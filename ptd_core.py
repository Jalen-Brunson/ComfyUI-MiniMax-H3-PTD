"""Per-hop PTD LoRA sidecar for MiniMax-H3 (taobao-content-ai/MiniMax-H3-PTD-LoRAs).

PTD (Parametric Trajectory Distillation) = PDD's per-interval output head bank --
byte-identical key names and shapes to the alibaba-pai PDD Acc files -- plus ONE
INDEPENDENT LoRA PER SAMPLING STEP ("hop") on the trunk. Hop index == PDD block
index == step index, on a uniform 32/hops grid.

Why a sidecar instead of comfy's normal LoRA path
-------------------------------------------------
A per-hop adapter cannot be baked into the trunk: that would need one 32.4 GB
baked trunk per hop, or a full re-patch of 208 modules (dequant -> add -> requant
on an int8_convrot build) between every step. Instead the base weight is left
untouched and each target Linear gains a parallel low-rank branch whose hop slice
is selected per step. Two side benefits fall out: quantized weights are never
dequantised, and character LoRAs keep composing through the normal add_patches
path underneath.

Measured on the 8-step release (analysis/h3_ptd/hop_divergence.py): the hops are
mutually ORTHOGONAL (median pairwise cosine +0.001), so collapsing them to a
single static LoRA destroys 7/8 of the signal. Per-hop is load-bearing; there is
no cheaper approximation.

Module mapping (diffusers file -> comfy model)
----------------------------------------------
    transformer_blocks.{i}.attn.to_q/to_k/to_v -> blocks.{i}.attn.qkv_proj  (fused: 3 branches)
    transformer_blocks.{i}.attn.to_out.0       -> blocks.{i}.attn.out_proj
    transformer_blocks.{i}.ff.net.0.proj       -> blocks.{i}.mlp.fc1  (B row-halves swapped)
    transformer_blocks.{i}.ff.net.2            -> blocks.{i}.mlp.fc2  (see comfy_force_forward)
    token_refiner.refiner_blocks.{i}.*         -> token_refiner.blocks.{i}.*

The fc2 trap
------------
core's MLP.forward is `comfy.ops.linear_input_act(self.fc2, self.fc1(x), "swiglu")`.
On a quantized trunk that fuses swiglu into the int8 GEMM epilogue and reads
fc2.weight DIRECTLY -- `fc2.forward` is never called, so a patched forward would
be a SILENT no-op. comfy.ops exposes the escape hatch: `comfy_force_forward=True`
on the module routes it back through `linear(_eager_input_act(x, "swiglu"))`, so
our forward runs and receives the post-SwiGLU activation (which is exactly what
diffusers' `ff.net.2` consumes). Cost is the fused epilogue on fc2 only.
qkv_proj / out_proj / fc1 are all plain module calls and need no such treatment.
"""
from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

try:                                        # inside ComfyUI, as a package
    from .pdd_heads import HEAD_KEYS, partition_starts, select_block
except ImportError:                          # as a plain module (tests, CLI)
    from pdd_heads import HEAD_KEYS, partition_starts, select_block

# diffusers suffix -> (comfy module suffix, kind). "qkv" is handled as a group.
_PLAIN_MAP = (
    ("attn.to_out.0", "attn.out_proj", "plain"),
    ("ff.net.0.proj", "mlp.fc1", "swiglu_fc1"),
    ("ff.net.2", "mlp.fc2", "plain_forced"),
)
_QKV_PARTS = ("to_q", "to_k", "to_v")


# ---------------------------------------------------------------------------
# file loading
# ---------------------------------------------------------------------------

def read_ptd_config(metadata, filename="<file>"):
    meta = metadata or {}
    if "ptd_num_hops" not in meta:
        raise ValueError(
            f"{filename}: no 'ptd_num_hops' in metadata -- this is not a PTD checkpoint. "
            f"Load alibaba-pai PDD Acc files with MiniMaxH3PDDAccApply instead.")
    hops = int(meta["ptd_num_hops"])
    num_steps = int(meta.get("pdd_num_steps", 32))
    if hops < 1 or num_steps % hops:
        raise ValueError(f"{filename}: pdd_num_steps={num_steps} not divisible by "
                         f"ptd_num_hops={hops}")
    rank = int(meta.get("lora_rank", 128))
    return {
        "hops": hops,
        "num_steps": num_steps,
        # derived, never read from the file: the reference apply_ptd_lora
        # recomputes it too and warns the stored field can be stale
        "block_size": num_steps // hops,
        "rank": rank,
        "alpha": float(meta.get("lora_alpha", float(rank))),
        "scaling": float(meta.get("lora_alpha", float(rank))) / rank,
        "targets": meta.get("lora_targets", ""),
    }


def split_ptd_state_dict(sd, metadata, filename="<file>"):
    """-> (adapters, heads, config). adapters[diffusers_module] = (A[H,r,in], B[H,out,r])."""
    config = read_ptd_config(metadata, filename)
    for k in HEAD_KEYS:
        if k not in sd:
            raise ValueError(f"{filename} has no '{k}' -- not a PTD/PDD head bank?")
    heads = tuple(sd.pop(k) for k in HEAD_KEYS)
    vw, _vb, aw, _ab = heads
    n = config["num_steps"]
    if vw.ndim != 3 or vw.shape[0] != n or aw.shape[0] != n:
        raise ValueError(f"{filename}: head bank shape mismatch -- proj_out {list(vw.shape)}, "
                         f"audio_proj_out {list(aw.shape)}, pdd_num_steps {n}")

    adapters, leftovers = {}, []
    for key, t in sd.items():
        k = key.replace("._checkpoint_wrapped_module", "")
        if k.endswith(("curv_weight", "curv_bias")):
            continue  # training-only curvature heads; inference never reads them
        if k.endswith(".lora_down"):
            mod, slot = k[: -len(".lora_down")], 0
        elif k.endswith(".lora_up"):
            mod, slot = k[: -len(".lora_up")], 1
        else:
            leftovers.append(k)
            continue
        if t.ndim != 3 or t.shape[0] != config["hops"]:
            raise ValueError(f"{filename}: {k} has shape {list(t.shape)}, expected a leading "
                             f"hop axis of {config['hops']} -- is this a de-hopped slice? "
                             f"(slice_ptd_hop.py output loads with MiniMaxH3PDDAccApply)")
        adapters.setdefault(mod, [None, None])[slot] = t
    if leftovers:
        raise ValueError(f"{filename}: {len(leftovers)} unrecognized keys, e.g. "
                         f"{sorted(leftovers)[:4]} -- format drift, refusing to load.")
    for mod, pair in adapters.items():
        if pair[0] is None or pair[1] is None:
            raise ValueError(f"{filename}: {mod} is missing its "
                             f"{'lora_down' if pair[0] is None else 'lora_up'}")
    return {m: tuple(p) for m, p in adapters.items()}, heads, config


def plan_modules(adapters, config, filename="<file>"):
    """Group diffusers adapters into comfy target modules.

    -> list of dicts {path, kind, tensors:[(A,B), ...]} where path is relative to
    "diffusion_model." and tensors holds 3 pairs for qkv, 1 otherwise.
    """
    prefixes = {}
    for mod in adapters:
        if mod.startswith("transformer_blocks."):
            i = mod.split(".")[1]
            prefixes.setdefault((f"transformer_blocks.{i}", f"blocks.{i}"), None)
        elif mod.startswith("token_refiner.refiner_blocks."):
            i = mod.split(".")[2]
            prefixes.setdefault(
                (f"token_refiner.refiner_blocks.{i}", f"token_refiner.blocks.{i}"), None)
        else:
            raise ValueError(f"{filename}: unexpected module prefix in '{mod}'")

    plan, used = [], set()
    for src, dst in sorted(prefixes, key=lambda p: (p[0].split(".")[0], int(p[0].split(".")[-1]))):
        pairs = []
        for part in _QKV_PARTS:
            key = f"{src}.attn.{part}"
            if key not in adapters:
                raise ValueError(f"{filename}: {key} missing (qkv must be complete)")
            pairs.append(adapters[key])
            used.add(key)
        # one stacked down-projection [H, 3r, in] so the hot path does a single
        # matmul and never re-concatenates per forward; ups stay separate so each
        # writes into its own slice of the fused [q|k|v] output
        A3 = torch.cat([A for A, _ in pairs], dim=1).contiguous()
        plan.append({"path": f"{dst}.attn.qkv_proj", "kind": "qkv",
                     "flat": [A3] + [B.contiguous() for _, B in pairs],
                     "src": f"{src}.attn.to_[qkv]"})
        for s_suffix, d_suffix, kind in _PLAIN_MAP:
            key = f"{src}.{s_suffix}"
            if key not in adapters:
                raise ValueError(f"{filename}: {key} missing")
            A, B = adapters[key]
            if kind == "swiglu_fc1":
                # diffusers SwiGLU chunks [value; gate]; comfy _swiglu_eager chunks
                # [gate; value] -> swap the lora_up row halves, once, at load
                half = B.shape[1] // 2
                B = torch.cat([B[:, half:], B[:, :half]], dim=1).contiguous()
            plan.append({"path": f"{dst}.{d_suffix}", "kind": kind,
                         "flat": [A.contiguous(), B.contiguous()], "src": key})
            used.add(key)
    unused = sorted(set(adapters) - used)
    if unused:
        raise ValueError(f"{filename}: {len(unused)} adapters did not map to a comfy module, "
                         f"e.g. {unused[:4]}")
    return plan


# ---------------------------------------------------------------------------
# the stack
# ---------------------------------------------------------------------------

class PTDStack:
    """All per-hop factors in ONE flat buffer, so a hop change is one copy (or none).

    residency="resident": the whole [H, N] buffer lives on the GPU and set_hop is
    a free index change. residency="streamed": [H, N] stays in CPU RAM and only
    the active hop ([1, N]) is on the GPU, refreshed with a single H2D per step.

    Device allocation is LAZY -- it happens inside the sampling wrapper, after
    model_management has loaded the UNET, so a resident stack cannot provoke a
    partial UNET offload at load time.
    """

    def __init__(self, plan, hops, residency="streamed", pin=True):
        self.hops = int(hops)
        self.residency = residency
        self.pin = bool(pin)
        self._hop = -1          # hop currently resident in the live buffer
        self._pending = None    # hop the sampler asked for, not yet bound
        self._device = None
        self._gpu = None

        # pack every factor into one flat [H, N] CPU tensor
        total = 0
        for entry in plan:
            layout = []
            for t in entry["flat"]:
                if t.shape[0] != self.hops:
                    raise ValueError(f"{entry['src']}: hop axis {t.shape[0]} != {self.hops}")
                n = t[0].numel()
                layout.append((total, tuple(t.shape[1:])))
                total += n
            entry["_layout"] = layout
        self.numel = total
        self.bytes_per_hop = total * 2  # bf16

        flat = torch.empty(self.hops, total, dtype=torch.bfloat16,
                           pin_memory=(self.pin and residency == "streamed"))
        for entry in plan:
            for (off, shape), t in zip(entry["_layout"], entry["flat"]):
                n = t[0].numel()
                flat[:, off:off + n] = t.to(torch.bfloat16).reshape(self.hops, n)
            entry["flat"] = None  # packed; drop the per-module references
        self._cpu = flat
        self.plan = plan

    # -- device -------------------------------------------------------------
    def ensure_device(self, device):
        if self._device == device:
            return
        self._device = device
        if self.residency == "resident":
            self._gpu = self._cpu.to(device, non_blocking=False)
        else:
            self._gpu = torch.empty(1, self.numel, dtype=torch.bfloat16, device=device)
        self._hop = -1
        # Views must be taken per ROW: a column slice of [H, N] is not contiguous,
        # so .view() on it raises. gpu[h, off:off+n] IS contiguous.
        rows = self._gpu.shape[0]
        for entry in self.plan:
            entry["_live"] = [[self._gpu[r, off:off + s[0] * s[1]].view(*s)
                               for off, s in entry["_layout"]] for r in range(rows)]

    def request_hop(self, hop):
        """Arm a hop WITHOUT touching the GPU -- no device is known yet here."""
        self._pending = min(max(int(hop), 0), self.hops - 1)

    def release(self):
        self._pending = None

    def bind(self, device=None):
        """Make the pending hop live on `device` and return its buffer row (-1 unarmed).

        Called from every sidecar forward, where the input tensor's device is
        unambiguous -- so correctness never depends on the wrapper guessing it.
        Idempotent and cheap: after the first call in a step it is two compares.
        """
        if device is not None and self._device != device:
            self.ensure_device(device)
        if self._gpu is None or self._pending is None:
            return -1
        if self._pending != self._hop:
            if self.residency != "resident":
                self._gpu[0].copy_(self._cpu[self._pending], non_blocking=True)
            self._hop = self._pending
        return self._hop if self.residency == "resident" else 0

    def set_hop(self, hop):
        """Arm and bind in one go (tests / already-placed stacks)."""
        self.request_hop(hop)
        return self.bind(self._device)

    @property
    def index(self):
        """Row of the live buffer holding the active hop, or -1 when unarmed."""
        if self._hop < 0 or self._gpu is None or self._pending is None:
            return -1
        return self._hop if self.residency == "resident" else 0

    def describe(self):
        gib = self.bytes_per_hop / 2 ** 30
        if self.residency == "resident":
            return f"resident, {gib * self.hops:.2f} GiB VRAM ({self.hops} hops), no per-step copy"
        return (f"streamed, {gib:.2f} GiB VRAM + {gib * self.hops:.2f} GiB CPU"
                f"{' (pinned)' if self.pin else ''}, one {gib:.2f} GiB H2D per step")


# ---------------------------------------------------------------------------
# the per-module forward
# ---------------------------------------------------------------------------

def first_tensor_device(x):
    """Device of the first real tensor in a DIFFUSION_MODEL `x`.

    MiniMax-H3 hands its wrappers ``x = [video_latent, audio_latent]`` -- a
    LIST, not a tensor (core itself reads ``x[0].device``). Returns None when no
    device can be determined; the per-module bind then settles it.
    """
    if torch.is_tensor(x):
        return x.device
    if isinstance(x, (list, tuple)):
        for item in x:
            if torch.is_tensor(item):
                return item.device
    return None


def _add_lora_(out, x, A, B, strength):
    """out += strength * (x @ A.T) @ B.T, in place, without a full-size temporary.

    The up-projection is the whole output width (fc1 is 28672 wide; at ~100k tokens
    a temporary would be several GiB), so it is written straight into `out` with
    addmm_. Non-2D or grad-tracking inputs fall back to the allocating form.
    """
    z = F.linear(x, A.to(x.dtype))
    Bt = B.to(x.dtype)
    if out.ndim == 2 and z.ndim == 2 and not out.requires_grad:
        try:
            out.addmm_(z, Bt.t(), alpha=strength)
            return out
        except RuntimeError:
            pass  # strided/dtype combination cuBLAS declined -- fall through
    return out.add_(F.linear(z, Bt), alpha=strength)


def make_sidecar_forward(module, entry, stack, strength):
    """Bound replacement for a target Linear's forward: base output + the hop's LoRA."""
    orig = type(module).forward
    kind = entry["kind"]

    def sidecar_forward(self, x, *args, **kwargs):
        out = orig(self, x, *args, **kwargs)
        hop = stack.bind(x.device if torch.is_tensor(x) else None)
        if hop < 0:
            return out  # not armed -- byte-identical to the base model
        live = entry["_live"][hop]
        if kind == "qkv":
            # live = [A3 [3r, in], B_q, B_k, B_v]; one stacked down-projection,
            # three narrow up-projections into the fused [q|k|v] output (core
            # splits it by heads*head_dim, in that order)
            z = F.linear(x, live[0].to(x.dtype))
            rank = z.shape[-1] // 3
            width = live[1].shape[0]          # B_q rows == inner
            for j in range(3):
                Bj = live[1 + j].to(x.dtype)
                zj = z.narrow(-1, j * rank, rank)
                dst = out.narrow(-1, j * width, width)
                if dst.ndim == 2 and not out.requires_grad:
                    try:
                        dst.addmm_(zj, Bj.t(), alpha=strength)
                        continue
                    except RuntimeError:
                        pass
                dst.add_(F.linear(zj, Bj), alpha=strength)
            return out
        return _add_lora_(out, x, live[0], live[1], strength)

    return sidecar_forward.__get__(module, type(module))


# ---------------------------------------------------------------------------
# attach
# ---------------------------------------------------------------------------

def attach_ptd_sidecar(m, plan, stack, strength, logger=logging):
    """Object-patch every target module on the ModelPatcher `m`. -> (n_modules, notes)."""
    forced = 0
    for entry in plan:
        path = "diffusion_model." + entry["path"]
        try:
            module = m.get_model_object(path)
        except Exception as exc:
            raise ValueError(f"MiniMaxH3PTDApply: target module '{path}' not found on the "
                             f"loaded model ({exc}) -- is this a MiniMax-H3 UNET?")
        exp_in = entry["_layout"][0][1][1]          # A: (r or 3r, in)
        wB = entry["_layout"][1][1][0]              # B: (out, r)
        exp_out = wB * 3 if entry["kind"] == "qkv" else wB
        got = tuple(module.weight.shape)
        if got != (exp_out, exp_in):
            raise ValueError(f"MiniMaxH3PTDApply: {path} weight {list(got)} does not match the "
                             f"adapter's [{exp_out}, {exp_in}] (from {entry['src']})")
        m.add_object_patch(path + ".forward", make_sidecar_forward(module, entry, stack, strength))
        if entry["kind"] == "plain_forced":
            # see "The fc2 trap" in the module docstring
            m.add_object_patch(path + ".comfy_force_forward", True)
            forced += 1
    note = (f"sidecar: {len(plan)} modules patched @ strength {strength} "
            f"({forced} fc2 forced through forward), {stack.describe()}")
    logger.info("MiniMaxH3PTDApply: %s", note)
    return len(plan), note


def hop_map(sizes, trained_block, hops):
    """Which hop drives each partition block.

    Hop h was trained on the fine interval [h*trained_block, (h+1)*trained_block),
    so a block STARTING at fine index s is served by hop s // trained_block --
    the adapter trained for the part of the trajectory that step begins in.

    With nfe == hops this is the identity (block h -> hop h), which is the only
    configuration PTD was distilled for. With fewer, larger steps it subsamples
    the trajectory: 8 hops at nfe 4 -> hops 0, 2, 4, 6. With a non-uniform
    partition (nfe 6 -> 8,8,4,4,4,4) -> hops 0, 2, 4, 5, 6, 7.
    """
    return [min(int(s) // int(trained_block), hops - 1) for s in partition_starts(sizes)]


def make_ptd_wrapper(holder, stack, bounds, on_off_grid, validate_shifts, hop_of_block=None):
    """`stack` may be None (lora_strength 0.0 = head bank only, no trunk sidecar).

    `hop_of_block` maps a partition block index to the hop that serves it; None
    means the identity (nfe == hops).
    """
    """DIFFUSION_MODEL wrapper: arm the hop BEFORE the trunk runs.

    The PDD head wrapper derives its block inside final_layer.forward, which runs
    LAST -- too late for a trunk sidecar. This sets holder.blk up front from the
    same sigma, so the trunk and the fused head always agree (final_layer then
    recomputes the identical value).
    """
    def ptd_wrapper(executor, x, timestep, context, transformer_options={}, **kwargs):
        validate_shifts(executor, transformer_options, kwargs)
        holder.sigma_v = float(timestep.flatten()[0]) / 1000.0
        holder.blk = select_block(holder.sigma_v, bounds, on_off_grid)
        if stack is not None:
            blk = holder.blk
            stack.request_hop(blk if hop_of_block is None else hop_of_block[blk])
            # best effort only: H3 passes x as [video, audio], and under offload the
            # compute device can differ anyway -- each sidecar re-binds on its own input
            stack.bind(first_tensor_device(x))
        try:
            return executor(x, timestep, context, transformer_options, **kwargs)
        finally:
            holder.sigma_v = None
            holder.blk = None
            if stack is not None:
                stack.release()
    return ptd_wrapper
