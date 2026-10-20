"""Tests for the per-hop PTD sidecar (ptd_core.py).

    PYTHONPATH=/workspace/ComfyUI python3 custom_nodes/minimax_h3_pdd_acc/tests/test_ptd.py

Numeric tests feed fp32 activations so the bf16 factors widen exactly and the
reference and the sidecar agree to fp32 precision.
"""
import json
import os
import struct
import sys
import traceback

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)
sys.path.insert(0, PACK)

from ptd_core import (  # noqa: E402
    PTDStack, first_tensor_device, make_ptd_wrapper, make_sidecar_forward, plan_modules,
    read_ptd_config, split_ptd_state_dict,
)
from pdd_heads import (  # noqa: E402
    AUDIO_SHIFT, VIDEO_SHIFT, block_boundaries, validate_sigma_shifts,
)

PTD_FILE = "/workspace/ComfyUI/models/ptd/MiniMax-H3-Ref2VA-PTD-4Step.safetensors"
TESTS, SKIP = [], []


def test(fn):
    TESTS.append(fn)
    return fn


# ---------------------------------------------------------------- fixtures

H, R, IN, INNER, FFN = 4, 8, 16, 12, 10


def synth_adapters(hops=H, rank=R, hidden=IN, inner=INNER, ffn=FFN, blocks=2, refiners=1):
    g = torch.Generator().manual_seed(0)
    def t(*shape):
        return torch.randn(*shape, generator=g).to(torch.bfloat16)
    sd = {}
    for prefix, n in (("transformer_blocks", blocks), ("token_refiner.refiner_blocks", refiners)):
        for i in range(n):
            s = f"{prefix}.{i}"
            for p in ("to_q", "to_k", "to_v"):
                sd[f"{s}.attn.{p}.lora_down"] = t(hops, rank, hidden)
                sd[f"{s}.attn.{p}.lora_up"] = t(hops, inner, rank)
            sd[f"{s}.attn.to_out.0.lora_down"] = t(hops, rank, inner)
            sd[f"{s}.attn.to_out.0.lora_up"] = t(hops, hidden, rank)
            sd[f"{s}.ff.net.0.proj.lora_down"] = t(hops, rank, hidden)
            sd[f"{s}.ff.net.0.proj.lora_up"] = t(hops, ffn * 2, rank)
            sd[f"{s}.ff.net.2.lora_down"] = t(hops, rank, ffn)
            sd[f"{s}.ff.net.2.lora_up"] = t(hops, hidden, rank)
    sd["proj_out.weight"] = t(32, 6, hidden)
    sd["proj_out.bias"] = t(32, 6)
    sd["audio_proj_out.weight"] = t(32, 4, hidden)
    sd["audio_proj_out.bias"] = t(32, 4)
    meta = {"ptd_num_hops": str(hops), "pdd_num_steps": "32", "lora_rank": str(rank),
            "lora_alpha": str(float(rank)), "lora_targets": "to_q,to_k,to_v,to_out.0,"
            "ff.net.0.proj,ff.net.2"}
    return sd, meta


def build(residency="streamed", hops=H):
    sd, meta = synth_adapters(hops=hops)
    raw = {k: v.clone() for k, v in sd.items()}
    adapters, heads, cfg = split_ptd_state_dict(sd, meta)
    plan = plan_modules(adapters, cfg)
    stack = PTDStack(plan, cfg["hops"], residency=residency, pin=False)
    stack.ensure_device(torch.device("cpu"))
    return raw, plan, stack, cfg


def entry_for(plan, suffix):
    return next(e for e in plan if e["path"].endswith(suffix) and e["path"].startswith("blocks.0."))


# ---------------------------------------------------------------- tests

@test
def test_split_rejects_dehopped_slice():
    sd, meta = synth_adapters()
    sd["transformer_blocks.0.attn.to_q.lora_down"] = torch.zeros(R, IN, dtype=torch.bfloat16)
    try:
        split_ptd_state_dict(sd, meta, "x.safetensors")
    except ValueError as e:
        assert "hop axis" in str(e), e
        return
    raise AssertionError("a de-hopped slice must be refused")


@test
def test_split_rejects_pdd_file():
    sd, meta = synth_adapters()
    meta.pop("ptd_num_hops")
    try:
        read_ptd_config(meta, "pdd.safetensors")
    except ValueError as e:
        assert "not a PTD checkpoint" in str(e), e
        return
    raise AssertionError("a PDD file must be refused by the PTD node")


@test
def test_plan_shape_and_kinds():
    _, plan, _, _ = build()
    assert len(plan) == 3 * 4, len(plan)          # 2 trunk + 1 refiner blocks, 4 modules each
    kinds = [e["kind"] for e in plan if e["path"].startswith("blocks.0.")]
    assert kinds == ["qkv", "plain", "swiglu_fc1", "plain_forced"], kinds
    paths = [e["path"] for e in plan if e["path"].startswith("blocks.0.")]
    assert paths == ["blocks.0.attn.qkv_proj", "blocks.0.attn.out_proj",
                     "blocks.0.mlp.fc1", "blocks.0.mlp.fc2"], paths
    assert any(e["path"].startswith("token_refiner.blocks.0.") for e in plan)


@test
def test_plain_sidecar_matches_reference():
    raw, plan, stack, cfg = build()
    e = entry_for(plan, "attn.out_proj")
    lin = torch.nn.Linear(INNER, IN, bias=False)
    base = lin.weight.detach().clone()
    x = torch.randn(7, INNER)
    for hop in range(H):
        stack.set_hop(hop)
        lin.forward = make_sidecar_forward(lin, e, stack, 1.0)
        got = lin(x)
        A = raw["transformer_blocks.0.attn.to_out.0.lora_down"][hop].float()
        B = raw["transformer_blocks.0.attn.to_out.0.lora_up"][hop].float()
        want = F.linear(x, base) + F.linear(F.linear(x, A), B)
        assert torch.allclose(got, want, rtol=1e-5, atol=1e-5), (hop, (got - want).abs().max())


@test
def test_qkv_sidecar_matches_three_branches():
    raw, plan, stack, _ = build()
    e = entry_for(plan, "attn.qkv_proj")
    lin = torch.nn.Linear(IN, INNER * 3, bias=False)
    base = lin.weight.detach().clone()
    x = torch.randn(5, IN)
    for hop in (0, H - 1):
        stack.set_hop(hop)
        lin.forward = make_sidecar_forward(lin, e, stack, 1.0)
        got = lin(x)
        parts = []
        for p in ("to_q", "to_k", "to_v"):
            A = raw[f"transformer_blocks.0.attn.{p}.lora_down"][hop].float()
            B = raw[f"transformer_blocks.0.attn.{p}.lora_up"][hop].float()
            parts.append(F.linear(F.linear(x, A), B))
        want = F.linear(x, base) + torch.cat(parts, dim=-1)
        assert torch.allclose(got, want, rtol=1e-5, atol=1e-5), (hop, (got - want).abs().max())


@test
def test_fc1_half_swap():
    """diffusers packs [value; gate]; comfy's _swiglu_eager chunks [gate; value]."""
    raw, plan, stack, _ = build()
    e = entry_for(plan, "mlp.fc1")
    lin = torch.nn.Linear(IN, FFN * 2, bias=False)
    base = lin.weight.detach().clone()
    x = torch.randn(4, IN)
    stack.set_hop(1)
    lin.forward = make_sidecar_forward(lin, e, stack, 1.0)
    got = lin(x)
    A = raw["transformer_blocks.0.ff.net.0.proj.lora_down"][1].float()
    B = raw["transformer_blocks.0.ff.net.0.proj.lora_up"][1].float()
    swapped = torch.cat([B[FFN:], B[:FFN]], dim=0)
    want = F.linear(x, base) + F.linear(F.linear(x, A), swapped)
    assert torch.allclose(got, want, rtol=1e-5, atol=1e-5), (got - want).abs().max()
    unswapped = F.linear(x, base) + F.linear(F.linear(x, A), B)
    assert not torch.allclose(got, unswapped, rtol=1e-3, atol=1e-3), "swap was a no-op"


@test
def test_strength_scales_and_zero_is_base():
    raw, plan, stack, _ = build()
    e = entry_for(plan, "attn.out_proj")
    lin = torch.nn.Linear(INNER, IN, bias=False)
    base = lin.weight.detach().clone()
    x = torch.randn(3, INNER)
    stack.set_hop(2)
    lin.forward = make_sidecar_forward(lin, e, stack, 0.0)
    assert torch.allclose(lin(x), F.linear(x, base), rtol=1e-6, atol=1e-6)
    lin.forward = make_sidecar_forward(lin, e, stack, 1.0)
    full = lin(x) - F.linear(x, base)
    lin.forward = make_sidecar_forward(lin, e, stack, 0.5)
    half = lin(x) - F.linear(x, base)
    assert torch.allclose(half, full * 0.5, rtol=1e-4, atol=1e-5)


@test
def test_unarmed_is_exact_base():
    _, plan, stack, _ = build()
    stack._hop = -1                       # as constructed / after ensure_device
    assert stack.index == -1
    e = entry_for(plan, "attn.out_proj")
    lin = torch.nn.Linear(INNER, IN, bias=False)
    base = lin.weight.detach().clone()
    x = torch.randn(3, INNER)
    lin.forward = make_sidecar_forward(lin, e, stack, 1.0)
    assert torch.equal(lin(x), F.linear(x, base)), "unarmed sidecar must be byte-identical"


@test
def test_streamed_matches_resident():
    out = {}
    for mode in ("streamed", "resident"):
        raw, plan, stack, _ = build(residency=mode)
        e = entry_for(plan, "mlp.fc2")
        lin = torch.nn.Linear(FFN, IN, bias=False)
        torch.manual_seed(5)
        lin.weight.data = torch.randn(IN, FFN)
        x = torch.randn(6, FFN, generator=torch.Generator().manual_seed(9))
        res = []
        for hop in range(H):
            stack.set_hop(hop)
            lin.forward = make_sidecar_forward(lin, e, stack, 1.0)
            res.append(lin(x).clone())
        out[mode] = res
    for hop in range(H):
        assert torch.equal(out["streamed"][hop], out["resident"][hop]), f"hop {hop} differs"


@test
def test_hop_clamped_and_index_semantics():
    _, _, stack, _ = build(residency="resident")
    stack.set_hop(99)
    assert stack.index == H - 1
    stack.set_hop(-5)
    assert stack.index == 0
    _, _, s2, _ = build(residency="streamed")
    s2.set_hop(3)
    assert s2.index == 0, "streamed keeps the active hop in row 0"


@test
def test_real_file_plan_is_208_modules():
    if not os.path.exists(PTD_FILE):
        SKIP.append("real_file_plan (PTD file missing)")
        return
    with open(PTD_FILE, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    meta = hdr.pop("__metadata__", {})
    cfg = read_ptd_config(meta, PTD_FILE)
    sd = {k: torch.zeros(v["shape"], dtype=torch.bfloat16) for k, v in hdr.items()}
    adapters, _heads, cfg = split_ptd_state_dict(sd, meta, PTD_FILE)
    plan = plan_modules(adapters, cfg, PTD_FILE)
    assert len(plan) == 50 * 4 + 2 * 4, len(plan)
    assert sum(e["kind"] == "plain_forced" for e in plan) == 52
    assert sum(e["kind"] == "qkv" for e in plan) == 52
    # fused qkv down-projection is the 3-way stack
    q = next(e for e in plan if e["path"] == "blocks.0.attn.qkv_proj")
    assert list(q["flat"][0].shape) == [cfg["hops"], 3 * cfg["rank"], 5376], q["flat"][0].shape
    assert list(q["flat"][1].shape) == [cfg["hops"], 7168, cfg["rank"]]


@test
def test_fc2_bypass_is_real_and_force_forward_fixes_it():
    """The whole reason fc2 gets comfy_force_forward.

    core's MLP calls comfy.ops.linear_input_act(self.fc2, ...), which on a
    quantized or fp16-accumulate path reads fc2.weight DIRECTLY and never calls
    fc2.forward -- a patched forward would be a silent no-op. Reproduced here for
    real (fp16 activations + allow_fp16_accumulation, the flag h3 mode runs with),
    not mocked.
    """
    if not torch.cuda.is_available():
        SKIP.append("fc2_bypass (no CUDA)")
        return
    import comfy.ops
    prev = torch.backends.cuda.matmul.allow_fp16_accumulation
    torch.backends.cuda.matmul.allow_fp16_accumulation = True
    try:
        calls = []

        # must be a comfy ops Linear: cast_bias_weight reads CastWeightBiasOp
        # attributes that a bare torch.nn.Linear does not carry
        class Probe(comfy.ops.disable_weight_init.Linear):
            def forward(self, x):
                calls.append(x.shape[-1])
                return super().forward(x)

        lin = Probe(FFN, IN, bias=False).to("cuda", torch.float16)
        h = torch.randn(4, FFN * 2, device="cuda", dtype=torch.float16)
        assert comfy.ops._fp16_linear_wanted(h), "test needs the fp16 fast path active"

        comfy.ops.linear_input_act(lin, h, "swiglu")
        assert not calls, ("fc2 was NOT bypassed on the fp16 path -- the premise for "
                           "comfy_force_forward may no longer hold; re-check comfy.ops")

        lin.comfy_force_forward = True
        comfy.ops.linear_input_act(lin, h, "swiglu")
        assert calls, "comfy_force_forward did not route fc2 back through forward"
        assert calls[-1] == FFN, (f"forward received width {calls[-1]}, expected the "
                                  f"POST-SwiGLU {FFN} (what diffusers' ff.net.2 consumes)")
    finally:
        torch.backends.cuda.matmul.allow_fp16_accumulation = prev


@test
def test_real_refiner_block_matches_merged_weights():
    """End-to-end ground truth on a REAL core block.

    Builds comfy's own RefinerBlock (same Attention/MLP classes DiTBlock uses),
    attaches the sidecar to its four Linears, and compares against an identical
    block whose weights have the hop's B@A merged in. Merging is the definition
    of applying a LoRA, so agreement proves the whole mapping at once: fused-qkv
    row order, the fc1 half-swap, the fc2 force-forward path, and hop selection.
    """
    if not torch.cuda.is_available():
        SKIP.append("real_refiner_block (no CUDA)")
        return
    import comfy.ops
    from comfy.ldm.minimax.model import RefinerBlock

    dev, dt = "cuda", torch.float32
    heads, head_dim = 2, 6
    assert heads * head_dim == INNER
    ops = comfy.ops.disable_weight_init
    torch.manual_seed(11)

    def make():
        b = RefinerBlock(IN, heads, head_dim, FFN, 1e-6, 1e-6,
                         dtype=dt, device=dev, operations=ops)
        return b.to(dev, dt).eval()

    # comfy ops Linear has a NO-OP reset_parameters: fresh weights are
    # uninitialised memory, so seeding cannot make two fixtures agree --
    # initialise explicitly, then copy.
    live = make()
    with torch.no_grad():
        g = torch.Generator(dev).manual_seed(11)
        for p in live.parameters():
            p.copy_(torch.randn(p.shape, device=dev, dtype=dt, generator=g) * 0.25)
    ref = make()
    ref.load_state_dict(live.state_dict())
    base_block = make()
    base_block.load_state_dict(live.state_dict())
    for (n1, p1), (_n2, p2) in zip(live.named_parameters(), ref.named_parameters()):
        assert torch.equal(p1, p2), f"fixtures diverged at {n1}"

    raw, plan, stack, cfg = build()
    stack.ensure_device(torch.device(dev))
    hop = 2
    stack.set_hop(hop)
    src = "transformer_blocks.0"

    def fac(mod):
        A = raw[f"{src}.{mod}.lora_down"][hop].to(dev, dt)
        B = raw[f"{src}.{mod}.lora_up"][hop].to(dev, dt)
        return A, B

    # ---- reference: merge B@A into the weights ----
    with torch.no_grad():
        qkv = []
        for p in ("to_q", "to_k", "to_v"):
            A, B = fac(f"attn.{p}")
            qkv.append(B @ A)
        ref.attn.qkv_proj.weight += torch.cat(qkv, dim=0)
        A, B = fac("attn.to_out.0")
        ref.attn.out_proj.weight += B @ A
        A, B = fac("ff.net.0.proj")
        Bsw = torch.cat([B[FFN:], B[:FFN]], dim=0)      # diffusers [value;gate] -> comfy [gate;value]
        ref.mlp.fc1.weight += Bsw @ A
        A, B = fac("ff.net.2")
        ref.mlp.fc2.weight += B @ A

    # ---- live: attach the sidecar exactly as attach_ptd_sidecar does ----
    for suffix, module in (("attn.qkv_proj", live.attn.qkv_proj),
                           ("attn.out_proj", live.attn.out_proj),
                           ("mlp.fc1", live.mlp.fc1),
                           ("mlp.fc2", live.mlp.fc2)):
        e = entry_for(plan, suffix)
        module.forward = make_sidecar_forward(module, e, stack, 1.0)
        if e["kind"] == "plain_forced":
            module.comfy_force_forward = True

    x = torch.randn(9, IN, device=dev, dtype=dt, generator=torch.Generator(dev).manual_seed(3))
    with torch.no_grad():
        got, want = live(x.clone()), ref(x.clone())
        base = base_block(x.clone())
    err = (got - want).abs().max().item()
    scale = want.abs().max().item()
    assert err / scale < 1e-4, f"sidecar != merged weights: max abs err {err} (scale {scale})"
    assert (want - base).abs().max().item() / scale > 1e-3, "LoRA had no effect; test is vacuous"


@test
def test_first_tensor_device_handles_h3_list():
    """MiniMax-H3 hands DIFFUSION_MODEL wrappers x = [video, audio], not a tensor."""
    v, a = torch.zeros(2, 3), torch.zeros(1)
    assert first_tensor_device([v, a]) == v.device
    assert first_tensor_device((v, a)) == v.device
    assert first_tensor_device(v) == v.device
    assert first_tensor_device([None, a]) == a.device
    assert first_tensor_device(None) is None
    assert first_tensor_device(["not a tensor"]) is None


@test
def test_wrapper_accepts_h3_list_x_and_arms_the_hop():
    """Regression: the wrapper must not assume x is a tensor (x.device -> AttributeError).

    Calls it exactly the way comfy/ldm/minimax/model.py does, including the
    [video, audio] list and the minimax_payload audio_scale.
    """
    _raw, plan, stack, cfg = build(residency="streamed")
    hops = cfg["hops"]
    bounds = block_boundaries(cfg["num_steps"], (cfg["num_steps"] // hops,) * hops)

    class FakeDM:
        sigma_shift_video, sigma_shift_audio = VIDEO_SHIFT, AUDIO_SHIFT

    seen = {}

    class FakeExecutor:
        class_obj = FakeDM()

        def __call__(self, x, timestep, context, transformer_options, **kwargs):
            seen["row"] = stack.index
            seen["x_is_list"] = isinstance(x, list)
            return [x[0], x[1]]

    wrapper = make_ptd_wrapper(
        __import__("types").SimpleNamespace(sigma_v=None, blk=None), stack, bounds, "error",
        lambda ex, to, kw: validate_sigma_shifts(ex, to, kw, "MiniMaxH3PTDApply"))

    for hop in range(hops):
        x = [torch.zeros(1, 4, 2, 2), torch.zeros(1, 4, 2)]
        ts = torch.tensor([bounds[hop] * 1000.0])
        out = wrapper(FakeExecutor(), x, ts, None, {},
                      minimax_payload={"audio_scale": VIDEO_SHIFT / AUDIO_SHIFT})
        assert seen["x_is_list"], "wrapper must pass x through untouched"
        assert seen["row"] == (0 if stack.residency != "resident" else hop), (hop, seen["row"])
        assert stack._hop == hop, f"hop {hop} not armed (got {stack._hop})"
        assert out is not None
    assert stack.index == -1, "stack must be released after the forward"


@test
def test_sidecar_binds_device_when_wrapper_could_not():
    """Correctness must not depend on the wrapper guessing a device."""
    raw, plan, stack, _ = build(residency="streamed")
    stack._device = None
    stack._gpu = None
    stack.request_hop(1)
    assert stack.bind(None) == -1, "no device known yet -> unarmed"
    e = entry_for(plan, "attn.out_proj")
    lin = torch.nn.Linear(INNER, IN, bias=False)
    base = lin.weight.detach().clone()
    x = torch.randn(3, INNER)
    lin.forward = make_sidecar_forward(lin, e, stack, 1.0)
    got = lin(x)                       # the forward itself must place the stack
    A = raw["transformer_blocks.0.attn.to_out.0.lora_down"][1].float()
    B = raw["transformer_blocks.0.attn.to_out.0.lora_up"][1].float()
    want = F.linear(x, base) + F.linear(F.linear(x, A), B)
    assert torch.allclose(got, want, rtol=1e-5, atol=1e-5), (got - want).abs().max()


if __name__ == "__main__":
    failed = 0
    for fn in TESTS:
        try:
            fn()
            print(f"  ok: {fn.__name__}")
        except Exception:
            failed += 1
            print(f"  FAIL: {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(TESTS) - failed} passed, {failed} failed" + (f", skipped: {SKIP}" if SKIP else ""))
    sys.exit(1 if failed else 0)
