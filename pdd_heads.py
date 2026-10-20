"""PDD head-bank machinery shared with ComfyUI-MiniMax-H3-PDD-Acc.

PTD reuses PDD's decoder verbatim -- the head-bank keys and shapes are
byte-identical between the two releases -- so this module is a VENDORED SUBSET
of `pdd_acc_core.py` from:

    https://github.com/Jalen-Brunson/ComfyUI-MiniMax-H3-PDD-Acc

Function bodies are extracted mechanically and are byte-identical to that
source; fix bugs there first, then re-vendor. Only what PTD needs is here (the
diffusers->ComfyUI LoRA key conversion, adaln curve rebasing and trunk baking
all stay upstream, since PTD carries no adaln LoRA and cannot be baked).

Grid math is float64 throughout; endpoints are analytically exact 1.0 / 0.0.
"""

import math

import torch
import torch.nn.functional as F

VIDEO_SHIFT = 12.0


AUDIO_SHIFT = 3.0


KNOT_TOLERANCE = 1e-6


HEAD_KEYS = ("proj_out.weight", "proj_out.bias", "audio_proj_out.weight", "audio_proj_out.bias")


def shifted_sigma(shift, t):
    return shift * t / (1.0 + (shift - 1.0) * t)


def fine_sigmas(shift, num_steps):
    """Descending [1.0 ... 0.0], num_steps+1 knots, python floats (float64)."""
    return [shifted_sigma(shift, (num_steps - i) / num_steps) for i in range(num_steps + 1)]


def validate_sigma_shifts(executor, transformer_options, kwargs, node="MiniMaxH3PDDAccApply"):
    """Refuse a run whose SigmaShift is not the 12/3 the heads were trained on.

    Shared by the PDD and PTD wrappers: both consume the same fused head bank, so
    both are invalid off that grid.
    """
    import math
    dm = executor.class_obj
    shift_v = float(transformer_options.get("minimax_h3_sigma_shift_video",
                                            getattr(dm, "sigma_shift_video", VIDEO_SHIFT)))
    shift_a = float(transformer_options.get("minimax_h3_sigma_shift_audio",
                                            getattr(dm, "sigma_shift_audio", AUDIO_SHIFT)))
    if not (math.isclose(shift_v, VIDEO_SHIFT, abs_tol=1e-6)
            and math.isclose(shift_a, AUDIO_SHIFT, abs_tol=1e-6)):
        raise ValueError(
            f"{node}: heads are trained on SigmaShift "
            f"{VIDEO_SHIFT}/{AUDIO_SHIFT}, got {shift_v}/{shift_a}.")
    payload = kwargs.get("minimax_payload") or {}
    scale = float(payload.get("audio_scale", VIDEO_SHIFT / AUDIO_SHIFT))
    if not math.isclose(scale, VIDEO_SHIFT / AUDIO_SHIFT, abs_tol=1e-6):
        raise ValueError(
            f"{node}: payload audio_scale {scale} != "
            f"{VIDEO_SHIFT / AUDIO_SHIFT} -- mismatched sampling shifts.")


DEFAULT_PARTITIONS = {
    6: (8, 8, 4, 4, 4, 4),
    # OFF-ENVELOPE experiment partitions (allow_off_envelope only): >8 blocks
    # over 32 fine steps force sizes < 4. Size-2 blocks are placed at the LOW
    # sigma end (later fine indices) so the off-grid starts sit where per-step
    # displacement is smallest.
    10: (4, 4, 4, 4, 4, 4, 2, 2, 2, 2),
    12: (4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2),
}


def resolve_partition(num_steps, nfe, partition_text="", trained_block=4,
                      allow_off_envelope=False):
    """Return the tuple of block sizes (in fine steps) for an nfe / partition spec.

    Sizes are restricted to {trained_block, 2*trained_block} (4 and 8 for the
    released files): PDD heads are conditioned on trunk features from block
    STARTS at multiples of L_min=trained_block with sizes up to L_max=2*L_min —
    the two officially demonstrated groupings. Evaluating the trunk anywhere
    else feeds the heads features they never trained on; empirically (fl2va,
    nfe 32, 2026-08-27 reports + local repro) that renders as heavy noise, so
    off-envelope partitions are rejected rather than allowed to degrade —
    unless allow_off_envelope=True (explicit experiment opt-in; expect
    degradation, the caller should surface a warning).
    """
    text = (partition_text or "").strip()
    if text:
        try:
            sizes = tuple(int(p) for p in text.replace(" ", "").split(",") if p)
        except ValueError:
            raise ValueError(f"partition '{partition_text}' is not a comma-separated int list")
        if any(s < 1 for s in sizes) or sum(sizes) != num_steps:
            raise ValueError(f"partition {sizes} must be positive block sizes summing to "
                             f"{num_steps} (got sum {sum(sizes)})")
    elif num_steps % nfe == 0:
        sizes = (num_steps // nfe,) * nfe
    elif nfe in DEFAULT_PARTITIONS and sum(DEFAULT_PARTITIONS[nfe]) == num_steps:
        sizes = DEFAULT_PARTITIONS[nfe]
    else:
        raise ValueError(f"nfe {nfe} does not divide pdd_num_steps {num_steps} and has no "
                         f"default partition — supply one via the partition field (block sizes "
                         f"summing to {num_steps}, e.g. '8,8,4,4,4,4')")
    allowed = (trained_block, 2 * trained_block)
    bad = sorted({s for s in sizes if s not in allowed})
    if bad and not allow_off_envelope:
        raise ValueError(
            f"partition {sizes}: block sizes {bad} are outside the trained envelope "
            f"{allowed}. PDD heads only ever saw trunk features from block starts on the "
            f"L_min={trained_block} grid with blocks of {allowed[0]} or {allowed[1]} fine "
            f"steps; anything else renders as heavy noise (verified on fl2va). Use nfe "
            f"8/6/4 or a partition made of {allowed[0]}s and {allowed[1]}s — or set "
            f"allow_off_envelope=True to run the experiment anyway.")
    return sizes


def partition_off_envelope(sizes, trained_block=4):
    """Sorted list of block sizes in `sizes` outside the trained envelope."""
    allowed = (trained_block, 2 * trained_block)
    return sorted({s for s in sizes if s not in allowed})


def partition_starts(sizes):
    starts = [0]
    for s in sizes[:-1]:
        starts.append(starts[-1] + s)
    return starts


def block_boundaries(num_steps, sizes):
    """Video-sigma boundaries of a partition: descending, len(sizes)+1, ends at 0."""
    fine = fine_sigmas(VIDEO_SHIFT, num_steps)
    knots = partition_starts(sizes) + [num_steps]
    return [fine[k] for k in knots]


def fuse_heads(bank_w, bank_b, fine, sizes):
    """Fuse per-fine-interval velocity heads into one head per partition block.

    Plan weights are the block's fine step sizes normalized to the block span,
    per modality — identical to the reference einsum('pn,noi->poi', plan, bank).
    """
    d = [fine[k] - fine[k + 1] for k in range(bank_w.shape[0])]
    w64 = bank_w.to(torch.float64)
    b64 = bank_b.to(torch.float64)
    fused_w = []
    fused_b = []
    for start, size in zip(partition_starts(sizes), sizes):
        ks = list(range(start, start + size))
        span = sum(d[k] for k in ks)
        fw = sum((d[k] / span) * w64[k] for k in ks)
        fb = sum((d[k] / span) * b64[k] for k in ks)
        fused_w.append(fw.to(torch.float32))
        fused_b.append(fb.to(torch.float32))
    return torch.stack(fused_w).contiguous(), torch.stack(fused_b).contiguous()


def select_block(sigma, bounds, on_off_grid):
    """Map a step-start sigma to its trained block. bounds is descending, len nfe+1."""
    nfe = len(bounds) - 1
    for b in range(nfe):
        if abs(sigma - bounds[b]) <= KNOT_TOLERANCE:
            return b
    if abs(sigma - bounds[-1]) <= KNOT_TOLERANCE:
        return nfe - 1
    if on_off_grid == "error":
        pretty = ", ".join(f"{s:.6f}" for s in bounds)
        raise ValueError(
            f"MiniMaxH3PDDAccApply: model evaluated at sigma {sigma:.6f}, which is not a trained "
            f"PDD block boundary [{pretty}]. Use the `sigmas` output of the Apply node (or the "
            f"MiniMaxH3PDDAccScheduler) with the plain `euler` sampler — multi-stage samplers "
            f"(er_sde, dpmpp, res_*) evaluate off-grid and cannot drive PDD heads. "
            f"Set on_off_grid=clamp to force nearest-block instead (degraded output)."
        )
    if sigma >= bounds[0]:
        return 0
    for b in range(nfe):
        if sigma > bounds[b + 1]:
            return b
    return nfe - 1


class HeadBank:
    """Fused per-block head weights, lazily mirrored to each device used."""

    def __init__(self, vw, vb, aw, ab):
        self.cpu = (vw, vb, aw, ab)
        self._cache = {}

    def for_device(self, device):
        d = torch.device(device)
        if d.type == "cpu":
            return self.cpu
        got = self._cache.get(d)
        if got is None:
            got = tuple(t.to(d) for t in self.cpu)
            self._cache[d] = got
        return got


class _PDDHeadLinear(torch.nn.Module):
    """Transient stand-in for final_layer.video_out / audio_out during a
    patched forward: applies the armed block's fused head, optionally blended
    with the native projection at head_strength."""

    def __init__(self, native, heads, is_video, holder, strength):
        super().__init__()
        self._native = (native,)  # tuple hides it from nn.Module registration
        self.heads = heads
        self.is_video = is_video
        self.holder = holder
        self.strength = strength

    def forward(self, h):
        vW, vB, aW, aB = self.heads.for_device(h.device)
        W, B = (vW, vB) if self.is_video else (aW, aB)
        blk = self.holder.blk
        out = F.linear(h, W[blk].to(h.dtype), B[blk].to(h.dtype))
        if self.strength != 1.0:
            native = self._native[0](h)
            out = native + (out - native) * self.strength
        return out

    # Core PR #15908 ("MiniMax-H3: Support PDD LoRA") makes FinalLayer.forward
    # probe `self.video_out.weight.shape[0] // self.video_out.out_features` to
    # detect PDD-fused checkpoints BEFORE calling the projection. During our
    # swap these must resolve to the NATIVE module's attributes: the native
    # head keeps its original size, so core computes n == 1, takes its stock
    # path, and calls video_out(h) — which is this module's forward.
    @property
    def weight(self):
        return self._native[0].weight

    @property
    def bias(self):
        return self._native[0].bias

    @property
    def in_features(self):
        return self._native[0].in_features

    @property
    def out_features(self):
        return self._native[0].out_features


def make_pdd_final_forward(final_layer, heads, holder, bounds, on_off_grid, strength):
    """Bound replacement for final_layer.forward.

    Delegates to the CLASS's own forward with the two projection modules
    temporarily swapped for the armed per-block fused heads. Core-version
    proof: every ComfyUI since the initial H3 merge mods the streams and then
    calls self.video_out(hv) / self.audio_out(ha), so the mod math (which
    changed in #15375 — the private _mod_row helper) is never replicated here.
    """
    orig_forward = type(final_layer).forward
    v_head = _PDDHeadLinear(final_layer.video_out, heads, True, holder, strength)
    a_head = _PDDHeadLinear(final_layer.audio_out, heads, False, holder, strength)

    def pdd_final_forward(self, x, t_emb, *args, **kwargs):
        if holder.sigma_v is None:
            raise RuntimeError(
                "MiniMaxH3PDDAccApply: final-layer patch invoked without active diffusion-call "
                "state. Do not stack caching packs (T8 blockcache / EasyCache / Spectrum) on a "
                "PDD-patched model.")
        holder.blk = select_block(holder.sigma_v, bounds, on_off_grid)
        native_v, native_a = self.video_out, self.audio_out
        try:
            self.video_out, self.audio_out = v_head, a_head
            return orig_forward(self, x, t_emb, *args, **kwargs)
        finally:
            self.video_out, self.audio_out = native_v, native_a
            holder.blk = None

    return pdd_final_forward.__get__(final_layer, final_layer.__class__)


PARTITION_TOLERANCE = 0.015   # >> cast/storage noise (~2e-3), << trunk gap (0.0503)


KNOWN_PARTITIONS = ("fl2va", "ref2va")


def partition_from_name(*names):
    """Which trunk a file targets, from its metadata/filename strings.

    Returns "fl2va" / "ref2va", or None when no string names exactly one
    partition token (unknown or ambiguous)."""
    for s in names:
        s = (s or "").lower()
        hits = {p for p in KNOWN_PARTITIONS if p in s}
        if len(hits) == 1:
            return hits.pop()
    return None


def identify_trunk(video_out_weight, fingerprints, tol=PARTITION_TOLERANCE):
    """Match a live final_layer.video_out.weight against shipped fingerprints.

    fingerprints: {partition: weight tensor}. Returns (partition or None,
    {partition: relative distance}).

    This is an ADVISORY check and must never abort a render, so every failure
    degrades to "no match". The non-obvious failure (issue #3): GGUF trunks
    hand us a tensor subclass whose reported .shape is the LOGICAL shape while
    arithmetic runs on the packed storage — the shape gate passes and the
    subtraction then raises a size mismatch (the reporter's 5376 -> 3024 is
    exactly Q4_K's 9/16 packing). An opaque quantized weight is simply
    unmatchable; `check_partition_pairing` already words the INCONCLUSIVE
    note for that case."""
    try:
        live = video_out_weight.detach().to(torch.float32).cpu()
    except Exception:
        return None, {}
    dists = {}
    for name, ref in fingerprints.items():
        ref = ref.to(torch.float32)
        if ref.shape != live.shape:
            continue
        try:
            dists[name] = float((live - ref).norm() / ref.norm())
        except Exception:
            continue
    best = min(dists, key=dists.get) if dists else None
    if best is not None and dists[best] <= tol:
        return best, dists
    return None, dists


def check_partition_pairing(video_out_weight, fingerprints, file_partition,
                            pdd_file, mode="error"):
    """Refuse (or warn about) a PDD file applied to the other trunk's model.

    Returns a note string for the info output. Raises ValueError only when BOTH
    sides are confidently identified and disagree (mode="error"); an unknown
    model (finetune/hybrid) or unidentifiable file degrades to a warning note —
    the check must never block checkpoints it has no fingerprint for."""
    if not fingerprints:
        return "partition check skipped: no fingerprints shipped"
    model_partition, dists = identify_trunk(video_out_weight, fingerprints)
    dtxt = ", ".join(f"{k} {v:.4f}" for k, v in sorted(dists.items())) or "no shape match"
    if model_partition is None:
        return (f"partition check INCONCLUSIVE: model video_out matches no known trunk "
                f"({dtxt}; tol {PARTITION_TOLERANCE}) — finetune or full-merge? "
                f"file/model pairing not verified")
    if file_partition is None:
        return (f"partition check INCONCLUSIVE: model is {model_partition} "
                f"({dtxt}) but the file's trunk could not be identified from its "
                f"name/metadata — pairing not verified")
    if file_partition != model_partition:
        msg = (f"{pdd_file} is a {file_partition.upper()} distill but the loaded UNET is the "
               f"{model_partition} trunk (video_out distance {dtxt}, trunks sit ~0.05 apart). "
               f"The key sets are identical, so this would apply cleanly and render silently "
               f"wrong — pair FL2VA with an fl2va UNET, Ref2VA with ref2va. If this pairing "
               f"is a deliberate experiment, set partition_check to 'warn'.")
        if mode == "error":
            raise ValueError(f"MiniMaxH3PDDAccApply: {msg}")
        return f"partition MISMATCH (continuing, partition_check=warn): {msg}"
    return f"partition check ok: {file_partition} file on {model_partition} model ({dtxt})"
