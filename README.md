# ComfyUI-MiniMax-H3-PTD

ComfyUI support for the **MiniMax-H3 PTD per-hop acceleration LoRAs**
([taobao-content-ai/MiniMax-H3-PTD-LoRAs](https://huggingface.co/taobao-content-ai/MiniMax-H3-PTD-LoRAs)) —
full audio+video generation in **8 or 4 sampler steps**, no CFG.

> **Status: unrendered.** The loading path is tested in depth (see [Tests](#tests)) and verified
> against the real ref2va trunk, but nobody has published a quality comparison of PTD vs PDD on
> MiniMax-H3 yet, and we have not finished one either. Treat output quality as unknown.

## What PTD is, and why it needs its own node

PTD ([Parametric Trajectory Distillation](https://alan-lanfeng.github.io/PTD/), Feng et al. 2026)
keeps PDD's **per-interval output head bank** — 32 copies of the final-layer video/audio
projections, fused into one mean-block-velocity head per sampler step — and replaces PDD's single
shared trunk LoRA with **one independent low-rank adapter per sampling step** ("hop"). Hop *h* is
selected for block *h* of a uniform 32/hops grid, so an 8-hop checkpoint runs in 8 NFE and a 4-hop
checkpoint in 4.

The head-bank keys and shapes are byte-identical to the
[alibaba-pai PDD Acc](https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs) release; the only
structural differences are a leading hop axis on every adapter and **no adaln LoRA at all**:

```
PDD Acc 8Step   rank 64,  alpha 64    to_q,to_k,to_v,to_out.0,ff.net.0.proj,ff.net.2,adaln_proj.linear
PTD   4Step     rank 128, alpha 128   to_q,to_k,to_v,to_out.0,ff.net.0.proj,ff.net.2
      transformer_blocks.N.attn.to_q.lora_down   [64, 5376]  ->  [H, 128, 5376]
```

**A per-hop adapter cannot be baked or flattened.** Baking would need one 32.4 GB trunk per hop;
re-patching 208 modules between every step means dequantise → add → requantise on an int8 build.
So this pack leaves the base weights untouched and gives each target `Linear` a parallel low-rank
branch whose hop slice is selected per step. Two things fall out of that: quantized weights are
never dequantised, and character LoRAs keep composing through the normal `add_patches` path
underneath.

**There is also no cheap approximation.** Measured on the 8-step release, the hops are mutually
orthogonal — median pairwise cosine between hop deltas **+0.001**, i.e. no more aligned than
random low-rank directions. For H orthogonal deltas of equal norm,
`‖M_h − M_mean‖ / ‖M_mean‖ = √(H−1) = 2.646`; measured 2.657. Collapsing the hops to a single
static LoRA therefore destroys 7/8 of the signal. (`slice_ptd_hop.py` will produce such a slice
anyway — as a plumbing probe, never as a production path.) The hops are likewise orthogonal to
PDD's own adapter (+0.0003) at roughly twice its magnitude, so PTD is an independent solution
rather than "PDD plus corrections".

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Jalen-Brunson/ComfyUI-MiniMax-H3-PTD
```

No dependencies beyond ComfyUI. Needs a core new enough for the MiniMax-H3 carried-audio rework
([ComfyUI#15243](https://github.com/comfyanonymous/ComfyUI/pull/15243), v0.33.0+); the node checks
and refuses otherwise.

Put the checkpoints in `ComfyUI/models/ptd/`:

| File | NFE | hops | block | size |
|---|---|---|---|---|
| `MiniMax-H3-Ref2VA-PTD-8Step.safetensors` | 8 | 8 | 4 | 11.1 GB |
| `MiniMax-H3-Ref2VA-PTD-4Step.safetensors` | 4 | 4 | 8 | 5.58 GB |

**Ref2VA only** — there is no FL2VA PTD release.

## Use

`MiniMax H3 PTD Per-Hop LoRA (Apply)` takes a `MODEL` and returns `MODEL` + `SIGMAS` + `info`.
A worked graph is in [`example_workflows/`](example_workflows), derived from ComfyUI's stock
`video_minimax_h3_r2v` template.

**The recipe is mandatory:**

- **`euler`**, driven by the node's own `sigmas` output. Multi-stage samplers (`res_multistep`,
  `er_sde`, `dpmpp`, `res_*`) evaluate the model *between* the trained block boundaries, which the
  heads cannot serve — the node fails closed on an off-grid sigma rather than degrade quietly.
- **CFG 1.0** (`BasicGuider`, or `CFGGuider` at 1.0).
- **SigmaShift 12 / 3.** The model already defaults to these, and the node refuses anything else.
- **A clean `ref2va` trunk.** PTD carries its own head bank, so a checkpoint that already has one
  baked in (`..._pdd_acc_..._baked_with_heads_...`) is rejected with an explicit message.
- No turbo/Lightning LoRA, no T8 BlockCache / EasyCache / Spectrum, and no PDD Acc node in the
  same chain.

### Steps (`nfe`)

Leave it on **`auto`**: the step count comes from the file's `ptd_num_hops`, because hop *h* was
trained for block *h* of a uniform 32/hops grid. Switching 8-step ↔ 4-step is just the `ptd_file`
dropdown.

The override exists for experiments. Picking a different `nfe` re-partitions the 32-step grid and
maps each block to **the hop trained for the sigma region that step starts in** — hop
`block_start // trained_block`:

| file | nfe | blocks | hops used |
|---|---|---|---|
| 8-step | `auto` (8) | 4,4,4,4,4,4,4,4 | 0,1,2,3,4,5,6,7 — trained |
| 8-step | 4 | 8,8,8,8 | 0,2,4,6 |
| 8-step | 6 | 8,8,4,4,4,4 | 0,2,4,5,6,7 |
| 4-step | `auto` (4) | 8,8,8,8 | 0,1,2,3 — trained |
| 4-step | 8 | 4,4,4,4,… | *refused* — blocks of 4 are outside this file's head envelope (8, 16) |

Fewer, larger steps therefore subsample the same trajectory rather than mismatching hops to
sigmas. It is still off the distilled configuration, so the node says so loudly in `info` and the
log. Block sizes the PDD heads never saw are refused outright unless you also set
`allow_off_envelope` (expect heavy noise).

Both widgets are appended **last** deliberately — inserting them mid-list would shift the stored
values of every already-saved graph.

### `residency`

| | VRAM | per step |
|---|---|---|
| `streamed` (default) | ~1.38 GiB (one hop) | one ~1.38 GiB H2D |
| `resident` | 5.6 / 11.1 GiB (whole stack) | none |

The whole stack is packed into one flat `[H, N]` buffer, so a hop change is a single copy rather
than 624 small ones. Per-hop size does not depend on hop count, so **`streamed` PTD-8 costs less
VRAM than `resident` PTD-4**. Allocation is lazy — it happens inside the sampling wrapper, after
model management has placed the UNET, so a resident stack cannot provoke a partial UNET offload at
load time.

## Implementation notes

Two things in core will silently break a naive implementation, and are worth knowing if you are
writing something similar:

**`fc2` never calls its own `forward`.** Core's MLP is
`comfy.ops.linear_input_act(self.fc2, self.fc1(x), "swiglu")`, which on a quantized or
fp16-accumulate path reads `fc2.weight` directly and fuses SwiGLU into the GEMM epilogue. A patched
`fc2.forward` is a **silent no-op** there — 52 of 208 adapters would vanish with no error. The fix
is `comfy_force_forward=True` on those modules, which routes them back through
`linear(_eager_input_act(x, "swiglu"))` so the forward runs and receives the post-SwiGLU activation
(exactly what diffusers' `ff.net.2` consumes). Covered by `test_fc2_bypass_is_real_...`, which
reproduces the bypass rather than mocking it.

**`x` is a list, not a tensor.** MiniMax-H3 hands `DIFFUSION_MODEL` wrappers
`x = [video_latent, audio_latent]` (core reads `x[0].device`). This pack binds its buffer
per-module, inside each sidecar forward where the input tensor is unambiguous — which is also
offload- and multi-device-safe — with the wrapper only doing a best-effort eager bind.

Other mapping details, all shared with the PDD Acc pack: core fuses q/k/v into one `qkv_proj`
(three branches are computed against one stacked down-projection and written into their own slices
of the fused output — exact, and cheaper than a block-diagonal rank-3r form), and core's
`_swiglu_eager` chunks `[gate; value]` where diffusers packs `[value; gate]`, so `fc1`'s `lora_up`
row-halves are swapped once at load.

`pdd_heads.py` is a **vendored subset** of `pdd_acc_core.py` from
[ComfyUI-MiniMax-H3-PDD-Acc](https://github.com/Jalen-Brunson/ComfyUI-MiniMax-H3-PDD-Acc) —
function bodies are extracted mechanically and are byte-identical. Fix bugs there first, then
re-vendor.

## Tests

```bash
PYTHONPATH=/path/to/ComfyUI python3 tests/test_ptd.py
```

16 tests. The strongest builds a real core `RefinerBlock` and compares the sidecar against the same
block with `B@A` **merged into its weights** — merging being the definition of applying a LoRA — so
agreement proves fused-qkv row order, the fc1 half-swap, the fc2 path and hop selection at once
(with a guard that the LoRA is not a no-op). Also covered: streamed ≡ resident bit-for-bit,
`lora_strength 0` and unarmed are byte-identical to base, the wrapper's `[video, audio]` calling
convention, and that a de-hopped slice or a PDD file is refused with a pointed message.

## Credit

- PTD: Lan Feng et al., *Parametric Trajectory Distillation for Few-Step Video Generation* (2026) —
  [project page](https://alan-lanfeng.github.io/PTD/)
- PDD: Neta Shaul et al., *Parallel Decoding Distillation for Fast Image and Video Generation* —
  [arXiv:2607.26004](https://arxiv.org/abs/2607.26004)
- Weights: [taobao-content-ai](https://huggingface.co/taobao-content-ai/MiniMax-H3-PTD-LoRAs),
  under the
  [MiniMax-H3 community license](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)
- Trunk-pairing fingerprint guard after
  [fblissjr/ComfyUI-h3-explorations](https://github.com/fblissjr/ComfyUI-h3-explorations)

Apache-2.0 (this pack). Model weights keep their own license.
