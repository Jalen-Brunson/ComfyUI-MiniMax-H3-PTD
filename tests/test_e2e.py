"""End-to-end tests on a tiny REAL native MiniMax-H3 (CPU, fp32).

The node's MODEL goes through core's own extra_conds -> sampler -> forward path.
Module-level tests (test_ptd.py) cannot see what happens around the forward, and
that is exactly where the refiner trap (ptd_core.py) lived.

    PYTHONPATH=/workspace/ComfyUI python3 custom_nodes/ComfyUI-MiniMax-H3-PTD/tests/test_e2e.py
"""
import copy
import importlib.util
import os
import sys
import tempfile
import traceback

import torch
from safetensors.torch import save_file

import comfy.cli_args
comfy.cli_args.args.cpu = True
import comfy.model_management  # noqa: E402
import comfy.model_patcher  # noqa: E402
import comfy.nested_tensor  # noqa: E402
import comfy.samplers  # noqa: E402
import comfy.supported_models  # noqa: E402
import folder_paths  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)
COMPACT_PDD = os.path.join(os.path.dirname(PACK), "ComfyUI-PTD-Compact", "pdd.py")

# Load the pack the way ComfyUI does (as a package), so nodes.py's relative imports
# resolve and ComfyUI's own top-level `nodes` module is not shadowed.
_spec = importlib.util.spec_from_file_location(
    "ptd_pack", os.path.join(PACK, "__init__.py"), submodule_search_locations=[PACK])
_pack = importlib.util.module_from_spec(_spec)
sys.modules["ptd_pack"] = _pack
_spec.loader.exec_module(_pack)
from ptd_pack import nodes as ptd_nodes  # noqa: E402
from ptd_pack.pdd_heads import (  # noqa: E402
    AUDIO_SHIFT, VIDEO_SHIFT, block_boundaries, fine_sigmas, fuse_heads,
)

TESTS, SKIP = [], []

# compact's tests/test_stacking.py tiny-model recipe
DIMS = dict(hidden_size=64, num_layers=2, token_refiner_num_layers=1, num_attention_heads=2,
            attention_head_dim=32, ffn_hidden_size=48, text_dim=40, timestep_input_dim=16,
            time_embed_hidden_size=32, time_embed_dim=16, rope_inv_freq_len=4)
HOPS, RANK = 4, 8
# diffusers suffix -> (comfy suffix, slice of the comfy weight rows it feeds)
PARTS = (("attn.to_q", "attn.qkv_proj"), ("attn.to_k", "attn.qkv_proj"),
         ("attn.to_v", "attn.qkv_proj"), ("attn.to_out.0", "attn.out_proj"),
         ("ff.net.0.proj", "mlp.fc1"), ("ff.net.2", "mlp.fc2"))


def test(fn):
    TESTS.append(fn)
    return fn


def native_model():
    config = comfy.supported_models.MiniMaxH3(dict(image_model="minimax_h3", **DIMS))
    config.set_inference_dtype(torch.float32, None)
    model = config.get_model({})
    torch.manual_seed(0)
    with torch.no_grad():
        for p in model.diffusion_model.parameters():
            p.normal_(0, 0.05)
        dm = model.diffusion_model
        n = dm.rope.inv_freq.numel()
        dm.rope.inv_freq.copy_(1.0 / 10000 ** (torch.arange(n, dtype=torch.float32) / n))
    cpu = torch.device("cpu")
    return comfy.model_patcher.ModelPatcher(model, cpu, cpu)


def write_ptd_file(folder, native, hops=HOPS, rank=RANK):
    """A synthetic taobao-layout PTD file sized from the native model's own modules."""
    g = torch.Generator().manual_seed(1)

    def t(*shape):
        return (torch.randn(*shape, generator=g) * 0.05).to(torch.bfloat16)

    sd = {}
    blocks = [(f"transformer_blocks.{i}", f"blocks.{i}") for i in range(len(native.blocks))]
    blocks += [(f"token_refiner.refiner_blocks.{i}", f"token_refiner.blocks.{i}")
               for i in range(len(native.token_refiner.blocks))]
    for src, dst in blocks:
        for part, comfy_part in PARTS:
            out_f, in_f = native.get_submodule(f"{dst}.{comfy_part}").weight.shape
            if comfy_part == "attn.qkv_proj":
                out_f //= 3
            sd[f"{src}.{part}.lora_down"] = t(hops, rank, in_f)
            sd[f"{src}.{part}.lora_up"] = t(hops, out_f, rank)
    for name, key in (("video_out", "proj_out"), ("audio_out", "audio_proj_out")):
        out_f, in_f = getattr(native.final_layer, name).weight.shape
        sd[f"{key}.weight"] = t(32, out_f, in_f)
        sd[f"{key}.bias"] = t(32, out_f)
    meta = {"ptd_num_hops": str(hops), "pdd_num_steps": "32", "lora_rank": str(rank),
            "lora_alpha": str(float(rank)),
            "lora_targets": "to_q,to_k,to_v,to_out.0,ff.net.0.proj,ff.net.2"}
    name = "tiny_ptd.safetensors"
    save_file(sd, os.path.join(folder, name), metadata=meta)
    folder_paths.add_model_folder_path("ptd", folder)
    return name, sd


def merged_refiner(base_refiner, sd, hop):
    """The token refiner with hop `hop`'s B@A merged into its weights (= the LoRA, by definition)."""
    ref = copy.deepcopy(base_refiner)
    with torch.no_grad():
        for i, block in enumerate(ref.blocks):
            src = f"token_refiner.refiner_blocks.{i}"

            def delta(part):
                A = sd[f"{src}.{part}.lora_down"][hop].float()
                B = sd[f"{src}.{part}.lora_up"][hop].float()
                return B, A

            block.attn.qkv_proj.weight += torch.cat(
                [B @ A for B, A in (delta(f"attn.{p}") for p in ("to_q", "to_k", "to_v"))], 0)
            B, A = delta("attn.to_out.0")
            block.attn.out_proj.weight += B @ A
            B, A = delta("ff.net.0.proj")
            half = B.shape[0] // 2
            block.mlp.fc1.weight += torch.cat([B[half:], B[:half]], 0) @ A  # [value;gate] -> [gate;value]
            B, A = delta("ff.net.2")
            block.mlp.fc2.weight += B @ A
    return ref


def sample(model, sigmas, seed=7):
    guider = comfy.samplers.CFGGuider(model)
    context = torch.randn(1, 6, DIMS["text_dim"], generator=torch.Generator().manual_seed(3))
    guider.set_conds([[context, {}]], [[context, {}]])
    guider.set_cfg(1.0)
    video, audio = torch.zeros(1, 24, 2, 4, 4), torch.zeros(1, 32, 2, 3)
    noise = comfy.nested_tensor.NestedTensor([
        torch.randn(video.shape, generator=torch.Generator().manual_seed(4)),
        torch.randn(audio.shape, generator=torch.Generator().manual_seed(5))])
    try:
        return guider.sample(noise, comfy.nested_tensor.NestedTensor([video, audio]),
                             comfy.samplers.sampler_object("euler"), sigmas,
                             disable_pbar=True, seed=seed)
    finally:
        comfy.model_management.unload_all_models()


def apply_node(base, name):
    m, sigmas, info = ptd_nodes.MiniMaxH3PTDApply().apply(
        base, name, 1.0, 1.0, "streamed", "error", partition_check="off", pin_memory=False)
    return m, sigmas, info


def capture_refiner(native):
    calls = []

    def hook(module, args, output):
        calls.append((args[0].detach().clone(), output.detach().clone()))

    return calls, native.token_refiner.register_forward_hook(hook)


# ---------------------------------------------------------------- tests

@test
def test_refiner_adapters_run_per_hop_through_core():
    """The fix: raw text reaches the forward, which refines it once per hop under that hop's adapters."""
    base = native_model()
    native = base.get_model_object("diffusion_model")
    base_refiner = copy.deepcopy(native.token_refiner)
    with tempfile.TemporaryDirectory() as tmp:
        name, sd = write_ptd_file(tmp, native)
        m, sigmas, info = apply_node(base, name)
        assert "token_refiner runs per hop" in info, info
        calls, handle = capture_refiner(native)
        try:
            sample(m, sigmas)
        finally:
            handle.remove()
    assert len(calls) == HOPS, f"{len(calls)} refiner calls, expected one per hop ({HOPS})"
    outs = []
    for hop, (inp, out) in enumerate(calls):
        with torch.no_grad():
            want = merged_refiner(base_refiner, sd, hop)(inp)
            plain = base_refiner(inp)
        scale = want.abs().max().item()
        err = (out - want).abs().max().item()
        assert err / scale < 1e-4, f"hop {hop}: refiner != merged weights ({err} / {scale})"
        assert (out - plain).abs().max().item() / scale > 1e-3, f"hop {hop}: adapters had no effect"
        outs.append(out)
    assert (outs[0] - outs[-1]).abs().max().item() > 1e-4, "every hop refined the text identically"


@test
def test_without_the_fix_the_refiner_runs_once_unarmed():
    """The trap, pinned: core refines in extra_conds, outside any hop, so the adapters never apply."""
    base = native_model()
    native = base.get_model_object("diffusion_model")
    base_refiner = copy.deepcopy(native.token_refiner)
    with tempfile.TemporaryDirectory() as tmp:
        name, _ = write_ptd_file(tmp, native)
        m, sigmas, _ = apply_node(base, name)
        m = m.clone()
        m.object_patches.pop("diffusion_model.preprocess_text_embeds")
        calls, handle = capture_refiner(native)
        try:
            sample(m, sigmas)
        finally:
            handle.remove()
    assert 0 < len(calls) < HOPS, f"{len(calls)} refiner calls: expected only the extra_conds pass"
    for inp, out in calls:
        with torch.no_grad():
            plain = base_refiner(inp)
        assert torch.allclose(out, plain, rtol=1e-5, atol=1e-6), "unarmed refiner was not the base"


@test
def test_core_audio_step_is_the_trained_chord():
    """core's carried audio + plain euler == x_a + (sigma_a_next - sigma_a) * u_a for a block-mean head.

    Calls core's REAL MiniMaxH3Model.forward (only _forward is stubbed to return the head's
    velocity), so this breaks if core ever changes its carried-audio conversion. Taobao's
    port and the compact pack need schedule.audio_head_corrections because ComfyUI-PTD
    converts audio with the clock tangent; ours must NOT add it.
    """
    from comfy.ldm.minimax.model import time_shift_sigma
    dm = native_model().get_model_object("diffusion_model")
    bounds = block_boundaries(32, [4] * 8)
    scale = VIDEO_SHIFT / AUDIO_SHIFT
    seen = {}

    def stub(x, timestep, context, transformer_options={}, **kwargs):
        seen["audio"] = x[1]
        return [torch.zeros_like(x[0]), seen["vbar"]]

    dm._forward = stub

    def carry(s):  # sigma_a / sigma_v; tends to AUDIO_SHIFT / VIDEO_SHIFT at sigma 0
        return time_shift_sigma(s, VIDEO_SHIFT, AUDIO_SHIFT) / s if s > 0 else 1.0 / scale

    g = torch.Generator().manual_seed(9)
    for k in range(8):
        s0, s1 = bounds[k], bounds[k + 1]
        a0 = time_shift_sigma(s0, VIDEO_SHIFT, AUDIO_SHIFT)
        a1 = time_shift_sigma(s1, VIDEO_SHIFT, AUDIO_SHIFT) if s1 > 0 else 0.0
        xa0 = torch.randn(1, 32, 2, 3, generator=g)
        seen["vbar"] = torch.randn(1, 32, 2, 3, generator=g)
        y0 = xa0 / carry(s0)                                   # the sampler's carried variable
        out = dm.forward([torch.zeros(1, 24, 1, 2, 2), y0], torch.tensor([s0 * 1000.0]),
                         torch.zeros(1, 2, DIMS["hidden_size"]), {},
                         minimax_payload={"audio_scale": scale})
        torch.testing.assert_close(seen["audio"], xa0, rtol=1e-5, atol=1e-6)
        y1 = y0 + (s1 - s0) * out[1]                           # euler: d = (x - denoised)/sigma = out
        chord = xa0 + (a1 - a0) * seen["vbar"]
        torch.testing.assert_close(y1 * carry(s1), chord, rtol=1e-4, atol=1e-5,
                                   msg=lambda m: f"hop {k}: carried euler != chord\n{m}")


@test
def test_head_plans_match_the_reference_port():
    """Cross-check our head fusion against the compact pack's (taobao's schedule.hop_plans)."""
    if not os.path.isfile(COMPACT_PDD):
        SKIP.append("head plans (ComfyUI-PTD-Compact not installed)")
        return
    spec = importlib.util.spec_from_file_location("compact_pdd", COMPACT_PDD)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    for hops in (4, 8):
        sizes = [32 // hops] * hops
        plans = ref.hop_plans(hops, 32, VIDEO_SHIFT, AUDIO_SHIFT)
        eye = torch.eye(32, dtype=torch.float64).unsqueeze(-1)
        for shift, idx in ((VIDEO_SHIFT, 0), (AUDIO_SHIFT, 1)):
            ours = fuse_heads(eye, torch.zeros(32, 1), fine_sigmas(shift, 32), sizes)[0].squeeze(-1)
            want = torch.stack([p[idx].reshape(-1) for p in plans]).float()
            torch.testing.assert_close(ours, want, rtol=1e-6, atol=1e-7)


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
