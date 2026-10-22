"""ComfyUI node for the MiniMax-H3 PTD per-hop acceleration LoRAs.

PTD (taobao-content-ai/MiniMax-H3-PTD-LoRAs) pairs PDD's per-interval output
head bank with ONE INDEPENDENT LoRA PER SAMPLING STEP on the trunk. The head
bank is loaded and fused exactly as in ComfyUI-MiniMax-H3-PDD-Acc (see
pdd_heads.py, a vendored subset of that project); the per-hop trunk adapter is
this pack's own machinery -- see ptd_core.py for why it cannot be baked.
"""

import inspect
import logging
import os
from types import SimpleNamespace

import torch

import comfy.patcher_extension
import comfy.utils
import folder_paths

try:
    from .pdd_heads import (
        AUDIO_SHIFT, VIDEO_SHIFT, HeadBank, block_boundaries, check_partition_pairing,
        fine_sigmas, fuse_heads, make_pdd_final_forward, partition_from_name,
        partition_off_envelope, resolve_partition, validate_sigma_shifts,
    )
    from .ptd_core import (
        PTDStack, attach_ptd_sidecar, hop_map, make_ptd_wrapper, make_raw_text_preprocess,
        plan_modules, split_ptd_state_dict,
    )
except ImportError:
    from pdd_heads import (
        AUDIO_SHIFT, VIDEO_SHIFT, HeadBank, block_boundaries, check_partition_pairing,
        fine_sigmas, fuse_heads, make_pdd_final_forward, partition_from_name,
        partition_off_envelope, resolve_partition, validate_sigma_shifts,
    )
    from ptd_core import (
        PTDStack, attach_ptd_sidecar, hop_map, make_ptd_wrapper, make_raw_text_preprocess,
        plan_modules, split_ptd_state_dict,
    )

WRAPPER_KEY = "minimax_h3_ptd"

_ptd_dir = os.path.join(folder_paths.models_dir, "ptd")
os.makedirs(_ptd_dir, exist_ok=True)
folder_paths.add_model_folder_path("ptd", _ptd_dir, is_default=True)

_fingerprints = None


def _load_partition_fingerprints():
    """{trunk: fp16 video_out weight} from partition_fingerprints/, cached."""
    global _fingerprints
    if _fingerprints is None:
        _fingerprints = {}
        fp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "partition_fingerprints")
        if os.path.isdir(fp_dir):
            from safetensors.torch import load_file
            for name in sorted(os.listdir(fp_dir)):
                if name.startswith("video_out_") and name.endswith(".safetensors"):
                    trunk = name[len("video_out_"):-len(".safetensors")]
                    _fingerprints[trunk] = load_file(
                        os.path.join(fp_dir, name))["video_out_weight"]
    return _fingerprints


def _check_core_supported():
    """Fail closed on cores predating the MiniMax-H3 carried-audio rework (#15243):
    the heads emit mean block velocities and rely on core's carried-variable audio
    mapping -- older stock-world mechanics would silently mis-integrate audio."""
    try:
        from comfy.ldm.minimax.model import MiniMaxH3Model
        src = inspect.getsource(MiniMaxH3Model.forward)
    except Exception:
        return  # can't probe (source unavailable) -- don't block on that alone
    if "audio_scale" not in src:
        raise RuntimeError(
            "MiniMaxH3PTDApply: this ComfyUI predates the MiniMax-H3 audio-mechanics rework "
            "(comfyanonymous/ComfyUI#15243), so the heads would mis-integrate audio. "
            "Update ComfyUI to v0.33.0 or newer.")


def _sigmas_tensor(bounds):
    # Snap the float32 endpoints to their analytic values -- but ONLY when they
    # already are those endpoints: an unconditional t[0] = 1.0 would silently
    # rewrite a denoise-sliced schedule and discard a resume latent at full noise.
    t = torch.tensor(bounds, dtype=torch.float32)
    if abs(float(t[0]) - 1.0) < 1e-4:
        t[0] = 1.0
    if abs(float(t[-1])) < 1e-4:
        t[-1] = 0.0
    return t


class MiniMaxH3PTDApply:
    """Per-hop PTD LoRA + PDD head bank.

    `nfe` is LOCKED to the file's `ptd_num_hops`: hop h was trained for block h
    of a uniform 32/hops grid, so there is no steps or partition widget. Drive
    the sampler with the `sigmas` output, plain `euler`, CFG 1.0, SigmaShift 12/3.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "ptd_file": (folder_paths.get_filename_list("ptd"),
                             {"tooltip": "MiniMax-H3-Ref2VA-PTD-{8,4}Step.safetensors (bf16) or "
                                         "nynxz's ptd_h3_ref2va_*_int8 files in models/ptd/. "
                                         "nfe is read from the file."}),
                "lora_strength": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0,
                                            "step": 0.01}),
                "head_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0,
                                            "step": 0.01}),
                "residency": (["streamed", "resident"], {
                    "default": "streamed",
                    "tooltip": "streamed: one hop on the GPU (~1.4 GiB), refreshed by a single "
                               "H2D per step. resident: the whole stack (5.6/11.1 GiB), no "
                               "per-step copy, but it can provoke a partial UNET offload."}),
                "on_off_grid": (["error", "clamp"], {
                    "default": "error",
                    "tooltip": "What to do when the model is evaluated away from a trained "
                               "block boundary (a multi-stage sampler will do this)."}),
            },
            "optional": {
                "enabled": ("BOOLEAN", {"default": True}),
                "bypass_sigmas": ("SIGMAS", {"tooltip": "Passed through when enabled=False."}),
                "partition_check": (["error", "warn", "off"], {"default": "error"}),
                "pin_memory": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Pin the CPU stack (streamed only): faster per-step H2D."}),
                # appended last on purpose: inserting a widget mid-list shifts the
                # values of every already-saved graph
                "nfe": (["auto", "8", "6", "4", "10", "12"], {
                    "default": "auto",
                    "tooltip": "auto = the file's ptd_num_hops, the ONLY configuration PTD was "
                               "distilled for. Anything else re-partitions the 32-step grid and "
                               "each step falls back to the hop trained for the sigma region it "
                               "starts in (8 hops at nfe 4 -> hops 0,2,4,6). Experiment only."}),
                "allow_off_envelope": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Permit block sizes the PDD heads never saw. Expect heavy noise."}),
            },
        }

    RETURN_TYPES = ("MODEL", "SIGMAS", "STRING")
    RETURN_NAMES = ("model", "sigmas", "info")
    FUNCTION = "apply"
    CATEGORY = "advanced/model"

    def apply(self, model, ptd_file, lora_strength, head_strength, residency, on_off_grid,
              enabled=True, bypass_sigmas=None, partition_check="error", pin_memory=True,
              nfe="auto", allow_off_envelope=False):
        if not enabled:
            if bypass_sigmas is None:
                raise ValueError(
                    "MiniMaxH3PTDApply: enabled=False needs bypass_sigmas wired (the schedule "
                    "for the un-distilled model) -- the PTD block-boundary sigmas would be "
                    "wrong for an unpatched model.")
            return (model, bypass_sigmas,
                    "PTD BYPASSED (enabled=False): model and bypass_sigmas passed through.")
        _check_core_supported()
        path = folder_paths.get_full_path_or_raise("ptd", ptd_file)
        sd, metadata = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
        adapters, (vw, vb, aw, ab), config = split_ptd_state_dict(sd, metadata, ptd_file)
        hops, num_steps = config["hops"], config["num_steps"]
        plan = plan_modules(adapters, config, ptd_file)
        del adapters, sd

        # nfe == hops is the trained configuration: hop h <-> block h of the
        # uniform 32/hops grid. An override re-partitions the grid and maps each
        # block to the hop trained for the sigma region it starts in.
        trained_block = config["block_size"]
        sel = str(nfe or "auto").strip().lower()
        want_nfe = hops if sel in ("", "auto") else int(sel)
        sizes = resolve_partition(num_steps, want_nfe, trained_block=trained_block,
                                  allow_off_envelope=allow_off_envelope)
        hop_of_block = hop_map(sizes, trained_block, hops)

        nfe_note = ""
        if want_nfe != hops:
            nfe_note = (
                f"WARNING: nfe {want_nfe} != the file's {hops} hops. PTD was distilled ONLY for "
                f"{hops} steps; blocks {','.join(map(str, sizes))} now run hops "
                f"{','.join(map(str, hop_of_block))}, each the adapter trained for the sigma "
                f"region that step starts in. Experiment -- A/B against nfe auto.\n")
            logging.warning("MiniMaxH3PTDApply: %s", nfe_note.strip())

        off_env = partition_off_envelope(sizes, trained_block=trained_block)
        off_env_note = ""
        if off_env:
            off_env_note = (f"WARNING: block size {off_env} is outside this file's PDD head "
                            f"envelope ({trained_block}, {2 * trained_block}) -- expect heavy "
                            f"noise.\n")
            logging.warning("MiniMaxH3PTDApply: %s", off_env_note.strip())

        meta = metadata or {}
        if partition_check == "off":
            fp_note = "model-type check OFF (partition_check=off)"
        else:
            fp_note = check_partition_pairing(
                model.get_model_object("diffusion_model.final_layer").video_out.weight,
                _load_partition_fingerprints(),
                partition_from_name(meta.get("pdd_partition"), meta.get("source_file"), ptd_file),
                ptd_file, mode=partition_check)
        logging.info("MiniMaxH3PTDApply: %s", fp_note)

        m = model.clone()

        # ---- head bank (same layout as PDD; reuse its fusion + final-layer patch) ----
        final_layer = m.get_model_object("diffusion_model.final_layer")
        for name, bank in (("video_out", vw), ("audio_out", aw)):
            native = getattr(final_layer, name).weight
            if tuple(native.shape) != tuple(bank.shape[1:]):
                raise ValueError(
                    f"MiniMaxH3PTDApply: head bank {name} {list(bank.shape[1:])} does not match "
                    f"model final_layer.{name} {list(native.shape)}. PTD brings its own head "
                    f"bank, so it needs a CLEAN ref2va trunk -- not a '..._baked_with_heads_...' "
                    f"checkpoint that already carries one.")
        # Audio needs NO chord/tangent correction here (taobao's and the compact pack's
        # schedule.audio_head_corrections): those run audio on ComfyUI-PTD's tangent clock.
        # core carries audio as (sigma_v/sigma_a)*x_audio, which makes a plain euler step land
        # exactly on the chord the mean-velocity head was trained for -- porting the factor
        # would double-correct (tests/test_e2e.py). The 1.0-endpoint grid is also the trained
        # one: taobao's README says Diffusers and training use 1.0; ComfyUI-PTD's 0.999 is
        # its own clock, not the checkpoint's.
        vW, vB = fuse_heads(vw, vb, fine_sigmas(VIDEO_SHIFT, num_steps), sizes)
        aW, aB = fuse_heads(aw, ab, fine_sigmas(AUDIO_SHIFT, num_steps), sizes)
        heads = HeadBank(vW, vB, aW, aB)
        bounds = block_boundaries(num_steps, sizes)

        holder = SimpleNamespace(sigma_v=None, blk=None)

        # ---- per-hop trunk sidecar ----
        if lora_strength == 0.0:
            stack = None
            sidecar_note = ("sidecar: SKIPPED (lora_strength 0.0) -- head bank only, which is "
                            "off the training envelope")
        else:
            stack = PTDStack(plan, hops, residency=residency, pin=pin_memory)
            _, sidecar_note = attach_ptd_sidecar(m, plan, stack, lora_strength)
            if any(e["path"].startswith("token_refiner.") for e in plan):
                # see "The refiner trap" in ptd_core.py
                m.add_object_patch(
                    "diffusion_model.preprocess_text_embeds",
                    make_raw_text_preprocess(m.get_model_object("diffusion_model").hidden_size))
                sidecar_note += "; token_refiner runs per hop (raw text kept for the forward)"

        m.add_object_patch(
            "diffusion_model.final_layer.forward",
            make_pdd_final_forward(final_layer, heads, holder, bounds, on_off_grid, head_strength))
        m.remove_wrappers_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY)
        m.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY,
            make_ptd_wrapper(holder, stack, bounds, on_off_grid,
                             lambda ex, to, kw: validate_sigma_shifts(
                                 ex, to, kw, "MiniMaxH3PTDApply"),
                             hop_of_block=hop_of_block))

        sigmas = _sigmas_tensor(bounds)
        info = (
            f"{nfe_note}{off_env_note}"
            f"PTD: {ptd_file} ({config['storage']})\n"
            f"grid {num_steps} fine steps -> {hops} hops x block {trained_block} trained; "
            f"running nfe {len(sizes)} blocks {','.join(map(str, sizes))} "
            f"-> hops {','.join(map(str, hop_of_block))}"
            f"{' (trained configuration)' if want_nfe == hops else ' (OVERRIDE)'}\n"
            f"{sidecar_note}\n"
            f"{fp_note}\n"
            f"heads: video {list(vW.shape)}, audio {list(aW.shape)} fp32, "
            f"strength {head_strength}\n"
            f"sigmas: {', '.join(f'{s:.6f}' for s in bounds)}\n"
            f"recipe: euler + these sigmas, CFG 1.0, SigmaShift 12/3. No turbo LoRA, no "
            f"T8/EasyCache/Spectrum, no PDD Acc node in the same chain."
        )
        logging.info("MiniMaxH3PTDApply: %s (%d hops, nfe %d) %s",
                     ptd_file, hops, len(sizes), sidecar_note)
        return (m, sigmas, info)


NODE_CLASS_MAPPINGS = {"MiniMaxH3PTDApply": MiniMaxH3PTDApply}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniMaxH3PTDApply": "MiniMax H3 PTD Per-Hop LoRA (Apply)"}
