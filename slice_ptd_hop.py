#!/usr/bin/env python3
"""Slice one hop (or the hop-mean) out of a PTD per-hop LoRA into a PDD-shaped file.

PTD (taobao-content-ai/MiniMax-H3-PTD-LoRAs) is structurally an alibaba-pai PDD Acc
file with a leading hop axis on every adapter and no adaln_proj LoRA:

    PDD  transformer_blocks.N.attn.to_q.lora_down   [64, 5376]
    PTD  transformer_blocks.N.attn.to_q.lora_down   [H, 128, 5376]

head bank keys/shapes (proj_out, audio_proj_out) are IDENTICAL in both.

Dropping the hop axis therefore yields a file the existing MiniMaxH3PDDAccApply
loads through its normal "original format" path (convert_pdd_lora + fuse_heads),
with no runtime changes at all.

The result is NOT correct PTD: every sampling step gets one hop's adapter, where
the real thing selects hop == block index per step. It exists to validate the
plumbing (key mapping onto fused qkv_proj, the fc1 half-swap, rank-128 alphas,
the head bank, the sigma grid) before any per-hop sidecar is written -- and the
--hop mean output doubles as the "what does per-hop actually buy" baseline.

Usage:
    python3 slice_ptd_hop.py /workspace/models/ptd/MiniMax-H3-Ref2VA-PTD-8Step.safetensors
    python3 slice_ptd_hop.py <file> --hop 0 --hop 7 --hop mean
    python3 slice_ptd_hop.py <file> --hop all --out-dir /somewhere
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HEAD_KEYS = ("proj_out.weight", "proj_out.bias", "audio_proj_out.weight", "audio_proj_out.bias")
_CKPT_WRAP = "._checkpoint_wrapped_module"
_ADAPTER_TAILS = (".lora_down", ".lora_up")


def _clean(key: str) -> str:
    return key.replace(_CKPT_WRAP, "")


def read_config(path: str) -> dict:
    with safe_open(path, framework="pt") as f:
        meta = dict(f.metadata() or {})
    if "ptd_num_hops" not in meta:
        raise SystemExit(f"{path}: no 'ptd_num_hops' in metadata -- not a PTD checkpoint?")
    hops = int(meta["ptd_num_hops"])
    num_steps = int(meta.get("pdd_num_steps", 32))
    if num_steps % hops:
        raise SystemExit(f"pdd_num_steps={num_steps} not divisible by ptd_num_hops={hops}")
    return {
        "hops": hops,
        "num_steps": num_steps,
        # derived, never read from the file: the reference warns the stored
        # pdd_block_size can be stale (apply_ptd_lora recomputes it too)
        "block_size": num_steps // hops,
        "rank": int(meta.get("lora_rank", 128)),
        "alpha": float(meta.get("lora_alpha", 128.0)),
        "targets": meta.get("lora_targets", ""),
        "raw": meta,
    }


def slice_file(path: str, hop, cfg: dict) -> dict:
    """hop: int index, or "mean". Returns the de-hopped state dict."""
    out, n_adapters, n_heads = {}, 0, 0
    with safe_open(path, framework="pt") as f:
        for raw_key in f.keys():
            key = _clean(raw_key)
            if key.endswith(("curv_weight", "curv_bias")):
                continue  # training-only curvature heads; inference never reads them
            if key in HEAD_KEYS:
                out[key] = f.get_tensor(raw_key)
                n_heads += 1
                continue
            if not key.endswith(_ADAPTER_TAILS):
                raise SystemExit(f"{path}: unexpected key {key} -- format drift, refusing.")
            if hop == "mean":
                t = f.get_tensor(raw_key)
                if t.shape[0] != cfg["hops"]:
                    raise SystemExit(f"{key}: hop axis {t.shape[0]} != {cfg['hops']}")
                # mean in fp32 then back to bf16: averaging 8 bf16 tensors in
                # bf16 loses ~3 bits of the tail
                out[key] = t.to(torch.float32).mean(0).to(t.dtype).contiguous()
            else:
                # get_slice avoids materialising all H hops per key
                out[key] = f.get_slice(raw_key)[hop].contiguous()
            n_adapters += 1
    if n_heads != len(HEAD_KEYS):
        raise SystemExit(f"{path}: found {n_heads}/{len(HEAD_KEYS)} head-bank keys")
    print(f"  sliced {n_adapters} adapters + {n_heads} head tensors")
    return out


def out_metadata(cfg: dict, hop, src: str) -> dict:
    return {
        "pdd_num_steps": str(cfg["num_steps"]),
        "pdd_block_size": str(cfg["block_size"]),
        "lora_rank": str(cfg["rank"]),
        "lora_alpha": str(cfg["alpha"]),
        "lora_targets": cfg["targets"],
        "pdd_partition": "ref2va",
        "source_file": os.path.basename(src),
        # provenance: this is a degenerate single-hop slice, not real PTD
        "ptd_hop": str(hop),
        "ptd_num_hops": str(cfg["hops"]),
        "ptd_slice_note": (
            "Degenerate PTD slice: one hop's adapter applied at EVERY step. "
            "Not equivalent to PTD inference, which selects hop == block index."),
    }


def verify(out_path: str, reference: str | None) -> None:
    """Round-trip the written file through the pack's own loader."""
    try:
        import comfy.utils  # noqa: E402
        from minimax_h3_pdd_acc.pdd_acc_core import split_pdd_state_dict  # noqa: E402
    except ImportError as exc:
        print(f"  (skipping loader round-trip: needs ComfyUI-MiniMax-H3-PDD-Acc on the path "
              f"-- {exc})")
        return

    sd, meta = comfy.utils.load_torch_file(out_path, safe_load=True, return_metadata=True)
    lora_sd, heads, config = split_pdd_state_dict(sd, meta, os.path.basename(out_path))
    mods = sorted(k[: -len(".lora_A.weight")] for k in lora_sd if k.endswith(".lora_A.weight"))
    print(f"  loader OK: {config['source_format']} format, {len(mods)} modules, "
          f"grid {config['num_steps']}, block {config['block_size']}, alpha {config['alpha']}")
    print(f"  heads: proj_out {list(heads[0].shape)}, audio_proj_out {list(heads[2].shape)}")

    if reference and os.path.exists(reference):
        rsd, rmeta = comfy.utils.load_torch_file(reference, safe_load=True, return_metadata=True)
        rlora, _, _ = split_pdd_state_dict(rsd, rmeta, os.path.basename(reference))
        rmods = sorted(k[: -len(".lora_A.weight")] for k in rlora if k.endswith(".lora_A.weight"))
        extra = sorted(set(mods) - set(rmods))
        missing = sorted(set(rmods) - set(mods))
        print(f"  vs {os.path.basename(reference)}: {len(rmods)} modules "
              f"({len(extra)} extra, {len(missing)} missing)")
        if extra:
            raise SystemExit(f"  FAIL: modules absent from the reference PDD file: {extra[:5]}")
        # PTD carries no adaln LoRA, so the adaln modules are the only legal gap
        non_adaln = [m for m in missing if "adaln" not in m]
        if non_adaln:
            raise SystemExit(f"  FAIL: non-adaln modules missing vs reference: {non_adaln[:5]}")
        print(f"  key sets match (the {len(missing)} missing are all adaln -- "
              f"expected, PTD does not adapt adaln_proj)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ptd_file")
    ap.add_argument("--hop", action="append", default=None,
                    help="hop index, 'mean', or 'all' (repeatable; default: mean)")
    ap.add_argument("--out-dir", default="/workspace/ComfyUI/models/pdd_acc")
    ap.add_argument("--reference", default=None,
                    help="a PDD Acc file to diff converted key sets against "
                         "(default: the ref2va 8Step in --out-dir, if present)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = read_config(args.ptd_file)
    print(f"{os.path.basename(args.ptd_file)}: {cfg['hops']} hops, grid {cfg['num_steps']}, "
          f"block {cfg['block_size']}, rank {cfg['rank']}, alpha {cfg['alpha']}")
    print(f"  targets: {cfg['targets']}")

    requested = args.hop or ["mean"]
    hops: list = []
    for h in requested:
        if h == "all":
            hops.extend(list(range(cfg["hops"])) + ["mean"])
        elif h == "mean":
            hops.append("mean")
        else:
            i = int(h)
            if not 0 <= i < cfg["hops"]:
                raise SystemExit(f"--hop {i} out of range (0..{cfg['hops'] - 1})")
            hops.append(i)
    seen = set()
    hops = [h for h in hops if not (h in seen or seen.add(h))]

    reference = args.reference
    if reference is None:
        cand = os.path.join(args.out_dir, "MiniMax-H3-Ref2VA-Acc-8Step.safetensors")
        reference = cand if os.path.exists(cand) else None

    os.makedirs(args.out_dir, exist_ok=True)
    for hop in hops:
        tag = "mean" if hop == "mean" else f"{hop:02d}"
        name = f"minimax_h3_ref2va_ptd_{cfg['hops']}step_hop{tag}.safetensors"
        dst = os.path.join(args.out_dir, name)
        print(f"\n-> {name}")
        if args.dry_run:
            print("  (dry run)")
            continue
        sd = slice_file(args.ptd_file, hop, cfg)
        save_file(sd, dst, metadata=out_metadata(cfg, hop, args.ptd_file))
        del sd
        print(f"  wrote {os.path.getsize(dst) / 2**30:.2f} GiB")
        verify(dst, reference)


if __name__ == "__main__":
    main()
