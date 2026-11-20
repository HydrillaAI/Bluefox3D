#!/usr/bin/env python3
"""Build a local 1536 cascade pack: new SS-64 / shape-1024 / tex-1024 + copied shape-512.

Does not upload. Does not touch Hydrilla/BlueFox3D.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file

ROOT = Path(os.environ.get("ROOT", "/home/dharani/work/BlueFox3D"))
OUT = Path(os.environ.get("PACK_DIR", str(ROOT / "results" / "retrained_characters_pack")))
STOCK = "Hydrilla/BlueFox3D"
SHAPE512_REPO = "Hydrilla/BlueFox3D-character-shape512"

STAGES = [
    {
        "name": "ss64",
        "pt": ROOT / "results/ss64_character/ckpts/denoiser_step0001000.pt",
        "stem": "ss_flow_img_dit_1_3B_64_bf16",
        "json_repo": STOCK,
    },
    {
        "name": "shape1024",
        "pt": ROOT / "results/shape1024_character/ckpts/denoiser_step0001000.pt",
        "stem": "slat_flow_img2shape_dit_1_3B_1024_bf16",
        "json_repo": STOCK,
    },
    {
        "name": "tex1024",
        "pt": ROOT / "results/tex1024_character/ckpts/denoiser_step0001000.pt",
        "stem": "slat_flow_imgshape2tex_dit_1_3B_1024_bf16",
        "json_repo": STOCK,
    },
]


def _to_cpu_sd(obj) -> dict:
    if isinstance(obj, dict) and all(torch.is_tensor(v) for v in obj.values()):
        sd = obj
    elif isinstance(obj, dict) and "state_dict" in obj:
        sd = obj["state_dict"]
    elif isinstance(obj, dict) and "denoiser" in obj and isinstance(obj["denoiser"], dict):
        sd = obj["denoiser"]
    else:
        sd = obj
    out = {}
    for k, v in sd.items():
        key = k[7:] if k.startswith("module.") else k
        if not torch.is_tensor(v):
            continue
        t = v.detach().to("cpu").contiguous()
        out[key] = t
    return out


def convert_pt(pt: Path, out_sf: Path) -> int:
    print("convert", pt, flush=True)
    sd = torch.load(pt, map_location="cpu", weights_only=True)
    sd = _to_cpu_sd(sd)
    out_sf.parent.mkdir(parents=True, exist_ok=True)
    save_file(sd, str(out_sf))
    print("wrote", out_sf, "keys", len(sd), "bytes", out_sf.stat().st_size, flush=True)
    return len(sd)


def copy_json(repo: str, stem: str, dest: Path) -> None:
    src = hf_hub_download(repo, f"ckpts/{stem}.json")
    shutil.copy2(src, dest)
    print("json", dest, flush=True)


def copy_shape512(ckpt_dir: Path) -> None:
    stem = "slat_flow_img2shape_dit_1_3B_512_bf16"
    for ext in (".safetensors", ".json"):
        src = hf_hub_download(SHAPE512_REPO, f"ckpts/{stem}{ext}")
        dst = ckpt_dir / f"{stem}{ext}"
        shutil.copy2(src, dst)
        print("copied shape-512", dst, dst.stat().st_size, flush=True)


def write_pipeline(out: Path) -> None:
    src = hf_hub_download(SHAPE512_REPO, "pipeline.json")
    with open(src) as f:
        pipe = json.load(f)
    models = pipe["args"]["models"]
    models["sparse_structure_flow_model"] = "ckpts/ss_flow_img_dit_1_3B_64_bf16"
    models["shape_slat_flow_model_512"] = "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16"
    models["shape_slat_flow_model_1024"] = "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16"
    models["tex_slat_flow_model_1024"] = "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16"
    # decoders stay Hydrilla/BlueFox3D
    dest = out / "pipeline.json"
    dest.write_text(json.dumps(pipe, indent=2) + "\n")
    print("pipeline", dest, flush=True)


def main() -> None:
    ckpt_dir = OUT / "ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    missing = [s["name"] for s in STAGES if not s["pt"].is_file()]
    if missing:
        raise SystemExit(f"missing checkpoints: {missing}")
    for stage in STAGES:
        stem = stage["stem"]
        convert_pt(stage["pt"], ckpt_dir / f"{stem}.safetensors")
        copy_json(stage["json_repo"], stem, ckpt_dir / f"{stem}.json")
    copy_shape512(ckpt_dir)
    write_pipeline(OUT)
    print("PACK_OK", OUT, flush=True)


if __name__ == "__main__":
    main()
