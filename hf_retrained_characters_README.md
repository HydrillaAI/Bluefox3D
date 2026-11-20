---
license: mit
library_name: bluefox3d
pipeline_tag: image-to-3d
base_model: Hydrilla/BlueFox3D
tags:
  - image-to-3d
  - pixal3d
  - bluefox3d
  - finetune
  - character
---

# BlueFox3D retrained characters (4-stage L4 pack)

**This is not a replacement for [`Hydrilla/BlueFox3D`](https://huggingface.co/Hydrilla/BlueFox3D).**  
Default BlueFox3D inference still uses that stock repo.

This pack is four 1536-cascade denoisers for a small character-like Objaverse-LVIS subset:

| Slot | Source | Retrained here? |
|------|--------|-----------------|
| SS-64 | last-4-block finetune from stock SS-64 | Yes (1000 steps) |
| Shape-512 | copy of [`Hydrilla/BlueFox3D-character-shape512`](https://huggingface.co/Hydrilla/BlueFox3D-character-shape512) | **No** (not rerun) |
| Shape-1024 | last-4-block finetune from stock shape-1024 | Yes (1000 steps) |
| Tex-1024 | last-4-block finetune from stock tex-1024 | Yes (1000 steps) |
| SS / shape / tex VAEs | [`Hydrilla/BlueFox3D`](https://huggingface.co/Hydrilla/BlueFox3D) | No |

This is **not** a full retrain of BlueFox3D. Same 149 meshes, last 4 of 30 blocks + `out_layer`, 8-bit Adam, 1000 steps per new stage. Do not call it a quality win until a hold-out compare says so.

## What was trained

| Item | Value |
|------|--------|
| Trainable | last **4** transformer blocks + `out_layer` |
| Steps | 1000 per new stage (50-step VRAM smoke first) |
| Batch | 1 / GPU, `max_tokens` 2048 on 1024/tex |
| GPU | 1× NVIDIA L4 24 GB (sequential; DDP does not pool VRAM) |
| Optimizer | AdamW 8-bit, lr `1e-5`, no EMA |
| Shape-512 | reused previous character finetune |

## Dataset

**Objaverse-LVIS only.** We did **not** download the ~8.9 TB `allenai/objaverse` snapshot.

LVIS keys (no class named `character`): person, teddy bear, doll, figurine, statue, mannequin, puppet, action figure, toy soldier, snowman, scarecrow.

| Split | Count | Used? |
|-------|-------|--------|
| Train GLBs | 150 downloaded, **149** dual-grid / SS / shape | Yes |
| PBR / tex | **120** (shader dump failed on the rest) | Tex-1024 only |
| Hold-out GLBs | 20 | **Never** |

## Comparisons (15-Image L4 Benchmark & `21_img.png`)

### 15-Image Open-Weights Benchmark (Single NVIDIA L4, 24 GB VRAM, Seed 42)

| Model | Checkpoint / Source | Completed | Avg Time / Asset | Avg Vertices | Avg Faces | Avg GLB Size | Output Type |
|---|---|---:|---:|---:|---:|---:|---|
| **BlueFox3D (Stock)** | [`Hydrilla/BlueFox3D`](https://huggingface.co/Hydrilla/BlueFox3D) | 15 / 15 | 4.2 min (254 s) | 199,836 | 214,576 | 11.3 MB | 2048 PBR maps |
| **BlueFox3D (Char Pack)** | **`Hydrilla/BlueFox3D-retrained-characters`** | 15 / 15 | 4.4 min (266 s) | 203,623 | 214,524 | 11.7 MB | 2048 PBR maps |
| **TRELLIS.2-4B** | `microsoft/TRELLIS.2-4B` | 15 / 15 | 9.4 min (564 s) | 219,553 | 239,819 | 19.7 MB | 2048 PBR maps |
| **Hunyuan3D-2.1** | `tencent/Hunyuan3D-2.1` | 15 / 15 | 11.8 min (709 s) | 31,939 | 40,000 | 6.2 MB | Painted mesh (40k remesh) |
| **Hi3DGen** | `Stable-X/trellis-normal-v0-1` | 15 / 15 | 1.5 min (88 s) | 426,673 | 853,476 | 19.5 MB | Clay mesh (no PBR) |

Across the 15 images, this pack produces outputs very close to stock `Hydrilla/BlueFox3D` (small local deltas) and reproduces stock artifacts (`21_img` stray staff, `5_img` base slab). On the isolated `21_img.png` test (seed 42, 1536 `--low_vram`, 250k decimation):

- **Stock [`Hydrilla/BlueFox3D`](https://huggingface.co/Hydrilla/BlueFox3D):** 211,389 verts, 246,891 faces, 14.0 MB, 376 s (**Recommended default**)
- **[`Hydrilla/BlueFox3D-character-shape512`](https://huggingface.co/Hydrilla/BlueFox3D-character-shape512):** 198,274 verts, 235,199 faces, 437.1 s
- **This 4-stage pack:** 197,039 verts, 236,397 faces, 13.2 MB, 396 s (more fragmented on `21_img`)

## Use

```bash
python inference.py --image assets/images/21_img.png --output ./pack.glb \
  --seed 42 --low_vram --resolution 1536 --preset baseline \
  --model_path Hydrilla/BlueFox3D-retrained-characters
```

`--low_vram` is required on 24 GB GPUs (~18–22 GB). Full 1536 without it needs ~40 GB+.

Stock (unchanged):

```bash
python inference.py --image assets/images/21_img.png --output ./stock.glb \
  --seed 42 --low_vram --resolution 1536 --preset baseline \
  --model_path Hydrilla/BlueFox3D
```

## Files

- `pipeline.json` — SS-64 / shape-512 / shape-1024 / tex-1024 from this repo; decoders from `Hydrilla/BlueFox3D`
- `ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors`
- `ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors` (copied, not retrained)
- `ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors`
- `ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors`

## Code

Training and convert scripts live in [`HydrillaAI/Bluefox3D`](https://github.com/HydrillaAI/Bluefox3D).

## License / citation

MIT, same as BlueFox3D. Upstream: Pixal3D / TRELLIS.2.

```bibtex
@article{li2026pixal3d,
    title={Pixal3D: Pixel-Aligned 3D Generation from Images},
    author={Li, Dong-Yang and Zhao, Wang and Chen, Yuxin and Hu, Wenbo and Guo, Meng-Hao and Zhang, Fang-Lue and Shan, Ying and Hu, Shi-Min},
    journal={arXiv preprint arXiv:2605.10922},
    year={2026}
}
```
