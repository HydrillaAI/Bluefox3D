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
