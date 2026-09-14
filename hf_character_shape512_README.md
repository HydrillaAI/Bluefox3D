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

# BlueFox3D character shape-512 (L4 finetune)

**This is not a replacement for [`Hydrilla/BlueFox3D`](https://huggingface.co/Hydrilla/BlueFox3D).**  
Default BlueFox3D inference still uses that stock repo.

This repo contains a **shape-512 denoiser** finetuned on a small character-like Objaverse-LVIS subset. All other cascade stages (sparse structure, shape-1024, texture-1024) are still loaded from stock `Hydrilla/BlueFox3D`.

## What was trained

| Item | Value |
|------|--------|
| Stage | shape-512 only (`ElasticSLatFlowModel`, 30 blocks, 1.39B params) |
| Trainable | last **4** transformer blocks + `out_layer` (~183M / 1.39B) |
| Steps | 1000 |
| Batch | 1 mesh / step |
| GPU | 1× NVIDIA L4 24 GB |
| Optimizer | AdamW 8-bit, lr `1e-5`, no EMA |
| Start weights | public Pixal3D / BlueFox3D shape-512 |

1536 inference still runs:

1. SS-64 — **stock**
2. Shape-512 — **this file**
3. Shape-1024 — **stock**
4. Texture-1024 — **stock**

Expect a **small mid-stage geometry shift**, not a new product model.

## Dataset

**Objaverse-LVIS only.** We did **not** download the ~8.9 TB `allenai/objaverse` snapshot.

LVIS keys (no class named `character`): person, teddy bear, doll, figurine, statue, mannequin, puppet, action figure, toy soldier, snowman, scarecrow. Matching was a bit loose, so extra toy/bear IDs mixed in.

| Split | Count | Trained on? |
|-------|-------|-------------|
| Train GLBs | 150 downloaded, **149 encoded** (1 mesh failed voxelize) | Yes |
| Hold-out GLBs | 20 | **Never** |
| Encoded | 2 Blender views + dual-grid 512 + shape-512 view latents | Yes |
| PBR / SS latents | not built | No |

## One-image compare (honest)

Same photo `assets/images/21_img.png` (jester), seed `42`, `--preset baseline`, `--resolution 1536`, `--low_vram`.

| | Stock Pixal3D / BlueFox3D | This shape-512 |
|---|---|---|
| Weights | `Hydrilla/BlueFox3D` | this repo for shape-512 only |
| Vertices | 210,075 | 198,274 |
| Faces | 246,515 | 235,199 |

The two GLBs are **not identical**, but they look similar. This is **not** a quality-win claim. `21_img` is a demo photo, not one of the 20 hold-out meshes.

## Use

```bash
python inference.py --image assets/images/21_img.png --output ./ft.glb \
  --seed 42 --low_vram --resolution 1536 --preset baseline \
  --model_path Hydrilla/BlueFox3D-character-shape512
```

`--low_vram` is required on 24 GB GPUs.

Stock (unchanged):

```bash
python inference.py --image assets/images/21_img.png --output ./stock.glb \
  --seed 42 --low_vram --resolution 1536 --preset baseline \
  --model_path Hydrilla/BlueFox3D
```

## Files

- `pipeline.json` — shape-512 from this repo; other stages from `Hydrilla/BlueFox3D`
- `ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors` — converted `denoiser_step0001000.pt`
- `ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.json` — same architecture as stock shape-512

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
