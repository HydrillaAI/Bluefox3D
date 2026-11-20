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

| | Stock [`Hydrilla/BlueFox3D`](https://huggingface.co/Hydrilla/BlueFox3D) | This `shape-512` finetune | [`Hydrilla/BlueFox3D-retrained-characters`](https://huggingface.co/Hydrilla/BlueFox3D-retrained-characters) |
|---|---|---|---|
| Weights | stock cascade | this repo for shape-512 only | 4-stage L4 pack |
| Vertices | 210,075 | 198,274 | 197,039 |
| Faces | 246,515 | 235,199 | 236,397 |
| Runtime | 436.4 s | 437.1 s | 396 s |
| Status | **Recommended default** | Experimental | Experimental |

The two GLBs are **not identical**, but they look similar. This is **not** a quality-win claim. `21_img` is a demo photo, not one of the 20 hold-out meshes.

### 15-Image Open-Weights Context (Single L4 24 GB, Seed 42)

For broader context across 15 test images on a single L4: **BlueFox3D (Stock)** averaged 4.2 min / 199,836 verts / 214,576 faces / 11.3 MB (2048 PBR), vs **BlueFox3D (Char Pack)** (4.4 min / 203,623 verts / 214,524 faces / 11.7 MB), **TRELLIS.2-4B** (9.4 min / 219,553 verts / 239,819 faces / 19.7 MB), **Hunyuan3D-2.1** (11.8 min / 31,939 verts / 40,000 faces / 6.2 MB), and **Hi3DGen** (1.5 min / 426,673 verts / 853,476 faces / 19.5 MB clay mesh). See [`Hydrilla/BlueFox3D`](https://huggingface.co/Hydrilla/BlueFox3D) and [`HydrillaAI/Bluefox3D`](https://github.com/HydrillaAI/Bluefox3D) for full details.

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
