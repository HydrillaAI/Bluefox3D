---
license: mit
license_link: LICENSE
library_name: bluefox3d
pipeline_tag: image-to-3d
tags:
  - image-to-3d
  - 3d-generation
  - pbr
  - bluefox3d
  - pixal3d
---

# BlueFox3D 1.0 (`Hydrilla/BlueFox3D`)

**BlueFox3D** generates textured 3D meshes with PBR maps (base color, metallic, roughness, alpha) from a single image or posed multi-view images, lifting pixel features into 3D through back-projection.

- **Code & training scripts:** [`HydrillaAI/Bluefox3D`](https://github.com/HydrillaAI/Bluefox3D)
- **Model card (Hydrilla):** [`hydrilla.ai/research#model-card`](https://hydrilla.ai/research#model-card)
- **Weights provenance:** Baseline Pixal3D / TRELLIS.2-architecture 1.3B 3-stage cascade weights hosted under the Hydrilla organization as the default checkpoint for `BlueFox3D`.

Optional experimental character L4 finetune packs (not drop-in quality replacements):
- [`Hydrilla/BlueFox3D-character-shape512`](https://huggingface.co/Hydrilla/BlueFox3D-character-shape512) — last-4-block 1000-step shape-512 finetune on an Objaverse-LVIS character subset.
- [`Hydrilla/BlueFox3D-retrained-characters`](https://huggingface.co/Hydrilla/BlueFox3D-retrained-characters) — 4-stage L4 pack (SS-64, copied shape-512, shape-1024, tex-1024).

---

## Architecture

3-stage flow-matching cascade (`1536_cascade` default):
1. **Sparse Structure (`64`):** `ckpts/ss_flow_img_dit_1_3B_64_bf16` + `ckpts/ss_dec_conv3d_16l8_fp16`
2. **Shape SLat (`512` → `1024`):** `ckpts/slat_flow_img2shape_dit_1_3B_{512,1024}_bf16` + `ckpts/shape_dec_next_dc_f16c32_fp16`
3. **Texture SLat (`1024`):** `ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16` + `ckpts/tex_dec_next_dc_f16c32_fp16`

Image conditioning uses DINOv3 (`facebook/dinov3-vitl16-pretrain-lvd1689m`) with MoGe-2 camera estimation and BiRefNet background removal.

---

## Quickstart

Install [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) CUDA extensions (`FlexGEMM`, `o_voxel`, `flash-attn`, `nvdiffrast`) first, then clone [`HydrillaAI/Bluefox3D`](https://github.com/HydrillaAI/Bluefox3D):

```bash
git clone https://github.com/HydrillaAI/Bluefox3D.git BlueFox3D1.0
cd BlueFox3D1.0
pip install -r requirements.txt
pip install "huggingface_hub>=0.34,<1.0"
```

Single-image inference (`--low_vram` is required on 24 GB GPUs such as NVIDIA L4 / RTX 4090):

```bash
python inference.py --image assets/images/1_img.png --output ./output.glb --low_vram --resolution 1536
```

Multi-view inference (`transforms.json` in Blender/NeRF camera-to-world format):

```bash
python inference_mv.py --views_dir assets/mv_images/example --output ./output_mv.glb --low_vram
```

---

## Hardware Notes

- **24 GB GPUs (NVIDIA L4 / RTX 3090 / RTX 4090):** Supported at `1536` with `--low_vram` (stages offloaded to CPU between steps; ~4.2 min warm sequential average per asset, ~6–9 min cold CLI run).
- **40 GB+ GPUs (A100 / H100):** Can run without `--low_vram` for faster execution.

---

## Benchmarks & Comparisons

### 15-Image Open-Weights Benchmark (Single NVIDIA L4, 24 GB VRAM, Seed 42)

Measured directly from generated GLBs and run metadata across 15 diverse images (objects, characters, scenes/dioramas):

| Model | Checkpoint / Source | Completed | Avg Time / Asset | Avg Vertices | Avg Faces | Avg GLB Size | Output Type |
|---|---|---:|---:|---:|---:|---:|---|
| **BlueFox3D (Stock)** | [`Hydrilla/BlueFox3D`](https://huggingface.co/Hydrilla/BlueFox3D) | 15 / 15 | 4.2 min (254 s) | 199,836 | 214,576 | 11.3 MB | 2048 PBR maps |
| **BlueFox3D (Char Pack)** | [`Hydrilla/BlueFox3D-retrained-characters`](https://huggingface.co/Hydrilla/BlueFox3D-retrained-characters) | 15 / 15 | 4.4 min (266 s) | 203,623 | 214,524 | 11.7 MB | 2048 PBR maps |
| **TRELLIS.2-4B** | `microsoft/TRELLIS.2-4B` | 15 / 15 | 9.4 min (564 s) | 219,553 | 239,819 | 19.7 MB | 2048 PBR maps |
| **Hunyuan3D-2.1** | `tencent/Hunyuan3D-2.1` | 15 / 15 | 11.8 min (709 s) | 31,939 | 40,000 | 6.2 MB | Painted mesh (40k remesh) |
| **Hi3DGen** | `Stable-X/trellis-normal-v0-1` | 15 / 15 | 1.5 min (88 s) | 426,673 | 853,476 | 19.5 MB | Clay mesh (no PBR) |

**Observed trade-offs & methodology (no uncomputed Chamfer/CLIP/FID scores):**
- **BlueFox3D (Stock):** Recommended default. Preserves input color fidelity with 2048 PBR maps and completes all 15/15 on a 24 GB L4 at `1536` (`--low_vram`); observed artifacts include a stray staff on `21_img`, a base slab on `5_img`, and rug fragments on `s_18_img`.
- **BlueFox3D (Char Pack):** Very close to stock on most images, reproduces stock artifacts on `21_img` / `5_img`, and fragmented worse on the isolated `21_img` check (experimental only).
- **TRELLIS.2-4B:** Strong geometry and 2048 PBR maps, with larger GLBs (19.7 MB avg) and ~2.2× slower runtime (~9.4 min avg) on L4.
- **Hunyuan3D-2.1:** Lightest GLBs (40k faces, 6.2 MB avg); averaged 11.8 min in this 5-model batch (3.9 min in an earlier warm 4-model run); failed on `s_15_img` (invented table disc), lost objects to background removal on `s_18_img`, and shifted colors on `7_img` / `9_img`.
- **Hi3DGen:** Fastest runtime and sharpest clay geometry (`853,476` avg faces), with no PBR textures (requires a separate texturing step).

### BlueFox3D Variants Compared (`21_img.png`, Seed 42, 1536 `--low_vram`)

| Repo | What Changed | Vertices | Faces | GLB Size | Status |
|---|---|---:|---:|---:|---|
| [`Hydrilla/BlueFox3D`](https://huggingface.co/Hydrilla/BlueFox3D) | Stock 1.3B 3-stage cascade | 210,075–211,389 | 246,515–246,891 | 14.0 MB | **Recommended default** |
| [`Hydrilla/BlueFox3D-character-shape512`](https://huggingface.co/Hydrilla/BlueFox3D-character-shape512) | Shape-512 only (last 4 blocks, 1000 steps) | 198,274 | 235,199 | — | Experimental |
| [`Hydrilla/BlueFox3D-retrained-characters`](https://huggingface.co/Hydrilla/BlueFox3D-retrained-characters) | 4-stage L4 pack (SS-64, shape-1024, tex-1024 + shape-512) | 197,039 | 236,397 | 13.2 MB | Experimental |

---

## License & Attribution

Released under the [MIT License](LICENSE). Built on Pixal3D and [TRELLIS.2](https://github.com/microsoft/TRELLIS.2); see [NOTICE](NOTICE) for third-party attribution.

```bibtex
@article{li2026pixal3d,
    title={Pixal3D: Pixel-Aligned 3D Generation from Images},
    author={Li, Dong-Yang and Zhao, Wang and Chen, Yuxin and Hu, Wenbo and Guo, Meng-Hao and Zhang, Fang-Lue and Shan, Ying and Hu, Shi-Min},
    journal={arXiv preprint arXiv:2605.10922},
    year={2026}
}
```
