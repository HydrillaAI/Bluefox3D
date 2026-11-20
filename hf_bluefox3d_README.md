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

- **24 GB GPUs (NVIDIA L4 / RTX 3090 / RTX 4090):** Supported at `1536` with `--low_vram` (stages offloaded to CPU between steps; ~4–8 min per asset depending on GPU and preset).
- **40 GB+ GPUs (A100 / H100):** Can run without `--low_vram` for faster execution.

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
