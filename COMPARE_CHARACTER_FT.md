# Pixal3D vs BlueFox3D character shape-512 finetune

**Date:** 14 Sep 2026  
**One-image compare:** `assets/images/21_img.png` (jester)  
**This is not a quality-win claim.** The two meshes differ, but they look similar.

Default inference still uses stock [`Hydrilla/BlueFox3D`](https://huggingface.co/Hydrilla/BlueFox3D).  
The finetuned shape-512 is a **separate** repo: [`Hydrilla/BlueFox3D-character-shape512`](https://huggingface.co/Hydrilla/BlueFox3D-character-shape512).

---

## 1. What was compared

Same photo, same seed, same wrapper. Only shape-512 weights differed.

| | Pixal3D / stock BlueFox3D | BlueFox3D character finetune |
|---|---|---|
| Image | `assets/images/21_img.png` | same |
| Seed | 42 | 42 |
| Preset | `baseline` (no island cleanup) | `baseline` |
| Resolution | 1536 cascade | 1536 cascade |
| `--low_vram` | yes | yes |
| `--decimation` | 250000 | 250000 |
| Model path | `Hydrilla/BlueFox3D` | local convert of step 1000, now [`Hydrilla/BlueFox3D-character-shape512`](https://huggingface.co/Hydrilla/BlueFox3D-character-shape512) |
| Vertices | 210,075 | 198,274 |
| Faces | 246,515 | 235,199 |
| Runtime | 436.4 s | 437.1 s |
| MD5 | `1350886158fa0643b93088d0bfa4d3f4` | `1ca6c5e57d468902d292999cb560a85a` |

`21_img` is a **demo photo**. It is **not** one of the 20 hold-out training meshes. Hold-out was never visually compared.

### Files (local)

GitHub does not store the 13 MB GLBs (`compare/.gitignore` ignores `*.glb`). Stills and metadata are in this repo:

- Stock stills: [`compare/ft_one_image/21_img/pixal3d_stock/stills/`](compare/ft_one_image/21_img/pixal3d_stock/stills/)
- Finetune stills: [`compare/ft_one_image/21_img/bluefox_ft/stills/`](compare/ft_one_image/21_img/bluefox_ft/stills/)
- Metadata: [`pixal3d_stock/meta.json`](compare/ft_one_image/21_img/pixal3d_stock/meta.json), [`bluefox_ft/meta.json`](compare/ft_one_image/21_img/bluefox_ft/meta.json)

If you still have the GLBs on the Mac from the VM copy:

- `compare/ft_one_image/21_img_pixal3d.glb`
- `compare/ft_one_image/21_img_bluefox_ft.glb`

Open those two files side by side. PNG stills from our preview renderer are noisy and not a substitute.

### What each cascade stage used

```text
SS-64            stock Hydrilla/BlueFox3D
shape-512        stock  vs  this 1000-step finetune
shape-1024       stock Hydrilla/BlueFox3D
tex-1024         stock Hydrilla/BlueFox3D
```

Only mid-resolution **shape** changed. High-res shape and texture are still Pixal3D/BlueFox3D public weights.

---

## 2. Dataset

Source: **Objaverse-LVIS only**. We did **not** snapshot the ~8.9 TB `allenai/objaverse` dump.

Collector: [`data_toolkit/collect_objaverse_lvis.py`](data_toolkit/collect_objaverse_lvis.py)  
Folder adapter: [`data_toolkit/datasets/MyCategory.py`](data_toolkit/datasets/MyCategory.py)

LVIS keys (there is no class named `character`): person, teddy bear, doll, figurine, statue, mannequin, puppet, action figure, toy soldier, snowman, scarecrow. Matching was a bit loose, so extra toy/bear IDs mixed in.

| Split | Count | Path on the training VM | Used in training? |
|-------|-------|-------------------------|-------------------|
| Train GLBs | 150 downloaded, **149 encoded** (1 mesh failed voxelize) | `datasets/MyCategory/raw/` | Yes |
| Hold-out GLBs | 20 | `datasets/MyCategory/raw_holdout/` | **Never** |
| Encoded | 2 Blender views, dual-grid 512, shape-512 view latents | `renders_cond/`, `shape_latents/` | Yes |
| PBR / SS latents | not built | — | No (those stages were not trained) |

Seed for the LVIS sample: `42`. Train/hold-out split is in `datasets/MyCategory/splits/` on the VM (not committed).

---

## 3. How we trained it

- **Start weights:** public Pixal3D / BlueFox3D shape-512, converted to `.pt` because the trainer `finetune_from` uses `torch.load`, not Hugging Face `from_pretrained`.
- **Config:** [`configs/gen/shape512_mycategory_l4.json`](configs/gen/shape512_mycategory_l4.json)
- **Orchestrator:** [`data_toolkit/run_character_finetune.sh`](data_toolkit/run_character_finetune.sh)
- **Hardware:** 1× NVIDIA L4 24 GB. Four L4s do **not** pool VRAM. 4-GPU DDP failed with freeze + checkpointing (`out_layer.weight marked ready twice`).
- **Recipe that fitted 24 GB:** last 4 of 30 transformer blocks + `out_layer` trainable (~183M / 1.39B params), AdamW 8-bit, batch 1, `batch_split` 1, lr `1e-5`, `max_tokens` 2048, no EMA, `i_sample: -1`.
- **Steps:** 1000 (~0.19 h after a 50-step VRAM smoke).
- **Checkpoint:** `results/shape512_character/ckpts/denoiser_step0001000.pt` on the VM, converted to safetensors for inference.

Official A100 recipe this is **not:** batch 8, `batch_split` 2, lr `1e-4`, full AdamW, EMA on.

---

## 4. Code and config that changed

### New

| File | Why |
|------|-----|
| `data_toolkit/datasets/MyCategory.py` | Toolkit has no “folder of GLBs” subset. |
| `data_toolkit/collect_objaverse_lvis.py` | Download ~170 LVIS UIDs only. |
| `configs/gen/shape512_mycategory_l4.json` | 1000-step L4 train config. |
| `configs/gen/shape512_mycategory_l4_smoke.json` | 50-step VRAM smoke. |
| `data_toolkit/run_character_finetune.sh` | Collect → preprocess → train. |

### Patched so 24 GB could train

| File | Change |
|------|--------|
| `bluefox3d/trainers/basic.py` | `trainable_last_blocks`, AdamW8bit, skip snapshots, safe TensorBoard close, `empty_cache` before `optimizer.step()`. |
| `bluefox3d/models/structured_latent_flow.py` | Frozen blocks under `torch.no_grad()`. |
| `bluefox3d/utils/dist_utils.py` | Each rank reads the checkpoint from disk (no 5.2 GB CUDA broadcast). |

---

## 5. What did **not** change

- SS-64, shape-1024, texture-1024 weights
- Default Hugging Face repo [`Hydrilla/BlueFox3D`](https://huggingface.co/Hydrilla/BlueFox3D)
- Mesh-cleanup / `quality` / `fast` presets (wrapper, not this neural run)
- Full 30-block finetune
- 4-GPU DDP training
- Hold-out Chamfer / multi-image visual eval

---

## 6. Reproduce inference

Stock (unchanged default):

```bash
python inference.py --image assets/images/21_img.png --output ./stock.glb \
  --seed 42 --low_vram --resolution 1536 --preset baseline \
  --model_path Hydrilla/BlueFox3D
```

Finetuned shape-512 (other stages still stock):

```bash
python inference.py --image assets/images/21_img.png --output ./ft.glb \
  --seed 42 --low_vram --resolution 1536 --preset baseline \
  --model_path Hydrilla/BlueFox3D-character-shape512
```

`--low_vram` is required on 24 GB GPUs.

Ops notes (VM, convert, billing): [CHARACTER_FINETUNE_L4.md](CHARACTER_FINETUNE_L4.md).  
Hugging Face model card source: [hf_character_shape512_README.md](hf_character_shape512_README.md).

---

## 7. Honest conclusion

The BlueFox GLB **did** use the trained step-1000 shape-512 weights. The files are not copies of stock.

The change is a **small mid-stage geometry shift**. Last 4 of 30 blocks, 149 LVIS meshes, 1000 steps, then stock 1024 shape + stock texture. Do not describe this as a new BlueFox3D that is better than Pixal3D.
