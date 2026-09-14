# Character shape-512 finetune on 4× L4 (what actually changed)

**Date:** 14 Sep 2026  
**VM:** `hydrillapixal3d` (`us-central1-a`), 4× NVIDIA L4 24 GB  
**Goal:** Finetune public Pixal3D / BlueFox3D **shape-512** on a small character-like Objaverse subset, on this box, without an 80 GB GPU.

One-image compare is done (`21_img` jester, seed 42). See **[COMPARE_CHARACTER_FT.md](COMPARE_CHARACTER_FT.md)** for dataset, training recipe, face counts, and honesty notes. The meshes differ; they look similar. This is **not** a quality-win claim.

---

## 1. What this new checkpoint is

| Item | Value |
|------|--------|
| Stage trained | **shape-512 denoiser only** (`ElasticSLatFlowModel`, 30 blocks, 1.39B params) |
| What was actually updated | **Last 4 transformer blocks + `out_layer` only** (~183M / 1.39B params) |
| Optimizer | **AdamW 8-bit** (`bitsandbytes`) |
| Steps | **1000** (plus a 50-step VRAM smoke) |
| Batch | 1 mesh / step, 1 GPU |
| LR | `1e-5` |
| Data | **149** train objects (150 dumped, 1 failed voxelize) |
| Hold-out | **20** GLBs in `raw_holdout/` — **never trained** |
| EMA | off |
| Snapshots during train | off (`i_sample: -1`) |

Official inference at `--resolution 1536` still runs:

1. SS-64 (stock)
2. **shape-512 (this is the only stage we changed)**
3. shape-1024 (stock)
4. tex-1024 (stock)

So even after you plug this file in, **geometry at 1024 and all textures are still the public model**. Expect a **small mid-stage** change on character-like inputs, not a new product model.

---

## 2. Where the new weights are (on the VM)

```text
/home/dharani/work/BlueFox3D/results/shape512_character/ckpts/
  denoiser_step0000250.pt
  denoiser_step0000500.pt
  denoiser_step0000750.pt
  denoiser_step0001000.pt   ← use this unless hold-out says otherwise

/home/dharani/work/BlueFox3D/results/shape512_character_smoke1gpu/ckpts/
  denoiser_step0000050.pt   ← 50-step smoke only
```

Each `denoiser_*.pt` is a **full** 5.2 GB state dict (frozen blocks kept, last 4 blocks trained). Step **1000** was converted and uploaded to [`Hydrilla/BlueFox3D-character-shape512`](https://huggingface.co/Hydrilla/BlueFox3D-character-shape512).

Default `inference.py` (no `--model_path`) still uses stock [`Hydrilla/BlueFox3D`](https://huggingface.co/Hydrilla/BlueFox3D). That repo was **not** overwritten.

---

## 3. Code / config we added or changed

### New files

| File | Why |
|------|-----|
| `data_toolkit/datasets/MyCategory.py` | Official toolkit has no “folder of GLBs” subset. This adapter hashes `datasets/MyCategory/raw/` and feeds dump/render/encode. |
| `data_toolkit/collect_objaverse_lvis.py` | Downloads **~170 LVIS UIDs only** (not the 8.9 TB Objaverse snapshot). Train → `raw/`, hold-out → `raw_holdout/`. |
| `configs/gen/shape512_mycategory_l4.json` | 1000-step L4 train config. |
| `configs/gen/shape512_mycategory_l4_smoke.json` | 50-step VRAM smoke. |
| `data_toolkit/run_character_finetune.sh` | Collect → smoke preprocess → full preprocess → train. Resume with `STAGE=train1` / `STAGE=train4`. |

LVIS keys used (no class named `character`): person, teddy bear, doll, figurine, statue, mannequin, puppet, action figure, toy soldier, snowman, scarecrow. Matching was a bit loose, so a few extra classes (e.g. toy / bear) mixed in.

### Trainer / model changes (needed so 24 GB could train)

| File | Change |
|------|--------|
| `bluefox3d/trainers/basic.py` | `trainable_last_blocks`: freeze all but last N blocks + `out_layer`. `AdamW8bit` optimizer. `i_sample: -1` skips snapshots. `prefetch_data` flag. Safe `writer.close()` when TensorBoard is off. `empty_cache` before `optimizer.step()`. |
| `bluefox3d/models/structured_latent_flow.py` | Frozen blocks run under `torch.no_grad()` so activations are not kept. |
| `bluefox3d/utils/dist_utils.py` | Each rank reads the checkpoint from disk. Broadcasting a 5.2 GB file as a CUDA `ByteTensor` OOM’d L4s. |

### Config knobs that are **not** the official A100 recipe

Official shape-512: batch 8, `batch_split` 2, lr `1e-4`, full AdamW, EMA on.  
This run: batch 1, `batch_split` 1 (batch must divide split), lr `1e-5`, `max_tokens` 2048, no EMA, 8-bit Adam, last-4-block freeze.

4× L4 is **four 24 GB cards**, not 96 GB. DDP does not merge VRAM. 4-GPU DDP **failed** with freeze + checkpointing (`out_layer.weight marked ready twice`). The 1000 steps ran on **1 L4**.

---

## 4. Data that was built (toolkit, not hand-made)

On the VM under `datasets/MyCategory/`:

```text
raw/                  150 train GLBs
raw_holdout/          20 hold-out GLBs (not in metadata / not trained)
metadata.csv
mesh_dumps/
renders_cond/         2 Blender views + transforms.json
dual_grid_view_512/   149 ok (1 mesh had no valid geometry)
shape_latents/shape_enc_next_dc_f16c32_fp16_512_view/
splits/train_sha256.txt  smoke_sha256.txt  holdout_uids.txt
```

We did **not** encode PBR or SS latents. Those are for texture / occupancy stages we did not train.

---

## 5. Compare that was actually run

Full write-up: **[COMPARE_CHARACTER_FT.md](COMPARE_CHARACTER_FT.md)**.

One image (`21_img` jester), seed `42`, `--preset baseline`, `--resolution 1536`. Stock vs converted step-1000 shape-512. Vertices 210,075 vs 198,274. Files are not identical; they look similar. `21_img` is a demo photo, **not** the 20 hold-out meshes.

Stills and `meta.json` (no GLBs in git): [`compare/ft_one_image/21_img/`](compare/ft_one_image/21_img/).

Load the finetune without a local convert:

```bash
python inference.py --image assets/images/21_img.png --output ./ft.glb \
  --seed 42 --low_vram --resolution 1536 --preset baseline \
  --model_path Hydrilla/BlueFox3D-character-shape512
```

---

## 6. What we did **not** change

- Sparse-structure (SS-64) weights  
- Shape-1024 and texture-1024 weights  
- Default Hugging Face repo `Hydrilla/BlueFox3D` (stock was not overwritten; finetune is a **new** repo)  
- Mesh-cleanup / presets wrapper (that is not this neural finetune)  
- Full-model (all 30 blocks) finetune  
- 4-GPU DDP training (incompatible with this freeze)  
- Hold-out Chamfer / visual eval (not run yet)

Do **not** ship or upload this checkpoint as “BlueFox3D better than Pixal3D” until hold-out character images/meshes beat stock at the same settings, and the 4 old mixed images do not collapse.

---

## 7. Disk / billing notes

Checkpoints for 250/500/750/1000 are ~**23 GB**. You can delete 250/500/750 after you pick a step. The VM still bills while `g2-standard-48` + 4×L4 is running.

To stop the instance when you are done:

```bash
gcloud compute instances stop hydrillapixal3d --zone us-central1-a --project hydrilla-486606
```
