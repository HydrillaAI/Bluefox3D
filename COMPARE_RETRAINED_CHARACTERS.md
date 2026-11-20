# Pixal3D vs BlueFox3D retrained-characters pack

**Date:** 17 Sep 2026  
**One-image compare:** `assets/images/21_img.png` (jester)  
**This is not a quality-win claim.** On this one image the pack is **more broken** than stock (extra islands, less coherent silhouette). Do not ship it as an upgrade from that still.

Default inference still uses stock [`Hydrilla/BlueFox3D`](https://huggingface.co/Hydrilla/BlueFox3D).  
The four-stage pack is a **separate** repo: [`Hydrilla/BlueFox3D-retrained-characters`](https://huggingface.co/Hydrilla/BlueFox3D-retrained-characters).  
[`Hydrilla/BlueFox3D-character-shape512`](https://huggingface.co/Hydrilla/BlueFox3D-character-shape512) was not overwritten.

---

## 1. What was compared

Same photo, same seed, same wrapper. The pack swaps three new denoisers plus the old shape-512.

| | Pixal3D / stock BlueFox3D | Retrained-characters pack |
|---|---|---|
| Image | `assets/images/21_img.png` | same |
| Seed | 42 | 42 |
| Preset | `baseline` | `baseline` |
| Resolution | 1536 cascade | 1536 cascade |
| `--low_vram` | yes | yes |
| `--decimation` | 250000 | 250000 |
| Model path | `Hydrilla/BlueFox3D` | local pack (same files as the Hub repo) |
| Vertices | 211,389 | 197,039 |
| Faces | 246,891 | 236,397 |
| GLB size | 14.0 MB | 13.2 MB |
| Wall time | 376 s | 396 s |

`21_img` is a demo photo, not one of the 20 unused hold-out meshes.

---

## 2. What was trained

| Slot | This pack | Notes |
|------|-----------|--------|
| SS-64 | last-4-block, 1000 steps | 149 SS latents from shape-1024 |
| Shape-512 | **copied**, not retrained | from `Hydrilla/BlueFox3D-character-shape512` |
| Shape-1024 | last-4-block, 1000 steps | `max_tokens` 2048 (official 32768 needs ~80 GB) |
| Tex-1024 | last-4-block, 1000 steps | 120 PBR latents (shader dump failed on the rest) |
| VAEs | stock `Hydrilla/BlueFox3D` | not trained |

Same 149 LVIS train GLBs as the earlier shape-512 run. Hold-out 20 unused. Optimizer AdamW 8-bit, 1× L4, no EMA. 50-step smoke passed on all three new stages before 1000 steps.

---

## 3. Honesty

Do not say “retrained BlueFox3D on characters.” Say: three extra last-4-block stages on a small LVIS set, plus the old shape-512. On `21_img` the pack **looks worse** than stock (fragmented hat/body, extra floating pieces). Mesh counts moved; that is not a quality metric. Hold-out 20 GLBs still unused until a real compare. Default app weights stay stock.

---

## Use

```bash
python inference.py --image assets/images/21_img.png --output ./pack.glb \
  --seed 42 --low_vram --resolution 1536 --preset baseline \
  --model_path Hydrilla/BlueFox3D-retrained-characters
```
