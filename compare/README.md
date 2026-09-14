# Pixal3D vs BlueFox3D compare

## Wrapper compare (same stock weights)

Same weights (`Hydrilla/BlueFox3D`). The PDF shows **pipeline** differences: current Pixal3D-equivalent defaults vs the BlueFox3D wrapper (preprocess stills, mesh cleanup, fast/quality presets).

Run on the VM after `../sync_bluefox3d.sh`:

```bash
source ~/activate_bluefox3d.sh
cd ~/work/compare
python run_compare.py --bluefox ~/work/BlueFox3D --out ~/work/compare/runs
python make_pdf.py --runs ~/work/compare/runs --out ~/work/compare/BlueFox3D_vs_Pixal3D.pdf
```

`run_compare.py` uses one worker per GPU. On a single L4 it runs jobs one after another.

## Character shape-512 finetune compare (different shape-512 weights)

One-image neural compare (`21_img`, seed 42): [COMPARE_CHARACTER_FT.md](../COMPARE_CHARACTER_FT.md). Stills and metadata: [`ft_one_image/21_img/`](ft_one_image/21_img/). GLBs are gitignored.

Run on the VM after `../sync_bluefox3d.sh`:

```bash
source ~/activate_bluefox3d.sh
cd ~/work/compare
python run_compare.py --bluefox ~/work/BlueFox3D --out ~/work/compare/runs
python make_pdf.py --runs ~/work/compare/runs --out ~/work/compare/BlueFox3D_vs_Pixal3D.pdf
```

`run_compare.py` uses one worker per GPU. On a single L4 it runs jobs one after another.
