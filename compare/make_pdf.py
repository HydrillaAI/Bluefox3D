#!/usr/bin/env python3
"""Build a readable honest comparison PDF from compare/runs."""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PAGE = (1600, 1100)
MARGIN = 36
COL_W = 500
ROW_H = 320

FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
)


@lru_cache(maxsize=None)
def font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if Path(path).is_file():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    raise RuntimeError("No TTF font found; install fonts-dejavu-core")


def _advance(f: ImageFont.ImageFont, text: str) -> float:
    if hasattr(f, "getlength"):
        return float(f.getlength(text))
    box = f.getbbox(text)
    return float(box[2] - box[0])


def draw_text(draw: ImageDraw.ImageDraw, xy, text: str, size: int, fill=(20, 20, 20)) -> None:
    """Draw words with an explicit pixel gap so spaces cannot collapse."""
    f = font(size)
    x, y = float(xy[0]), float(xy[1])
    gap = max(_advance(f, "n") * 0.45, size * 0.28)
    for i, word in enumerate(text.split(" ")):
        if i:
            x += gap
        if not word:
            continue
        draw.text((x, y), word, fill=fill, font=f)
        x += _advance(f, word)


def fit(path: Path, size: tuple[int, int]) -> Image.Image:
    canvas = Image.new("RGB", size, (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    if not path or not path.is_file():
        draw_text(draw, (16, size[1] // 2 - 10), "missing (job failed or no GLB)", 20, fill=(120, 120, 120))
        return canvas
    im = Image.open(path).convert("RGB")
    im.thumbnail(size, Image.Resampling.LANCZOS)
    canvas.paste(im, ((size[0] - im.width) // 2, (size[1] - im.height) // 2))
    return canvas


def find_raw(bluefox: Path, stem: str) -> Path | None:
    folder = bluefox / "assets" / "images"
    if not folder.is_dir():
        return None
    for p in folder.iterdir():
        if p.stem == stem:
            return p
    return None


def load_meta(run_dir: Path) -> dict:
    meta = run_dir / "meta.json"
    if meta.is_file():
        return json.loads(meta.read_text())
    return {}


def _meta_note(name: str, folder: Path) -> str:
    meta = load_meta(folder)
    glb_ok = (folder / "output.glb").is_file()
    if not meta:
        return f"{name}: failed (no GLB)" if not glb_ok else f"{name}: GLB ok, no meta"
    bits = [f"{name}: {meta.get('resolution')}px", f"steps={meta.get('tex_slat_sampling_steps')}"]
    if meta.get("seconds"):
        bits.append(f"{meta['seconds']:.0f}s")
    if meta.get("decimation_target"):
        bits.append(f"decim={meta['decimation_target']}")
    if meta.get("remesh") is False:
        bits.append("no-remesh")
    if meta.get("cleanup"):
        bits.append("cleanup")
    bits.append("pass" if glb_ok else "fail")
    return ", ".join(bits)


def page_cover(manifest: dict) -> Image.Image:
    img = Image.new("RGB", PAGE, (255, 255, 255))
    d = ImageDraw.Draw(img)
    draw_text(d, (MARGIN, 56), "Pixal3D-equivalent defaults vs BlueFox3D wrapper (same weights)", 28)
    lines = [
        "Checkpoint: Hydrilla/BlueFox3D (public Pixal3D weights).",
        "This report is not a new-model claim. It only compares pipeline settings:",
        "stock defaults (baseline) vs wrapper cleanup at 1536 (quality) vs 1024 / 8 steps (fast).",
        "",
        f"Seed: {manifest.get('seed', 42)}",
        f"Images: {', '.join(manifest.get('images', []))}",
        "Instance: hydrillapixal3d, us-central1-a, --low_vram",
        "Failed 1536 remesh jobs were retried one GPU at a time with --decimation 250000.",
        "4_img 1536 recovered with --no-remesh. 9_img 1536 OOM'd in fill_holes (CuMesh).",
        "",
        "Stills are shaded basecolor views of the exported GLB, each in a fresh process.",
    ]
    y = 130
    for line in lines:
        draw_text(d, (MARGIN, y), line, 22)
        y += 40
    return img


def page_image(stem: str, runs: Path, bluefox: Path) -> Image.Image:
    img = Image.new("RGB", PAGE, (255, 255, 255))
    d = ImageDraw.Draw(img)
    draw_text(d, (MARGIN, 16), f"{stem}    seed=42    same Hydrilla/BlueFox3D weights", 24)

    raw = find_raw(bluefox, stem)
    base = runs / stem / "pixal3d_baseline"
    quality = runs / stem / "bluefox_quality"
    fast = runs / stem / "bluefox_fast"

    rows = (
        ((raw, "Raw photo"), (base / "preprocessed.png", "Baseline preprocess"), (quality / "preprocessed.png", "Wrapper preprocess")),
        ((base / "stills" / "front.png", "Baseline front"), (quality / "stills" / "front.png", "Quality front"), (fast / "stills" / "front.png", "Fast front")),
        ((base / "stills" / "three_quarter.png", "Baseline 3/4"), (quality / "stills" / "three_quarter.png", "Quality 3/4"), (fast / "stills" / "three_quarter.png", "Fast 3/4")),
    )
    y = 58
    for row in rows:
        x = MARGIN
        for path, title in row:
            draw_text(d, (x, y), title, 18)
            img.paste(fit(path, (COL_W, ROW_H - 28)), (x, y + 24))
            x += COL_W + 16
        y += ROW_H + 6

    notes = [_meta_note(name, folder) for name, folder in (("baseline", base), ("quality", quality), ("fast", fast))]
    draw_text(d, (MARGIN, PAGE[1] - 72), notes[0], 16)
    draw_text(d, (MARGIN, PAGE[1] - 46), "  |  ".join(notes[1:]), 16)
    return img


def page_summary(runs: Path) -> Image.Image:
    img = Image.new("RGB", PAGE, (255, 255, 255))
    d = ImageDraw.Draw(img)
    draw_text(d, (MARGIN, 50), "What this report can and cannot say", 32)
    lines = [
        "What the stills actually show:",
        "- 21_img: baseline front has a small disconnected island by the lantern; quality and fast do not.",
        "- That is wrapper cleanup dropping a floater, not a better network.",
        "- 1_img and 4_img: baseline / quality / fast silhouettes look the same. No forced win.",
        "- Fast (1024 / 8) is quicker (about 6-10 min vs 7-12 min) and not sharply lower detail here.",
        "- 4_img 1536 needed --decimation 250000 and --no-remesh. 1M remesh OOM'd earlier.",
        "- 9_img 1536 is missing: both retries OOM'd in fill_holes after sampling. Only fast made a crab.",
        "",
        "Not allowed (and not claimed):",
        "- BlueFox3D is a better neural model than Pixal3D.",
        "- New or retrained weights. Hydrilla/BlueFox3D is the public Pixal3D checkpoint.",
        "- FID / CLIP scores. None were computed.",
    ]
    y = 120
    for line in lines:
        draw_text(d, (MARGIN, y), line, 22)
        y += 40
    return img


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path("/home/dharani/work/compare/runs"))
    parser.add_argument("--out", type=Path, default=Path("/home/dharani/work/compare/BlueFox3D_vs_Pixal3D.pdf"))
    parser.add_argument("--bluefox", type=Path, default=Path("/home/dharani/work/BlueFox3D"))
    args = parser.parse_args()

    manifest_path = args.runs / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    stems = sorted(p.name for p in args.runs.iterdir() if p.is_dir())
    pages = [page_cover(manifest)]
    for stem in stems:
        pages.append(page_image(stem, args.runs, args.bluefox))
    pages.append(page_summary(args.runs))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pages[0].save(args.out, save_all=True, append_images=pages[1:])
    png_dir = args.out.parent / "pdf_pages"
    png_dir.mkdir(exist_ok=True)
    for i, page in enumerate(pages, 1):
        page.save(png_dir / f"page_{i:02d}.png")
    print(f"wrote {args.out} ({len(pages)} pages) and {png_dir}")


if __name__ == "__main__":
    main()
