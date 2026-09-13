#!/usr/bin/env python3
"""Rerun only missing 1536 GLBs, one GPU job at a time, lower decimation."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

BLUEFOX = Path("/home/dharani/work/BlueFox3D")
RUNS = Path("/home/dharani/work/compare/runs")
SEED = 42
JOBS = (
    ("4_img", "pixal3d_baseline", "baseline"),
    ("4_img", "bluefox_quality", "quality"),
    ("9_img", "pixal3d_baseline", "baseline"),
    ("9_img", "bluefox_quality", "quality"),
)


def main() -> None:
    py = sys.executable
    for stem, variant, preset in JOBS:
        run_dir = RUNS / stem / variant
        glb = run_dir / "output.glb"
        if glb.is_file() and glb.stat().st_size > 1000:
            print(f"skip existing {glb}")
            continue
        src = next((BLUEFOX / "assets" / "images").glob(f"{stem}.*"))
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            py, str(BLUEFOX / "inference.py"),
            "--image", str(src),
            "--output", str(glb),
            "--seed", str(SEED),
            "--preset", preset,
            "--resolution", "1536",
            "--decimation", "250000",
            "--texture-size", "2048",
            "--no-remesh",
            "--low_vram",
            "--save-preprocessed", str(run_dir / "preprocessed.png"),
            "--meta", str(run_dir / "meta.json"),
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"
        env["PYTHONUNBUFFERED"] = "1"
        log = run_dir / "retry.log"
        print(f"START {stem} {variant}")
        with log.open("w") as f:
            code = subprocess.call(cmd, cwd=str(BLUEFOX), env=env, stdout=f, stderr=subprocess.STDOUT)
        print(f"DONE {stem} {variant} exit={code}")
        if code != 0:
            print(log.read_text()[-1500:])


if __name__ == "__main__":
    main()
