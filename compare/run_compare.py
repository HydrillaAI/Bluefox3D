#!/usr/bin/env python3
"""Run Pixal3D-equivalent baseline vs BlueFox3D wrapper on the same images."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_IMAGES = [
    "assets/images/1_img.png",
    "assets/images/4_img.png",
    "assets/images/9_img.png",
    "assets/images/21_img.png",
]

VARIANTS = (
    {
        "name": "pixal3d_baseline",
        "preset": "baseline",
        "resolution": 1536,
        "extra": [],
    },
    {
        "name": "bluefox_quality",
        "preset": "quality",
        "resolution": 1536,
        "extra": [],
    },
    {
        "name": "bluefox_fast",
        "preset": "fast",
        "resolution": 1024,
        "extra": [],
    },
)


def gpu_count() -> int:
    try:
        import torch
        return max(int(torch.cuda.device_count()), 1)
    except Exception:
        return 1


def image_stem(rel: str) -> str:
    return Path(rel).stem


def build_jobs(bluefox: Path, out: Path, images: list[str], seed: int, low_vram: bool) -> list[dict]:
    jobs = []
    for rel in images:
        src = bluefox / rel
        if not src.is_file():
            print(f"skip missing image: {src}", file=sys.stderr)
            continue
        stem = image_stem(rel)
        for variant in VARIANTS:
            run_dir = out / stem / variant["name"]
            run_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable, str(bluefox / "inference.py"),
                "--image", str(src),
                "--output", str(run_dir / "output.glb"),
                "--seed", str(seed),
                "--preset", variant["preset"],
                "--resolution", str(variant["resolution"]),
                "--save-preprocessed", str(run_dir / "preprocessed.png"),
                "--meta", str(run_dir / "meta.json"),
            ]
            if low_vram:
                cmd.append("--low_vram")
            cmd.extend(variant["extra"])
            jobs.append({
                "stem": stem,
                "variant": variant["name"],
                "cmd": cmd,
                "cwd": str(bluefox),
                "log": str(run_dir / "run.log"),
                "meta": str(run_dir / "meta.json"),
            })
    return jobs


def run_jobs(jobs: list[dict], n_gpu: int) -> None:
    print(f"{len(jobs)} jobs, {n_gpu} GPU(s)")
    pending = list(jobs)
    free = list(range(n_gpu))
    running = []
    while pending or running:
        while pending and free:
            job = pending.pop(0)
            gpu = free.pop(0)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            log_f = open(job["log"], "w")
            print(f"START gpu={gpu} {job['stem']} {job['variant']}")
            proc = subprocess.Popen(
                job["cmd"],
                cwd=job["cwd"],
                env=env,
                stdout=log_f,
                stderr=subprocess.STDOUT,
            )
            running.append((proc, job, log_f, gpu))
        still = []
        for proc, job, log_f, gpu in running:
            code = proc.poll()
            if code is None:
                still.append((proc, job, log_f, gpu))
                continue
            log_f.close()
            free.append(gpu)
            print(f"DONE gpu={gpu} {job['stem']} {job['variant']} exit={code}")
        running = still
        if running:
            time.sleep(5)


def write_manifest(out: Path, jobs: list[dict], extra: dict) -> None:
    payload = {
        "jobs": [{k: v for k, v in job.items() if k != "cmd"} | {"cmd": job["cmd"]} for job in jobs],
        **extra,
    }
    (out / "manifest.json").write_text(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bluefox", type=Path, default=Path("/home/dharani/work/BlueFox3D"))
    parser.add_argument("--out", type=Path, default=Path("/home/dharani/work/compare/runs"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--images", nargs="*", default=DEFAULT_IMAGES)
    parser.add_argument("--max-images", type=int, default=0, help="0 = all listed images")
    parser.add_argument("--no-low-vram", action="store_true")
    parser.add_argument("--max-parallel", type=int, default=2,
                        help="Cap concurrent GPU jobs (remesh is unstable with 4 at once)")
    args = parser.parse_args()

    n_gpu = min(gpu_count(), max(1, args.max_parallel))
    images = list(args.images)
    if args.max_images > 0:
        images = images[: args.max_images]
    elif n_gpu == 1:
        images = images[:2]
        print("1 GPU detected; using 2 hold-out images so the batch can finish.")

    args.out.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(args.bluefox, args.out, images, args.seed, low_vram=not args.no_low_vram)
    write_manifest(args.out, jobs, {
        "seed": args.seed,
        "gpu_count": n_gpu,
        "images": images,
        "model": "Hydrilla/BlueFox3D",
        "note": "Same checkpoint. pixal3d_baseline is current defaults; bluefox_* is the wrapper.",
    })
    started = time.time()
    run_jobs(jobs, n_gpu)
    print(f"all jobs finished in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
