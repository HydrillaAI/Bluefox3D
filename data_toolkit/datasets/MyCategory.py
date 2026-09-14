"""Local GLB/OBJ folder for a single-category BlueFox3D finetune.

Official toolkit subsets are ObjaverseXL / ABO / TexVerse only. This adapter
lets dump/render/encode scripts see files already in datasets/MyCategory/raw/.
"""

from __future__ import annotations

import argparse
import os

import pandas as pd
from utils import get_file_hash


def add_args(parser: argparse.ArgumentParser):
    pass


def get_metadata(**kwargs):
    root = kwargs.get("root") or kwargs.get("download_root")
    if not root:
        raise ValueError("MyCategory.get_metadata needs --root")
    raw = os.path.join(root, "raw")
    if not os.path.isdir(raw):
        raise FileNotFoundError(f"No GLBs yet: {raw}")
    exts = (".glb", ".obj", ".fbx", ".gltf", ".usdz")
    records = []
    for dirpath, _, filenames in os.walk(raw):
        for fname in filenames:
            if not fname.lower().endswith(exts):
                continue
            full = os.path.join(dirpath, fname)
            sha256 = get_file_hash(full)
            rel = os.path.relpath(full, root)
            records.append({
                "sha256": sha256,
                "local_path": rel,
                "file_identifier": os.path.splitext(fname)[0],
                "aesthetic_score": 5.0,
            })
    if not records:
        raise FileNotFoundError(f"No meshes under {raw}")
    return pd.DataFrame.from_records(records)


def download(metadata, output_dir, **kwargs):
    """Meshes are collected by collect_objaverse_lvis.py; nothing to fetch here."""
    rows = []
    for rec in metadata.to_dict("records"):
        if rec.get("local_path"):
            rows.append({"sha256": rec["sha256"], "local_path": rec["local_path"]})
    return pd.DataFrame.from_records(rows)


def _process_instance(args):
    metadatum, output_dir, func = args
    try:
        local_path = metadatum["local_path"]
        sha256 = metadatum["sha256"]
        file = os.path.join(output_dir, local_path)
        return func(file, sha256)
    except Exception as e:
        print(f"Error processing object {metadatum.get('sha256', '?')}: {e}")
        return None


def foreach_instance(metadata, output_dir, func, max_workers=None, desc="Processing objects", timeout=None, **_kwargs) -> pd.DataFrame:
    from concurrent.futures import ProcessPoolExecutor, TimeoutError, as_completed

    from tqdm import tqdm

    metadata = metadata.to_dict("records")
    max_workers = max_workers or min(8, os.cpu_count() or 4)
    if max_workers < 1:
        max_workers = 1
    records = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_instance, (m, output_dir, func)): m["sha256"]
            for m in metadata
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            sha256 = futures[future]
            try:
                r = future.result(timeout=timeout)
                if r is not None:
                    records.append(r)
            except TimeoutError:
                print(f"Timeout processing object {sha256}")
            except Exception as e:
                print(f"Error processing object {sha256}: {e}")
    return pd.DataFrame.from_records(records)
