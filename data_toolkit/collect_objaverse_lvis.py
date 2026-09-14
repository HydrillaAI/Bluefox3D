#!/usr/bin/env python3
"""Download a small Objaverse-LVIS subset. Never snapshot the 8.9TB dataset."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path

CHARACTER_KEYS = (
    "person",
    "teddy bear",
    "doll",
    "figurine",
    "statue",
    "mannequin",
    "puppet",
    "action figure",
    "toy soldier",
    "snowman",
    "scarecrow",
)


def match_lvis_keys(lvis: dict) -> dict[str, list[str]]:
    keys_lower = {k.lower(): k for k in lvis}
    picked = {}
    for want in CHARACTER_KEYS:
        if want in keys_lower:
            real = keys_lower[want]
            picked[real] = list(lvis[real])
            continue
        hits = [k for k in lvis if want in k.lower() or k.lower() in want]
        for k in hits:
            picked[k] = list(lvis[k])
    return picked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("datasets/MyCategory"))
    parser.add_argument("--train", type=int, default=150)
    parser.add_argument("--holdout", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download-processes", type=int, default=8)
    args = parser.parse_args()

    import objaverse

    lvis = objaverse.load_lvis_annotations()
    groups = match_lvis_keys(lvis)
    if not groups:
        # last resort: any key with person/human/character in the name
        groups = {k: v for k, v in lvis.items() if any(w in k.lower() for w in ("person", "human", "character", "teddy"))}
    if not groups:
        raise SystemExit(f"No character-like LVIS keys. Sample keys: {sorted(lvis)[:40]}")

    rng = random.Random(args.seed)
    uids = []
    seen = set()
    for key, ids in sorted(groups.items()):
        print(f"LVIS {key}: {len(ids)} uids")
        for uid in ids:
            if uid not in seen:
                seen.add(uid)
                uids.append(uid)
    rng.shuffle(uids)
    need = args.train + args.holdout
    if len(uids) < need:
        print(f"Only {len(uids)} character UIDs; using all of them")
        need = len(uids)
        args.holdout = min(args.holdout, max(len(uids) // 8, 1))
        args.train = need - args.holdout
    selected = uids[:need]
    train = selected[: args.train]
    holdout = selected[args.train :]

    print(f"Downloading {len(selected)} GLBs (train={len(train)} holdout={len(holdout)})")
    paths = objaverse.load_objects(uids=selected, download_processes=args.download_processes)

    raw = args.root / "raw"
    raw_holdout = args.root / "raw_holdout"
    raw.mkdir(parents=True, exist_ok=True)
    raw_holdout.mkdir(parents=True, exist_ok=True)
    splits = args.root / "splits"
    splits.mkdir(parents=True, exist_ok=True)

    def copy_uid(uid: str, dest_dir: Path) -> bool:
        src = paths.get(uid)
        if not src:
            print(f"missing download for {uid}")
            return False
        src_path = Path(src)
        dest = dest_dir / f"{uid}{src_path.suffix.lower() or '.glb'}"
        if not dest.exists():
            shutil.copy2(src_path, dest)
        return dest.exists()

    train_ok = [uid for uid in train if copy_uid(uid, raw)]
    hold_ok = [uid for uid in holdout if copy_uid(uid, raw_holdout)]
    (splits / "train_uids.txt").write_text("\n".join(train_ok) + "\n")
    (splits / "holdout_uids.txt").write_text("\n".join(hold_ok) + "\n")
    (args.root / "lvis_groups.json").write_text(json.dumps({k: len(v) for k, v in groups.items()}, indent=2))
    print(f"Train GLBs in {raw}: {len(train_ok)}")
    print(f"Hold-out GLBs in {raw_holdout}: {len(hold_ok)} (never scanned by MyCategory metadata)")
    if len(train_ok) < 8:
        raise SystemExit(f"Only {len(train_ok)} train GLBs; need at least 8 for smoke")


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    main()
