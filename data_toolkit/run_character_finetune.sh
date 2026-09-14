#!/usr/bin/env bash
# Character-category preprocess + shape-512 finetune on 4x L4.
# Does NOT snapshot allenai/objaverse (~8.9TB). Downloads ~170 LVIS GLBs only.
set -euo pipefail

ROOT="${ROOT:-/home/dharani/work/BlueFox3D}"
DATA="${DATA:-$ROOT/datasets/MyCategory}"
DT="$ROOT/data_toolkit"
CKPT_PT="$ROOT/ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.pt"
LOG_DIR="${LOG_DIR:-$ROOT/results/logs}"
STAGE="${STAGE:-all}"   # collect|smoke|full|train1|train4|all

cd "$ROOT"
# shellcheck disable=SC1090
source ~/activate_bluefox3d.sh
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_XET=1
export PYTHONPATH="$ROOT/data_toolkit:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$LOG_DIR" "$ROOT/ckpts" "$DATA/splits"

echo "===== $(date -Is) host=$(hostname) stage=$STAGE ====="
nvidia-smi -L || true
df -h "$ROOT" | tail -n 1

echo "Installing toolkit extras if needed..."
pip install -q pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless pandas objaverse bitsandbytes || true

download_ckpt() {
  if [ -f "$CKPT_PT" ]; then
    echo "Checkpoint already at $CKPT_PT"
    return 0
  fi
  echo "Downloading public shape-512 weights (safetensors -> torch.load .pt)..."
  python - <<'PY'
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import torch
st = hf_hub_download("TencentARC/Pixal3D", "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors")
sd = load_file(st)
out = "/home/dharani/work/BlueFox3D/ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.pt"
torch.save(sd, out)
print("wrote", out, "keys", len(sd))
PY
}

collect_data() {
  mkdir -p "$DATA/raw"
  n_raw="$(find "$DATA/raw" -type f \( -iname '*.glb' -o -iname '*.obj' -o -iname '*.gltf' \) | wc -l | tr -d ' ')"
  if [ "${n_raw:-0}" -ge 8 ]; then
    echo "Found $n_raw train meshes in $DATA/raw; skip collect"
  else
    python "$DT/collect_objaverse_lvis.py" --root "$DATA" --train 150 --holdout 20
  fi
}

rebuild_splits() {
  python - <<PY
import pandas as pd
from pathlib import Path
root = Path("$DATA")
meta = pd.read_csv(root / "metadata.csv")
sha = [str(s) for s in meta["sha256"].tolist()]
# Hold-out lives in raw_holdout/ and is not in metadata. Keep skip_list anyway.
hold = []
hp = root / "splits" / "holdout_uids.txt"
if hp.exists() and "file_identifier" in meta.columns:
    hold_uids = {x.strip() for x in hp.read_text().splitlines() if x.strip()}
    hold = meta[meta["file_identifier"].isin(hold_uids)]["sha256"].astype(str).tolist()
hold_set = set(hold)
train_sha = [s for s in sha if s not in hold_set]
smoke = train_sha[:8]
(root / "splits").mkdir(exist_ok=True)
(root / "splits" / "train_sha256.txt").write_text("\n".join(train_sha) + "\n")
(root / "splits" / "smoke_sha256.txt").write_text("\n".join(smoke) + "\n")
(root / "splits" / "skip_list.txt").write_text("\n".join(f"MyCategory/{s}" for s in hold) + "\n")
print(f"metadata={len(sha)} train={len(train_sha)} holdout_in_meta={len(hold)} smoke={len(smoke)}")
if len(smoke) < 5:
    raise SystemExit("Need at least 5 train hashes for smoke")
PY
}

run_toolkit() {
  local inst="$1"
  echo "dump_mesh $inst"
  python "$DT/dump_mesh.py" MyCategory --root "$DATA" --instances "$inst" --max_workers 4
  python "$DT/build_metadata.py" MyCategory --root "$DATA"
  echo "render_cond $inst"
  python "$DT/render_cond.py" MyCategory --root "$DATA" --num_cond_views 2 --instances "$inst" --max_workers 4
  python "$DT/build_metadata.py" MyCategory --root "$DATA"
  echo "dual_grid_view 512 $inst"
  python "$DT/dual_grid_view.py" MyCategory --root "$DATA" --resolution 512 --view_indices 0-1 --instances "$inst" --max_workers 4
  python "$DT/build_metadata.py" MyCategory --root "$DATA"
}

encode_one_gpu() {
  local inst="$1"
  python "$DT/encode_shape_latent_view.py" --root "$DATA" --resolution 512 --view_indices 0-1 --instances "$inst"
  python "$DT/build_metadata.py" MyCategory --root "$DATA"
}

encode_four_gpu() {
  local inst="$1"
  local pids=()
  local fail=0
  local r
  for r in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$r python "$DT/encode_shape_latent_view.py" \
      --root "$DATA" --resolution 512 --view_indices 0-1 --instances "$inst" \
      --rank "$r" --world_size 4 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
  done
  python "$DT/build_metadata.py" MyCategory --root "$DATA"
  if [ "$fail" -ne 0 ]; then
    echo "One or more encode ranks failed" >&2
    return 1
  fi
}

DATA_DIR_JSON="{\"MyCategory\":{\"base\":\"$DATA\",\"shape_latent\":\"$DATA/shape_latents/shape_enc_next_dc_f16c32_fp16_512_view\",\"render_cond\":\"$DATA/renders_cond\"}}"

train_one_gpu() {
  echo "===== 1-GPU smoke train (50 steps) ====="
  python train.py \
    --config configs/gen/shape512_mycategory_l4_smoke.json \
    --output_dir results/shape512_character_smoke1gpu \
    --num_gpus 1 \
    --auto_retry 0 \
    --data_dir "$DATA_DIR_JSON"
}

train_four_gpu() {
  echo "===== 1-GPU shape-512 finetune (1000 steps; 4-GPU DDP incompatible with last-block freeze) ====="
  python train.py \
    --config configs/gen/shape512_mycategory_l4.json \
    --output_dir results/shape512_character \
    --num_gpus 1 \
    --auto_retry 0 \
    --data_dir "$DATA_DIR_JSON"
}

case "$STAGE" in
  collect)
    download_ckpt &
    collect_data
    wait
    python "$DT/build_metadata.py" MyCategory --root "$DATA"
    rebuild_splits
    ;;
  smoke)
    python "$DT/build_metadata.py" MyCategory --root "$DATA"
    rebuild_splits
    run_toolkit "$DATA/splits/smoke_sha256.txt"
    encode_one_gpu "$DATA/splits/smoke_sha256.txt"
    echo "SMOKE_OK"
    ;;
  full)
    python "$DT/build_metadata.py" MyCategory --root "$DATA"
    rebuild_splits
    run_toolkit "$DATA/splits/train_sha256.txt"
    encode_four_gpu "$DATA/splits/train_sha256.txt"
    echo "FULL_ENCODE_OK"
    ;;
  train1)
    train_one_gpu
    ;;
  train4)
    train_four_gpu
    ;;
  resume)
    python "$DT/build_metadata.py" MyCategory --root "$DATA"
    rebuild_splits
    run_toolkit "$DATA/splits/train_sha256.txt"
    encode_four_gpu "$DATA/splits/train_sha256.txt"
    echo "FULL_ENCODE_OK"
    train_one_gpu
    echo "TRAIN1_OK"
    train_four_gpu
    echo "TRAIN4_OK"
    ;;
  all)
    download_ckpt &
    collect_data
    wait
    python "$DT/build_metadata.py" MyCategory --root "$DATA"
    rebuild_splits
    run_toolkit "$DATA/splits/smoke_sha256.txt"
    encode_one_gpu "$DATA/splits/smoke_sha256.txt"
    echo "SMOKE_OK"
    run_toolkit "$DATA/splits/train_sha256.txt"
    encode_four_gpu "$DATA/splits/train_sha256.txt"
    echo "FULL_ENCODE_OK"
    train_one_gpu
    echo "TRAIN1_OK"
    train_four_gpu
    echo "TRAIN4_OK"
    ;;
  *)
    echo "Unknown STAGE=$STAGE" >&2
    exit 2
    ;;
esac

echo "===== $(date -Is) done stage=$STAGE ====="
