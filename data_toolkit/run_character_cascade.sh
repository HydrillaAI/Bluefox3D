#!/usr/bin/env bash
# Encode missing SS/PBR/1024 latents and last-4-block finetune SS-64, shape-1024, tex-1024.
# Does NOT retrain shape-512. Same 149 MyCategory train meshes.
# STAGE=encode|encode_shape_ss|ckpts|smoke_ss|smoke_shape|smoke_tex|train_ss|train_shape|train_tex|all
set -euo pipefail

ROOT="${ROOT:-/home/dharani/work/BlueFox3D}"
DATA="${DATA:-$ROOT/datasets/MyCategory}"
DT="$ROOT/data_toolkit"
LOG_DIR="${LOG_DIR:-$ROOT/results/logs}"
STAGE="${STAGE:-encode}"
INST="${INST:-$DATA/splits/train_sha256.txt}"

cd "$ROOT"
# shellcheck disable=SC1090
source ~/activate_bluefox3d.sh
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_XET=1
export PYTHONPATH="$ROOT/data_toolkit:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$LOG_DIR" "$ROOT/ckpts"

echo "===== $(date -Is) host=$(hostname) stage=$STAGE ====="
nvidia-smi -L || true
df -h / | tail -n 1

download_pt() {
  local hf_name="$1"
  local out="$2"
  if [ -f "$out" ]; then
    echo "already $out"
    return 0
  fi
  python - <<PY
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import torch
st = hf_hub_download("Hydrilla/BlueFox3D", "ckpts/${hf_name}.safetensors")
sd = load_file(st)
torch.save(sd, "${out}")
print("wrote", "${out}", "keys", len(sd))
PY
}

download_ckpts() {
  download_pt "ss_flow_img_dit_1_3B_64_bf16" "$ROOT/ckpts/ss_flow_img_dit_1_3B_64_bf16.pt"
  download_pt "slat_flow_img2shape_dit_1_3B_1024_bf16" "$ROOT/ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.pt"
  download_pt "slat_flow_imgshape2tex_dit_1_3B_1024_bf16" "$ROOT/ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.pt"
}

encode_all() {
  python "$DT/build_metadata.py" MyCategory --root "$DATA"
  echo "dump_pbr"
  python "$DT/dump_pbr.py" MyCategory --root "$DATA" --instances "$INST" --max_workers 4
  python "$DT/build_metadata.py" MyCategory --root "$DATA"
  echo "dual_grid_view 1024"
  python "$DT/dual_grid_view.py" MyCategory --root "$DATA" --resolution 1024 --view_indices 0-1 --instances "$INST" --max_workers 4
  python "$DT/build_metadata.py" MyCategory --root "$DATA"
  echo "voxelize_pbr_view 1024"
  python "$DT/voxelize_pbr_view.py" MyCategory --root "$DATA" --resolution 1024 --view_indices 0-1 --instances "$INST" --max_workers 4
  python "$DT/build_metadata.py" MyCategory --root "$DATA"
  echo "encode_shape 1024"
  python "$DT/encode_shape_latent_view.py" --root "$DATA" --resolution 1024 --view_indices 0-1 --instances "$INST"
  python "$DT/build_metadata.py" MyCategory --root "$DATA"
  echo "encode_pbr 1024"
  python "$DT/encode_pbr_latent_view.py" --root "$DATA" --resolution 1024 --view_indices 0-1 --instances "$INST"
  python "$DT/build_metadata.py" MyCategory --root "$DATA"
  echo "encode_ss 64"
  python "$DT/encode_ss_latent_view.py" --root "$DATA" \
    --shape_latent_name shape_enc_next_dc_f16c32_fp16_1024 \
    --resolution 64 --view_indices 0-1 --instances "$INST"
  python "$DT/build_metadata.py" MyCategory --root "$DATA"
  echo "ENCODE_OK"
}

encode_shape_and_ss() {
  echo "encode_shape 1024"
  python "$DT/encode_shape_latent_view.py" --root "$DATA" --resolution 1024 --view_indices 0-1 --instances "$INST"
  python "$DT/build_metadata.py" MyCategory --root "$DATA"
  echo "encode_ss 64"
  python "$DT/encode_ss_latent_view.py" --root "$DATA" \
    --shape_latent_name shape_enc_next_dc_f16c32_fp16_1024 \
    --resolution 64 --view_indices 0-1 --instances "$INST"
  python "$DT/build_metadata.py" MyCategory --root "$DATA"
  echo "ENCODE_SHAPE_SS_OK"
}

SS_DATA="{\"MyCategory\":{\"base\":\"$DATA\",\"ss_latent\":\"$DATA/ss_latents/ss_enc_conv3d_16l8_fp16_64_view\",\"render_cond\":\"$DATA/renders_cond\"}}"
SHAPE_DATA="{\"MyCategory\":{\"base\":\"$DATA\",\"shape_latent\":\"$DATA/shape_latents/shape_enc_next_dc_f16c32_fp16_1024_view\",\"render_cond\":\"$DATA/renders_cond\"}}"
TEX_DATA="{\"MyCategory\":{\"base\":\"$DATA\",\"shape_latent\":\"$DATA/shape_latents/shape_enc_next_dc_f16c32_fp16_1024_view\",\"pbr_latent\":\"$DATA/pbr_latents/tex_enc_next_dc_f16c32_fp16_1024_view_fix\",\"render_cond\":\"$DATA/renders_cond\"}}"

train_stage() {
  local cfg="$1"
  local out="$2"
  python train.py --config "$cfg" --output_dir "$out" --num_gpus 1 --auto_retry 0 --data_dir "$3"
}

case "$STAGE" in
  ckpts) download_ckpts ;;
  encode) encode_all ;;
  encode_shape_ss) encode_shape_and_ss ;;
  smoke_ss)
    download_ckpts
    train_stage configs/gen/ss64_mycategory_l4_smoke.json results/ss64_character_smoke "$SS_DATA"
    ;;
  smoke_shape)
    download_ckpts
    train_stage configs/gen/shape1024_mycategory_l4_smoke.json results/shape1024_character_smoke "$SHAPE_DATA"
    ;;
  smoke_tex)
    download_ckpts
    train_stage configs/gen/tex1024_mycategory_l4_smoke.json results/tex1024_character_smoke "$TEX_DATA"
    ;;
  train_ss)
    train_stage configs/gen/ss64_mycategory_l4.json results/ss64_character "$SS_DATA"
    ;;
  train_shape)
    train_stage configs/gen/shape1024_mycategory_l4.json results/shape1024_character "$SHAPE_DATA"
    ;;
  train_tex)
    train_stage configs/gen/tex1024_mycategory_l4.json results/tex1024_character "$TEX_DATA"
    ;;
  all)
    download_ckpts
    encode_all
    train_stage configs/gen/ss64_mycategory_l4_smoke.json results/ss64_character_smoke "$SS_DATA"
    train_stage configs/gen/ss64_mycategory_l4.json results/ss64_character "$SS_DATA"
    train_stage configs/gen/shape1024_mycategory_l4_smoke.json results/shape1024_character_smoke "$SHAPE_DATA"
    train_stage configs/gen/shape1024_mycategory_l4.json results/shape1024_character "$SHAPE_DATA"
    train_stage configs/gen/tex1024_mycategory_l4_smoke.json results/tex1024_character_smoke "$TEX_DATA"
    train_stage configs/gen/tex1024_mycategory_l4.json results/tex1024_character "$TEX_DATA"
    ;;
  *)
    echo "Unknown STAGE=$STAGE" >&2
    exit 2
    ;;
esac

echo "===== $(date -Is) done stage=$STAGE ====="
