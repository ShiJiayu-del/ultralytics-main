#!/usr/bin/env bash
set -euo pipefail

# FLIR + PID training launcher with stable single-line tqdm refresh.
# Uses conda run --no-capture-output to avoid multiline progress bar rendering.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

ENV_NAME="${ENV_NAME:-sjyPID}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-2}"

MODEL="${MODEL:-ultralytics/cfg/models/v8/yolov8-pid-external.yaml}"
DATA="${DATA:-ultralytics/cfg/datasets/flir.yaml}"
IMGSZ="${IMGSZ:-640}"
EPOCHS="${EPOCHS:-100}"
BATCH="${BATCH:-64}"
DEVICE="${DEVICE:-0}"
WORKERS="${WORKERS:-0}"
PID_ENABLE="${PID_ENABLE:-True}"

# Optional plain text interval logs: 0 disables extra LOGGER.info batch lines.
YOLO_DISABLE_TQDM="${YOLO_DISABLE_TQDM:-1}"
YOLO_TEXT_LOG_INTERVAL="${YOLO_TEXT_LOG_INTERVAL:-10}"

# Optional extra args pass-through, e.g.:
# ./train_flir_pid.sh lr0=0.002 mosaic=0.5
EXTRA_ARGS=("$@")

echo "[train_flir_pid] cwd=$ROOT_DIR"
echo "[train_flir_pid] env=$ENV_NAME gpu=$CUDA_VISIBLE_DEVICES_VALUE"
echo "[train_flir_pid] model=$MODEL data=$DATA epochs=$EPOCHS batch=$BATCH device=$DEVICE workers=$WORKERS"

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE" \
  YOLO_DISABLE_TQDM="$YOLO_DISABLE_TQDM" \
  YOLO_TEXT_LOG_INTERVAL="$YOLO_TEXT_LOG_INTERVAL" \
  conda run --no-capture-output -n "$ENV_NAME" \
  yolo detect train \
  model="$MODEL" \
  data="$DATA" \
  imgsz="$IMGSZ" \
  epochs="$EPOCHS" \
  batch="$BATCH" \
  device="$DEVICE" \
  pid_enable="$PID_ENABLE" \
  workers="$WORKERS" \
  "${EXTRA_ARGS[@]}"
