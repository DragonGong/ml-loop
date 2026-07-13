#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:?Usage: $0 d12_25|d12_50|d12_100|d3 <initial-adapter-for-d3>}"
ROOT="${COMPACT_SFT_ROOT:-artifacts/appworld_sft/compact_lora_20260713}"
MODEL_PATH="${QWEN25_7B_PATH:-.model_cache/Qwen/Qwen2.5-7B-Instruct}"
PYTHON_BIN="${PYTHON_BIN:-python}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

common_args=(
  --model-name Qwen/Qwen2.5-7B-Instruct
  --model-path "$MODEL_PATH"
  --max-length 16384
  --lora-rank 32
  --lora-alpha 64
  --lora-dropout 0.05
  --weight-decay 0.01
  --max-grad-norm 1.0
  --lr-scheduler-type cosine
  --attention-backend auto
  --seed 20260713
  --logging-steps 5
)

case "$STAGE" in
  d12_25|d12_50|d12_100)
    data_dir="$ROOT/data/d12_nested/$STAGE"
    run_dir="$ROOT/runs/$STAGE"
    stage_args=(
      --train-jsonl "$data_dir/qwen_sft_train.jsonl"
      --validation-jsonl "$data_dir/qwen_sft_validation.jsonl"
      --output-dir "$run_dir"
      --learning-rate 5e-5
      --epochs 2
      --gradient-accumulation-steps 8
      --warmup-ratio 0.05
    )
    ;;
  d3)
    initial_adapter="${2:?Pass the best D12 checkpoint lora directory for D3 continuation.}"
    data_dir="$ROOT/data/d3_supervision_v2"
    run_name="${D3_RUN_NAME:-d3_from_best_d12}"
    stage_args=(
      --train-jsonl "$data_dir/qwen_sft_train.jsonl"
      --validation-jsonl "$data_dir/qwen_sft_validation.jsonl"
      --output-dir "$ROOT/runs/$run_name"
      --initial-adapter "$initial_adapter"
      --learning-rate 1e-5
      --epochs 2
      --gradient-accumulation-steps 4
      --warmup-ratio 0.10
    )
    ;;
  *)
    echo "Unknown stage: $STAGE" >&2
    exit 2
    ;;
esac

exec "$PYTHON_BIN" -m phi_agents.sft.launch -- \
  "$PYTHON_BIN" -m phi_agents.sft.trainer "${stage_args[@]}" "${common_args[@]}"
