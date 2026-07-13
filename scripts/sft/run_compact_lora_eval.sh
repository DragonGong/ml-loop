#!/usr/bin/env bash
set -euo pipefail

: "${APPWORLD_ROOT:=/home/yunlong/dragongong/appworld-data}"
export APPWORLD_ROOT
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SPLIT="${1:?Usage: $0 SPLIT CHECKPOINT_NAME ADAPTER_OR_null}"
CHECKPOINT_NAME="${2:?Pass a stable checkpoint label.}"
ADAPTER_PATH="${3:?Pass a checkpoint directory or null for the base model.}"
ROOT="${COMPACT_SFT_ROOT:-artifacts/appworld_sft/compact_lora_20260713}"
EXPERIMENT_NAME="compact_${CHECKPOINT_NAME}_${SPLIT}"
EVAL_DIR="$ROOT/evaluation/$SPLIT/$CHECKPOINT_NAME"
mkdir -p "$EVAL_DIR"

repo_split="data/appworld_splits/$SPLIT.txt"
appworld_split="$APPWORLD_ROOT/data/datasets/$SPLIT.txt"
if [[ -f "$repo_split" ]]; then
  if [[ -f "$appworld_split" ]] && ! cmp -s "$repo_split" "$appworld_split"; then
    echo "Refusing to overwrite a different AppWorld split: $appworld_split" >&2
    exit 2
  fi
  install -D -m 0644 "$repo_split" "$appworld_split"
fi

if [[ "$ADAPTER_PATH" == "null" ]]; then
  export LOOP7B_EVAL_LLM=qwen_2_5_7b_eval
else
  ADAPTER_PATH="$(realpath "$ADAPTER_PATH")"
  export LOOP7B_EVAL_LLM=qwen_2_5_7b_lora32_eval
fi

bash scripts/loop7b/eval_4x3090.sh \
  "$SPLIT" "$ADAPTER_PATH" "$EXPERIMENT_NAME" \
  "log_dir=$EVAL_DIR" \
  "llm.base_model_path=$PWD/.model_cache/Qwen/Qwen2.5-7B-Instruct"

all_results="$EVAL_DIR/all_results.txt"
if [[ ! -f "$all_results" ]]; then
  all_results="$(find "$EVAL_DIR" -name all_results.txt -print -quit)"
fi

python -m scripts.loop7b.analyze_appworld_behavior \
  --appworld-root "$APPWORLD_ROOT" \
  --experiment-name "$EXPERIMENT_NAME" \
  --all-results "$all_results" \
  --checkpoint-name "$CHECKPOINT_NAME" \
  --eval-result-path "$all_results" \
  --eval-log-path "$EVAL_DIR" \
  --output-json "$EVAL_DIR/behavior.json" \
  --output-csv "$EVAL_DIR/behavior.csv"
