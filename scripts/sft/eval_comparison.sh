#!/usr/bin/env bash
set -euo pipefail

: "${APPWORLD_ROOT:?Set APPWORLD_ROOT before running.}"

SPLIT="${1:-dev_small64}"
SFT_ADAPTER="${2:?Pass the SFT final_adapter or checkpoint directory as argument 2.}"
OUTPUT_ROOT="${3:-artifacts/appworld_sft/evaluation/${SPLIT}}"
LOOP_ADAPTER="${LOOP_CHECKPOINT_130:-experiments/qwen25_7b_loop_200x24x6_lora16/2026-06-08_10-12-49/checkpoint-130/lora}"

if [[ "$SPLIT" != "dev" && "$SPLIT" != "dev_small64" ]]; then
  echo "SFT comparison only permits dev or dev_small64, got: $SPLIT" >&2
  exit 2
fi
if [[ ! -d "$SFT_ADAPTER" ]]; then
  echo "SFT adapter not found: $SFT_ADAPTER" >&2
  exit 2
fi
if [[ ! -d "$LOOP_ADAPTER" ]]; then
  echo "LOOP checkpoint-130 adapter not found: $LOOP_ADAPTER" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT"

run_one() {
  local label="$1"
  local adapter="$2"
  local llm_config="$3"
  local experiment="sft_compare_${label}_${SPLIT}"
  local result_dir="$OUTPUT_ROOT/$label"
  mkdir -p "$result_dir"
  LOOP7B_EVAL_LLM="$llm_config" \
    bash scripts/loop7b/eval_4x3090.sh \
      "$SPLIT" "$adapter" "$experiment" "log_dir=$result_dir"
  conda run -n ml-loop-py312 python -m scripts.loop7b.analyze_appworld_behavior \
    --experiment-name "$experiment" \
    --all-results "$result_dir/all_results.txt" \
    --checkpoint-name "$label" \
    --eval-result-path "$result_dir/all_results.txt" \
    --eval-log-path "$result_dir" \
    --output-json "$result_dir/behavior.json" \
    --output-csv "$result_dir/behavior.csv"
}

run_one "qwen25_7b_base" "null" "qwen_2_5_7b_eval"
run_one "loop_checkpoint_130" "$LOOP_ADAPTER" "qwen_2_5_7b_lora16_eval"
run_one "qwen25_7b_sft" "$SFT_ADAPTER" "qwen_2_5_7b_lora64_eval"

conda run -n ml-loop-py312 python -m scripts.sft.summarize_comparison \
  --input "$OUTPUT_ROOT/qwen25_7b_base/behavior.json" \
  --input "$OUTPUT_ROOT/loop_checkpoint_130/behavior.json" \
  --input "$OUTPUT_ROOT/qwen25_7b_sft/behavior.json" \
  --output-dir "$OUTPUT_ROOT"
