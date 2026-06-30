#!/usr/bin/env bash
# 中文注释：启动 GRPO 训练，并同时运行 dev checkpoint 评测 watcher。
set -euo pipefail

: "${APPWORLD_ROOT:?Set APPWORLD_ROOT before running.}"
: "${HF_TOKEN:?Set HF_TOKEN before running.}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen25_7b_grpo_200x24x6_lora16}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y-%m-%d_%H-%M-%S)}"
RUN_DIR="${RUN_DIR:-$PWD/experiments/$EXPERIMENT_NAME/$RUN_STAMP}"
EVAL_SUMMARY_DIR="${EVAL_SUMMARY_DIR:-artifacts/grpo7b_stage2_dev_eval}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0}"
EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-1,2,3}"
EVAL_EVERY="${EVAL_EVERY:-10}"
EVAL_POLL_SECONDS="${EVAL_POLL_SECONDS:-300}"
EVAL_NUM_SCENARIO_RUNNERS="${EVAL_NUM_SCENARIO_RUNNERS:-12}"
EVAL_LLM="${EVAL_LLM:-qwen_2_5_7b_lora16_eval}"
EVAL_MAX_GPU_MEM_UTILIZATION="${EVAL_MAX_GPU_MEM_UTILIZATION:-0.82}"
EVAL_MAX_MODEL_LEN="${EVAL_MAX_MODEL_LEN:-16384}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-1200}"
EVAL_EAGER_MODE="${EVAL_EAGER_MODE:-true}"
if [[ "$EVAL_EAGER_MODE" == "0" || "$EVAL_EAGER_MODE" == "false" || "$EVAL_EAGER_MODE" == "False" ]]; then
  EVAL_EAGER_FLAG="--no-eager-mode"
else
  EVAL_EAGER_FLAG="--eager-mode"
fi

mkdir -p logs "$RUN_DIR" "$EVAL_SUMMARY_DIR" "$APPWORLD_ROOT/data/datasets"
cp data/appworld_splits/dev.txt "$APPWORLD_ROOT/data/datasets/dev.txt"

TRAIN_LOG="${TRAIN_LOG:-logs/${EXPERIMENT_NAME}_${RUN_STAMP}_train.log}"
WATCH_LOG="${WATCH_LOG:-logs/${EXPERIMENT_NAME}_${RUN_STAMP}_dev_eval_watch.log}"

watcher_pid=""

stop_watcher() {
  if [[ -z "${watcher_pid}" ]]; then
    return 0
  fi
  if kill -0 "$watcher_pid" 2>/dev/null; then
    echo "Stopping dev eval watcher pid=${watcher_pid}"
    kill "$watcher_pid" 2>/dev/null || true
    wait "$watcher_pid" 2>/dev/null || true
  fi
  watcher_pid=""
}

cleanup() {
  stop_watcher
}
trap cleanup EXIT INT TERM

run_watcher() {
  CUDA_VISIBLE_DEVICES="$EVAL_CUDA_VISIBLE_DEVICES" "$PYTHON_BIN" -m scripts.loop7b.eval_watch \
    --mode watch \
    --checkpoint-root "$RUN_DIR" \
    --run-name "$EXPERIMENT_NAME" \
    --summary-dir "$EVAL_SUMMARY_DIR" \
    --split dev \
    --eval-every "$EVAL_EVERY" \
    --poll-seconds "$EVAL_POLL_SECONDS" \
    --cuda-visible-devices "$EVAL_CUDA_VISIBLE_DEVICES" \
    --num-scenario-runners "$EVAL_NUM_SCENARIO_RUNNERS" \
    --llm "$EVAL_LLM" \
    --max-gpu-mem-utilization "$EVAL_MAX_GPU_MEM_UTILIZATION" \
    "$EVAL_EAGER_FLAG" \
    --max-model-len "$EVAL_MAX_MODEL_LEN" \
    --max-new-tokens "$EVAL_MAX_NEW_TOKENS" \
    "$@"
}

echo "run_dir=${RUN_DIR}"
echo "train_cuda_visible_devices=${TRAIN_CUDA_VISIBLE_DEVICES}"
echo "eval_cuda_visible_devices=${EVAL_CUDA_VISIBLE_DEVICES}"
echo "eval_summary_dir=${EVAL_SUMMARY_DIR}"

run_watcher >"$WATCH_LOG" 2>&1 &
watcher_pid=$!
echo "Started dev eval watcher pid=${watcher_pid}, log=${WATCH_LOG}"

train_status=0
CUDA_VISIBLE_DEVICES="$TRAIN_CUDA_VISIBLE_DEVICES" \
EXPERIMENT_NAME="$EXPERIMENT_NAME" \
TOTAL_ITERATIONS="${TOTAL_ITERATIONS:-200}" \
bash scripts/grpo7b/train_stage2_rtxpro6000.sh "$@" rl.cloud_path="$RUN_DIR" \
  >"$TRAIN_LOG" 2>&1 || train_status=$?

stop_watcher

if [[ "$train_status" -eq 0 ]]; then
  echo "Training completed; running one final dev eval watcher pass."
  run_watcher --once >>"$WATCH_LOG" 2>&1
else
  echo "Training failed with status=${train_status}; see ${TRAIN_LOG}" >&2
fi

exit "$train_status"
