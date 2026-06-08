#!/usr/bin/env bash
# 中文注释：在 4×3090 机器上并行评测 Qwen2.5-7B base 或 LoRA checkpoint。
set -euo pipefail

: "${APPWORLD_ROOT:?Set APPWORLD_ROOT before running.}"

SPLIT="${1:-dev_small64}"
ADAPTER_PATH="${2:-null}"
EXPERIMENT_NAME="${3:-}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PATH="$PWD/appworld-env/bin:$PATH"

if [[ -z "$EXPERIMENT_NAME" ]]; then
  if [[ "$ADAPTER_PATH" == "null" ]]; then
    EXPERIMENT_NAME="eval_base_qwen25_7b_${SPLIT}"
  else
    CKPT_NAME="$(basename "$ADAPTER_PATH")"
    EXPERIMENT_NAME="eval_loop7b_${CKPT_NAME}_${SPLIT}"
  fi
fi

"$PYTHON_BIN" -m scripts.appworld.run_inference \
  experiment_name="$EXPERIMENT_NAME" \
  llm=qwen_2_5_7b_eval \
  llm.adapter_path="$ADAPTER_PATH" \
  scenario_sampler.dataset_name="$SPLIT" \
  num_scenario_runners=16 \
  llm.max_gpu_mem_utilization="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}" \
  llm.vllm_server.gpus_per_vllm_server=1 \
  llm.vllm_server.max_model_len=16384 \
  llm.vllm_class.max_new_tokens=1200 \
  "${@:4}"

"$PYTHON_BIN" -m scripts.appworld.eval_parse_and_log \
  experiment_name="$EXPERIMENT_NAME" \
  scenario_sampler.dataset_name="$SPLIT"

"$PYTHON_BIN" -m scripts.loop7b.summarize_appworld_episodes "$EXPERIMENT_NAME"
