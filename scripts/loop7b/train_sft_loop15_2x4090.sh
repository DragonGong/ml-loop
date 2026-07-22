#!/usr/bin/env bash
# Resume the fixed SFT -> LOOP experiment on a local WSL2 host with two RTX 4090s.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_dir"

export APPWORLD_ROOT="${APPWORLD_ROOT:-/root/appworld-root}"
export CONDA_BASE="${CONDA_BASE:-/root/miniconda3}"
export CONDA_ENV="${CONDA_ENV:-appworld}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export GPU_ALLOCATION="two_gpu_shared"
export NUM_LEARNING_PROCESSES=2
export ACCELERATE_CONFIG_FILE="${ACCELERATE_CONFIG_FILE:-./phi_agents/rl/conf/accelerate_config_2x4090.yaml}"
# This value is per learner rank. Two ranks therefore provide 16 AppWorld runners.
export NUM_SCENARIO_RUNNERS="${NUM_SCENARIO_RUNNERS:-8}"
export TMPDIR="${TMPDIR:-/root/appworld-tmp}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
# NCCL detects that WSL cannot use CUDA peer access and selects SHM transport.
# Keeping discovery enabled is measurably faster than forcing P2P off.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export FSDP_OFFLOAD_PARAMS="${FSDP_OFFLOAD_PARAMS:-true}"
export FSDP_OFFLOAD_PIN_MEMORY="${FSDP_OFFLOAD_PIN_MEMORY:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PATH="$repo_dir/.codex_tmp/appworld-bin:$PATH"

mkdir -p "$TMPDIR"

wsl_model_cache="/root/model-cache/Qwen/Qwen2.5-7B-Instruct-d12_100_1epoch-merged-bf16"
if [[ -d "$wsl_model_cache" ]]; then
  export MERGED_BASE_MODEL="${MERGED_BASE_MODEL:-$wsl_model_cache}"
fi

IFS=',' read -r -a visible_gpus <<< "$CUDA_VISIBLE_DEVICES"
if [[ "${#visible_gpus[@]}" -ne 2 ]]; then
  echo "Expected exactly two CUDA devices, got: $CUDA_VISIBLE_DEVICES" >&2
  exit 2
fi

python_bin="$CONDA_BASE/envs/$CONDA_ENV/bin/python"
appworld_bin="$repo_dir/.codex_tmp/appworld-bin/appworld"
[[ -x "$python_bin" ]] || { echo "Python not found: $python_bin" >&2; exit 2; }
[[ -x "$appworld_bin" ]] || { echo "AppWorld wrapper not found: $appworld_bin" >&2; exit 2; }

exec bash scripts/loop7b/train_sft_loop15_rtxpro6000.sh \
  "$@" \
  rl.scenario_sampler.distributed_shard=true \
  llm.vllm_server.max_tries=900
