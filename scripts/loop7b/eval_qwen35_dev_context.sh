#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "Usage: $0 MAX_MODEL_LEN SUMMARY_DIR" >&2
  exit 2
fi

max_model_len="$1"
summary_dir="$2"
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_dir"

: "${APPWORLD_ROOT:?Set APPWORLD_ROOT to an installed AppWorld data root.}"

python_bin="${PYTHON_BIN:-$(command -v python)}"
default_env_bin="$(dirname "$python_bin")"
appworld_env_bin="${APPWORLD_ENV_BIN:-$default_env_bin}"
model_path="${MODEL_PATH:-.model_cache/Qwen/Qwen3.5-4B}"
cuda_devices="${CUDA_VISIBLE_DEVICES:-0,1}"
num_runners="${NUM_SCENARIO_RUNNERS:-16}"
max_num_seqs="${MAX_NUM_SEQS:-8}"
gpu_utilization="${GPU_MEMORY_UTILIZATION:-0.82}"
run_name="${RUN_NAME:-qwen35_4b_ctx${max_model_len}}"
eval_seed="${EVAL_SEED:-20260722}"
rollout_seed="${ROLLOUT_SEED:-2026072200}"
tmpdir="${TMPDIR:-/tmp/qwen35-appworld}"
resource_log="${RESOURCE_LOG:-$summary_dir/resource_samples.log}"
main_log="${MAIN_LOG:-$summary_dir/eval.log}"

mkdir -p "$summary_dir" "$tmpdir"
summary_dir="$(realpath "$summary_dir")"
resource_log="$(realpath -m "$resource_log")"
main_log="$(realpath -m "$main_log")"

export CUDA_VISIBLE_DEVICES="$cuda_devices"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TMPDIR="$tmpdir"
export RAY_TMPDIR="$tmpdir"

expected_gpu_count="$(awk -F, '{print NF}' <<< "$cuda_devices")"
visible_gpu_count="$($python_bin -c 'import torch; print(torch.cuda.device_count())')"
if [[ "$visible_gpu_count" != "$expected_gpu_count" ]]; then
  echo "Expected $expected_gpu_count visible GPUs, found $visible_gpu_count" >&2
  exit 3
fi

if [[ ! -f "$model_path/model.safetensors.index.json" ]]; then
  echo "Model is incomplete or missing: $model_path" >&2
  exit 4
fi

available_kb="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
if (( available_kb < 32 * 1024 * 1024 )); then
  echo "MemAvailable is below 32 GiB before evaluation" >&2
  exit 5
fi

cleanup() {
  status=$?
  trap - EXIT INT TERM
  if [[ -n "${monitor_pid:-}" ]]; then
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
  fi
  if [[ -n "${eval_pid:-}" ]] && kill -0 "$eval_pid" 2>/dev/null; then
    kill -TERM -- "-$eval_pid" 2>/dev/null || kill -TERM "$eval_pid" 2>/dev/null || true
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

{
  echo "===== $(date --iso-8601=seconds) ====="
  echo "run_name=$run_name max_model_len=$max_model_len cuda_devices=$cuda_devices"
  echo "num_runners=$num_runners max_num_seqs=$max_num_seqs gpu_utilization=$gpu_utilization"
  "$python_bin" - <<'PY'
import torch
import transformers
import vllm

print(f"torch={torch.__version__} cuda={torch.version.cuda} devices={torch.cuda.device_count()}")
print(f"transformers={transformers.__version__} vllm={vllm.__version__}")
PY
  free -h
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader
} >> "$main_log" 2>&1

set +e
setsid "$python_bin" -m scripts.loop7b.eval_watch \
  --mode base \
  --run-name "$run_name" \
  --summary-dir "$summary_dir" \
  --split dev \
  --no-wait-for-gpu-idle \
  --cuda-visible-devices "$cuda_devices" \
  --num-scenario-runners "$num_runners" \
  --llm qwen_3_5_4b_eval \
  --max-gpu-mem-utilization "$gpu_utilization" \
  --no-eager-mode \
  --max-model-len "$max_model_len" \
  --max-new-tokens 1200 \
  --appworld-env-bin "$appworld_env_bin" \
  --hydra-override "llm.base_model_path=$model_path" \
  --hydra-override "llm.vllm_server.max_num_seqs=$max_num_seqs" \
  --hydra-override "eval_seed=$eval_seed" \
  --hydra-override "rollout_seeds=[$rollout_seed]" \
  >> "$main_log" 2>&1 &
eval_pid=$!

bash scripts/loop7b/monitor_sft_loop15_2x4090.sh \
  "$eval_pid" "$resource_log" 1 &
monitor_pid=$!

wait "$eval_pid"
eval_status=$?
wait "$monitor_pid"
monitor_status=$?
set -e

printf '%s eval_exit eval_status=%s monitor_status=%s\n' \
  "$(date --iso-8601=seconds)" "$eval_status" "$monitor_status" >> "$main_log"

if (( eval_status != 0 )); then
  exit "$eval_status"
fi
if (( monitor_status != 0 )); then
  exit "$monitor_status"
fi
