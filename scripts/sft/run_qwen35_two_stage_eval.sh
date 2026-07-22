#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "Usage: $0 CHECKPOINT_LABEL ADAPTER_PATH" >&2
  exit 2
fi

label="$1"
adapter_path="$(realpath "$2")"
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_dir"

root="${QWEN35_SFT_ROOT:-artifacts/appworld_sft/qwen35_4b_two_stage_20260722}"
summary_dir="$root/evaluation/$label"
model_path="${QWEN35_4B_PATH:-.model_cache/Qwen/Qwen3.5-4B}"
python_bin="${PYTHON_BIN:-/data/ganpa/miniconda3/envs/appworld/bin/python}"
appworld_env_bin="${APPWORLD_ENV_BIN:-/data/ganpa/miniconda3/envs/appworld/bin}"
export APPWORLD_ROOT="${APPWORLD_ROOT:-/data/ganpa/dragongong/appworld-data}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TMPDIR="${TMPDIR:-/data/ganpa/q35-$label}"
export RAY_TMPDIR="$TMPDIR"
mkdir -p "$summary_dir" "$TMPDIR"

for filename in adapter_config.json adapter_model.safetensors; do
  if [[ ! -f "$adapter_path/$filename" ]]; then
    echo "Incomplete adapter: $adapter_path/$filename" >&2
    exit 3
  fi
done
if [[ ! -f "$model_path/model.safetensors.index.json" ]]; then
  echo "Qwen3.5 base model is incomplete: $model_path" >&2
  exit 4
fi
visible_gpu_count="$($python_bin -c 'import torch; print(torch.cuda.device_count())')"
if [[ "$visible_gpu_count" != "4" ]]; then
  echo "Expected four visible GPUs on ganpa, found $visible_gpu_count" >&2
  exit 5
fi
compute_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
  | sed '/^[[:space:]]*$/d' || true)"
if [[ -n "$compute_pids" ]]; then
  echo "Refusing to start while GPU compute processes are active: $compute_pids" >&2
  exit 6
fi
available_kb="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
if ((available_kb < 32 * 1024 * 1024)); then
  echo "MemAvailable is below 32 GiB before evaluation" >&2
  exit 7
fi

main_log="$summary_dir/eval.log"
resource_log="$summary_dir/resource_samples.log"
smoke_json="$summary_dir/lora_smoke.json"

if [[ ! -f "$smoke_json" ]]; then
  echo "Running fixed-prompt Qwen3.5 LoRA smoke on GPU 0" >> "$main_log"
  CUDA_VISIBLE_DEVICES=0 "$python_bin" -m scripts.sft.smoke_qwen35_lora_vllm \
    --model-path "$model_path" \
    --adapter-path "$adapter_path" \
    --output "$smoke_json" \
    --gpu-memory-utilization 0.88 \
    >> "$main_log" 2>&1
fi

run_attempt() {
  local attempt="$1"
  local runners="$2"
  local max_num_seqs="$3"
  local gpu_utilization="$4"
  local attempt_log="$summary_dir/eval_attempt_${attempt}.log"
  local monitor_log="$summary_dir/resource_attempt_${attempt}.log"
  local eval_pid
  local monitor_pid
  local eval_status
  local monitor_status
  local -a command=(
    "$python_bin" -m scripts.loop7b.eval_watch
    --mode adapter
    --adapter-path "$adapter_path"
    --checkpoint-name "$label"
    --run-name "qwen35_4b_sft_${label}"
    --summary-dir "$summary_dir"
    --split dev
    --no-run-base-if-missing
    --no-wait-for-gpu-idle
    --cuda-visible-devices "$CUDA_VISIBLE_DEVICES"
    --num-scenario-runners "$runners"
    --llm qwen_3_5_4b_lora32_eval
    --max-gpu-mem-utilization "$gpu_utilization"
    --no-eager-mode
    --max-model-len 16384
    --max-new-tokens 1200
    --appworld-env-bin "$appworld_env_bin"
    --hydra-override "llm.base_model_path=$model_path"
    --hydra-override "llm.vllm_server.max_num_seqs=$max_num_seqs"
    --hydra-override "eval_seed=20260722"
    --hydra-override "rollout_seeds=[2026072200]"
  )

  {
    echo "===== $(date --iso-8601=seconds) ====="
    echo "attempt=$attempt runners=$runners max_num_seqs=$max_num_seqs gpu_utilization=$gpu_utilization"
    printf 'command='
    printf '%q ' "${command[@]}"
    echo
  } >> "$attempt_log"

  set +e
  setsid "${command[@]}" >> "$attempt_log" 2>&1 &
  eval_pid=$!
  bash scripts/sft/monitor_qwen35_resources.sh "$eval_pid" "$monitor_log" 15 &
  monitor_pid=$!
  wait "$eval_pid"
  eval_status=$?
  wait "$monitor_pid"
  monitor_status=$?
  set -e
  printf '%s attempt_exit attempt=%s eval_status=%s monitor_status=%s\n' \
    "$(date --iso-8601=seconds)" "$attempt" "$eval_status" "$monitor_status" \
    >> "$attempt_log"
  cat "$attempt_log" >> "$main_log"
  cat "$monitor_log" >> "$resource_log"
  ((eval_status == 0 && monitor_status == 0))
}

if ! run_attempt primary 32 8 0.88; then
  if grep -Eqi 'CUDA out of memory|OutOfMemoryError|out of memory|retryable' \
    "$summary_dir/eval_attempt_primary.log" "$summary_dir/logs/"*.log 2>/dev/null; then
    cat > "$summary_dir/FALLBACK.json" <<EOF
{
  "reason": "primary resource-pressure failure",
  "num_scenario_runners": 24,
  "max_num_seqs": 6,
  "max_gpu_mem_utilization": 0.84
}
EOF
    run_attempt fallback 24 6 0.84
  else
    echo "Primary evaluation failed for a non-resource reason" >&2
    exit 8
  fi
fi

"$python_bin" - "$summary_dir/summary.json" "$label" <<'PY'
import json
import sys

path, label = sys.argv[1:]
rows = json.load(open(path))
matching = [row for row in rows if row.get("checkpoint_name") == label]
if len(matching) != 1:
    raise SystemExit(f"Expected one summary row for {label}, found {len(matching)}")
row = matching[0]
if row.get("split") != "dev" or row.get("episode_count") != 57:
    raise SystemExit(f"Incomplete dev evaluation: {row}")
if row.get("num_rollouts_analyzed") != 57:
    raise SystemExit(f"Behavior analysis did not cover 57 rollouts: {row}")
print(json.dumps(row, sort_keys=True))
PY

echo "$label dev evaluation complete: $summary_dir/summary.csv"
