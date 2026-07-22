#!/usr/bin/env bash
# Preflight and launch the local two-4090 R1 continuation with resource monitoring.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_dir"

training_log="${TRAINING_LOG:-$repo_dir/logs/sft_loop15_r1_local_2x4090_20260719.log}"
resource_log="${RESOURCE_LOG:-$repo_dir/.codex_tmp/sft_loop15_r1_local_2x4090_resources.log}"
pid_file="${TRAINING_PID_FILE:-$repo_dir/.codex_tmp/sft_loop15_r1_local_2x4090.pid}"
monitor_pid_file="${MONITOR_PID_FILE:-$repo_dir/.codex_tmp/sft_loop15_r1_local_2x4090_monitor.pid}"
monitor_log="${MONITOR_LOG:-$repo_dir/.codex_tmp/sft_loop15_r1_local_2x4090_monitor.log}"
train_script="$repo_dir/scripts/loop7b/train_sft_loop15_2x4090.sh"
monitor_script="$repo_dir/scripts/loop7b/monitor_sft_loop15_2x4090.sh"

mkdir -p "$repo_dir/logs" "$repo_dir/.codex_tmp"

if [[ -s "$pid_file" ]]; then
  existing_pid="$(cat "$pid_file")"
  if kill -0 "$existing_pid" 2>/dev/null; then
    echo "Training is already running with PID $existing_pid" >&2
    exit 2
  fi
  rm -f "$pid_file"
fi

if ss -ltn | grep -Eq ':(5555|5556|5557|5558)[[:space:]]'; then
  echo "One of the required vLLM ports 5555-5558 is already in use" >&2
  exit 2
fi

available_kb="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
if (( available_kb < 32 * 1024 * 1024 )); then
  echo "Less than 32 GiB of host memory is available" >&2
  exit 2
fi

PREFLIGHT_ONLY=1 bash "$train_script" >> "$training_log" 2>&1

nohup setsid bash "$train_script" >> "$training_log" 2>&1 </dev/null &
training_pid=$!
printf '%s\n' "$training_pid" > "$pid_file"

sleep 3
if ! kill -0 "$training_pid" 2>/dev/null; then
  echo "Training exited during startup; inspect $training_log" >&2
  exit 1
fi

nohup setsid bash "$monitor_script" "$training_pid" "$resource_log" 15 "$pid_file" \
  >> "$monitor_log" 2>&1 </dev/null &
monitor_pid=$!
printf '%s\n' "$monitor_pid" > "$monitor_pid_file"

echo "training_pid=$training_pid"
echo "monitor_pid=$monitor_pid"
echo "training_log=$training_log"
echo "resource_log=$resource_log"
