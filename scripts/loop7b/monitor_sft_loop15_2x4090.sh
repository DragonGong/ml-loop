#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 2 || "$#" -gt 4 ]]; then
  echo "Usage: $0 TRAINING_PID RESOURCE_LOG [INTERVAL_SECONDS] [PID_FILE]" >&2
  exit 2
fi

training_pid="$1"
resource_log="$2"
interval_seconds="${3:-15}"
pid_file="${4:-}"
min_available_kb=$((32 * 1024 * 1024))
max_swap_used_kb=$((8 * 1024 * 1024))

mkdir -p "$(dirname "$resource_log")"

collect_process_tree() {
  local parent_pid="$1"
  local child_pid
  while read -r child_pid; do
    [[ -n "$child_pid" ]] || continue
    collect_process_tree "$child_pid"
  done < <(pgrep -P "$parent_pid" 2>/dev/null || true)
  printf '%s\n' "$parent_pid"
}

terminate_training_tree() {
  local reason="$1"
  local pid
  local -a process_tree=()
  mapfile -t process_tree < <(collect_process_tree "$training_pid")
  printf '%s guard_stop reason=%s root_pid=%s process_count=%s\n' \
    "$(date --iso-8601=seconds)" "$reason" "$training_pid" "${#process_tree[@]}" >> "$resource_log"
  ((${#process_tree[@]} == 0)) || kill -TERM "${process_tree[@]}" 2>/dev/null || true
  sleep 15
  for pid in "${process_tree[@]}"; do
    kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" 2>/dev/null || true
  done
}

while kill -0 "$training_pid" 2>/dev/null; do
  available_kb="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
  swap_total_kb="$(awk '/^SwapTotal:/ {print $2}' /proc/meminfo)"
  swap_free_kb="$(awk '/^SwapFree:/ {print $2}' /proc/meminfo)"
  swap_used_kb=$((swap_total_kb - swap_free_kb))

  {
    printf '\n===== %s root_pid=%s =====\n' "$(date --iso-8601=seconds)" "$training_pid"
    free -m
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits
    ps -eo pid,ppid,pgid,rss,stat,cmd --sort=-rss | head -n 18
  } >> "$resource_log" 2>&1

  if (( available_kb < min_available_kb )); then
    terminate_training_tree "mem_available_below_32gib"
    exit 3
  fi
  if (( swap_used_kb > max_swap_used_kb )); then
    terminate_training_tree "swap_used_above_8gib"
    exit 4
  fi

  sleep "$interval_seconds"
done

if [[ -n "$pid_file" && -f "$pid_file" && "$(cat "$pid_file")" == "$training_pid" ]]; then
  rm -f "$pid_file"
fi
printf '%s monitor_exit root_pid=%s\n' "$(date --iso-8601=seconds)" "$training_pid" >> "$resource_log"
