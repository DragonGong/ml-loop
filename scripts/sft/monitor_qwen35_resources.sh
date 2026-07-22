#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 2 || "$#" -gt 3 ]]; then
  echo "Usage: $0 ROOT_PID RESOURCE_LOG [INTERVAL_SECONDS]" >&2
  exit 2
fi

root_pid="$1"
resource_log="$2"
interval_seconds="${3:-15}"
min_available_kb=$((32 * 1024 * 1024))
max_swap_growth_kb=$((4 * 1024 * 1024))
baseline_swap_used_kb="$(awk '
  /^SwapTotal:/ {total=$2}
  /^SwapFree:/ {free=$2}
  END {print total-free}
' /proc/meminfo)"

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

terminate_process_tree() {
  local reason="$1"
  local pid
  local -a process_tree=()
  mapfile -t process_tree < <(collect_process_tree "$root_pid")
  printf '%s guard_stop reason=%s root_pid=%s process_count=%s\n' \
    "$(date --iso-8601=seconds)" "$reason" "$root_pid" "${#process_tree[@]}" \
    >> "$resource_log"
  ((${#process_tree[@]} == 0)) || kill -TERM "${process_tree[@]}" 2>/dev/null || true
  sleep 15
  for pid in "${process_tree[@]}"; do
    kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" 2>/dev/null || true
  done
}

printf '%s monitor_start root_pid=%s baseline_swap_used_kb=%s\n' \
  "$(date --iso-8601=seconds)" "$root_pid" "$baseline_swap_used_kb" >> "$resource_log"

while kill -0 "$root_pid" 2>/dev/null; do
  available_kb="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
  swap_used_kb="$(awk '
    /^SwapTotal:/ {total=$2}
    /^SwapFree:/ {free=$2}
    END {print total-free}
  ' /proc/meminfo)"
  swap_growth_kb=$((swap_used_kb - baseline_swap_used_kb))

  {
    printf '\n===== %s root_pid=%s swap_growth_kb=%s =====\n' \
      "$(date --iso-8601=seconds)" "$root_pid" "$swap_growth_kb"
    free -m
    nvidia-smi \
      --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw \
      --format=csv,noheader,nounits
    ps -eo pid,ppid,pgid,rss,stat,cmd --sort=-rss | head -n 18
  } >> "$resource_log" 2>&1

  if ((available_kb < min_available_kb)); then
    terminate_process_tree "mem_available_below_32gib"
    exit 3
  fi
  if ((swap_growth_kb > max_swap_growth_kb)); then
    terminate_process_tree "swap_growth_above_4gib"
    exit 4
  fi
  sleep "$interval_seconds"
done

printf '%s monitor_exit root_pid=%s\n' \
  "$(date --iso-8601=seconds)" "$root_pid" >> "$resource_log"
