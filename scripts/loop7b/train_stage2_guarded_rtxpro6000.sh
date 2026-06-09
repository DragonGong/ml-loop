#!/usr/bin/env bash
set -euo pipefail

: "${APPWORLD_ROOT:?Set APPWORLD_ROOT before running.}"
: "${HF_TOKEN:?Set HF_TOKEN before running.}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen25_7b_loop_200x24x6_lora16}"
MAX_HARD_DEAD_RESTARTS="${MAX_HARD_DEAD_RESTARTS:-3}"
HARD_DEAD_RESTART_WAIT_SECONDS="${HARD_DEAD_RESTART_WAIT_SECONDS:-90}"

repo_dir="$(pwd)"
run_dir="${RUN_DIR:-}"

infer_run_dir_from_args() {
  local arg
  for arg in "$@"; do
    case "$arg" in
      rl.cloud_path=*)
        printf '%s\n' "${arg#rl.cloud_path=}"
        return 0
        ;;
    esac
  done
  return 1
}

latest_run_dir() {
  if [[ -n "$run_dir" ]]; then
    printf '%s\n' "$run_dir"
    return 0
  fi
  if run_dir="$(infer_run_dir_from_args "$@")"; then
    printf '%s\n' "$run_dir"
    return 0
  fi
  ls -td "$repo_dir/experiments/$EXPERIMENT_NAME"/* 2>/dev/null | head -n 1
}

latest_manifest() {
  local dir="$1"
  ls -t "$dir"/iteration_manifests/iteration-*.json 2>/dev/null | head -n 1
}

is_hard_dead_abort() {
  local manifest="$1"
  [[ -f "$manifest" ]] || return 1
  python - "$manifest" <<'PY'
import json
import sys

path = sys.argv[1]
data = json.loads(open(path, encoding="utf-8").read())
reason = str(data.get("reason") or "")
hard_reasons = {"appworld_server_rss_65gb", "host_memory_used_85pct"}
sys.exit(0 if data.get("status") == "aborted" and reason in hard_reasons else 1)
PY
}

cleanup_training_residuals() {
  local pattern_repo
  pattern_repo="$(printf '%s' "$repo_dir" | sed 's/[.[\*^$()+?{}|]/\\&/g')"
  echo "Cleaning current-user residual training processes for $repo_dir"
  pkill -TERM -u "$(id -u)" -f "$pattern_repo/.+train.py.+$EXPERIMENT_NAME" || true
  pkill -TERM -u "$(id -u)" -f "VLLM::EngineCore" || true
  pkill -TERM -u "$(id -u)" -f "ray::VLLMServer" || true
  pkill -TERM -u "$(id -u)" -f "${TMPDIR:-/home/yunlong/dragongong/tmp}/ray/session_" || true
  pkill -TERM -u "$(id -u)" -f "uvicorn.*appworld|appworld.*server" || true
  sleep 10
  pkill -KILL -u "$(id -u)" -f "$pattern_repo/.+train.py.+$EXPERIMENT_NAME" || true
  pkill -KILL -u "$(id -u)" -f "VLLM::EngineCore" || true
  pkill -KILL -u "$(id -u)" -f "ray::VLLMServer" || true
  pkill -KILL -u "$(id -u)" -f "${TMPDIR:-/home/yunlong/dragongong/tmp}/ray/session_" || true
  pkill -KILL -u "$(id -u)" -f "uvicorn.*appworld|appworld.*server" || true
}

attempt=0
while true; do
  set +e
  bash scripts/loop7b/train_stage2_rtxpro6000.sh "$@"
  exit_code=$?
  set -e

  if [[ "$exit_code" -eq 0 ]]; then
    exit 0
  fi

  current_run_dir="$(latest_run_dir "$@")"
  manifest="$(latest_manifest "$current_run_dir")"
  if ! is_hard_dead_abort "$manifest"; then
    echo "Training failed with exit_code=$exit_code and latest manifest is not hard-dead aborted: $manifest" >&2
    exit "$exit_code"
  fi

  attempt=$((attempt + 1))
  if (( attempt > MAX_HARD_DEAD_RESTARTS )); then
    echo "Hard-dead restart limit exceeded: $MAX_HARD_DEAD_RESTARTS" >&2
    exit "$exit_code"
  fi

  echo "Hard-dead detected in $manifest; restarting attempt $attempt/$MAX_HARD_DEAD_RESTARTS after ${HARD_DEAD_RESTART_WAIT_SECONDS}s"
  sleep "$HARD_DEAD_RESTART_WAIT_SECONDS"
  cleanup_training_residuals
done
