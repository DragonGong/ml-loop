#!/usr/bin/env bash
set -euo pipefail

set -a
source "$HOME/.config/ml-loop/deepseek.env"
set +a

export APPWORLD_ROOT=/home/yunlong/dragongong/appworld-data

mkdir -p artifacts/appworld_sft/run_logs

run_with_restart() {
  local split="$1"
  local successes_per_task="$2"
  local max_attempts_per_task="$3"
  local cost_limit_cny="$4"
  local output_dir="$5"
  local log_path="$6"

  while true; do
    if conda run --no-capture-output -n ml-loop-py312 python -u -m phi_agents.sft.teacher \
      --split "$split" \
      --successes-per-task "$successes_per_task" \
      --max-attempts-per-task "$max_attempts_per_task" \
      --cost-limit-cny "$cost_limit_cny" \
      --output-dir "$output_dir" \
      2>&1 | tee -a "$log_path"; then
      return 0
    fi
    printf '%s teacher worker exited unexpectedly; restarting from manifest in 15 seconds\n' \
      "$(date --iso-8601=seconds)" | tee -a "$log_path"
    sleep 15
  done
}

run_with_restart \
  train_difficulty_1_2 14 28 100 \
  artifacts/appworld_sft/teacher_d12_v1 \
  artifacts/appworld_sft/run_logs/teacher_d12_v1.log

run_with_restart \
  train_difficulty_3 14 28 25 \
  artifacts/appworld_sft/teacher_d3_v1 \
  artifacts/appworld_sft/run_logs/teacher_d3_v1.log
