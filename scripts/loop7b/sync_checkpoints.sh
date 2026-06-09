#!/usr/bin/env bash
# 中文注释：把完整 LOOP checkpoint 从训练机安全同步到评测机。
set -euo pipefail

: "${SOURCE_RUN_DIR:?Set SOURCE_RUN_DIR to the training run directory.}"
: "${DEST_RUN_DIR:?Set DEST_RUN_DIR to the destination checkpoint directory.}"

DEST_HOST="${DEST_HOST:-ganpa@10.103.68.253}"
DEST_SSH_OPTS="${DEST_SSH_OPTS:--p 22}"
EVAL_EVERY="${EVAL_EVERY:-10}"
POLL_SECONDS="${POLL_SECONDS:-300}"
ONCE="${ONCE:-0}"

read -r -a DEST_SSH_ARGS <<< "$DEST_SSH_OPTS"

is_complete_checkpoint() {
  local ckpt_dir="$1"
  [[ -f "$ckpt_dir/trainer_state.pt" ]] || return 1
  [[ -f "$ckpt_dir/lora/adapter_config.json" || -f "$ckpt_dir/lora_vllmqwen25/adapter_config.json" ]]
}

sync_one_checkpoint() {
  local ckpt_dir="$1"
  local ckpt_name
  ckpt_name="$(basename "$ckpt_dir")"
  local tmp_dir="${DEST_RUN_DIR}/.${ckpt_name}.tmp.$$"
  local final_dir="${DEST_RUN_DIR}/${ckpt_name}"

  if ssh "${DEST_SSH_ARGS[@]}" "$DEST_HOST" "[ -f '$final_dir/.complete' ]"; then
    echo "skip existing complete checkpoint: $ckpt_name"
    return 0
  fi

  echo "syncing $ckpt_name -> ${DEST_HOST}:${final_dir}"
  ssh "${DEST_SSH_ARGS[@]}" "$DEST_HOST" "mkdir -p '$tmp_dir' '$(dirname "$final_dir")'"
  rsync -a -e "ssh ${DEST_SSH_OPTS}" "${ckpt_dir}/" "${DEST_HOST}:${tmp_dir}/"
  ssh "${DEST_SSH_ARGS[@]}" "$DEST_HOST" "\
    if [ -e '$final_dir' ] && [ ! -f '$final_dir/.complete' ]; then \
      echo 'destination exists without .complete: $final_dir' >&2; exit 2; \
    fi; \
    if [ ! -e '$final_dir' ]; then mv '$tmp_dir' '$final_dir'; fi; \
    touch '$final_dir/.complete'"
}

while true; do
  shopt -s nullglob
  for ckpt_dir in "$SOURCE_RUN_DIR"/checkpoint-*; do
    ckpt_name="$(basename "$ckpt_dir")"
    iteration="${ckpt_name#checkpoint-}"
    if ! [[ "$iteration" =~ ^[0-9]+$ ]]; then
      continue
    fi
    if (( iteration % EVAL_EVERY != 0 )); then
      continue
    fi
    if ! is_complete_checkpoint "$ckpt_dir"; then
      echo "skip incomplete checkpoint: $ckpt_name"
      continue
    fi
    sync_one_checkpoint "$ckpt_dir"
  done

  if [[ "$ONCE" == "1" ]]; then
    break
  fi
  sleep "$POLL_SECONDS"
done
