#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "Usage: $0 LABEL ADAPTER_PATH" >&2
  exit 2
fi

label="$1"
adapter_path="$(realpath "$2")"
host="${GANPA_HOST:-ganpa253}"
remote_root="${GANPA_QWEN35_SFT_ROOT:-/data/ganpa/dragongong/ml-loop/artifacts/appworld_sft/qwen35_4b_two_stage_20260722}"
remote_path="$remote_root/adapters/$label"
remote_temp="$remote_root/adapters/.incoming-${label}-$$"

if [[ ! "$label" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "Unsafe adapter label: $label" >&2
  exit 3
fi
for filename in adapter_config.json adapter_model.safetensors; do
  if [[ ! -f "$adapter_path/$filename" ]]; then
    echo "Incomplete adapter: $adapter_path/$filename" >&2
    exit 4
  fi
done

sha256="$(sha256sum "$adapter_path/adapter_model.safetensors" | awk '{print $1}')"
existing_sha="$(ssh "$host" "if [ -f '$remote_path/adapter_model.safetensors' ]; then sha256sum '$remote_path/adapter_model.safetensors' | cut -d' ' -f1; fi")"
if [[ -n "$existing_sha" ]]; then
  if [[ "$existing_sha" == "$sha256" ]]; then
    echo "$remote_path already contains adapter sha256=$sha256"
    exit 0
  fi
  echo "Refusing to overwrite a different remote adapter: $remote_path" >&2
  exit 5
fi

ssh "$host" "mkdir -p '$remote_temp'"
cleanup() {
  local status=$?
  trap - EXIT
  if ((status != 0)); then
    ssh "$host" "rm -rf -- '$remote_temp'" 2>/dev/null || true
  fi
  exit "$status"
}
trap cleanup EXIT

files=("$adapter_path/adapter_config.json" "$adapter_path/adapter_model.safetensors")
[[ ! -f "$adapter_path/README.md" ]] || files+=("$adapter_path/README.md")
scp "${files[@]}" "$host:$remote_temp/"

ssh "$host" bash -s -- "$remote_temp" "$remote_path" "$sha256" <<'REMOTE'
set -euo pipefail
temp_path="$1"
final_path="$2"
expected_sha="$3"
for filename in adapter_config.json adapter_model.safetensors; do
  [[ -f "$temp_path/$filename" ]]
done
actual_sha="$(sha256sum "$temp_path/adapter_model.safetensors" | awk '{print $1}')"
[[ "$actual_sha" == "$expected_sha" ]]
printf '%s  adapter_model.safetensors\n' "$actual_sha" > "$temp_path/adapter_model.sha256"
printf 'complete\n' > "$temp_path/.complete"
if [[ -e "$final_path" ]]; then
  echo "Remote destination appeared during transfer: $final_path" >&2
  exit 6
fi
mv "$temp_path" "$final_path"
REMOTE

echo "Transferred $label to $host:$remote_path sha256=$sha256"
