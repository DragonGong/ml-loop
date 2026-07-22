#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 audit|smoke|d12|d3" >&2
  exit 2
fi

stage="$1"
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_dir"

root="${QWEN35_SFT_ROOT:-artifacts/appworld_sft/qwen35_4b_two_stage_20260722}"
data_root="${QWEN35_SFT_DATA_ROOT:-artifacts/appworld_sft/compact_lora_20260713/data}"
model_path="${QWEN35_4B_PATH:-.model_cache/Qwen/Qwen3.5-4B}"
python_bin="${PYTHON_BIN:-/root/miniconda3/envs/appworld/bin/python}"
cuda_devices="${CUDA_VISIBLE_DEVICES:-1,0}"
cpu_offload="${FSDP_CPU_OFFLOAD:-0}"

mkdir -p "$root"
root="$(realpath "$root")"
data_root="$(realpath "$data_root")"
model_path="$(realpath "$model_path")"

if [[ ! -x "$python_bin" ]]; then
  echo "Python environment is missing: $python_bin" >&2
  exit 3
fi
if [[ ! -f "$model_path/model.safetensors.index.json" ]]; then
  echo "Qwen3.5 model is incomplete: $model_path" >&2
  exit 4
fi

export CUDA_VISIBLE_DEVICES="$cuda_devices"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export TOKENIZERS_PARALLELISM=false
export TMPDIR="${TMPDIR:-/root/appworld-tmp/qwen35-sft}"
mkdir -p "$TMPDIR"

if [[ "$stage" == "audit" ]]; then
  exec "$python_bin" -m scripts.sft.audit_qwen35_sft_data \
    --repo-root "$repo_dir" \
    --data-root "$data_root" \
    --model-path "$model_path" \
    --output-dir "$root/audit"
fi

expected_gpu_count="$(awk -F, '{print NF}' <<< "$cuda_devices")"
if [[ "$expected_gpu_count" != "2" ]]; then
  echo "Training requires exactly two CUDA devices; got CUDA_VISIBLE_DEVICES=$cuda_devices" >&2
  exit 5
fi
visible_gpu_count="$($python_bin -c 'import torch; print(torch.cuda.device_count())')"
if [[ "$visible_gpu_count" != "2" ]]; then
  echo "Expected two visible GPUs, found $visible_gpu_count" >&2
  exit 6
fi
compute_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
  | sed '/^[[:space:]]*$/d' || true)"
if [[ -n "$compute_pids" ]]; then
  echo "Refusing to start while GPU compute processes are active: $compute_pids" >&2
  exit 7
fi
available_kb="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
if ((available_kb < 32 * 1024 * 1024)); then
  echo "MemAvailable is below 32 GiB before training" >&2
  exit 8
fi

common_args=(
  --model-name Qwen/Qwen3.5-4B
  --model-path "$model_path"
  --max-length 16384
  --lora-rank 32
  --lora-alpha 64
  --lora-dropout 0.05
  --per-device-train-batch-size 1
  --per-device-eval-batch-size 1
  --weight-decay 0.01
  --max-grad-norm 1.0
  --lr-scheduler-type cosine
  --attention-backend sdpa
  --seed 20260722
  --logging-steps 1
  --save-steps 10
  --save-strategy steps
  --save-total-limit 10
  --sparse-assistant-logits
  --fsdp-full-shard
)
if [[ "$cpu_offload" == "1" ]]; then
  common_args+=(--fsdp-cpu-offload)
fi

latest_resumable_checkpoint() {
  local output_dir="$1"
  local checkpoint
  local candidate=""
  local candidate_step=-1
  shopt -s nullglob
  for checkpoint in "$output_dir"/checkpoint-*; do
    local step="${checkpoint##*-}"
    [[ "$step" =~ ^[0-9]+$ ]] || continue
    [[ -f "$checkpoint/.complete" ]] || continue
    [[ -f "$checkpoint/trainer_state.json" ]] || continue
    [[ -f "$checkpoint/optimizer.pt" || -f "$checkpoint/optimizer.bin" ]] || continue
    [[ -f "$checkpoint/scheduler.pt" ]] || continue
    compgen -G "$checkpoint/rng_state*.pth" >/dev/null || continue
    [[ -f "$checkpoint/lora/adapter_config.json" ]] || continue
    [[ -f "$checkpoint/lora/adapter_model.safetensors" ]] || continue
    if ((step > candidate_step)); then
      candidate="$checkpoint"
      candidate_step="$step"
    fi
  done
  shopt -u nullglob
  printf '%s' "$candidate"
}

case "$stage" in
  smoke)
    smoke_stamp="$(date +%Y%m%d_%H%M%S)"
    output_dir="$root/smoke/run_${smoke_stamp}_offload${cpu_offload}"
    data_dir="$output_dir/data"
    "$python_bin" -m scripts.sft.build_qwen35_fsdp_smoke_data \
      --model-path "$model_path" \
      --output-dir "$data_dir" \
      --max-length 16384
    stage_args=(
      --train-jsonl "$data_dir/train.jsonl"
      --validation-jsonl "$data_dir/validation.jsonl"
      --output-dir "$output_dir"
      --learning-rate 1e-5
      --epochs 1
      --gradient-accumulation-steps 1
      --warmup-ratio 0
      --max-steps 1
      --save-steps 1
    )
    ;;
  d12)
    output_dir="$root/training/d12"
    data_dir="$data_root/d12_supervision_v2"
    stage_args=(
      --train-jsonl "$data_dir/qwen_sft_train.jsonl"
      --validation-jsonl "$data_dir/qwen_sft_validation.jsonl"
      --output-dir "$output_dir"
      --learning-rate 5e-5
      --epochs 1
      --gradient-accumulation-steps 4
      --warmup-ratio 0.05
    )
    ;;
  d3)
    output_dir="$root/training/d3"
    data_dir="$data_root/d3_supervision_v2"
    d12_adapter="$root/training/d12/final_adapter"
    if [[ ! -f "$d12_adapter/adapter_model.safetensors" ]]; then
      echo "D1/2 final adapter is missing: $d12_adapter" >&2
      exit 9
    fi
    stage_args=(
      --train-jsonl "$data_dir/qwen_sft_train.jsonl"
      --validation-jsonl "$data_dir/qwen_sft_validation.jsonl"
      --output-dir "$output_dir"
      --initial-adapter "$d12_adapter"
      --learning-rate 1e-5
      --epochs 2
      --gradient-accumulation-steps 2
      --warmup-ratio 0.10
    )
    ;;
  *)
    echo "Unknown stage: $stage" >&2
    exit 2
    ;;
esac

mkdir -p "$output_dir"
if [[ -f "$output_dir/final_adapter/adapter_model.safetensors" ]]; then
  echo "$stage is already complete: $output_dir/final_adapter"
  exit 0
fi

resume_checkpoint="${RESUME_FROM_CHECKPOINT:-$(latest_resumable_checkpoint "$output_dir")}"
if [[ -n "$resume_checkpoint" ]]; then
  echo "Resuming $stage from $resume_checkpoint"
  stage_args+=(--resume-from-checkpoint "$resume_checkpoint")
fi

main_log="$output_dir/train.log"
resource_log="$output_dir/resource_samples.log"
command=(
  "$python_bin" -m phi_agents.sft.launch --
  "$python_bin" -m torch.distributed.run
  --standalone
  --nproc-per-node 2
  -m phi_agents.sft.trainer
  "${stage_args[@]}"
  "${common_args[@]}"
)

{
  echo "===== $(date --iso-8601=seconds) ====="
  printf 'command='
  printf '%q ' "${command[@]}"
  echo
  echo "stage=$stage cuda_devices=$cuda_devices cpu_offload=$cpu_offload"
  echo "nccl_p2p=$NCCL_P2P_DISABLE nccl_ib=$NCCL_IB_DISABLE " \
    "nccl_cumem=$NCCL_CUMEM_ENABLE nccl_nvls=$NCCL_NVLS_ENABLE"
  free -h
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader
} >> "$main_log" 2>&1

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -n "${monitor_pid:-}" ]]; then
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
  fi
  if [[ -n "${training_pid:-}" ]] && kill -0 "$training_pid" 2>/dev/null; then
    kill -TERM -- "-$training_pid" 2>/dev/null || kill -TERM "$training_pid" 2>/dev/null || true
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

set +e
setsid "${command[@]}" >> "$main_log" 2>&1 &
training_pid=$!
bash scripts/sft/monitor_qwen35_resources.sh "$training_pid" "$resource_log" 15 &
monitor_pid=$!
wait "$training_pid"
training_status=$?
wait "$monitor_pid"
monitor_status=$?
set -e

printf '%s stage_exit stage=%s training_status=%s monitor_status=%s\n' \
  "$(date --iso-8601=seconds)" "$stage" "$training_status" "$monitor_status" >> "$main_log"
if ((training_status != 0)); then
  if grep -Eqi 'CUDA out of memory|OutOfMemoryError' "$main_log"; then
    echo "$stage failed with CUDA OOM; rerun with FSDP_CPU_OFFLOAD=1" >&2
  fi
  exit "$training_status"
fi
if ((monitor_status != 0)); then
  exit "$monitor_status"
fi

if [[ ! -f "$output_dir/final_adapter/adapter_model.safetensors" ]]; then
  echo "Training exited successfully but final adapter is missing: $output_dir" >&2
  exit 10
fi
sha256sum "$output_dir/final_adapter/adapter_model.safetensors" \
  > "$output_dir/final_adapter/adapter_model.sha256"
echo "$stage complete: $output_dir/final_adapter"
