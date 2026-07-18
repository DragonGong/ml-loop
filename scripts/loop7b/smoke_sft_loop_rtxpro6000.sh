#!/usr/bin/env bash
# One real AppWorld rollout/learner smoke for the merged SFT -> LOOP initialization.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_dir"

export APPWORLD_ROOT="${APPWORLD_ROOT:-/home/yunlong/dragongong/appworld-data}"
export HF_TOKEN="${HF_TOKEN:-local-model-no-network}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CONDA_ENV="${CONDA_ENV:-ml-loop-py312}"
CONDA_BASE="${CONDA_BASE:-$(conda info --base)}"
PYTHON_BIN="${PYTHON_BIN:-$CONDA_BASE/envs/$CONDA_ENV/bin/python}"
export ACCELERATE_BIN="${ACCELERATE_BIN:-$CONDA_BASE/envs/$CONDA_ENV/bin/accelerate}"

MERGED_BASE_MODEL="${MERGED_BASE_MODEL:-$repo_dir/.model_cache/Qwen/Qwen2.5-7B-Instruct-d12_100_1epoch-merged-bf16}"
MANIFEST_PATH="${MANIFEST_PATH:-$repo_dir/artifacts/sft_loop15/r2_first15_scenario_manifest.json}"
RUN_DIR="${RUN_DIR:-$repo_dir/experiments/qwen25_7b_d12_100x1_loop15_lora16/cuda_smoke_real_20260718}"
MERGED_BASE_MODEL="$(realpath -e "$MERGED_BASE_MODEL")"
MANIFEST_PATH="$(realpath -e "$MANIFEST_PATH")"
RUN_DIR="$(realpath -m "$RUN_DIR")"

"$PYTHON_BIN" scripts/loop7b/extract_r2_first15_manifest.py \
  --validate-manifest "$MANIFEST_PATH"
"$PYTHON_BIN" - "$MERGED_BASE_MODEL" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(
    (Path(sys.argv[1]) / "sft_loop_merge_manifest.json").read_text(encoding="utf-8")
)
if manifest.get("status") != "verified" or not manifest["verification"]["passed"]:
    raise SystemExit("The merged SFT initialization is not verified")
PY

if [[ -f "$RUN_DIR/checkpoint-1/done.txt" ]]; then
  echo "CUDA smoke already passed: $RUN_DIR/checkpoint-1"
  exit 0
fi

export EXPERIMENT_NAME="qwen25_7b_d12_100x1_loop_cuda_smoke"
export RUN_DIR
export TRAIN_SPLIT=train_difficulty_1_2
export TOTAL_ITERATIONS=1
export EPOCHS_PER_ITERATION=2
export SCENARIOS_PER_ITERATION=1
export ROLLOUTS_PER_SCENARIO=2
export MINIBATCH_SIZE=1
export ABS_ADV_THRESHOLD=0.0
export TRAIN_MAX_INTERACTIONS=6
export LEARNING_MAX_SEQ_LEN=16000
export VLLM_MAX_MODEL_LEN=20000
export MAX_NEW_TOKENS=1200
export VLLM_GPU_MEMORY_UTILIZATION=0.72
export INFERENCE_REQUIRES_MEMORY_GB=null
export LEARNING_REQUIRES_MEMORY_GB=null
export NUM_SCENARIO_RUNNERS=2
export MAX_CKPTS=1
export STRESS_TEST_ITERS=1
export WANDB_ENABLE=False
export MAX_HARD_DEAD_RESTARTS=0

echo "CUDA smoke contract: 1 real scenario x 2 rollouts, 2 learner epochs"
echo "run_dir=$RUN_DIR"

"$PYTHON_BIN" -m phi_agents.sft.launch -- \
  bash scripts/loop7b/train_stage2_guarded_rtxpro6000.sh \
  rl/scenario_sampler=appworld_manifest \
  rl.scenario_sampler.manifest_path="$MANIFEST_PATH" \
  rl.scenario_sampler.dataset_name=train_difficulty_1_2 \
  rl.scenario_sampler.start_iteration=1 \
  rl.scenario_sampler.cycle=true \
  rl.scenario_sampler.max_parallel=1 \
  rl.cloud_path="$RUN_DIR" \
  rl.rollout_diagnostics.enabled=true \
  rl.rollout_diagnostics.save_trajectories=true \
  rl.seed=20260718 \
  rl.params.total_iterations=1 \
  rl.params.epochs_per_iteration=2 \
  rl.params.scenarios_per_iteration=1 \
  rl.params.rollouts_per_scenario=2 \
  rl.params.minibatch_size=1 \
  rl.params.abs_adv_threshold=0.0 \
  rl.params.baseline=leave_one_out \
  rl.params.adv_normalization=false \
  rl.params.max_grad_norm=0.1 \
  rl.optimization.optimizer.lr=5e-5 \
  rl.optimization.optimizer.weight_decay=0.01 \
  rl.optimization.lr_scheduler._target_=phi_agents.rl.rl_utils.get_const_lr \
  rl.learning_max_seq_len=16000 \
  rl.scenario_runner.appworld_config.env.max_interactions=6 \
  rl.scenario_runner.appworld_config.env.sparse_reward=false \
  llm.base_model_path="$MERGED_BASE_MODEL" \
  llm.adapter_path=null \
  llm.lora_rank=16 \
  llm.lora_alpha=32 \
  llm.lora_dropout=0.0 \
  llm.temperature=1.0 \
  llm.vllm_server.max_model_len=20000 \
  llm.vllm_class.max_new_tokens=1200 \
  llm.vllm_server.gpus_per_vllm_server=1 \
  llm.max_gpu_mem_utilization=0.72 \
  rl.num_scenario_runners=2 \
  rl.inference_requires_memory_gb=null \
  rl.learning_requires_memory_gb=null \
  rl.max_ckpts=1 \
  rl.stress_test_iters=1
