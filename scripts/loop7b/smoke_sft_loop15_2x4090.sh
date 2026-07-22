#!/usr/bin/env bash
# Exercise two vLLM servers and a 16k two-rank FSDP backward from checkpoint-1.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_dir"

export APPWORLD_ROOT="${APPWORLD_ROOT:-/root/appworld-root}"
export CONDA_BASE="${CONDA_BASE:-/root/miniconda3}"
export CONDA_ENV="${CONDA_ENV:-appworld}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export GPU_ALLOCATION=two_gpu_shared
export NUM_LEARNING_PROCESSES=2
export ACCELERATE_CONFIG_FILE="${ACCELERATE_CONFIG_FILE:-./phi_agents/rl/conf/accelerate_config_2x4090.yaml}"
export NUM_SCENARIO_RUNNERS=1
export TMPDIR="${TMPDIR:-/root/appworld-tmp}"
export PATH="$repo_dir/.codex_tmp/appworld-bin:$PATH"
export HF_TOKEN="${HF_TOKEN:-local-model-no-network}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_CUMEM_ENABLE=0
export FSDP_OFFLOAD_PIN_MEMORY="${FSDP_OFFLOAD_PIN_MEMORY:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python_bin="$CONDA_BASE/envs/$CONDA_ENV/bin/python"
export ACCELERATE_BIN="$CONDA_BASE/envs/$CONDA_ENV/bin/accelerate"
wsl_model_cache="/root/model-cache/Qwen/Qwen2.5-7B-Instruct-d12_100_1epoch-merged-bf16"
if [[ -d "$wsl_model_cache" ]]; then
  merged_base_model="${MERGED_BASE_MODEL:-$wsl_model_cache}"
else
  merged_base_model="${MERGED_BASE_MODEL:-$repo_dir/.model_cache/Qwen/Qwen2.5-7B-Instruct-d12_100_1epoch-merged-bf16}"
fi
manifest_path="${MANIFEST_PATH:-$repo_dir/artifacts/sft_loop15/r2_first15_scenario_manifest.json}"
export RUN_DIR="${RUN_DIR:-$repo_dir/.codex_tmp/sft_loop15_2x4090_smoke}"

[[ -f "$RUN_DIR/checkpoint-1/done.txt" ]] || {
  echo "Smoke run requires a copied checkpoint-1 in $RUN_DIR" >&2
  exit 2
}
if [[ -f "$RUN_DIR/checkpoint-2/done.txt" ]]; then
  echo "Two-GPU CUDA smoke already passed: $RUN_DIR/checkpoint-2"
  exit 0
fi

export EXPERIMENT_NAME=qwen25_7b_d12_100x1_loop15_2x4090_smoke
export TRAIN_SPLIT=train_difficulty_1_2
export TOTAL_ITERATIONS=2
export EPOCHS_PER_ITERATION=1
export SCENARIOS_PER_ITERATION=2
export ROLLOUTS_PER_SCENARIO=2
export MINIBATCH_SIZE=4
export ABS_ADV_THRESHOLD=0.0
export TRAIN_MAX_INTERACTIONS=1
export LEARNING_MAX_SEQ_LEN=16000
export VLLM_MAX_MODEL_LEN=20000
export MAX_NEW_TOKENS=64
export VLLM_GPU_MEMORY_UTILIZATION=0.72
export INFERENCE_REQUIRES_MEMORY_GB=null
export LEARNING_REQUIRES_MEMORY_GB=null
export MAX_CKPTS=2
export STRESS_TEST_ITERS=1
export WANDB_ENABLE=False
export MAX_HARD_DEAD_RESTARTS=0

mkdir -p "$TMPDIR"

"$python_bin" -m phi_agents.sft.launch -- \
  bash scripts/loop7b/train_stage2_guarded_rtxpro6000.sh \
  rl/scenario_sampler=appworld_manifest \
  rl.scenario_sampler.manifest_path="$manifest_path" \
  rl.scenario_sampler.dataset_name=train_difficulty_1_2 \
  rl.scenario_sampler.start_iteration=2 \
  rl.scenario_sampler.cycle=true \
  rl.scenario_sampler.max_parallel=1 \
  rl.scenario_sampler.distributed_shard=true \
  rl.cloud_path="$RUN_DIR" \
  rl.rollout_diagnostics.enabled=false \
  rl.seed=20260718 \
  rl.params.total_iterations=2 \
  rl.params.epochs_per_iteration=1 \
  rl.params.scenarios_per_iteration=2 \
  rl.params.rollouts_per_scenario=2 \
  rl.params.minibatch_size=4 \
  rl.params.loss_type=pg_per_token \
  rl.params.do_ppo_clipping=true \
  rl.params.ppo_epsilon=0.1 \
  rl.params.abs_adv_threshold=0.0 \
  rl.params.baseline=leave_one_out \
  rl.params.adv_normalization=false \
  rl.params.max_grad_norm=0.1 \
  rl.optimization.optimizer.lr=5e-5 \
  rl.optimization.optimizer.weight_decay=0.01 \
  rl.optimization.lr_scheduler._target_=phi_agents.rl.rl_utils.get_const_lr \
  rl.learning_max_seq_len=16000 \
  rl.scenario_runner.appworld_config.env.max_interactions=1 \
  rl.scenario_runner.appworld_config.env.sparse_reward=false \
  llm.base_model_path="$merged_base_model" \
  llm.adapter_path=null \
  llm.lora_rank=16 \
  llm.lora_alpha=32 \
  llm.lora_dropout=0.0 \
  llm.temperature=1.0 \
  llm.vllm_server.max_model_len=20000 \
  llm.vllm_server.max_tries=900 \
  llm.vllm_class.max_new_tokens=64 \
  llm.vllm_server.gpus_per_vllm_server=1 \
  llm.max_gpu_mem_utilization=0.72 \
  rl.num_scenario_runners=1 \
  rl.inference_requires_memory_gb=null \
  rl.learning_requires_memory_gb=null \
  rl.max_ckpts=2 \
  rl.stress_test_iters=1 \
  rl.stress_test_on_resume=true
