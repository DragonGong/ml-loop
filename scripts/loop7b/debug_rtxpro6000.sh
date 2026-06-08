#!/usr/bin/env bash
# 中文注释：在 RTX PRO 6000 上运行 7B LOOP debug 训练。
set -euo pipefail

: "${APPWORLD_ROOT:?Set APPWORLD_ROOT before running.}"
: "${HF_TOKEN:?Set HF_TOKEN before running.}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PATH="$PWD/appworld-env/bin:$PATH"

mkdir -p logs

accelerate launch \
  --config_file ./phi_agents/rl/conf/accelerate_config.yaml \
  --num_processes=1 \
  ./phi_agents/rl/train.py \
  +global@_global_=appworld \
  rl/gpu_allocation=single_gpu \
  llm=qwen_2_5_7b_train \
  experiment_name=debug_qwen25_7b_loop \
  wandb.enable=False \
  rl.eval.enable=False \
  rl.params.total_iterations=2 \
  rl.scenario_runner.appworld_config.env.max_interactions=5 \
  rl.num_scenario_runners=1 \
  rl.params.scenarios_per_iteration=2 \
  rl.params.minibatch_size=2 \
  rl.params.rollouts_per_scenario=3 \
  rl.scenario_sampler.dataset_name=train_small128 \
  rl.learning_max_seq_len=12000 \
  llm.vllm_server.max_model_len=16384 \
  llm.vllm_class.max_new_tokens=1200 \
  llm.vllm_server.gpus_per_vllm_server=1 \
  llm.max_gpu_mem_utilization=0.72 \
  rl.inference_requires_memory_gb=64 \
  rl.learning_requires_memory_gb=56 \
  rl.stress_test_iters=1 \
  "$@"
