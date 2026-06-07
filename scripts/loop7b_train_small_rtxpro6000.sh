#!/usr/bin/env bash
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
  experiment_name=qwen25_7b_loop_small128_v1 \
  wandb.enable=False \
  rl.eval.enable=False \
  rl.params.total_iterations=30 \
  rl.params.epochs_per_iteration=1 \
  rl.params.scenarios_per_iteration=8 \
  rl.params.rollouts_per_scenario=4 \
  rl.params.minibatch_size=8 \
  rl.params.loss_type=pg_per_token \
  rl.params.do_ppo_clipping=True \
  rl.params.ppo_epsilon=0.1 \
  rl.params.abs_adv_threshold=0.01 \
  rl.params.baseline=leave_one_out \
  rl.params.adv_normalization=False \
  rl.scenario_sampler.dataset_name=train_small128 \
  rl.scenario_runner.appworld_config.env.max_interactions=8 \
  rl.scenario_runner.appworld_config.env.sparse_reward=False \
  rl.learning_max_seq_len=12000 \
  llm.vllm_server.max_model_len=16384 \
  llm.vllm_class.max_new_tokens=1200 \
  llm.vllm_server.gpus_per_vllm_server=1 \
  rl.num_scenario_runners=4 \
  llm.max_gpu_mem_utilization=0.72 \
  rl.inference_requires_memory_gb=64 \
  rl.learning_requires_memory_gb=56 \
  rl.max_ckpts=10 \
  rl.stress_test_iters=1 \
  "$@"
