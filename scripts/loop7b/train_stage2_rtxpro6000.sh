#!/usr/bin/env bash
# 中文注释：在 RTX PRO 6000 上启动第二阶段 Qwen2.5-7B LOOP LoRA16 训练。
set -euo pipefail

: "${APPWORLD_ROOT:?Set APPWORLD_ROOT before running.}"
: "${HF_TOKEN:?Set HF_TOKEN before running.}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen25_7b_loop_200x24x6_lora16}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train_difficulty_1_2}"
ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PATH="$PWD/appworld-env/bin:$PATH"

mkdir -p logs "$APPWORLD_ROOT/data/datasets"
cp "data/appworld_splits/${TRAIN_SPLIT}.txt" "$APPWORLD_ROOT/data/datasets/${TRAIN_SPLIT}.txt"

echo "experiment_name=${EXPERIMENT_NAME}"
echo "train_split=${TRAIN_SPLIT} repo_count=$(wc -l < "data/appworld_splits/${TRAIN_SPLIT}.txt") appworld_count=$(wc -l < "$APPWORLD_ROOT/data/datasets/${TRAIN_SPLIT}.txt")"
echo "note=trainer saves every iteration; sync/eval watcher defaults to every 10 iterations."

"$ACCELERATE_BIN" launch \
  --config_file ./phi_agents/rl/conf/accelerate_config.yaml \
  --num_processes=1 \
  ./phi_agents/rl/train.py \
  +global@_global_=appworld \
  rl/gpu_allocation=single_gpu \
  llm=qwen_2_5_7b_lora16_train \
  experiment_name="$EXPERIMENT_NAME" \
  wandb.enable="${WANDB_ENABLE:-False}" \
  rl.eval.enable=False \
  rl.params.total_iterations="${TOTAL_ITERATIONS:-200}" \
  rl.params.epochs_per_iteration="${EPOCHS_PER_ITERATION:-2}" \
  rl.params.scenarios_per_iteration="${SCENARIOS_PER_ITERATION:-24}" \
  rl.params.rollouts_per_scenario="${ROLLOUTS_PER_SCENARIO:-6}" \
  rl.params.minibatch_size="${MINIBATCH_SIZE:-8}" \
  rl.params.loss_type=pg_per_token \
  rl.params.do_ppo_clipping=True \
  rl.params.ppo_epsilon=0.1 \
  rl.params.abs_adv_threshold="${ABS_ADV_THRESHOLD:-0.01}" \
  rl.params.baseline=leave_one_out \
  rl.params.adv_normalization=False \
  rl.scenario_sampler.dataset_name="$TRAIN_SPLIT" \
  rl.scenario_runner.appworld_config.env.max_interactions="${TRAIN_MAX_INTERACTIONS:-40}" \
  rl.scenario_runner.appworld_config.env.sparse_reward=False \
  rl.learning_max_seq_len="${LEARNING_MAX_SEQ_LEN:-20000}" \
  llm.vllm_server.max_model_len="${VLLM_MAX_MODEL_LEN:-24576}" \
  llm.vllm_class.max_new_tokens="${MAX_NEW_TOKENS:-1500}" \
  llm.vllm_server.gpus_per_vllm_server=1 \
  llm.max_gpu_mem_utilization="${VLLM_GPU_MEMORY_UTILIZATION:-0.72}" \
  rl.num_scenario_runners="${NUM_SCENARIO_RUNNERS:-4}" \
  rl.inference_requires_memory_gb="${INFERENCE_REQUIRES_MEMORY_GB:-32}" \
  rl.learning_requires_memory_gb="${LEARNING_REQUIRES_MEMORY_GB:-56}" \
  rl.max_ckpts="${MAX_CKPTS:-240}" \
  rl.stress_test_iters="${STRESS_TEST_ITERS:-1}" \
  "$@"
