#!/usr/bin/env bash
# Run the single R1 experiment: merged d12_100x1 SFT base -> 15 LOOP iterations.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_dir"

export APPWORLD_ROOT="${APPWORLD_ROOT:-/home/yunlong/dragongong/appworld-data}"
# The verified merged model is fully local.  Older launch wrappers still assert
# that HF_TOKEN exists, so provide a non-secret sentinel only when it is unset.
export HF_TOKEN="${HF_TOKEN:-local-model-no-network}"

CONDA_ENV="${CONDA_ENV:-ml-loop-py312}"
CONDA_BASE="${CONDA_BASE:-$(conda info --base)}"
PYTHON_BIN="${PYTHON_BIN:-$CONDA_BASE/envs/$CONDA_ENV/bin/python}"
export ACCELERATE_BIN="${ACCELERATE_BIN:-$CONDA_BASE/envs/$CONDA_ENV/bin/accelerate}"
[[ -x "$PYTHON_BIN" ]] || { echo "Python not found: $PYTHON_BIN" >&2; exit 2; }
[[ -x "$ACCELERATE_BIN" ]] || { echo "Accelerate not found: $ACCELERATE_BIN" >&2; exit 2; }

EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen25_7b_d12_100x1_loop15_lora16}"
MERGED_BASE_MODEL="${MERGED_BASE_MODEL:-$repo_dir/.model_cache/Qwen/Qwen2.5-7B-Instruct-d12_100_1epoch-merged-bf16}"
MANIFEST_PATH="${MANIFEST_PATH:-$repo_dir/artifacts/sft_loop15/r2_first15_scenario_manifest.json}"
RUN_DIR="${RUN_DIR:-$repo_dir/experiments/$EXPERIMENT_NAME/r1_loop15_seed20260718}"
NUM_SCENARIO_RUNNERS="${NUM_SCENARIO_RUNNERS:-8}"
RL_SEED="${RL_SEED:-20260718}"

MERGED_BASE_MODEL="$(realpath -e "$MERGED_BASE_MODEL")"
MANIFEST_PATH="$(realpath -e "$MANIFEST_PATH")"
RUN_DIR="$(realpath -m "$RUN_DIR")"

"$PYTHON_BIN" scripts/loop7b/extract_r2_first15_manifest.py \
  --validate-manifest "$MANIFEST_PATH"

"$PYTHON_BIN" - "$MERGED_BASE_MODEL" "$MANIFEST_PATH" <<'PY'
import json
import sys
from pathlib import Path

model_dir = Path(sys.argv[1])
scenario_manifest_path = Path(sys.argv[2])
required_model_files = ("config.json", "tokenizer_config.json", "model.safetensors.index.json")
missing = [name for name in required_model_files if not (model_dir / name).is_file()]
if missing:
    raise SystemExit(f"Merged model is incomplete; missing: {missing}")

merge_manifest_path = model_dir / "sft_loop_merge_manifest.json"
if not merge_manifest_path.is_file():
    raise SystemExit(f"Missing merge verification manifest: {merge_manifest_path}")
merge_manifest = json.loads(merge_manifest_path.read_text(encoding="utf-8"))
if merge_manifest.get("status") != "verified" or not merge_manifest.get("verification", {}).get(
    "passed"
):
    raise SystemExit("Merged SFT initialization has not passed its consistency checks")

sft = merge_manifest.get("sft_adapter", {})
loop = merge_manifest.get("loop_adapter_initialization", {})
if (sft.get("r"), sft.get("lora_alpha")) != (32, 64):
    raise SystemExit("Merge manifest does not describe the expected rank-32/alpha-64 SFT adapter")
if (loop.get("r"), loop.get("lora_alpha"), loop.get("lora_dropout")) != (16, 32, 0.0):
    raise SystemExit("Merge manifest does not describe a fresh rank-16/alpha-32 LOOP adapter")
if not loop.get("init_lora_weights"):
    raise SystemExit("Merge manifest does not prove zero/fresh LOOP LoRA initialization")

scenario_manifest = json.loads(scenario_manifest_path.read_text(encoding="utf-8"))
expected = (15, 24, 6, "train_difficulty_1_2")
observed = (
    scenario_manifest.get("num_iterations"),
    scenario_manifest.get("scenarios_per_iteration"),
    scenario_manifest.get("rollouts_per_scenario"),
    scenario_manifest.get("dataset_name"),
)
if observed != expected:
    raise SystemExit(f"Unexpected scenario manifest contract: {observed!r} != {expected!r}")
PY

mkdir -p "$RUN_DIR"

latest_checkpoint="$($PYTHON_BIN - "$RUN_DIR" <<'PY'
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

run_dir = Path(sys.argv[1])
checkpoints: dict[int, Path] = {}
for path in run_dir.glob("checkpoint-*"):
    if path.is_dir() and path.name.removeprefix("checkpoint-").isdigit():
        checkpoints[int(path.name.removeprefix("checkpoint-"))] = path

if any(number > 15 for number in checkpoints):
    raise SystemExit("Run directory contains a checkpoint beyond the 15-iteration contract")

incomplete = [number for number, path in checkpoints.items() if not (path / "done.txt").is_file()]
if incomplete:
    quarantine = (
        run_dir
        / "recovery_quarantine"
        / datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
        / "incomplete_checkpoints"
    )
    quarantine.mkdir(parents=True, exist_ok=False)
    for number in sorted(incomplete):
        source = checkpoints.pop(number)
        destination = quarantine / source.name
        source.replace(destination)
        print(
            f"Quarantined incomplete checkpoint-{number}: {destination}",
            file=sys.stderr,
        )

if checkpoints:
    expected_numbers = set(range(1, max(checkpoints) + 1))
    if set(checkpoints) != expected_numbers:
        raise SystemExit("Run directory does not contain a contiguous checkpoint prefix")

for number, path in checkpoints.items():
    required = (
        path / "lora" / "adapter_config.json",
        path / "lora" / "adapter_model.safetensors",
        path / "trainer_state.pt",
    )
    missing = [str(item) for item in required if not item.is_file()]
    if missing:
        raise SystemExit(f"checkpoint-{number} is incomplete; missing: {missing}")
    state = torch.load(path / "trainer_state.pt", map_location="cpu", weights_only=True)
    if state.get("iterations_completed") != number:
        raise SystemExit(
            f"checkpoint-{number} trainer state says iterations_completed="
            f"{state.get('iterations_completed')!r}"
        )

print(max(checkpoints, default=0))
PY
)"

if (( latest_checkpoint >= 15 )); then
  echo "R1 already complete: checkpoint-15 exists in $RUN_DIR"
  exit 0
fi
start_iteration=$((latest_checkpoint + 1))

# A failure after rollout diagnostics but before checkpoint completion leaves
# immutable per-rollout files for the unfinished iteration.  Preserve them for
# audit and clear the canonical paths before a real retry.
"$PYTHON_BIN" - "$RUN_DIR" "$start_iteration" <<'PY'
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

run_dir = Path(sys.argv[1])
iteration = int(sys.argv[2])
relative_paths = (
    Path("rollouts") / f"iteration-{iteration:06d}",
    Path("rollout_diagnostics") / f"iteration-{iteration:06d}.json",
    Path("training_metrics") / f"iteration-{iteration:06d}.json",
)
existing = [relative for relative in relative_paths if (run_dir / relative).exists()]
if existing:
    quarantine = (
        run_dir
        / "recovery_quarantine"
        / datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
        / f"unfinished_iteration_{iteration:06d}"
    )
    moved = []
    for relative in existing:
        source = run_dir / relative
        destination = quarantine / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.replace(destination)
        moved.append({"source": str(source), "destination": str(destination)})
    manifest = {
        "schema_version": 1,
        "kind": "sft_loop15_recovery_quarantine",
        "iteration": iteration,
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "moved": moved,
    }
    manifest_path = quarantine / "quarantine_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Quarantined unfinished iteration {iteration} artifacts: {quarantine}")
PY

echo "R1 preflight passed"
echo "run_dir=$RUN_DIR"
echo "merged_base_model=$MERGED_BASE_MODEL"
echo "scenario_manifest=$MANIFEST_PATH"
echo "resume_checkpoint=$latest_checkpoint start_iteration=$start_iteration total_iterations=15"
echo "scenario_contract=24x6 iterations=15 total_rollouts=2160"
echo "core_config=minibatch4 epochs2 lr5e-5_constant max_grad_norm0.1 learn_len16000 vllm_len20000 max_new_tokens1200"
echo "num_scenario_runners=$NUM_SCENARIO_RUNNERS rl_seed=$RL_SEED"

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "preflight_only=1; training was not started"
  exit 0
fi

export EXPERIMENT_NAME RUN_DIR NUM_SCENARIO_RUNNERS
export TRAIN_SPLIT="train_difficulty_1_2"
export TOTAL_ITERATIONS=15
export EPOCHS_PER_ITERATION=2
export SCENARIOS_PER_ITERATION=24
export ROLLOUTS_PER_SCENARIO=6
export MINIBATCH_SIZE=4
export ABS_ADV_THRESHOLD=0.01
export TRAIN_MAX_INTERACTIONS=40
export LEARNING_MAX_SEQ_LEN=16000
export VLLM_MAX_MODEL_LEN=20000
export MAX_NEW_TOKENS=1200
export VLLM_GPU_MEMORY_UTILIZATION=0.72
export INFERENCE_REQUIRES_MEMORY_GB=null
export LEARNING_REQUIRES_MEMORY_GB=null
export MAX_CKPTS=15
export STRESS_TEST_ITERS=1
export WANDB_ENABLE="${WANDB_ENABLE:-False}"
# The generic guard's in-process retry reuses the original Hydra overrides,
# including the manifest start iteration.  Recovery for this fixed manifest
# must return through this outer launcher so it can select exactly N+1 from the
# newest complete checkpoint.
export MAX_HARD_DEAD_RESTARTS=0

# User-supplied Hydra arguments are accepted for operational flags, while the
# experiment-defining overrides below remain authoritative.
"$PYTHON_BIN" -m phi_agents.sft.launch -- \
  bash scripts/loop7b/train_stage2_guarded_rtxpro6000.sh \
  "$@" \
  rl/scenario_sampler=appworld_manifest \
  rl.scenario_sampler.manifest_path="$MANIFEST_PATH" \
  rl.scenario_sampler.dataset_name=train_difficulty_1_2 \
  rl.scenario_sampler.start_iteration="$start_iteration" \
  rl.scenario_sampler.cycle=true \
  rl.scenario_sampler.max_parallel=1 \
  rl.cloud_path="$RUN_DIR" \
  rl.rollout_diagnostics.enabled=true \
  rl.rollout_diagnostics.save_trajectories=true \
  rl.seed="$RL_SEED" \
  rl.params.total_iterations=15 \
  rl.params.epochs_per_iteration=2 \
  rl.params.scenarios_per_iteration=24 \
  rl.params.rollouts_per_scenario=6 \
  rl.params.minibatch_size=4 \
  rl.params.loss_type=pg_per_token \
  rl.params.do_ppo_clipping=true \
  rl.params.ppo_epsilon=0.1 \
  rl.params.abs_adv_threshold=0.01 \
  rl.params.baseline=leave_one_out \
  rl.params.adv_normalization=false \
  rl.params.max_grad_norm=0.1 \
  rl.optimization.optimizer.lr=5e-5 \
  rl.optimization.optimizer.weight_decay=0.01 \
  rl.optimization.lr_scheduler._target_=phi_agents.rl.rl_utils.get_const_lr \
  rl.learning_max_seq_len=16000 \
  rl.scenario_runner.appworld_config.env.max_interactions=40 \
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
  rl.num_scenario_runners="$NUM_SCENARIO_RUNNERS" \
  rl.inference_requires_memory_gb=null \
  rl.learning_requires_memory_gb=null \
  rl.max_ckpts=15 \
  rl.stress_test_iters=1
