# AppWorld DeepSeek teacher → Qwen2.5-7B SFT experiment

## Scope and safety invariants

This pipeline uses `deepseek-v4-flash-thinking` as a teacher and preserves complete, real
AppWorld action/observation turns. The production dataset allowlist is restricted to
`train_difficulty_1_2` (72 tasks). Difficulty 3 is derived from the remaining 18 tasks in the
original `train` split and is stored as `train_difficulty_3`; it is generated and built separately.
`dev`, `test_normal`, and `test_challenge` are rejected by the dataset builder.

The API key is read only from `DEEPSEEK_API_KEY`. It is never accepted as a CLI/config value.
Saved configs are redacted, and output scanning rejects key-like strings. DeepSeek
`reasoning_content` is disabled in new persisted trajectories and is never copied into SFT
messages when converting legacy trajectories. Evaluator passes/failures are reduced to numeric
metadata (`partial_pass`) and never enter model context.

## Outputs

Teacher generation writes an atomic `generation_manifest.json` and one immutable raw trajectory
per task attempt under `raw/<task_id>/attempt-N/trajectory.json`. The manifest supports resume,
records task status, retries, tokens, duration, per-rollout cost, cumulative cost, and repair
lineage.

Dataset construction creates:

- `filtered_success/`: strict, complete, parseable, deduplicated teacher trajectories.
- `failures/`: failed, unsafe, malformed, leaked, repetitive, and duplicate attempts with reasons.
- `quality_decisions.jsonl`: acceptance/rejection and near-duplicate lineage.
- `qwen_sft_train.jsonl` and `qwen_sft_validation.jsonl`: scenario-disjoint Qwen messages.
- `data_manifest.json` and `quality_report.md`: version, counts, costs, and quality statistics.

Each JSONL message has `role`, `content`, and `loss`. Prompt demonstrations, system/user messages,
and AppWorld observations have `loss=false`; only visible assistant actions from the real rollout
have `loss=true`. The final `complete_task` action and its environment result are retained.

## Failure continuation

For an unsuccessful attempt, generation locates the first `execution_failed` or no-code turn.
Because this client does not expose a portable AppWorld snapshot, it initializes the same task in
a fresh deterministic world and replays the exact successful action prefix. It then restores the
visible conversation and asks the teacher for a new suffix. The complete repaired attempt is
executed and strictly evaluated again. No evaluator content is supplied to the teacher. A repaired
trajectory is tagged `recovered_success`; the source failure remains in `raw` and later in
`failures` for preference/error-analysis use.

## Commands

All commands use the required `ml-loop-py312` conda environment.

Set runtime-only secrets and AppWorld location:

```bash
export APPWORLD_ROOT=/path/to/appworld-data
export DEEPSEEK_API_KEY=...
```

One-task teacher smoke with a hard cost ceiling:

```bash
conda run -n ml-loop-py312 python -m phi_agents.sft.teacher \
  --split train_difficulty_1_2 \
  --limit 1 \
  --successes-per-task 1 \
  --max-attempts-per-task 2 \
  --cost-limit-cny 0.20 \
  --output-dir artifacts/appworld_sft/smoke_teacher_d12
```

Main difficulty 1/2 generation (1,008 target successes, balanced at 14 per task):

```bash
conda run -n ml-loop-py312 python -m phi_agents.sft.teacher \
  --split train_difficulty_1_2 \
  --successes-per-task 14 \
  --max-attempts-per-task 28 \
  --cost-limit-cny 80 \
  --output-dir artifacts/appworld_sft/teacher_d12_v1
```

Separate difficulty 3 supplement (never mixed into the main dataset by default):

```bash
conda run -n ml-loop-py312 python -m phi_agents.sft.teacher \
  --split train_difficulty_3 \
  --successes-per-task 14 \
  --max-attempts-per-task 28 \
  --cost-limit-cny 20 \
  --output-dir artifacts/appworld_sft/teacher_d3_v1
```

Build the fair difficulty 1/2 dataset:

```bash
conda run -n ml-loop-py312 python -m phi_agents.sft.dataset \
  --mode difficulty_1_2 \
  --input-root artifacts/appworld_sft/teacher_d12_v1 \
  --output-dir artifacts/appworld_sft/dataset_d12_v1
```

Build the explicit difficulty 1/2/3 ablation:

```bash
conda run -n ml-loop-py312 python -m phi_agents.sft.dataset \
  --mode difficulty_1_2_3 \
  --input-root artifacts/appworld_sft/teacher_d12_v1 \
  --input-root artifacts/appworld_sft/teacher_d3_v1 \
  --output-dir artifacts/appworld_sft/dataset_d123_ablation_v1
```

Tokenization-only smoke, followed by LoRA SFT. A complete prompt+turn that exceeds `max-length`
causes a hard error; no tail is silently truncated. Longer trajectories are split only at complete
assistant-action/observation boundaries.

```bash
conda run -n ml-loop-py312 python -m phi_agents.sft.trainer \
  --train-jsonl artifacts/appworld_sft/dataset_d12_v1/qwen_sft_train.jsonl \
  --validation-jsonl artifacts/appworld_sft/dataset_d12_v1/qwen_sft_validation.jsonl \
  --output-dir experiments/qwen25_7b_appworld_sft_d12_v1 \
  --max-length 16384 \
  --tokenize-only

conda run -n ml-loop-py312 torchrun --nproc_per_node=4 -m phi_agents.sft.trainer \
  --train-jsonl artifacts/appworld_sft/dataset_d12_v1/qwen_sft_train.jsonl \
  --validation-jsonl artifacts/appworld_sft/dataset_d12_v1/qwen_sft_validation.jsonl \
  --output-dir experiments/qwen25_7b_appworld_sft_d12_v1 \
  --max-length 16384 \
  --gradient-accumulation-steps 16
```

Resume passes the Transformers checkpoint directory. It restores LoRA weights, optimizer,
scheduler, RNG, and trainer state:

```bash
conda run -n ml-loop-py312 torchrun --nproc_per_node=4 -m phi_agents.sft.trainer \
  --train-jsonl artifacts/appworld_sft/dataset_d12_v1/qwen_sft_train.jsonl \
  --validation-jsonl artifacts/appworld_sft/dataset_d12_v1/qwen_sft_validation.jsonl \
  --output-dir experiments/qwen25_7b_appworld_sft_d12_v1 \
  --resume-from-checkpoint experiments/qwen25_7b_appworld_sft_d12_v1/checkpoint-100
```

Evaluate the base Qwen, original LOOP checkpoint-130, and the new rank-64 SFT adapter. The wrapper
refuses any split other than `dev` or `dev_small64`:

```bash
bash scripts/sft/eval_comparison.sh \
  dev_small64 \
  experiments/qwen25_7b_appworld_sft_d12_v1/final_adapter \
  artifacts/appworld_sft/evaluation/dev_small64
```

It emits official TGC/SGC and difficulty groups plus partial pass, execution/no-code/invalid API,
API-doc use, average turns, error recovery, context-limit rate, and remaining failure types. The
combined outputs are `comparison.json` and `comparison.md`.

## Defaults and compatibility

SFT uses `Qwen/Qwen2.5-7B-Instruct`, LoRA rank 64, alpha 128, dropout 0, all Qwen attention/MLP
projection targets, assistant-only loss, bf16, gradient checkpointing, and gradient accumulation.
Every checkpoint contains `sft_artifact.json` with the base model, exact data SHA-256, data/training
configuration, token/split/truncation statistics, and a `loop_compatible` marker. The
`final_adapter` directory can be passed directly as the AppWorld/LOOP adapter path.

## Smoke status and cost estimate

Local tests cover the train-only allowlist, balanced plan, failure-prefix repair, strict filtering,
scenario-disjoint split, reasoning/evaluator exclusion, Qwen-style assistant-only masks, and
turn-boundary windowing. They pass without API calls. Planning estimates are based on the observed
DeepSeek dev cost (~0.025 CNY per rollout) with retry allowance:

- difficulty 1/2, 1,008 successes: approximately 25.2–80.6 CNY.
- separate difficulty 3, 252 successes: approximately 6.3–20.2 CNY.
- one-task smoke: approximately 0.03–0.08 CNY, capped at 0.20 CNY by the command above.

The current development shell did not provide `DEEPSEEK_API_KEY` or `APPWORLD_ROOT`, so a real API
teacher smoke and GPU training/evaluation were intentionally not launched. This avoids creating
fake train trajectories and complies with the requirement not to start bulk generation. An
additional cached-Qwen tokenizer/GPU smoke was attempted, but this host currently blocks while
retaining the NVIDIA UVM device (`uvm_gpu_retain_by_uuid`); the pure token-mask/window tests pass,
but the real tokenizer/GPU smoke should be rerun after the host driver state is healthy.

## Remaining operational risks

- Replayed repair prefixes assume AppWorld task initialization is deterministic. Every repaired
  result is still re-evaluated strictly, so nondeterministic replay becomes a rejected failure.
- Rank-64 vLLM serving requires the supplied `qwen_2_5_7b_lora64_eval` config; the older rank-16
  eval config cannot load this adapter.
- Actual throughput and the best max sequence length depend on the target GPU memory. Reduce batch
  size or increase accumulation, but do not lower max length until tokenize-only statistics show
  that complete turns still fit.
- Training quality depends on teacher diversity after per-task near-duplicate removal. Review
  `quality_report.md` before launching SFT and increase failed-task resampling selectively when
  coverage is imbalanced.
