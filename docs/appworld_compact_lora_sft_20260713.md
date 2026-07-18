# AppWorld compact LoRA SFT experiment (2026-07-13)

This experiment rebuilds supervision-v2 data from the immutable raw teacher trajectories and
trains independent rank-32 Qwen2.5-7B adapters. It does not use dev, test_normal, or
test_challenge trajectories for SFT.

## Data audit

| dataset | accepted | train | validation | train/validation scenarios | clean | recovered | supervised / masked failed actions |
|---|---:|---:|---:|---:|---:|---:|---:|
| difficulty 1/2 | 638 | 509 | 129 | 20 / 4 | 161 | 477 | 8,334 / 120 |
| difficulty 3 | 97 | 81 | 16 | 4 / 1 | 7 | 90 | 2,048 / 70 |

The D1/2 mask count is 119 `execution_failed` actions plus one `no_code` action. The D3 mask
count is 70 `execution_failed` actions. Every remaining action is supervised exactly once after
windowing; masked actions are supervised zero times. Both datasets report zero legacy samples,
zero token truncation, zero train/validation scenario overlap, no dev/test tasks, and no API key.

The fixed seed is `20260713`. The nested D1/2 subsets are:

| subset | rows | D1 / D2 | clean / recovered | scenarios / tasks | effective supervised tokens | train SHA-256 |
|---|---:|---:|---:|---:|---:|---|
| d12_25 | 127 | 80 / 47 | 35 / 92 | 20 / 58 | 100,472 | `47656d8fe967df51f063c3a0cf15e66ae8266b973a4cb4599192edc256b0dd07` |
| d12_50 | 255 | 162 / 93 | 70 / 185 | 20 / 58 | 206,682 | `a1e4416541d031f56414914c05a6d1cb91e371db958de61ebafa0e200b2e2a81` |
| d12_100 | 509 | 327 / 182 | 140 / 369 | 20 / 58 | 417,483 | `d5e64406e73350d0ce75fd52f8c4761125007e054e5efd85a50ac2ca5ce25f9e` |

All three use the identical 129-row validation file, SHA-256
`bb27a7a5a05c1b281f94bb54bd113e238a1bac235b52fc2a2df93e81fafeca38`.
`d12_25 ⊂ d12_50 ⊂ d12_100` is checked on trajectory IDs.

## Rebuild commands

Run all Python commands in the `ml-loop-py312` conda environment.

```bash
conda run -n ml-loop-py312 python -m phi_agents.sft.dataset \
  --input-root artifacts/appworld_sft/teacher_d12_v1/raw \
  --output-dir artifacts/appworld_sft/compact_lora_20260713/data/d12_supervision_v2 \
  --mode difficulty_1_2 --validation-fraction 0.15

conda run -n ml-loop-py312 python -m phi_agents.sft.dataset \
  --input-root artifacts/appworld_sft/teacher_d3_v1/raw \
  --output-dir artifacts/appworld_sft/compact_lora_20260713/data/d3_supervision_v2 \
  --mode difficulty_3 --validation-fraction 0.15

conda run -n ml-loop-py312 python -m phi_agents.sft.subsets \
  --train-jsonl artifacts/appworld_sft/compact_lora_20260713/data/d12_supervision_v2/qwen_sft_train.jsonl \
  --validation-jsonl artifacts/appworld_sft/compact_lora_20260713/data/d12_supervision_v2/qwen_sft_validation.jsonl \
  --output-dir artifacts/appworld_sft/compact_lora_20260713/data/d12_nested \
  --model-name .model_cache/Qwen/Qwen2.5-7B-Instruct \
  --max-length 16384 --seed 20260713
```

## Training commands

Each D1/2 command starts from the same unmodified base model. The wrapper emits a Dragon
Sentinel `CRITICAL` event named `training_failed` for any non-zero child exit. CUDA OOM emits
`cuda_oom`. Logs are written to `/var/log/dragon-sentinel/appworld/appworld.jsonl`.

```bash
conda run -n ml-loop-py312 scripts/sft/run_compact_lora_train.sh d12_25
conda run -n ml-loop-py312 scripts/sft/run_compact_lora_train.sh d12_50
conda run -n ml-loop-py312 scripts/sft/run_compact_lora_train.sh d12_100

# Fresh Base→LoRA one-epoch controls; these reuse the existing supervision-v2 files.
conda run -n ml-loop-py312 scripts/sft/run_compact_lora_train.sh d12_50_1epoch
conda run -n ml-loop-py312 scripts/sft/run_compact_lora_train.sh d12_100_1epoch

# Run only after validation selects BEST_D12. This updates that same adapter without merging or
# stacking another adapter and creates a fresh optimizer/scheduler.
conda run -n ml-loop-py312 scripts/sft/run_compact_lora_train.sh d3 \
  artifacts/appworld_sft/compact_lora_20260713/runs/BEST_D12/epoch-2/lora
```

## CUDA smoke and measured throughput

The real-window CUDA smoke completed one optimizer step with SDPA, BF16, TF32, gradient
checkpointing, rank 32, and no packing. It used about 31.34 GiB peak allocated GPU memory. A
second measurement trained 20 real trajectories (22 windows) for one epoch:

- 220,362 input tokens and 15,906 effective supervised tokens.
- 3 optimizer steps with accumulation 8.
- 61.3203 seconds, 3,593.62 input tokens/second, 0.359 windows/second.
- 49,345,972,736 bytes peak allocated GPU memory (about 45.96 GiB).
- train loss 0.47940 and one-window smoke validation loss 0.51926.

The three full D1/2 jobs completed in 5,953.71 CUDA training seconds (1.65 GPU hours), including
their token-level validation. The measured results were:

| run | windows | optimizer steps | supervised token exposures | train loss | validation loss | seconds | peak GPU memory |
|---|---:|---:|---:|---:|---:|---:|---:|
| d12_25 | 136 | 34 | 200,944 | 0.39666 | 0.32094 | 976.08 | 49,869,134,336 B |
| d12_50 | 271 | 68 | 413,364 | 0.35991 | 0.28897 | 1,711.16 | 49,868,940,800 B |
| d12_100 | 545 | 138 | 834,966 | 0.30586 | 0.27368 | 3,266.48 | 49,878,828,032 B |

The selected D3 continuation used `d12_50/epoch-2/lora`. It completed 60 optimizer steps over
117 windows in 1,046.39 seconds, exposed 189,040 supervised tokens, and reached train loss
0.33234. Epoch 1/2 validation losses were 0.25178 and 0.25229; peak allocated GPU memory was
49,956,072,960 bytes. All four formal jobs used BF16, TF32, gradient checkpointing, SDPA, no
packing, no QLoRA, and zero token truncation.

## Validation commands

The internal validation task lists are scenario-disjoint from SFT train and are installed under
the configured AppWorld data root. They contain 12 D1/2 tasks and 2 D3 tasks represented by the
held-out successful trajectories.

```bash
export APPWORLD_ROOT=/home/yunlong/dragongong/appworld-data
conda run -n ml-loop-py312 scripts/sft/run_compact_lora_eval.sh \
  sft_d12_validation_20260713 base null
conda run -n ml-loop-py312 scripts/sft/run_compact_lora_eval.sh \
  sft_d12_validation_20260713 d12_25 \
  artifacts/appworld_sft/compact_lora_20260713/runs/d12_25/epoch-2
conda run -n ml-loop-py312 scripts/sft/run_compact_lora_eval.sh \
  sft_d12_validation_20260713 d12_50 \
  artifacts/appworld_sft/compact_lora_20260713/runs/d12_50/epoch-2
conda run -n ml-loop-py312 scripts/sft/run_compact_lora_eval.sh \
  sft_d12_validation_20260713 d12_100 \
  artifacts/appworld_sft/compact_lora_20260713/runs/d12_100/epoch-2
conda run -n ml-loop-py312 scripts/sft/run_compact_lora_eval.sh \
  sft_d12_validation_20260713 d12_50_1epoch \
  artifacts/appworld_sft/compact_lora_20260713/runs/d12_50_1epoch/epoch-1
conda run -n ml-loop-py312 scripts/sft/run_compact_lora_eval.sh \
  sft_d12_validation_20260713 d12_100_1epoch \
  artifacts/appworld_sft/compact_lora_20260713/runs/d12_100_1epoch/epoch-1

# Pre-continuation D3 baseline, both D3 checkpoints, and final D1/2 forgetting check.
conda run -n ml-loop-py312 scripts/sft/run_compact_lora_eval.sh \
  sft_d3_validation_20260713 d12_50_before_d3 \
  artifacts/appworld_sft/compact_lora_20260713/runs/d12_50/epoch-2
conda run -n ml-loop-py312 scripts/sft/run_compact_lora_eval.sh \
  sft_d3_validation_20260713 d3_epoch1 \
  artifacts/appworld_sft/compact_lora_20260713/runs/d3_from_d12_50/epoch-1
conda run -n ml-loop-py312 scripts/sft/run_compact_lora_eval.sh \
  sft_d3_validation_20260713 d3_epoch2 \
  artifacts/appworld_sft/compact_lora_20260713/runs/d3_from_d12_50/epoch-2
conda run -n ml-loop-py312 scripts/sft/run_compact_lora_eval.sh \
  sft_d12_validation_20260713 d3_epoch1_on_d12 \
  artifacts/appworld_sft/compact_lora_20260713/runs/d3_from_d12_50/epoch-1
```

No formal test split is used. Best D12 selection is lexicographic: D1/2 strict success, partial
pass, fewer `execution_failed + no_code`, then lower validation loss. D3 epoch 1 and epoch 2 are
evaluated on the D3 validation list; the selected D3 checkpoint is then re-evaluated on D1/2.
Replay is launched only if D3 improves while D1/2 strict success drops by more than two percentage
points.

## Environment evaluation results

All results below are from the scenario-disjoint internal validation task lists, not the formal
AppWorld test splits.

| model | D1/2 TGC | SGC | partial | execution failed | no-code | invalid API | doc calls / rollout | avg turns | recovery | truncation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | 0.0 | 0.0 | 0.4500 | 142 | 2 | 17 | 0.25 | 21.25 | 0.0761 | 0.4167 |
| d12_25 | 8.3 | 0.0 | 0.4542 | 47 | 0 | 0 | 20.25 | 26.83 | 0.0909 | 0.8333 |
| d12_50 | 8.3 | 0.0 | 0.5875 | 91 | 0 | 2 | 85.17 | 19.50 | 0.1231 | 0.5000 |
| d12_100 | 0.0 | 0.0 | 0.4625 | 76 | 0 | 0 | 233.42 | 14.00 | 0.0000 | 0.8333 |

`d12_50` is Best D12: it ties `d12_25` on the first-priority strict metric and wins on
second-priority partial pass. Difficulty-group strict success was D1=33.3/D2=0 for `d12_25`,
D1=0/D2=11.1 for `d12_50`, and zero for Base and `d12_100`. SFT eliminates no-code and nearly
eliminates invalid API output, showing better initially valid API use, but it also over-learns
documentation lookup. More data is not monotonically better: `d12_100` averages 233.42 document
queries per rollout and returns to zero strict success.

## Fresh one-epoch controls (2026-07-14)

These two added runs started independently from the original Qwen base. They did not resume from
an old checkpoint and did not modify the existing `d12_25`, `d12_50`, or `d12_100` artifacts.
Before model loading, the trainer printed and persisted the following schedule audit:

| run | train windows | supervised exposure | optimizer steps | warmup | scheduler total | final LR | train loss | validation loss | seconds | peak GPU memory |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| d12_50_1epoch | 271 | 206,682 | 34 | 2 | 34 | 0.0 | 0.404975 | 0.321946 | 857.57 | 49,844,460,032 B |
| d12_100_1epoch | 545 | 417,483 | 69 | 4 | 69 | 0.0 | 0.355638 | 0.287145 | 1,641.06 | 49,878,828,032 B |

The actual optimizer steps matched the preflight values exactly. Both cosine schedules completed
their full decay within one epoch. The training data hashes remained
`a1e4416541d031f56414914c05a6d1cb91e371db958de61ebafa0e200b2e2a81` and
`d5e64406e73350d0ce75fd52f8c4761125007e054e5efd85a50ac2ca5ce25f9e`;
the common validation hash remained
`bb27a7a5a05c1b281f94bb54bd113e238a1bac235b52fc2a2df93e81fafeca38`.

### A: approximately 200k supervised-token exposure

| run | independent trajectories | epochs | TGC | partial | execution failed | no-code | invalid API | doc queries / rollout | truncation | avg turns |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| d12_25 | 127 | 2 | 8.3 | 0.4542 | 47 | 0 | 0 | 20.25 | 0.8333 | 26.83 |
| d12_50_1epoch | 255 | 1 | 8.3 | 0.4250 | 66 | 0 | 0 | 14.08 | 0.7500 | 28.83 |

At this exposure, doubling independent trajectories did not improve strict success or partial
pass: TGC tied and partial fell by 0.0292. It did reduce documentation lookup and truncation, but
execution failures increased, so it is not an overall win.

### B: approximately 410k supervised-token exposure

| run | independent trajectories | epochs | TGC | partial | execution failed | no-code | invalid API | doc queries / rollout | truncation | avg turns |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| d12_50 | 255 | 2 | 8.3 | 0.5875 | 91 | 0 | 2 | 85.17 | 0.5000 | 19.50 |
| d12_100_1epoch | 509 | 1 | 8.3 | 0.5542 | 19 | 0 | 0 | 8.83 | 0.4167 | 22.92 |

Here strict success again tied and the repeated-data run retained a 0.0333 partial-pass lead.
However, the full-data one-epoch run reduced execution failures by 72, invalid API hits from two
to zero, document queries by about 90%, and truncation from 0.50 to 0.4167. Its validation loss
was also slightly lower (0.287145 versus 0.288975).

The full-data one-epoch model does **not** exhibit the old severe documentation loop: 8.83 document
queries per rollout versus 233.42 for `d12_100` epoch 2 and 418.0 for the checkpoint taken after
epoch 1 of that two-epoch schedule. Since the same 509 trajectories behave normally when the
cosine schedule is defined over one epoch, the old anomaly is not explained by the second half of
the data. It is primarily associated with the two-epoch regimen: the old epoch-1 checkpoint was
only halfway through a 138-step schedule, and epoch 2 added a full repeated exposure. This control
cannot separate those two aspects further, but it rules out the full-data distribution by itself.

Under the originally declared lexicographic selection rule, `d12_50` remains the formal winner
because TGC ties and its partial pass is 0.0333 higher. For a stable LOOP initializer, however,
`d12_100_1epoch` is the more reasonable default recipe: 509 trajectories, one epoch, and a full
69-step cosine schedule give the same strict success with far cleaner execution and exploration.
The validation set has only 12 rollouts, so the small partial-pass difference should not be treated
as statistically decisive without additional seeds; no extra rollout or training was run here.

### Larger dev comparison (2026-07-14)

The two approximately 410k-exposure checkpoints were subsequently evaluated on the repository's
`dev_small64` split with the same inference configuration (temperature 0.1, 50 interactions,
16,384-token context, and 16 runners). Despite its historical name, the checked-in split currently
contains 57 unique tasks: 30 difficulty-1, 24 difficulty-2, and 3 difficulty-3 tasks. Both runs
completed all 57 tasks with zero cancellations; no train, D3 continuation, or test-split job was
started.

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n ml-loop-py312 \
  scripts/sft/run_compact_lora_eval.sh dev_small64 d12_50_x2 \
  artifacts/appworld_sft/compact_lora_20260713/runs/d12_50/epoch-2
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n ml-loop-py312 \
  scripts/sft/run_compact_lora_eval.sh dev_small64 d12_100_x1 \
  artifacts/appworld_sft/compact_lora_20260713/runs/d12_100_1epoch/epoch-1
```

| metric | d12_50 × 2 | d12_100 × 1 |
|---|---:|---:|
| completed / cancelled | 57 / 0 | 57 / 0 |
| TGC | 14.0 | 19.3 |
| SGC | 0.0 | 5.3 |
| D1 TGC | 13.3 | 26.7 |
| D2 TGC | 16.7 | 12.5 |
| D3 TGC | 0.0 | 0.0 |
| average partial pass | 0.39710 | 0.39620 |
| execution failed | 220 | 329 |
| no-code | 1 | 1 |
| invalid API hits | 0 | 8 |
| document queries / rollout | 45.32 | 10.47 |
| document-or-description queries / rollout | 46.60 | 11.53 |
| average turns | 24.72 | 25.46 |
| error recovery success | 0.0407 | 0.1286 |
| context truncation | 0.5614 | 0.6316 |

On this larger sample, `d12_100 × 1` improves strict task completion by 5.3 percentage points
and produces the only scenario-level success. The gain is concentrated in difficulty 1; difficulty
2 falls by 4.2 points and neither checkpoint solves difficulty 3. Partial pass is effectively tied
(a 0.00090 absolute difference), so the strict-success gain is not a broad partial-credit gain.

Most importantly, the one-epoch full-data checkpoint again does **not** show the severe document
query loop: it makes 10.47 document queries per rollout versus 45.32 for `d12_50 × 2`, a 76.9%
reduction. This strengthens the earlier conclusion that, at matched supervised-token exposure,
more independent trajectories with a one-epoch cosine schedule are a better default than repeating
half the data. The trade-off is real: `d12_100 × 1` has 109 more execution failures, eight
invalid-API regex hits, and a 7.0-point higher truncation rate. It should therefore be preferred for
strict success and exploration diversity, while execution robustness remains the next optimization
target.

Machine-readable JSON, CSV, and Markdown for this comparison are under
`artifacts/appworld_sft/compact_lora_20260713/evaluation/dev_small64/one_epoch_comparison/`.

| checkpoint | D3 TGC | D3 partial | execution failed | no-code | invalid API | doc calls / rollout | truncation |
|---|---:|---:|---:|---:|---:|---:|---:|
| Best D12 before D3 | 0.0 | 0.4000 | 0 | 0 | 0 | 26.5 | 1.0 |
| D3 epoch 1 | 0.0 | 0.4000 | 20 | 0 | 0 | 384.5 | 1.0 |
| D3 epoch 2 | 0.0 | 0.4000 | 21 | 0 | 0 | 436.5 | 1.0 |

Epoch 1 is Best D12→D3 because strict and partial tie while it has fewer execution failures and
the lower checkpoint validation loss. D3 did not improve from zero strict or 0.40 partial pass;
it instead greatly increased document queries and execution failures. Re-evaluating epoch 1 on
D1/2 gives TGC 0.0 and partial 0.5542: changes of -8.3 strict percentage points and -0.0333
partial pass from Best D12. The D1/2 drop exceeds two points, but the required D3-improvement
condition is false, so replay was not run.

The remaining failures are dominated by Spotify relationship/pagination tasks on D1/2 and the
held-out file-system D3 scenario. Context truncation and repeated API-document exploration remain
the principal behavioral failure modes; the D3 continuation did not fix them.

## Machine-readable report

```bash
conda run -n ml-loop-py312 python -m scripts.sft.build_compact_experiment_report
```

This writes JSON, CSV, and Markdown under
`artifacts/appworld_sft/compact_lora_20260713/reports/`. It is safe to run repeatedly while the
experiment is in progress.

## Verification

- Full repository test suite: 61 passed.
- Ruff and formatting on the changed SFT/logging files, shell syntax, and `git diff --check`:
  passed. A repository-wide Ruff run still reports pre-existing issues outside this change.
- Sensitive-data scan: zero `DEEPSEEK_API_KEY`, Authorization, or Bearer literal hits.
- Dataset split scan: zero dev/test tasks and zero train/validation scenario overlap.
- Evaluation-config dry-run: local Qwen base, rank-32 LoRA config, and both custom validation
  splits resolve successfully.
- Live logrotate verification: the active training continued writing valid one-line JSON to the
  newly created `appworld.jsonl` after the previous file was rotated to `appworld.jsonl.1`.

## Operational note

Alert-path development checks produced historical `CRITICAL/training_failed` records between
`2026-07-13T14:45:56.769Z` and `15:04:08.084Z`, including an explicit alert-delivery test and
non-zero subprocess test cases. None came from a formal training run. The tests have since been
isolated from production handlers, and no CRITICAL record was emitted during the six formal
training jobs. A whole-host power loss cannot be logged by the terminated host itself and needs an
external heartbeat/dead-man alert if that case must also page.
