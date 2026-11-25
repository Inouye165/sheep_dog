# Stage timing records

Every curriculum promotion appends one record to `artifacts/stage-timing.jsonl`
(append-only JSON lines; torn trailing lines are tolerated). Code:
`src/sheepdog/training/stage_timing.py`. API: `GET /api/stage-timings`. Viewer:
Progress view → "Stage timings: how long each stage took to train" (visible without opening anything; follows the Progress stage filter).

## Where to find it

Open the **Progress** view and scroll to the section titled *Stage timings: how long each stage took to train*, just below the learning charts. It follows the view's stage filter (current / all / a specific stage). With no data, or only one run, the section says so and names the file where data is stored (`artifacts/stage-timing.jsonl`). It is no longer in the History tab or the Settings telemetry log.

## Definitions (seconds unless noted)

| Field | Meaning |
| --- | --- |
| `wall_seconds` | `ended_at - started_at`; includes pauses, evaluation, checkpointing |
| `training_seconds` | Training-loop time from `training-runtime.json` (excludes pauses/evaluation); `null` for derived records |
| `policy_timesteps` | PPO timesteps used in the stage (difference of `_ts_<n>` in trigger checkpoint ids) |
| `world_steps` | Not recorded by the trainer; `null` (use `policy_timesteps`) |
| `steps_per_second` | `policy_timesteps / training_seconds`, else `/ wall_seconds` (`steps_per_second_basis`) |
| `source` | `recorded` (written at promotion) or `derived` (backfilled from `promotion-history.json`) |
| `timing_confidence` | `medium` start = previous promotion of the same run; `low` start = run-id timestamp (stage 1); `none` unknown |

Each record also stores run label (`SHEEPDOG_RUN_LABEL` env var, default null), git revision + dirty flag,
effective-config hash and compact snapshot, `assistance_mode` (= `observation_mode`), python/torch/SB3 versions and
hostname-free hardware info.

## Comparing runs

The endpoint compares the latest run with the previous run and with the fastest run per stage
(percent delta of wall seconds, training seconds, timesteps, steps/s). Comparisons are flagged
`comparable: false` when the config hash or assistance mode differs (or is unknown, as for derived records);
differences are shown, never hidden. A stage's start is persisted in `artifacts/stage-timing-state.json`, so restarts do
not lose it. Backfill runs lazily on each `GET /api/stage-timings` and when a promotion is logged; it is idempotent
by `promotion_event_id`.

## Limitations

Training seconds depend on the runtime ledger being flushed before promotion; after a rewind the same stage can
complete twice (the second record uses ledger differences). Derived wall times include any pause between promotions.

## Model Pass Lineage (`artifacts/model-history.sqlite`)

Long-lived model lineage and per-stage pass timings are persisted in SQLite at `artifacts/model-history.sqlite`
(`ModelHistoryStore`, `SCHEMA_VERSION = 1` in [src/sheepdog/tracking/model_history.py](../src/sheepdog/tracking/model_history.py)).

- **Tables**:
  - `model_passes`: records `pass_id`, `label`, `run_id`, `parent_pass_id`, `parent_checkpoint_*`, `resulting_checkpoint_*`, `pass_type`, `stage_spec`, `planned_stages_json`, `status`, phase totals, `total_wall_seconds`, `total_policy_timesteps`, `assistance_mode`, `observation_mode`, `git_revision`, and `source`.
  - `stage_runs`: records `stage_run_id`, `pass_id`, `stage`, `sequence_order`, `status`, `start_checkpoint_*`, `end_checkpoint_*`, `policy_timesteps`, `training_seconds`, `evaluation_seconds`, `checkpoint_save_seconds`, `replay_serialization_seconds`, `active_seconds`, `wall_seconds`, `steps_per_second`, `initial_success_rate`, `final_success_rate`, `best_success_rate`, `final_avg_steps`, `final_avg_reward`, and `baseline_runtime_json`.
- **Preservation across resets**: `artifacts/model-history.sqlite` is preserved across `_clear_training_outputs` and `_archive_training_outputs` (Clear / Reset Journey) so pass history and parent-child lineage survive across journeys.
- **API & UI**:
  - `GET /api/model-history` returns recorded passes and stage runs; `GET /api/model-history/lineage` resolves parent/child ancestry and cumulative totals. Both are read-only and work in normal and `--read-only` preview modes.
  - The Progress view's **Speed & Timing** tab (`StageTimingPanel` in [web/src/components/StageTimingPanel.tsx](../web/src/components/StageTimingPanel.tsx)) displays **Model Training Passes & Lineage**, including ancestry chains, cumulative totals, and stage-by-stage comparisons (`Δ vs Previous` / `Δ vs First`) across passes.
- **Stage-queue execution from Training Controls**:
  - `POST /api/training/start` and the Train view (`#train-stage-spec-input` *Stages to run (optional)* and `#train-pass-label-input` *Pass label*) accept arbitrary stage queues (including non-sequential or descending lists such as `10-12, 15-20, 25, 33-38`, parsed by `parse_stage_spec` / `parseStageSpec`).
  - `TrainingManager._run_training` executes the queued stages in order, stops cleanly after the final queued stage, supports mid-queue Pause/Resume, and resets per-stage best-model tracking in `MaskablePPOTrainer.train()` when `best_model_curriculum_stage != active_stage` while still loading weights from `policy_state_path`.
- **Backfill CLI**:
  - [scripts/backfill_model_history.py](../scripts/backfill_model_history.py) imports existing artifacts into `artifacts/model-history.sqlite` without modifying checkpoints or other artifacts (`.\.venv\Scripts\python.exe scripts/backfill_model_history.py --report`).
