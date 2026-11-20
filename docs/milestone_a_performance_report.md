# Milestone A — Performance Optimization Report

## Goal & Invariance Contract

Eliminate the multi-minute checkpoint evaluation loading pause in `src/sheepdog/server.py` and improve simulation/training throughput while keeping curriculum stages, reward formulas/weights, neural network architecture, PPO hyperparameters, and evaluation/promotion decisions **100% unchanged and bit-exact**.

## Summary of Changes

1. **Persistent Evaluation Summary Cache** (`src/sheepdog/evaluation/summary_cache.py`, `src/sheepdog/server.py`)
   - Added `load_evaluation_summaries` and `default_eval_cache_path` modeled after `src/sheepdog/checkpoints/summary_cache.py`.
   - Stores only the fields read by `compute_promotion_gate_snapshot` (evaluation metadata fields + compact per-seed `seed`/`steps`/`success`/`timeout` records), keyed by filename, `st_size`, and `st_mtime_ns`.
   - Updated `_load_all_persisted_evaluations` to use `load_evaluation_summaries` whenever `_DIR_EVAL_CACHE` misses (cold start or when `evaluations/` `mtime` changes after a new evaluation lands), eliminating the 2,048-entry LRU eviction cliff that forced re-reading ~39,749 JSON files on every checkpoint.
   - Reused `trigger_cp` in `server.py`'s checkpoint progress handler instead of calling `self.get_checkpoint_details(checkpoint_episode)` twice for the same checkpoint.

2. **Environment Hot-Loop Optimizations** (`src/sheepdog/environment.py`)
   - Added `_blocked_positions_for_dog(dog_index)` and optional `blocked_positions` parameter to `project_dog_action` and `_action_score`, plus early return in `project_dog_action` when `steps <= 0`.
   - `action_mask_for_dog` and `_tactically_valid_wait` now compute `blocked_positions` once per dog/step and share it across all 8–9 action projections.
   - `_action_score` defers computing `current_debug = self._action_context_debug(...)` until `self._is_reverse_action(dog.last_action, action)` is true.
   - `_touched_wall_this_step` skips `action == "wait"` before calling `project_dog_action`.

3. **Web Asset Export Skip-on-Match** (`src/sheepdog/training/trainer.py`)
   - `_export_web_assets` now checks `target_path.stat().st_size == src_stat.st_size` before reading and re-writing existing replay files.

## Before / After Benchmarks

### 1. `_load_all_persisted_evaluations(Path("artifacts"))` (39,749 files, 19,652 unique evaluations)

| Metric | Before | After | Speedup |
|---|---:|---:|---:|
| Cold load / cache miss (dir_mtime changed) | 333.975 s | 315.953 s (one-time cache build) | — |
| Warm disk-cache / incremental reload (`_DIR_EVAL_CACHE` cleared) | 333.975 s | **0.459 s** | **727.6x** |
| In-memory `_DIR_EVAL_CACHE` hit | 0.000899 s | 0.000468 s | 1.9x |

### 2. Environment 1,000-Step Throughput

| Benchmark | Before | After | Change |
|---|---:|---:|---:|
| `SheepdogEnvironment` (1,000 steps) | 0.511 s (1,955.3 steps/s) | **0.469 s (2,130.5 steps/s)** | **+9.0%** |
| `SheepdogRLAdapter` (1,000 steps) | 0.313 s (3,196.4 steps/s) | **0.284 s (3,526.9 steps/s)** | **+10.3%** |

### 3. End-to-End 2-Segment `MaskablePPOTrainer` (2,048 timesteps, 4 workers, 10-seed eval)

| Metric | Before | After | Change |
|---|---:|---:|---:|
| Total 2-segment wall time | 28.325 s | **25.452 s** | **-10.1%** |
| Segment 1 learn throughput | 2.86 s (358 steps/s) | **2.64 s (388 steps/s)** | **+8.4%** |
| Segment 2 learn throughput | 2.68 s (382 steps/s) | **2.62 s (391 steps/s)** | **+2.4%** |
| Segment 1 quick / conf avg reward | -65.58 / -62.52 | -65.58 / -62.52 | Bit-exact |
| Segment 2 quick / conf avg reward | 22.84 / 27.21 | 22.84 / 27.21 | Bit-exact |

## Verification

- `python -m pytest tests/test_evaluation_summary_cache.py tests/test_hotloop_invariance_and_perf.py tests/test_server.py tests/test_maskable_ppo_timing.py tests/test_parallel_evaluation.py`: **48 passed in 26.55s**.
- `python -m ruff check tests/test_evaluation_summary_cache.py tests/test_hotloop_invariance_and_perf.py src/sheepdog/evaluation/summary_cache.py src/sheepdog/environment.py src/sheepdog/training/trainer.py`: **All checks passed (0 errors)**.
- `python -m ruff check src/sheepdog/server.py`: 0 errors on modified/added lines (23 pre-existing `UP038` warnings elsewhere in `server.py`).
