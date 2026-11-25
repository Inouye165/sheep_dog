# Stage 34 Strided (`232`-Input) Continuation Test

**Date**: 2026-11-17  
**Experiment Workspace**: `artifacts/experiments/strided_gate_cont_20261117_163800/`  
**Runner**: [scripts/continue_strided_stage34.py](../scripts/continue_strided_stage34.py)  
**Assistance Mode**: **Guided / Assisted** (`observation_mode="guided_gate_control"`, `observation_history_strides=(0, 4, 8, 16)`, `include_loop_flags_in_observation=True`, `TeamStrategy` role/target inputs, and simulation gate-surge physics `sheep_surge_multiplier=1.3`, `sheep_d_gate=12.0`). Not autonomous.

## Goal & Setup

Test whether continuing the two `8/10` `232`-input strided checkpoints from `artifacts/experiments/strided_gate_run_20261117_085600/arm_a_control_zero_gate/` (`seed_101` and `seed_202`) while **preserving their populated Adam optimizer state** (`len(model.policy.optimizer.state) == 16`) under `ENV_CONFIG_VERSION="1.1"` and `50%` failure-directed starts (`boundary_split_flock`, `wall_stray_extraction`, `high_pen_standoff`, `gate_wall_flock`, equal weight `1.0`) breaks the `8/10` ceiling on `DEV_SEEDS` (`11, 23, 37, 41, 53, 59, 61, 67, 71, 73`) and solves Seed 59 and Seed 73.

- **Protocol**: `stage34_strided_gate_continuation_v1`
- **Source Revision**: `bfd12db856c1c2d5b41462758d7eb29d41ba8221` (`git_dirty=True`, `dirty_patch_hash=4adc461659cbc43e7386a1460eb43fe9558a1e7685242c0b04e484d3852584f5`)
- **Budget**: `3 × 65,536 = 196,608` target transitions per seed, `max_seed_seconds=500.0`, `max_total_seconds=1100.0`, deterministic `Evaluator(..., capture_step_telemetry=False)` on `DEV_SEEDS` after each segment.
- **Held-out Seeds**: Unused.
- **Live Artifacts**: Untouched (`artifacts/training-state.json` SHA-256 `2EABBDB9CE980910B44545AD58821F9107193D6DB10444692697D85672AFD80A` verified identical before and after).

## Execution Summary

Total wall-clock elapsed: **`1074.125s`** (within `1100s` total budget). Both seeds completed Segments 1 and 2 (`65,536` and `131,072` transitions) and hit the `500s` per-seed wall-clock limit during Segment 3 (including evaluation time inside the seed wall-clock window), saving and evaluating their Segment 3 milestones cleanly:

- **`seed_101`**: `144,804 / 196,608` transitions (`48,268` world steps, `146` training episodes, `530.64s`, `stop_reason="seed_or_global_time_budget_exhausted"`). Worker sampling counts: `normal=69`, `boundary_split_flock=25`, `high_pen_standoff=24`, `wall_stray_extraction=15`, `gate_wall_flock=14`.
- **`seed_202`**: `144,084 / 196,608` transitions (`48,028` world steps, `182` training episodes, `543.109s`, `stop_reason="seed_or_global_time_budget_exhausted"`). Worker sampling counts: `normal=92`, `high_pen_standoff=24`, `boundary_split_flock=23`, `gate_wall_flock=22`, `wall_stray_extraction=22`.

## Per-Segment & Per-Seed Outcomes (`DEV_SEEDS`, Guided/Assisted)

| Seed | Transitions | Success | Avg Reward | Seed 59 | Seed 73 | Failing Seeds |
| --- | ---: | :---: | ---: | --- | --- | --- |
| `101` | `0` (baseline) | **8/10** | `-347.90` | `0/6` (`no-progress`, 862st) | `0/6` (`no-progress`, 873st) | `59, 73` |
| `101` | `65,536` | `6/10` | `-790.58` | `5/6` (`no-progress`, 1270st) | `0/6` (`no-progress`, 866st) | `37, 53, 59, 73` |
| `101` | `131,072` | `6/10` | `-474.09` | `0/6` (`no-progress`, 904st) | `0/6` (`no-progress`, 869st) | `53, 59, 67, 73` |
| `101` | `144,804` | `7/10` | `-331.28` | `0/6` (`no-progress`, 889st) | `0/6` (`no-progress`, 867st) | `37, 59, 73` |
| `202` | `0` (baseline) | **8/10** | `-359.53` | **`6/6` (`success`, 266st)** | `0/6` (`no-progress`, 1008st) | `53, 73` |
| `202` | `65,536` | **8/10** | `-470.54` | **`6/6` (`success`, 918st)** | `0/6` (`no-progress`, 876st) | `71, 73` |
| `202` | `131,072` | `6/10` | `-562.43` | **`6/6` (`success`, 362st)** | `0/6` (`no-progress`, 873st) | `53, 67, 71, 73` |
| `202` | `144,084` | `5/10` | `-801.15` | `0/6` (`no-progress`, 1032st) | **`6/6` (`success`, 533st)** | `11, 53, 59, 67, 71` |

## Assessment & Limitations

- **Did not break the `8/10` ceiling**: Neither seed exceeded `8/10` on `DEV_SEEDS`. `seed_101` dropped to `6/10`–`7/10` (failing Seed 59 and Seed 73 throughout, plus regressions on Seed 37/53/67), and `seed_202` held `8/10` at `65,536` (briefly solving Seed 53 while losing Seed 71) before dropping to `6/10` and `5/10`.
- **Seed 73 solved once with regressions**: At `144,084` transitions, `seed_202` solved Seed 73 (`6/6`, 533 steps), but simultaneously regressed on Seeds 11, 59, 67, and 71 (`5/10` overall).
- **Incomplete Segment 3**: Both seeds reached ~`144k` of `196,608` transitions before hitting `max_seed_seconds=500`.
- **No promotion**: Preserve the `8/10` baseline checkpoints; do not promote any continuation checkpoint. All results are **guided/assisted** on `DEV_SEEDS` only.

---

## Gentle `arm_b_treatment_gate_info` (`232`-Input Live Gate) Run — 2026-11-17

- **Experiment Workspace**: `artifacts/experiments/strided_gate_gentle_armb_20261117_174300/`
- **Runner**: [scripts/run_strided_gate_validation.py](../scripts/run_strided_gate_validation.py) (`prepare` + `run` with `--arms arm_b_treatment_gate_info --seeds 202,101 --transitions 196608 --eval-interval 65536 --lr 3e-5 --target-kl 0.015 --target-ratio 0.25 --max-arm-seconds 420`)
- **Parent**: Immutable `pv_10483` (`checkpoint 40897`, timestep `343,787,786`, SHA-256 `26aecd34c7d1723feb4ae2f44a03fbddc16c494b2855a6c26216b31703c54dd2`), expanded `54 -> 232` (`observation_history_strides=(0, 4, 8, 16)`, `include_loop_flags_in_observation=True`, `observation_mode="guided_gate_control"`, `ENV_CONFIG_VERSION="1.1"`).
- **Assistance Mode**: **Guided / Assisted** (`TeamStrategy` role/target inputs + simulation gate-surge physics `sheep_surge_multiplier=1.3`, `sheep_d_gate=12.0`). Not autonomous.
- **Held-out Seeds**: Unused (`DEV_SEEDS` only).
- **Live Artifacts**: Untouched (`artifacts/training-state.json` SHA-256 `2EABBDB9CE980910B44545AD58821F9107193D6DB10444692697D85672AFD80A` verified identical before and after).

### Execution Summary

Both seeds preserved Adam state across segments (`optimizer_state_len=16` at every milestone) and hit the `420s` per-seed wall-clock limit (`max_arm_seconds=420`, `complete_training_budget=False`):

- **`seed_101`**: `131,072 / 196,608` transitions (`43,690` world steps, `138` training episodes: `103` normal, `35` targeted — `11` `high_pen_standoff`, `10` `boundary_split_flock`, `8` `wall_stray_extraction`, `6` `gate_wall_flock`), `461.406s` total elapsed.
- **`seed_202`**: `132,395 / 196,608` transitions (`44,131` world steps, `150` training episodes: `108` normal, `42` targeted — `11` `wall_stray_extraction`, `11` `boundary_split_flock`, `11` `gate_wall_flock`, `9` `high_pen_standoff`), `469.359s` total elapsed.

### Per-Segment & Per-Seed Outcomes (`DEV_SEEDS`, Guided/Assisted)

| Seed | Segment | Transitions | World Steps | Elapsed | Success | Avg Reward | Seed 59 | Seed 73 | Failing Seeds |
| --- | :---: | ---: | ---: | ---: | :---: | ---: | --- | --- | --- |
| `101` | `1` | `65,536` | `21,845` | `222.062s` | `6/10` | `-598.46` | `0/6` (`no-progress`, 872st) | `1/6` (`no-progress`, 1139st) | `59, 61, 71, 73` |
| `101` | `2` | `131,072` | `43,690` | `461.375s` | `5/10` | `-751.38` | `0/6` (`no-progress`, 870st) | `0/6` (`no-progress`, 1173st) | `41, 53, 59, 61, 73` |
| `202` | `1` | `65,536` | `21,845` | `202.359s` | `6/10` | `-464.84` | `0/6` (`no-progress`, 883st) | `0/6` (`no-progress`, 929st) | `59, 67, 71, 73` |
| `202` | `2` | `131,072` | `43,690` | `416.734s` | `7/10` | `-885.90` | `0/6` (`no-progress`, 863st) | `0/6` (`no-progress`, 1102st) | `59, 71, 73` |
| `202` | `3` | `132,395` | `44,131` | `469.328s` | `7/10` | `-885.90` | `0/6` (`no-progress`, 863st) | `0/6` (`no-progress`, 1102st) | `59, 71, 73` |

### Assessment & Limitations

- Gentle `arm_b_treatment_gate_info` (`lr=3e-5`, `target_kl=0.015`, `target_ratio=0.25`) peaked at `6/10` (`seed_101`) and `7/10` (`seed_202`) on `DEV_SEEDS` without solving Seed 59 or Seed 73, and both seeds hit `max_arm_seconds=420` near `131k–132k` transitions. No promotion; `pv_10483` remains preserved.
