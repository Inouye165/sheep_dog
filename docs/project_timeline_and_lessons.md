# Sheepdog Herding Lab: Project Timeline, Improvements, and Lessons Learned

**Coverage**: April 21, 2026 – November 24, 2026  
**Benchmark Goal**: Three autonomous dogs learning to herd all six sheep into a pen—including recovery from oscillation, blocked approaches, walls, corners, and split flocks.

> **Provenance & Scope Note**  
> This document distinguishes between:
> - **Committed Git History**: `89` commits on `HEAD` from `2026-04-21` (`80aff42`) through `2026-11-20` (`5956385`).
> - **Working-Tree / Uncommitted Milestones**: Work from `2026-11-20` through `2026-11-24` recorded in [handoff.md](../handoff.md) and sibling reports in `docs/`.
> - **Guided/Assisted vs. Autonomous Policies**: Guided modes (`guided`, `guided_gate_control`) supply `TeamStrategy` role/target vectors or loop flags as training aids and baselines, whereas the autonomous target path (`emergent_gate`, `208` strided inputs, `include_loop_flags_in_observation: false`, `enable_instinct_rewards: false`) learns without `TeamStrategy` or loop-flag assistance.
> - **Development vs. Held-Out Evaluations**: `DEV_SEEDS` (e.g., 10-seed dev sets such as `11..73`) are used for iteration and monitoring, while held-out sets (`holdout10` such as `101..149`, and `Test-30` with seeds `> 1000`) are reserved for generalization checks.

---

## 1. Executive Summary & Current Milestone Snapshot

Between April and November 2026, Sheepdog Herding Lab evolved from a basic grid-world prototype with heuristic hill-climbing into a full-stack reinforcement-learning research and teaching studio powered by `MaskablePPO`, deterministic replay forensics, spatial bottleneck analytics, and an interactive React/TypeScript web viewer.

### Current Training State (Stage 38 Snapshot)

As recorded in [artifacts/training-state.json](../artifacts/training-state.json) (`run_id`: `run_stage9_from_stage8_20261119_063250_40a1`, `policy_version`: `3490`), the active autonomous training lineage has reached **Curriculum Stage 38**:

| Metric / Field | Observed Value |
| :--- | :--- |
| **Observation Mode & Size** | `emergent_gate` (`208` inputs: `52` base features $\times$ `4` strided frames `[0, 4, 8, 16]`, `include_loop_flags_in_observation: false`) |
| **Observation Schema** | `emergent_gate_v1\|strides=0,4,8,16\|no_loop_flags` |
| **Policy Architecture** | `MaskablePPO`, 3-layer MLP `[256, 256, 128]`, `9` discrete actions, `16` environment workers |
| **Instinct Rewards** | Disabled (`enable_instinct_rewards: false`) |
| **Total Timesteps** | `211,933,846` |
| **Total Environment Episodes** | `260,913` (`11,734` `total_episodes_trained`) |
| **Best Stage Reached** | Stage `38` (`best_model_curriculum_stage: 38`) |
| **Best Dev Eval Success Rate** | `1.0` (`100%`), `best_average_reward: 272.93`, `best_completion_steps: 234.5` |
| **Adaptive Step State** | `Stage 4 of 4 (0.50x • Holding Gate)` (`effective_learning_rate: 5e-05`, `effective_entropy_coef: 0.005`, `ema_success_rate: 0.9935`) |

*Caveat*: `best_success_rate: 1.0` in [training-state.json](../artifacts/training-state.json) reflects the recorded development evaluation checkpoint (`artifacts/checkpoints/stage38/stage38-best-model.zip`). On `2026-11-24`, before resuming Stage 37/38 training under `ENV_CONFIG_VERSION = "1.2"` and stage-aware PPO overrides, an un-retrained Stage 37 checkpoint (`maskable-ppo-stage37-201906838.zip`) evaluated across 40 deterministic episodes scored **90% (9/10)** on Stage 37 `dev10`, **80% (8/10)** on Stage 37 `holdout10`, **100% (10/10)** on Stage 38 `dev10`, and **80% (8/10)** on Stage 38 `holdout10` ([handoff.md](../handoff.md#L1318-L1323)).

---

## 2. Month-by-Month Timeline (April – November 2026)

### April 2026 — Foundation, Role-Aware Baselines, and MaskablePPO (Committed: `80aff42` – `be92789`)
- **Apr 21 (`80aff42` – `a83440a`)**: Initialized the repository, built the core Sheepdog Herding Lab environment, added herding-instinct reward shaping and early curriculum stages (`65db18b`), and separated instinct-only training from target-strategy guidance (`a83440a`).
- **Apr 22 (`75386c1` – `f9e9d97`)**: Fixed field rendering and cell occupancy (`75386c1`), introduced dynamic dog roles (`f358552`), and hardened the role-aware hill-climbing baseline (`f9e9d97`).
- **Apr 23 (`8fc1c2c` – `be92789`)**: Introduced the `MaskablePPO` neural policy (`8fc1c2c`), replay end controls (`5452dc1`), sprint herding (`80af0d1`), live PPO training status (`60e50da`), and disengaged-dog penalties (`be92789`).

### May 2026 — Sheep Personalities, Masking Fixes, and UI Split (Committed: `c811a6f` – `e712567`)
- **May 1 – May 16 (`c811a6f` – `57b467b`)**: Polished PPO resume and sprint herding (`c811a6f`), and fixed Stage 1 `pen_opening=bottom` so heuristic herding works from below (`57b467b`).
- **May 24 (`d67d2b3`)**: Fixed neural wait masking and training diagnostics (`d67d2b3`).
- **May 26 – May 28 (`029f014` – `e712567`)**: Added per-sheep personalities with configurable strength (`d8dbb3d`), split the web Config panel into Dogs and Sheep sections (`033181a`), replaced `pen_curious` with `pen_fearful` (`dcc853d`), and standardized docstrings and linting (`029f014`, `e712567`).

### June 2026 — Scenario Evaluation, Emergent Mode, and Architecture/Stages UI (Committed: `bcc7778` – `1269235`)
- **Jun 1 (`bcc7778` – `7ef3b84`)**: Added an interactive Testing tab for scenario evaluation and personality testing (`eb99c82`, merged in `7ef3b84`).
- **Jun 2 (`b4d2d6c` – `f743c6c`)**: Added emergent PPO mode and stray-recovery rewards (`db8215f`, PR #14 `e9bcf73`) and resolved sheep idle and vision-detection bugs (`5505bb9`, PR #15 `f743c6c`).
- **Jun 10 – Jun 12 (`7f92036` – `2259f13`)**: Refactored the Results tab milestone layout (`7f92036`), expanded training/curriculum controls (`a4f1dc6`), and improved stray-recovery curriculum defaults (`2259f13`).
- **Jun 26 (`1269235`)**: Added Model Architecture and Training Stages tabs alongside a Master Notes Log (`1269235`).

### July 2026 — Lifecycle Hardening, Auto-Promotion, and Live Telemetry (Committed: `87bc656` – `bd0414a`)
- **Jul 1 – Jul 10 (`87bc656` – `918a808`)**: Added mid-batch auto-promotion, stage rewinding, atomic I/O, and app lifecycle scripts (`87bc656`), graceful training control and session auto-resume (`3694e19`), network architecture docs ([sheep_dog_rl_network_guide.md](sheep_dog_rl_network_guide.md), `0c59cd2`), startup replay-export optimization (`4c4c340`), redesigned Neural Network Layers and Training sub-tabs (`a317c46`, `9ba3c42`), and fixed socket leaks and resume crashes (`918a808`).
- **Jul 13 – Jul 14 (`eefda6e` – `939856d`)**: Added stage selection summaries (`eefda6e`), displayed total vs. stage episodes (`0ea4587`), and implemented a per-seed consistency promotion gate (`939856d`).
- **Jul 27 – Jul 31 (`321bc49` – `bd0414a`)**: Hardened concurrent startup (`321bc49`), implemented rolling curriculum promotion gates (`5fd9382`), and introduced the telemetry manager (`bd0414a`).

### August 2026 — Wall Recovery, Spatial Bottlenecks, and Adaptive Learning (Committed: `2233186` – `67a6ac7`)
- **Aug 3 – Aug 7 (`2233186` – `ddda066`)**: Implemented durable live-training episode telemetry (`2233186`, `d91d436`), added per-sheep wall-stall metrics (`4238ada`), inward sheep boundary steering (`0cc5032`), safe open-field role-target positioning (`689e7e6`), Stage 9 wall-recovery remediation (`d777a3f`, `788f638`, `cc6a5ad`), standardized 10-seed evaluation (`c25bd87`), and aligned telemetry X-axes with checkpoints (`ddda066`).
- **Aug 13 – Aug 18 (`b5c70fe` – `c8cc6c2`)**: Consolidated the performance chart pipeline (`b5c70fe`), added stage-completion hover cards (`6eac9bd`), and shipped the formal 10-seed auto-promote readiness engine and hourly backups (`0a77747`, `c8cc6c2`).
- **Aug 21 (`10298b4`)**: Added the 4-stage adaptive learning-rate and step-size controller (`10298b4`) and diagnosed/remediated the Stage 7 lone-straggler plateau to reach 10/10 at PV 1398 ([model_training_history.md](model_training_history.md)).
- **Aug 26 (`67a6ac7`)**: Added the Evaluation Episodes Inspector with deterministic replays and 9-zone spatial stage bottleneck diagnostics (`67a6ac7`).

### September 2026 — Replay Lifecycle, Continuity, Net2WiderNet, and Stage 26 (Committed: `fa44ed6` – `82a7f66` + Changelog)
- **Sep 1 – Sep 18 (`fa44ed6` – `82a7f66`)**: Resolved telemetry discrepancies (`fa44ed6`), added replay retention lifecycle and user pinning (`9be32f6`), preserved neural model continuity across checkpoint restore and stage revert (`b3d9a29`), and enforced authentic evaluation replay recording with strict 100% penned pass criteria (`82a7f66`).
- **Sep 27 – Sep 29 ([MODEL_CHANGELOG.md](../MODEL_CHANGELOG.md))**: Upgraded the policy MLP from `[128, 128, 128]` (`41,098` params) to `[256, 256, 128]` (~`115,000` params) via `Net2WiderNet` weight transplant with near-zero divergence (`0.000102` max logit diff, `0.000000` value diff), enabled `failure_directed_training_enabled`, and remediated the Stage 26 plateau (`max_steps` `1820 -> 2200`, `entropy_coef: 0.020`, `success_threshold: 0.85`).

### October 2026 — Ovine Biomechanics Refactor ([MODEL_CHANGELOG.md](../MODEL_CHANGELOG.md))
- **Oct 1**: Replaced omnidirectional radial pen repulsion (`pen_fearful`/`pen_shy` forcefields up to `~35.0` that overpowered dog pressure `~11.25`) with biologically grounded mechanics: directional threshold drag (`sheep_d_gate: 12.0`, `sheep_alpha_hesitation: 0.65`), flank pressure-relief containment (`sheep_theta_containment: 1.8` rad), leader-follower surge upon first pen breach (`sheep_surge_multiplier: 1.3`, `sheep_surge_cohesion_radius: 24.0`), and tonic immobility / packing (`sheep_p_crit: 8.0`).

### November 2026 — Forensics, Gate Info, Strided Autonomous Policy, Scaling, and Hardening
- **Nov 6 ([handoff.md](../handoff.md#L35-L100))**: Audited Stage 34 (`guided`, `observation_mode: guided`, `6/10` quick eval) and built the offline temporal dog–sheep interaction analyzer (`src/sheepdog/training/diagnostics/temporal_interaction.py`) and web diagnostics panel, surfacing corner trapping, `TeamStrategy._stray_sheep` 2-sheep edge cases, and `no_progress_steps` jitter resets.
- **Nov 12 – Nov 13 (Committed `bfd12db` + [handoff.md](../handoff.md#L1050-L1057))**: Hardened Stage 34 checkpoint continuity, backend concurrency, and web polling races (`bfd12db`), and ran a bounded 6-arm gate-information comparison (`17/30` control vs. `22/30` gate treatment on `DEV_SEEDS` in `guided` mode).
- **Nov 17 – Nov 19 ([handoff.md](../handoff.md#L1059-L1150))**:
  - Evaluated `TeamStrategy` 1.1.0 split-flock options (`team_strategy_split_flock_eval.md`) and fixed `no_progress_steps` consolidation jitter (`ENV_CONFIG_VERSION` `"1.0" -> "1.1"`, [no_progress_jitter_fix.md](no_progress_jitter_fix.md)).
  - Implemented strided temporal observation history (`strides=(0, 4, 8, 16)`) and `emergent_gate` (`208` inputs, no loop flags). Demonstrated clean-sheet `emergent_gate` Stage 1->2 promotion (`10/10` at `114,688` steps with `lr=3e-4`, `enable_instinct_rewards=False`, whereas `enable_instinct_rewards=True` scored `0/10`, [clean_sheet_emergent_gate_pilot.md](clean_sheet_emergent_gate_pilot.md)).
  - Shipped the unified dark 4-tab UI redesign (`Train`, `Progress`, `Trouble Spots & Replays`, `Settings`), fixed SQLite registry `INSERT OR REPLACE` cascade bugs and circular diagnostic imports, and added per-segment timing + parallel `Evaluator.evaluate`.
- **Nov 20 (Committed `704577a`, `5956385` + [handoff.md](../handoff.md#L1151-L1217))**:
  - **Milestone A (`704577a`)**: Added persistent evaluation summary cache (`727.6x` faster warm reload: `333.975s -> 0.459s`) and bit-exact environment hot-loop optimizations (`+9.0%`–`+10.3%` step throughput, [milestone_a_performance_report.md](milestone_a_performance_report.md)).
  - **Integrity (`5956385`)**: Isolated the promotion gate, excluded transition baselines, and tracked actual SB3 timesteps.
  - **Stage 8 Curriculum Experiments (working tree)**: Evaluated `wrong_side_recovery`, `coherent_wall_recovery`, `wrong_side_mild`, `wrong_side_safe_radius`, and 3-arm regrouping mechanisms across `DEV_SEEDS`, `Diagnostic-20`, `Test-30`, and `Confirmation-20` ([stage8_curriculum_v2_report.md](stage8_curriculum_v2_report.md), [stage8_regrouping_report.md](stage8_regrouping_report.md)).
- **Nov 22 – Nov 24 (Working tree, [handoff.md](../handoff.md#L1218-L1342))**:
  - **Nov 22**: Resolved `ENOSPC` disk exhaustion (`27.52 GB` pruned from unreferenced duplicate web replays, atomic I/O and pre-segment disk guards added) and diagnosed/fixed browser OOM (`Aw, Snap!`) crashes via live Chrome CDP profiling (dropping Train tab post-GC memory `168.8 MB -> 17.19 MB` and Settings DOM nodes `223,160 -> 1,866`).
  - **Nov 23**: Landed Round 1 correctness and throughput fixes (`+35.2%` rollout throughput `2,209.67 -> 2,987.07 trans/s`, real terminal observations on timeout truncation for `V(s_T)` bootstrapping, `0` false-positive wall penalties down from `517`, and removal of `TeamStrategy` / heuristic wait penalties in `emergent_gate` mode).
  - **Nov 24**: Fixed the `escapist` sheep anti-flock force bug (`ESCAPIST_MAX_BOLT_DISTANCE = 10.0`, `ENV_CONFIG_VERSION = "1.2"`), lifting an un-retrained Stage 37 checkpoint from `28/40 (70.0%)` to `35/40 (87.5%)` across Stage 37/38 `dev10` + `holdout10`, and wired stage-aware PPO overrides (`gamma=0.997` for 3 dogs, late-stage `lr=1e-4`, `target_kl=0.015`).

---

## 3. Major Improvements by Subsystem

### 3.1 Simulation Physics & Environment
- **Biologically Grounded Ovine Mechanics**: Replaced artificial radial pen-repulsion forcefields with directional threshold drag (`sheep_d_gate`), rear containment arc pressure relief (`sheep_theta_containment`), leader-follower surge (`sheep_surge_multiplier`), and tonic immobility packing (`sheep_p_crit`) ([MODEL_CHANGELOG.md](../MODEL_CHANGELOG.md)).
- **Boundary & Escapist Fixes**: Added inward sheep boundary steering (`0cc5032`), fixed sheep idle/vision bugs (`5505bb9`), and capped `escapist` anti-flock bolt distance at `ESCAPIST_MAX_BOLT_DISTANCE = 10.0` (`ENV_CONFIG_VERSION = "1.2"`) so lone escapist sheep on `144x104` arenas remain pushable ([handoff.md](../handoff.md#L1309-L1323)).
- **Accounting & Bootstrapping Accuracy**: Gated `no_progress_steps` on strict improvement over best-so-far spread (`ENV_CONFIG_VERSION = "1.1"`, [no_progress_jitter_fix.md](no_progress_jitter_fix.md)), replaced post-move wall-touch false positives with `any(invalid_action_attempts)`, and returned real terminal observations on timeout truncation so SB3 bootstraps `V(s_T)` rather than `0` ([handoff.md](../handoff.md#L1292-L1293)).
- **Bit-Exact Hot-Loop Speedups**: Shared per-dog `blocked_positions` across masking/projection/scoring and removed redundant per-step work, improving `SheepdogRLAdapter` throughput by `+10.3%` ([milestone_a_performance_report.md](milestone_a_performance_report.md)).

### 3.2 Neural Policy & Training Pipeline
- **Capacity & Temporal Memory**: Upgraded MLP from `[128, 128, 128]` to `[256, 256, 128]` via `Net2WiderNet` ([MODEL_CHANGELOG.md](../MODEL_CHANGELOG.md)), then added strided observation history (`strides=(0, 4, 8, 16)`) and the autonomous `emergent_gate` (`208`-input, no loop flags, no `TeamStrategy`) mode ([handoff.md](../handoff.md#L1073-L1078)).
- **PPO Stability & Continuity**: Added 4-stage adaptive learning-rate/entropy cooldown (`10298b4`), preserved neural weights and optimizer continuity across checkpoint restores (`b3d9a29`, `bfd12db`), fixed segment-boundary `reset_num_timesteps=False` accounting (`test_round1_segment_continuity.py`), and added stage-aware `gamma`/`lr`/`target_kl` overrides ([handoff.md](../handoff.md#L1328-L1335)).
- **Throughput Scaling**: Added `torch_num_threads=4` main-process tuning + `1` thread per worker (`+35.2%` `trans/s` at 16 workers), per-`(world_step, dog_index)` action-mask caching, and parallel `ProcessPoolExecutor` evaluation (`Evaluator.evaluate`).

### 3.3 Curriculum & Rewards
- **Stray & Gate Shaping**: Fixed lone-straggler reward deactivation when 1 unpenned sheep remained (`dist_to_centroid == 0.0`) ([model_training_history.md](model_training_history.md)), calibrated gate corridor/alignment rewards, and disabled heuristic wait penalties in `neural_policy` mode ([handoff.md](../handoff.md#L1294)).
- **Autonomous vs. Instinct Rewards**: Clean-sheet Stage 1 testing showed `emergent_gate` (`208` inputs) promoted `10/10` with `enable_instinct_rewards=False` (`lr=3e-4`) while `enable_instinct_rewards=True` scored `0/10` ([clean_sheet_emergent_gate_pilot.md](clean_sheet_emergent_gate_pilot.md)).
- **Disciplined Curriculum Screening**: Used paired multi-seed experiments with `DEV_SEEDS`, `Diagnostic-20`, `Test-30`, and `Confirmation-20` to test recovery spawns (`wrong_side_recovery`, `coherent_wall_recovery`, `wrong_side_mild`, `wrong_side_safe_radius`) and regrouping mechanisms—rejecting interventions that failed unseen confirmation ([stage8_regrouping_report.md](stage8_regrouping_report.md)).

### 3.4 Diagnostics, Tracking & Evaluation
- **Spatial & Temporal Forensics**: Built 9-zone spatial bottleneck analytics (`src/sheepdog/training/spatial_analytics.py`) and the offline temporal interaction analyzer (`src/sheepdog/training/diagnostics/temporal_interaction.py`) with strict provenance labeling and capture budgets.
- **Registry & Evaluation Integrity**: Fixed SQLite `INSERT OR REPLACE` cascade deletions in `registry.py`, excluded `*.inprogress.json` files from scans, isolated promotion gates (`5956385`), enforced 100% penned pass criteria (`82a7f66`), and added a persistent evaluation summary cache (`727.6x` warm speedup, [milestone_a_performance_report.md](milestone_a_performance_report.md)).

### 3.5 Web Studio & Operational Reliability
- **Calm 4-Tab Studio**: Consolidated the UI into `Train`, `Progress`, `Trouble Spots & Replays`, and `Settings` ([handoff.md](../handoff.md#L1113-L1115)) with intra-segment rollout progress visibility.
- **Memory & Disk Hardening**: Eliminated browser OOMs via slimmed `checkpoint-index.json` exports, `ETag`/`304`, gzip, lazy `<details>` mounting, pagination, hidden-tab poll pausing, and closure isolation (`168.8 MB -> 17.19 MB` on Train); prevented `ENOSPC` crashes with pre-segment disk checks, atomic `.tmp` cleanup, and unreferenced web replay pruning (`26.85 GB` reclaimed, [handoff.md](../handoff.md#L1220-L1282)).

---

## 4. Key Lessons Learned

1. **Physics Bugs Masquerade as Policy Plateaus**: For 65.1M timesteps (994 checkpoints), Stage 37 plateaued around `55–61%` because unnormalized `escapist` anti-flock repulsion overpowered dog pressure on `144x104` arenas. Capping bolt distance at `10.0` immediately boosted an unchanged checkpoint from `70.0%` to `87.5%` across 40 dev/holdout episodes ([handoff.md](../handoff.md#L1309-L1323)). *Lesson: Always audit force balances and edge-case geometry in failed replays before tuning PPO.*
2. **Reward Edge Cases Can Silently Zero Out Gradients**: In Stage 7, gating stray rewards on `dist_to_centroid > 8.0` deactivated the gradient whenever only `1` sheep remained (`flock_centroid == lone_sheep`, `dist == 0.0`). Fixing that single edge case unlocked `10/10` mastery ([model_training_history.md](model_training_history.md)).
3. **High Entropy Helps Escape Local Minima—Then Causes Drift**: Boosting `entropy_coef` to `0.016` helped break Stage 7 gate standoffs, but keeping it high after `100%` mastery degraded performance back to `50%` within 5 policy versions (`PV 1399–1402`). Adaptive cooldown (`10298b4`) and stage-aware `target_kl` are essential to lock in gains.
4. **Development Scores Can Mask Overfitting and Instability**: Stage 8 curriculum treatments that hit `10/10` on `DEV_SEEDS` still scored only `5/20–7/20` on unseen `Confirmation-20` wrong-side scenarios ([stage8_regrouping_report.md](stage8_regrouping_report.md)). *Lesson: Held-out seeds (`holdout10`, `Test-30`, `Confirmation-20`) and multi-seed runs are non-negotiable before promoting curriculum or reward changes.*
5. **Guided Baselines and Autonomous Policies Learn Differently**: On clean-sheet `emergent_gate` (`208` strided inputs), `enable_instinct_rewards=True` failed Stage 1 (`0/10`) while `enable_instinct_rewards=False` passed `10/10` ([clean_sheet_emergent_gate_pilot.md](clean_sheet_emergent_gate_pilot.md)). Heuristic penalties (`_tactically_valid_wait`) and `TeamStrategy` work in guided baselines but must be cleanly separated from the autonomous RL path.
6. **Observability Without Forensics Wastes Compute**: Building 9-zone spatial analytics and offline temporal replay analyzers turned multi-day guessing into pinpointed fixes (e.g., `no_progress_steps` jitter, false-positive wall penalties, timeout truncation `V(s_T)` bootstrapping).
7. **Long-Running Local RL Labs Need Production-Grade Ops**: At `200M+` timesteps and `44,000+` evaluation files, naive directory globs (`333s`), unpruned replay exports (`27.5 GB`), uncleaned `.tmp` files, and un-slimmed `78 MB` JSON polling will crash both disk (`ENOSPC`) and browser (`Aw, Snap!`). Atomic I/O, sidecar caches, disk guards, and `ETag`/pagination are prerequisites for uninterrupted multi-day training.
