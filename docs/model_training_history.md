# Model Training History & Stage 7 Remediation Log

**Created**: August 21, 2026  
**Repository**: `sheep_dog`  
**Focus**: Curriculum Stage Diagnostics, Stage 7 Failure Remediation (2 Dogs, 4 Sheep, Nearby Strays), and Stabilization  

---

## 1. Overview & Context

Curriculum Stage 7 introduces a significant step up in multi-agent coordination and spatial reasoning:
- **Field Configuration**: $96 \times 72$ grid with a $12 \times 12$ corner pen (bottom opening).
- **Entities**: 2 Shepherd Dogs, 4 Sheep.
- **Spawn Mix**: $45\%$ Fixed Easy, $45\%$ Randomized Flock, $10\%$ Nearby Stray (distance 6.0–9.0).
- **Promotion Threshold**: $\ge 90\%$ success across multi-seed evaluations with consistent step efficiency and no persistent seed failures.

---

## 2. Phase 1: Spatial Analytics & Root-Cause Diagnostics System

### Condition Needed to Fix
* **Problem**: When curriculum stages stalled or plateaued, developers had no historical spatial telemetry to prove *why* the stage was failing (e.g., whether sheep were trapped in specific corners, pinned against perimeter walls, hindered by pen placement, or failing on specific spawn presets).
* **Objective**: Build an end-to-end telemetry and diagnostics engine to continuously track zone residencies, trapped failures, and automated bottleneck classifications across the lifetime of every stage.

### Changes Made
1. **Spatial Analytics Engine** (`src/sheepdog/training/spatial_analytics.py`):
   - Discretized the arena into 9 semantic zones: 4 Corners (`top_left`, `top_right`, `bottom_left`, `bottom_right`), 4 Walls (`top_wall`, `bottom_wall`, `left_wall`, `right_wall`), and Open Field (`center`).
   - Implemented `SpatialEpisodeTracker` to evaluate flock centroid, unpenned sheep, and dog coordinates on every environment step.
   - Built automated diagnostic rules (`diagnose_stage_bottlenecks`) detecting:
     * **Corner Entrapment**: Low win rate & trapped timeouts in corners vs open field.
     * **Corner/Axis Bias**: Left vs Right or Top vs Bottom asymmetries.
     * **Pen Placement Friction**: Win rates indexed by pen coordinate.
     * **Spawn Setup Vulnerability**: Win rates indexed by spawn layout mode.
2. **Environment & Telemetry Pipeline**:
   - `src/sheepdog/entities.py`: Added spatial attributes (`initial_sheep_zone`, `final_sheep_zone`, `pen_zone`, `corner_time_pct`, `wall_time_pct`, `corner_stuck_at_end`, `spatial_metrics`) to `EpisodeStats`.
   - `src/sheepdog/environment.py`: Initialized and stepped `SpatialEpisodeTracker` during episode lifecycles.
   - `src/sheepdog/training/episode_store.py`: Added self-healing SQLite migrations and `get_stage_diagnostics(stage, run_id)` aggregation queries.
   - `src/sheepdog/server.py`: Added `GET /api/stage-diagnostics` endpoint.
3. **Interactive UI** (`web/src/components/StageBottlenecksPanel.tsx` & `DiagnosticsPanel.tsx`):
   - Created an interactive 2D 9-Zone Spatial Heatmap Matrix, Stage Selector (Stages 1–8), and Bottleneck Diagnostic Insight Cards.

### Expectations
- Provide immediate, indisputable evidence of why any curriculum stage is struggling.
- Accurately categorize failure modes and highlight lone stragglers, wall traps, and corner stalls.

---

## 3. Phase 2: Stage 7 Historical Data Audit & Root Cause Isolation

### Condition Needed to Fix
* **Observed Baseline (PV 736 – PV 1392)**:
  - Total Stage 7 episodes analyzed: 13,807+.
  - Overall win rate stalled at **55.6%**, with **27.0% hard timeouts (980 steps)** and **17.4% premature stagnation stops**.
  - **30.5% of all failures penned 3 out of 4 sheep**, and another **22.8% penned 2 out of 4 sheep**.
* **Root Cause Isolated**:
  1. **Lone Straggler Approach Reward Deactivation** (`src/sheepdog/rewards.py`):
     - In `_compute_stray_terms()`, isolated stray incentives were gated on `dist_to_centroid > 8.0`.
     - When 3 sheep were penned, only 1 unpenned sheep remained on the field, meaning `flock_centroid == lone_sheep` ($\text{distance} = 0.0$).
     - This **completely deactivated** the approach reward gradient, leaving the dogs with zero reinforcement signal to leave the pen and traverse the field to retrieve the 4th sheep.
  2. **Gate Hovering Local Minimum**: Dogs received high gate alignment/corridor rewards near the pen entrance, creating an attractor state that locked them into micro-oscillations ($\Delta x \approx 1\text{–}3, \Delta y \approx 5\text{–}9$).
  3. **Patience Window Too Short**: `no_progress_window` was set to 240 steps, aborting cross-field round-trip retrievals on the large $96 \times 72$ grid.
  4. **Insufficient Exploration Entropy**: Baseline `entropy_coef: 0.010` was too low for dogs to break gate standoff equilibrium.

### Historical Performance Summary (PV 736 – PV 1392)

| Policy Version Window | Timestamp Range (UTC) | Avg Eval Success | Max Eval Success | % Evals $\ge 80\%$ | Avg Steps | Avg Reward |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **PV 736 – 785** (Early Stage 7) | `2026-08-18T22:33:00Z` – `2026-08-19T00:58:56Z` | 37.0% | 70.0% | 0.0% | 591.7 | -61.6 |
| **PV 786 – 885** (Initial climb) | `2026-08-19T01:04:39Z` – `2026-08-19T17:40:20Z` | 59.1% | 80.0% | 10.0% | 424.3 | +87.6 |
| **PV 886 – 1285** (Flocking plateau) | `2026-08-19T17:47:46Z` – `2026-08-21T13:42:23Z` | 56.1% | 80.0% | 10.5% | 434.7 | +46.8 |
| **PV 1286 – 1385** (Pre-tweak baseline) | `2026-08-21T13:44:16Z` – `2026-08-21T18:43:57Z` | 61.4% | 90.0% | 22.0% | 420.7 | +42.3 |
| **PV 1386 – 1392** (Pre-tweak slump) | `2026-08-21T18:48:28Z` – `2026-08-21T19:00:09Z` | 44.0% | 70.0% | 0.0% | 598.0 | -76.8 |

---

## 4. Phase 3: Stage 7 Remediation & Hyperparameter Optimization

### Condition Needed to Fix
* Enable continuous approach reinforcement for single unpenned stragglers, extend step allowances for cross-field journeys, and boost exploration entropy.

### Changes Made
1. **Single-Straggler Approach Gradient** (`src/sheepdog/rewards.py`):
   - Updated `_compute_stray_terms` so that when `len(sheep_positions) == 1` and the lone sheep is far from target ($> 10.0$), the dense approach reward gradient ($(\text{prev\_min\_dog\_dist} - \text{min\_dog\_dist}) \times \text{scale}$) remains fully active.
   - Capped maximum step stray penalty at $-15.0$ to prevent passive risk-aversion.
2. **Exploration & Learning Rate Overrides** (`src/sheepdog/curriculum.py`):
   - Added `CURRICULUM_TRAINING_OVERRIDES[7]`:
     ```python
     7: {
         "entropy_coef": 0.016,          # Boosted from 0.010 to break gate standoff
         "learning_rate": 1.2e-4,         # Initial learning rate
         "learning_rate_final": 4.5e-5,   # Cooldown target
         "gae_lambda": 0.98,
     }
     ```
3. **Reward Scaling & Patience Budget** (`src/sheepdog/curriculum.py`):
   - `farthest_sheep_progress_scale`: Increased from $0.35 \to 0.55$.
   - `stray_ignore_penalty_scale`: Set to $0.0045$.
   - `no_progress_window`: Lengthened from $240 \to 300$ steps in `CURRICULUM_STAGES[7]`.

### Expectations
- Dogs will disengage from the pen after penning the primary flock and navigate directly to the 4th sheep.
- Eliminate lone-straggler timeouts and convert fragile spikes into sustained $\ge 90\%$ evaluation passes.

---

## 5. Phase 4: The Breakthrough Surge to 100% Mastery (PV 1393 – PV 1398)

Following deployment of the remediation changes, the model achieved the highest performance in Stage 7 history, reaching a flawless **100% (10/10 seeds passed)** benchmark at PV 1398:

| Policy Version | Timestamp (UTC) | Success Rate | Avg Steps | Avg Reward | Evaluation Result Breakdown |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **PV 1393** | `2026-08-21T20:00:01Z` | **90%** (9/10) | 143.6 | +279.5 | 9 wins (94–121 steps), 1 fail on Seed 11 (no-progress @ 468 steps) |
| **PV 1394** | `2026-08-21T20:01:49Z` | **80%** (8/10) | 241.0 | +246.8 | 8 wins (90–206 steps), 1 fail on Seed 11 (no-progress), 1 timeout on Seed 53 |
| **PV 1395** | `2026-08-21T20:03:08Z` | **80%** (8/10) | 232.8 | +161.5 | 8 wins (91–105 steps), 1 timeout on Seed 11, 1 no-progress on Seed 53 |
| **PV 1396** | `2026-08-21T20:04:36Z` | **80%** (8/10) | 289.5 | +122.8 | 8 wins (89–192 steps), timeouts on Seed 11 and Seed 53 |
| **PV 1397** | `2026-08-21T20:05:55Z` | **80%** (8/10) | 188.2 | +185.2 | 8 wins (99–116 steps), no-progress on Seed 11 (471 steps) & Seed 53 (568 steps) |
| **PV 1398** | `2026-08-21T20:07:04Z` | <mark>**100% (10/10)**</mark> | **117.4** | **+340.9** | **ALL 10 SEEDS PASSED** (All completed in 106–142 steps) |

*The optimal model weights from PV 1398 were persisted to `artifacts/models/best-model.zip` (`best_success_rate: 1.0`, `best_completion_steps: 117.4`).*

---

## 6. Phase 5: Post-Surge Drift Analysis & Final Cooldown Tuning

### Condition Needed to Fix
* Continued training under elevated entropy ($0.016$) and high learning rate ($1.2 \times 10^{-4}$) after achieving 100% mastery caused noisy rollout gradient updates to destabilize performance on difficult diagonal split seeds (Seeds 11 and 37).

### Performance Drift Log (PV 1399 – PV 1403)

| Policy Version | Timestamp (UTC) | Success Rate | Avg Steps | Avg Reward | Failure Root Cause |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **PV 1399** | `2026-08-21T20:08:50Z` | **80%** (8/10) | 277.4 | +193.2 | Timeouts on Seed 11 (980 steps) & Seed 37 (980 steps) |
| **PV 1400** | `2026-08-21T20:10:13Z` | **70%** (7/10) | 313.5 | +156.7 | Timeouts on Seed 11 & 37; no-progress on Seed 41 (463 steps) |
| **PV 1401** | `2026-08-21T20:11:43Z` | **70%** (7/10) | 372.0 | +71.5 | Timeouts on Seed 11, Seed 37, and Seed 41 |
| **PV 1402** | `2026-08-21T20:13:18Z` | **50%** (5/10) | 437.5 | -12.4 | High entropy oscillation across opposite corners |
| **PV 1403** | `2026-08-21T20:14:41Z` | **70%** (7/10) | 365.2 | +68.3 | Inconsistent stray retrieval on edge seeds |

### Changes Made (Cooldown & Fine-Tuning)
To lock in the optimal weights discovered at PV 1398 and suppress action churn, `CURRICULUM_TRAINING_OVERRIDES[7]` in `src/sheepdog/curriculum.py` was adjusted for fine-tuning:

```python
# Stage 7 fine-tuning & cooldown (locks in discovered herding policy, minimizes destructive drift)
7: {
    "entropy_coef": 0.005,          # Reduced from 0.016 -> suppresses action churn
    "learning_rate": 3.0e-5,         # Reduced from 1.2e-4 -> stabilizes neural network weights
    "learning_rate_final": 1.5e-5,   # Reduced from 4.5e-5
    "gae_lambda": 0.98,
}
```

### Expectations
- Prevent destructive policy drift and lock in the 100% mastery herding reflexes.
- Satisfy the 6-consecutive-evaluation auto-promotion criteria ($\ge 90\%$) for seamless progression to Stage 8.

---

## 7. Verification & Operational Status

* **Automated Test Suite**:
  - `pytest tests/test_rewards.py tests/test_curriculum.py tests/test_spatial_analytics.py tests/test_server.py`: **69 passed** in 12.64s.
  - `npm test --prefix web`: **105 passed** in 8.44s.
* **Active Status**: Server running with cooldown overrides in place, backed by best-checkpoint [best-model.zip](file:///c:/Users/inouy/source/sheep_dog/artifacts/models/best-model.zip) (PV 1398).
Before we accept Phase 1A, perform a strict compliance audit of the implementation against the refinements made after the original Phase 1A plan.

Do NOT begin Phase 1B.
Do NOT add UI.
Do NOT add AI/LLM functionality.
Do NOT change training, rewards, PPO, curriculum, or auto-promotion behavior.

First inspect the code you just implemented and answer whether each requirement below is fully implemented, partially implemented, or missing.

1. Rename/remove `needs_more_training_score`.

We explicitly do not want the deterministic system claiming to know whether more training is the correct treatment.

Replace it with an objective concept such as:

- `continued_learning_evidence_score`
or
- `failure_progress_score`

The score must represent evidence that failed episodes are improving, not a probability that additional training will solve the stage.

Update schemas, implementation, tests, serialization, comments, and documentation consistently.

2. Failure classification must permit uncertainty.

Verify that failed episodes can explicitly be classified as:

- `unknown`
- `multiple_candidate_causes`
- `insufficient_telemetry`

The classifier must not force every failure into one of the known behavioral failure modes.

3. Seed failure persistence must have severity.

Do not treat the third consecutive failure as automatic proof of a systematic defect.

Represent something approximately like:

1 failure = normal
2 failures = watch
3 failures = persistent_candidate
4+ failures = strong_persistence

Exact names may be adjusted if there is a cleaner implementation, but preserve the distinction.

4. Diagnostic thresholds must be centralized/configurable.

Values such as:

- persistent streak thresholds
- seed outlier failure rate
- minimum stage success rate
- minimum checkpoint history
- trend windows
- evidence thresholds

must not be scattered as unexplained magic constants throughout the algorithms.

Use a clearly named deterministic diagnostics configuration/constants structure.

This is diagnostic configuration only and must NOT alter RL training configuration.

5. Minimum-history safeguards.

Success slopes, failure-progress trends, step-efficiency trends, and similar calculations must not claim a trend when there is insufficient history.

Support explicit states such as:

- `insufficient_history`
- `improving`
- `stable`
- `regressing`

Where appropriate, report:

- sample_count
- checkpoint_count
- window_used

6. Supporting evidence.

Major diagnostic findings should preserve the facts that caused the finding.

For example, rather than only:

`corner_entrapment_top_left`

the report should be capable of carrying information resembling:

finding:
  type: corner_entrapment
  zone: top_left

support:
  affected_failures: 9
  total_failures: 11
  checkpoints_observed: 5
  affected_seeds: [...]
  longest_seed_streak: 5
  success_rate_with_condition: ...
  success_rate_without_condition: ...

evidence_strength: strong

Do not fabricate metrics that the existing telemetry cannot calculate. Use only evidence actually available.

7. Failure-signature persistence.

Verify that the system tracks not only:

“Seed 17 failed repeatedly”

but also:

“Seed 17 repeatedly failed with the SAME behavioral signature.”

Those are different levels of evidence.

The output should distinguish repeated seed failure with varying causes from repeated seed failure with a stable behavioral signature.

8. Evidence strength versus AI confidence.

`evidence_strength` must mean only the strength/amount/consistency of deterministic evidence.

It must never be described as confidence that the inferred causal explanation is correct.

There is no AI diagnostic confidence in Phase 1A.

9. Tests.

Add or modify tests as necessary to cover these requirements.

Particularly test:

- insufficient history
- unknown failure
- ambiguous/multiple candidate failure
- repeated seed with changing failure signatures
- repeated seed with identical failure signature
- streak severity transitions
- configurable thresholds
- improving failures while headline success is flat
- strong evidence versus weak evidence based on sample depth

Run the new deterministic diagnostics tests.

Also run the relevant existing regression suites again.

10. Produce sample reports.

After all tests pass, produce at least THREE serialized sample `DeterministicDiagnosticReport` outputs using test/synthetic histories:

A. Healthy learning:
headline success may be improving and no systematic bottleneck exists.

B. Flat headline success but improving failures:
success rate is approximately unchanged, but failed episodes are clearly getting closer to success. This should demonstrate why success percentage alone is insufficient.

C. Systematic behavioral bottleneck:
a seed or group of seeds repeatedly fails, preferably with a persistent behavioral signature and strong deterministic evidence.

For every report, explain in plain English what the deterministic engine is saying.

IMPORTANT:

The deterministic engine should report observations and evidence.

It should NOT recommend:
- changing a reward
- changing PPO parameters
- continuing training
- stopping training
- restarting training
- changing curriculum parameters

Those decisions will belong to later analyst layers.

At the end give me:

1. Compliance audit before changes
2. Files changed
3. Tests added/changed
4. Exact test results
5. The three sample reports
6. Any requirement that could not be implemented with current telemetry
7. Confirmation that Phase 1B has NOT begun