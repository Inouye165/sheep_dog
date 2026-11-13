# Checkpoint 40575 Head/Tail Telemetry Evaluation Findings

**Evaluation Date:** 2026-11-11 (America/Los_Angeles)  
**Target Historical Checkpoint:** Checkpoint 40575 (`chk_run_20260817_063453_91f4_pv_10369_ts_339740239`)  
**Verified Model Weights:** `artifacts/checkpoints/stage34/maskable-ppo-stage34-339740239.zip`  
**Weights SHA-256:** `bc6b9883ab5f9343255c15e59cbf0252e8268808731482fec2f5b417d3196a00`  
**Experiment Root:** `artifacts/experiments/head_tail_diagnostic_chk40575_20261111_210123/`  
**Original Manifest:** `artifacts/experiments/head_tail_diagnostic_chk40575_20261111_210123/manifest.json`  
**Corrected Manifest:** `artifacts/experiments/head_tail_diagnostic_chk40575_20261111_210123/manifest_corrected.json`  
**Corrected Replays Directory:** `artifacts/experiments/head_tail_diagnostic_chk40575_20261111_210123/replays_corrected/`  
**Evaluation Script:** `scripts/run_head_tail_evaluation.py`  
**Assistance Classification:** `guided` (Neural policy with guided observation builder, 3 dogs, 6 sheep, Stage 34 full arena)  

---

## 1. Executive Summary & Required Disclosures

This investigation executed a bounded, isolated development evaluation of historical Checkpoint 40575 using bounded dual-window head/tail telemetry capture ($H=200$ head steps, $L=500$ rolling tail steps). The primary objectives were:
1. Establish verifiable identity and configuration provenance for historical Checkpoint 40575.
2. Certify bit-for-bit capture invariance between uninstrumented execution (Capture OFF) and windowed telemetry capture (Capture ON) on development seed 53.
3. Eliminate late-stall decision blindness by retaining real policy observations, action masks, value estimates, and action probabilities across the final 500 steps of failed episodes.
4. Analyze empirical evidence across 10 development seeds using the offline temporal interaction analyzer (`src/sheepdog/training/diagnostics/temporal_interaction.py`).

### Key Results
- **Phase 2 Invariance Gate:** **PASSED**. Executed action sequences ($4,062$ actions), all 28 `RewardBreakdown` components, complete environment snapshot states, episode length ($T=1,354$), termination status (`no-progress`), and ending PRNG states (Python, NumPy 5-element tuple, PyTorch, and Environment `_rng`) were **100% bit-for-bit identical** between Capture OFF and Capture ON.
- **Ten-Seed Benchmark Reproduction:** Evaluated 10 development seeds (`11, 23, 37, 41, 53, 59, 61, 67, 71, 73`). Achieved **7/10 full-flock successes (70.0%)**, with **2 no-progress stops** (Seed 53 at 5/6 penned, Seed 67 at 5/6 penned) and **1 timeout** (Seed 73 at 0/6 penned). Matching historical 7/10 success totals confirms aggregate performance under current environment code, but does not prove bit-for-bit reproduction of historical trajectories.
- **Late Telemetry Preservation:** Zero truncation of late-episode failure decisions. For every episode with $T > 700$, exactly 200 head frames and 500 rolling tail frames were preserved with complete decision records. Frame 201 contained the explicit omission marker (`steps_201_to_<tail_start-1>`).
- **Replay Storage Efficiency:** Compressed replay files averaged ~670 KB, with the largest episode (Seed 73, $T=2,320$) taking 979 KB—well below the 2.5 MB project target.

### Required Disclosures & Audit Corrections
1. **Actual Execution vs Reproduction Commands:**
   - The reported invocation executed:
     ```bash
     .venv/Scripts/python.exe scripts/run_head_tail_evaluation.py --output-dir artifacts/experiments/head_tail_diagnostic_chk40575_20261111_210123 --reuse-replays
     ```
   - A clean from-scratch reproduction command is:
     ```bash
     .venv/Scripts/python.exe scripts/run_head_tail_evaluation.py --output-dir artifacts/experiments/head_tail_diagnostic_chk40575_<timestamp>
     ```
2. **Elapsed Time Accounting on Reused Replays:**
   - In the reported run, 9 seeds (`11, 23, 37, 41, 59, 61, 67, 71, 73`) were reused from an earlier run in that directory. Their original execution times were unrecorded and are marked `unknown` (`None`), not 0.0s.
   - The reported total elapsed time of 60.85s was for the reuse invocation only (running Seed 53 control OFF and treatment ON, serializing Seed 53, and running the offline temporal interaction analysis on all 10 replays).
3. **Database Unregistered Checkpoint Status:**
   - Checkpoint 40575 was created prior to the relational tracking registry (`PRAGMA user_version = 1`). While its weights file SHA-256 (`bc6b9883...`) and checkpoint JSON match perfectly, it is not registered in the SQLite `registry_checkpoints` table.
4. **Configuration and Discount Provenance Correction:**
   - Checkpoint 40575 training used a `sequential_shared_dog` timebase (3 sequential dog transitions per world step in MaskablePPO) with training discount $\gamma = 0.997$ (verified from internal SB3 zip `data` JSON).
   - In-memory evaluation configuration defaulted to `LabConfig().training.gamma = 0.99`, creating an evaluation gamma mismatch (`0.99 != 0.997`).
   - **Behavioral Impact on Evaluation:** Zero impact. Deterministic evaluation rollouts use only actor forward-pass logits (argmax action selection); gamma is not used during policy inference or environment stepping.
   - **Replay Header Correction:** The initial runner script hardcoded `discount_timebase="joint_world_step"` and `evaluation_gamma=0.997` in replay headers, contradicting the in-memory config and training timebase. This has been corrected in `scripts/run_head_tail_evaluation.py`, documented in `manifest_corrected.json`, and corrected replay bundles are staged in `replays_corrected/`.
5. **Scope of Capture Guarantee:** The head/tail capture invariance verified here certifies standalone evaluation under fixed settings. Active background training rolling tail capture is not certified by this run.

---

## 2. Phase 1 — Verification of Input Identity & Configuration

Before any model loading or simulation occurred, input identity was cryptographically verified:
- **Verified weights path:** `artifacts/checkpoints/stage34/maskable-ppo-stage34-339740239.zip`.
- **SHA-256 Digest:** `bc6b9883ab5f9343255c15e59cbf0252e8268808731482fec2f5b417d3196a00` (Newly computed file hash matches expected digest).
- **SQLite Registry Status:** Unregistered historical checkpoint (created before relational schema).
- **Linkage Artifacts:**
  - `artifacts/checkpoints/checkpoint-chk_run_20260817_063453_91f4_pv_10369_ts_339740239.json` explicitly references `artifacts\checkpoints\stage34\maskable-ppo-stage34-339740239.zip`.
  - `artifacts/training-summary.json` records Checkpoint 40575 with `success_rate: 0.7` and evaluation ID `eval_chk_run_20260817_063453_91f4_pv_10369_ts_339740239_confidence_ff85663fc223`.
- **Model Architecture & Hyperparameters:**
  - Policy Architecture: Feedforward `MaskableActorCriticPolicy` (`MlpPolicy`).
  - Network Hidden Sizes: `[256, 256, 128]`.
  - Observation Size: 54 continuous features (`observation_mode: "guided"`).
  - Action Size: 9 discrete actions (wait, 4 cardinal moves at speed 2, 4 cardinal sprint moves at speed 4).
  - Discount Provenance: Training discount $\gamma = 0.997$, transition timebase `sequential_shared_dog`; evaluation effective config $\gamma = 0.99$.
  - Environment Dimensions: Arena $144 \times 104$, Pen $14 \times 14$ (`pen_placement: "random"`).
  - Dynamics: Dog speed 2.0 (sprint 4.0), Sheep speed 1.0 (surge multiplier 1.3, surge cohesion radius 24.0, $d_{\text{gate}} = 12.0$, $\alpha_{\text{hesitation}} = 0.65$, $\theta_{\text{containment}} = 1.8$, $p_{\text{crit}} = 8.0$).
  - Episode Budget: 2,320 world steps; no-progress window: 780 steps.

---

## 3. Phase 2 — One-Seed Capture Invariance Gate (Seed 53)

To ensure that telemetry capture introduces zero passive simulation side-effects or PRNG divergence, development seed 53 was run twice in a standalone deterministic test harness with identical initial seeds (`seed=53`, PRNG seed `53`):
- **Run A (Control):** `capture_step_telemetry=False`, `capture_replays=True`.
- **Run B (Treatment):** `capture_step_telemetry=True`, `telemetry_mode="head_tail"`, `telemetry_head_steps=200`, `telemetry_tail_steps=500`, `capture_replays=True`.

### Verification Evidence & Storage Boundary
Independent verification of the saved replay artifacts on disk (`checkpoint-040575-seed-000053_control_off.json.gz` vs `checkpoint-040575-seed-000053.json.gz`) confirms:
1. **Episode Length ($T$):** Exactly 1,354 frames on both runs.
2. **Action Sequence Invariance:** Step-by-step comparison across all 1,354 frames showed **zero action mismatches** ($1,354 \times 3 = 4,062$ identical actions).
3. **Reward Breakdown Invariance:** All 28 reward components per frame matched bit-for-bit with **zero numerical drift** (0 mismatches across 37,912 reward floats).
4. **Snapshot & Position Invariance:** Post-step dog and sheep coordinates, penned status, and spatial metrics were identical across all 1,354 frames (0 snapshot mismatches).
5. **PRNG State Invariance (Verified in Memory at Runtime):**
   - Python `random.getstate()`: Bit-for-bit identical at termination.
   - NumPy `np.random.get_state()`: All 5 tuple elements (`'MT19937'`, 624-key array, pos, has_gauss, cached_gaussian) bit-for-bit identical.
   - PyTorch `torch.get_rng_state()`: Identical byte-tensor match.
   - Environment internal `env._rng.getstate()`: Identical state tuple.
   *(Note: The saved replay files store frames, actions, snapshots, and rewards; PRNG states were verified in memory at runtime and recorded in the manifest.)*
6. **Telemetry Structure Validation:**
   - Head frames (1..200): All 200 frames contained full decision records (54-dim observation, action probabilities, critic values, masks).
   - Omission marker: Frame 201 contained `telemetry_omitted=True`, `omitted_range="steps_201_to_854"`, `omitted_steps_count=654`.
   - Tail frames (855..1354): All 500 frames contained complete decision records.

**Verdict:** Invariance Gate **PASSED**. Telemetry capture has zero behavioral impact on simulation outcomes.

---

## 4. Phase 3 — Bounded Ten-Seed Development Capture Results

With the gate certified, the 10 development seeds were evaluated under identical settings ($H=200, L=500$). Total evaluation consumed 9,944 world steps across 11 episodes.

### Frame vs Decision Counts
To avoid ambiguity, we distinguish **unique detailed frames** from **per-dog decision counts** (each detailed frame contains decisions for all 3 dogs, i.e., 3 decisions per detailed frame):
- **Seed 11 ($T=233$):** 233 unique detailed frames (head 1..200 and tail 1..233 overlap on 1..200); $233 \times 3 = 699$ per-dog decisions.
- **Seed 53 ($T=1,354$):** 700 unique detailed frames (200 head + 500 tail); $700 \times 3 = 2,100$ per-dog decisions.
- **Seed 67 ($T=1,369$):** 700 unique detailed frames (200 head + 500 tail); $700 \times 3 = 2,100$ per-dog decisions.
- **Seed 73 ($T=2,320$):** 700 unique detailed frames (200 head + 500 tail); $700 \times 3 = 2,100$ per-dog decisions.

| Seed | Steps ($T$) | Penned | Outcome | Total Reward | Replay Size | Detailed Frames | Per-Dog Decisions | Coverage Status | Reused | Elapsed |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **11** | 233 | 6/6 | **Success** | +272.19 | 329.6 KB | 233 | 699 | Complete ($T \le 700$) | Yes | Unknown |
| **23** | 236 | 6/6 | **Success** | +354.39 | 324.8 KB | 236 | 708 | Complete ($T \le 700$) | Yes | Unknown |
| **37** | 415 | 6/6 | **Success** | +242.75 | 565.5 KB | 415 | 1,245 | Complete ($T \le 700$) | Yes | Unknown |
| **41** | 280 | 6/6 | **Success** | -8.59 | 356.6 KB | 280 | 840 | Complete ($T \le 700$) | Yes | Unknown |
| **53** | 1,354 | 5/6 | **No-Progress** | -1,977.93 | 968.0 KB | 700 | 2,100 | Windowed (gap 201..854) | No | 17.49s |
| **59** | 971 | 6/6 | **Success** | -1,102.97 | 893.1 KB | 700 | 2,100 | Windowed (gap 201..471) | Yes | Unknown |
| **61** | 803 | 6/6 | **Success** | -418.25 | 947.1 KB | 700 | 2,100 | Windowed (gap 201..303) | Yes | Unknown |
| **67** | 1,369 | 5/6 | **No-Progress** | -2,228.92 | 941.6 KB | 700 | 2,100 | Windowed (gap 201..869) | Yes | Unknown |
| **71** | 609 | 6/6 | **Success** | -186.60 | 795.4 KB | 609 | 1,827 | Complete ($T \le 700$) | Yes | Unknown |
| **73** | 2,320 | 0/6 | **Timeout** | -3,900.39 | 979.5 KB | 700 | 2,100 | Windowed (gap 201..1820) | Yes | Unknown |

### Aggregate Metrics
- **Success Rate:** 7 / 10 (70.0%)
- **Average Penned:** 5.0 / 6 sheep
- **Average Steps on Successes:** 507.0 steps
- **Zero Capture Errors:** 0 serialization or forward-pass extraction exceptions across all seeds.

---

## 5. Phase 4 — Deep Replay & Late-Stall Evidence Analysis

The offline temporal analyzer (`analyze_episode_temporal_interactions`) processed all generated replays. The preserved late-tail decision telemetry provides rigorous empirical measurements across the failure states.

### Exact Failure Tail Windows & Empirical Measurements

#### 1. Seed 53: Exterior Closed-Fence Pin ($T=1,354$, Stopped at Step 1,354, 5/6 Penned)
- **Exact Tail Boundary:** Steps **855..1354** (500 steps, 1,500 dog decisions).
- **Geometry & Final State:** Pen located at $(117, 88)$ with opening facing **top** ($y=88$). Five sheep penned. Straggler Sheep 2 (personality: `pen_shy`) is trapped at $(115, 93)$ along the **west exterior closed fence**. Dogs 0, 1, and 2 are positioned at $(106, 91)$, $(111, 90)$, and $(109, 95)$—crowding the open field immediately west of Sheep 2.
- **Action Replacements vs Physical Displacement:**
  - Zero action replacements occurred (`replacement_reason=None` for all 1,500 decisions).
  - Out of 1,500 non-wait actions executed across the tail: Dog 0 had 0 zero-displacement non-waits; Dog 1 had 0; Dog 2 had 1 zero-displacement non-wait (0.2%).
  - Total zero-displacement non-waits: 1 out of 1,500 (0.067%). Execution collisions were essentially absent; dogs moved freely according to policy outputs.
- **Dog-to-Nearest-Unpenned-Sheep Distance Distribution ($N=1,500$ observations):**
  - Minimum: 2.24 cells
  - Median: 8.06 cells
  - Maximum: 13.60 cells
  - Mean: 7.97 cells
  - Per-dog medians: Dog 0 median 8.06 cells (range 3.16..13.60); Dog 1 median 6.32 cells (range 2.24..12.81); Dog 2 median 9.22 cells (range 5.66..13.00).
- **Entropy vs Cycles:** Mean policy distribution entropy across the late tail is $1.900$ nats. Although distribution entropy is high, the policy selects actions via deterministic argmax. High distribution entropy over 9 actions does **not** preclude deterministic cycles; the policy can alternate between two high-probability moves (e.g., `up` and `down`, or `left` and `right`) despite assigning small non-zero probabilities to remaining actions.
- **Critic Value $V(s)$:** Predicted discounted returns fluctuate between $-32.76$ and $+54.49$ (mean $+8.38$). Rather than demonstrating policy "recognition", this reflects the policy value function's learned expectation that 5 penned sheep typically yield positive terminal returns.
- **Stall Dynamics:** The dogs press Sheep 2 eastward directly into the closed west fence $(117, 93)$. Entering the pen requires navigating north around the corner to the top gate at $y=88$. The dogs hover at $y=90..95$, continuously exerting eastward pressure against the solid fence without flanking south to herd the sheep northward.

#### 2. Seed 67: North Exterior Closed-Fence Compression ($T=1,369$, Stopped at Step 1,369, 5/6 Penned)
- **Exact Tail Boundary:** Steps **870..1369** (500 steps, 1,500 dog decisions).
- **Geometry & Final State:** Pen located at $(128, 60)$ with opening facing **left** ($x=128$). Five sheep penned. Straggler Sheep 5 (personality: `escapist`) is pinned at $(133, 58)$ against the **north exterior closed fence**. Dogs 0, 1, and 2 are at $(137, 52)$, $(133, 49)$, and $(128, 48)$—north of Sheep 5 in the corridor between pen and top arena wall.
- **Action Replacements vs Physical Displacement:**
  - Zero action replacements occurred (`replacement_reason=None` for all 1,500 decisions).
  - Out of 1,497 non-wait actions executed across the tail: Dog 0 had 0 zero-displacement non-waits; Dog 1 had 3 zero-displacement non-waits (0.6%); Dog 2 had 0.
  - Total zero-displacement non-waits: 3 out of 1,497 (0.200%).
- **Dog-to-Nearest-Unpenned-Sheep Distance Distribution ($N=1,500$ observations):**
  - Minimum: 4.00 cells
  - Median: 8.94 cells
  - Maximum: 13.89 cells
  - Mean: 8.71 cells
  - Per-dog medians: Dog 0 median 7.21 cells (range 4.00..12.17); Dog 1 median 8.06 cells (range 5.00..13.60); Dog 2 median 10.79 cells (range 7.21..13.89).
- **Entropy & Action Patterns:** Mean policy entropy is $1.954$ nats. Dogs predominantly oscillate horizontally: Dog 0 executes `right` 227 times and `left` 140 times; Dog 1 executes `right` 203 times and `left` 130 times; Dog 2 executes `right` 192 times and `left` 133 times.
- **Critic Value $V(s)$:** Value estimates average $+12.68$, again reflecting optimism from 5 penned sheep.
- **Stall Dynamics:** All three dogs remain clustered north of the sheep, driving it southward into the closed north fence of the pen ($y=60$). Reaching the gate at $(128, 67)$ requires Sheep 5 to move west and south. The dogs maintain north-side pressure that prevents the sheep from breaking away from the solid fence.

#### 3. Seed 73: Split-Flock Corner Oscillation ($T=2,320$, Timeout, 0/6 Penned)
- **Exact Tail Boundary:** Steps **1821..2320** (500 steps, 1,500 dog decisions).
- **Geometry & Final State:** Pen located at $(128, 73)$ with opening facing **left**. Zero sheep penned. Flock is split: Sheep 2 (personality: `escapist`) is in the bottom-left corner at $(23, 102)$, while the remaining 5 sheep are clustered in the open field at $(60..64, 53..56)$—over 60 cells from the dogs and over 70 cells from the pen.
- **Action Replacements vs Physical Displacement:**
  - Zero action replacements occurred (`replacement_reason=None` for all 1,500 decisions).
  - Out of 1,500 non-wait actions: Dog 0 had 0 zero-displacement non-waits; Dog 1 had 0; Dog 2 had 0.
  - Total zero-displacement non-waits: 0 out of 1,500 (0.0%).
- **Dog-to-Nearest-Unpenned-Sheep Distance Distribution ($N=1,500$ observations):**
  - Minimum: 7.62 cells
  - Median: 13.04 cells
  - Maximum: 28.02 cells
  - Mean: 15.65 cells
  - Per-dog medians: Dog 0 median 10.82 cells (range 8.06..13.60); Dog 1 median 25.02 cells (range 23.02..28.02); Dog 2 median 9.22 cells (range 7.62..13.04).
- **Oscillation Pattern:** Action counts reveal symmetric vertical oscillation:
  - Dog 0: `up` 167 times, `down` 167 times, `right` 84 times, `left` 82 times.
  - Dog 1: `up` 209 times, `down` 208 times, `left` 42 times, `right` 41 times.
  - Dog 2: `up` 212 times, `down` 210 times, `left` 40 times, `right` 38 times.
- **Critic Value $V(s)$:** Deeply negative value estimates averaging $-20.69$ (range $-38.54$ to $-6.76$), reflecting the cumulative time and distance penalties when 0 sheep are penned.
- **Stall Dynamics:** Dogs 0 and 2 are locked in a vertical oscillation near straggler Sheep 2, while Dog 1 oscillates in open field at $(22, 75)$ (median distance 25.02 cells from Sheep 2). With 0 sheep penned, the surge pull never activates, and the dogs never disengage from the corner straggler to gather the main flock.

---

## 6. Disentangling Candidate Failure Causes

With late-tail decision telemetry captured and measured, we evaluate the 6 hypotheses from the readiness review:

| Candidate Hypothesis | Measured Telemetry Evidence | Evaluation Verdict |
| :--- | :--- | :--- |
| **1. Late-Stall Decision Blindness** | Complete decision records (observations, masks, distributions, values) preserved across steps 855..1354 (Seed 53), 870..1369 (Seed 67), and 1821..2320 (Seed 73). | **RESOLVED**: Decision blindness is fully eliminated. |
| **2. Sequential Action Masking Collision Discrepancies** | Zero action replacements occurred across all 4,500 tail decisions (`replacement_reason=None`). Across all 3 failed tails, only 4 zero-displacement non-waits occurred out of 4,497 non-wait actions (0.089%). | **DISPROVED as Direct Cause**: Execution collisions and mask overrides did not force dog actions or cause stalls in late failure states. |
| **3. Policy Entropy Collapse (Deadlock)** | Tail policy distribution entropy remained high ($1.73$ to $1.95$ nats across failed seeds). However, deterministic argmax selection can alternate between two high-probability moves despite high distribution entropy. | **DISPROVED as Zero-Entropy Freeze**: Policy is not frozen in a 0-entropy argmax collapse, but deterministic argmax cycling remains an active failure mode. |
| **4. Closed-Fence Target Penetration & Gate Feature Absence** | In Seeds 53 and 67, dogs repeatedly selected moves driving sheep into solid exterior pen walls. The 54-dim observation provides only pen center $(x, y)$ and scalar distance to pen. | **CANDIDATE EXPLANATION**: Missing gate/fence features is a leading candidate explanation, but not a proven cause. Guided targets and spatial observations may convey indirect gate information. Requires controlled experimental verification. |
| **5. Boundary Pressure Release Failure** | Measured dog-to-nearest-sheep distances show medians of 8.06 cells (Seed 53), 8.94 cells (Seed 67), and 13.04 cells (Seed 73). Dogs maintain persistent proximity without backing off to create an escape corridor. | **SUPPORTED BY EMPIRICAL DISTRIBUTIONS**: Dogs fail to learn that temporary retreat away from trapped sheep enables them to turn toward the open gate. |
| **6. Unobserved Sheep Kinematic Forces** | Replays record post-step sheep positions, but internal forces ($F_{\text{predator}}$, $F_{\text{surge}}$, $F_{\text{wall}}$, candidate scores) remain unrecorded in simulation frames. | **UNRESOLVED / REQUIRES LOGGING**: While dog decision telemetry is transparent, sheep internal force dynamics remain inferred. |

---

## 7. Next Experiment Proposal: Controlled Gate-Information Ablation

> [!NOTE]
> This section is strictly an experiment proposal. No training or evaluation re-run has been performed.

### 7.1 Hypothesis & Objective
**Hypothesis:** Dogs stall against closed exterior pen walls in Seeds 53 and 67 because the policy observation vector provides only the pen center $(x_{\text{pen}}, y_{\text{pen}})$ and cannot distinguish open gate openings from solid fences. Providing explicit relative gate features will allow the policy to learn flanking behaviors around solid walls.

### 7.2 Experimental Protocol: Control vs Treatment
- **Control:** Unchanged policy architecture and 54-dimensional observation vector, initialized from Checkpoint 40575 weights.
- **Treatment:** Augmented observation vector with 4 new continuous features:
  1. $\Delta x_{\text{gate}} = (x_{\text{gate}} - x_{\text{dog}}) / W$ (relative gate opening x-offset)
  2. $\Delta y_{\text{gate}} = (y_{\text{gate}} - y_{\text{dog}}) / H$ (relative gate opening y-offset)
  3. $n_x$ (gate opening outward normal vector x-component)
  4. $n_y$ (gate opening outward normal vector y-component)
  Total observation size: $54 \to 58$ dimensions.

### 7.3 Weight Initialization & Compatibility Preservation
- To ensure numerical compatibility when resuming from Checkpoint 40575:
  - First layer weight matrix $W_1 \in \mathbb{R}^{256 \times 58}$:
    - Retain historical weights for the first 54 inputs: $W_1[:, :54] = W_{1, \text{chk40575}}$.
    - Initialize the 4 new feature columns to exact zero: $W_1[:, 54:58] = 0$.
  - This guarantees that at step 0 of training, the treatment policy's forward pass outputs and value predictions are **numerically identical** to the baseline checkpoint.

### 7.4 Training & Evaluation Protocol
- **Compute Budget:** Equal training budget for Control and Treatment (e.g., 50,000 policy transitions under MaskablePPO, Stage 34 full arena).
- **Training Seeds:** 3 independent training seeds per condition (e.g., seeds 101, 102, 103) to ensure statistical reliability.
- **Evaluation Sets:**
  - *Development Evaluation:* The 10 development seeds (`11, 23, 37, 41, 53, 59, 61, 67, 71, 73`) used for diagnostic comparisons.
  - *Held-Out Reserve:* 20 separate held-out seeds evaluated once post-training to prevent overfitting to development failure cases.
- **Stopping Conditions & Success Metrics:**
  - Full-flock 6/6 penning success rate (target $\ge 80\%$ on dev seeds, no regression on seeds 11, 23, 37, 41, 59, 61, 71).
  - Specific recovery verification: Seed 53 and Seed 67 converting from no-progress exterior closed-fence stalls to full successes.
  - Wall-pressure penalty reduction and elimination of closed-fence pin intervals in offline temporal interaction analysis.
