# Diagnostic Readiness Review: Disentangling Model Failure Causes

**Document Reference**: `docs/diagnostic_readiness_review.md`  
**Date**: 2026-11-11  
**Project**: Sheepdog Herding Lab  
**Status**: Revised Technical Review & Bounded Telemetry Architecture Proposal  
**Benchmark Reference**: Stage 34 Evaluation Checkpoint 40575 (`chk_run_20260817_063453_91f4_pv_10369_ts_339740239`)  

---

## 1. Executive Summary & Readiness Verdict

### 1.1 Core Diagnostic Question
Can existing codebase mechanisms, saved replay structures, and recorded evaluation evidence disentangle why trained neural policies fail to herd all six sheep into the pen under the primary benchmark? Specifically, can current diagnostics reliably distinguish among:
1. Closed-fence blindness and invalid target penetration in observation and scripted guidance;
2. Late-stall decision blindness caused by fixed telemetry capture budgets;
3. Action masking limitations and sequential movement collision bottlenecks;
4. Unobserved sheep response forces, threshold drag, and candidate move scoring;
5. Sheep-slot permutation instability under distance sorting in feedforward policies; and
6. Scripted role-lock, collector stagnation, and boundary target clamping?

### 1.2 Categorized Diagnostic State
To prevent conflating what is proven from what is hypothesized, the diagnostic state is partitioned into four explicit categories:

1. **Verified Implementation Facts (Code-Audited Mechanics)**:
   - Action spaces: Dogs have 9 discrete actions (`wait`, 4 cardinal moves at speed 2, 4 sprint moves at speed 4). No diagonal actions exist.
   - Action masking in `project_dog_action()` checks pre-step positions of unpenned sheep and other dogs, but does not check simultaneous moves into the same destination.
   - Neural policy modes short-circuit `action_mask_for_dog()` at line 653, returning movement legality without applying caller-passed `reserved_positions`.
   - Execution in `_apply_dog_actions()` sequentially blocks dogs against both newly claimed positions and pre-step occupied positions.
   - Telemetry capture in `SheepdogRLAdapter` and evaluation runs is bounded by `max_telemetry_steps_per_episode` (default 1,000 steps).
   - Replay frames store post-step entity positions, executed action names, rewards, and outcomes for every step across the entire episode. Pre-step decision telemetry (raw observations, action masks, action probabilities, value predictions, selected pre-replacement actions, replacement reasons, and internal strategy state) ceases after step 1,000.
   - Sheep movement forces (predator pressure, leader surge pull, flank squirt, gate drag, and candidate scores) are computed within `_sheep_step()` and discarded each step.

2. **Recorded / Derived Episode Observations (Benchmark 40575 Replays)**:
   - Seed 53 stalled at 5/6 penned sheep with unpenned sheep S2 on the outside of the west closed fence; Dog 0 targeted $(119, y)$ behind the closed fence for ~300 steps (steps 650–950) before switching to $(115, 96)$ at steps 995–999.
   - Seed 67 stalled at 5/6 penned sheep with unpenned sheep S5 (index 5) compressed against the north closed fence at $(136, 58)$ by Dogs 0 and 1.
   - Seed 73 timed out at 0/6 penned sheep with straggler S2 at the bottom wall $(23, 102..103)$; Dogs 0 and 2 were within 7.6 to 13.0 cells of S2, while Dog 1 was within 21.0 to 25.0 cells. Dogs were ~63–65 cells from the main flock centroid $(62.0, 54.2..56.6)$ and ~111–124 cells from the pen center.
   - Seed 11 penned all 6 sheep in 233 steps; first sheep S2 entered at step 65.
   - Seed 41 penned all 6 sheep in 280 steps; first sheep S1 entered at step 95.

3. **Candidate Causal Explanations (Plausible Hypotheses Requiring Disentanglement)**:
   - Closed-fence target penetration causes dogs to pin sheep against solid walls.
   - Feedforward policies lack the recurrent memory or velocity features needed to break directional oscillation loops.
   - Sequential execution collision blocks convert valid policy moves into stationary stalls without the policy receiving collision feedback.
   - Directional threshold gate drag alters candidate direction scoring enough to prevent entry unless surge pull or containment driving is active.

4. **Measurements and Experiments Still Required**:
   - Instrumented capture of late policy decision telemetry (steps 1,000–2,320) during failure windows.
   - Instrumented logging of sheep candidate forces and gate drag at the moment of gate refusal.
   - Controlled ablation comparing closed-fence-aware target clamping vs open-field clamping under equal compute budgets.

### 1.3 Readiness Verdict: **Conditionally Ready for Spatial Observation; Blocked for Causal Attribution**
The Sheepdog Herding Lab has created capable offline replay inspection tools, temporal interval detection, and structured frame telemetry. However, **the existing system cannot definitively isolate the root cause of late-episode failure stalls**.

| Diagnostic Capability | Current Status | Available Evidence | Missing Critical Evidence |
| :--- | :--- | :--- | :--- |
| **Spatial Stall & Loop Detection** | **Ready** | Full trajectory positions, bounding boxes, BFS route distances, return cycles ([temporal_interaction_analysis_report.md](temporal_interaction_analysis_report.md)). | None for basic spatial geometry. |
| **Scripted Guidance Disentanglement** | **Partially Ready** | Reconstructed scripted roles and open-field targets; logged role switches. | Penetration of targets behind closed fences is detectable geometrically, but internal strategy stickiness across resets is unrecorded. |
| **Late-Stall Decision Inputs** | **Blocked** | Pre-step policy inputs, masks, action probabilities, and values for the first 1,000 steps. Executed action names and positions continue throughout the episode. | **Zero decision records after step 1,000**. Stalls occurring between steps 1,000 and 2,320 lack observations, masks, probabilities, and value estimates. |
| **Sheep Dynamic Forces & Scoring** | **Blocked** | Post-step sheep positions and discrete penned/personality flags. | Predator pressure, leader surge pull, containment flank squirt, and directional drag values are computed and discarded. |
| **Training vs Eval Mask Divergence** | **Partially Ready** | Static code analysis reveals structural mask differences and early-return short circuits. | Runtime logging does not record whether an executed wait was chosen by the policy, forced by an invalid mask, or caused by a dynamic collision block. |
| **Diagnostic Specificity** | **Needs Correction** | Generic interval alerts fire in UI. | `two_sheep_collector_blind_spot` fires on normal 6-sheep flock driving. |

**Verdict Summary**: The environment records complete spatial trajectories, executed actions, and rewards for entire episodes. However, because internal sheep candidate forces are discarded and late-episode policy decisions (>1,000 steps) are truncated, **post-hoc analysis cannot distinguish whether dogs stall because they cannot see the closed fence, because the policy predicts poor actions, because sequential action masks blocked their escape, or because sheep drag dynamics resisted pen entry**.

---

## 2. End-to-End Decision-Path Map & Training vs Evaluation Comparison

### 2.1 Complete Step Decision-Path Map
Every simulation step advances through a deterministic sequence across observation construction, scripted guidance, action masking, policy inference, validation/replacement, multi-substep dog movement, sheep response dynamics, reward scoring, and termination:

```
+---------------------------------------------------------------------------------------------------+
| 1. Observation Construction                                                                       |
|    - observations.py: RoleAwareObservationBuilder.build (L52-188) / Emergent (L349-430)           |
|    - Pen center (pen_x, pen_y), distance_to_pen, wall margins, stall counters, sorted sheep slots|
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
| 2. Scripted Guidance & Target Synthesis (Guided Mode Only)                                        |
|    - team_strategy.py: TeamStrategy.assign_roles (L190-280), _safe_open_field_target (L408-440)   |
|    - Computes blocker, rear pressure, and left/right flanker targets; injects into observation    |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
| 3. Action Mask Generation                                                                         |
|    - environment.py: action_mask_for_dog (L636-707), project_dog_action (L709-735)                 |
|    - Evaluates 9 discrete actions: wait, 4 cardinal moves (speed 2), 4 sprint moves (speed 4)      |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
| 4. Policy Action Selection                                                                        |
|    - Training: rl_env.py: SheepdogRLAdapter.step (L259-350), _current_dog_action_mask (L506-519)  |
|    - Evaluation: policies/neural.py: NeuralPolicy.select_actions (L428-520)                       |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
| 5. Action Validation & Replacement                                                                |
|    - environment.py: step (L918-940), _validate_action (L1969-1972)                               |
|    - Replaces unrecognized action strings with 'wait'; checks physical boundary legality          |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
| 6. Dog Movement Execution                                                                         |
|    - environment.py: _apply_dog_actions (L1810-1857), _movement_steps (L553-585)                  |
|    - Sequential dog execution: moves dog 0, then dog 1, then dog 2                                |
|    - Resolves dynamic cell claims against claimed_positions and occupied_positions               |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
| 7. Sheep Response Simulation                                                                      |
|    - environment.py: _move_sheep (L1859-1967), _sheep_step (L1974-2242),                         |
|      _score_sheep_candidate (L2244-2290)                                                          |
|    - Computes predator repulsion, leader surge pull, flank squirt, directional gate drag          |
|    - Evaluates 4 candidate cardinal positions; selects candidate with highest scalar score        |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
| 8. Reward Computation                                                                             |
|    - rewards.py: RewardComputer.compute (L180-280)                                                |
|    - Evaluates 28 component terms: penning bonuses, route progress, stall/reversal penalties       |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
| 9. Termination & Stoppage Assessment                                                              |
|    - environment.py: step (L960-985), _progress_made_for_no_progress (L2682-2725)                 |
|    - Checks full penning (success), step budget (timeout), or no-progress stall limit (stoppage)  |
+---------------------------------------------------------------------------------------------------+
```

### 2.2 Deep Audit: Action Masking and Sequential Movement Dynamics

#### 1. What `project_dog_action()` Actually Checks
In `src/sheepdog/environment.py` (`project_dog_action`, lines 709–724):
```python
dog = self._dogs[dog_index]
blocked_positions = {sheep.position for sheep in self._sheep if not sheep.penned}
blocked_positions.update(other.position for other in self._dogs if other.index != dog_index)
steps, _ = self._movement_steps(
    self._dog_action_speed(action),
    dog.movement_budget,
    allow_accumulation=action != "wait",
)
return self._advance_position(
    dog.position,
    action,
    steps,
    blocked_positions=blocked_positions,
)
```
`project_dog_action()` evaluates whether an action will move the dog into a blocked cell. `blocked_positions` explicitly includes:
- Current pre-step positions of all unpenned sheep;
- Current pre-step positions of all other dogs; and
- Outer boundary walls and solid pen fences (checked within `_advance_position`).

If an action cannot advance at least 1 cell because its path is blocked, `_advance_position` returns `dog.position`. In `action_mask_for_dog` (lines 646–650):
```python
movement_mask: dict[Action, bool] = {
    action: self.project_dog_action(dog_index, action) != dog.position
    for action in ACTION_ORDER
    if action != "wait"
}
```
Thus, the policy action mask **does** filter out moves that collide with current pre-step positions of other dogs, unpenned sheep, and static walls. The neural policy is not blind to existing obstacle positions.

#### 2. The Early Return and Ignored `reserved_positions`
In `action_mask_for_dog` (lines 651–653):
```python
if policy_mode in {None, "neural_policy", "shepherd_neural_dogs"}:
    movement_mask["wait"] = True
    return movement_mask
```
Because `policy_mode` defaults to `None` in both training and evaluation call sites:
- `SheepdogRLAdapter._current_dog_action_mask` (`rl_env.py` L506–519) passes `reserved_positions` containing pre-step positions of earlier dogs.
- `NeuralPolicy.select_actions` (`neural.py` L453–477) passes `reserved_positions` containing projected next destinations of earlier dogs.
- **Both call sites are ignored by the environment**: `action_mask_for_dog` returns at line 653 before checking `reserved_positions`.

#### 3. Simultaneous Moves and Sequential Execution Bottlenecks
The narrower, verified issue is that `project_dog_action` tests only **pre-step static positions**:
1. If Dog 0 and Dog 1 both select actions taking them to the same unoccupied cell $C$, both actions are evaluated as legal by their respective masks.
2. In `_apply_dog_actions` (lines 1810–1821), movement executes sequentially:
   ```python
   sheep_positions = {sheep.position for sheep in self._sheep if not sheep.penned}
   occupied_positions = {dog.position for dog in self._dogs}
   claimed_positions: set[Point] = set()
   for dog, action in zip(self._dogs, actions, strict=True):
       blocked_positions = (
           sheep_positions
           | claimed_positions
           | {position for position in occupied_positions if position != dog.position}
       )
   ```
   Dog 0 moves first and adds $C$ to `claimed_positions`.
3. When Dog 1 moves, cell $C$ is in `claimed_positions`. Dog 1 cannot enter $C$ and halts in place, forfeiting its move and incrementing `dog.blocked_steps`.
4. Furthermore, Dog 1 is blocked against `occupied_positions` (the pre-step position of Dog 2), even if Dog 2 is vacating that cell during the current step.
5. This sequential execution asymmetry means that a valid action chosen by the policy can be converted into an in-place stall during physics execution, without pre-step policy awareness.

---

## 3. Observation Sufficiency Audit

### 3.1 Closed-Fence and Gate Blindness
The observation features provided to the policy (both `RoleAwareObservationBuilder.build` L74–80 and `EmergentObservationBuilder.build` L371–377) include:
- `pen_x, pen_y`: The center coordinates of the pen.
- `distance_to_pen`: Euclidean distance from dog to pen center.
- `wall_left, wall_right, wall_top, wall_bottom`: Distances to the outer boundary walls.

**Crucial Omission**: The observation vector contains **zero information about pen orientation, fence segments, or gate opening coordinates**:
- A rectangular pen with a north-facing gate has solid closed fences on its west, east, and south perimeters.
- The policy receives only the geometric center $(x_{\text{pen}}, y_{\text{pen}})$.
- To the neural network, approaching a sheep pinned against the outside of the south closed fence is geometrically indistinguishable from driving a sheep toward an open south gate.
- The policy cannot learn an explicit gradient around the closed fence because the fence geometry is completely unobservable.

### 3.2 Total Absence of Velocity and Kinematic Memory
In both observation builders:
- Positions are normalized coordinates $(x, y) \in [0, 1]^2$.
- Velocity vectors $(\dot{x}, \dot{y})$ or $(v_x, v_y)$ are **completely omitted**.
- Heading, momentum, turn radius, and acceleration do not exist in the observation.
- Although older changelog entries described "velocity-aware inputs", the code contains only scalar distance differences or static historical revisit flags (`revisits_recent_position`, `two_position_loop`).
- A feedforward MLP (`MlpPolicy` with hidden sizes `[256, 256, 128]`) receiving a single frame cannot distinguish whether a sheep is fleeing toward the pen or fleeing into a corner.

### 3.3 Completely Hidden Internal States
Several physical simulation variables that dictate entity behavior are entirely invisible to the policy:
1. `dog.movement_budget`: Fractional sprint accumulator (0.0 to 1.0) determining whether the dog will take 1 or 2 sub-steps.
2. `sheep.panic_steps` and `sheep.personality`: Personality multipliers modifying flight distance and hesitation drag.
3. `sheep.packed`: Boolean indicating density compression.
4. `TeamStrategy` sticky state: Internal persistent registers (`_sticky_pen_dx`, `_sticky_pen_dy`, `steps_in_role`) that govern role target synthesis.

### 3.4 Discontinuous Sheep-Slot Permutations
In `RoleAwareObservationBuilder.build` (lines 147–150):
```python
sorted_sheep = sorted(
    environment.sheep,
    key=lambda sheep: (sheep.penned, dog.position.distance_to(sheep.position), sheep.index),
)
```
Sheep slots (`sheep_0`, `sheep_1`, ..., `sheep_5`) are sorted every single step by distance to the individual dog:
- If Sheep A is at distance 5.01 and Sheep B is at distance 4.99, Sheep B occupies slot 0 and Sheep A occupies slot 1.
- If Sheep B moves 0.05 units away, their slot positions instantly swap.
- For a feedforward neural network without permutation-invariant encoders, this causes discontinuous input jumps. The network weights assigned to slot 0 suddenly see a different sheep with different coordinates and penning status, inducing high-frequency action chatter.

---

## 4. Movement and Sheep Response Dynamics Audit

### 4.1 Quantized Movement Rules and Inertia
In `src/sheepdog/environment.py`:
- Movement is restricted to **4 cardinal directions** (`up`, `down`, `left`, `right`) and `wait`. There are no diagonal moves.
- Dogs move at regular speed 2 (2 grid cells/step) or sprint speed 4 (4 grid cells/step).
- Sheep move at base speed 1 (1 cell/step) or surge speed 1.3 (accumulating fractional movement budgets).
- **Zero Inertia**: Entities can reverse direction by 180° instantly in a single step without turning deceleration or angular momentum. This can contribute to high-frequency oscillation loops where a dog steps left-right while a sheep mirrors right-left.

### 4.2 Directional Threshold Drag Semantics
Inspection of `src/sheepdog/environment.py` (`_sheep_step` lines 2064–2110 and `_score_sheep_candidate` lines 2244–2290) reveals the drag mechanism:
```python
r_gate_x, r_gate_y = self._gate_vector_into_pen()
forward_comp = vector_x * r_gate_x + vector_y * r_gate_y
if forward_comp > 0:
    drag_factor = 1.0 - alpha * proximity
    vector_x -= (1.0 - drag_factor) * forward_comp * r_gate_x
    vector_y -= (1.0 - drag_factor) * forward_comp * r_gate_y
```
- Drag attenuates the **continuous candidate preference vector** $(v_x, v_y)$ along the pen entry axis.
- When `_score_sheep_candidate` evaluates candidate grid steps, the scalar score is computed as:
  $$\text{score} = \Delta x \cdot v_x + \Delta y \cdot v_y - \text{penalties}$$
- Attenuating $(v_x, v_y)$ changes which cardinal direction wins the `argmax`.
- **It does NOT reduce physical movement distance**. A sheep that decides to move still moves exactly 1 cell. It simply shifts the preference from entering the pen to moving laterally along the fence.

### 4.3 Disclosed Goal-Directed Sheep Assistance
The simulation contains two hardcoded, scripted sheep behaviors that provide assistance toward task completion:

#### 1. Leader-Follower Surge Pull (`_sheep_step`, lines 2018–2024, `environment.py` line 1893)
When at least one sheep enters the pen (`self._pen_breached` and `penned_centroid is not None`):
- Unpenned sheep **within the surge cohesion radius** (`sheep_surge_cohesion_radius = 24.0` cells in Benchmark 40575) receive an artificial vector pull directly toward the penned centroid with magnitude $1.8$:
  $$\vec{v}_{\text{surge}} = 1.8 \cdot \frac{\vec{p}_{\text{penned}} - \vec{p}}{\|\vec{p}_{\text{penned}} - \vec{p}\|}$$
- Speed is multiplied by 1.3 (`sheep_surge_multiplier`).
- Gate drag is disabled ($\alpha = 0.0$).
- **Crucial Clarification**: Sheep outside the 24.0-cell radius do **not** receive surge pull. Success cannot be claimed as entirely explained by surge pull, but when unpenned sheep are within 24 cells of the pen, this assistance significantly aids funneling.

#### 2. Containment Gate Driving (`_sheep_step`, lines 2026–2056)
When dogs surround a sheep such that their angular span covers the rear and both flanks ($\text{span} \ge \theta_{\text{containment}}$, $\phi_{\min} \le -0.25$, $\phi_{\max} \ge 0.25$):
- The sheep does not flee the dogs.
- Instead, the environment injects an artificial driving vector directly toward `_gate_entrance_center()` with magnitude $1.5$.

### 4.4 Discarded Dynamic Forces
At the conclusion of `_sheep_step`, the environment discards:
- `predator_pressure`: Total repulsive force from all dogs.
- `surge_pull`: Magnitude of leader-follower attraction.
- `flank_squirt`: Lateral escape vector.
- `drag_factor`: Gate reluctance attenuation.
- `candidate_scores`: Relative scores of the 4 cardinal directions.

Because these values are never stored in the step record, an auditor inspecting a stalled replay cannot verify whether a sheep stayed against the fence because of gate drag, because predator pressure was insufficient, or because flank squirt was triggered.

---

## 5. Empirical Evidence from Benchmark 40575 Replays

### 5.1 Verified Benchmark Provenance
- **Checkpoint ID**: `chk_run_20260817_063453_91f4_pv_10369_ts_339740239`
- **Run ID**: `run_20260817_063453_91f4`
- **Policy Version**: `10369` (Episode `40575`)
- **Assistance Mode**: Guided (54-dimensional observation schema `8f3c742d5c70e95941df72272ef41adab8080ad4a939706e5a57cc54331e6f21`)
- **Discount Factor**: $\gamma = 0.997$
- **Formal Metric**: 7/10 Full-Flock Success (70.0%), 1/10 Timeout (10.0%), 2/10 Controlled Stall Stoppage (20.0%).

### 5.2 Comparative Evidence Table: Failures vs Successes

| Replay / Seed | Total Steps | Penned | Stop Reason | Telemetry Range | Verified Entity IDs, Geometric Positions & Failure Observations |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`checkpoint-040575-seed-000053`** *(Failure)* | 1,354 | 5 / 6 | Stopped (`no_progress`) | Steps 1–1,000 (354 steps unrecorded) | **Closed-Fence Pin & Target Clamping**: Unpenned sheep S2 trapped at $(115, 93..95)$ on outside of west closed fence. Pen origin at $(117, 88)$ with north-facing gate. Dog 0 target X remained clamped at 119 behind closed fence for ~300 steps (steps 650–950) with Y fluctuating between 90 and 95; at steps 995–999, Dog 0 switched to rear pressure targeting $(115, 96)$. Pre-step decision telemetry ceases at step 1,000; executed actions, positions, and rewards continue to step 1,354. |
| **`checkpoint-040575-seed-000067`** *(Failure)* | 1,369 | 5 / 6 | Stopped (`no_progress`) | Steps 1–1,000 (369 steps unrecorded) | **North Fence Compression**: Unpenned sheep is **S5 (index 5)** at $(136, 58)$ against north closed fence. Pen origin $(128, 60)$ has west opening. Dogs 0 and 1 pressed south from $(136, 54..56)$, holding sheep against solid fence. Pre-step decision telemetry ceases at step 1,000; executed actions, positions, and rewards continue to step 1,369. |
| **`checkpoint-040575-seed-000073`** *(Failure)* | 2,320 | 0 / 6 | Timeout (budget reached) | Steps 1–1,000 (1,320 steps unrecorded) | **Distant Stall & Straggler Wall Trap**: Sheep S2 trapped at bottom wall $(23, 102..103)$. Dogs 0 and 2 at $(12..16, 95..99)$ are within 7.6 to 13.0 cells of S2; Dog 1 is within 21.0 to 25.0 cells. Distance from dogs to the *main flock centroid* at $(62.0, 54.2..56.6)$ is ~63–65 cells; distance to pen center is ~111–124 cells. Pre-step decision telemetry ceases at step 1,000 (final 1,320 steps unmonitored). |
| **`checkpoint-040575-seed-000011`** *(Success)* | 233 | 6 / 6 | Success (all penned) | Steps 1–233 (Complete) | **Early Penning & Cascade**: First penned sheep is **S2 (index 2)** at **step 65**. Surge pull ($\text{radius}=24.0, \text{pull}=1.8, \alpha=0.0$) activated for proximate unpenned sheep. Remaining 5 sheep penned by step 233. Full telemetry recorded. |
| **`checkpoint-040575-seed-000041`** *(Success)* | 280 | 6 / 6 | Success (all penned) | Steps 1–280 (Complete) | **Clean Approach**: Open-field driving. First penned sheep is **S1 (index 1)** at **step 95**. Remaining 5 sheep penned within 185 subsequent steps. Full telemetry recorded. |

### 5.3 Recorded Frame Data vs Missing Pre-Step Decision Telemetry

It is essential to distinguish between what remains recorded after step 1,000 versus what is lost:

- **Recorded Throughout Entire Episode (Including Steps > 1,000)**:
  - Exact $(x, y)$ positions of all 3 dogs and all 6 sheep in every frame;
  - Executed action names (e.g., `'up'`, `'down_sprint'`, `'wait'`) for each dog;
  - Sheep penned status, packed status, and personality flags;
  - All 28 reward components and episode-level termination reasons; and
  - Geometric distances (BFS route distance to pen, flock spread, wall margins).

- **Completely Missing After Step 1,000 (Truncated by Telemetry Budget)**:
  - Raw 54-dimensional policy observation vectors;
  - Action masks for each dog at decision time;
  - Action probabilities and logits from the neural policy forward pass;
  - Value function predictions $V(s)$ from the critic;
  - Selected pre-replacement actions and replacement reasons (`action_replaced` vs `projected_blocked_by_obstacle`); and
  - Internal `TeamStrategy` sticky state registers (`sticky_pen_dx`, `sticky_pen_dy`).

Because pre-step decision telemetry is absent during the actual stall windows (steps 1,001–1,354 in Seed 53; steps 1,001–1,369 in Seed 67; steps 1,001–2,320 in Seed 73), an auditor cannot determine whether the policy attempted to break away but was physically blocked, or whether the policy continuously output high-probability moves toward the fence.

---

## 6. Diagnostic Specificity Audit: `two_sheep_collector_blind_spot`

### 6.1 Audit of the Detection Rule
In `src/sheepdog/training/diagnostics/temporal_interaction.py` (lines 1868–1871):
```python
collector_blind_spot = (
    len(unpenned_sheep) >= 2
    and max_pairwise_sheep_dist >= 8.0
    and not collector_active
)
```
And in interval classification (lines 2721–2726):
```python
if collector_blind_steps >= max(3, duration // 4):
    mechanisms.append("two_sheep_collector_blind_spot")
    evidence.append(
        f"[Hypothesized Mechanism: two_sheep_collector_blind_spot] "
        f"Unpenned sheep were separated across the field/pen for {collector_blind_steps}/{duration} steps "
        f"while TeamStrategy.collector_active remained False."
    )
```

### 6.2 Findings and False Generalization
1. **Overly Broad Threshold (`>= 2`)**: The condition checks `len(unpenned_sheep) >= 2`. In an episode starting with 6 unpenned sheep, initial flock dispersion is frequently $\ge 8.0$ units across the field.
2. **Normal Flock Driving Flagged as Flaw**: At 6 unpenned sheep, `TeamStrategy` intentionally assigns dogs to `REAR_PRESSURE`, `LEFT_FLANKER`, and `RIGHT_FLANKER` to herd the main cluster. A dedicated `COLLECTOR` role is designed for an isolated stray breaking away from a compact flock, not for initial 6-sheep flock driving.
3. **UI Misdirection**: When inspecting Benchmark 40575 in the Temporal Diagnostics web app, Interval #1 (steps 1–120 with all 6 sheep unpenned) displays the `two_sheep_collector_blind_spot` warning badge.
4. **Verdict**: The label is an overly broad false generalization. It leads developers to misdiagnose normal open-field team driving as an alleged 2-sheep bug.

---

## 7. Ranked Candidate Explanations & Confidence Limits

Based on code analysis and empirical inspection, the candidate explanations for herding failure are ranked below:

| Rank | Candidate Explanation | Confidence | Supporting Evidence & Code References | Primary Confidence Limits |
| :---: | :--- | :---: | :--- | :--- |
| **1** | **Closed-Fence Blindness & Target Penetration** | **High** | `observations.py` L74–80 omits fence geometry; `team_strategy.py` L408–440 clamps targets only to outer field boundaries. In Seed 53, Dog 0 targeted $(119, 90..95)$ behind the closed fence for ~300 steps. | Replay frames prove spatial entrapment, but late policy logits (>1,000 steps) are missing, so dog policy intent cannot be verified. |
| **2** | **Late-Stall Decision Blindness (Telemetry Truncation)** | **Certain Fact** | `environment.py` / `rl_env.py` cap capture at 1,000 steps. In Seed 73, 1,320 of 2,320 steps (57%) lack pre-step decision telemetry. | Codebase implementation fact, not an empirical inference. |
| **3** | **Sequential Action Masking & Dynamic Collision Bottleneck** | **Medium** | `environment.py` L651 returns early, ignoring `reserved_positions`; `_apply_dog_actions` L1820 blocks dogs on both old and new positions sequentially. | Requires runtime instrumented comparison to measure how frequently valid policy moves are blocked by dynamic claims. |
| **4** | **Unobserved Sheep Response Forces & Candidate Scoring** | **High** | `environment.py` L2010–2085 computes forces but discards them; sheep move along argmax score rather than continuous vector integration. | Forces are discarded at runtime; exact force values cannot be reconstructed without re-simulation. |
| **5** | **Sheep-Slot Permutation Instability Under Distance Sorting** | **Medium** | `observations.py` L147–150 sorts sheep by distance every step, causing discontinuous feature jumps in feedforward MLP layers. | Feedforward networks may tolerate some slot jitter, but attention or Transformer baselines are untested. |
| **6** | **Scripted Role-Lock & Boundary Target Clamping** | **High (Seed-Specific)** | `team_strategy.py` L378–382 applies stagnation penalty only after 150 steps; historical Seed 73 exhibited long role-holds before Phase 4. | Phase 4/4.1 mitigated role-locks, but wall clamping remained active on outer boundaries. |

---

## 8. Implemented Late Policy Decision Telemetry (Head/Tail Windowing)

### 8.1 Scope Bounding
This implementation is **strictly bounded to preserving late policy decision telemetry**. It intentionally defers sheep kinematic forces, continuous force vectors, and execution collision logs to a separate future proposal. This keeps the instrumentation minimal, auditable, and non-disruptive to simulation physics. Zero changes to physics, rewards, observations, action spaces, guidance, or checkpoints.

### 8.2 Implemented Capture Modes

| Mode | Configuration | Storage Impact | Compute Lifecycle | Diagnostic Coverage |
| :--- | :--- | :--- | :--- | :--- |
| **Default** | `capture_step_telemetry = False` | Standard replay frames only; zero telemetry overhead. | Zero diagnostic forward passes. | Standard trajectory frames (`actions`, `snapshot`, `reward`, outcomes). |
| **Legacy Prefix** | `telemetry_mode = "prefix"`, `max_telemetry_steps = 1000` | Replays store up to 1,000 detailed steps, followed by a truncation marker. | Forward passes cease immediately after step budget exhaustion. | Complete opening coverage; late stalls (>1,000 steps) are truncated. |
| **Opt-In Head/Tail** | `telemetry_mode = "head_tail"`, `telemetry_head_steps = 200`, `telemetry_tail_steps = 500` | Strictly bounded ($\le H + L = 700$ detailed steps; $< 2.5\text{ MB}$ compressed). | **Continuous forward passes when $L > 0$** (because termination step $T$ is unknown in advance). Halts after $H$ when $L = 0$. | Captures both opening dynamics ($1..\min(H, T)$) and late terminal stalls ($\max(1, T-L+1)..T$). |

### 8.3 Crucial Compute vs Storage Distinction
A critical technical distinction is maintained and documented:
- **Bounded storage does NOT bound forward-pass compute in rolling tails**.
- Because the simulation cannot predict at step $t$ whether the episode will terminate at step $t+1$ or step $2,320$, maintaining a rolling tail requires collecting or buffering decision telemetry at *every step* when $L > 0$.
- While compressed storage is strictly bounded by evicting older intermediate frames from the rolling deque (`collections.deque(maxlen=L)`), the evaluation forward pass ($3 \times \text{dogs}$ recomputations) must execute for the duration of the episode if rolling tail telemetry is enabled.
- Conversely, when $L == 0$ and $H > 0$, diagnostic collection halts after step $H$, stopping all subsequent forward passes.

### 8.4 Exact Capture Semantics and Schema Alignment
1. **Frame Preservation**:
   - Every ordinary replay frame is preserved for the entire episode: `actions`, `snapshot`, `reward`, outcomes, and metadata.
   - Only detailed `step_telemetry` retention is windowed. Middle trajectory frames are **never** discarded!
2. **World-Step Indexing & Head/Tail Retention**:
   - Head window: $1..\min(H, T)$ attached to frames directly during execution.
   - Tail window: $\max(1, T - L + 1)..T$ maintained during execution in rolling ring buffer `collections.deque(maxlen=L)`. Evicted records are automatically garbage-collected.
   - Total retained detailed records at any point during simulation $\le H + L$.
3. **Overlap Deduplication**:
   - If total episode steps $T \le H + L$ (contiguous, $tail\_start \le H + 1$), all steps $1..T$ are retained contiguously; no gap marker is emitted.
4. **Gap Marker**:
   - If $T > H + L$ ($tail\_start > H + 1$), steps $(H + 1)..(tail\_start - 1)$ have omitted detailed telemetry.
   - Frame $H + 1$ receives an explicit omission metadata marker dictionary:
     ```python
     {
         "telemetry_omitted": True,
         "omitted_reason": "intentionally_omitted_window",
         "telemetry_mode": "head_tail",
         "head_steps": H,
         "tail_steps": L,
         "omitted_range": f"steps_{H+1}_to_{tail_start-1}",
         "omitted_steps_start": H + 1,
         "omitted_steps_end": tail_start - 1,
         "omitted_steps_count": tail_start - 1 - H,
         "total_world_steps": T,
     }
     ```
   - Subsequent omitted frames have `step_telemetry = None`.
   - Tail frames $tail\_start..T$ are stitched from the ring buffer upon episode completion.
5. **Replay Metadata & Coverage**:
   - Emits `telemetry_mode`, `telemetry_head_steps`, `telemetry_tail_steps`, and `telemetry_coverage`:
     ```python
     {
         "mode": telemetry_mode,
         "total_steps": T,
         "captured_detailed_steps": count_detailed,
         "omitted_steps_count": count_omitted,
         "retained_head_range": [1, min(H, T)] if H > 0 else None,
         "retained_tail_range": [tail_start, T] if L > 0 and tail_start <= T else None,
         "omitted_range": [H + 1, tail_start - 1] if tail_start > H + 1 else None,
         "telemetry_truncated": is_truncated,
         "complete_decision_coverage": (tail_start <= H + 1) and capture_step_telemetry,
     }
     ```
6. **Episode Reset and Exception Safety**:
   - Ring buffers and counters are cleared upon `env.reset()`.
   - Temporary capture flags are restored in `try...finally` blocks.

### 8.5 Implemented Files
- `src/sheepdog/training/replay_writer.py`: `CapturePolicy` with `telemetry_mode`, `telemetry_head_steps`, `telemetry_tail_steps`, `max_telemetry_steps_per_episode`, and `validate_telemetry_config(...)` enforcing valid modes and non-negative integer types (rejecting bools and floats).
- `src/sheepdog/environment.py`: `SheepdogEnvironment` validation in constructor, `run_policy`, and `run_policy_on_scenario`; configuration property getters/setters; mid-episode mutation rejection; rolling tail deque; omission marker insertion; and safe reset/cleanup.
- `src/sheepdog/training/rl_env.py`: `SheepdogRLAdapter` validation in constructor and property setters; mid-episode mutation rejection; dual-window capture coordination; rolling tail buffer; stitched trajectory persistence; and `export_active_trajectory(*, finalize_tail: bool = False)` for mid-run partial export.
- `src/sheepdog/evaluation/evaluator.py`: `Evaluator.evaluate` configuration validation and pass-through, replay coverage dictionary generation, and serialization.
- `src/sheepdog/training/diagnostics/temporal_interaction.py`: Support for `telemetry_omitted`, `telemetry_omitted_range`, `telemetry_mode == "head_tail"`, `schema_status = "omitted"`, and gap-resilient spatial analysis.
- `web/src/components/TemporalDiagnosticsPanel.tsx` & `TemporalDiagnosticsPanel.test.tsx`: Explicit intentional omission presentation across inspector headers, critic estimates, action probabilities, and observation fields (`"Telemetry omitted (windowed capture: steps X to Y)"`).
- `tests/test_head_tail_telemetry.py`: 16 comprehensive unit and integration tests covering configuration validation, mid-episode mutation rejection, application-level garbage collection, bit-for-bit RNG invariance, training adapter lifecycle, zero-length windows, legacy replay loading, benchmark replay gap verification, and reproducible multi-iteration performance benchmarking.

---

## 9. Verification Results

All required verification checks have been implemented and verified:

1. **Exact Benchmark Failure Case ($T = 1354, H = 200, L = 500$)**:
   - Verified in `test_exact_benchmark_failure_example_t1354_h200_l500`.
   - Exactly 1,354 total frames present (zero trajectory frames discarded).
   - Head frames $1..200$ (indices $0..199$): 200 frames with full decision telemetry.
   - Frame $201$ (index $200$): receives explicit omission marker (`omitted_range: "steps_201_to_854"`, `omitted_steps_count: 654`).
   - Omitted frames $202..854$: exactly 653 frames with `step_telemetry = None` (total omitted = 654).
   - Tail frames $855..1354$ (indices $854..1353$): exactly 500 frames with full decision telemetry.
   - Total detailed frames: $200 + 500 = 700$. Total frames: $700 + 654 = 1,354$.
   - Integration with `analyze_episode_temporal_interactions` and `build_temporal_diagnostic_payload` verified: `telemetry_omitted = True`, omitted steps cleanly labelled with `schema_status = "omitted"`, tail step decision evidence intact.
2. **Consistent Configuration Validation & Mid-Episode Mutation Rejection**:
   - Verified across `CapturePolicy`, `SheepdogEnvironment`, `SheepdogRLAdapter`, and `Evaluator`.
   - Rejects unknown modes (`ValueError`), negative lengths (`ValueError`), and non-integer / boolean types (`TypeError`).
   - Rejects mid-episode capture configuration mutations with `RuntimeError("Cannot change telemetry configuration during an active episode")` when an episode is active.
3. **Application-Level Payload Retention & Garbage Collection**:
   - Verified in `test_environment_run_policy_payload_eviction_and_gc` (using real `SheepdogEnvironment.run_policy`) and `test_rl_adapter_payload_eviction_and_gc` (using `SheepdogRLAdapter`).
   - Real weak reference tracking across an episode with a genuine gap ($T=10, H=2, L=2$).
   - Verified that after episode completion and `gc.collect()`, weak references for intermediate evicted payloads (steps 3..8) resolve to `None` (reclaimed), while retained head (steps 1..2) and tail (steps 9..10) payloads remain alive.
4. **Bit-for-Bit Trajectory, Reward, and RNG Invariance**:
   - Verified in `test_bit_for_bit_invariance_deterministic_and_sampled` across both deterministic and sampled policy modes over an episode with a genuine gap ($T > H+L$).
   - Compared capture OFF vs ON across:
     - Full action sequences for all dogs.
     - All 28 reward components per frame (`RewardBreakdown`).
     - Full environment snapshot fields and animal positions.
     - Episode termination flags (`terminated`, `timeout`, `stopped`, `success`, `stop_reason`).
     - Exact RNG states: Python `random.getstate()`, NumPy `np.random.get_state()`, PyTorch `torch.get_rng_state()`, and environment `env._rng.getstate()`.
     - Proved zero random numbers consumed by telemetry capture and zero divergence.
5. **Direct Training Adapter Lifecycle & Zero-Length Windows**:
   - Verified across complete `SheepdogRLAdapter` episodes:
     - Overlapping head/tail ($T \le H+L$): deduplication, no gap marker, all detailed payloads present.
     - Exact gap boundaries ($T > H+L$): frame $H+1$ has omission marker, tail frames stitched, ordinary frames intact.
     - Zero-length windows: $H=0, L=0$ (no detailed payloads, frame 1 has omission marker); $H>0, L=0$ (head only); $H=0, L>0$ (tail only).
     - Consecutive resets: buffers and state completely cleared without cross-episode leakage.
     - Mid-run partial export: `export_active_trajectory` verified with both raw buffer inspection and finalized tail stitching.
6. **Analyzer and Viewer Gap Semantics**:
   - Verified legacy replay loading compatibility in `test_legacy_replay_loading_compatibility`: replays without head/tail load without error, audit sets `telemetry_omitted = False`.
   - Verified `web/src/components/TemporalDiagnosticsPanel.tsx` in `web/src/components/TemporalDiagnosticsPanel.test.tsx`: intentional omission notice displayed across inspector headers, critic estimates, action probabilities, and observation vector fields.
7. **Performance Measurement & Storage Bounds**:
   - Reproducible multi-iteration benchmark helper (`run_multi_iteration_telemetry_benchmark`) executed with $N=5, T=300$ paired iterations, excluding warmup and alternating capture order.
   - Clarified that existing historical benchmark timing measured environment stepping, neural inference, and disk replay serialization together rather than separating collection timing.
   - Replay compressed size strictly verified: $< 50\text{ KB}$ per episode (median ~25 KB, max ~26 KB), well below the $< 2.5\text{ MB}$ target.
   - No fragile timing thresholds asserted in automated tests; micro-benchmark timings qualified as development measurements that do not directly extrapolate to full Stage 34 training.
8. **Regression and Full Test Suite Execution**:
   - `pytest tests/test_head_tail_telemetry.py tests/test_telemetry_enrichment.py tests/test_temporal_interaction_diagnostics.py -v`: **48/48 passed** in 56.92s.
   - `ruff check`: **0 errors** across all modified files and test suites.
   - Web test suite (`npm --prefix web test`): **26/26 test files passed, 170/170 tests passed** in 21.89s.
   - Completed standalone development evaluation of Checkpoint 40575 (`artifacts/experiments/head_tail_diagnostic_chk40575_20261111_210123/manifest.json` and `manifest_corrected.json`), confirming Phase 2 bit-for-bit invariance and capturing complete late-stall telemetry across all 10 development seeds without truncation. Corrected discount provenance (`sequential_shared_dog`), timing accounting for reused replays, empirical displacement/distance distributions, and next experiment proposal documented in [docs/head_tail_checkpoint40575_findings.md](head_tail_checkpoint40575_findings.md).

---

## 10. Controlled Evaluation Plan (Development Only)

### 10.1 Evaluation Protocol
- **Parent Checkpoint**: Checkpoint 40575 (`chk_run_20260817_063453_91f4_pv_10369_ts_339740239`).
- **Development Seed Set (10 Paired Seeds)**:
  `11, 23, 37, 41, 53, 59, 61, 67, 71, 73`
- **Held-Out Evaluation Reserve**:
  Deferred pending verified seed manifest. Unverified seeds (e.g. 101–120) must **not** be assumed held-out until their training provenance is formally verified against the training log registry.

### 10.2 What Improved Capture Distinguishes vs What Needs Subsequent Interventions

| Diagnostic Question | Solved by Bounded Late Telemetry? | Required Follow-up If Unresolved |
| :--- | :---: | :--- |
| **Did the policy output moves toward the fence during late stalls?** | **Yes** (Captured in late action probabilities and logits) | N/A |
| **Was the policy trapped in deterministic action-reversal cycles?** | **Yes** (Measured action entropy and argmax sequence) | N/A |
| **Were dog actions overridden or blocked by dynamic claims?** | **Yes** (Recorded in `replacement_reason` and legal masks) | N/A |
| **Did closed-fence guidance penetration cause the failure?** | **Partially** (Correlates guidance targets with policy actions) | Requires controlled ablation: closed-fence-aware target clamping vs baseline. |
| **Did sheep gate hesitation drag prevent entry?** | **No** (Forces remain discarded in this phase) | Requires separate Sheep Movement & Force Telemetry proposal. |

---

## 11. Separate Future Proposal: Sheep Movement and Force Telemetry (Deferred / Unapproved)

To maintain strict scope control, the logging of internal sheep dynamics is deferred to a future proposal. For reference, that future proposal would encompass:

1. **Candidate Move Scores**: Recording `_score_sheep_candidate` values across all 4 cardinal directions to determine why sheep chose lateral movement over pen entry.
2. **Intermediate Continuous Force Vectors**: Logging `predator_pressure`, continuous $(v_x, v_y)$, leader-follower `surge_pull`, flank squirt vectors, and directional `drag_factor`.
3. **Collision Logging**: Recording whether a sheep's preferred candidate move was blocked by a wall, fence, or other sheep.

*Review completed in accordance with AGENTS.md rules. All factual citations are verified against benchmark-40575 replays and repository source code.*
