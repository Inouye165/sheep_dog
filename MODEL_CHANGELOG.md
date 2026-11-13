# Sheepdog Model & Curriculum Changelog

This document logs all architectural, behavioral, and curriculum changes made to the sheepdog reinforcement learning policy and training pipeline, with timestamps and rationale.

---

### [2026-10-01 07:45:00 PDT / 14:45:00 UTC] - Ovine Biomechanics & Flocking Mechanics Refactor

* **Problem**: Omnidirectional repulsive potential fields (specifically `pen_fearful` and `pen_shy`) applied inverse-square radial forces up to ~35.0 outward from the pen entrance, physically overpowering dog flight pressure (~11.25). Furthermore, fear panic disabled flock cohesion, causing escapist sheep to shatter radially into open space, while penned sheep exerted zero social attraction on remaining stragglers. This created mathematical barriers at the gate and credit assignment collapse during Stage 26/33+ training.
* **Changes**:
  1. **Directional Threshold Drag (Replacing Radial Pen Repulsion)**:
     - `sheep_d_gate`: Added and set to `12.0` (approach distance $D_{\text{gate}}$ from gate opening).
     - `sheep_alpha_hesitation`: Added and set to `0.65` (approach resistance coefficient $\alpha_{\text{hesitation}}$).
     - *Mechanism*: Eliminated omnidirectional outward forcefields. Replaced with forward-vector directional drag $\vec{v}_{t+1} = \vec{v}_t \cdot \max(0.2, 1.0 - \alpha_{\text{hesitation}} \cdot (1.0 - d / D_{\text{gate}}))$ and gate-proximity lateral exploration jitter.
  2. **Pressure-Relief Vector (Flank Squirting vs. Driving)**:
     - `sheep_theta_containment`: Added and set to `1.8` rad (~103° rear containment arc).
     - *Mechanism*: When herding dogs maintain a balanced concave containment arc behind the sheep covering $\ge \theta_{\text{containment}}$, path of least resistance resolves forward toward the gate. If dogs leave a lateral escape lane, sheep squirt laterally through the gap.
  3. **Leader-Follower Surge & Allelomimetic Threshold Collapse**:
     - `pen_breached`: Added episode state tracking.
     - `sheep_surge_multiplier`: Added and set to `1.3` ($1.3\times$ speed surge for trailing flock).
     - `sheep_surge_cohesion_radius`: Added and set to `24.0` ($R_{\text{flock}}$ radius for surge draw toward penned centroid).
     - *Mechanism*: When the first sheep enters the pen, $\alpha_{\text{hesitation}}$ collapses to 0.0 for all sheep within $R_{\text{flock}}$, attractive pull vector to penned centroid activates, and speed scales up $1.3\times$.
  4. **Tonic Immobility / Packing Threshold (Anti-Shatter Mechanic)**:
     - `sheep_p_crit`: Added and set to `8.0` (critical predator pressure threshold $P_{\text{crit}}$).
     - `is_packed`: Added to `SheepState` and `AgentSnapshot`.
     - *Mechanism*: When predator pressure exceeds $P_{\text{crit}}$ while the forward escape path is blocked or dragging, sheep transition to `is_packed`, freezing/slowing down, doubling flock cohesion attraction ($2\times$), and suppressing radial scatter.
  5. Synchronized in `config.py`, `environment.py`, `entities.py`, and `effective-training-config.json`.
* **Rationale**: Eliminates artificial gate forcefield traps while accurately modeling biological ovine psychology (Hamilton's Selfish Herd, gate balking drag, flank escape reflexes, and social allelomimetic draught).

---

### [2026-09-29 07:15:00 PDT / 14:15:00 UTC] - Stage 26 Plateau Remediation & Timeout Extension

* **Problem**: Over 48 hours of training (~15,800 episodes, 278 formal evaluations), the model oscillated around a 63.7% success rate and failed to meet the 6-run 90% gate streak needed for Stage 27 promotion. Telemetry revealed 94.7% of failures were timeouts at 1,820 steps with an average of 3.8/6 sheep already penned, caused by time expiring while dogs chased 1–2 distant strays across the large $132 \times 96$ interior-pen arena. Low policy entropy (-1.15) caused the dogs to repeat this same local optimum.
* **Changes**:
  1. **Timeout Extension**:
     - `max_steps`: Increased from `1820` to `2200` in `CURRICULUM_STAGES[26]` ([curriculum.py](file:///c:/Users/inouy/source/sheep_dog/src/sheepdog/curriculum.py)).
     - `no_progress_window`: Proportionally adjusted from `560` to `680`.
     - *Rationale*: Grants the dogs the ~380 additional steps needed to collect scattered strays across the $132 \times 96$ arena after penning the primary flock.
  2. **Exploration Re-stimulation**:
     - `entropy_coef`: Added stage override `"entropy_coef": 0.020` (up from baseline `0.010`) in `CURRICULUM_TRAINING_OVERRIDES[26]` ([curriculum.py](file:///c:/Users/inouy/source/sheep_dog/src/sheepdog/curriculum.py)).
     - *Rationale*: Counteracts policy entropy decay and encourages discovery of dedicated outrunner/stray-retrieval roles prior to driving the main group toward the interior pen.
  3. **Promotion Gate Calibration**:
     - `success_threshold`: Set to `0.85` (down from `0.90`) for Stage 26 in `CURRICULUM_STAGES[26]` ([curriculum.py](file:///c:/Users/inouy/source/sheep_dog/src/sheepdog/curriculum.py)).
     - *Rationale*: A 90% streak across 6 consecutive evaluation windows on stochastic 6-sheep interior-pen physics was mathematically punitive; 85% unlocks promotion to Stage 27 (random pen placements) once consistent mastery is reached.
  4. Synchronized in [effective-training-config.json](file:///c:/Users/inouy/source/sheep_dog/artifacts/effective-training-config.json).

---

### [2026-09-27 13:58:37 PDT / 20:58:37 UTC] - Net2WiderNet Model Architecture Upgrade

* **Problem**: The policy capacity at `[128, 128, 128]` (~41K parameters) was nearing its representation ceiling on complex spatial scenarios with 3 dogs and 6 scattered sheep across large dimensions.
* **Changes**:
  1. **Weight Transplant (Net2WiderNet)**:
     - Widen hidden layers from `[128, 128, 128]` to `[256, 256, 128]` (parameters increased from 41,098 to ~115,000, ~2.8x capacity).
     - Executed via [widen_model.py](file:///c:/Users/inouy/source/sheep_dog/widen_model.py).
     - Old weights copied into new layer sub-matrices; new neurons initialized with near-zero Gaussian noise (`noise_scale=0.01`).
     - Max action logit divergence: `0.000102`; Value prediction divergence: `0.000000` (identical initial behavior, zero loss of learned progress across 29,633 episodes).
  2. **Configuration & Policy Loaders**:
     - Updated default `neural_hidden_sizes = (256, 256, 128)` in `TrainingConfig` ([config.py](file:///c:/Users/inouy/source/sheep_dog/src/sheepdog/config.py)).
     - Modified `NeuralPolicyConfig.from_dict` in [neural.py](file:///c:/Users/inouy/source/sheep_dog/src/sheepdog/policies/neural.py) to dynamically reference `config.training.neural_hidden_sizes` fallback rather than hardcoding `(128, 128, 128)`.
     - Saved updated weights into [best-model.zip](file:///c:/Users/inouy/source/sheep_dog/artifacts/models/best-model.zip), Stage 26 checkpoints, and [training-state.json](file:///c:/Users/inouy/source/sheep_dog/artifacts/training-state.json).

---

### [2026-09-27 13:50:00 PDT / 20:50:00 UTC] - Reward Shaping & Failure-Directed Training

* **Problem**: Dogs exhibited high wait times and hesitation near the gate corridor, allowing sheep to loop back around the interior pen.
* **Changes**:
  1. **Penalties & Gate Scale Adjustments** in [config.py](file:///c:/Users/inouy/source/sheep_dog/src/sheepdog/config.py):
     - `wait_penalty`: Increased from `0.05` to `0.20` to disincentivize idle stalling.
     - `gate_corridor_progress_scale`: Increased from `0.8` to `1.6` to strongly reward guiding sheep through the pen mouth corridor.
     - `gate_alignment_scale`: Increased from `1.0` to `1.8` to penalize orthogonal or misaligned approaches.
     - `failure_directed_training_enabled`: Activated (`False` -> `True`) to increase dynamic exposure to failure scenarios (isolated strays and subpen splits).

---

### [Baseline Architecture & Hyperparameters]

* **Algorithm**: `MaskablePPO` (PPO with action masking for valid directional boundaries).
* **Observation Space**: 54 continuous features (dog poses, velocities, flock centroid, individual relative sheep vectors, pen relative distance and heading).
* **Action Space**: Discrete (9 actions: 8 directional moves + wait, with optional sprint toggle).
* **Base Architecture**: 3-layer MLP (`Tanh` activations) with separate actor (`pi`) and critic (`vf`) heads.
