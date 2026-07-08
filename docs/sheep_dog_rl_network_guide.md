# Sheepdog RL Network & Project Guide

This guide is a comprehensive reference documenting the entire reinforcement learning system, neural network topology, environment physics, hyperparameters, and application architecture for the **Sheepdog Herding Simulation** project. It is structured specifically for use in NotebookLM to assist in understanding, navigating, and troubleshooting the codebase.

---

## Table of Contents

* [Source Map](#source-map)
* [Where to Look When... (Navigation Guide)](#where-to-look-when-navigation-guide)
* [1. Project Overview](#1-project-overview)
* [2. Main Entry Points](#2-main-entry-points)
* [3. Neural Network / RL Model](#3-neural-network--rl-model)
* [4. Hyperparameters Catalog](#4-hyperparameters-catalog)
* [5. Observation Features / Input Nodes](#5-observation-features--input-nodes)
* [6. Actions / Output Nodes](#6-actions--output-nodes)
* [7. Dog Agent System](#7-dog-agent-system)
* [8. Sheep Movement and Reaction System](#8-sheep-movement-and-reaction-system)
* [9. Reward System](#9-reward-system)
* [10. Curriculum / Stages](#10-curriculum--stages)
* [11. Training Loop](#11-training-loop)
* [12. Replay / Watch / Results](#12-replay--watch--results)
* [13. Scenario System](#13-scenario-system)
* [14. UI / Frontend Map](#14-ui--frontend-map)
* [15. Backend / API Map](#15-backend--api-map)
* [16. Data Files / Saved State](#16-data-files--saved-state)
* [17. Tests](#17-tests)
* [18. Known Issues / Things to Verify](#18-known-issues--things-to-verify)
* [19. Glossary](#19-glossary)
* [20. Diagrams](#20-diagrams)
* [21. Final Summary](#21-final-summary)

---

## Source Map

Below is a map of the repository's major files and folders along with the systems they house.

| Component / Layer | Relative File Path | Primary Class / Function / Enum |
| :--- | :--- | :--- |
| **Environment Physics** | [src/sheepdog/environment.py](file:///c:/Users/inouy/source/sheep_dog/src/sheepdog/environment.py) | `SheepdogEnvironment`, `ACTION_ORDER`, `ACTION_DELTAS` |
| **Agent Definitions** | [src/sheepdog/entities.py](file:///c:/Users/inouy/source/sheep_dog/src/sheepdog/entities.py) | `DogState`, `SheepState`, `Pen`, `Point`, `DogRole` |
| **Observation Building** | [src/sheepdog/observations.py](file:///c:/Users/inouy/source/sheep_dog/src/sheepdog/observations.py) | `RoleAwareObservationBuilder`, `EmergentObservationBuilder`, `HierarchicalObservationBuilder` |
| **Reward Computers** | [src/sheepdog/rewards.py](file:///c:/Users/inouy/source/sheep_dog/src/sheepdog/rewards.py) | `RewardComputer`, `HierarchicalRewardComputer` |
| **Configuration Schema** | [src/sheepdog/config.py](file:///c:/Users/inouy/source/sheep_dog/src/sheepdog/config.py) | `EnvironmentConfig`, `RewardConfig`, `TrainingConfig`, `LabConfig` |
| **Curriculum Stages** | [src/sheepdog/curriculum.py](file:///c:/Users/inouy/source/sheep_dog/src/sheepdog/curriculum.py) | `CURRICULUM_STAGES` |
| **Team Role Assigner** | [src/sheepdog/team_strategy.py](file:///c:/Users/inouy/source/sheep_dog/src/sheepdog/team_strategy.py) | `TeamStrategy`, `RoleAssignment`, `StrategySnapshot` |
| **Shepherd NPC Logic** | [src/sheepdog/shepherd.py](file:///c:/Users/inouy/source/sheep_dog/src/sheepdog/shepherd.py) | `ScriptedShepherd`, `ShepherdCommand`, `COMMAND_ORDER` |
| **Sequential RL Adapter** | [src/sheepdog/training/rl_env.py](file:///c:/Users/inouy/source/sheep_dog/src/sheepdog/training/rl_env.py) | `SheepdogRLAdapter` |
| **Joint RL Adapter** | [src/sheepdog/training/joint_rl_env.py](file:///c:/Users/inouy/source/sheep_dog/src/sheepdog/training/joint_rl_env.py) | `JointActionRLEnv` |
| **Hill-Climbing Trainer** | [src/sheepdog/training/trainer.py](file:///c:/Users/inouy/source/sheep_dog/src/sheepdog/training/trainer.py) | `Trainer` |
| **Maskable PPO Trainer** | [src/sheepdog/training/maskable_ppo.py](file:///c:/Users/inouy/source/sheep_dog/src/sheepdog/training/maskable_ppo.py) | `MaskablePPOTrainer` |
| **Hierarchical Trainer** | [src/sheepdog/training/hierarchical_trainer.py](file:///c:/Users/inouy/source/sheep_dog/src/sheepdog/training/hierarchical_trainer.py) | `HierarchicalMaskablePPOTrainer` |
| **Neural Policy Wrap** | [src/sheepdog/policies/neural.py](file:///c:/Users/inouy/source/sheep_dog/src/sheepdog/policies/neural.py) | `NeuralPolicy`, `NeuralPolicyConfig` |
| **Hierarchical Policy Wrap**| [src/sheepdog/policies/hierarchical.py](file:///c:/Users/inouy/source/sheep_dog/src/sheepdog/policies/hierarchical.py) | `ShepherdNeuralDogPolicy` |
| **Scenario Structure** | [src/sheepdog/evaluation/scenarios.py](file:///c:/Users/inouy/source/sheep_dog/src/sheepdog/evaluation/scenarios.py) | `Scenario`, `PenLayout`, `AgentLayout` |
| **CLI Commands** | [src/sheepdog/cli.py](file:///c:/Users/inouy/source/sheep_dog/src/sheepdog/cli.py) | `train_command`, `evaluate_command`, `benchmark_command` |
| **API Server** | [src/sheepdog/server.py](file:///c:/Users/inouy/source/sheep_dog/src/sheepdog/server.py) | `HTTPHandler`, `ServerManager` |
| **UI App Entry** | [web/src/App.tsx](file:///c:/Users/inouy/source/sheep_dog/web/src/App.tsx) | `App` React component |
| **UI Config Tab** | [web/src/components/ConfigPanel.tsx](file:///c:/Users/inouy/source/sheep_dog/web/src/components/ConfigPanel.tsx) | `ConfigPanel` React component |
| **UI Network Tab** | [web/src/components/NetworkTab.tsx](file:///c:/Users/inouy/source/sheep_dog/web/src/components/NetworkTab.tsx) | `NetworkTab`, `NetworkTopologyViewer` |
| **UI Layers Tab** | [web/src/components/LayersTab.tsx](file:///c:/Users/inouy/source/sheep_dog/web/src/components/LayersTab.tsx) | `LayersTab` React component |
| **UI Insights Tab** | [web/src/components/DiagnosticsPanel.tsx](file:///c:/Users/inouy/source/sheep_dog/web/src/components/DiagnosticsPanel.tsx) | `DiagnosticsPanel` React component |
| **UI Results Tab** | [web/src/components/ResultsPanel.tsx](file:///c:/Users/inouy/source/sheep_dog/web/src/components/ResultsPanel.tsx) | `ResultsPanel` React component |
| **UI Stages Tab** | [web/src/components/StagesTab.tsx](file:///c:/Users/inouy/source/sheep_dog/web/src/components/StagesTab.tsx) | `StagesTab` React component |

---

## Where to Look When... (Navigation Guide)

Use this table as a quick troubleshooting guide when you need to inspect or adjust specific systems.

> [!WARNING]
> Do not modify observation vector layouts in production configurations unless you intend to retrain all models from scratch; changing the features in `src/sheepdog/observations.py` breaks checkpoint compatibility!

| Goal / Concern | File to Inspect | Class / Function / Constant | What to Check / Modify |
| :--- | :--- | :--- | :--- |
| **Modify Sheep Reactivity** | `src/sheepdog/environment.py` | `SheepdogEnvironment._sheep_step` | Modify force vectors, personality behaviors, dog pressure calculations, or wall avoidance margins. |
| **Adjust Dog Movement Speeds** | `src/sheepdog/config.py` | `EnvironmentConfig.dog_speed` | Adjust base speeds, sprint multipliers, or sprint penalties (`sprint_cost_scale` in `RewardConfig`). |
| **Add/Modify Dog Actions** | `src/sheepdog/environment.py` | `ACTION_ORDER`, `ACTION_DELTAS` | Add new action keys (e.g., barking) and define their spatial offsets. |
| **Debug Dog Oscillation** | `src/sheepdog/team_strategy.py` | `_LATERAL_DEAD_ZONE` | Adjust the dead-zone (default 3 cells) where flanking targets are locked to prevent target swapping. |
| **Change NN Architecture** | `src/sheepdog/config.py` | `TrainingConfig.neural_hidden_sizes` | Modify the hidden layers tuple (e.g., change `(128, 128)` to `(256, 128)`). |
| **Adjust Reward Weights** | `src/sheepdog/config.py` | `RewardConfig` / `InstinctRewardConfig` | Adjust weights of progress, cohesion, overpressure, and time penalties. |
| **Understand why a stage regressed** | `artifacts/promotion-history.json` | JSON structure | Inspect candidate success rates and avg step metrics across checkpoints. |
| **Add a Curriculum Stage** | `src/sheepdog/curriculum.py` | `CURRICULUM_STAGES` | Insert a new stage dict with grid, sheep, dog, and mix variables. |
| **Change UI Layout** | `web/src/App.tsx` | React tab structure | Inspect the `APP_TABS` array and tab rendering switches. |
| **Inspect Checked-In Model Parameters** | `artifacts/checkpoints/` | JSON files | Read the metadata and `policy_state_path` string linking to PPO zip weights. |

---

## 1. Project Overview

The **Sheepdog Herding Lab** is a reinforcement learning workspace. Its primary goal is to simulate and train dog agents to collaborate on grouping and herding sheep into a predefined pen.

### Simulation Rules
* **Grid Environment**: The simulation operates on a logical 2D coordinate grid (by default, `80` cells wide and `60` cells high, configured in `EnvironmentConfig.width` / `EnvironmentConfig.height`).
* **Deterministic Transitions**: Transitions are fully deterministic. A specific seed sequence resolves positions, collision priorities, and sheep reactions exactly.
* **Success Condition**: All sheep must be inside the pen boundaries (`Pen.contains` returns `True` for all sheep) before the step limit is reached (`EnvironmentConfig.max_steps`) and before the no-progress termination guard kicks in.
* **No-Overlap Constraint**: Dogs and sheep occupy logical cells. Occupancy conflicts are resolved deterministically: lower-index agents retain priority to enter a contested cell, and blocked agents remain in place.

### Agent Paradigms
* **Sheep**: Fully scripted reactive agents that run away from dog pressure (`sheep_vision` controls panic range), cohere together (`flock_radius` and `sheep_flock_cohesion_weight`), and avoid boundary walls. They can also possess randomized personalities (obedient, pen-shy, pen-fearful, escapist, bold).
* **Dogs**: Cooperating agents controlled by either:
  1. A shared linear policy trained via hill climbing (`trained_policy`).
  2. A shared MLP neural policy trained via Maskable PPO (`neural_policy`).
  3. A hierarchical model (`shepherd_neural_dogs`) where a high-level command shepherd passes commands, and dogs choose grid moves.

---

## 2. Main Entry Points

These are the primary entry points and classes within the codebase:

### Start Training
* **CLI Entry**: `sheepdog-train` maps to `train_command()` in [src/sheepdog/cli.py](file:///c:/Users/inouy/source/sheep_dog/src/sheepdog/cli.py).
* **API Entry**: POST `/api/training/start` is handled by `ServerManager.start()` in [src/sheepdog/server.py](file:///c:/Users/inouy/source/sheep_dog/src/sheepdog/server.py).
* **Details**: Launches the configured trainer (e.g. `HillClimbTrainer` or `MaskablePPOTrainer`) to run a training run.

### Configure Training & Environment
* **File Path**: [src/sheepdog/config.py](file:///c:/Users/inouy/source/sheep_dog/src/sheepdog/config.py)
* **Classes**:
  * `EnvironmentConfig`: Stores grid height, width, agent count, speeds, and vision settings.
  * `RewardConfig`: Stores scale factors for distance progress, cohesion, time penalty, and stray weights.
  * `TrainingConfig`: Stores learning rates, rollout sizes, PPO hyperparameters, hidden sizes, and seed.
  * `LabConfig`: Bundles config dataclasses together for serialization.

### Environment Loop
* **File Path**: [src/sheepdog/environment.py](file:///c:/Users/inouy/source/sheep_dog/src/sheepdog/environment.py)
* **Class**: `SheepdogEnvironment`
* **Methods**:
  * `reset(seed)`: Seeds environment, sets up pen, places dogs and sheep.
  * `step(actions)`: Moves dogs sequentially by index, executes sheep movement physics, computes rewards, checks success/timeouts.

### Observation Construction
* **File Path**: [src/sheepdog/observations.py](file:///c:/Users/inouy/source/sheep_dog/src/sheepdog/observations.py)
* **Classes**:
  * `RoleAwareObservationBuilder`: Builds standard 54-feature vector for guided mode (incorporates roles like left flanker, blocker, pusher).
  * `EmergentObservationBuilder`: Builds 50-feature vector for emergent mode (no role labels or targets).
  * `HierarchicalObservationBuilder`: Builds 69-feature vector for hierarchical mode (adds shepherd commands + normalized IDs).

### Reward Calculation
* **File Path**: [src/sheepdog/rewards.py](file:///c:/Users/inouy/source/sheep_dog/src/sheepdog/rewards.py)
* **Classes**:
  * `RewardComputer`: Computes regular PPO/hill-climb step rewards.
  * `HierarchicalRewardComputer`: Computes rewards for the hierarchical neural training path.

### Replay Export & Rendering
* **File Path**: [src/sheepdog/replay/store.py](file:///c:/Users/inouy/source/sheep_dog/src/sheepdog/replay/store.py) and [web/src/components/FieldView.tsx](file:///c:/Users/inouy/source/sheep_dog/web/src/components/FieldView.tsx).
* **Explanation**: Replay data contains step snapshots exported as JSON. The React frontend reads this JSON and draws it on a canvas in `FieldView.tsx` with smooth transitions.

---

## 3. Neural Network / RL Model

The experimental neural policy path uses **Maskable PPO** (`MaskablePPO` from the `sb3_contrib` package) to train a shared MLP policy.

### Model Architecture
* **Algorithm**: Maskable PPO (Proximal Policy Optimization with invalid action masking).
* **Invalid Action Masking**: Handled by `SheepdogEnvironment.action_mask_for_dog`. Actions that would cause out-of-bounds movement, collisions, or forbidden wait steps are masked out (unselectable by PPO).
* **Hidden Layers**: 2 layers of size `128` (default configuration `neural_hidden_sizes: (128, 128)`).
* **Activation Function**: `tanh` (default in Stable-Baselines3's `MlpPolicy`).
* **Optimizers**: Adam optimizer.
* **Outputs**:
  * **Actor (Policy)**: 9 discrete logit scores (one for each grid direction action).
  * **Critic (Value Function)**: A single scalar state-value estimate $V(s)$.

### Network Parameter & Tensor Counts
The network consists of separate Actor and Critic networks with no shared parameters (outside of the identity feature extractor). We can compute the exact trainable parameter counts for each observation mode:

#### 1. Guided Mode (54 Input Features, 9 Actions)
* **Actor MLP**:
  * Layer 1 Weight: $54 \times 128 = 6,912$ | Bias: $128$
  * Layer 2 Weight: $128 \times 128 = 16,384$ | Bias: $128$
  * Output Weight: $128 \times 9 = 1,152$ | Bias: $9$
  * *Actor Total*: $24,713$ parameters
* **Critic MLP**:
  * Layer 1 Weight: $54 \times 128 = 6,912$ | Bias: $128$
  * Layer 2 Weight: $128 \times 128 = 16,384$ | Bias: $128$
  * Output Weight: $128 \times 1 = 128$ | Bias: $1$
  * *Critic Total*: $23,681$ parameters
* **Total Trainable Parameters**: **48,394**

#### 2. Emergent Mode (50 Input Features, 9 Actions)
* **Actor MLP**:
  * Layer 1 Weight: $50 \times 128 = 6,400$ | Bias: $128$
  * Layer 2 Weight: $128 \times 128 = 16,384$ | Bias: $128$
  * Output Weight: $128 \times 9 = 1,152$ | Bias: $9$
  * *Actor Total*: $24,201$ parameters
* **Critic MLP**:
  * Layer 1 Weight: $50 \times 128 = 6,400$ | Bias: $128$
  * Layer 2 Weight: $128 \times 128 = 16,384$ | Bias: $128$
  * Output Weight: $128 \times 1 = 128$ | Bias: $1$
  * *Critic Total*: $23,169$ parameters
* **Total Trainable Parameters**: **47,370**

#### 3. Hierarchical Mode (69 Input Features, 9 Actions)
* **Actor MLP**:
  * Layer 1 Weight: $69 \times 128 = 8,832$ | Bias: $128$
  * Layer 2 Weight: $128 \times 128 = 16,384$ | Bias: $128$
  * Output Weight: $128 \times 9 = 1,152$ | Bias: $9$
  * *Actor Total*: $26,633$ parameters
* **Critic MLP**:
  * Layer 1 Weight: $69 \times 128 = 8,832$ | Bias: $128$
  * Layer 2 Weight: $128 \times 128 = 16,384$ | Bias: $128$
  * Output Weight: $128 \times 1 = 128$ | Bias: $1$
  * *Critic Total*: $25,601$ parameters
* **Total Trainable Parameters**: **52,234**

### Training Settings
* **Gamma**: `0.99` (discount factor).
* **GAE Lambda**: `0.98` (generalized advantage estimation parameter).
* **Learning Rate**: `1e-4` starting, decayed linearly to `3e-5` (`learning_rate_final`).
* **Rollout Steps**: `2048` (rollout buffer collection size per update).
* **Batch Size**: `64` (minibatch size).
* **Entropy Coefficient**: `0.01` (incentivizes exploration).
* **Value Coefficient**: `0.5` (weight of value loss).

---

## 4. Hyperparameters Catalog

Below are the primary hyperparameters exposed in `src/sheepdog/config.py`.

| Hyperparameter | Default | Type | Class / Group | Controls | Safe Range | Immediate / Next Run | UI Visibility |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `width` | `80` | `int` | `EnvironmentConfig` | Logical grid width. | `20` - `200` | Next Run | Yes (Config tab) |
| `height` | `60` | `int` | `EnvironmentConfig` | Logical grid height. | `15` - `150` | Next Run | Yes (Config tab) |
| `dogs` | `3` | `int` | `EnvironmentConfig` | Number of dog agents. | `1` - `5` | Next Run | Yes (Config tab) |
| `sheep` | `6` | `int` | `EnvironmentConfig` | Number of sheep agents. | `1` - `12` | Next Run | Yes (Config tab) |
| `max_steps` | `600` | `int` | `EnvironmentConfig` | Step limit per episode. | `100` - `3000` | Immediate | Yes (Config tab) |
| `dog_vision` | `16` | `int` | `EnvironmentConfig` | Dog sensing radius. | `5` - `100` | Immediate | Yes (Config tab) |
| `sheep_vision` | `12` | `int` | `EnvironmentConfig` | Sheep dog-panic trigger radius. | `4` - `30` | Immediate | Yes (Config tab) |
| `flock_radius` | `10` | `int` | `EnvironmentConfig` | Radius for flock cohesion check. | `5` - `30` | Immediate | Yes (Config tab) |
| `dog_speed` | `1.0` | `float` | `EnvironmentConfig` | Dog step displacement speed. | `0.5` - `3.0` | Immediate | Yes (Config tab) |
| `sheep_speed` | `0.75` | `float` | `EnvironmentConfig` | Sheep step displacement speed. | `0.2` - `2.0` | Immediate | Yes (Config tab) |
| `dog_sprint_multiplier` | `2.0` | `float` | `EnvironmentConfig` | Speed multiplier for sprints. | `1.0` - `4.0` | Immediate | Yes (Config tab) |
| `sheep_flock_cohesion_weight` | `0.2` | `float` | `EnvironmentConfig` | Gravity attraction weight to flock center. | `0.0` - `1.0` | Immediate | Yes (Config tab) |
| `sheep_cohere_without_dog_pressure` | `True` | `bool` | `EnvironmentConfig` | Cohesion active without nearby dogs. | `True` / `False` | Immediate | Yes (Config tab) |
| `sheep_personality_strength` | `0.0` | `float` | `EnvironmentConfig` | Scalar multiplier for sheep traits. | `0.0` - `2.0` | Immediate | Yes (Config tab) |
| `role_minimum_hold_steps` | `8` | `int` | `EnvironmentConfig` | Minimum steps a dog must keep a role. | `1` - `30` | Immediate | Yes (Config tab) |
| `blocker_min_remaining_dogs` | `1` | `int` | `EnvironmentConfig` | Min herding dogs before blocker active. | `0` - `3` | Immediate | Yes (Config tab) |
| `progress_scale` | `2.0` | `float` | `RewardConfig` | Reward factor for pen progress. | `0.0` - `10.0` | Immediate | Yes (Config tab) |
| `sheep_penned_reward` | `8.0` | `float` | `RewardConfig` | Flat reward per penned sheep. | `0.0` - `50.0` | Immediate | Yes (Config tab) |
| `flock_cohesion_scale` | `0.35` | `float` | `RewardConfig` | Reward factor for grouping spread. | `0.0` - `2.0` | Immediate | Yes (Config tab) |
| `scatter_penalty_scale` | `0.2` | `float` | `RewardConfig` | Penalty factor for grouping scatter. | `0.0` - `2.0` | Immediate | Yes (Config tab) |
| `time_penalty` | `0.05` | `float` | `RewardConfig` | Negative step penalty. | `0.0` - `1.0` | Immediate | Yes (Config tab) |
| `no_progress_penalty` | `0.1` | `float` | `RewardConfig` | Negative step penalty on stalling. | `0.0` - `2.0` | Immediate | Yes (Config tab) |
| `terminal_success_reward` | `20.0` | `float` | `RewardConfig` | Large bonus when all sheep are penned. | `0.0` - `200.0`| Immediate | Yes (Config tab) |
| `terminal_failure_penalty` | `12.0` | `float` | `RewardConfig` | Penalty on timeout. | `0.0` - `100.0`| Immediate | Yes (Config tab) |
| `wait_penalty` | `0.05` | `float` | `RewardConfig` | Cost for wait actions. | `0.0` - `0.5` | Immediate | Yes (Config tab) |
| `sprint_cost_scale` | `0.12` | `float` | `RewardConfig` | Cost per sprint step. | `0.0` - `1.0` | Immediate | Yes (Config tab) |
| `farthest_sheep_progress_scale`| `0.0` | `float` | `RewardConfig` | Reward for drawing in strays. | `0.0` - `5.0` | Immediate | Yes (Config tab) |
| `stray_ignore_penalty_scale` | `0.0` | `float` | `RewardConfig` | Cost of keeping unpenned strays. | `0.0` - `1.0` | Immediate | Yes (Config tab) |
| `learning_rate` | `1e-4` | `float` | `TrainingConfig` | Starting neural PPO learning rate. | `1e-6` - `1e-3` | Next Run | Yes (Config tab) |
| `rollout_steps` | `2048` | `int` | `TrainingConfig` | Rollout buffer collection size. | `128` - `8192` | Next Run | Yes (Config tab) |
| `batch_size` | `64` | `int` | `TrainingConfig` | Optimization minibatch size. | `16` - `512` | Next Run | Yes (Config tab) |
| `invalid_action_masking` | `True` | `bool` | `TrainingConfig` | Enables invalid action masking. | `True` / `False` | Next Run | Yes (Config tab) |
| `observation_mode` | `"guided"`| `str` | `TrainingConfig` | Observation vector builder mode. | `"guided"`/`"emergent"` | Next Run | No (Internal) |

* **Increased Effect**: Increasing speed makes agents travel faster but may cause overpressure or overshoot. Increasing rewards highlights specific herding aspects.
* **Decreased Effect**: Decreasing limits may lead to timeouts. Decreasing learning rate stabilizes training but slows down convergence.

---

## 5. Observation Features / Input Nodes

Observations represent the sensory inputs received by the dog agent's policy network. They are scaled or normalized into the range $[-1.0, 1.0]$ based on the grid dimensions.

### Detailed Mappings by Mode

#### 1. Guided Mode (`observation_mode = "guided"`)
Features are built by `RoleAwareObservationBuilder.build()`. Total feature count is **54**.

| Index | Feature Name | Description | Normalization / Scaling |
| :--- | :--- | :--- | :--- |
| `0` | `own_x` | Dog's own absolute X position | `own_x / (width - 1)` |
| `1` | `own_y` | Dog's own absolute Y position | `own_y / (height - 1)` |
| `2` | `pen_x` | Pen center absolute X position | `pen_x / (width - 1)` |
| `3` | `pen_y` | Pen center absolute Y position | `pen_y / (height - 1)` |
| `4` | `flock_center_x` | Flock centroid absolute X coordinate | `flock_cx / (width - 1)` |
| `5` | `flock_center_y` | Flock centroid absolute Y coordinate | `flock_cy / (height - 1)` |
| `6` | `target_x` | Scripted target absolute X coordinate | `target_x / (width - 1)` |
| `7` | `target_y` | Scripted target absolute Y coordinate | `target_y / (height - 1)` |
| `8` | `distance_to_pen` | Dog distance to pen center | `dist / diagonal` |
| `9` | `distance_to_flock` | Dog distance to flock center | `dist / diagonal` |
| `10`| `distance_to_target`| Dog distance to assigned role target | `dist / diagonal` |
| `11`| `flock_spread` | Spread (standard dev) of sheep positions| `spread / diagonal` |
| `12`| `average_distance_to_pen` | Mean distance of all sheep to pen | `avg_dist / diagonal` |
| `13`| `wall_left` | Proximity to left wall boundary | `own_x / (width - 1)` |
| `14`| `wall_right` | Proximity to right wall boundary | `(width - 1 - own_x) / (width - 1)` |
| `15`| `wall_top` | Proximity to top wall boundary | `own_y / (height - 1)` |
| `16`| `wall_bottom` | Proximity to bottom wall boundary | `(height - 1 - own_y) / (height - 1)` |
| `17`| `blocked_steps` | Steps dog has been blocked (collision) | `min(blocked, 10) / 10.0` |
| `18`| `no_progress_steps` | Steps with no herd progress | `min(no_progress, window) / window` |
| `19`| `revisits_recent_position` | Binary indicator for position revisit | `1.0` if revisit loop, else `0.0` |
| `20`| `two_position_loop` | Binary indicator for 2-cell oscillation | `1.0` if oscillating, else `0.0` |
| `21`| `stray_present` | Binary indicator for stray sheep presence| `1.0` if stray exists, else `0.0` |
| `22`| `role_rear_pressure`| Dog is assigned `REAR_PRESSURE` | `1.0` if true, else `0.0` |
| `23`| `role_left_flanker` | Dog is assigned `LEFT_FLANKER` | `1.0` if true, else `0.0` |
| `24`| `role_right_flanker`| Dog is assigned `RIGHT_FLANKER` | `1.0` if true, else `0.0` |
| `25`| `role_collector` | Dog is assigned `COLLECTOR` | `1.0` if true, else `0.0` |
| `26`| `role_blocker` | Dog is assigned `BLOCKER` | `1.0` if true, else `0.0` |
| `27`| `focus_sheep_dx` | Relative X coordinate to closest unpenned sheep | `(focus_x - own_x) / (width - 1)` |
| `28`| `focus_sheep_dy` | Relative Y coordinate to closest unpenned sheep | `(focus_y - own_y) / (height - 1)` |
| `29`| `focus_sheep_distance` | Distance to focus sheep | `dist / diagonal` |
| `30`| `stray_sheep_dx` | Relative X coordinate to stray sheep | `(stray_x - own_x) / (width - 1)` |
| `31`| `stray_sheep_dy` | Relative Y coordinate to stray sheep | `(stray_y - own_y) / (height - 1)` |
| `32-49`| `sheep_0` to `sheep_5` | Slots for 6 closest sheep (relative X, relative Y, penned flag) | 6 slots, each taking 3 variables. |
| `50-53`| `other_dog_0` to `other_dog_1` | Slots for 2 other dogs (relative X, relative Y) | 2 slots, each taking 2 variables. |

#### 2. Emergent Mode (`observation_mode = "emergent"`)
Features are built by `EmergentObservationBuilder.build()`. Total feature count is **50**.

* Removes all scripted roles (`role_rear_pressure` to `role_blocker`) and target features (`target_x`, `target_y`, `distance_to_target`).
* Appends one-hot dog slot identity columns at indices `2-4`:
  * `dog_id_slot_0`: `1.0` if dog index is 0.
  * `dog_id_slot_1`: `1.0` if dog index is 1.
  * `dog_id_slot_2`: `1.0` if dog index is 2.
* Adds explicit closest unpenned features (`nearest_unpenned_dx`, `nearest_unpenned_dy`, `nearest_unpenned_distance`) and farthest unpenned features (`farthest_unpenned_dx`, `farthest_unpenned_dy`, `farthest_unpenned_distance`).

#### 3. Hierarchical Mode (Scripted Shepherd + Neural Dogs)
Features are built by `HierarchicalObservationBuilder.build_hierarchical()`. Total feature count is **69**.

* Starts with the standard **54** guided observation features.
* Appends **8** one-hot shepherd command nodes at indices `54-61`:
  * `shepherd_cmd_gather`, `shepherd_cmd_drive_to_pen`, `shepherd_cmd_hold_left`, `shepherd_cmd_hold_right`, `shepherd_cmd_block_escape`, `shepherd_cmd_apply_pressure`, `shepherd_cmd_back_off`, `shepherd_cmd_stop`.
* Appends **7** dog identity nodes at indices `62-68`:
  * `dog_id_normalized`: `dog_index / max(1, dog_count - 1)`
  * `dog_count_normalized`: `dog_count / max(1, MAX_DOG_SLOTS = 5)`
  * `dog_id_slot_0` to `dog_id_slot_4`: One-hot indicators for 5 possible dog slots.

---

## 6. Actions / Output Nodes

The dog action space is discrete, containing **9** distinct actions.

| Action ID | Action Name | Movement Offset (X, Y) | Type |
| :--- | :--- | :---: | :--- |
| `0` | `"up"` | $(0, -1)$ | Standard Step |
| `1` | `"down"` | $(0, 1)$ | Standard Step |
| `2` | `"left"` | $(-1, 0)$ | Standard Step |
| `3` | `"right"` | $(1, 0)$ | Standard Step |
| `4` | `"sprint_up"` | $(0, -2)$ | Sprint Step (Speed $\times$ Sprint Multiplier) |
| `5` | `"sprint_down"` | $(0, 2)$ | Sprint Step (Speed $\times$ Sprint Multiplier) |
| `6` | `"sprint_left"` | $(-2, 0)$ | Sprint Step (Speed $\times$ Sprint Multiplier) |
| `7` | `"sprint_right"`| $(2, 0)$ | Sprint Step (Speed $\times$ Sprint Multiplier) |
| `8` | `"wait"` | $(0, 0)$ | Hold Position |

### Action Resolution and Collision
1. **Budget-Based Movement**: The environment updates dog coordinates based on accumulated `movement_budget`. Standard steps consume 1 step from the budget. Sprint steps consume steps scaled by `dog_sprint_multiplier`.
2. **Priority Resolution**: Moving dogs are resolved in sequential index order (0 to $N-1$). When resolving dog $i$, its candidate next position is computed via `project_dog_action()`.
3. **Collisions**: A movement is blocked if the target cell is occupied by a sheep, an already-moved dog, or a dog that is holding its position. In this case, the dog remains in its original cell and its `blocked_steps` counter increases.

---

## 7. Dog Agent System

Dogs are modeled as state objects of type `DogState` containing:
* `index`: Unique dog slot ID.
* `position`: Grid coordinate `Point(x, y)`.
* `current_role`: Enum value of type `DogRole`.
* `last_action`: Last action string (e.g. `"wait"`, `"sprint_left"`).
* `blocked_steps`: Consecutive steps where movement was blocked.
* `movement_budget`: Fractional movement accumulator.
* `recent_positions`: Circular list tracking positions for oscillation checks.
* `steps_in_role`: Consecutive steps spent in the current role.

### Shared Policy Architecture
* **Single Policy Network**: Rather than unique weights per dog, a single policy network selects actions for all dogs. This is a shared-policy setup.
* **Observation Input**: The model receives observations matching the specific dog index, complete with role-specific target offsets (Guided) or one-hot identity indices (Emergent / Hierarchical).
* **Credit Assignment**:
  * In the **sequential baseline adapter** (`SheepdogRLAdapter`), dogs choose actions one at a time. The environment steps only after all dogs act, meaning intermediate steps return zero reward.
  * In the **joint hierarchical adapter** (`JointActionRLEnv`), actions are collected for all dogs before a single step, and the shared team reward is immediately returned.

---

## 8. Sheep Movement and Reaction System

Sheep behave as reactive state entities of type `SheepState`. Their movement is computed in `SheepdogEnvironment._move_sheep()` using a force-vector physics model.

### Movement Vector Calculation
1. **Idleness**: If panic steps $\le 0$, no dog is within `sheep_vision`, the sheep is near the flock center, and is away from walls, there is a **70% chance** the sheep idles (remains in place).
2. **Dog Pressure**: Flee vector pointing away from dogs within `sheep_vision` range:
   $$\vec{F}_{\text{dog}} = \sum_{d \in \text{Dogs}} \frac{\text{sheep\_vision} - \text{distance} + 1.0}{\text{distance}} \times (\vec{X}_{\text{sheep}} - \vec{X}_{\text{dog}})$$
   * *Bold Personality Modifier*: If the sheep is `bold` and the dog is further than 3 cells away, pressure is scaled down by $\max(0.0, 1.0 - 0.4 \times \text{strength})$.
3. **Flock Cohesion**: Attraction force toward the flock centroid:
   $$\vec{F}_{\text{cohesion}} = (\vec{X}_{\text{flock}} - \vec{X}_{\text{sheep}}) \times \text{cohesion\_weight}$$
   * Cohesion applies when dogs are nearby or when the sheep is far away (`distance > flock_radius`) and `sheep_cohere_without_dog_pressure = True`.
   * *Escapist Personality*: Panicked escapist sheep run *away* from the flock center: $\vec{F} \mathrel{{+}{=}} (\vec{X}_{\text{sheep}} - \vec{X}_{\text{flock}}) \times 0.35 \times \text{strength}$.
4. **Boundary Avoidance**: A repulsive vector is added when near borders to keep sheep on the field.
5. **Pen Repulsion**:
   * *Pen-Shy*: Constant mild push away from the pen center: $\vec{F} \mathrel{{-}{=}} \frac{\vec{X}_{\text{pen}} - \vec{X}_{\text{sheep}}}{\text{distance}} \times \text{strength}$.
   * *Pen-Fearful*: Proximity-scaled repulsion away from pen: scales up to 4x at entrance and fades smoothly with distance.
6. **Move Selection**: The candidate move (up, down, left, right) that maximizes the force-vector projection is chosen. Tied moves are broken randomly.
7. **Pen Retention**: Once a sheep enters the pen opening, `sheep.penned` is set to `True`, and it remains permanently locked inside (cannot exit or move).

---

## 9. Reward System

Total reward is computed in [src/sheepdog/rewards.py](file:///c:/Users/inouy/source/sheep_dog/src/sheepdog/rewards.py).

### Reward Terms (Sequential Path)
* **Pen Progress**: $(d_{t-1} - d_t) \times \text{progress\_scale}$ (encourages moving flock toward pen center).
* **Penned Sheep**: $N_{\text{newly\_penned}} \times \text{sheep\_penned\_reward}$ (incentivizes driving sheep in).
* **Flock Cohesion**: $(\text{spread}_{t-1} - \text{spread}_t) \times \text{flock\_cohesion\_scale}$ (incentivizes grouping sheep).
* **Scatter Penalty**: $\max(0, \text{spread}_t - \text{spread}_{t-1}) \times \text{scatter\_penalty\_scale}$ (penalizes flock dispersion).
* **Time Penalty**: $-\text{time\_penalty}$ per step (encourages quick resolution).
* **No Progress Penalty**: $-\text{no\_progress\_penalty}$ when stalled.
* **Wall Pressure Penalty**: $-\text{wall\_pressure\_penalty}$ if dog touches a boundary cell.
* **Wait Penalty**: $-\text{wait\_penalty}$ if dog waits without reason.
* **Sprint Cost**: $-N_{\text{sprints}} \times \text{sprint\_cost\_scale}$ (penalizes excessive sprinting).
* **Gate Progress**: Rewards moving sheep through the pen gate corridor.
* **Terminal Success**: Large bonus ($\text{terminal\_success\_reward}$) when all sheep are penned.
* **Terminal Failure**: Penalty ($-\text{terminal\_failure\_penalty}$) on timeout.

### Stray Terms (Emergent Path)
* **Farthest Progress**: $(f_{t-1} - f_t) \times \text{farthest\_sheep\_progress\_scale}$ (rewards drawing in farthest stray).
* **Stray Ignore Penalty**: $-\text{farthest\_dist} \times \text{stray\_ignore\_penalty\_scale}$ (penalizes letting a stray remain isolated). Contains approach rewards and progress bonuses for isolated strays.

### Hierarchical Path Reward Terms
Computed by `HierarchicalRewardComputer`:
* **Sheep Closer to Pen**: $+2.5$ scale / **Sheep Away from Pen**: $-0.5$ scale.
* **Flock Grouped**: $+0.4$ scale / **Scatter**: $-0.6$ scale.
* **Pressure from Behind**: $+0.8$ bonus when dogs sit opposite to pen relative to flock center.
* **Dog Spread**: $+0.3$ bonus for spreading out, and $-0.5$ penalty if stacked together (`dog_stack_distance < 2.0`).
* **Dog Blocking Escape**: $+0.2$ bonus for plugging escape lanes.
* **Overpressure Penalty**: $-1.2$ penalty if dog distance to sheep is $< 1.5$.
* **Gate Blocking**: Up to $-1.0$ penalty if a non-blocker dog blocks the gate corridor.
* **Task Completion**: $+25.0$ / **Timeout**: $-15.0$ / **Wandering Penalty**: $-0.04$ per step.

---

## 10. Curriculum / Stages

The project defines **38** distinct herding curriculum stages in [src/sheepdog/curriculum.py](file:///c:/Users/inouy/source/sheep_dog/src/sheepdog/curriculum.py).

### Curriculum Progression
* **Early Stages (1-5)**: Small maps (60x45 to 84x60), 1-2 dogs, 1-3 sheep, obedient personality only, corner pen placement, simple spawn layout mix (`fixed_easy` or `randomized_flock`).
* **Middle Stages (6-20)**: Map dimensions scale up to 120x90. Dog count increases to 3, sheep count to 5. Stray layouts introduced (`nearby_stray`, `farther_stray`). Minor personality strength added.
* **Late Stages (21-38)**: Map dimensions up to 144x104. 3 dogs, 6 sheep. Pen placements generalize (`same_wall`, `any_wall`, `interior`, `random`). Personality strength climbs to `1.25` (incorporating escapists, bold, pen-shy, pen-fearful sheep). Flock cohesion attraction is weakened to force dog pressure to matter.

### Promotion and Checks
* **Resumption**: Trainer signatures are saved in `training-state.json`. If a new run has a matching signature, training resumes using the saved weights. If config signatures differ, the state is cleared and training starts fresh.
* **Auto-Promotion**: During training, checkpoints are evaluated. If success rate exceeds promotion thresholds, the curriculum automatically moves to the next stage, keeping the learned network parameters intact.

---

## 11. Training Loop

Below is the execution flow of the reinforcement learning training loop (specifically for Maskable PPO):

```
       +---------------------------------------------+
       |             Start Episode                   |
       +---------------------------------------------+
                             |
                             v
       +---------------------------------------------+
       |   Sample/Init Grid, Pen, Dogs, & Sheep      |
       +---------------------------------------------+
                             |
                             v
+----->|       Iterate Dogs Sequentially 0..N-1      |
|      +---------------------------------------------+
|                            |
|                            v
|      +---------------------------------------------+
|      | Build Observation & Fetch Legal Action Mask |
|      +---------------------------------------------+
|                            |
|                            v
|      +---------------------------------------------+
|      | Predict Action (using MaskablePPO MLP)      |
|      +---------------------------------------------+
|                            |
|                            v
|      +---------------------------------------------+
|      | Accumulate Action into Team Buffer          |
|      +---------------------------------------------+
|                            |
|                            +------------------+
|                                               |
|                                     Has every dog acted?
|                                               |
|                                       No      Yes
|                                       |       |
+---------------------------------------+       v
                                       +------------------+
                                       | Step Environment |
                                       +------------------+
                                                |
                                                v
                                       +------------------+
                                       | Move Dogs & Sheep|
                                       +------------------+
                                                |
                                                v
                                       +------------------+
                                       | Compute Rewards  |
                                       +------------------+
                                                |
                                                v
                                       +------------------+
                                       | Check Terminate? |
                                       +------------------+
                                                |
                                        No      Yes
                                        |       |
                                        +       v
                                                |
                                       +------------------+
                                       | Update NN (PPO)  |
                                       +------------------+
                                                |
                                                v
                                       +------------------+
                                       | Save Checkpoint  |
                                       +------------------+
```

---

## 12. Replay / Watch / Results

### Replay Data Format
Steps are stored as a JSON array of `StepRecord` elements containing:
* `step`: Integer index.
* `actions`: Array of string actions committed.
* `snapshot`: `EnvironmentSnapshot` representing positions of all agents.
* `reward`: Dict breakdown of all reward components.

### Checkpoint vs Live
* **Live Replay**: Replayed dynamically from the API server `/api/replay/run` based on current configuration and model weights.
* **Evaluation Replays**: Saved statically under `artifacts/evaluations/replays/checkpoint-NNNNNN-seed-SSSSSS.json` during training evaluation checkpoints.

---

## 13. Scenario System

A scenario represents a fixed, reproducible environment setup. Scenarios are defined using the `Scenario` class in [src/sheepdog/evaluation/scenarios.py](file:///c:/Users/inouy/source/sheep_dog/src/sheepdog/evaluation/scenarios.py).

### Scenario Layout Fields
* `id`: Unique UUID string.
* `name`: Display name.
* `seed`: Random seed.
* `width` / `height`: Grid dimensions.
* `dogs` / `sheep`: Arrays of `AgentLayout` objects (index, x, y coordinates, personality).
* `pen`: `PenLayout` (origin, dimensions, opening side).
* `description`: Text notes.

### Execution
The user can save a scenario via the Scenarios tab. Replays or evaluations can be triggered on saved scenarios to test specific layouts (e.g., scattered corner sheep or blocked pens).

---

## 14. UI / Frontend Map

The web application is located in `web/` and built using React and TypeScript.

### Interactive Grid Views
* **Train Tab (`train`)** [web/src/components/TrainingPanel.tsx](file:///c:/Users/inouy/source/sheep_dog/web/src/components/TrainingPanel.tsx)
  * Displays: Live training logs, PPO parameters, episode counts, success charts, auto-promote toggle.
  * Endpoints: POST `/api/training/start`, POST `/api/training/pause`, POST `/api/training/stop`.
* **Watch Tab (`watch`)** [web/src/components/StatusPanel.tsx](file:///c:/Users/inouy/source/sheep_dog/web/src/components/StatusPanel.tsx)
  * Displays: Step-by-step canvas animation of dogs/sheep, playback speed, checkpoint selectors, active role labels.
  * Endpoints: GET `/api/config`, POST `/api/replay/run`.
* **Scenarios Tab (`test`)** [web/src/components/SavedScenariosPanel.tsx](file:///c:/Users/inouy/source/sheep_dog/web/src/components/SavedScenariosPanel.tsx)
  * Displays: Saved layouts library, evaluate scenario buttons, create custom start layouts.
  * Endpoints: GET `/api/scenarios`, POST `/api/scenarios`.

### Fullscreen Informational Tabs
* **Stages Tab (`stages`)** [web/src/components/StagesTab.tsx](file:///c:/Users/inouy/source/sheep_dog/web/src/components/StagesTab.tsx)
  * Displays: Grid of all 38 curriculum stages, active stage marker, difficulty progressions.
* **Network Tab (`network`)** [web/src/components/NetworkTab.tsx](file:///c:/Users/inouy/source/sheep_dog/web/src/components/NetworkTab.tsx)
  * Displays: Real-time weight parameters, neuron connection charts, bias distributions.
  * Endpoints: GET `/api/network/topology`.
* **Layers Tab (`layers`)** [web/src/components/LayersTab.tsx](file:///c:/Users/inouy/source/sheep_dog/web/src/components/LayersTab.tsx)
  * Displays: Node-by-node listings for Guided, Hierarchical, and Emergent modes.
* **History Tab (`history`)** [web/src/components/HistoryTab.tsx](file:///c:/Users/inouy/source/sheep_dog/web/src/components/HistoryTab.tsx)
  * Displays: Revisions of configurations and parameters.
* **Insights Tab (`insights`)** [web/src/components/DiagnosticsPanel.tsx](file:///c:/Users/inouy/source/sheep_dog/web/src/components/DiagnosticsPanel.tsx)
  * Displays: Deep training metrics, loss functions, reward component breakdown charts.
  * Endpoints: GET `/api/training/history`.
* **Results Tab (`results`)** [web/src/components/ResultsPanel.tsx](file:///c:/Users/inouy/source/sheep_dog/web/src/components/ResultsPanel.tsx)
  * Displays: Tabulated success rates, average steps, and completion metrics across checkpoints.

---

## 15. Backend / API Map

The backend is a custom python server utilizing `http.server.BaseHTTPRequestHandler` in [src/sheepdog/server.py](file:///c:/Users/inouy/source/sheep_dog/src/sheepdog/server.py).

### GET Endpoints
* `/api/health`: Returns `{"ok": true}`.
* `/api/config`: Returns active `LabConfig` parameters dict.
* `/api/training/history`: Returns logs list from `artifacts/training_history.json`.
* `/api/training/status`: Returns current server manager state.
* `/api/network/topology`: Returns shapes and parameter names of model layers.
* `/api/scenarios`: Returns list of saved scenarios.

### POST Endpoints
* `/api/training/start`: Payload `{ episodes: int, fast_mode: bool, ... }`. Starts a training batch.
* `/api/training/pause`: Pauses training execution.
* `/api/training/stop`: Terminates training and updates logs.
* `/api/training/clear`: Deletes model weights, resets training state back to episode 0.
* `/api/replay/run`: Payload `{ seed: int, policy_mode: str, ... }`. Runs a simulation step sequence.

---

## 16. Data Files / Saved State

All persisted outputs reside in `artifacts/`.

* **`effective-training-config.json`**: Active configuration copy. Created by training start. Read by evaluation processes.
* **`training-state.json`**: Main hill-climbing state tracker (contains `total_episodes_trained`, `weights`, `best_score`). If deleted, training starts fresh.
* **`hierarchical-training-state.json`**: State tracker for hierarchical PPO runs.
* **`training-summary.json`**: Compiled logs, success rates, and metrics.
* **`checkpoints/checkpoint-NNNNNN.json`**: JSON files with checkpoint details and policy paths.
* **`models/maskable-ppo-XXXXXXXX.zip`**: persistent PPO neural network weights.
* **`evaluations/replays/checkpoint-NNNNNN-seed-SSSSSS.json`**: Saved replay step records.

---

## 17. Tests

The project includes a Python regression test suite in `tests/`.

* **`test_curriculum.py`**: Validates sorting and stage progression parameters.
* **`test_emergent_ppo.py`**: Verifies emergent observation builder shape and stray rewards.
* **`test_environment.py`**: Verifies dog/sheep collision physics, wall avoidance, and logical steps.
* **`test_hierarchical_obs.py`**: Verifies hierarchical shepherd command one-hot and identity variables.
* **`test_rewards.py`**: Validates reward components and shapes.
* **`test_scenarios.py`**: Verifies scenario serialization.
* **`test_server.py`**: Tests http GET/POST endpoints.

> [!NOTE]
> All 319 core unit tests pass successfully. The slow benchmark tests (`test_stage1_benchmark.py`) can be bypassed for quick verification using `.venv/Scripts/pytest -k "not test_stage1_benchmark"`.

---

## 18. Known Issues / Things to Verify

* **Windows Spawn Deadlock**: Python 3.13 on Windows can deadlock when vectorized environments spawn SubprocVecEnv workers. To prevent this, the codebase automatically overrides worker count to `1` on Windows with Python 3.13 (`src/sheepdog/policies/neural.py:32`).
* **Config History File Size**: `artifacts/config-history.json` grows large over time due to frequent UI writes. It is safe to clear or truncate if storage becomes a concern.

---

## 19. Glossary

* **Reinforcement Learning (RL)**: Machine learning paradigm where agents learn to act to maximize cumulative rewards.
* **PPO (Proximal Policy Optimization)**: Policy gradient method using a clipped surrogate objective to prevent destabilizing updates.
* **Maskable PPO**: Extension of PPO filtering out illegal actions.
* **Policy**: The network mapping observations to action distributions.
* **Observation**: Sensory input vector scaled to $[-1.0, 1.0]$.
* **Action Space**: Valid actions (the 9 directions).
* **Action Mask**: Boolean array indicating valid directions at each step.
* **Reward Shaping**: Intermediate rewards (cohesion, progress) to guide exploration.
* **Flock Radius**: Distance around centroid defining flock boundary.
* **Sheep Vision**: Range where sheep detect dogs and panic.
* **Dog Vision**: Range where dogs detect entities.
* **Instinct**: Scripted force vectors or shaping rewards mimicking biological herding.
* **Overpressure**: Penalty when dog gets too close ($< 1.5$ cells) to a sheep.
* **Guided Observation**: Observation containing scripted role targets and labels.
* **Emergent Observation**: Observation with roles stripped, relying on raw spatial data and slot IDs.

---

## 20. Diagrams

### Observation-Network-Action Loop
```
       +---------------------------------------------+
       |            Environment Grid                 |
       +---------------------------------------------+
             |                                 ^
      (Build Observation)                (Step & Apply)
             |                                 |
             v                                 |
       +-----------+                     +-----------+
       | 54/50/69  |                     | Chosen    |
       | Inputs    |                     | Action    |
       +-----------+                     +-----------+
             |                                 ^
             v                                 |
       +---------------------------------------------+
       |            Neural Network (PPO)             |
       |  Input -> Linear -> Tanh -> Linear -> Logits|
       +---------------------------------------------+
```

### Shepherd-Dog Interaction Flow (Hierarchical Mode)
```
       +---------------------------------------------+
       |             Scripted Shepherd               |
       |  (Computes high-level command from flock)   |
       +---------------------------------------------+
                             |
                  (Command one-hot label)
                             |
                             v
       +---------------------------------------------+
       |                 Neural Dog                  |
       | (Predicts grid movement move using command) |
       +---------------------------------------------+
                             |
                       (Grid Move)
                             |
                             v
       +---------------------------------------------+
       |            Environment Physics              |
       |  (Resolves collisions & moves sheep)        |
       +---------------------------------------------+
```

---

## 21. Final Summary

### Top 10 Files in the Project
1. [src/sheepdog/environment.py](file:///c:/Users/inouy/source/sheep_dog/src/sheepdog/environment.py): Simulation logic, actions, and stepping physics.
2. [src/sheepdog/config.py](file:///c:/Users/inouy/source/sheep_dog/src/sheepdog/config.py): Hyperparameters and defaults.
3. [src/sheepdog/observations.py](file:///c:/Users/inouy/source/sheep_dog/src/sheepdog/observations.py): Feature builder for network inputs.
4. [src/sheepdog/rewards.py](file:///c:/Users/inouy/source/sheep_dog/src/sheepdog/rewards.py): Reward scoring calculations.
5. [src/sheepdog/curriculum.py](file:///c:/Users/inouy/source/sheep_dog/src/sheepdog/curriculum.py): Stage definitions.
6. [src/sheepdog/team_strategy.py](file:///c:/Users/inouy/source/sheep_dog/src/sheepdog/team_strategy.py): Scripted role and target assignment rules.
7. [src/sheepdog/shepherd.py](file:///c:/Users/inouy/source/sheep_dog/src/sheepdog/shepherd.py): High-level scripted shepherd commands.
8. [src/sheepdog/training/maskable_ppo.py](file:///c:/Users/inouy/source/sheep_dog/src/sheepdog/training/maskable_ppo.py): Maskable PPO training wrapper.
9. [src/sheepdog/policies/neural.py](file:///c:/Users/inouy/source/sheep_dog/src/sheepdog/policies/neural.py): Neural network execution wrapper.
10. [web/src/App.tsx](file:///c:/Users/inouy/source/sheep_dog/web/src/App.tsx): React frontend router.

### Safe Places to Experiment
* **Reward scale overrides**: Modifying parameters in the Config tab or `RewardConfig` changes training incentives without breaking weights.
* **Stage mixes & layouts**: Modifying `CURRICULUM_STAGES` allows trying new maps or layouts immediately.

### Risky Places to Modify
* **Adding/removing observations**: Modifying `observations.py` changes layer shape, causing shape mismatches with checkpoints.
* **Altering environment coordinates**: Shrinking grid bounds below Pen bounds will cause out-of-bounds initialization errors.
