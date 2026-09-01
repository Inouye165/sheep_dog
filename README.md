# Sheepdog Herding Lab

Sheepdog Herding Lab is a deterministic sheepdog-herding simulation where a team of dogs learns to herd multiple sheep into a pen.

The project is split into two parts:

- A Python simulation, training, and evaluation pipeline.
- A React + TypeScript replay viewer that reads exported checkpoint data.

The current baseline is intentionally honest about what it does. It trains a role-aware shared linear policy with hill climbing, measures real checkpoint results, and exports replays that the UI can inspect.

This repository now also includes an experimental comparison path: a small role-aware neural policy trained with MaskablePPO. The baseline is still present and still first-class. The point of this round is comparison and benchmarking, not replacing the hill-climbing path.

## Why This Exists

The goal is to make herd behavior visible, measurable, and comparable across checkpoints without hiding the actual results behind demo logic or hand-waved metrics.

## Quick Start

1. Create a Python environment and install the backend package in editable mode with dev tools.
2. Start the backend and web viewer from PowerShell.
3. Open the UI and train or run the current dogs.
4. Run the training pipeline when you want exported checkpoints and replay files.

## Start From PowerShell

From the repository root in PowerShell, run:

```powershell
.\start-app.ps1
```

That launcher will:

- install the backend in editable mode if needed
- install the web dependencies if `web/node_modules` is missing
- start the Python API server at `http://127.0.0.1:8000`
- start the Vite viewer at `http://127.0.0.1:5173`
- write process ids and startup log paths to `artifacts/startup/pids.json`

To stop both services:

```powershell
.\stop-app.ps1
```

If training is active, `stop-app.ps1` now asks the backend to persist the last complete checkpoint before it shuts the processes down. The next `start-app.ps1` run will prompt you to resume a paused or stopped training session if one is available. If the app was interrupted unexpectedly during a running batch, startup now auto-resumes from the last persisted safe point instead.

To restart both services:

```powershell
.\restart-app.ps1
```

If you want to start the services manually from PowerShell instead of using the launcher:

```powershell
python -m pip install -e .[dev]
python -m sheepdog.server
cd web
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```

Example commands:

```powershell
python -m pip install -e .[dev]
.\start-app.ps1
python -m sheepdog.cli train
python -m sheepdog.cli export-demo
```

To run the experimental PPO path, install the RL extras once:

```powershell
python -m pip install -e .[dev,rl]
```

## Repo Structure

- `src/sheepdog/` Python simulation, policies, training, evaluation, checkpointing, and replay export.
- `tests/` Python regression tests.
- `web/` React + TypeScript UI for replay playback and checkpoint comparison.
- `docs/architecture.md` High-level architecture notes.
- [`docs/rl-evaluation-roadmap.md`](docs/rl-evaluation-roadmap.md) RL experiment results, decisions, and prioritized improvement queue.
- `artifacts/` Generated training and evaluation output.

## Architecture Overview

The simulation is grid based and deterministic for a fixed seed. Dogs act first, sheep react to nearby pressure, and the environment tracks progress toward the pen. Rewards are decomposed into named pieces so the total can be audited.

The trainer uses hill climbing to optimize a simple shared linear policy. Hill climbing is the training algorithm, not the model. The baseline is not PPO, not MaskablePPO, and not a neural network. Checkpoints are real evaluation snapshots, not fabricated milestones.

The current baseline adds dynamic scripted team roles on top of that same hill-climbing trainer. Dogs still share one linear weight vector, but each step the team strategy assigns tactical jobs such as rear pressure, flanking, collecting strays, and blocking near the pen, then the role-aware linear policy scores actions against those temporary responsibilities.

The experimental path keeps that scripted role assignment and swaps only the model and trainer pieces:

- baseline model: role-aware linear policy
- baseline trainer: hill climbing
- experimental model: small neural policy
- experimental trainer: MaskablePPO

The role system is still scripted in this phase. The neural policy does not learn role assignment from scratch yet.

The web app is a viewer, not a second simulation engine. It loads exported checkpoint and replay JSON files from `web/public/generated/` and plays them back frame by frame.

## Policy Modes

- `random_untrained` picks legal moves uniformly and has no herding intelligence.
- `instinct_only` can chase, circle, avoid diving into the flock, and recover nearby sheep, but it does not know where the pen is.
- `heuristic_expert` is a scripted, pen-aware expert that uses pressure positioning behind the flock relative to the target.
- `trained_policy` uses learned linear-policy weights from hill-climb training checkpoints.
- `neural_policy` uses a learned small neural-network policy from MaskablePPO checkpoints.

By default the no-training playback path uses `instinct_only`, not the expert heuristic. Pen-directed behavior now requires training, `heuristic_expert`, or an explicit handler target command.
us

Action masking keeps illegal movement out of policy choices. For heuristic-style modes, `wait` is also gated by the same tactical scoring threshold used for movement decisions. For neural/RL modes such as `neural_policy`, `shepherd_neural_dogs`, and unspecified policy mode inside RL wrappers, `wait` remains legal even when a movement action scores better, so MaskablePPO can learn hold-position behavior instead of having it masked away.

## Simulation Concepts

- Dogs are AI-controlled and share one role-aware policy implementation.
- Dogs can switch dynamic tactical roles each step while still using the same shared policy instance.
- Sheep flee from nearby dogs, stay loosely flocked, and avoid walls.
- The pen is a goal area inside the field.
- Episodes succeed only when every sheep is penned.
- The environment stops on success, timeout, or no-progress conditions.

## Movement and Occupancy Rules

- The logical field now runs on a significantly denser grid than the original release, so agents can reposition in smaller increments without visually shrinking to dots.
- The default field now renders at one fixed world size instead of zooming to the current group positions.
- Dogs move two logical cells per simulation step by default.
- Sheep move one logical cell per simulation step.
- Curriculum stages can still override dog speed for slower early training scenarios.
- The viewer keeps icons roughly the same apparent size by scaling the rendered agent markers relative to grid density instead of tying icon size to one logical cell.
- Dog-dog overlap is disallowed.
- Dog-sheep overlap is disallowed.
- Sheep-sheep overlap is also prevented in normal play.
- If multiple agents want the same cell, resolution is deterministic: lower-index agents keep priority and blocked agents remain in place for that step.
- Sheep add a small seed-driven tie-break jitter to local movement scoring so wall escapes and local minima do not always collapse to the same deterministic axis choice.
- When a sheep stays pinned near a wall or pen edge and progress stalls, the dog scorer boosts flank/lateral repositioning and penalizes repeated bounce loops more strongly.
- The pen now sits flush against the right wall so there is no dead strip behind it where sheep can get trapped.

## Training

The project now supports two clean trainer/model combinations:

- hill-climbing trainer plus role-aware linear policy baseline
- MaskablePPO trainer plus role-aware neural policy experiment
- hierarchical MaskablePPO trainer plus shepherd-level neural dog coordination experiment

Hill climbing and MaskablePPO are trainers, not models. Linear and neural are models, not trainers.

Run training with:

```powershell
python -m sheepdog.cli train
```

This writes checkpoint JSON files under `artifacts/checkpoints/`, evaluation JSON and CSV files under `artifacts/evaluations/`, and web assets under `web/public/generated/`.

Baseline hill-climb training:

```powershell
python -m sheepdog.cli train --trainer-type hill_climb --policy-type linear --output-dir artifacts/hill_climb
```

Experimental MaskablePPO training:

```powershell
python -m sheepdog.cli train --trainer-type maskable_ppo --policy-type neural --total-timesteps 20000 --output-dir artifacts/maskable_ppo
```

Using separate output roots is the recommended comparison workflow so the latest linear and neural artifacts do not overwrite each other.

## Checkpoints

Default checkpoint episodes:

`0, 5, 10, 25, 50, 100, 500, 1000`

Each checkpoint includes:

- checkpoint episode
- total training episodes so far
- policy name
- trainer type and policy type
- policy weights for linear checkpoints or model path plus policy config for neural checkpoints
- environment configuration
- reward configuration
- success rate over evaluation seeds
- average completion steps and seconds
- timeout rate
- average sheep penned
- average reward
- replay path for the exported run

## Evaluation

Evaluation uses fixed seeds so checkpoints are compared fairly.

The evaluator writes:

- JSON summary files
- CSV per-seed records
- replay JSON files for each evaluated seed

Those files live under `artifacts/evaluations/` and are copied to the web public folder for playback.

Evaluate fixed seeds for one policy mode:

```powershell
python -m sheepdog.cli evaluate --policy trained_policy --seeds 11 23 37
python -m sheepdog.cli evaluate --policy neural_policy --config path\to\ppo-config.json --seeds 11 23 37
```

Compare random, instinct, heuristic, hill-climb linear, and MaskablePPO neural policies on the same seeds:

```powershell
python -m sheepdog.cli benchmark --seeds 11 23 37 41 53 --linear-output-dir artifacts/hill_climb --neural-output-dir artifacts/maskable_ppo
```

The benchmark command writes:

- machine-readable JSON
- machine-readable CSV
- human-readable Markdown summary

by default under `artifacts/benchmarks/`.

## Herding Instincts and Curriculum

Real sheepdogs start with herding instincts; the agents in this lab do not.
To narrow that gap without scripting the answer, the project adds optional
reward shaping that simulates instinct, plus a curriculum that starts simple
and grows harder. The dog still selects its own actions through the existing
policy; instincts only shape the reward signal and the training scenario.

Training reward shaping may still use target progress toward the pen. That is separate from action selection: untrained action choice no longer gets a hidden pen-aware controller through the default playback policy.

Configurable instinct reward terms (`InstinctRewardConfig` in
`src/sheepdog/config.py`):

- `pressure_zone_weight` – reward being on the opposite side of the flock from the pen.
- `safe_pressure_weight` – reward staying in a useful distance band from the flock.
- `grouping_weight` – reward decreasing flock spread.
- `target_progress_weight` – reward the flock centroid moving toward the pen.
- `chaos_penalty_weight` – penalize entering the flock or causing sudden scatter.
- `overpressure_penalty_weight` – penalize crowding individual sheep.
- `split_flock_penalty_weight` – penalize one sheep straying far from the rest.

Toggles:

- `enable_instinct_rewards` – off by default. Turn on to add the instinct terms to training.
- `debug_reward_breakdown` – flag for downstream tooling to surface per-term values.
- `curriculum_stage` – non-zero to apply a curriculum-stage environment override.

Policy config (`PolicyConfig` in `src/sheepdog/config.py`):

- `policy_mode` – select `random_untrained`, `instinct_only`, `heuristic_expert`, or `trained_policy`.
- `allow_instinct_target_awareness` – off by default. Leave disabled unless you explicitly want instinct mode to read a target.
- `handler_target_enabled` – off by default. Reserved for explicit handler-driven pen targeting.

Training config (`TrainingConfig` in `src/sheepdog/config.py`):

- `candidate_evaluation_seeds` – fixed seeds used to score hill-climber candidates before promotion.
- `candidate_pool_size` – number of mutated candidate policies considered each training episode.
- `mutation_scale` – per-weight mutation magnitude for the hill climber.

Environment display config (`EnvironmentConfig` in `src/sheepdog/config.py`):

- `sheep_personality_colors` – mapping of personality name → hex color used to color-code sheep in the replay viewer. Keys must match entries in `entities.SHEEP_PERSONALITIES` (`obedient`, `pen_fearful`, `pen_shy`, `escapist`, `bold`). When `sheep_personality_strength` is `0.0` every sheep is `obedient` and therefore shares the `obedient` color – raise the strength (e.g. `0.2`) to see distinct colors and behaviors.

Curriculum stages (`CURRICULUM_STAGES` in `src/sheepdog/curriculum.py`):

1. One dog, one sheep, dense open field, one-cell motion, nearby pen.
2. One dog, small flock, dense-grid grouping in an open field.
3. One dog, small flock, larger dense field for sustained drive/fetch.
4. Two dogs, medium flock, dense-grid pressure control near guarded pens.
5. Three dogs, larger flock, dense-grid cooperation without jump movement.

Reward breakdown is always returned per step on `RewardBreakdown`, so the
viewer or any debugging tool can inspect contributions term by term without
log spam. To compare shaped vs. unshaped training, run with
`enable_instinct_rewards=False` (the default) and again with it set to
`True`.

The UI training path sends `enable_instinct_rewards`, `curriculum_stage`,
and `debug_reward_breakdown` to the backend API server. By default the Train panel starts
new runs with instinct rewards enabled and curriculum stage 1 unless you
override them.

Compatibility warning: older checkpoints trained without instinct rewards can
work against the new pressure-position curriculum. Do not expect a 1100-episode
run from the old reward mix to improve just because the viewer changed. Clear
existing training data before starting a fresh instincts curriculum run, but do
that manually so you can keep old artifacts if you still want them for
comparison.

Recommended first run:

- clear training data
- enable instincts
- use curriculum stage 1
- keep the stage-1 one-cell dog speed defaults while the policy is learning pressure control

Recommended progression:

- stage 1 until the dog reliably holds pressure behind a single sheep
- stage 2 until grouping three sheep looks stable
- stage 3 for longer driving and fetch behavior in a larger field
- stage 4 and stage 5 only after pressure positioning stays stable with one dog

Realism note: the denser logical grid is doing the main visual work here. Instead of letting dogs jump 2-3 cells in one update, the simulation now advances with smaller logical steps and more total steps. That improves wall interactions, makes flanking space available, and gives the UI enough intermediate positions for smoother playback.

## Scenario-Based Training

Scenario-based training mixes difficult starting situations with normal random starts to make the trained policy more robust by regularly exposing it to edge-case recovery situations during training.

### Why It Helps

Normal random starts can leave the policy undertrained on difficult recovery scenarios like:
- **Scattered sheep**: Sheep spread widely across the field, requiring dispersed flock recovery
- **Split flock**: Sheep separated into two groups, requiring regrouping behavior
- **Corner huddle**: Sheep clustered in a corner away from the pen, requiring extraction and movement

By regularly training on these edge cases, the policy learns robust recovery strategies that generalize better to unseen situations.

### How to Enable

Add scenario training configuration to your training config:

```json
{
  "training": {
    "scenario_training_enabled": true,
    "scenario_mix": {
      "random": 0.50,
      "scattered_sheep": 0.20,
      "split_flock": 0.15,
      "corner_huddle": 0.15
    }
  }
}
```

The `scenario_mix` weights determine the probability of each scenario type at episode reset. Weights must sum to 1.0.

### Important Warning

**Always mix scenario training with random starts.** Do not set the random weight to 0.0. Training exclusively on difficult scenarios can cause overfitting to those specific patterns and degrade performance on normal situations. A mix of 50% random with 50% difficult scenarios is a good starting point.

### Reproducibility

Scenario selection uses the existing RNG/seed system. Two training runs with the same seed produce the same sequence of scenario choices, ensuring reproducible training.

### Observability

The scenario sampler tracks usage statistics. After training, you can query the scenario usage summary to see how many episodes started from each scenario type. This helps verify that the configured mix is being applied correctly.

### Evaluation

You can evaluate a trained policy against each scenario type separately to identify performance differences:

```python
from sheepdog.training.scenario_evaluator import evaluate_policy_on_scenario_types

results = evaluate_policy_on_scenario_types(
    policy,
    config,
    evaluation_seeds=(11, 23, 37, 41, 53),
    scenario_types=("random", "scattered_sheep", "split_flock", "corner_huddle"),
)
```

This returns per-scenario-type metrics (success rate, average reward, steps, etc.) so you can see if the policy performs differently on edge cases compared with normal random starts.

### Available Scenario Types

- **random**: Normal randomized starting positions (default behavior)
- **scattered_sheep**: Sheep spread widely across the field
- **split_flock**: Sheep separated into two groups
- **corner_huddle**: Sheep clustered in a corner away from the pen

All scenarios use relative placement where possible to work with different grid sizes.

## UI

Start the web viewer with:

```powershell
cd web
npm install
npm run dev
```

The UI shows:

- the field, dogs, sheep, and pen
- color-coded sheep markers with a sheep legend
- compact dog role labels during replay
- the selected checkpoint and seed
- policy mode
- run state
- sheep penned count
- elapsed simulated time and steps
- reward breakdown
- no-progress warning
- start, pause, resume, stop, reset, evaluation refresh, and export controls

Replay exports keep the dynamic role labels. Evaluation and benchmark replays now also preserve policy and trainer identity through the exported metadata so baseline and experimental runs can be distinguished during review.

## Interactive Training

Start the API server in one terminal:

```powershell
python -m sheepdog.server
```

Then open the UI, choose an episode count in the Train panel, toggle Fast mode if you want the replay to advance without visible delay, and press Train. The page will poll training status and refresh the latest checkpoint as each episode finishes.

The Train panel also exposes instinct shaping, curriculum stage, and replay
debug toggles. If you are switching from an older reward setup, clear the
persisted training artifacts before you start the new curriculum run so the
trainer does not continue mutating weights that learned to dive into the flock.

## Tests

Python:

```powershell
python -m pip install -e .[dev]
python -m ruff format src tests
python -m ruff check src tests
python -m pylint src/sheepdog tests
python -m pytest
```

Ruff is the source of truth for formatting, imports, unused code, and Python
style. Pylint is intentionally limited to fatal and error diagnostics so the
VS Code Problems panel stays focused on defects instead of duplicating Ruff or
reporting legacy complexity metrics. VS Code formats Python with Ruff on save.
The repository-wide Ruff lint and Pylint checks are clean; a full-tree
`ruff format --check` remains deferred until legacy formatting is normalized.

Web:

```powershell
cd web
npm install
npm run lint
npm run test
```

Full validation pass from PowerShell:

```powershell
python -m ruff check src tests
python -m pylint src/sheepdog tests
python -m pytest
cd web
npm run lint
npm run test
```

## Configuration Example

You can provide a JSON config file to the CLI. Example:

```json
{
  "environment": {
    "width": 48,
    "height": 32,
    "dogs": 3,
    "sheep": 8,
    "max_steps": 360
  },
  "rewards": {
    "progress_scale": 2.0,
    "sheep_penned_reward": 8.0,
    "terminal_success_reward": 20.0
  },
  "training": {
    "episodes": 1000,
    "output_dir": "artifacts"
  }
}
```

## Known Limitations

- The first release uses a simple hill-climbing policy rather than PPO.
- Dynamic cooperation is role-aware, but it is still a shared linear baseline rather than a neural policy.
- The UI is a replay viewer for exported runs, not a live training dashboard.
- There is no backend API server yet; the browser reads generated files directly.

## The Next Steps

1. Compare this role-aware hill-climbing baseline against PPO or MaskablePPO rather than replacing it blindly.
2. Add richer role analytics and per-dog trajectory overlays in the replay viewer.
3. Expand replay analytics with per-step trajectory charts.
4. Add a lightweight local API if live browser control becomes necessary.
