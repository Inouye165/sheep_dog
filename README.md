# Sheepdog Herding Lab

Sheepdog Herding Lab is a deterministic reinforcement-learning simulation where a team of dogs learns to herd multiple sheep into a pen.

The project is split into two parts:

- A Python simulation, training, and evaluation pipeline.
- A React + TypeScript replay viewer that reads exported checkpoint data.

The first version is intentionally honest about what it does. It trains a simple shared policy with hill climbing, measures real checkpoint results, and exports replays that the UI can inspect.

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

## Repo Structure

- `src/sheepdog/` Python simulation, policies, training, evaluation, checkpointing, and replay export.
- `tests/` Python regression tests.
- `web/` React + TypeScript UI for replay playback and checkpoint comparison.
- `docs/architecture.md` High-level architecture notes.
- `artifacts/` Generated training and evaluation output.

## Architecture Overview

The simulation is grid based and deterministic for a fixed seed. Dogs act first, sheep react to nearby pressure, and the environment tracks progress toward the pen. Rewards are decomposed into named pieces so the total can be audited.

The trainer uses a simple shared trainable policy with hill climbing. It does not pretend to be PPO or another full RL algorithm. Checkpoints are real evaluation snapshots, not fabricated milestones.

The web app is a viewer, not a second simulation engine. It loads exported checkpoint and replay JSON files from `web/public/generated/` and plays them back frame by frame.

## Policy Modes

- `random_untrained` picks legal moves uniformly and has no herding intelligence.
- `instinct_only` can chase, circle, avoid diving into the flock, and recover nearby sheep, but it does not know where the pen is.
- `heuristic_expert` is a scripted, pen-aware expert that uses pressure positioning behind the flock relative to the target.
- `trained_policy` uses learned weights from training checkpoints.

By default the no-training playback path uses `instinct_only`, not the expert heuristic. Pen-directed behavior now requires training, `heuristic_expert`, or an explicit handler target command.

## Simulation Concepts

- Dogs are AI-controlled and share one policy.
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

Run training with:

```powershell
python -m sheepdog.cli train
```

This writes checkpoint JSON files under `artifacts/checkpoints/`, evaluation JSON and CSV files under `artifacts/evaluations/`, and web assets under `web/public/generated/`.

## Checkpoints

Default checkpoint episodes:

`0, 5, 10, 25, 50, 100, 500, 1000`

Each checkpoint includes:

- checkpoint episode
- total training episodes so far
- policy name and weights
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

The UI training path now sends `enable_instinct_rewards`, `curriculum_stage`,
and `debug_reward_breakdown` to the backend. By default the Train panel starts
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


## UI

Start the web viewer with:

```powershell
cd web
npm install
npm run dev
```

The UI shows:

- the field, dogs, sheep, and pen
- the selected checkpoint and seed
- policy mode
- run state
- sheep penned count
- elapsed simulated time and steps
- reward breakdown
- no-progress warning
- start, pause, resume, stop, reset, evaluation refresh, and export controls

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
python -m pytest
ruff check src tests
ruff format --check src tests
```

Web:

```powershell
cd web
npm install
npm run lint
npm run test
```

Full validation pass from PowerShell:

```powershell
python -m pytest
ruff check src tests
ruff format --check src tests
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
- The UI is a replay viewer for exported runs, not a live training dashboard.
- There is no backend API server yet; the browser reads generated files directly.

## Next Steps

1. Replace the baseline trainer with a fuller RL implementation if needed.
2. Add richer dog coordination policies.
3. Expand replay analytics with per-step trajectory charts.
4. Add a lightweight local API if live browser control becomes necessary.
