# Project rules

## Mission and ownership

Build Sheepdog Herding Lab into a rigorous, accessible reinforcement-learning tool that students can learn from and professionals can inspect, reproduce, and extend. The target task is **three dogs learning to herd all six sheep into a pen**, including recovery from oscillation, blocked approaches, walls, corners, and unhelpful team positions.

The primary assistant is the manager and owns scoping, priorities, implementation prompts, review, verification, integration decisions, and communication with the user. Keep the user informed about material issues, ideas, tradeoffs, and evidence. Never assign project management or user-communication ownership to an implementation agent. The primary assistant remains accountable for the result.

The user gives standing authorization to delegate implementation. All implementation, including code, tests, configuration, repository documentation, and follow-up fixes, must be performed by an implementation agent through a concrete prompt from the manager. The manager may inspect files, plan, review changes, and run verification directly, but must delegate edits and fixes. Any exception allowing the manager to implement directly must first be explicitly discussed with the user; convenience or a small change is not an exception.

Each implementation prompt must state the goal, bounded scope and allowed files, constraints, acceptance criteria, and required verification. The implementation agent performs those edits and checks, preserves unrelated work, and reports changed files, evidence, limitations, and blockers back to the manager. The agent must not take over management, change priorities, communicate decisions on the manager's behalf, or expand the task without returning to the manager. The manager reviews the result, delegates any corrections, decides whether acceptance criteria are met, and communicates the outcome to the user.

At the beginning of work, read `handoff.md` and `todo.md`, inspect Git status, and inspect relevant code before proposing changes. Preserve unrelated and uncommitted work. At the end of meaningful work, update the handoff and task status with actual evidence, unresolved issues, and the next concrete action. Never mark an unverified outcome as achieved.

## Learning contract

- The primary benchmark uses exactly three dogs and six sheep. Simpler curricula are training aids; report them separately.
- Success requires all six sheep inside the pen under the environment's validated completion rule within the episode budget. Partial completion and high reward are not success.
- Dogs must learn how to break back-and-forth cycles, release pressure, move out of the way, reposition around sheep, and coordinate a new approach. Temporary movement away from the pen can enable eventual success.
- Do not solve recovery by teleporting entities, forcing sheep toward the goal, injecting scripted dog maneuvers, or resetting a stalled episode and counting it as success.
- Separate physical legality masks from tactical guidance. Any scripted roles, targets, shepherd commands, tactical masks, expert demonstrations, or recovery overrides must be disclosed and evaluated as assisted behavior. Do not label an assisted result autonomous.
- Keep guided and scripted policies as clearly labeled teaching aids and baselines. Establish a separately evaluated autonomous policy path; audit that path for hidden guidance before making independence claims.
- Rewards should favor eventual task completion and useful recovery without requiring monotonic distance reduction. Check for reward farming, repeated entry/exit credit, waiting exploits, and cycles that earn reward without solving the task.
- Give a policy enough observable information or memory to learn recovery. Any observation, action, reward, physics, or termination change needs a compatibility decision and recorded rationale; never silently reinterpret an old checkpoint.
- Diagnostics may identify failures and select training scenarios. They must not secretly take control of evaluation actions. Keep held-out evaluation cases out of failure-directed training.

## Evidence and experimental discipline

Before modifying training, record a baseline and a specific hypothesis. Use comparable configurations, episode budgets, and evaluation seeds. Report changes to compute budget and assistance explicitly.

Maintain separate training, development evaluation, and held-out evaluation seed/scenario sets. Tune on development cases; reserve held-out cases for release decisions. Evaluate multiple independently trained seeds and report sample counts, uncertainty, and per-scenario failures, not only the best run.

Recovery coverage must include oscillation, dog-blocked gate/approach, counterproductive pressure, wall/corner traps, split flock, stragglers, and interference among dogs. Measure full-flock success, timeout rate, steps to success, recovery success/time, repeated-state or oscillation duration, and blocked-corridor duration. Define each metric and denominator before using it for promotion.

Every reported experiment must identify source revision (and dirty-tree patch when relevant), full configuration, dependency versions, training/evaluation seeds, training budget, checkpoint identity, assistance mode, and artifact locations. Preserve representative failures as well as successes. Replays and displayed metrics must come from the stated checkpoint and evaluation run.

Do not claim the project is solved from training reward, a single seed, a selected replay, or an assisted run. Define release thresholds in advance, compare with random/scripted/guided baselines, and document regressions and limitations.

## Engineering and verification

- Keep simulation, training, evaluation, persistence, API, and presentation responsibilities distinct. Reuse existing capabilities before introducing parallel systems.
- Use seeded randomness and explicit configuration. Keep run state isolated; avoid global mutable state shared across experiments.
- Add focused tests for meaningful behavior changes: physics invariants, success/termination, recovery diagnostics, reward exploits, checkpoint compatibility, and replay fidelity as appropriate.
- Run the checks relevant to changed behavior. For Python use the project environment with `python -m pytest` and `python -m ruff check`; for the viewer use `npm test`, `npm run lint`, and `npm run build` in `web`. Record exact commands, results, and checks not run. Documentation-only changes need link/content checks, not training runs.
- Do not overwrite checkpoints or reset active training without a clear task reason and a recoverable copy. Do not start expensive training as an incidental verification step; establish the budget and stopping conditions first.
- Keep secrets, local environments, generated builds, logs, and large experiment artifacts out of Git. Use artifact manifests and external storage for transferable experiment state.

## Laptop, desktop, and future hosting

Use repository-relative or explicitly configured paths; never embed a user's drive, home directory, or machine identity. Document clean setup and CPU operation, with accelerator support optional and explicit. Record dependencies reproducibly; do not copy virtual environments or `node_modules` between machines.

Move source through Git and move experiment state through versioned artifact bundles. A transfer must include configuration, model and optimizer state where supported, normalization/RNG/curriculum state where applicable, a manifest, integrity checks, and restore verification. State clearly whether resumption is exact or approximate. Avoid concurrent writers to one experiment.

Design the web viewer to work through a configurable API and portable replay formats. Keep long-running training outside request handlers as the architecture evolves. Before public hosting, implement authenticated mutation controls, job/resource limits, durable storage, deployment configuration, and separation of public viewing from training control. Local launch scripts are not a production deployment.

## Teaching and professional usability

Explain observations, actions, rewards, policy architecture, training steps versus world steps, assistance, and evaluation in plain language. Provide a quick learning path and deeper technical references. Make failures inspectable with pause/step/replay, trajectories, and understandable diagnostics. Distinguish observed facts from inferred causes. Label live, recorded, training, and evaluation views accurately; include accessible controls, readable visuals, and units/definitions for metrics.

## Completion standard

A change is complete when its stated acceptance criteria are satisfied, relevant checks are recorded, documentation matches actual behavior, experiment/checkpoint compatibility is addressed where affected, and `handoff.md`/`todo.md` reflect the outcome. Communicate the result, material limitations, and next action directly to the user.
