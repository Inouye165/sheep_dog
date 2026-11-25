# Failure-directed training plumbing correction — 2026-11-13

## Baseline and hypothesis

The saved policy versions 10479–10481 reported active failure weights but zero exposure episodes and empty scenario counts. Code inspection established why: the trainer queried its management-only `ScenarioSampler`, which updates weights but never samples starts, rather than the actual training workers. Zero counts therefore did not establish zero recovery practice.

Manager baseline inspection: policy version 10481 / checkpoint 40891 completed 6/10 development episodes; seeds 53 (0 penned), 59 (5 penned, corner), 67 (0 penned, 1188 gate-corridor-failure steps, corner), and 73 (0 penned) each timed out at 2320 world steps. This is guided behavior, not autonomous learning evidence. Manager reran the repetition/recovery analyzer on saved failure replays: repeated movement steps were 1875/1384/2190/2308 respectively; stall counts 4/7/1/2, recovered stalls 2/6/0/0, relapses 1/0/0/1, and one unresolved stall each. The analyzer counts 2321 frames including step zero, so its reported total is not the 2320-step evaluation budget. These observations support testing more recovery exposure, not a causal conclusion.

Hypothesis: trustworthy worker telemetry and reproducible, legal initial conditions are prerequisites for assessing failure-focused training. Do not change rewards or observations on the basis of incorrect exposure counts.

## Changes and compatibility

- Trainer reads actual vector-worker sampler summaries, aggregates scenario counts, and reports source, worker count and cumulative vector-environment-lifetime scope. Reading does not reset counters. These are episode **starts**, not completions or per-checkpoint counts. Empty exposure has undefined percentages (`null`); query failure is explicitly unavailable with null counts and error details.
- Actual worker weights are reported separately from requested manager weights; propagation status/error and warning logs expose failures. Existing state persistence still stores requested failure weights for restoration. Historical zero-count records are not rewritten.
- Adapter passes the requested reset seed into the sampler, uses the safe selected seed for both normal and scenario resets, and records the actual `Scenario.id`. Previously the normal path discarded exclusion remapping, and targeted explicit/fixed resets could use a different seed than their replay metadata.
- Real reset integration exposed wall-stray sheep spawned on closed fence cells, causing `PenNavigation.distance` to raise. The wall-stray builder now locates its entrance for every gate orientation and deterministically moves only invalid/overlapping **initial placements** to the nearest legal free cell. This is scenario initialization, not an in-episode recovery override. Dimensions, counts, seed and personality settings remain intact. A bounded scan of 120 high-pen initial layouts found no analogous fence/overlap issue; no other builder was changed.

Model observation/action/reward schemas and weights are unchanged, so existing models remain loadable. Training initial-condition sequences and seed metadata are intentionally corrected: continuation is not bit-for-bit equivalent to the previous sampling path. Already-loaded runtime code requires a safe restart to adopt these fixes; this task neither restarts nor resets the active writer. Defaults, including the 25% target ratio, remain unchanged.

## Limitations and next action

The historical seed-to-class shortcut can assign a stale archetype to a changed failure (e.g. seed 53 always maps to gate-wall). A missing class also decays without proof it passed. These classifier issues remain explicit limitations, not fixes in this patch. Worker RNG/config duplication during vector construction and exact sampler RNG resumption remain separate work; current counts restart with new worker instances. Configured development cases are used for targeting; this must never be called untouched held-out evaluation.

Verify actual exposure after safe runtime adoption. Compare an isolated 25% versus 60% recovery-sampling pilot from the same immutable parent and fixed development-derived weights, equal compute and unchanged observations/rewards. Ordinary full-task success and recovery-start success must be reported separately. No learning improvement or stage completion is claimed by this plumbing patch.

## Verification

Focused checks and final results are recorded in handoff.md. No policy optimization, benchmark promotion, checkpoint overwrite, active config mutation, or frontend changes were performed by this implementation task.


## Read-only runtime adoption audit - 2026-11-13

Local API access was successfully verified using sandbox-escalated execution: manager health check returned ready; a direct GET /api/training/status confirmed running=true, phase=learning, curriculum_stage=34. Process inspection identified backend PID 8324 listening on 127.0.0.1:8000 with command `python.exe -m sheepdog.server`. This supersedes earlier blanket statements that local health access was blocked: ordinary sandbox access was restricted, while the escalated local-socket check succeeded.

Code inspection found that the current pause message overstates checkpoint safety. A pause can interrupt the current learning segment; the trainer checks the stop flag before saving its new policy and returns the prior durable checkpoint. Optimizer updates since that checkpoint can therefore be discarded. During evaluation the new model ZIP already exists, but its training-state reference is persisted only after evaluation/replay. There is no atomic API pause-at-durable-checkpoint barrier, and racing the next checkpoint cannot guarantee zero lost updates.

The fixes are verified in source, but adoption by the already-loaded active backend is deliberately pending. No pause, shutdown, restart, live configuration mutation or checkpoint replacement was performed. No code expansion was made for this audit.

Next action: wait for natural job completion, verify stable durable state and model identity, create a unique recoverable state/config/checkpoint snapshot with integrity manifest, gracefully restart only the backend while leaving the UI running, and verify loaded checkpoint identity. Do not use the start/stop scripts' force-stop fallbacks. Natural completion exhausts the current request; restarting must not automatically repeat it or invent a new training budget. Any later approximate-resume alternative needs an explicit manager decision and disclosure of lost in-memory progress. After separately authorized learning resumes, verify actual worker exposure at the next checkpoint. The isolated learning pilot can proceed without changing the live job.


## Stopped-run backup and fixed-source restart - 2026-11-13

The user stopped training and closed the app instead of waiting for the remaining long job. Inspection found no backend/viewer listeners or training writer. The runtime session had ended with `manually_stopped`; persisted session state was `stopped`. The durable current model is policy version 10483, checkpoint 40897, Stage 34, timestep identity 343787786. Its archive integrity passed and metadata confirms three dogs, six sheep, 54 guided inputs and hidden layers 256/256/128. SHA-256: `26aecd34c7d1723feb4ae2f44a03fbddc16c494b2855a6c26216b31703c54dd2`.

Created `artifacts/experiments/stopped_adoption_20261113_171410/manifest.json` with twelve verified state/config/session/current-and-best-model files. Source hashes were stable across repeated reads and all copied hashes were checked before and after restart. SQLite was not copied. The source current checkpoint remains unchanged. Any unsaved optimizer updates lost when stopping cannot be recovered or quantified from these artifacts; continuation remains approximate.

Started the backend and viewer directly in hidden processes, bypassing launcher resume prompts. The stopped session did not auto-resume. API health reports ready; viewer returned HTTP 200; repeated training status reports `running=false`, `phase=stopped`, Stage 34 and the exact saved checkpoint identity. `adoption.json` records listener PIDs and relevant fixed-source hashes. The existing status message still contains historical wording about training continuing; the `running=false` field and stopped phase establish the actual idle state.

The source fixes are now loaded by the restarted backend. The old job's persisted remaining budget (399,996 transitions) was not resumed, reset or cleared. No new policy optimization was started. Actual live worker exposure still needs verification in the next deliberately budgeted run; readiness alone does not establish sampling behavior. The separate gate-information preparation may use the immutable checkpoint/metadata copies in this backup. No new application tests were needed for this operational/documentation task; backup integrity, ZIP integrity, process/port checks, API/model identity and documentation encoding/content were checked.
