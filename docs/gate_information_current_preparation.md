# Current-parent gate-information experiment preparation

Prepared and verified on 2026-11-13, with **no optimization or new training run**.
The active app was separately restarted idle; these workspaces do not change it.

The final workspace is
`artifacts/experiments/gate_current_prep_20261113_173000/manifest.json`.
The earlier `gate_current_prep_20261113_172400` preparation is preserved and
superseded by this final verification. Directory suffixes are identifiers, not
measured start times.

The explicit protocol is `stage34_gate_information_current_parent_v2`.
The historical [v1 experiment](gate_information_experiment.md) and its pinned
checkpoint/hyperparameter requirements remain unchanged. The v2 parent is the
verified immutable stopped-run backup: policy version 10483, checkpoint episode
40897, timestep 343787786; SHA-256
`26aecd34c7d1723feb4ae2f44a03fbddc16c494b2855a6c26216b31703c54dd2`.
Its saved confidence development evaluation scored 8/10, failing seeds 59 and
73 with zero sheep penned. That historical baseline is copied and hashed;
a fresh baseline is still required before any optimization.

The hypothesis is that explicit gate location and outward orientation help
learning recovery compared with four zero appended inputs. Six workspaces use
paired continuation seeds 101, 202 and 303. Both arms retain the same parent
physics/reward, three dogs, six sheep, 2,320-world-step episode limit, architecture
256/256/128, gamma 0.997, GAE lambda 0.98, entropy coefficient 0.015, learning
rate 0.0001, rollout 2,048, batch 1,024 and archive n_epochs 10. Adaptive changes
are disabled equally. Planned recovery-start probability remains 25%, with
the same three fixed development-derived failure-class weights in both arms.

Both models explicitly use 58 inputs. Control uses `guided_gate_control` (54
existing features plus four zeros); treatment uses `guided_gate_treatment`
(the same 54 features plus gate dx/dy and outward normal x/y). The schema is
`guided_gate_v1`; feature names are recorded in the manifest. Existing parameters
and biases are preserved exactly, new input columns are zero, and both Adam
states are reset identically. This is approximate continuation in new explicit
modes, not silent reinterpretation of a 54-input checkpoint. The policy remains
guided/assisted, including existing physics assistance.

Every pair passed the existing masked probability/logit/value numerical checks.
That legacy verifier's 15-step seed-11 trace rounds reward totals; it is not
used as proof of exact full-reward equality. A separate check passed **20 world
steps on development seed 53**, comparing unrounded complete snapshots, all
reward fields and actions exactly, and checking every existing policy parameter
plus zero new columns. Equal traces before training establish compatibility,
not a learning improvement. All six configs/models and the copied parent are
covered by 16 verified file checksums. Full configs, versions, source revision,
dirty patch and untracked source hashes are retained. Held-out cases were unused.

The proposed next experiment is limited to 12,288 transitions per arm/seed,
73,728 total, 120 seconds of training per arm and 900 seconds overall, with at
most 70 development evaluation episodes including the fresh parent baseline.
This is a proposed budget only. No optimizer runner or training was started;
the original v1 90,112-transition budget does not apply to this preparation.
No automatic promotion or live default change is warranted.

Verification commands:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_gate_information_preparation.py tests/test_gate_information_experiment.py -q
.venv/Scripts/python.exe -m ruff check src/sheepdog/gate_information_preparation.py scripts/prepare_gate_information_current.py tests/test_gate_information_preparation.py
.venv/Scripts/python.exe scripts/prepare_gate_information_current.py verify --root artifacts/experiments/gate_current_prep_20261113_173000
```

Results: **15 passed, 1 skipped** in 5.27 seconds; the skipped legacy test needs
the absent historical v1 parent. Ruff passed. Actual `prepare` and `verify`
passed against the available immutable parent, including all six converted
models and bounded simulations. UTF-8, content and whitespace checks passed.
No viewer code changed, so viewer checks were not run. No full training benchmark
or held-out evaluation was run. Next action is manager review and a separately
bounded training comparison after a fresh baseline, with the app kept idle.
