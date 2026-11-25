# Stage 8 Progressive Recovery Curriculum v2 (`stage8_curriculum_v2`)

## 1. Experimental Setup & Integrity

- **Experiment ID**: `stage8_curriculum_v2`
- **Script**: [`scripts/run_stage8_curriculum_v2.py`](file:///c:/Users/inouy/source/sheep_dog/scripts/run_stage8_curriculum_v2.py)
- **Results Artifact**: [`artifacts/experiments/stage8_curriculum_v2/v2_results.json`](file:///c:/Users/inouy/source/sheep_dog/artifacts/experiments/stage8_curriculum_v2/v2_results.json)
- **Parent Checkpoint**: `artifacts/models/maskable-ppo-05065228.zip` (Ckpt 613, SHA-256 `da8ce02fcdcd9100601def6fb733762c5c9ce134d11d07058e57919763530e60`, verified `parent_unchanged=true`).
- **Training Budget**: 3 independent training seeds (`7, 101, 202`) × `327,680` timesteps (`10 segments × 32,768` timesteps) per run, writing exclusively to `artifacts/experiments/stage8_curriculum_v2/seed_<S>/treatment_v2/`.
- **Fixed Hyperparameters**: `effective_learning_rate=0.00015`, `effective_entropy_coef=0.005`, `ppo_env_workers=16`, unchanged rewards, model architecture, and evaluation criteria.
- **Spawn Mixes**:
  - **Control** (from `stage8_curriculum_multiseed_v1`): `{"fixed_easy": 0.30, "randomized_flock": 0.55, "nearby_stray": 0.15}`
  - **Treatment-v1** (from `stage8_curriculum_multiseed_v1`): `{"randomized_flock": 0.35, "nearby_stray": 0.15, "split_flock": 0.15, "wall_recovery": 0.10, "fixed_easy": 0.10, "wrong_side_recovery": 0.15}`
  - **Treatment-v2**: `{"randomized_flock": 0.45, "nearby_stray": 0.15, "split_flock": 0.15, "coherent_wall_recovery": 0.10, "wrong_side_mild": 0.08, "fixed_easy": 0.07}`
- **New Progressive Recovery Modes**:
  - `coherent_wall_recovery`: Spawns all sheep around a single anchor `0–3` cells from one wall/corner (avoiding the pen zone `94..107, 1..14`, `jitter_x=2, jitter_y=2`, spread `<= 6.0`).
  - `wrong_side_mild`: Samples a flock center `>= 25` cells from `pen.center`, spawns sheep (`jitter_x=2, jitter_y=2`), and repositions `self._dogs` with a mild forward pen-side offset (`4–10` cells) plus lateral flank offset (`±(4–8)` cells, `jitter=1`).
  - *Solvability note*: `HeuristicExpertPolicy` solves `1/5` on `coherent_wall_recovery` (`2016..2020`) and `3/5` (`_make_env`) / `2/5` (Stage 8 config) on `wrong_side_mild` (similar to `0/5` on `wall_recovery` and `2/5` on `wrong_side_recovery`), whereas neural `Ckpt 613` already solves `5/5` on `coherent_wall_recovery` and `0/5` on `wrong_side_mild`.

---

## 2. Late-Segment Drift Check (`seg_2..seg_10` on Dev-10)

| Arm | Train Seed | Best Seg (Timesteps) | Seg 2 SR (Rew) | Seg 4 SR (Rew) | Seg 6 SR (Rew) | Seg 8 SR (Rew) | Seg 10 SR (Rew) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Control** | 7 | `seg_6` (196,608) | 0.80 (343.1) | 0.90 (382.2) | **0.90 (411.9)** | 0.80 (264.9) | 0.70 (244.0) |
| **Control** | 101 | `seg_8` (262,144) | 0.80 (303.6) | 0.90 (364.5) | 0.80 (358.3) | **0.90 (399.9)** | 0.80 (353.8) |
| **Control** | 202 | `seg_4` (131,072) | 0.80 (358.9) | **0.90 (382.8)** | 0.90 (382.2) | 0.90 (370.3) | 0.90 (378.2) |
| **Treatment-v1** | 7 | `seg_2` (65,536) | **0.90 (404.7)** | 0.90 (378.7) | 0.80 (275.6) | 0.80 (280.8) | 0.80 (269.1) |
| **Treatment-v1** | 101 | `seg_2` (65,536) | **0.90 (365.2)** | 0.80 (296.8) | 0.70 (170.0) | 0.70 (179.9) | 0.80 (241.8) |
| **Treatment-v1** | 202 | `seg_6` (196,608) | 0.80 (312.6) | 0.90 (398.3) | **1.00 (467.9)** | 0.80 (321.4) | 0.70 (207.9) |
| **Treatment-v2** | 7 | `seg_10` (327,680) | 0.90 (349.6) | 0.70 (250.9) | 0.80 (364.0) | 0.80 (389.0) | **1.00 (483.2)** |
| **Treatment-v2** | 101 | `seg_10` (327,680) | 0.70 (256.7) | 0.90 (355.7) | 0.80 (305.1) | 0.80 (354.4) | **1.00 (471.6)** |
| **Treatment-v2** | 202 | `seg_10` (327,680) | 0.80 (301.8) | 0.80 (333.0) | 0.90 (373.1) | 0.90 (413.1) | **1.00 (477.5)** |

**Key finding**: Treatment-v2 completely eliminated the late-segment dev drift seen in Treatment-v1 (`seg_2`–`seg_6` peak followed by `0.70–0.80` decay). Every Treatment-v2 seed improved through `seg_10` (`327,680` ts) and achieved **1.00 (10/10)** on Dev-10.

---

## 3. Aggregated Comparison: Baseline vs Control vs Treatment-v1 vs Treatment-v2

*(Mean ± Std [Min, Max] across 3 training seeds `[7, 101, 202]`)*

| Suite | Arm | Success Rate | Avg Steps (All) | Avg Reward | Avg No-Progress Steps |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Dev-10** | Baseline (Ckpt 613) | 0.900 (9/10) | 133.9 | 412.1 | 30.6 |
| **Dev-10** | Control | 0.900 ± 0.000 [0.900, 0.900] | 134.7 | 398.2 ± 14.6 | 31.5 |
| **Dev-10** | Treatment-v1 | 0.933 ± 0.058 [0.900, 1.000] | 138.7 | 412.6 ± 51.8 | 24.2 |
| **Dev-10** | **Treatment-v2** | **1.000 ± 0.000** [1.000, 1.000] | **109.7** | **477.4 ± 5.8** | **2.2** |
| **Diag-20** | Baseline (Ckpt 613) | 0.800 (16/20) | 154.8 | 238.3 | 53.5 |
| **Diag-20** | Control | 0.850 ± 0.050 [0.800, 0.900] | 142.7 | 279.9 ± 55.4 | 40.1 |
| **Diag-20** | Treatment-v1 | 0.867 ± 0.029 [0.850, 0.900] | 142.0 | 283.1 ± 25.6 | 39.6 |
| **Diag-20** | **Treatment-v2** | **0.883 ± 0.029** [0.850, 0.900] | **129.9** | **306.9 ± 22.8** | **31.1** |
| **Test-30** | Baseline (Ckpt 613) | 0.900 (27/30) | 129.9 | 359.6 | 29.0 |
| **Test-30** | Control | 0.900 ± 0.033 [0.867, 0.933] | 133.0 | 366.8 ± 13.3 | 31.7 |
| **Test-30** | **Treatment-v1** | **0.933 ± 0.000** [0.933, 0.933] | **125.8** | 368.4 ± 7.6 | **20.9** |
| **Test-30** | Treatment-v2 | 0.911 ± 0.038 [0.867, 0.933] | 126.6 | **371.6 ± 30.3** | 26.0 |
| **Probe-25** | Baseline (Ckpt 613) | 0.320 (8/25) | 305.2 | -141.2 | 184.4 |
| **Probe-25** | Control | 0.280 ± 0.040 [0.240, 0.320] | 312.2 | -146.6 ± 41.0 | 194.1 |
| **Probe-25** | **Treatment-v2** | **0.360 ± 0.040** [0.320, 0.400] | **298.0** | **-99.7 ± 25.1** | **173.6** |

*(Note: Treatment-v1 was only evaluated on Probe-15 in `multiseed_results.json`, scoring `0.289 ± 0.038`.)*

---

## 4. Stress Probe-25 Category Breakdown (5 seeds per category)

| Category | Baseline (Ckpt 613) | Control Mean ± Std (Rew) | Treatment-v1 (Probe-15) | Treatment-v2 Mean ± Std (Rew) |
| :--- | :---: | :---: | :---: | :---: |
| `coherent_wall_recovery` (5) | **1.000** (546.9) | 0.933 ± 0.115 (512.1) | — | 0.933 ± 0.115 (515.0) |
| `split_flock` (5) | 0.600 (166.3) | 0.400 ± 0.000 (34.1) | **0.800 ± 0.000** (290.5) | 0.667 ± 0.115 (197.2) |
| `wall_recovery` (5) | 0.000 (-455.6) | 0.067 ± 0.115 (-336.3) | 0.067 ± 0.115 (-384.3) | **0.133 ± 0.115** (-332.6) |
| `wrong_side_mild` (5) | 0.000 (-456.6) | 0.000 ± 0.000 (-464.7) | — | 0.000 ± 0.000 (-453.7) |
| `wrong_side_recovery` (5) | 0.000 (-506.9) | 0.000 ± 0.000 (-478.3) | 0.000 ± 0.000 (-479.7) | **0.067 ± 0.115** (-424.5) |

---

## 5. Forgetting vs Baseline & Per-Seed Treatment-v2

| Train Seed | Dev-10 SR (Rew) | Diag-20 SR (Rew) | Test-30 SR (Rew) | Probe-25 SR (Rew) | Regressed / Gained (Dev / Diag / Test) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **7** | 1.000 (483.2) | 0.900 (314.9) | 0.867 (336.9) | 0.360 (-89.5) | `[]`/`[37]` ; `[]`/`[109, 131]` ; `[1193]`/`[]` |
| **101** | 1.000 (471.6) | 0.900 (324.6) | 0.933 (385.0) | 0.320 (-128.4) | `[]`/`[37]` ; `[]`/`[109, 131]` ; `[]`/`[1171]` |
| **202** | 1.000 (477.5) | 0.850 (281.1) | 0.933 (393.0) | 0.400 (-81.4) | `[]`/`[37]` ; `[]`/`[109]` ; `[]`/`[1171]` |

- **Mean forgetting (Regressed / Gained / Net)**:
  - **Control**: Dev `0.00 / 0.00 / 0.00`, Diag `0.67 / 1.67 / +1.00`, Test `1.00 / 1.00 / 0.00`
  - **Treatment-v1**: Dev `0.33 / 0.67 / +0.33`, Diag `0.00 / 1.33 / +1.33`, Test `0.00 / 1.00 / +1.00`
  - **Treatment-v2**: Dev `0.00 / 1.00 / +1.00`, Diag `0.00 / 1.67 / +1.67`, Test `0.33 / 0.67 / +0.33`

---

## 6. Wall-Clock Cost

- **Treatment-v2 training (327,680 ts, 10 segments, 16 workers)**: `219.2s` (seed 7), `212.1s` (seed 101), `212.4s` (seed 202) — mean **`214.6 ± 4.0s` (~3.6 min/seed)**.

---

## 7. Recommendation

- **Do NOT auto-promote Treatment-v2 to production `spawn_mix` yet**, because:
  1. `wrong_side_mild` remained `0.000` (0/5) across all 3 seeds despite 8% exposure (`wrong_side_recovery` ticked up to `1/15`), showing that even mild pen-side starts are not learned via standard PPO updates from Ckpt 613 at 8% mix without an explicit flanking/wrong-side curriculum or reward/observation signal.
  2. Untouched Test-30 was `0.911 ± 0.038` (1 regression on seed 7) vs Treatment-v1's `0.933 ± 0.000` (0 regressions).
- **Strong positive signal**: Treatment-v2 **completely cured late-segment dev drift** (`seg_10 = 1.000` on all 3 seeds, Dev-10 NP steps `30.6 -> 2.2`, Diag-20 `0.800 -> 0.883` with 0 regressions, Probe-25 `0.320 -> 0.360`). Keep `coherent_wall_recovery` / `randomized_flock` balance as the stable base, and diagnose why `wrong_side_mild` stays `0/5` (e.g., replay inspection of `2021..2025` or higher wrong-side exposure / lower LR) before promotion.
