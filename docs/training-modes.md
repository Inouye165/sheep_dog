# Training Modes

Sheepdog supports two observation modes that control what information each dog
receives per step. The mode is set via `TrainingConfig.observation_mode`.

---

## Guided mode (default)

`observation_mode = "guided"` (default)

Each dog is assigned a **scripted role** by the role assigner (pusher, flanker,
or holder). The observation vector includes:

- Own position and velocity
- Per-role one-hot label (`role_pusher`, `role_flanker`, `role_holder`)
- Scripted target coordinates (`target_x`, `target_y`, `distance_to_target`)
- Flock centroid and spread
- Per-sheep relative positions (up to `N_SHEEP` slots)
- Other dogs' relative positions

This gives the neural policy strong inductive bias and usually converges faster,
but the dogs lean on the scripted roles rather than developing truly emergent
coordination strategies.

---

## Emergent mode

`observation_mode = "emergent"`

Scripted roles and target coordinates are **removed** from the observation.
The dogs must learn herding behaviour purely from raw spatial observations and
the reward signal. The observation vector includes:

- Own position
- Per-dog slot one-hot ID (which slot am I?)
- Pen centre relative position and distance
- Flock centroid relative position and spread
- Farthest unpenned sheep relative position (`farthest_unpenned_dx`,
  `farthest_unpenned_dy`, `farthest_unpenned_distance`)
- Per-sheep relative positions (up to `N_SHEEP` slots)
- Other dogs' relative positions

Because the observation size differs from guided mode, **emergent and guided
checkpoints are not interchangeable**.

### Recommended reward config for emergent training

Add pressure to bring the stray in by enabling the optional reward terms:

```python
rewards=RewardConfig(
    farthest_sheep_progress_scale=1.0,  # reward for reducing farthest distance
    stray_ignore_penalty_scale=0.5,     # penalty when dogs ignore the stray
),
```

Both scales default to `0.0` (terms are off), so guided mode is unaffected.

---

## Example: emergent training config

```python
from sheepdog.config import LabConfig, TrainingConfig, RewardConfig

config = LabConfig(
    training=TrainingConfig(
        trainer_type="maskable_ppo",
        policy_type="neural",
        observation_mode="emergent",
        scenario_training_enabled=True,
        scenario_mix={
            "random": 0.50,
            "scattered_sheep": 0.20,
            "split_flock": 0.15,
            "corner_huddle": 0.10,
            "partial_pen_with_stray": 0.05,
        },
        output_dir="artifacts/emergent",
    ),
    rewards=RewardConfig(
        farthest_sheep_progress_scale=1.0,
        stray_ignore_penalty_scale=0.5,
    ),
)
```

The `partial_pen_with_stray` scenario places most sheep near the pen with one
sheep in the far corner, giving the dogs a concrete stray-retrieval task to
practise each episode it is sampled.

---

## Choosing a mode

| Concern | Guided | Emergent |
|---|---|---|
| Training speed | Faster | Slower |
| Role dependency | High | None |
| Emergent coordination | Limited | Goal |
| Checkpoint portability | Guided only | Emergent only |
| Stray reward terms useful | No | Yes |
