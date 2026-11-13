from dataclasses import replace

import pytest

from sheepdog.config import RewardConfig
from sheepdog.rewards import RewardComputer, RewardInputs


def transition(old_dogs, dogs, old_sheep, sheep):
    return RewardInputs(
        previous_average_distance=50.0,
        current_average_distance=50.0,
        previous_flock_spread=5.0,
        current_flock_spread=5.0,
        newly_penned=0,
        no_progress_step=False,
        touched_wall=False,
        waited_without_reason=False,
        terminated=False,
        timeout=False,
        success=False,
        previous_farthest_distance=50.0,
        current_farthest_distance=50.0,
        previous_dog_positions=old_dogs,
        dog_positions=dogs,
        previous_sheep_positions=old_sheep,
        sheep_positions=sheep,
        flock_centroid=(50.0, 0.0),
        target_position=(100.0, 0.0),
    )


def test_approach_and_retreat_cancel_without_positive_ignore_penalty():
    computer = RewardComputer(RewardConfig(stray_ignore_penalty_scale=0.013))
    sheep = ((50.0, 0.0),)
    forward = computer.compute(transition(((0.0, 0.0),), ((4.0, 0.0),), sheep, sheep))
    backward = computer.compute(transition(((4.0, 0.0),), ((0.0, 0.0),), sheep, sheep))
    assert forward.stray_approach_progress > 0
    assert backward.stray_approach_progress < 0
    assert forward.stray_approach_progress + backward.stray_approach_progress == pytest.approx(0)
    assert forward.stray_ignore_penalty <= 0
    assert backward.stray_ignore_penalty <= 0
    assert forward.total + backward.total < 0
    assert forward.to_dict()["stray_approach_progress"] == forward.stray_approach_progress


def test_closed_loop_with_moving_sheep_and_changing_nearest_dog_cancels():
    computer = RewardComputer(RewardConfig(stray_ignore_penalty_scale=0.013))
    states = [
        (((0.0, 0.0), (90.0, 0.0)), ((30.0, 0.0), (60.0, 0.0))),
        (((40.0, 0.0), (85.0, 0.0)), ((65.0, 0.0), (25.0, 0.0))),
        (((10.0, 0.0), (70.0, 0.0)), ((35.0, 0.0), (55.0, 0.0))),
    ]
    rewards = [
        computer.compute(transition(old[0], new[0], old[1], new[1])).stray_approach_progress
        for old, new in zip(states, states[1:] + states[:1], strict=True)
    ]
    assert sum(rewards) == pytest.approx(0)


def test_approach_progress_is_bounded_without_isolation_amplification():
    computer = RewardComputer(RewardConfig(stray_ignore_penalty_scale=0.013))
    sheep = ((1000000.0, 0.0),)
    inputs = transition(((0.0, 0.0),), sheep, sheep, sheep)
    reward = computer.compute(inputs)
    assert 0 < reward.stray_approach_progress <= 0.26
    other = replace(inputs, flock_centroid=(-1000000.0, 0.0))
    assert computer.compute(other).stray_approach_progress == reward.stray_approach_progress


@pytest.mark.parametrize(
    "overrides",
    [
        {"newly_penned": 1},
        {"terminated": True},
        {"timeout": True},
        {"success": True},
        {"sheep_positions": ()},
        {"previous_sheep_positions": ()},
    ],
)
def test_penning_terminal_and_missing_states_have_no_approach_bonus(overrides):
    computer = RewardComputer(RewardConfig(stray_ignore_penalty_scale=0.013))
    inputs = transition(((0.0, 0.0),), ((4.0, 0.0),), ((50.0, 0.0),), ((50.0, 0.0),))
    assert computer.compute(replace(inputs, **overrides)).stray_approach_progress == 0


def test_stationary_and_disabled_shaping_have_no_approach_bonus():
    inputs = transition(((0.0, 0.0),), ((0.0, 0.0),), ((50.0, 0.0),), ((50.0, 0.0),))
    computer = RewardComputer(RewardConfig(stray_ignore_penalty_scale=0.013))
    assert computer.compute(inputs).stray_approach_progress == 0
    assert RewardComputer(RewardConfig()).compute(inputs).stray_approach_progress == 0


def test_old_reward_version_remains_loadable_with_warning():
    from sheepdog.checkpoints.store import (
        get_observation_schema_hash,
        verify_checkpoint_compatibility,
    )
    from sheepdog.config import LabConfig

    config = LabConfig()
    metadata = {
        "observation_schema_hash": get_observation_schema_hash(config),
        "reward_schema_version": "1.0",
    }
    result = verify_checkpoint_compatibility(metadata, config)
    assert result["compatible"]
    assert result["warnings"]
    metadata["reward_schema_version"] = "unknown"
    assert not verify_checkpoint_compatibility(metadata, config)["compatible"]
