"""Tests for the optional sheepdog-instinct reward shaping."""

from __future__ import annotations

from sheepdog.config import InstinctRewardConfig, RewardConfig
from sheepdog.rewards import RewardComputer, RewardInputs


def _base_inputs(**overrides: object) -> RewardInputs:
    defaults: dict[str, object] = {
        "previous_average_distance": 10.0,
        "current_average_distance": 10.0,
        "previous_flock_spread": 2.0,
        "current_flock_spread": 2.0,
        "newly_penned": 0,
        "no_progress_step": False,
        "touched_wall": False,
        "waited_without_reason": False,
        "terminated": False,
        "timeout": False,
        "success": False,
    }
    defaults.update(overrides)
    return RewardInputs(**defaults)  # type: ignore[arg-type]


def _instinct_config(**overrides: object) -> RewardConfig:
    instincts = InstinctRewardConfig(enable_instinct_rewards=True, **overrides)  # type: ignore[arg-type]
    return RewardConfig(instincts=instincts)


def test_instinct_rewards_default_off_produces_zero_terms() -> None:
    breakdown = RewardComputer(RewardConfig()).compute(
        _base_inputs(
            dog_positions=((0.0, 0.0),),
            sheep_positions=((5.0, 5.0),),
            flock_centroid=(5.0, 5.0),
            previous_flock_centroid=(5.0, 5.0),
            target_position=(10.0, 10.0),
        )
    )

    assert breakdown.pressure_zone == 0.0
    assert breakdown.safe_pressure == 0.0
    assert breakdown.grouping == 0.0
    assert breakdown.target_progress == 0.0
    assert breakdown.chaos_penalty == 0.0
    assert breakdown.overpressure_penalty == 0.0
    assert breakdown.split_flock_penalty == 0.0


def test_pressure_zone_rewards_dog_behind_flock_relative_to_target() -> None:
    config = _instinct_config()
    flock = (10.0, 10.0)
    target = (20.0, 10.0)
    behind_dog_inputs = _base_inputs(
        dog_positions=((4.0, 10.0),),
        flock_centroid=flock,
        target_position=target,
    )
    in_front_dog_inputs = _base_inputs(
        dog_positions=((16.0, 10.0),),
        flock_centroid=flock,
        target_position=target,
    )

    behind = RewardComputer(config).compute(behind_dog_inputs)
    in_front = RewardComputer(config).compute(in_front_dog_inputs)

    assert behind.pressure_zone > 0
    assert in_front.pressure_zone < 0
    assert behind.pressure_zone > in_front.pressure_zone


def test_safe_pressure_penalizes_dog_too_close_to_flock() -> None:
    config = _instinct_config()
    flock = (10.0, 10.0)
    target = (20.0, 10.0)
    too_close = _base_inputs(
        dog_positions=((10.0, 10.5),),
        flock_centroid=flock,
        target_position=target,
    )
    in_band = _base_inputs(
        dog_positions=((6.0, 10.0),),
        flock_centroid=flock,
        target_position=target,
    )

    close_breakdown = RewardComputer(config).compute(too_close)
    band_breakdown = RewardComputer(config).compute(in_band)

    assert close_breakdown.safe_pressure < 0
    assert band_breakdown.safe_pressure > 0


def test_safe_pressure_penalizes_dog_too_far_from_flock() -> None:
    config = _instinct_config()
    flock = (10.0, 10.0)
    target = (20.0, 10.0)
    too_far = _base_inputs(
        dog_positions=((22.0, 10.0),),
        flock_centroid=flock,
        target_position=target,
    )

    far_breakdown = RewardComputer(config).compute(too_far)

    assert far_breakdown.safe_pressure < 0


def test_pressure_zone_reward_decays_for_disengaged_dog() -> None:
    config = _instinct_config()
    flock = (10.0, 10.0)
    target = (20.0, 10.0)
    engaged = _base_inputs(
        dog_positions=((4.0, 10.0),),
        flock_centroid=flock,
        target_position=target,
    )
    disengaged = _base_inputs(
        dog_positions=((-10.0, 10.0),),
        flock_centroid=flock,
        target_position=target,
    )

    engaged_breakdown = RewardComputer(config).compute(engaged)
    disengaged_breakdown = RewardComputer(config).compute(disengaged)

    assert engaged_breakdown.pressure_zone > disengaged_breakdown.pressure_zone


def test_grouping_reward_tracks_flock_spread_change() -> None:
    config = _instinct_config()
    flock = (10.0, 10.0)
    target = (20.0, 10.0)
    grouping_in = _base_inputs(
        previous_flock_spread=4.0,
        current_flock_spread=2.0,
        dog_positions=((4.0, 10.0),),
        flock_centroid=flock,
        previous_flock_centroid=flock,
        target_position=target,
    )
    scattering = _base_inputs(
        previous_flock_spread=2.0,
        current_flock_spread=4.0,
        dog_positions=((4.0, 10.0),),
        flock_centroid=flock,
        previous_flock_centroid=flock,
        target_position=target,
    )

    grouped_breakdown = RewardComputer(config).compute(grouping_in)
    scattered_breakdown = RewardComputer(config).compute(scattering)

    assert grouped_breakdown.grouping > 0
    assert scattered_breakdown.grouping < 0


def test_target_progress_reward_tracks_centroid_movement() -> None:
    config = _instinct_config()
    target = (20.0, 10.0)
    moving_in = _base_inputs(
        previous_flock_centroid=(8.0, 10.0),
        flock_centroid=(12.0, 10.0),
        dog_positions=((4.0, 10.0),),
        target_position=target,
    )
    moving_away = _base_inputs(
        previous_flock_centroid=(12.0, 10.0),
        flock_centroid=(8.0, 10.0),
        dog_positions=((4.0, 10.0),),
        target_position=target,
    )

    closer = RewardComputer(config).compute(moving_in)
    farther = RewardComputer(config).compute(moving_away)

    assert closer.target_progress > 0
    assert farther.target_progress < 0


def test_training_rewards_can_still_include_target_progress() -> None:
    config = _instinct_config(target_progress_weight=0.75)
    breakdown = RewardComputer(config).compute(
        _base_inputs(
            previous_flock_centroid=(8.0, 10.0),
            flock_centroid=(12.0, 10.0),
            dog_positions=((4.0, 10.0),),
            target_position=(20.0, 10.0),
        )
    )

    assert breakdown.target_progress > 0


def test_chaos_penalty_when_dog_inside_flock() -> None:
    config = _instinct_config()
    flock = (10.0, 10.0)
    target = (20.0, 10.0)
    inside_inputs = _base_inputs(
        dog_positions=((10.0, 10.0),),
        sheep_positions=((10.0, 10.0),),
        flock_centroid=flock,
        previous_flock_centroid=flock,
        target_position=target,
    )

    breakdown = RewardComputer(config).compute(inside_inputs)

    assert breakdown.chaos_penalty < 0
    assert breakdown.overpressure_penalty < 0


def test_split_flock_penalty_triggers_when_one_sheep_strays() -> None:
    config = _instinct_config()
    flock = (10.0, 10.0)
    target = (20.0, 10.0)
    inputs = _base_inputs(
        previous_flock_spread=1.0,
        current_flock_spread=1.0,
        dog_positions=((4.0, 10.0),),
        sheep_positions=((10.0, 10.0), (10.0, 11.0), (24.0, 10.0)),
        flock_centroid=flock,
        previous_flock_centroid=flock,
        target_position=target,
    )

    breakdown = RewardComputer(config).compute(inputs)

    assert breakdown.split_flock_penalty < 0
