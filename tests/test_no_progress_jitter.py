"""Regression tests for best-consolidation gating of no_progress_steps."""

from unittest.mock import patch

from sheepdog.config import EnvironmentConfig, InstinctRewardConfig, LabConfig, RewardConfig
from sheepdog.entities import DogState, Pen, Point, SheepState
from sheepdog.environment import ENV_CONFIG_VERSION, SheepdogEnvironment
from sheepdog.evaluation.scenarios import scenario_from_snapshot


def _consolidation_environment(*, no_progress_window: int = 760) -> SheepdogEnvironment:
    """Create a minimal deterministic environment with collection progress enabled."""
    config = LabConfig(
        environment=EnvironmentConfig(
            width=144,
            height=104,
            dogs=3,
            sheep=6,
            curriculum_stage=34,
            spawn_mix={},
            count_collection_progress=True,
            no_progress_window=no_progress_window,
        ),
        rewards=RewardConfig(instincts=InstinctRewardConfig(enable_instinct_rewards=True)),
    )
    env = SheepdogEnvironment(config)
    env.reset(seed=1234)
    env._pen = Pen(Point(128, 43), 14, 14, "left")
    env._fence_cells = env.pen.fence_cells()
    # Two unpenned sheep symmetric around y=50; four already penned.
    env._sheep = [
        SheepState(0, Point(20, 30), personality="calm"),
        SheepState(1, Point(20, 70), personality="calm"),
        *(SheepState(i, Point(132, 46 + i), penned=True) for i in range(2, 6)),
    ]
    env._dogs = [
        DogState(0, Point(5, 5)),
        DogState(1, Point(6, 5)),
        DogState(2, Point(7, 5)),
    ]
    env.invalidate_role_assignments()
    env._previous_farthest_distance = env._farthest_distance_to_pen()
    return env


def test_env_config_version_bumped() -> None:
    assert ENV_CONFIG_VERSION == "1.2"


def test_spread_oscillation_cannot_reset_or_decrement_no_progress_steps() -> None:
    env = _consolidation_environment()

    # Step 1: genuine new best consolidation (20.0 -> 19.0) while moving away from pen.
    def improve_to_19() -> None:
        env.sheep[0].position = Point(18, 31)
        env.sheep[1].position = Point(18, 69)

    with patch.object(env, "_move_sheep", side_effect=improve_to_19):
        env.step(["wait"] * 3)
    assert env._best_flock_spread == 19.0
    assert env._best_farthest_to_flock_center == 19.0
    assert env._no_progress_steps == 0

    # Subsequent 20.0 <-> 19.0 oscillation must never beat best (19.0) or decrement no_progress_steps.
    for step in range(1, 21):
        y_offset = 0 if step % 2 == 1 else 1

        def oscillate(offset: int = y_offset) -> None:
            env.sheep[0].position = Point(18, 30 + offset)
            env.sheep[1].position = Point(18, 70 - offset)

        with patch.object(env, "_move_sheep", side_effect=oscillate):
            env.step(["wait"] * 3)
        assert env._best_flock_spread == 19.0
        assert env._best_farthest_to_flock_center == 19.0
        assert env._no_progress_steps == step


def test_genuine_consolidation_counts_as_progress_and_penning_reanchors_best() -> None:
    env = _consolidation_environment()

    # Accumulate 5 no-progress steps with no movement.
    for _ in range(5):
        with patch.object(env, "_move_sheep"):
            env.step(["wait"] * 3)
    assert env._no_progress_steps == 5
    assert env._best_flock_spread == 20.0
    assert env._best_farthest_to_flock_center == 20.0

    # Genuine new consolidation: 20.0 -> 15.0 (while x moves away from pen).
    def consolidate_15() -> None:
        env.sheep[0].position = Point(16, 35)
        env.sheep[1].position = Point(16, 65)

    with patch.object(env, "_move_sheep", side_effect=consolidate_15):
        env.step(["wait"] * 3)
    assert env._best_flock_spread == 15.0
    assert env._best_farthest_to_flock_center == 15.0
    assert env._no_progress_steps == 4

    # Further consolidation: 15.0 -> 12.0.
    def consolidate_12() -> None:
        env.sheep[0].position = Point(14, 38)
        env.sheep[1].position = Point(14, 62)

    with patch.object(env, "_move_sheep", side_effect=consolidate_12):
        env.step(["wait"] * 3)
    assert env._best_flock_spread == 12.0
    assert env._best_farthest_to_flock_center == 12.0
    assert env._no_progress_steps == 3

    # Add a third unpenned sheep far away, then pen sheep[1] so geometry changes and re-anchors.
    env.sheep[2].penned = False
    env.sheep[2].position = Point(14, 80)

    def pen_one_sheep() -> None:
        env.sheep[1].penned = True
        env.sheep[1].position = Point(132, 48)

    with patch.object(env, "_move_sheep", side_effect=pen_one_sheep):
        env.step(["wait"] * 3)
    # Remaining unpenned sheep are at (14, 38) and (14, 80) -> spread/farthest = 21.0.
    assert env._best_flock_spread == 21.0
    assert env._best_farthest_to_flock_center == 21.0
    assert env._no_progress_steps == 2


def test_reset_and_reset_from_scenario_clear_best_consolidation_state() -> None:
    env = _consolidation_environment()
    scenario = scenario_from_snapshot(
        name="consolidation reset", seed=42, snapshot=env.get_state_snapshot().to_dict()
    )
    with patch.object(env, "_move_sheep"):
        env.step(["wait"] * 3)
    assert env._best_flock_spread is not None
    assert env._best_farthest_to_flock_center is not None

    env.reset(seed=99)
    assert env._best_flock_spread is None
    assert env._best_farthest_to_flock_center is None

    with patch.object(env, "_move_sheep"):
        env.step(["wait"] * 3)
    assert env._best_flock_spread is not None
    assert env._best_farthest_to_flock_center is not None

    env.reset_from_scenario(scenario)
    assert env._best_flock_spread is None
    assert env._best_farthest_to_flock_center is None
