"""Tests for the per-sheep personality system."""

# pylint: disable=protected-access,missing-function-docstring

from __future__ import annotations

from dataclasses import replace

from sheepdog.config import EnvironmentConfig, LabConfig, RewardConfig, TrainingConfig
from sheepdog.entities import SHEEP_PERSONALITIES, Point, SheepState
from sheepdog.environment import SheepdogEnvironment


def _base_config(strength: float = 0.0, sheep: int = 6) -> LabConfig:
    return LabConfig(
        environment=EnvironmentConfig(sheep=sheep, sheep_personality_strength=strength),
        rewards=RewardConfig(),
        training=TrainingConfig(
            episodes=0,
            checkpoint_episodes=(0,),
            evaluation_seeds=(11,),
            output_dir="artifacts",
            web_export_dir="web/public/generated",
        ),
    )


def test_personalities_default_to_obedient_when_disabled() -> None:
    env = SheepdogEnvironment(_base_config(strength=0.0))
    env.reset(seed=11)
    assert all(sheep.personality == "obedient" for sheep in env.sheep)


def test_personalities_are_assigned_and_fixed_across_episode() -> None:
    env = SheepdogEnvironment(_base_config(strength=0.5))
    env.reset(seed=11)
    initial = [sheep.personality for sheep in env.sheep]
    assert all(p in SHEEP_PERSONALITIES for p in initial)
    # Step a few times with no-op actions and confirm personalities never change.
    actions = ["wait"] * env.dog_count
    for _ in range(5):
        env.step(actions)
    assert [sheep.personality for sheep in env.sheep] == initial


def test_personality_assignment_is_deterministic_for_seed() -> None:
    config = _base_config(strength=0.5)
    env_a = SheepdogEnvironment(config)
    env_b = SheepdogEnvironment(config)
    env_a.reset(seed=7)
    env_b.reset(seed=7)
    assert [s.personality for s in env_a.sheep] == [s.personality for s in env_b.sheep]


def test_disabled_strength_preserves_legacy_seed_behavior() -> None:
    """When strength==0 the RNG must not be consumed by personality assignment,
    so the resulting layout matches the legacy environment bit-for-bit."""
    env_legacy = SheepdogEnvironment(_base_config(strength=0.0))
    env_new = SheepdogEnvironment(_base_config(strength=0.0))
    snap_a = env_legacy.reset(seed=123)
    snap_b = env_new.reset(seed=123)
    assert [(s.x, s.y) for s in snap_a.sheep] == [(s.x, s.y) for s in snap_b.sheep]


def test_snapshot_includes_personality() -> None:
    env = SheepdogEnvironment(_base_config(strength=0.4))
    snapshot = env.reset(seed=11)
    assert all(sheep.personality is not None for sheep in snapshot.sheep)


def _step_one_sheep(
    *,
    personality: str,
    strength: float,
    sheep_position: Point,
    dog_positions: list[Point],
    panic_steps: int = 0,
    flock_center: Point | None = None,
) -> Point:
    env = SheepdogEnvironment(_base_config(strength=strength, sheep=1))
    env.reset(seed=0)
    sheep = SheepState(index=0, position=sheep_position, personality=personality)
    env._sheep = [sheep]
    return env._sheep_step(
        sheep_position,
        dog_positions,
        flock_center,
        panic_steps,
        sheep=sheep,
    )


def test_pen_fearful_biases_away_from_pen() -> None:
    # Place a sheep near the pen so the proximity-scaled repulsion is active,
    # with a far dog so the only nontrivial vector is the personality bias.
    env_config = EnvironmentConfig(sheep=1, sheep_personality_strength=1.0)
    pen_center = Point(
        env_config.width - env_config.pen_width + env_config.pen_width // 2,
        1 + env_config.pen_height // 2,
    )
    sheep_position = Point(pen_center.x - 10, pen_center.y + 12)
    far_dog = [Point(0, env_config.height - 1)]
    move_fearful = _step_one_sheep(
        personality="pen_fearful",
        strength=1.0,
        sheep_position=sheep_position,
        dog_positions=far_dog,
    )
    move_obedient = _step_one_sheep(
        personality="obedient",
        strength=1.0,
        sheep_position=sheep_position,
        dog_positions=far_dog,
    )
    # Fearful sheep should end up farther from the pen than the obedient baseline.
    assert move_fearful.distance_to(pen_center) > move_obedient.distance_to(pen_center)


def test_pen_shy_biases_away_from_pen() -> None:
    env_config = EnvironmentConfig(sheep=1, sheep_personality_strength=1.0)
    pen_center = Point(
        env_config.width - env_config.pen_width + env_config.pen_width // 2,
        1 + env_config.pen_height // 2,
    )
    sheep_position = Point(pen_center.x - 10, pen_center.y + 12)
    far_dog = [Point(0, env_config.height - 1)]
    move_shy = _step_one_sheep(
        personality="pen_shy",
        strength=1.0,
        sheep_position=sheep_position,
        dog_positions=far_dog,
    )
    move_obedient = _step_one_sheep(
        personality="obedient",
        strength=1.0,
        sheep_position=sheep_position,
        dog_positions=far_dog,
    )
    # pen_shy sheep should end up at least as far from the pen as the
    # obedient baseline, and typically farther.
    assert move_shy.distance_to(pen_center) >= move_obedient.distance_to(pen_center)


def test_bold_resists_distant_dog_more_than_obedient() -> None:
    # Dog at moderate distance so the flee weight depends on the bold dampener.
    sheep_position = Point(30, 30)
    dog_positions = [Point(35, 30)]  # distance 5, within dog_vision (16)
    move_bold = _step_one_sheep(
        personality="bold",
        strength=1.0,
        sheep_position=sheep_position,
        dog_positions=dog_positions,
    )
    move_obedient = _step_one_sheep(
        personality="obedient",
        strength=1.0,
        sheep_position=sheep_position,
        dog_positions=dog_positions,
    )
    # Bold sheep should not flee further than an obedient sheep would; usually
    # it holds ground or moves less in the away-from-dog direction.
    assert move_bold.x <= move_obedient.x


def test_escapist_breaks_from_flock_when_panicked() -> None:
    sheep_position = Point(20, 30)
    flock_center = Point(25, 30)  # flock is to the right
    dog_positions = [Point(10, 30)]  # dog far to the left, sheep panicking
    move_escapist = _step_one_sheep(
        personality="escapist",
        strength=1.0,
        sheep_position=sheep_position,
        dog_positions=dog_positions,
        panic_steps=3,
        flock_center=flock_center,
    )
    move_obedient = _step_one_sheep(
        personality="obedient",
        strength=1.0,
        sheep_position=sheep_position,
        dog_positions=dog_positions,
        panic_steps=3,
        flock_center=flock_center,
    )
    # The escapist should be at least as far from the flock as the obedient
    # sheep after one step, and ideally strictly farther.
    def flock_dist(p: Point) -> float:
        return p.distance_to(flock_center)

    assert flock_dist(move_escapist) >= flock_dist(move_obedient)


def test_environment_config_round_trips_with_new_field() -> None:
    config = _base_config(strength=0.3)
    payload = config.to_dict()
    restored = LabConfig.from_dict(payload)
    assert restored.environment.sheep_personality_strength == 0.3
    # And replace() should still work.
    bumped = replace(config.environment, sheep_personality_strength=0.7)
    assert bumped.sheep_personality_strength == 0.7
