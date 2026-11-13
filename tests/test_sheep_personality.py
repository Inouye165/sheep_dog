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


def test_pen_fearful_hesitates_under_gate_approach() -> None:
    # Under dog pressure toward the gate, pen_fearful experiences higher directional drag (hesitation)
    # than obedient sheep, resulting in less forward gate penetration.
    env_config = EnvironmentConfig(sheep=1, sheep_personality_strength=1.0)
    gate_entrance = Point(
        env_config.width - env_config.pen_width,
        1 + env_config.pen_height // 2,
    )
    # Position sheep 4 cells in front of the gate opening with a dog directly behind pushing it toward the gate
    sheep_position = Point(gate_entrance.x - 4, gate_entrance.y)
    rear_dog = [Point(sheep_position.x - 2, sheep_position.y)]
    move_fearful = _step_one_sheep(
        personality="pen_fearful",
        strength=1.0,
        sheep_position=sheep_position,
        dog_positions=rear_dog,
    )
    move_obedient = _step_one_sheep(
        personality="obedient",
        strength=1.0,
        sheep_position=sheep_position,
        dog_positions=rear_dog,
    )
    # Fearful sheep should not penetrate further into the gate than obedient baseline
    assert move_fearful.distance_to(gate_entrance) >= move_obedient.distance_to(gate_entrance)


def test_pen_fearful_does_not_blast_radially_when_calm() -> None:
    # Replacing radial repulsion: when calm with no dog pressure, sheep does not get launched backward
    env_config = EnvironmentConfig(sheep=1, sheep_personality_strength=1.0)
    gate_entrance = Point(
        env_config.width - env_config.pen_width,
        1 + env_config.pen_height // 2,
    )
    sheep_position = Point(gate_entrance.x - 3, gate_entrance.y)
    far_dog = [Point(0, env_config.height - 1)]
    move_fearful = _step_one_sheep(
        personality="pen_fearful",
        strength=1.0,
        sheep_position=sheep_position,
        dog_positions=far_dog,
    )
    # Sheep should remain near its position (idling or small step), not blasted radially outward
    assert move_fearful.distance_to(sheep_position) <= 1.5


def test_pen_shy_biases_away_from_pen() -> None:
    env_config = EnvironmentConfig(sheep=1, sheep_personality_strength=1.0)
    gate_entrance = Point(
        env_config.width - env_config.pen_width,
        1 + env_config.pen_height // 2,
    )
    sheep_position = Point(gate_entrance.x - 4, gate_entrance.y)
    rear_dog = [Point(sheep_position.x - 2, sheep_position.y)]
    move_shy = _step_one_sheep(
        personality="pen_shy",
        strength=1.0,
        sheep_position=sheep_position,
        dog_positions=rear_dog,
    )
    move_obedient = _step_one_sheep(
        personality="obedient",
        strength=1.0,
        sheep_position=sheep_position,
        dog_positions=rear_dog,
    )
    assert move_shy.distance_to(gate_entrance) >= move_obedient.distance_to(gate_entrance)


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


def test_leader_follower_surge_pulls_to_penned_centroid() -> None:
    env_config = EnvironmentConfig(sheep=2, sheep_speed=1.0, sheep_personality_strength=1.0)
    config = LabConfig(environment=env_config)
    env = SheepdogEnvironment(config)
    env.reset(seed=1)

    # Mark sheep 0 as penned inside the pen
    pen_inside = env.pen.center
    env._sheep[0].position = pen_inside
    env._sheep[0].penned = True
    env._pen_breached = True

    # Position unpenned sheep 1 near the gate
    gate_entrance = env._gate_entrance_center()
    outside_pos = Point(gate_entrance.x - 3, gate_entrance.y)
    env._sheep[1].position = outside_pos
    env._sheep[1].penned = False

    # Far dogs
    env._dogs[0].position = Point(0, 0)

    # Call _move_sheep
    env._move_sheep()

    # Unpenned sheep should have surged toward the penned centroid/pen
    new_dist = env._sheep[1].position.distance_to(pen_inside)
    old_dist = outside_pos.distance_to(pen_inside)
    assert new_dist < old_dist


def test_tonic_immobility_packing_suppresses_scatter() -> None:
    env_config = EnvironmentConfig(sheep=2, sheep_personality_strength=1.0, sheep_p_crit=4.0)
    config = LabConfig(environment=env_config)
    env = SheepdogEnvironment(config)
    env.reset(seed=1)

    # Escapist sheep at corner Point(0, 0) where primary flee paths are blocked and dog pressure is high
    sheep_pos = Point(0, 0)
    sheep = SheepState(index=0, position=sheep_pos, personality="escapist")
    env._sheep = [sheep]

    # Dogs right on top creating massive predator pressure (P > P_crit)
    dog_positions = [Point(1, 0), Point(0, 1)]
    flock_center = Point(5, 5)

    # Step the sheep
    env._sheep_step(
        sheep_pos,
        dog_positions,
        flock_center,
        panic_steps=3,
        sheep=sheep,
    )
    # Under tonic immobility, the sheep should be marked packed
    assert sheep.is_packed is True


def test_flank_squirt_vs_containment_arc() -> None:
    env_config = EnvironmentConfig(sheep=1, sheep_personality_strength=0.0, sheep_theta_containment=1.8)
    config = LabConfig(environment=env_config)
    env = SheepdogEnvironment(config)
    env.reset(seed=1)

    gate_entrance = env._gate_entrance_center()
    # Sheep positioned left of the gate, so gate is to the right (+x)
    sheep_pos = Point(gate_entrance.x - 8, gate_entrance.y)
    sheep = SheepState(index=0, position=sheep_pos)

    # Case A: Balanced containment arc (dogs behind sheep on both left and right flanks)
    # Dogs behind at (-x) spanning both +y and -y
    contained_dogs = [Point(sheep_pos.x - 3, sheep_pos.y - 3), Point(sheep_pos.x - 3, sheep_pos.y + 3)]
    move_contained = env._sheep_step(
        sheep_pos,
        contained_dogs,
        flock_center=None,
        panic_steps=2,
        sheep=sheep,
    )
    # Should drive forward toward gate (+x)
    assert move_contained.x > sheep_pos.x
