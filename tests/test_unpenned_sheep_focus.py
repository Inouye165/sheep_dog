"""Unit tests verifying that unpenned sheep take priority in observations and shepherd commands."""


from sheepdog.config import EnvironmentConfig, LabConfig
from sheepdog.entities import Point
from sheepdog.environment import SheepdogEnvironment
from sheepdog.observations import EmergentObservationBuilder
from sheepdog.shepherd import ScriptedShepherd, ShepherdCommand, ShepherdContext


def test_observation_builder_sorts_unpenned_sheep_first():
    """Verify that unpenned sheep always occupy sheep_0 in observations regardless of physical proximity to penned sheep."""
    config = EnvironmentConfig(width=32, height=32, dogs=2, sheep=4)
    env = SheepdogEnvironment(LabConfig(environment=config))
    env.reset(seed=42)

    # Place dog 0 right next to pen (27, 2)
    env.dogs[0].position = Point(27, 2)

    # 3 sheep inside pen at (28, 2), (28, 3), (29, 2) -> penned
    env.sheep[0].position = Point(28, 2)
    env.sheep[0].penned = True
    env.sheep[1].position = Point(28, 3)
    env.sheep[1].penned = True
    env.sheep[2].position = Point(29, 2)
    env.sheep[2].penned = True

    # 1 sheep far away at (5, 5) -> unpenned
    env.sheep[3].position = Point(5, 5)
    env.sheep[3].penned = False

    builder = EmergentObservationBuilder()
    obs = builder.build(env, dog_index=0)

    # Extract sheep_0_penned feature value
    sheep_0_penned_idx = obs.feature_names.index("sheep_0_penned")
    sheep_0_penned_val = obs.values[sheep_0_penned_idx]

    # sheep_0 must be unpenned (0.0)
    assert sheep_0_penned_val == 0.0, f"Expected sheep_0_penned == 0.0, got {sheep_0_penned_val}"


def test_shepherd_bypasses_escape_holding_for_single_unpenned_sheep():
    """Verify that when 3 of 4 sheep are penned, shepherd commands APPLY_PRESSURE rather than BLOCK_ESCAPE."""
    shepherd = ScriptedShepherd()

    # Scenario: 3 of 4 sheep penned, near pen, escape left and right true
    ctx = ShepherdContext(
        flock_center_x=20.0,
        flock_center_y=5.0,
        flock_spread=1.0,
        average_distance_to_pen=8.0,
        sheep_penned=3,
        total_sheep=4,
        average_dog_sheep_distance=5.0,
        escape_left=True,
        escape_right=True,
        near_pen=True,
    )

    cmd = shepherd._decide(ctx, None)

    # Should NOT return BLOCK_ESCAPE; should return APPLY_PRESSURE to drive final sheep
    assert cmd == ShepherdCommand.APPLY_PRESSURE
