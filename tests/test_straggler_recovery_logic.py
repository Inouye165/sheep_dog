"""Unit tests for straggler recovery role assignment and observation prioritization."""

from sheepdog.config import EnvironmentConfig, LabConfig
from sheepdog.entities import DogRole, DogState, Pen, Point, SheepState
from sheepdog.environment import SheepdogEnvironment
from sheepdog.observations import RoleAwareObservationBuilder
from sheepdog.team_strategy import TeamStrategy


def test_single_unpenned_sheep_assigned_collector_role():
    w, h = 108, 78
    config = EnvironmentConfig(width=w, height=h)
    strategy = TeamStrategy(w, h, config)
    pen = Pen(Point(94, 1), 14, 14, "bottom")

    dogs = [
        DogState(0, Point(105, 15)),
        DogState(1, Point(106, 15)),
        DogState(2, Point(107, 15)),
    ]
    sheep = [
        SheepState(0, Point(100, 10), penned=True),
        SheepState(1, Point(102, 10), penned=True),
        SheepState(2, Point(104, 10), penned=True),
        SheepState(3, Point(16, 19), penned=False),
    ]

    assignments, snapshot = strategy.assign_roles(dogs, sheep, pen)

    # Verify one dog gets assigned COLLECTOR for sheep index 3
    collector_assignments = [a for a in assignments.values() if a.role == DogRole.COLLECTOR]
    assert len(collector_assignments) == 1, f"Expected 1 collector assignment, got {len(collector_assignments)}"
    collector = collector_assignments[0]
    assert collector.target_sheep_index == 3
    assert collector.reason == "split_sheep_detected"

    # Target should be positioned to drive sheep 3 toward pen (pen center is ~101, 8, sheep is at 16, 19)
    # Target should be behind sheep relative to pen (x < 16)
    assert collector.target.x < 16, f"Expected collector target x < 16, got {collector.target.x}"


def test_observation_prioritizes_unpenned_sheep():
    config = LabConfig()
    env = SheepdogEnvironment(config)
    env.reset(seed=100)

    # Set up scenario: dogs at pen (105, 15), sheep 0-2 penned inside pen (100, 10), sheep 3 unpenned far away (16, 19)
    env._sheep = [
        SheepState(0, Point(100, 10), penned=True),
        SheepState(1, Point(102, 10), penned=True),
        SheepState(2, Point(104, 10), penned=True),
        SheepState(3, Point(16, 19), penned=False),
    ]
    env._dogs = [
        DogState(0, Point(105, 15)),
        DogState(1, Point(106, 15)),
        DogState(2, Point(107, 15)),
    ]

    builder = RoleAwareObservationBuilder()
    obs = builder.build(env, dog_index=0)
    feat_dict = obs.as_feature_dict()

    # sheep_0 slot MUST represent the unpenned sheep (sheep 3), so sheep_0_penned MUST be 0.0
    assert feat_dict["sheep_0_penned"] == 0.0, "Expected sheep_0 to be unpenned sheep, but sheep_0_penned == 1.0"

    # sheep_0 relative coordinates should point to (16, 19) relative to dog at (105, 15)
    # dx = (16 - 105) / field_width = -89 / 108 ~ -0.824
    assert feat_dict["sheep_0_dx"] < 0, f"Expected sheep_0_dx < 0, got {feat_dict['sheep_0_dx']}"
