"""Regression tests for the herding environment."""

# pylint: disable=protected-access,missing-function-docstring

from __future__ import annotations

from dataclasses import asdict, replace
from unittest.mock import patch

from sheepdog.config import EnvironmentConfig, LabConfig, PolicyConfig, RewardConfig, TrainingConfig
from sheepdog.entities import DogRole, Point
from sheepdog.environment import SheepdogEnvironment
from sheepdog.policies.heuristic import HeuristicExpertPolicy
from sheepdog.policies.trainable import PolicyWeights
from sheepdog.team_strategy import RoleAssignment


def make_config(**environment_overrides):
    environment = EnvironmentConfig(**environment_overrides)
    return LabConfig(
        environment=environment,
        rewards=RewardConfig(),
        training=TrainingConfig(
            episodes=0,
            checkpoint_episodes=(0,),
            evaluation_seeds=(11,),
            output_dir="artifacts",
            web_export_dir="web/public/generated",
        ),
    )


def role_weights(**overrides):
    payload = {name: 0.0 for name in asdict(PolicyWeights())}
    payload.update(overrides)
    return PolicyWeights(**payload)


def test_reset_is_deterministic_for_fixed_seed() -> None:
    config = make_config()
    first = SheepdogEnvironment(config).reset(seed=42).to_dict()
    second = SheepdogEnvironment(config).reset(seed=42).to_dict()

    assert first == second


def test_reset_starts_with_unique_dog_and_sheep_cells() -> None:
    config = make_config(dogs=4, sheep=6)
    snapshot = SheepdogEnvironment(config).reset(seed=42)

    dog_positions = {(dog.x, dog.y) for dog in snapshot.dogs}
    sheep_positions = {(sheep.x, sheep.y) for sheep in snapshot.sheep}

    assert len(dog_positions) == len(snapshot.dogs)
    assert len(sheep_positions) == len(snapshot.sheep)
    assert dog_positions.isdisjoint(sheep_positions)


def test_sheep_flee_from_nearby_dog() -> None:
    config = make_config()
    environment = SheepdogEnvironment(config)
    environment.reset(seed=7)
    sheep = environment.sheep[0]
    dog = environment.dogs[0]
    sheep.position = sheep.position.__class__(8, 8)
    dog.position = dog.position.__class__(7, 8)

    before = sheep.position
    environment.step(["wait"] * config.environment.dogs)

    after = environment.sheep[0].position
    assert after.x >= before.x


def test_snapshot_exposes_field_dimension_aliases() -> None:
    snapshot = SheepdogEnvironment(make_config()).reset(seed=2)

    assert snapshot.field_width == snapshot.grid_width
    assert snapshot.field_height == snapshot.grid_height


def test_default_environment_uses_doubled_grid_resolution() -> None:
    config = make_config()

    assert config.environment.width == 80
    assert config.environment.height == 60
    assert config.environment.pen_width == 10
    assert config.environment.pen_height == 10


def test_sheep_enter_pen_and_counted() -> None:
    config = make_config()
    environment = SheepdogEnvironment(config)
    snapshot = environment.reset(seed=11)
    pen = snapshot.pen

    for sheep in environment.sheep:
        sheep.position = sheep.position.__class__(pen.origin.x, pen.origin.y)
        sheep.penned = False

    next_snapshot, _ = environment.step(["wait"] * config.environment.dogs)

    assert next_snapshot.penned_count == config.environment.sheep
    assert next_snapshot.success is True


def test_episode_stops_when_no_progress_occurs() -> None:
    config = make_config(no_progress_window=1, no_progress_distance_delta=100.0)
    environment = SheepdogEnvironment(config)
    environment.reset(seed=5)

    snapshot, _ = environment.step(["wait"] * config.environment.dogs)

    assert snapshot.stopped is True
    assert snapshot.timeout is False


def test_collection_progress_counts_flock_spread_reduction_as_progress() -> None:
    config = make_config(
        curriculum_stage=9,
        count_collection_progress=True,
        no_progress_distance_delta=0.2,
    )
    environment = SheepdogEnvironment(config)

    progress = environment._progress_made_for_no_progress(
        newly_penned=0,
        average_distance_delta=0.0,
        flock_spread_delta=0.10,
        farthest_to_flock_center_delta=0.0,
        farthest_to_pen_delta=0.0,
    )

    assert progress is True


def test_collection_progress_counts_farthest_sheep_progress_as_progress() -> None:
    config = make_config(
        curriculum_stage=9,
        count_collection_progress=True,
        no_progress_distance_delta=0.2,
    )
    environment = SheepdogEnvironment(config)

    progress = environment._progress_made_for_no_progress(
        newly_penned=0,
        average_distance_delta=0.0,
        flock_spread_delta=0.0,
        farthest_to_flock_center_delta=0.11,
        farthest_to_pen_delta=0.0,
    )

    assert progress is True


def test_non_collection_stage_ignores_spread_only_progress_signal() -> None:
    config = make_config(curriculum_stage=5, no_progress_distance_delta=0.2)
    environment = SheepdogEnvironment(config)

    progress = environment._progress_made_for_no_progress(
        newly_penned=0,
        average_distance_delta=0.0,
        flock_spread_delta=0.08,
        farthest_to_flock_center_delta=0.11,
        farthest_to_pen_delta=0.11,
    )

    assert progress is False


def test_legacy_stage_6_without_spawn_mix_keeps_collection_progress() -> None:
    config = make_config(curriculum_stage=6, no_progress_distance_delta=0.2)
    environment = SheepdogEnvironment(config)

    progress = environment._progress_made_for_no_progress(
        newly_penned=0,
        average_distance_delta=0.0,
        flock_spread_delta=0.10,
        farthest_to_flock_center_delta=0.0,
        farthest_to_pen_delta=0.0,
    )

    assert progress is True


def test_action_mask_disallows_useless_wait_when_movement_exists() -> None:
    # Heuristic modes must preserve the wait-scoring threshold gate.
    config = make_config()
    environment = SheepdogEnvironment(config)
    environment.reset(seed=3)

    mask = environment.action_mask_for_dog(0, policy_mode="instinct_only")

    assert mask["wait"] is False


def test_wait_always_legal_in_neural_policy_mode() -> None:
    # neural_policy must expose wait even when the heuristic score would suppress it.
    config = make_config()
    environment = SheepdogEnvironment(config)
    environment.reset(seed=3)

    # seed=3 is the same position where instinct_only suppresses wait;
    # neural_policy must override that gate.
    mask = environment.action_mask_for_dog(0, policy_mode="neural_policy")

    assert mask["wait"] is True


def test_neural_action_mask_skips_heuristic_action_scoring() -> None:
    environment = SheepdogEnvironment(make_config())
    environment.reset(seed=12)

    with patch.object(environment, "_action_score", side_effect=AssertionError):
        mask = environment.action_mask_for_dog(0, policy_mode="neural_policy")

    assert mask["wait"] is True


def test_wait_always_legal_in_shepherd_neural_dogs_mode() -> None:
    config = make_config()
    environment = SheepdogEnvironment(config)
    environment.reset(seed=3)

    mask = environment.action_mask_for_dog(0, policy_mode="shepherd_neural_dogs")

    assert mask["wait"] is True


def test_wait_always_legal_when_policy_mode_is_none() -> None:
    # policy_mode=None is what both RL adapters use during training.
    config = make_config()
    environment = SheepdogEnvironment(config)
    environment.reset(seed=3)

    mask = environment.action_mask_for_dog(0, policy_mode=None)

    assert mask["wait"] is True


def test_heuristic_expert_mode_preserves_wait_threshold() -> None:
    config = make_config()
    environment = SheepdogEnvironment(config)
    environment.reset(seed=3)

    mask_heuristic = environment.action_mask_for_dog(0, policy_mode="heuristic_expert")
    mask_instinct = environment.action_mask_for_dog(0, policy_mode="instinct_only")

    # Both heuristic modes must agree: wait is gated by scoring threshold.
    assert mask_heuristic["wait"] == mask_instinct["wait"]


def test_non_wait_actions_unchanged_by_neural_mode() -> None:
    # Switching to neural mode must not alter movement action legality.
    config = make_config()
    environment = SheepdogEnvironment(config)
    environment.reset(seed=3)

    mask_heuristic = environment.action_mask_for_dog(0, policy_mode="instinct_only")
    mask_neural = environment.action_mask_for_dog(0, policy_mode="neural_policy")

    non_wait_actions = [action for action in mask_heuristic if action != "wait"]
    for action in non_wait_actions:
        assert mask_neural[action] == mask_heuristic[action], (
            f"action '{action}' legality changed between instinct_only and neural_policy"
        )


def test_multiple_dogs_keep_consistent_state_updates() -> None:
    config = make_config(dogs=4)
    environment = SheepdogEnvironment(config)
    environment.reset(seed=9)

    snapshot, _ = environment.step(["right", "left", "up", "down"])

    assert len(snapshot.dogs) == 4
    assert {dog.index for dog in snapshot.dogs} == {0, 1, 2, 3}


def test_grouped_flock_assigns_rear_and_flank_roles() -> None:
    config = make_config(dogs=3, sheep=3)
    environment = SheepdogEnvironment(config)
    environment.reset(seed=9)
    for index, sheep in enumerate(environment.sheep):
        sheep.position = sheep.position.__class__(18 + index, 10 + index)

    roles = environment.current_role_assignments()

    assert set(roles.values()) == {
        DogRole.REAR_PRESSURE.value,
        DogRole.LEFT_FLANKER.value,
        DogRole.RIGHT_FLANKER.value,
    }


def test_split_sheep_assigns_collector() -> None:
    config = make_config(dogs=3, sheep=3)
    environment = SheepdogEnvironment(config)
    environment.reset(seed=10)
    environment.sheep[0].position = environment.sheep[0].position.__class__(18, 10)
    environment.sheep[1].position = environment.sheep[1].position.__class__(19, 11)
    environment.sheep[2].position = environment.sheep[2].position.__class__(6, 24)

    roles = environment.current_role_assignments()

    assert DogRole.COLLECTOR.value in roles.values()


def test_role_aware_observation_includes_expected_features() -> None:
    config = make_config(dogs=3, sheep=3)
    environment = SheepdogEnvironment(config)
    environment.reset(seed=14)
    environment.dogs[0].position = Point(10, 10)
    environment.sheep[0].position = Point(18, 10)
    environment.sheep[1].position = Point(20, 11)
    environment.sheep[2].position = Point(6, 24)

    observation = environment.build_observation_for_dog(0)
    feature_map = observation.as_feature_dict()

    assert observation.role in {role.value for role in DogRole}
    assert feature_map["own_x"] == 10 / (config.environment.width - 1)
    assert "role_collector" in feature_map
    assert "focus_sheep_dx" in feature_map
    assert "sheep_0_dx" in feature_map
    assert "other_dog_0_dx" in feature_map


def test_near_pen_flock_assigns_blocker_and_flanker() -> None:
    config = make_config(dogs=3, sheep=3)
    environment = SheepdogEnvironment(config)
    snapshot = environment.reset(seed=11)
    pen = snapshot.pen
    for index, sheep in enumerate(environment.sheep):
        sheep.position = sheep.position.__class__(pen.origin.x - 4, pen.origin.y + 2 + index)

    roles = environment.current_role_assignments()

    assert DogRole.BLOCKER.value in roles.values()
    assert any(
        role in roles.values() for role in (DogRole.LEFT_FLANKER.value, DogRole.RIGHT_FLANKER.value)
    )


def test_roles_change_when_flock_state_changes() -> None:
    config = make_config(dogs=3, sheep=3)
    environment = SheepdogEnvironment(config)
    environment.reset(seed=12)
    for index, sheep in enumerate(environment.sheep):
        sheep.position = sheep.position.__class__(18 + index, 10)
    first_roles = environment.current_role_assignments()

    environment.invalidate_role_assignments()
    environment.sheep[2].position = environment.sheep[2].position.__class__(6, 24)
    second_roles = environment.current_role_assignments()

    assert first_roles != second_roles
    assert DogRole.COLLECTOR.value in second_roles.values()


def test_roles_stay_stable_when_flock_only_moves_slightly() -> None:
    config = make_config(dogs=3, sheep=3)
    environment = SheepdogEnvironment(config)
    environment.reset(seed=15)
    environment.dogs[0].position = Point(14, 16)
    environment.dogs[1].position = Point(13, 21)
    environment.dogs[2].position = Point(13, 11)
    for index, sheep in enumerate(environment.sheep):
        sheep.position = Point(20 + index, 16)

    first_roles = environment.current_role_assignments()

    environment.invalidate_role_assignments()
    for sheep in environment.sheep:
        sheep.position = Point(sheep.position.x, sheep.position.y + 1)
    second_roles = environment.current_role_assignments()

    assert first_roles == second_roles


def test_dog_speed_is_used_for_clear_moves() -> None:
    config = make_config(dogs=1, dog_speed=2, sheep=0)
    environment = SheepdogEnvironment(config)
    environment.reset(seed=3)
    dog = environment.dogs[0]
    dog.position = dog.position.__class__(5, 5)

    environment.step(["right"])

    assert environment.dogs[0].position.x == 7


def test_sprint_action_moves_farther_than_walk() -> None:
    config = make_config(dogs=1, sheep=0)
    environment = SheepdogEnvironment(config)
    environment.reset(seed=31)
    environment.dogs[0].position = Point(5, 5)

    environment.step(["sprint_right"])

    assert environment.dogs[0].position.x == 7


def test_action_mask_blocks_sprint_into_walls() -> None:
    config = make_config(dogs=1, sheep=0)
    environment = SheepdogEnvironment(config)
    environment.reset(seed=32)
    environment.dogs[0].position = Point(0, 0)

    mask = environment.action_mask_for_dog(0)

    assert mask["up"] is False
    assert mask["left"] is False
    assert mask["sprint_up"] is False
    assert mask["sprint_left"] is False


def test_valid_action_is_not_counted_invalid_after_reaching_wall() -> None:
    config = make_config(dogs=1, sheep=0)
    environment = SheepdogEnvironment(config)
    environment.reset(seed=32)
    environment.dogs[0].position = Point(config.environment.width - 2, 5)

    environment.step(["right"])

    assert environment._stats.num_invalid_actions == 0


def test_default_dog_speed_moves_one_cell() -> None:
    config = make_config(dogs=1, sheep=0)
    environment = SheepdogEnvironment(config)
    environment.reset(seed=4)
    dog = environment.dogs[0]
    dog.position = dog.position.__class__(5, 5)

    environment.step(["right"])

    assert environment.dogs[0].position.x == 6


def test_fractional_sheep_speed_moves_three_cells_in_four_steps() -> None:
    config = make_config(dogs=1, sheep=1, dog_vision=20, sheep_speed=0.75)
    environment = SheepdogEnvironment(config)
    environment.reset(seed=21)

    sheep = environment.sheep[0]
    dog = environment.dogs[0]
    sheep.position = sheep.position.__class__(20, 20)
    dog.position = dog.position.__class__(10, 20)

    start_x = sheep.position.x
    for _ in range(4):
        environment.step(["wait"])

    assert environment.sheep[0].position.x - start_x == 3


def test_dog_action_score_prefers_herding_standoff() -> None:
    config = make_config(dogs=1, sheep=1)
    environment = SheepdogEnvironment(config)
    environment.reset(seed=4)

    dog = environment.dogs[0]
    sheep = environment.sheep[0]
    sheep.position = sheep.position.__class__(10, 10)
    dog.position = dog.position.__class__(6, 10)

    better_spacing = environment.score_action_for_dog(0, "right")
    too_far = environment.score_action_for_dog(0, "left")

    assert better_spacing > too_far


def test_sprint_near_sheep_is_scored_lower_than_walk() -> None:
    config = make_config(dogs=1, sheep=1)
    environment = SheepdogEnvironment(config)
    environment.reset(seed=33)
    environment.dogs[0].position = Point(6, 10)
    environment.sheep[0].position = Point(10, 10)

    walk_score = environment.score_action_for_dog(0, "right")
    sprint_score = environment.score_action_for_dog(0, "sprint_right")

    assert walk_score > sprint_score


def test_dog_action_score_prefers_spread_out_team_positions() -> None:
    config = make_config(dogs=2, sheep=1, dog_speed=1)
    environment = SheepdogEnvironment(config)
    environment.reset(seed=6)

    first_dog = environment.dogs[0]
    second_dog = environment.dogs[1]
    first_dog.position = first_dog.position.__class__(6, 10)
    second_dog.position = second_dog.position.__class__(8, 10)
    environment.sheep[0].position = environment.sheep[0].position.__class__(18, 18)

    spacing_weights = type(
        "SpacingWeights",
        (),
        {
            "nearest_sheep": 0.0,
            "flock_center": 0.0,
            "pen_pressure": 0.0,
            "behind_flock": 0.0,
            "team_formation": 0.0,
            "dog_spacing": 5.0,
            "wall_margin": 0.0,
            "wait_bias": 0.0,
            "rear_drive": 0.0,
            "flank_control": 0.0,
            "collector_focus": 0.0,
            "blocker_cover": 0.0,
            "anti_stack_penalty": 0.0,
            "oscillation_penalty": 0.0,
        },
    )()

    spread_out = environment.score_action_for_dog(0, "left", weights=spacing_weights)
    bunched_up = environment.score_action_for_dog(0, "right", weights=spacing_weights)

    assert spread_out > bunched_up


def test_dogs_avoid_stacking_when_target_cells_conflict() -> None:
    config = make_config(dogs=2, sheep=1, dog_speed=1)
    environment = SheepdogEnvironment(config)
    environment.reset(seed=13)
    environment.dogs[0].position = environment.dogs[0].position.__class__(5, 5)
    environment.dogs[1].position = environment.dogs[1].position.__class__(7, 5)
    environment.sheep[0].position = environment.sheep[0].position.__class__(6, 5)

    environment.step(["right", "left"])

    positions = {(dog.position.x, dog.position.y) for dog in environment.dogs}
    assert len(positions) == 2


def test_rear_pressure_behavior_prefers_behind_flock_positions() -> None:
    config = make_config(dogs=1, sheep=3)
    environment = SheepdogEnvironment(config)
    environment.reset(seed=30)
    environment.dogs[0].position = Point(16, 20)
    environment.sheep[0].position = Point(20, 20)
    environment.sheep[1].position = Point(20, 21)
    environment.sheep[2].position = Point(21, 20)
    environment._role_assignments = {
        0: RoleAssignment(0, DogRole.REAR_PRESSURE, Point(15, 20), reason="test_rear")
    }
    environment._roles_prepared_step = environment._step_count

    weights = role_weights(
        rear_drive=0.8,
        rear_behind_flock=2.0,
        rear_drive_to_pen=1.2,
        rear_avoid_overpressure=1.0,
    )

    behind = environment.score_action_for_dog(0, "left", weights=weights)
    overpush = environment.score_action_for_dog(0, "right", weights=weights)

    assert behind > overpush


def test_flanker_behavior_prefers_side_control_positions() -> None:
    config = make_config(dogs=1, sheep=3)
    environment = SheepdogEnvironment(config)
    environment.reset(seed=31)
    environment.dogs[0].position = Point(17, 20)
    environment.sheep[0].position = Point(20, 20)
    environment.sheep[1].position = Point(21, 20)
    environment.sheep[2].position = Point(20, 21)
    environment._role_assignments = {
        0: RoleAssignment(0, DogRole.LEFT_FLANKER, Point(17, 24), reason="test_flank")
    }
    environment._roles_prepared_step = environment._step_count

    weights = role_weights(
        flank_control=0.8,
        flank_side_control=2.0,
        flank_escape_blocking=1.0,
        flank_wall_margin=0.1,
    )

    side_control = environment.score_action_for_dog(0, "down", weights=weights)
    straight_drive = environment.score_action_for_dog(0, "right", weights=weights)

    assert side_control > straight_drive


def test_flank_handedness_bias_differs_for_left_and_right_roles() -> None:
    config = make_config(dogs=1, sheep=3)
    environment = SheepdogEnvironment(config)
    environment.reset(seed=35)
    environment.dogs[0].position = Point(16, 20)
    environment.sheep[0].position = Point(20, 20)
    environment.sheep[1].position = Point(21, 20)
    environment.sheep[2].position = Point(20, 21)

    left_weights = role_weights(flank_control=0.0, flank_side_control=0.0, flank_handedness=3.0)
    right_weights = role_weights(flank_control=0.0, flank_side_control=0.0, flank_handedness=3.0)

    environment._role_assignments = {
        0: RoleAssignment(0, DogRole.LEFT_FLANKER, Point(17, 24), reason="left_flank")
    }
    environment._roles_prepared_step = environment._step_count
    left_clockwise = environment.score_action_for_dog(0, "down", weights=left_weights)
    left_counter_clockwise = environment.score_action_for_dog(0, "up", weights=left_weights)

    environment._role_assignments = {
        0: RoleAssignment(0, DogRole.RIGHT_FLANKER, Point(17, 16), reason="right_flank")
    }
    environment._roles_prepared_step = environment._step_count
    right_counter_clockwise = environment.score_action_for_dog(0, "up", weights=right_weights)
    right_clockwise = environment.score_action_for_dog(0, "down", weights=right_weights)

    assert left_clockwise > left_counter_clockwise
    assert right_counter_clockwise > right_clockwise


def test_collector_behavior_prioritizes_stray_sheep() -> None:
    config = make_config(dogs=1, sheep=3)
    environment = SheepdogEnvironment(config)
    environment.reset(seed=32)
    environment.dogs[0].position = Point(10, 20)
    environment.sheep[0].position = Point(8, 20)
    environment.sheep[1].position = Point(20, 20)
    environment.sheep[2].position = Point(21, 20)
    environment._role_assignments = {
        0: RoleAssignment(
            0,
            DogRole.COLLECTOR,
            Point(6, 20),
            target_sheep_index=0,
            reason="test_collect",
        )
    }
    environment._roles_prepared_step = environment._step_count

    weights = role_weights(
        collector_focus=0.6,
        collector_stray_focus=2.4,
        collector_return_to_flock=0.4,
        collector_rejoin_angle=0.7,
    )

    toward_stray = environment.score_action_for_dog(0, "left", weights=weights)
    abandon_stray = environment.score_action_for_dog(0, "right", weights=weights)

    assert toward_stray > abandon_stray


def test_blocker_behavior_prefers_gate_control_positions() -> None:
    config = make_config(dogs=1, sheep=3)
    environment = SheepdogEnvironment(config)
    snapshot = environment.reset(seed=33)
    pen = snapshot.pen
    environment.dogs[0].position = Point(pen.origin.x - 4, pen.center.y)
    environment.sheep[0].position = Point(pen.origin.x - 6, pen.center.y - 1)
    environment.sheep[1].position = Point(pen.origin.x - 6, pen.center.y)
    environment.sheep[2].position = Point(pen.origin.x - 5, pen.center.y + 1)
    environment._role_assignments = {
        0: RoleAssignment(
            0,
            DogRole.BLOCKER,
            Point(pen.origin.x - 2, pen.center.y),
            reason="test_block",
        )
    }
    environment._roles_prepared_step = environment._step_count

    weights = role_weights(
        blocker_cover=0.8,
        blocker_escape_route_cover=1.0,
        blocker_gate_control=2.2,
        blocker_hold_position=0.6,
    )

    toward_gate = environment.score_action_for_dog(0, "right", weights=weights)
    drift_away = environment.score_action_for_dog(0, "left", weights=weights)

    assert toward_gate > drift_away


def test_blocker_target_stays_off_gate_center_to_preserve_funnel_lane() -> None:
    config = make_config(dogs=3, sheep=3)
    environment = SheepdogEnvironment(config)
    snapshot = environment.reset(seed=36)
    pen = snapshot.pen
    for index, sheep in enumerate(environment.sheep):
        sheep.position = Point(pen.origin.x - 5, pen.center.y - 1 + index)

    environment.prepare_policy_step()
    blocker_assignment = next(
        assignment
        for assignment in environment._role_assignments.values()
        if assignment.role == DogRole.BLOCKER
    )
    gate_position = environment._gate_position()

    assert blocker_assignment.target != gate_position
    assert blocker_assignment.target.distance_to(gate_position) <= 3.5


def test_controlled_stall_metrics_and_penalty_trigger_away_from_gate() -> None:
    config = make_config(dogs=1, sheep=3, stalled_control_activation_steps=2)
    environment = SheepdogEnvironment(config)
    environment.reset(seed=37)
    environment.dogs[0].position = Point(4, 4)
    environment.sheep[0].position = Point(1, 1)
    environment.sheep[1].position = Point(1, 2)
    environment.sheep[2].position = Point(2, 1)

    _, first_breakdown = environment.step(["wait"])
    _, second_breakdown = environment.step(["wait"])

    assert first_breakdown.wrong_hold_penalty == 0.0
    assert second_breakdown.wrong_hold_penalty < 0
    assert environment._stats.controlled_stall_steps >= 2


def test_short_hold_near_gate_is_not_penalized_as_wrong_hold() -> None:
    config = make_config(dogs=1, sheep=2)
    environment = SheepdogEnvironment(config)
    snapshot = environment.reset(seed=38)
    pen = snapshot.pen
    environment.dogs[0].position = Point(pen.origin.x - 3, pen.center.y)
    environment.sheep[0].position = Point(pen.origin.x - 2, pen.center.y)
    environment.sheep[1].position = Point(pen.origin.x - 2, pen.center.y + 1)

    _, breakdown = environment.step(["wait"])

    assert breakdown.wrong_hold_penalty == 0.0


def test_oscillation_penalty_changes_action_score() -> None:
    config = make_config(dogs=1, sheep=0)
    environment = SheepdogEnvironment(config)
    environment.reset(seed=34)
    dog = environment.dogs[0]
    dog.position = Point(6, 5)
    dog.recent_positions = [Point(5, 5), Point(6, 5), Point(5, 5), Point(6, 5)]

    weights = role_weights(oscillation_penalty=4.0)

    revisiting_loop = environment.score_action_for_dog(0, "left", weights=weights)
    breaking_loop = environment.score_action_for_dog(0, "up", weights=weights)

    assert breaking_loop > revisiting_loop


def test_sheep_do_not_stack_when_fleeing_into_same_cell() -> None:
    config = make_config(dogs=1, sheep=2, dog_vision=20, sheep_speed=1)
    environment = SheepdogEnvironment(config)
    environment.reset(seed=22)
    dog = environment.dogs[0]
    dog.position = dog.position.__class__(9, 10)
    environment.sheep[0].position = environment.sheep[0].position.__class__(10, 9)
    environment.sheep[1].position = environment.sheep[1].position.__class__(10, 11)

    environment.step(["wait"])

    positions = {(sheep.position.x, sheep.position.y) for sheep in environment.sheep}
    assert len(positions) == 2


def test_sheep_uses_side_escape_when_pressed_against_wall() -> None:
    config = make_config(dogs=2, sheep=1, dog_vision=20, sheep_speed=1)
    environment = SheepdogEnvironment(config)
    environment.reset(seed=23)
    sheep = environment.sheep[0]
    sheep.position = sheep.position.__class__(0, 10)
    environment.dogs[0].position = environment.dogs[0].position.__class__(1, 10)
    environment.dogs[1].position = environment.dogs[1].position.__class__(1, 11)

    environment.step(["wait", "wait"])

    assert environment.sheep[0].position == sheep.position.__class__(0, 9)


def test_replay_frames_include_dog_roles() -> None:
    config = make_config(dogs=3, sheep=3)
    environment = SheepdogEnvironment(config)
    result = environment.run_policy(HeuristicExpertPolicy(), seed=14, capture_replay=True)

    assert result.replay
    assert all(frame.snapshot.dogs[0].role for frame in result.replay)


def test_heuristic_policy_runs_to_completion_or_stop() -> None:
    config = make_config()
    environment = SheepdogEnvironment(config)
    result = environment.run_policy(HeuristicExpertPolicy(), seed=13, capture_replay=True)

    assert result.stats.terminated is True
    assert result.final_snapshot.status in {"success", "timeout", "no-progress", "stopped"}
    assert isinstance(result.stats.role_distribution, dict)


def test_instinct_only_action_score_is_pen_invariant_without_target_awareness() -> None:
    base_config = make_config(dogs=1, sheep=1)
    narrow_pen_config = replace(
        base_config,
        environment=replace(base_config.environment, pen_width=6, pen_height=6),
        policy=PolicyConfig(policy_mode="instinct_only"),
    )

    first = SheepdogEnvironment(base_config)
    second = SheepdogEnvironment(narrow_pen_config)
    first.reset(seed=4)
    second.reset(seed=4)

    for environment in (first, second):
        dog = environment.dogs[0]
        sheep = environment.sheep[0]
        sheep.position = sheep.position.__class__(10, 10)
        dog.position = dog.position.__class__(6, 10)

    assert first.score_action_for_dog(
        0,
        "right",
        policy_mode="instinct_only",
    ) == second.score_action_for_dog(
        0,
        "right",
        policy_mode="instinct_only",
    )


def test_debug_snapshot_contains_pressure_payload_when_enabled() -> None:
    config = replace(
        make_config(),
        rewards=replace(
            RewardConfig(),
            instincts=replace(RewardConfig().instincts, debug_reward_breakdown=True),
        ),
    )
    snapshot = SheepdogEnvironment(config).reset(seed=5)

    assert snapshot.debug["policy_mode"] == config.policy.policy_mode
    assert len(snapshot.debug["dogs"]) == config.environment.dogs


def test_pen_has_three_closed_fences_and_one_opening() -> None:
    config = make_config()
    environment = SheepdogEnvironment(config)
    snapshot = environment.reset(seed=11)

    pen = snapshot.pen
    assert pen.opening == "left"
    assert pen.origin.x + pen.width == config.environment.width
    segments = pen.fence_segments()
    assert len(segments) == 3

    fence_cells = pen.fence_cells()
    # No fence on the open side (left column outside the pen).
    open_x = pen.origin.x - 1
    assert all(cell.x != open_x for cell in fence_cells)


def test_dog_cannot_step_into_fence_cell() -> None:
    config = make_config()
    environment = SheepdogEnvironment(config)
    snapshot = environment.reset(seed=11)
    pen = snapshot.pen
    fence_cell = next(iter(pen.fence_cells()))

    dog = environment.dogs[0]
    # Place dog adjacent to a fence cell from outside the pen.
    dog.position = dog.position.__class__(fence_cell.x, fence_cell.y + 1)
    if not environment.pen.contains(dog.position):
        environment.step(["up"] + ["wait"] * (config.environment.dogs - 1))
        # Dog should not have moved onto the fence.
        assert environment.dogs[0].position != fence_cell


def test_dog_cannot_enter_pen_from_outside() -> None:
    config = make_config(dogs=1, sheep=0)
    environment = SheepdogEnvironment(config)
    snapshot = environment.reset(seed=15)
    pen = snapshot.pen

    dog = environment.dogs[0]
    dog.position = dog.position.__class__(pen.origin.x - 1, pen.origin.y + 1)

    environment.step(["right"])

    assert environment.dogs[0].position == dog.position


def test_sheep_use_sheep_vision_not_dog_vision_for_detection() -> None:
    # Test A: Sheep use sheep_vision not dog_vision for detection
    # Set dog_vision very large, sheep_vision small
    config = make_config(dog_vision=50, sheep_vision=8, dogs=1, sheep=1)
    environment = SheepdogEnvironment(config)
    environment.reset(seed=42)

    sheep = environment.sheep[0]
    dog = environment.dogs[0]

    # Place dog outside sheep_vision but inside dog_vision
    sheep.position = sheep.position.__class__(20, 20)
    # 10 units away, > sheep_vision(8), < dog_vision(50)
    dog.position = dog.position.__class__(30, 20)

    # Sheep should not panic/flee since dog is outside sheep_vision
    environment.step(["wait"])

    # Sheep should not have moved significantly
    # (may have small random movement due to flock cohesion).
    # With the fix, sheep should mostly stay still when dog is outside sheep_vision.
    assert sheep.panic_steps == 0, "Sheep should not panic when dog is outside sheep_vision"


def test_sheep_stay_mostly_still_when_dogs_far_away() -> None:
    # Test B: Sheep stay still when dogs are far away
    config = make_config(dog_vision=50, sheep_vision=12, dogs=1, sheep=3)
    environment = SheepdogEnvironment(config)
    environment.reset(seed=43)

    # Place dogs far away from all sheep
    environment.dogs[0].position = environment.dogs[0].position.__class__(5, 5)
    for sheep in environment.sheep:
        sheep.position = sheep.position.__class__(50, 30)

    initial_positions = [(s.position.x, s.position.y) for s in environment.sheep]

    # Run several steps with dogs waiting
    for _ in range(10):
        environment.step(["wait"])

    # Sheep should not have moved significantly across the field
    final_positions = [(s.position.x, s.position.y) for s in environment.sheep]
    total_movement = sum(
        abs(final_positions[i][0] - initial_positions[i][0])
        + abs(final_positions[i][1] - initial_positions[i][1])
        for i in range(len(initial_positions))
    )
    # With idle behavior, total movement should be minimal (< 5 cells over 10 steps for 3 sheep)
    assert total_movement < 5, f"Sheep moved too much when dogs were far away: {total_movement}"


def test_sheep_do_not_self_pen_with_frozen_dogs() -> None:
    # Test C: Sheep do not self-pen with frozen/far dogs
    config = make_config(dog_vision=50, sheep_vision=12, dogs=1, sheep=3)
    environment = SheepdogEnvironment(config)
    snapshot = environment.reset(seed=44)
    pen = snapshot.pen

    # Place dogs far from pen
    environment.dogs[0].position = environment.dogs[0].position.__class__(10, 10)

    # Place sheep near but not in pen
    for i, sheep in enumerate(environment.sheep):
        sheep.position = sheep.position.__class__(pen.origin.x - 15, pen.origin.y + i)

    initial_penned = sum(1 for s in environment.sheep if s.penned)

    # Run simulation with frozen dogs
    for _ in range(30):
        environment.step(["wait"])

    final_penned = sum(1 for s in environment.sheep if s.penned)

    # Sheep should not have entered the pen without dog pressure
    assert final_penned == initial_penned, "Sheep should not self-pen without dog pressure"


def test_sheep_move_when_dogs_within_sheep_vision() -> None:
    # Test D: Sheep still move/react when dogs within sheep_vision
    config = make_config(dog_vision=50, sheep_vision=12, dogs=1, sheep=1, sheep_speed=1.0)
    environment = SheepdogEnvironment(config)
    environment.reset(seed=45)

    sheep = environment.sheep[0]
    dog = environment.dogs[0]

    # Place dog within sheep_vision
    sheep.position = sheep.position.__class__(20, 20)
    dog.position = dog.position.__class__(15, 20)  # 5 units away, < sheep_vision(12)

    before = sheep.position
    environment.step(["wait"])

    after = environment.sheep[0].position

    # Sheep should have moved away from the dog (or at least not toward it)
    # With the flee vector, sheep should move right (increase x) to escape
    assert after.x >= before.x, "Sheep should not move toward dog within sheep_vision"
    # After the step, panic should be set for the next step
    assert sheep.panic_steps > 0, "Sheep should panic when dog is within sheep_vision"


def test_episode_stats_to_dict() -> None:
    from sheepdog.entities import EpisodeStats
    stats = EpisodeStats(steps=10, reward_total=100.5, success=True)
    d = stats.to_dict()
    assert isinstance(d, dict)
    assert d["steps"] == 10
    assert d["reward_total"] == 100.5
    assert d["success"] is True

