"""Regression tests for the herding environment."""

from __future__ import annotations

from dataclasses import replace

from sheepdog.config import EnvironmentConfig, LabConfig, PolicyConfig, RewardConfig, TrainingConfig
from sheepdog.environment import SheepdogEnvironment
from sheepdog.policies.heuristic import HeuristicExpertPolicy, InstinctOnlyPolicy
from sheepdog.policies.random_policy import RandomPolicy


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


def test_reset_is_deterministic_for_fixed_seed() -> None:
    config = make_config()
    first = SheepdogEnvironment(config).reset(seed=42).to_dict()
    second = SheepdogEnvironment(config).reset(seed=42).to_dict()

    assert first == second


def test_sheep_flee_from_nearby_dog() -> None:
    config = make_config()
    environment = SheepdogEnvironment(config)
    environment.reset(seed=7)
    sheep = environment._sheep[0]
    dog = environment._dogs[0]
    sheep.position = sheep.position.__class__(8, 8)
    dog.position = dog.position.__class__(7, 8)

    before = sheep.position
    environment.step(["wait"] * config.environment.dogs)

    after = environment._sheep[0].position
    assert after.x >= before.x


def test_sheep_enter_pen_and_counted() -> None:
    config = make_config()
    environment = SheepdogEnvironment(config)
    snapshot = environment.reset(seed=11)
    pen = snapshot.pen

    for sheep in environment._sheep:
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


def test_action_mask_disallows_useless_wait_when_movement_exists() -> None:
    config = make_config()
    environment = SheepdogEnvironment(config)
    environment.reset(seed=3)

    mask = environment.action_mask_for_dog(0)

    assert mask["wait"] is False


def test_multiple_dogs_keep_consistent_state_updates() -> None:
    config = make_config(dogs=4)
    environment = SheepdogEnvironment(config)
    environment.reset(seed=9)

    snapshot, _ = environment.step(["right", "left", "up", "down"])

    assert len(snapshot.dogs) == 4
    assert {dog.index for dog in snapshot.dogs} == {0, 1, 2, 3}
    assert len({(dog.x, dog.y) for dog in snapshot.dogs}) == 4


def test_default_dog_movement_uses_two_cell_steps() -> None:
    config = make_config(dogs=1, sheep=0, dog_speed=2)
    environment = SheepdogEnvironment(config)
    environment.reset(seed=3)
    dog = environment._dogs[0]
    dog.position = dog.position.__class__(5, 5)

    environment.step(["right"])

    assert environment._dogs[0].position.x == 7


def test_dog_speed_stops_before_blocking_sheep() -> None:
    config = make_config(dogs=1, sheep=1, dog_speed=2, sheep_speed=1)
    environment = SheepdogEnvironment(config)
    environment.reset(seed=3)
    dog = environment._dogs[0]
    sheep = environment._sheep[0]
    dog.position = dog.position.__class__(5, 5)
    sheep.position = sheep.position.__class__(7, 5)

    environment.step(["right"])

    assert environment._dogs[0].position.x == 6


def test_dog_action_score_prefers_herding_standoff() -> None:
    config = make_config(dogs=1, sheep=1)
    environment = SheepdogEnvironment(config)
    environment.reset(seed=4)

    dog = environment._dogs[0]
    sheep = environment._sheep[0]
    sheep.position = sheep.position.__class__(10, 10)
    dog.position = dog.position.__class__(6, 10)

    better_spacing = environment.score_action_for_dog(0, "right")
    too_far = environment.score_action_for_dog(0, "left")

    assert better_spacing > too_far


def test_action_score_prefers_behind_flock_over_pen_side_position() -> None:
    config = make_config(dogs=1, sheep=1)
    environment = SheepdogEnvironment(config)
    environment.reset(seed=4)

    sheep = environment._sheep[0]
    sheep.position = sheep.position.__class__(18, 10)
    dog = environment._dogs[0]

    dog.position = dog.position.__class__(14, 10)
    behind_score = environment.score_action_for_dog(0, "wait", policy_mode="heuristic_expert")

    dog.position = dog.position.__class__(22, 10)
    pen_side_score = environment.score_action_for_dog(0, "wait", policy_mode="heuristic_expert")

    assert behind_score > pen_side_score


def test_action_score_penalizes_positions_inside_or_too_close_to_flock() -> None:
    config = make_config(dogs=1, sheep=1)
    environment = SheepdogEnvironment(config)
    environment.reset(seed=8)

    sheep = environment._sheep[0]
    sheep.position = sheep.position.__class__(10, 10)
    dog = environment._dogs[0]
    dog.position = dog.position.__class__(7, 10)

    safer_pressure = environment.score_action_for_dog(0, "wait", policy_mode="heuristic_expert")
    too_close = environment.score_action_for_dog(0, "right", policy_mode="heuristic_expert")

    assert safer_pressure > too_close


def test_action_score_no_longer_rewards_diving_toward_flock_center() -> None:
    config = make_config(dogs=1, sheep=1)
    environment = SheepdogEnvironment(config)
    environment.reset(seed=10)

    sheep = environment._sheep[0]
    sheep.position = sheep.position.__class__(10, 10)
    dog = environment._dogs[0]
    dog.position = dog.position.__class__(7, 10)

    hold_pressure = environment.score_action_for_dog(0, "wait", policy_mode="heuristic_expert")
    step_toward_center = environment.score_action_for_dog(
        0, "right", policy_mode="heuristic_expert"
    )

    assert hold_pressure > step_toward_center


def test_anti_oscillation_penalty_lowers_immediate_reverse_actions() -> None:
    config = make_config(dogs=1, sheep=1)
    environment = SheepdogEnvironment(config)
    environment.reset(seed=12)

    sheep = environment._sheep[0]
    sheep.position = sheep.position.__class__(10, 10)
    dog = environment._dogs[0]
    dog.position = dog.position.__class__(7, 10)
    dog.last_action = "right"

    reverse_score = environment.score_action_for_dog(0, "left", policy_mode="heuristic_expert")
    hold_score = environment.score_action_for_dog(0, "wait", policy_mode="heuristic_expert")

    assert hold_score > reverse_score


def test_debug_snapshot_includes_pressure_position_fields() -> None:
    config = make_config(dogs=1, sheep=1)
    config = replace(
        config,
        rewards=replace(
            config.rewards,
            instincts=replace(
                config.rewards.instincts,
                enable_instinct_rewards=True,
                debug_reward_breakdown=True,
                curriculum_stage=2,
            ),
        ),
    )
    environment = SheepdogEnvironment(config)

    snapshot = environment.reset(seed=2)

    assert snapshot.debug["curriculum_stage"] == 2
    assert snapshot.debug["enable_instinct_rewards"] is True
    dog_debug = snapshot.debug["dogs"][0]
    assert "desired_pressure_target" in dog_debug
    assert "distance_to_pressure_target" in dog_debug
    assert "pressure_side_alignment" in dog_debug
    assert "between_flock_and_pen" in dog_debug
    assert "inside_or_too_close_to_flock" in dog_debug


def test_dog_action_score_prefers_spread_out_team_positions() -> None:
    config = make_config(dogs=2, sheep=1)
    environment = SheepdogEnvironment(config)
    environment.reset(seed=6)

    first_dog = environment._dogs[0]
    second_dog = environment._dogs[1]
    first_dog.position = first_dog.position.__class__(6, 10)
    second_dog.position = second_dog.position.__class__(8, 10)

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
        },
    )()

    spread_out = environment.score_action_for_dog(0, "left", weights=spacing_weights)
    bunched_up = environment.score_action_for_dog(0, "right", weights=spacing_weights)

    assert spread_out > bunched_up


def test_heuristic_policy_runs_to_completion_or_stop() -> None:
    config = make_config()
    environment = SheepdogEnvironment(config)
    result = environment.run_policy(HeuristicExpertPolicy(), seed=13, capture_replay=True)

    assert result.stats.terminated is True
    assert result.final_snapshot.status in {"success", "timeout", "no-progress", "stopped"}


def test_instinct_only_action_scoring_does_not_depend_on_pen_position() -> None:
    left_pen_config = LabConfig(
        environment=EnvironmentConfig(dogs=1, sheep=1),
        rewards=RewardConfig(),
        training=TrainingConfig(episodes=0, checkpoint_episodes=(0,), evaluation_seeds=(11,)),
        policy=PolicyConfig(policy_mode="instinct_only"),
    )
    right_pen_config = LabConfig(
        environment=EnvironmentConfig(dogs=1, sheep=1, pen_opening="right"),
        rewards=RewardConfig(),
        training=TrainingConfig(episodes=0, checkpoint_episodes=(0,), evaluation_seeds=(11,)),
        policy=PolicyConfig(policy_mode="instinct_only"),
    )
    left_environment = SheepdogEnvironment(left_pen_config)
    right_environment = SheepdogEnvironment(right_pen_config)
    left_environment.reset(seed=21)
    right_environment.reset(seed=21)

    for environment in (left_environment, right_environment):
        sheep = environment._sheep[0]
        dog = environment._dogs[0]
        sheep.position = sheep.position.__class__(10, 10)
        dog.position = dog.position.__class__(7, 10)

    left_score = left_environment.score_action_for_dog(0, "wait", policy_mode="instinct_only")
    right_score = right_environment.score_action_for_dog(0, "wait", policy_mode="instinct_only")

    assert left_score == right_score


def test_heuristic_expert_scoring_depends_on_pen_position() -> None:
    left_pen_config = make_config(dogs=1, sheep=1, width=40)
    right_pen_config = make_config(dogs=1, sheep=1, width=60)
    left_environment = SheepdogEnvironment(left_pen_config)
    right_environment = SheepdogEnvironment(right_pen_config)
    left_environment.reset(seed=22)
    right_environment.reset(seed=22)

    for environment in (left_environment, right_environment):
        sheep = environment._sheep[0]
        dog = environment._dogs[0]
        sheep.position = sheep.position.__class__(18, 10)
        dog.position = dog.position.__class__(14, 10)

    left_score = left_environment.score_action_for_dog(0, "wait", policy_mode="heuristic_expert")
    right_score = right_environment.score_action_for_dog(0, "wait", policy_mode="heuristic_expert")

    assert left_score != right_score


def test_random_untrained_policy_does_not_reliably_pick_pen_directed_action() -> None:
    config = make_config(dogs=1, sheep=1)
    pen_directed_actions = 0

    for seed in range(60):
        environment = SheepdogEnvironment(config)
        environment.reset(seed=seed)
        sheep = environment._sheep[0]
        dog = environment._dogs[0]
        sheep.position = sheep.position.__class__(10, 10)
        dog.position = dog.position.__class__(7, 10)
        action = RandomPolicy(seed=seed).select_actions(environment)[0]
        if action == "right":
            pen_directed_actions += 1

    assert pen_directed_actions < 25


def test_instinct_only_policy_runs_without_pen_target_knowledge() -> None:
    config = make_config()
    environment = SheepdogEnvironment(config)

    result = environment.run_policy(InstinctOnlyPolicy(), seed=17, capture_replay=True)

    assert result.policy_name == "instinct_only"
    assert result.stats.terminated is True


def test_dog_and_sheep_cannot_occupy_same_cell() -> None:
    config = make_config(dogs=1, sheep=1)
    environment = SheepdogEnvironment(config)
    environment.reset(seed=31)
    dog = environment._dogs[0]
    sheep = environment._sheep[0]
    dog.position = dog.position.__class__(5, 5)
    sheep.position = sheep.position.__class__(6, 5)

    environment.step(["right"])

    assert environment._dogs[0].position != environment._sheep[0].position
    assert environment._dogs[0].position == dog.position.__class__(5, 5)


def test_two_dogs_cannot_occupy_same_cell() -> None:
    config = make_config(dogs=2, sheep=0)
    environment = SheepdogEnvironment(config)
    environment.reset(seed=32)
    environment._dogs[0].position = environment._dogs[0].position.__class__(5, 5)
    environment._dogs[1].position = environment._dogs[1].position.__class__(7, 5)

    environment.step(["right", "left"])

    positions = {(dog.position.x, dog.position.y) for dog in environment._dogs}
    assert len(positions) == 2


def test_denser_grid_defaults_are_active() -> None:
    config = make_config()

    assert config.environment.width >= 120
    assert config.environment.height >= 90
    assert config.environment.pen_width >= 15
    assert config.environment.dog_speed == 2


def test_sheep_randomness_changes_tie_break_outcomes_by_seed() -> None:
    left_environment = SheepdogEnvironment(make_config(dogs=1, sheep=1))
    right_environment = SheepdogEnvironment(make_config(dogs=1, sheep=1))
    left_environment.reset(seed=41)
    right_environment.reset(seed=42)

    for environment in (left_environment, right_environment):
        environment._dogs[0].position = environment._dogs[0].position.__class__(10, 10)
        environment._sheep[0].position = environment._sheep[0].position.__class__(12, 12)

    left_environment.step(["wait"])
    right_environment.step(["wait"])

    assert left_environment._sheep[0].position != right_environment._sheep[0].position


def test_deadlock_detection_triggers_for_wall_pinned_sheep() -> None:
    config = make_config(dogs=1, sheep=1)
    environment = SheepdogEnvironment(config)
    environment.reset(seed=51)
    sheep = environment._sheep[0]
    dog = environment._dogs[0]
    sheep.position = sheep.position.__class__(0, 10)
    sheep.blocked_steps = 3
    dog.position = dog.position.__class__(2, 10)
    environment._no_progress_steps = 5

    state = environment._deadlock_state()

    assert state["active"] is True
    assert state["wall_pinned_sheep"] == (0,)


def test_two_position_loop_penalty_reduces_back_and_forth_behavior() -> None:
    config = make_config(dogs=1, sheep=1)
    clean_environment = SheepdogEnvironment(config)
    loop_environment = SheepdogEnvironment(config)
    clean_environment.reset(seed=61)
    loop_environment.reset(seed=61)

    for environment in (clean_environment, loop_environment):
        sheep = environment._sheep[0]
        dog = environment._dogs[0]
        sheep.position = sheep.position.__class__(20, 20)
        dog.position = dog.position.__class__(16, 20)

    loop_environment._dogs[0].recent_positions[:] = [
        loop_environment._dogs[0].position.__class__(16, 20),
        loop_environment._dogs[0].position.__class__(17, 20),
        loop_environment._dogs[0].position.__class__(16, 20),
        loop_environment._dogs[0].position.__class__(17, 20),
    ]

    clean_score = clean_environment.score_action_for_dog(0, "right", policy_mode="heuristic_expert")
    loop_score = loop_environment.score_action_for_dog(0, "right", policy_mode="heuristic_expert")

    assert loop_score < clean_score


def test_pen_has_three_closed_fences_and_one_opening() -> None:
    config = make_config()
    environment = SheepdogEnvironment(config)
    snapshot = environment.reset(seed=11)

    assert snapshot.field_width == config.environment.width
    assert snapshot.field_height == config.environment.height

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

    dog = environment._dogs[0]
    # Place dog adjacent to a fence cell from outside the pen.
    dog.position = dog.position.__class__(fence_cell.x, fence_cell.y + 1)
    if not environment._pen.contains(dog.position):
        environment.step(["up"] + ["wait"] * (config.environment.dogs - 1))
        # Dog should not have moved onto the fence.
        assert environment._dogs[0].position != fence_cell
