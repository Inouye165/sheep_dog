"""Tests for opt-in Stage 8 regrouping reward shaping and no-progress reset."""

from __future__ import annotations

from dataclasses import replace

import pytest

from sheepdog.config import EnvironmentConfig, LabConfig, RewardConfig
from sheepdog.entities import Point
from sheepdog.environment import SheepdogEnvironment
from sheepdog.rewards import RewardComputer, RewardInputs


def _transition(
    old_dogs: tuple[tuple[float, float], ...],
    dogs: tuple[tuple[float, float], ...],
    old_sheep: tuple[tuple[float, float], ...],
    sheep: tuple[tuple[float, float], ...],
    *,
    previous_average_distance: float = 50.0,
    current_average_distance: float = 50.0,
) -> RewardInputs:
    return RewardInputs(
        previous_average_distance=previous_average_distance,
        current_average_distance=current_average_distance,
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


def test_defaults_are_bit_exact_with_baseline() -> None:
    """Default RewardConfig and EnvironmentConfig leave behavior bit-exact."""
    default_cfg = LabConfig()
    explicit_off_cfg = LabConfig(
        environment=EnvironmentConfig(
            regroup_no_progress_reset=False,
            regroup_no_progress_max_resets=80,
        ),
        rewards=RewardConfig(regrouping_approach_scale=0.0),
    )
    assert default_cfg.rewards.regrouping_approach_scale == 0.0
    assert default_cfg.environment.regroup_no_progress_reset is False
    assert default_cfg.environment.regroup_no_progress_max_resets == 80

    # Round-trip serialization preserves new fields.
    round_trip = LabConfig.from_dict(
        LabConfig(
            environment=EnvironmentConfig(
                regroup_no_progress_reset=True,
                regroup_no_progress_max_resets=42,
            ),
            rewards=RewardConfig(regrouping_approach_scale=2.0),
        ).to_dict()
    )
    assert round_trip.rewards.regrouping_approach_scale == 2.0
    assert round_trip.environment.regroup_no_progress_reset is True
    assert round_trip.environment.regroup_no_progress_max_resets == 42

    env_a = SheepdogEnvironment(default_cfg)
    env_b = SheepdogEnvironment(explicit_off_cfg)
    snap_a = env_a.reset(seed=17)
    snap_b = env_b.reset(seed=17)
    assert snap_a == snap_b

    actions = ("right", "down", "left")
    for _ in range(15):
        s_a, r_a = env_a.step(actions)
        s_b, r_b = env_b.step(actions)
        assert s_a == s_b
        assert r_a == r_b
        assert r_a.regrouping_approach == 0.0


def test_adversarial_closed_loop_and_oscillation_zero_net_reward() -> None:
    """Closed-loop dog movement and 2-step oscillation yield 0.0 net regrouping reward."""
    computer = RewardComputer(RewardConfig(regrouping_approach_scale=2.0))
    sheep = ((50.0, 20.0), (54.0, 22.0))

    # 2-step oscillation: (10, 20) -> (12, 20) -> (10, 20)
    fwd = computer.compute(_transition(((10.0, 20.0),), ((12.0, 20.0),), sheep, sheep))
    bwd = computer.compute(_transition(((12.0, 20.0),), ((10.0, 20.0),), sheep, sheep))
    assert fwd.regrouping_approach > 0.0
    assert bwd.regrouping_approach < 0.0
    assert fwd.regrouping_approach + bwd.regrouping_approach == pytest.approx(0.0, abs=1e-12)
    assert fwd.total + bwd.total < 0.0  # time_penalty makes closed loop strictly negative

    # Multi-step closed loop with moving dogs and sheep
    states = [
        (((0.0, 0.0), (90.0, 0.0)), ((30.0, 10.0), (60.0, 10.0))),
        (((20.0, 5.0), (85.0, 5.0)), ((45.0, 12.0), (50.0, 8.0))),
        (((35.0, 10.0), (60.0, 10.0)), ((55.0, 15.0), (40.0, 12.0))),
        (((10.0, 5.0), (75.0, 5.0)), ((35.0, 8.0), (55.0, 10.0))),
    ]
    loop_terms = [
        computer.compute(_transition(old[0], new[0], old[1], new[1])).regrouping_approach
        for old, new in zip(states, states[1:] + states[:1], strict=True)
    ]
    assert sum(loop_terms) == pytest.approx(0.0, abs=1e-12)


def test_adversarial_wrong_side_push_remains_net_negative() -> None:
    """Pushing sheep away from the pen while approaching still yields net negative reward."""
    computer = RewardComputer(RewardConfig(regrouping_approach_scale=2.0))
    bd = computer.compute(
        _transition(
            ((48.0, 0.0),),
            ((49.0, 0.0),),
            ((50.0, 0.0),),
            ((49.0, 0.0),),
            previous_average_distance=50.0,
            current_average_distance=51.0,
        )
    )
    assert bd.progress_to_pen == pytest.approx(-2.0)
    assert bd.total < -1.5


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
def test_regrouping_approach_zeroed_on_penning_and_terminal_transitions(
    overrides: dict[str, object],
) -> None:
    """Regrouping shaping is strictly 0.0 on penning, terminal, or empty-entity steps."""
    computer = RewardComputer(RewardConfig(regrouping_approach_scale=2.0))
    inputs = _transition(((0.0, 0.0),), ((10.0, 0.0),), ((50.0, 0.0),), ((50.0, 0.0),))
    assert computer.compute(replace(inputs, **overrides)).regrouping_approach == 0.0


def test_regroup_no_progress_reset_conditions_and_boundedness() -> None:
    """Opt-in regroup reset counts spread reduction above best spread, but is strictly bounded."""
    env_cfg = EnvironmentConfig(
        width=80,
        height=60,
        dogs=3,
        sheep=6,
        max_steps=400,
        no_progress_window=20,
        regroup_no_progress_reset=True,
        regroup_no_progress_max_resets=3,
    )
    env = SheepdogEnvironment(LabConfig(environment=env_cfg))
    env.reset(seed=7)

    # Place dogs within 15 cells of unpenned sheep, and set _best_flock_spread tight (4.0)
    # while current flock is scattered (> 6.0) and shrinking by >= 0.15 on this step.
    env._best_flock_spread = 4.0
    env._no_progress_steps = 10
    env._regroup_resets_used = 0
    for idx, dog in enumerate(env._dogs):
        dog.position = Point(20 + idx, 30)
    # Put sheep at spread > 6.0 and let's test step() with wait + controlled sheep positions
    # by monkeypatching _move_sheep for a deterministic unit check of the reset gate.
    spread_positions = [
        [Point(20, 20), Point(40, 20), Point(20, 40), Point(40, 40), Point(30, 20), Point(30, 40)],
        [Point(22, 22), Point(38, 22), Point(22, 38), Point(38, 38), Point(30, 22), Point(30, 38)],
    ]
    for sheep, pt in zip(env._sheep, spread_positions[0], strict=True):
        sheep.position = pt
        sheep.penned = False

    step_idx = 0

    def _fake_move_sheep() -> None:
        nonlocal step_idx
        step_idx += 1
        # Alternate shrinking (odd steps) and expanding (even steps) while spread > 6.0
        target = spread_positions[1] if step_idx % 2 == 1 else spread_positions[0]
        for s, p in zip(env._sheep, target, strict=True):
            s.position = p

    env._move_sheep = _fake_move_sheep  # type: ignore[method-assign]

    # Step 1 (shrinks): uses 1st regroup reset, decrements _no_progress_steps 10 -> 9
    env.step(["wait", "wait", "wait"])
    assert env._regroup_resets_used == 1
    assert env._no_progress_steps == 9

    # Continue alternating until max_resets=3 is exhausted, then no-progress terminates well before max_steps=400
    for _ in range(100):
        if env._terminated:
            break
        env.step(["wait", "wait", "wait"])

    assert env._terminated is True
    assert env._stop_reason == "no-progress"
    assert env._regroup_resets_used == 3
    assert env._step_count < 60  # well before max_steps=400


def test_regroup_no_progress_reset_requires_nearby_dog() -> None:
    """Far-field spread reduction (> 15.0 cells from all dogs) does NOT trigger regroup reset."""
    env_cfg = EnvironmentConfig(
        width=80,
        height=60,
        dogs=3,
        sheep=6,
        max_steps=200,
        no_progress_window=20,
        regroup_no_progress_reset=True,
        regroup_no_progress_max_resets=10,
    )
    env = SheepdogEnvironment(LabConfig(environment=env_cfg))
    env.reset(seed=7)
    env._best_flock_spread = 4.0
    env._no_progress_steps = 5
    for idx, dog in enumerate(env._dogs):
        dog.position = Point(2 + idx, 2)  # > 15 cells from all sheep below

    initial_pts = [
        Point(40, 20), Point(60, 20), Point(40, 40), Point(60, 40), Point(50, 20), Point(50, 40)
    ]
    shrunk_pts = [
        Point(42, 22), Point(58, 22), Point(42, 38), Point(58, 38), Point(50, 22), Point(50, 38)
    ]
    for s, p in zip(env._sheep, initial_pts, strict=True):
        s.position = p
        s.penned = False

    env._move_sheep = lambda: [  # type: ignore[method-assign, assignment]
        setattr(s, "position", p) for s, p in zip(env._sheep, shrunk_pts, strict=True)
    ]
    env.step(["wait", "wait", "wait"])
    assert env._regroup_resets_used == 0
    assert env._no_progress_steps == 6
