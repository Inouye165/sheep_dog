"""Tests for optional (assisted/guided) TeamStrategy split-flock/corner options."""

from __future__ import annotations

import json
from pathlib import Path

from sheepdog.entities import DogRole, DogState, Pen, Point, SheepState
from sheepdog.team_strategy import TeamStrategy

W, H = 120, 104
PEN = Pen(origin=Point(50, 2), width=12, height=12, opening="bottom")


def _sheep(points: list[tuple[int, int]]) -> list[SheepState]:
    return [SheepState(index=i, position=Point(x, y)) for i, (x, y) in enumerate(points)]


def _dogs(points: list[tuple[int, int]], roles=None, held=0) -> list[DogState]:
    dogs = [DogState(index=i, position=Point(x, y)) for i, (x, y) in enumerate(points)]
    for dog, role in zip(dogs, roles or [None] * len(dogs), strict=True):
        if role is not None:
            dog.current_role = role
            dog.steps_in_role = held
    return dogs


SPLIT = [(70, 44), (72, 44), (71, 46), (73, 45), (70, 46), (60, 85)]
CORNER = [(70, 44), (72, 44), (71, 46), (73, 45), (70, 46), (0, 103)]
DOG_POS = [(30, 60), (90, 60), (60, 20)]


def _targets(strategy: TeamStrategy, sheep_pts, dog_pts=DOG_POS):
    roles, snap = strategy.assign_roles(_dogs(dog_pts), _sheep(sheep_pts), PEN)
    return (
        {i: (a.role.value, a.target.x, a.target.y) for i, a in sorted(roles.items())},
        snap,
    )


def test_defaults_are_off_and_match_explicit_off() -> None:
    assert not TeamStrategy.DEFAULT_MAIN_GROUP_TARGETS
    assert not TeamStrategy.DEFAULT_COLLECTOR_YIELD
    assert not TeamStrategy.DEFAULT_CORNER_CLEARANCE
    for pts in (SPLIT, CORNER, SPLIT[:5]):
        default, _ = _targets(TeamStrategy(W, H), pts)
        off, _ = _targets(
            TeamStrategy(
                W, H, main_group_targets=False, collector_yield=False, corner_clearance=False
            ),
            pts,
        )
        assert default == off


def test_defaults_reproduce_v1_0_0_golden_targets() -> None:
    golden = json.loads((Path(__file__).parent / "golden_team_strategy_v1_0_0.json").read_text())
    for name, pts in (("split", SPLIT), ("corner", CORNER), ("clustered", SPLIT[:5])):
        got, _ = _targets(TeamStrategy(W, H), pts)
        assert {str(k): list(v) for k, v in got.items()} == golden[name]


def test_main_group_option_excludes_stray_from_centroid() -> None:
    _, snap_off = _targets(TeamStrategy(W, H), SPLIT)
    assert snap_off.stray_sheep_index == 5
    off, _ = _targets(TeamStrategy(W, H), SPLIT)
    on, snap_on = _targets(TeamStrategy(W, H, main_group_targets=True), SPLIT)
    assert snap_on.flock_center == snap_off.flock_center  # snapshot unchanged
    rear_off = next(v for v in off.values() if v[0] == DogRole.REAR_PRESSURE.value)
    rear_on = next(v for v in on.values() if v[0] == DogRole.REAR_PRESSURE.value)
    # Main group centre is ~(71, 45); all-sheep centre is dragged to ~(69, 54).
    assert abs(rear_on[2] - 45) < abs(rear_off[2] - 45)
    assert rear_on != rear_off


def test_main_group_option_noop_without_stray() -> None:
    a, _ = _targets(TeamStrategy(W, H), SPLIT[:5])
    b, _ = _targets(TeamStrategy(W, H, main_group_targets=True), SPLIT[:5])
    assert a == b


def test_collector_yields_after_stall_only_with_option() -> None:
    held = 900
    target = Point(60, 54)
    holder = _dogs([(60, 52), (90, 90)], roles=[DogRole.COLLECTOR, None], held=held)[0]
    other = DogState(index=1, position=Point(60, 62))
    base = TeamStrategy(W, H)
    new = TeamStrategy(W, H, collector_yield=True)
    # Baseline: stagnation floor keeps the holder ahead of a nearby challenger.
    assert base._best_dog_for_role([holder, other], target, DogRole.COLLECTOR).index == 0
    assert new._best_dog_for_role([holder, other], target, DogRole.COLLECTOR).index == 1
    # Fresh holder (not stalled) is unaffected by the option.
    holder.steps_in_role = 10
    assert new._best_dog_for_role([holder, other], target, DogRole.COLLECTOR).index == 0


def test_corner_clearance_collector_and_others() -> None:
    off = TeamStrategy(W, H)
    on = TeamStrategy(W, H, corner_clearance=True)
    stray = Point(0, 103)
    c_off = off._collector_target(stray, Point(71, 45), PEN)
    c_on = on._collector_target(stray, Point(71, 45), PEN)
    assert max(abs(c_off.x - stray.x), abs(c_off.y - stray.y)) == 3
    assert c_on == Point(4, 99)
    got, _ = _targets(on, CORNER)
    for _role, x, y in got.values():
        assert max(abs(x - stray.x), abs(y - stray.y)) >= 4


def test_corner_clearance_noop_in_open_field() -> None:
    a, _ = _targets(TeamStrategy(W, H), SPLIT)
    b, _ = _targets(TeamStrategy(W, H, corner_clearance=True), SPLIT)
    assert a == b
