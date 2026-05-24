"""Diagnostic tests for pen geometry and success conditions.

Verifies that:
- The pen opening is clear of fence cells for each Stage 1-5 config
- Sheep placed inside the pen are detected as penned via pen.contains()
- The environment's success condition fires when all sheep are penned
- average_distance_to_pen approaches 0.0 when all sheep are inside the pen
"""

from __future__ import annotations

import pytest

from sheepdog.config import LabConfig
from sheepdog.curriculum import apply_training_profile
from sheepdog.entities import Pen, Point
from sheepdog.environment import SheepdogEnvironment


def _stage_config(stage: int) -> LabConfig:
    base = LabConfig()
    return apply_training_profile(base, curriculum_stage=stage)


# ---------------------------------------------------------------------------
# Pen entity tests
# ---------------------------------------------------------------------------


class TestPenFenceCells:
    """Unit tests for Pen.fence_cells() correctness."""

    def test_left_opening_clears_left_column(self) -> None:
        """No fence cell should sit in the gate column when opening='left'."""
        pen = Pen(Point(40, 1), width=18, height=18, opening="left")
        fence_cells = pen.fence_cells()
        gate_x = pen.origin.x - 1  # column just left of pen
        gate_cells = [c for c in fence_cells if c.x == gate_x]
        assert gate_cells == [], (
            f"Gate column x={gate_x} should be fully clear, "
            f"but found {len(gate_cells)} fence cells there"
        )

    def test_fence_cells_do_not_overlap_pen_interior(self) -> None:
        """Fence cells should not cover any cell inside the pen bounds."""
        pen = Pen(Point(40, 1), width=18, height=18, opening="left")
        fence_cells = pen.fence_cells()
        for cell in fence_cells:
            assert not pen.contains(cell), (
                f"Fence cell {cell} is inside the pen interior - "
                "sheep would be blocked from entering/exiting"
            )

    def test_pen_contains_interior_cells(self) -> None:
        """Every cell in the pen bounds should be detected as inside."""
        pen = Pen(Point(40, 1), width=18, height=18, opening="left")
        interior = [
            Point(x, y)
            for y in range(pen.origin.y, pen.origin.y + pen.height)
            for x in range(pen.origin.x, pen.origin.x + pen.width)
        ]
        for cell in interior:
            assert pen.contains(cell), f"Pen should contain {cell}"

    @pytest.mark.parametrize("stage", [1, 2, 3, 4, 5])
    def test_stage_pen_gate_is_clear(self, stage: int) -> None:
        """All curriculum stages should have a clear pen opening."""
        config = _stage_config(stage)
        env = SheepdogEnvironment(config)
        env.reset(seed=0)
        pen = env.pen
        fence_cells = env._fence_cells  # noqa: SLF001 (test-only access)
        opening = pen.opening
        if opening == "left":
            gate_cells_in_fence = [c for c in fence_cells if c.x == pen.origin.x - 1]
            location = f"column x={pen.origin.x - 1}"
        elif opening == "right":
            gate_cells_in_fence = [c for c in fence_cells if c.x == pen.origin.x + pen.width]
            location = f"column x={pen.origin.x + pen.width}"
        elif opening == "top":
            gate_cells_in_fence = [c for c in fence_cells if c.y == pen.origin.y - 1]
            location = f"row y={pen.origin.y - 1}"
        else:  # bottom
            gate_cells_in_fence = [c for c in fence_cells if c.y == pen.origin.y + pen.height]
            location = f"row y={pen.origin.y + pen.height}"
        assert gate_cells_in_fence == [], (
            f"Stage {stage}: gate {location} has {len(gate_cells_in_fence)} fence cells: "
            f"{[(c.x, c.y) for c in gate_cells_in_fence[:5]]}"
        )

    @pytest.mark.parametrize("stage", [1, 2, 3, 4, 5])
    def test_stage_pen_interior_not_fenced(self, stage: int) -> None:
        """Pen interior must be free of fence cells for every curriculum stage."""
        config = _stage_config(stage)
        env = SheepdogEnvironment(config)
        env.reset(seed=0)
        pen = env.pen
        fence_cells = env._fence_cells  # noqa: SLF001 (test-only access)
        interior_cells_in_fence = [c for c in fence_cells if pen.contains(c)]
        assert interior_cells_in_fence == [], (
            f"Stage {stage}: found {len(interior_cells_in_fence)} fence cells inside the pen"
        )


# ---------------------------------------------------------------------------
# Success condition tests
# ---------------------------------------------------------------------------


class TestSuccessCondition:
    """Verify that pen-entry mechanics and success detection work correctly."""

    def test_sheep_move_not_blocked_at_gate(self) -> None:
        """_sheep_move_blocked() must return False for a move from gate to pen interior."""
        config = _stage_config(1)
        env = SheepdogEnvironment(config)
        env.reset(seed=0)
        pen = env.pen
        # Gate cell: immediately below pen at center x (opening="bottom")
        gate_cell = Point(pen.center.x, pen.origin.y + pen.height)
        # First interior cell: just inside the pen from the gate
        first_interior = Point(pen.center.x, pen.origin.y + pen.height - 1)
        # Sanity: first_interior must be inside pen
        assert pen.contains(first_interior), f"first_interior {first_interior} should be inside pen"
        # Should NOT be blocked (sheep coming from gate into pen)
        blocked = env._sheep_move_blocked(  # noqa: SLF001 (test-only access)
            position=gate_cell,
            candidate=first_interior,
            dog_positions=[],
            blocked_positions=set(),
        )
        assert not blocked, (
            f"Sheep move from gate {gate_cell} into pen interior {first_interior} "
            "should NOT be blocked"
        )

    def test_sheep_penned_flag_after_step(self) -> None:
        """Sheep placed inside pen bounds should have penned=True after any step."""
        config = _stage_config(1)
        env = SheepdogEnvironment(config)
        env.reset(seed=0)

        # Manually teleport sheep to pen center (bypassing environment movement)
        pen = env.pen
        for sheep in env._sheep:  # noqa: SLF001 (test-only access)
            sheep.position = pen.center
            sheep.penned = False  # reset to confirm step() sets it

        # One no-op step to trigger penning logic
        actions = ["wait"] * len(env.dogs)
        _, _ = env.step(actions)

        for sheep in env.sheep:
            assert sheep.penned, (
                f"Sheep at {sheep.position} inside pen bounds should be penned after step"
            )

    def test_success_fires_when_all_sheep_penned(self) -> None:
        """Success must become True when all sheep are inside the pen."""
        config = _stage_config(1)
        env = SheepdogEnvironment(config)
        env.reset(seed=0)

        pen = env.pen
        for sheep in env._sheep:  # noqa: SLF001 (test-only access)
            sheep.position = pen.center
            sheep.penned = True  # pre-mark so step does not move them

        actions = ["wait"] * len(env.dogs)
        snapshot, _ = env.step(actions)

        assert snapshot.penned_count == len(env.sheep), (
            f"Expected all {len(env.sheep)} sheep penned, got {snapshot.penned_count}"
        )
        assert snapshot.success, "success should be True when all sheep are penned"

    def test_average_distance_to_pen_near_zero_when_all_penned(self) -> None:
        """average_distance_to_pen should be very close to 0 when all sheep are at pen center."""
        config = _stage_config(1)
        env = SheepdogEnvironment(config)
        env.reset(seed=0)

        pen = env.pen
        for sheep in env._sheep:  # noqa: SLF001 (test-only access)
            sheep.position = pen.center
            sheep.penned = True

        actions = ["wait"] * len(env.dogs)
        snapshot, _ = env.step(actions)

        assert snapshot.average_distance_to_pen < 1.0, (
            f"average_distance_to_pen should be near 0 when sheep are at pen center, "
            f"got {snapshot.average_distance_to_pen:.2f}"
        )

    def test_sheep_can_enter_pen_from_bottom(self) -> None:
        """Sheep should be able to step from the gate cell into the pen without being blocked."""
        config = _stage_config(1)
        env = SheepdogEnvironment(config)
        env.reset(seed=0)
        pen = env.pen

        # Gate cell: immediately below pen at center x (opening="bottom")
        gate_cell = Point(pen.center.x, pen.origin.y + pen.height)
        first_interior = Point(pen.center.x, pen.origin.y + pen.height - 1)

        # Confirm gate is not a fence cell
        assert gate_cell not in env._fence_cells, (  # noqa: SLF001
            f"Gate cell {gate_cell} should NOT be a fence cell"
        )
        # Confirm first_interior is not a fence cell
        assert first_interior not in env._fence_cells, (  # noqa: SLF001
            f"Interior cell {first_interior} should NOT be a fence cell"
        )

    @pytest.mark.parametrize("stage", [1, 2, 3, 4, 5])
    def test_pen_origin_is_at_right_edge(self, stage: int) -> None:
        """Pen origin x should equal width - pen_width for every curriculum stage."""
        config = _stage_config(stage)
        env = SheepdogEnvironment(config)
        env.reset(seed=0)
        expected_x = config.environment.width - config.environment.pen_width
        assert env.pen.origin.x == expected_x, (
            f"Stage {stage}: pen.origin.x={env.pen.origin.x} != "
            f"width({config.environment.width}) - pen_width({config.environment.pen_width}) "
            f"= {expected_x}"
        )
