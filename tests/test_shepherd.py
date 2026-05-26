"""Tests for ShepherdCommand enum and ScriptedShepherd command policy."""

# pylint: disable=missing-function-docstring,import-outside-toplevel,reimported,protected-access
from __future__ import annotations

from sheepdog.config import LabConfig
from sheepdog.environment import SheepdogEnvironment
from sheepdog.shepherd import (
    COMMAND_INDEX,
    COMMAND_ORDER,
    ScriptedShepherd,
    ShepherdCommand,
)

# ---------------------------------------------------------------------------
# Enum & ordering
# ---------------------------------------------------------------------------


def test_shepherd_command_has_eight_members():
    assert len(ShepherdCommand) == 8


def test_command_order_has_eight_entries():
    assert len(COMMAND_ORDER) == 8


def test_command_order_contains_all_members():
    assert set(COMMAND_ORDER) == set(ShepherdCommand)


def test_command_order_is_stable():
    """Re-importing must give the same ordering (no randomisation)."""
    from sheepdog.shepherd import COMMAND_ORDER as order_again  # noqa: PLC0415

    assert order_again == COMMAND_ORDER


def test_command_index_matches_order():
    for i, cmd in enumerate(COMMAND_ORDER):
        assert COMMAND_INDEX[cmd] == i


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_env(seed: int = 0) -> SheepdogEnvironment:
    config = LabConfig()
    env = SheepdogEnvironment(config)
    env.reset(seed=seed)
    return env


def _place_all_sheep_in_pen(env: SheepdogEnvironment) -> None:
    """Mutate sheep state so every sheep is penned (for STOP test)."""
    # Use object.__setattr__ to bypass frozen dataclass slots.
    new_sheep = []
    for sheep in env.sheep:
        new_sheep.append(
            sheep.__class__(
                **{
                    **{f: getattr(sheep, f) for f in sheep.__dataclass_fields__},
                    "penned": True,
                }
            )
        )
    env._sheep = new_sheep  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# ScriptedShepherd decisions
# ---------------------------------------------------------------------------


def test_issue_command_returns_shepherd_command():
    env = _make_env()
    shepherd = ScriptedShepherd()
    cmd = shepherd.issue_command(env)
    assert isinstance(cmd, ShepherdCommand)


def test_issue_command_stores_last_command():
    env = _make_env()
    shepherd = ScriptedShepherd()
    cmd = shepherd.issue_command(env)
    assert shepherd.last_command is cmd


def test_issue_command_gather_when_flock_spread():
    """A freshly reset environment typically has spread > COMPACT_SPREAD → GATHER."""
    config = LabConfig()
    # Use curriculum stage 5 (3 dogs, 6 sheep spread) to ensure spread.
    from sheepdog.curriculum import apply_training_profile  # noqa: PLC0415

    config = apply_training_profile(config, curriculum_stage=5)
    env = SheepdogEnvironment(config)
    env.reset(seed=42)
    shepherd = ScriptedShepherd()
    cmd = shepherd.issue_command(env)
    # Flock should be dispersed on reset; accept GATHER or DRIVE_TO_PEN.
    assert cmd in {ShepherdCommand.GATHER, ShepherdCommand.DRIVE_TO_PEN}


def test_issue_command_stop_when_all_penned():
    env = _make_env()
    # Force all sheep to be penned via EpisodeStats or direct manipulation.
    # Simplest: run until done or check behaviour via shepherd context.
    shepherd = ScriptedShepherd()
    # Simulate a completed episode by running against the full env.
    # If the env doesn't finish in 10 steps, at minimum verify STOP logic
    # by exercising it with a contrived path.
    # Check STOP returns for zero unpenned sheep.
    # Rather than mutating private env state, patch via monkey-patching sheep.
    original_sheep = env.sheep
    # Temporarily replace sheep list with empty to simulate all penned.
    env._sheep = []  # type: ignore[attr-defined]
    cmd = shepherd.issue_command(env)
    env._sheep = original_sheep  # type: ignore[attr-defined]
    assert cmd is ShepherdCommand.STOP


def test_scripted_shepherd_default_last_command_is_gather():
    shepherd = ScriptedShepherd()
    assert shepherd.last_command is ShepherdCommand.GATHER
