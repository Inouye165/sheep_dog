"""Predefined difficult training scenarios for robust policy training."""

from __future__ import annotations

from random import Random

from sheepdog.config import EnvironmentConfig
from sheepdog.evaluation.scenarios import AgentLayout, PenLayout, Scenario


def create_scattered_sheep_scenario(
    *,
    seed: int,
    config: EnvironmentConfig,
) -> Scenario:
    """Create a scenario with sheep scattered across the field.

    Sheep are spread widely across valid positions to force the policy to
    handle dispersed flock recovery.
    """
    width = config.width
    height = config.height
    sheep_count = config.sheep
    dog_count = config.dogs

    # Place sheep in a scattered pattern across the field
    # Use relative positions to work with different grid sizes
    margin = max(2, min(width, height) // 10)
    safe_width = width - 2 * margin
    safe_height = height - 2 * margin

    sheep: list[AgentLayout] = []
    for i in range(sheep_count):
        # Distribute sheep across the field in a grid-like pattern
        row = i // 3
        col = i % 3
        x = margin + (col * safe_width // 3) + (safe_width // 6)
        y = margin + (row * safe_height // 3) + (safe_height // 6)
        sheep.append(AgentLayout(index=i, x=x, y=y))

    # Place dogs in a reasonable recovery position (center-left)
    dogs: list[AgentLayout] = []
    for i in range(dog_count):
        x = margin + (i * (safe_width // (dog_count + 1)))
        y = height // 2
        dogs.append(AgentLayout(index=i, x=x, y=y))

    pen = PenLayout(
        origin_x=width - config.pen_width,
        origin_y=1,
        width=config.pen_width,
        height=config.pen_height,
        opening=config.pen_opening,
    )

    return Scenario(
        id=f"scattered_sheep_{seed}",
        name="scattered_sheep",
        created_at="",
        seed=seed,
        width=width,
        height=height,
        dogs=tuple(dogs),
        sheep=tuple(sheep),
        pen=pen,
        sheep_personality_strength=config.sheep_personality_strength,
        sheep_personality_seed_offset=config.sheep_personality_seed_offset,
        seed_offset=config.seed_offset,
        description="Sheep scattered across the field for dispersed flock recovery training",
    )


def create_split_flock_scenario(
    *,
    seed: int,
    config: EnvironmentConfig,
) -> Scenario:
    """Create a scenario with the flock split into two separated groups.

    Two groups of sheep are placed on opposite sides of the field to force
    the policy to handle flock splitting and regrouping.
    """
    width = config.width
    height = config.height
    sheep_count = config.sheep
    dog_count = config.dogs

    # Split sheep into two groups
    group1_size = sheep_count // 2
    group2_size = sheep_count - group1_size

    margin = max(2, min(width, height) // 10)

    sheep: list[AgentLayout] = []
    # Group 1: top-left quadrant
    for i in range(group1_size):
        x = margin + (i * (width // 4 // max(1, group1_size)))
        y = margin + (height // 4)
        sheep.append(AgentLayout(index=i, x=x, y=y))

    # Group 2: bottom-right quadrant
    for i in range(group2_size):
        x = width - margin - (i * (width // 4 // max(1, group2_size)))
        y = height - margin - (height // 4)
        sheep.append(AgentLayout(index=group1_size + i, x=x, y=y))

    # Place dogs in the center to recover both groups
    dogs: list[AgentLayout] = []
    for i in range(dog_count):
        x = width // 2 + (i - dog_count // 2) * 3
        y = height // 2
        dogs.append(AgentLayout(index=i, x=x, y=y))

    pen = PenLayout(
        origin_x=width - config.pen_width,
        origin_y=1,
        width=config.pen_width,
        height=config.pen_height,
        opening=config.pen_opening,
    )

    return Scenario(
        id=f"split_flock_{seed}",
        name="split_flock",
        created_at="",
        seed=seed,
        width=width,
        height=height,
        dogs=tuple(dogs),
        sheep=tuple(sheep),
        pen=pen,
        sheep_personality_strength=config.sheep_personality_strength,
        sheep_personality_seed_offset=config.sheep_personality_seed_offset,
        seed_offset=config.seed_offset,
        description="Flock split into two separated groups for regrouping training",
    )


def create_corner_huddle_scenario(
    *,
    seed: int,
    config: EnvironmentConfig,
) -> Scenario:
    """Create a scenario with sheep huddled in a corner.

    Sheep are clustered in a corner away from the pen to force the policy
    to handle corner extraction and movement.
    """
    width = config.width
    height = config.height
    sheep_count = config.sheep
    dog_count = config.dogs

    margin = max(2, min(width, height) // 10)

    # Huddle sheep in the bottom-left corner (opposite from pen which is top-right)
    sheep: list[AgentLayout] = []
    huddle_radius = max(2, min(width, height) // 15)
    for i in range(sheep_count):
        # Arrange in a small cluster
        row = i // 3
        col = i % 3
        x = margin + col * huddle_radius
        y = height - margin - row * huddle_radius
        sheep.append(AgentLayout(index=i, x=x, y=y))

    # Place dogs between the huddle and the pen
    dogs: list[AgentLayout] = []
    for i in range(dog_count):
        x = width // 3 + (i * (width // 3 // max(1, dog_count)))
        y = height // 2
        dogs.append(AgentLayout(index=i, x=x, y=y))

    pen = PenLayout(
        origin_x=width - config.pen_width,
        origin_y=1,
        width=config.pen_width,
        height=config.pen_height,
        opening=config.pen_opening,
    )

    return Scenario(
        id=f"corner_huddle_{seed}",
        name="corner_huddle",
        created_at="",
        seed=seed,
        width=width,
        height=height,
        dogs=tuple(dogs),
        sheep=tuple(sheep),
        pen=pen,
        sheep_personality_strength=config.sheep_personality_strength,
        sheep_personality_seed_offset=config.sheep_personality_seed_offset,
        seed_offset=config.seed_offset,
        description="Sheep huddled in a corner for extraction training",
    )


def create_normal_random_scenario(
    *,
    seed: int,
    config: EnvironmentConfig,
) -> Scenario:
    """Create a scenario with normal random starting positions.

    This scenario type is equivalent to a normal env.reset() but represented
    as a scenario for consistency in the scenario training framework.
    """
    width = config.width
    height = config.height
    sheep_count = config.sheep
    dog_count = config.dogs

    rng = Random(seed + config.seed_offset)

    margin = max(2, min(width, height) // 10)

    # Random sheep positions (similar to env._initial_sheep logic)
    sheep: list[AgentLayout] = []
    for i in range(sheep_count):
        x = rng.randint(margin, width - margin - 1)
        y = rng.randint(margin, height - margin - 1)
        sheep.append(AgentLayout(index=i, x=x, y=y))

    # Random dog positions (similar to env._initial_dogs logic)
    dogs: list[AgentLayout] = []
    for i in range(dog_count):
        x = rng.randint(margin, width - margin - 1)
        y = rng.randint(margin, height - margin - 1)
        dogs.append(AgentLayout(index=i, x=x, y=y))

    pen = PenLayout(
        origin_x=width - config.pen_width,
        origin_y=1,
        width=config.pen_width,
        height=config.pen_height,
        opening=config.pen_opening,
    )

    return Scenario(
        id=f"normal_random_{seed}",
        name="normal_random",
        created_at="",
        seed=seed,
        width=width,
        height=height,
        dogs=tuple(dogs),
        sheep=tuple(sheep),
        pen=pen,
        sheep_personality_strength=config.sheep_personality_strength,
        sheep_personality_seed_offset=config.sheep_personality_seed_offset,
        seed_offset=config.seed_offset,
        description="Normal random starting positions",
    )


# Scenario type registry
SCENARIO_BUILDERS: dict[str, callable] = {
    "scattered_sheep": create_scattered_sheep_scenario,
    "split_flock": create_split_flock_scenario,
    "corner_huddle": create_corner_huddle_scenario,
    "normal_random": create_normal_random_scenario,
}


def get_scenario_builder(scenario_type: str) -> callable:
    """Return the scenario builder function for the given type."""
    if scenario_type not in SCENARIO_BUILDERS:
        raise ValueError(
            f"Unknown scenario type: {scenario_type}. "
            f"Available types: {list(SCENARIO_BUILDERS.keys())}"
        )
    return SCENARIO_BUILDERS[scenario_type]


def list_scenario_types() -> tuple[str, ...]:
    """Return all available scenario type names."""
    return tuple(SCENARIO_BUILDERS.keys())
