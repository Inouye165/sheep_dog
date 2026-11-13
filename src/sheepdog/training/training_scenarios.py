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


def create_partial_pen_with_stray_scenario(
    *,
    seed: int,
    config: EnvironmentConfig,
) -> Scenario:
    """Create a scenario with most sheep near the pen and 1–2 stray outliers.

    Dogs are placed near the pen entrance, tempting them to guard already-close
    sheep instead of fetching the distant stray.  This forces the policy to
    learn to prioritise the worst-case sheep.
    """
    width = config.width
    height = config.height
    sheep_count = config.sheep
    dog_count = config.dogs

    rng = Random(seed + config.seed_offset)

    pen_right = width - 1
    pen_top = 1
    margin = max(2, min(width, height) // 10)

    # Number of strays: 1 if fewer than 4 sheep, else 2
    stray_count = 1 if sheep_count < 4 else 2
    near_count = sheep_count - stray_count

    sheep: list[AgentLayout] = []

    # Near-pen sheep: scatter them within ~20% of field width from the right edge
    near_zone_x_min = max(margin, pen_right - max(3, width // 5))
    near_zone_x_max = pen_right - margin
    near_zone_y_min = max(margin, pen_top)
    near_zone_y_max = min(height - margin - 1, pen_top + config.pen_height + margin * 2)

    for i in range(near_count):
        x = rng.randint(near_zone_x_min, near_zone_x_max)
        y = rng.randint(near_zone_y_min, near_zone_y_max)
        sheep.append(AgentLayout(index=i, x=x, y=y))

    # Stray sheep: place in the far corner (bottom-left) from the pen
    stray_zone_x_max = max(margin + 1, width // 5)
    stray_zone_y_min = max(margin, height // 2)
    stray_zone_y_max = height - margin - 1

    for i in range(stray_count):
        x = rng.randint(margin, stray_zone_x_max)
        y = rng.randint(stray_zone_y_min, stray_zone_y_max)
        sheep.append(AgentLayout(index=near_count + i, x=x, y=y))

    # Dogs: place near pen entrance (just left of the opening)
    dogs: list[AgentLayout] = []
    pen_entrance_x = pen_right - config.pen_width - margin
    pen_entrance_y = pen_top + config.pen_height // 2
    for i in range(dog_count):
        x = max(margin, pen_entrance_x - i * 2)
        y = pen_entrance_y + rng.randint(-margin, margin)
        y = max(margin, min(height - margin - 1, y))
        dogs.append(AgentLayout(index=i, x=x, y=y))

    pen = PenLayout(
        origin_x=pen_right - config.pen_width,
        origin_y=pen_top,
        width=config.pen_width,
        height=config.pen_height,
        opening=config.pen_opening,
    )

    return Scenario(
        id=f"partial_pen_with_stray_{seed}",
        name="partial_pen_with_stray",
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
        description=(
            f"Most sheep near pen, {stray_count} stray in far corner — "
            "trains dogs to fetch outliers not guard the already-close flock"
        ),
    )


def create_subpen_flock_scenario(
    *,
    seed: int,
    config: EnvironmentConfig,
) -> Scenario:
    """Create a scenario with sheep spawned in the sub-pen zone (below the pen).

    Models the geometry of Seed 11: sheep spawn beneath the top-right corner pen,
    and dogs approach from the left. Tests split prevention and lone-straggler recovery.
    """
    width = config.width
    height = config.height
    sheep_count = config.sheep
    dog_count = config.dogs
    rng = Random(seed + config.seed_offset)

    pen_origin_x = width - config.pen_width
    pen_origin_y = 1
    pen = PenLayout(
        origin_x=pen_origin_x,
        origin_y=pen_origin_y,
        width=config.pen_width,
        height=config.pen_height,
        opening=config.pen_opening,
    )

    # Subpen zone: beneath the corner pen (x in [0.70W, 0.85W], y in [0.30H, 0.48H])
    center_x = min(width - config.pen_width - 2, max(10, int(width * 0.78))) + rng.randint(-3, 3)
    center_y = max(config.pen_height + 4, min(height - 10, int(height * 0.38))) + rng.randint(-3, 3)

    sheep: list[AgentLayout] = []
    occupied: set[tuple[int, int]] = set()
    for i in range(sheep_count):
        placed = False
        for _ in range(50):
            sx = max(2, min(width - 3, center_x + rng.randint(-3, 3)))
            sy = max(2, min(height - 3, center_y + rng.randint(-3, 3)))
            if (sx, sy) not in occupied:
                occupied.add((sx, sy))
                sheep.append(AgentLayout(index=i, x=sx, y=sy))
                placed = True
                break
        if not placed:
            sheep.append(AgentLayout(index=i, x=center_x + i, y=center_y))

    # Standard left-side dogs
    dogs: list[AgentLayout] = []
    for i in range(dog_count):
        dx = 2 + i * 2
        dy = max(2, min(height - 3, (height // 2) + rng.randint(-2, 2)))
        dogs.append(AgentLayout(index=i, x=dx, y=dy))

    return Scenario(
        id=f"subpen_flock_{seed}",
        name="subpen_flock",
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
        description="Flock spawned in sub-pen region beneath pen entrance (Seed-11-like archetype)",
    )


def create_gate_wall_flock_scenario(
    *,
    seed: int,
    config: EnvironmentConfig,
) -> Scenario:
    """Create a scenario with sheep along the top wall adjacent to the pen entrance.

    Models the geometry of Seed 53: sheep spawn near the top perimeter wall to the left
    of the pen opening, testing gate-corridor entry and preventing hovering standoff.
    """
    width = config.width
    height = config.height
    sheep_count = config.sheep
    dog_count = config.dogs
    rng = Random(seed + config.seed_offset)

    pen_origin_x = width - config.pen_width
    pen_origin_y = 1
    pen = PenLayout(
        origin_x=pen_origin_x,
        origin_y=pen_origin_y,
        width=config.pen_width,
        height=config.pen_height,
        opening=config.pen_opening,
    )

    # Top-wall gate flank zone: x in [0.55W, 0.70W], y in [3, 8]
    center_x = max(10, min(pen_origin_x - 10, int(width * 0.62))) + rng.randint(-3, 3)
    center_y = rng.randint(3, 7)

    sheep: list[AgentLayout] = []
    occupied: set[tuple[int, int]] = set()
    for i in range(sheep_count):
        placed = False
        for _ in range(50):
            sx = max(2, min(pen_origin_x - 2, center_x + rng.randint(-3, 3)))
            sy = max(2, min(10, center_y + rng.randint(-2, 2)))
            if (sx, sy) not in occupied:
                occupied.add((sx, sy))
                sheep.append(AgentLayout(index=i, x=sx, y=sy))
                placed = True
                break
        if not placed:
            sheep.append(AgentLayout(index=i, x=center_x + i, y=center_y))

    # Standard left-side dogs
    dogs: list[AgentLayout] = []
    for i in range(dog_count):
        dx = 2 + i * 2
        dy = max(2, min(height - 3, (height // 2) + rng.randint(-2, 2)))
        dogs.append(AgentLayout(index=i, x=dx, y=dy))

    return Scenario(
        id=f"gate_wall_flock_{seed}",
        name="gate_wall_flock",
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
        description="Flock along top wall adjacent to pen entrance corridor (Seed-53-like archetype)",
    )


def create_isolated_stray_scenario(
    *,
    seed: int,
    config: EnvironmentConfig,
) -> Scenario:
    """Create a scenario with a main flock and one separated stray.

    Models the geometry of Seed 41/37: an isolated stray separated from the main flock.
    """
    width = config.width
    height = config.height
    sheep_count = config.sheep
    dog_count = config.dogs
    rng = Random(seed + config.seed_offset)

    pen_origin_x = width - config.pen_width
    pen_origin_y = 1
    pen = PenLayout(
        origin_x=pen_origin_x,
        origin_y=pen_origin_y,
        width=config.pen_width,
        height=config.pen_height,
        opening=config.pen_opening,
    )

    main_count = max(1, sheep_count - 1)
    center_x = rng.randint(width // 3, width // 2)
    center_y = rng.randint(height // 4, 3 * height // 4)

    sheep: list[AgentLayout] = []
    occupied: set[tuple[int, int]] = set()
    for i in range(main_count):
        placed = False
        for _ in range(50):
            sx = max(2, min(width - 3, center_x + rng.randint(-2, 2)))
            sy = max(2, min(height - 3, center_y + rng.randint(-2, 2)))
            if (sx, sy) not in occupied:
                occupied.add((sx, sy))
                sheep.append(AgentLayout(index=i, x=sx, y=sy))
                placed = True
                break
        if not placed:
            sheep.append(AgentLayout(index=i, x=center_x + i, y=center_y))

    # Isolated stray: displaced 8 to 14 units away from center
    stray_offset_x = rng.choice([-1, 1]) * rng.randint(8, 14)
    stray_offset_y = rng.choice([-1, 1]) * rng.randint(6, 12)
    stray_x = max(2, min(width - 3, center_x + stray_offset_x))
    stray_y = max(2, min(height - 3, center_y + stray_offset_y))
    sheep.append(AgentLayout(index=main_count, x=stray_x, y=stray_y))

    # Standard left-side dogs
    dogs: list[AgentLayout] = []
    for i in range(dog_count):
        dx = 2 + i * 2
        dy = max(2, min(height - 3, (height // 2) + rng.randint(-2, 2)))
        dogs.append(AgentLayout(index=i, x=dx, y=dy))

    return Scenario(
        id=f"isolated_stray_{seed}",
        name="isolated_stray",
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
        description="Main flock with one separated stray (Seed-41/37-like archetype)",
    )


def create_high_pen_standoff_scenario(
    *,
    seed: int,
    config: EnvironmentConfig,
) -> Scenario:
    """Create a scenario with an upper-boundary pen and south-facing gate.

    Models the failure geometry of Seeds 37 and 73: pen positioned near top edge
    (y in [3, 24]), opening="bottom", with sheep flock located in the central field.
    Forces dogs to coordinate north-pushing without 2-position oscillation deadlock.
    """
    width = config.width
    height = config.height
    sheep_count = config.sheep
    dog_count = config.dogs
    rng = Random(seed + config.seed_offset)

    # Upper boundary pen: y between 3 and min(24, int(height * 0.25))
    pen_origin_y = rng.randint(3, max(3, min(24, int(height * 0.22))))
    pen_min_x = 4
    pen_max_x = max(pen_min_x, width - config.pen_width - 6)
    pen_origin_x = rng.randint(pen_min_x, pen_max_x)

    pen = PenLayout(
        origin_x=pen_origin_x,
        origin_y=pen_origin_y,
        width=config.pen_width,
        height=config.pen_height,
        opening="bottom",
    )

    # Central flock below the pen
    center_x = max(8, min(width - 8, pen_origin_x + config.pen_width // 2 + rng.randint(-8, 8)))
    min_sheep_y = pen_origin_y + config.pen_height + 5
    max_sheep_y = max(min_sheep_y + 2, min(height - 12, int(height * 0.55)))
    center_y = rng.randint(min_sheep_y, max_sheep_y)

    sheep: list[AgentLayout] = []
    occupied: set[tuple[int, int]] = set()
    for i in range(sheep_count):
        placed = False
        for _ in range(50):
            sx = max(2, min(width - 3, center_x + rng.randint(-5, 5)))
            sy = max(min_sheep_y, min(height - 4, center_y + rng.randint(-4, 4)))
            if (sx, sy) not in occupied:
                occupied.add((sx, sy))
                sheep.append(AgentLayout(index=i, x=sx, y=sy))
                placed = True
                break
        if not placed:
            sheep.append(AgentLayout(index=i, x=center_x + i, y=center_y))

    # Dogs positioned south and flanking the flock to herd north toward the pen
    dogs: list[AgentLayout] = []
    dog_base_y = min(height - 3, center_y + rng.randint(8, 14))
    dog_spread = max(4, min(12, width // (dog_count + 1)))
    for i in range(dog_count):
        dx = max(2, min(width - 3, center_x + (i - dog_count // 2) * dog_spread + rng.randint(-1, 1)))
        dy = max(2, min(height - 3, dog_base_y + rng.randint(-2, 2)))
        dogs.append(AgentLayout(index=i, x=dx, y=dy))

    return Scenario(
        id=f"high_pen_standoff_{seed}",
        name="high_pen_standoff",
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
        description="High boundary pen with south-facing entrance to remediate 2-position oscillation standoff (Seed-37/73 archetype)",
    )


def create_wall_stray_extraction_scenario(
    *,
    seed: int,
    config: EnvironmentConfig,
) -> Scenario:
    """Create a scenario with majority of sheep at/in pen and 1–2 strays against walls/corners.

    Models the 5-penned failure mode of Seeds 11, 23, 41, 53, 59, 61, 71.
    Trains the team to detach from gate defense and retrieve wall/corner strays.
    """
    width = config.width
    height = config.height
    sheep_count = config.sheep
    dog_count = config.dogs
    rng = Random(seed + config.seed_offset)

    # Determine pen placement (interior if configured, else corner/wall)
    placement = (getattr(config, "pen_placement", "corner") or "corner").lower()
    if placement in ("interior", "random"):
        min_x = 4
        max_x = max(min_x, width - config.pen_width - 6)
        min_y = 4
        max_y = max(min_y, height - config.pen_height - 6)
        pen_origin_x = rng.randint(min_x, max_x)
        pen_origin_y = rng.randint(min_y, max_y)
    else:
        pen_origin_x = width - config.pen_width - 2
        pen_origin_y = 2

    pen = PenLayout(
        origin_x=pen_origin_x,
        origin_y=pen_origin_y,
        width=config.pen_width,
        height=config.pen_height,
        opening=config.pen_opening,
    )

    # 1 stray (or 2 if large flock and rng rolls 20%)
    stray_count = 1 if sheep_count <= 4 or rng.random() > 0.20 else 2
    near_count = sheep_count - stray_count

    sheep: list[AgentLayout] = []
    occupied: set[tuple[int, int]] = set()

    # Place near-pen sheep directly in front of or inside the pen entrance
    entrance_center_x = pen_origin_x + config.pen_width // 2
    entrance_y = pen_origin_y + config.pen_height + 2 if config.pen_opening == "bottom" else max(2, pen_origin_y - 2)

    for i in range(near_count):
        placed = False
        for _ in range(50):
            sx = max(2, min(width - 3, entrance_center_x + rng.randint(-3, 3)))
            sy = max(2, min(height - 3, entrance_y + rng.randint(-2, 3)))
            if (sx, sy) not in occupied:
                occupied.add((sx, sy))
                sheep.append(AgentLayout(index=i, x=sx, y=sy))
                placed = True
                break
        if not placed:
            sheep.append(AgentLayout(index=i, x=entrance_center_x + i, y=entrance_y))

    # Place stray sheep tight against one of the perimeter walls or corners
    wall_choices = ["bottom", "right", "left", "top"]
    for i in range(stray_count):
        wall = rng.choice(wall_choices)
        if wall == "bottom":
            sx = rng.randint(4, max(4, width - 5))
            sy = rng.randint(max(2, height - 3), height - 2)
        elif wall == "right":
            sx = rng.randint(max(2, width - 3), width - 2)
            sy = rng.randint(4, max(4, height - 5))
        elif wall == "left":
            sx = rng.randint(2, min(4, width - 2))
            sy = rng.randint(4, max(4, height - 5))
        else:  # top
            sx = rng.randint(4, max(4, width - 5))
            sy = rng.randint(2, min(4, height - 2))
        sheep.append(AgentLayout(index=near_count + i, x=sx, y=sy))

    # Dogs: 1 holding near pen entrance, other dogs positioned toward the stray
    dogs: list[AgentLayout] = []
    dogs.append(AgentLayout(index=0, x=entrance_center_x - 2, y=entrance_y + 1))
    stray_ref_x = sheep[-1].x
    stray_ref_y = sheep[-1].y
    for i in range(1, dog_count):
        t = i / dog_count
        dx = int(entrance_center_x + t * (stray_ref_x - entrance_center_x))
        dy = int(entrance_y + t * (stray_ref_y - entrance_y))
        dogs.append(AgentLayout(index=i, x=max(2, min(width - 3, dx)), y=max(2, min(height - 3, dy))))

    return Scenario(
        id=f"wall_stray_extraction_{seed}",
        name="wall_stray_extraction",
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
        description="Near-pen flock with lone stray trapped against perimeter wall/corner (Seeds 11/23/41/53/59/61/71 archetype)",
    )


# Scenario type registry
SCENARIO_BUILDERS: dict[str, callable] = {
    "scattered_sheep": create_scattered_sheep_scenario,
    "split_flock": create_split_flock_scenario,
    "corner_huddle": create_corner_huddle_scenario,
    "normal_random": create_normal_random_scenario,
    "partial_pen_with_stray": create_partial_pen_with_stray_scenario,
    "subpen_flock": create_subpen_flock_scenario,
    "gate_wall_flock": create_gate_wall_flock_scenario,
    "isolated_stray": create_isolated_stray_scenario,
    "high_pen_standoff": create_high_pen_standoff_scenario,
    "wall_stray_extraction": create_wall_stray_extraction_scenario,
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
