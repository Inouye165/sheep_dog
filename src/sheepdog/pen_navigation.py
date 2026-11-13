"""Cached static routes through a pen's open entrance."""

from __future__ import annotations

from collections import deque
from functools import lru_cache

from sheepdog.entities import Pen, Point


class PenNavigation:
    """Shortest cardinal entry distances and fence-safe approach waypoints."""

    def __init__(self, pen: Pen, width: int, height: int) -> None:
        self.pen = pen
        self.width = width
        self.height = height
        self.fences = pen.fence_cells()
        self.distances: dict[Point, int] = {}
        queue: deque[Point] = deque()
        for x in range(max(0, pen.origin.x), min(width, pen.origin.x + pen.width)):
            for y in range(max(0, pen.origin.y), min(height, pen.origin.y + pen.height)):
                point = Point(x, y)
                self.distances[point] = 0
                queue.append(point)
        while queue:
            point = queue.popleft()
            for neighbor in self._neighbors(point):
                if neighbor not in self.fences and neighbor not in self.distances:
                    self.distances[neighbor] = self.distances[point] + 1
                    queue.append(neighbor)

    def _neighbors(self, point: Point) -> tuple[Point, ...]:
        return tuple(
            neighbor
            for neighbor in (
                Point(point.x - 1, point.y),
                Point(point.x + 1, point.y),
                Point(point.x, point.y - 1),
                Point(point.x, point.y + 1),
            )
            if 0 <= neighbor.x < self.width and 0 <= neighbor.y < self.height
        )

    def distance(self, point: Point) -> int:
        """Return legal grid moves to entry, rejecting unreachable positions."""
        if point not in self.distances:
            raise ValueError(
                f"No route to pen entrance from {point}: "
                f"pen={self.pen}, field={self.width}x{self.height}, "
                f"on_fence={point in self.fences}."
            )
        return self.distances[point]

    def clear_segment(self, start: Point, end: Point) -> bool:
        """Check every touched grid cell, including both sides of diagonal corners."""
        x, y = start.x, start.y
        dx, dy = end.x - x, end.y - y
        nx, ny = abs(dx), abs(dy)
        sx = 1 if dx > 0 else -1
        sy = 1 if dy > 0 else -1
        ix = iy = 0
        if start in self.fences:
            return False
        while ix < nx or iy < ny:
            decision = (1 + 2 * ix) * ny - (1 + 2 * iy) * nx
            if decision == 0:
                if Point(x + sx, y) in self.fences or Point(x, y + sy) in self.fences:
                    return False
                x, y = x + sx, y + sy
                ix, iy = ix + 1, iy + 1
            elif decision < 0:
                x += sx
                ix += 1
            else:
                y += sy
                iy += 1
            if Point(x, y) in self.fences:
                return False
        return True

    def target(self, point: Point) -> Point:
        """Keep direct approaches; otherwise aim at the first route corner."""
        self.distance(point)
        if self.clear_segment(point, self.pen.center):
            return self.pen.center
        current = point
        direction: tuple[int, int] | None = None
        while self.distance(current) > 0:
            candidate = min(
                (
                    neighbor
                    for neighbor in self._neighbors(current)
                    if self.distances.get(neighbor) == self.distance(current) - 1
                ),
                key=lambda neighbor: neighbor.distance_to(self.pen.center),
            )
            next_direction = (candidate.x - current.x, candidate.y - current.y)
            if direction is not None and next_direction != direction:
                return current
            direction = next_direction
            current = candidate
        return current


@lru_cache(maxsize=32)
def pen_navigation(pen: Pen, width: int, height: int) -> PenNavigation:
    """Reuse static routing across steps and identical episode layouts."""
    return PenNavigation(pen, width, height)
