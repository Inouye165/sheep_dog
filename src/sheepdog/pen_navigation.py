"""Cached static routes through a pen's open entrance."""

from __future__ import annotations

from collections import deque
from functools import lru_cache
from math import hypot, inf

from sheepdog.entities import Pen, Point


class PenNavigation:
    """Shortest cardinal entry distances and fence-safe approach waypoints."""

    def __init__(self, pen: Pen, width: int, height: int) -> None:
        self.pen = pen
        self.width = width
        self.height = height
        self.fences = pen.fence_cells()
        self._fence_grid = bytearray(width * height)
        for cell in self.fences:
            if 0 <= cell.x < width and 0 <= cell.y < height:
                self._fence_grid[cell.y * width + cell.x] = 1
        self._dist_grid = [-1] * (width * height)
        self._target_cache: dict[int, Point] = {}
        self.distances: dict[Point, int] = {}
        queue: deque[Point] = deque()
        for x in range(max(0, pen.origin.x), min(width, pen.origin.x + pen.width)):
            for y in range(max(0, pen.origin.y), min(height, pen.origin.y + pen.height)):
                point = Point(x, y)
                self.distances[point] = 0
                self._dist_grid[y * width + x] = 0
                queue.append(point)
        fence_grid = self._fence_grid
        dist_grid = self._dist_grid
        while queue:
            point = queue.popleft()
            px, py = point.x, point.y
            next_dist = dist_grid[py * width + px] + 1
            for nx, ny in ((px - 1, py), (px + 1, py), (px, py - 1), (px, py + 1)):
                if 0 <= nx < width and 0 <= ny < height:
                    n_idx = ny * width + nx
                    if not fence_grid[n_idx] and dist_grid[n_idx] < 0:
                        dist_grid[n_idx] = next_dist
                        neighbor = Point(nx, ny)
                        self.distances[neighbor] = next_dist
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
        px, py = point.x, point.y
        if 0 <= px < self.width and 0 <= py < self.height:
            d = self._dist_grid[py * self.width + px]
            if d >= 0:
                return d
        raise ValueError(
            f"No route to pen entrance from {point}: "
            f"pen={self.pen}, field={self.width}x{self.height}, "
            f"on_fence={point in self.fences}."
        )

    def _clear_segment_fallback(self, start: Point, end: Point) -> bool:
        """Fallback segment check for out-of-bounds endpoints."""
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

    def clear_segment(self, start: Point, end: Point) -> bool:
        """Check every touched grid cell, including both sides of diagonal corners."""
        w = self.width
        h = self.height
        x, y = start.x, start.y
        ex, ey = end.x, end.y
        if not (0 <= x < w and 0 <= y < h and 0 <= ex < w and 0 <= ey < h):
            return self._clear_segment_fallback(start, end)
        fences = self._fence_grid
        if fences[y * w + x]:
            return False
        dx, dy = ex - x, ey - y
        nx, ny = abs(dx), abs(dy)
        sx = 1 if dx > 0 else -1
        sy = 1 if dy > 0 else -1
        ix = iy = 0
        while ix < nx or iy < ny:
            decision = (1 + 2 * ix) * ny - (1 + 2 * iy) * nx
            if decision == 0:
                if fences[y * w + (x + sx)] or fences[(y + sy) * w + x]:
                    return False
                x, y = x + sx, y + sy
                ix, iy = ix + 1, iy + 1
            elif decision < 0:
                x += sx
                ix += 1
            else:
                y += sy
                iy += 1
            if fences[y * w + x]:
                return False
        return True

    def target(self, point: Point) -> Point:
        """Keep direct approaches; otherwise aim at the first route corner."""
        cur_dist = self.distance(point)
        w = self.width
        idx = point.y * w + point.x
        cached = self._target_cache.get(idx)
        if cached is not None:
            return cached
        pen_center = self.pen.center
        if self.clear_segment(point, pen_center):
            self._target_cache[idx] = pen_center
            return pen_center
        h = self.height
        dist_grid = self._dist_grid
        pcx, pcy = pen_center.x, pen_center.y
        cx, cy = point.x, point.y
        dir_x = dir_y = 0
        has_dir = False
        while cur_dist > 0:
            target_dist = cur_dist - 1
            best_nx = cx
            best_ny = cy
            best_key = inf
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nx = cx + dx
                ny = cy + dy
                if 0 <= nx < w and 0 <= ny < h and dist_grid[ny * w + nx] == target_dist:
                    key = hypot(nx - pcx, ny - pcy)
                    if key < best_key:
                        best_key = key
                        best_nx = nx
                        best_ny = ny
            next_dx = best_nx - cx
            next_dy = best_ny - cy
            if has_dir and (next_dx != dir_x or next_dy != dir_y):
                result = Point(cx, cy)
                self._target_cache[idx] = result
                return result
            dir_x, dir_y = next_dx, next_dy
            has_dir = True
            cx, cy = best_nx, best_ny
            cur_dist = target_dist
        result = Point(cx, cy)
        self._target_cache[idx] = result
        return result


@lru_cache(maxsize=32)
def pen_navigation(pen: Pen, width: int, height: int) -> PenNavigation:
    """Reuse static routing across steps and identical episode layouts."""
    return PenNavigation(pen, width, height)
