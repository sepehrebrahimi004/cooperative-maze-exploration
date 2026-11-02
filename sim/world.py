"""A 2-D maze world for offline testing: walls, obstacles and floor tiles.

The layout mimics the course world: two zones (x < 0 and x > 0) joined by a
single opening that is covered by a blue border tile, red tiles scattered in
both zones and box obstacles in zone 2. Dimensions are in metres.
"""

import numpy as np

WALL = 0.02  # wall thickness


def hwall(y, x0, x1):
    return (x0, y - WALL / 2, x1, y + WALL / 2)


def vwall(x, y0, y1):
    return (x - WALL / 2, y0, x + WALL / 2, y1)


def box(cx, cy, size):
    return (cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2)


def tile(cx, cy, size=0.15):
    return box(cx, cy, size)


class MazeWorld:
    """Axis-aligned rectangles for walls/obstacles and coloured floor tiles."""

    FLOOR_WHITE, FLOOR_RED, FLOOR_BLUE = 750.0, 960.0, 320.0   # ground-sensor readings

    def __init__(self):
        X, Y = 1.5, 1.0
        self.bounds = (-X, X, -Y, Y)
        self.walls = [
            # outer walls
            hwall(-Y, -X, X), hwall(Y, -X, X), vwall(-X, -Y, Y), vwall(X, -Y, Y),
            # divider with one opening (the border) at y in [-0.15, 0.15]
            vwall(0.0, -Y, -0.15), vwall(0.0, 0.15, Y),
            # zone 1
            hwall(0.4, -X, -0.75), vwall(-0.9, -Y, -0.3), hwall(-0.3, -0.6, 0.0),
            vwall(-0.4, 0.4, Y), vwall(-1.2, 0.4, 0.75),
            # zone 2
            vwall(0.6, -0.5, Y), hwall(-0.5, 0.6, 1.2), vwall(1.0, 0.1, 0.6),
        ]
        self.obstacles = [box(0.3, 0.55, 0.10), box(0.95, -0.75, 0.10),
                          box(1.25, 0.35, 0.10), box(0.3, -0.35, 0.08)]
        self.red_tiles = [tile(-1.3, 0.8), tile(-0.45, -0.7), tile(-0.2, 0.75),
                          tile(1.35, -0.2), tile(0.3, -0.85), tile(0.8, 0.8)]
        self.border_tile = (-0.12, -0.15, 0.12, 0.15)

        rects = np.array(self.walls + self.obstacles, dtype=float)
        self.rects = rects
        # Each rectangle contributes its four edges as segments P -> Q
        x0, y0, x1, y1 = rects.T
        self.seg_p = np.concatenate([np.c_[x0, y0], np.c_[x1, y0], np.c_[x1, y1], np.c_[x0, y1]])
        self.seg_q = np.concatenate([np.c_[x1, y0], np.c_[x1, y1], np.c_[x0, y1], np.c_[x0, y0]])

    # ------------------------------------------------------------- geometry
    def raycast(self, ox, oy, angles, max_range):
        """Distance to the first wall along each ray (inf beyond max_range)."""
        d = np.c_[np.cos(angles), np.sin(angles)]                # (n, 2)
        e = self.seg_q - self.seg_p                               # (m, 2)
        w = self.seg_p - np.array([ox, oy])                       # (m, 2)
        denom = d[:, None, 0] * e[None, :, 1] - d[:, None, 1] * e[None, :, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (w[None, :, 0] * e[None, :, 1] - w[None, :, 1] * e[None, :, 0]) / denom
            u = (w[None, :, 0] * d[:, None, 1] - w[None, :, 1] * d[:, None, 0]) / denom
        valid = (np.abs(denom) > 1e-12) & (t > 0) & (u >= 0) & (u <= 1)
        t = np.where(valid, t, np.inf).min(axis=1)
        return np.where(t <= max_range, t, np.inf)

    def collides(self, x, y, radius):
        """True if a disc of ``radius`` at (x, y) overlaps any wall/obstacle."""
        r = self.rects
        dx = np.maximum(np.maximum(r[:, 0] - x, 0), x - r[:, 2])
        dy = np.maximum(np.maximum(r[:, 1] - y, 0), y - r[:, 3])
        return bool(np.any(dx * dx + dy * dy < radius * radius))

    def floor_value(self, x, y):
        def inside(rect):
            return rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]
        if inside(self.border_tile):
            return self.FLOOR_BLUE
        if any(inside(t) for t in self.red_tiles):
            return self.FLOOR_RED
        return self.FLOOR_WHITE

    def free_cells(self, grid):
        """Ground-truth free cells of an OccupancyGrid (centre not in a wall)."""
        rr, cc = np.mgrid[0:grid.rows, 0:grid.cols]
        cx = grid.x_min + (cc + 0.5) * grid.cell
        cy = grid.y_min + (rr + 0.5) * grid.cell
        x0, x1, y0, y1 = self.bounds
        free = (cx > x0) & (cx < x1) & (cy > y0) & (cy < y1)
        for a, b, c, d in self.rects:
            free &= ~((cx >= a) & (cx <= c) & (cy >= b) & (cy <= d))
        return free
