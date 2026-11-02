"""Log-odds occupancy grid built from 2-D LiDAR scans."""

import numpy as np

UNKNOWN, FREE, OCCUPIED = 0, 1, 2


class OccupancyGrid:
    """Occupancy grid in world coordinates.

    Cell (row, col) covers x in [x_min + col*cell, x_min + (col+1)*cell) and
    y in [y_min + row*cell, ...), so row 0 is the bottom of the map (plot with
    ``origin='lower'``).
    """

    L_FREE, L_OCC = -0.4, 0.9            # log-odds increments per observation
    L_MIN, L_MAX = -4.0, 4.0             # clamping keeps the map responsive
    T_FREE, T_OCC = -0.3, 0.8            # classification thresholds

    def __init__(self, x_limits, y_limits, cell_size):
        self.x_min, self.x_max = x_limits
        self.y_min, self.y_max = y_limits
        self.cell = cell_size
        self.rows = int(round((self.y_max - self.y_min) / cell_size))
        self.cols = int(round((self.x_max - self.x_min) / cell_size))
        self.logodds = np.zeros((self.rows, self.cols))
        self.forbidden = np.zeros((self.rows, self.cols), dtype=bool)

    # ------------------------------------------------------------ coordinates
    def world_to_grid(self, x, y):
        col = int(np.floor((x - self.x_min) / self.cell))
        row = int(np.floor((y - self.y_min) / self.cell))
        return (min(max(row, 0), self.rows - 1), min(max(col, 0), self.cols - 1))

    def grid_to_world(self, row, col):
        return (self.x_min + (col + 0.5) * self.cell,
                self.y_min + (row + 0.5) * self.cell)

    # ------------------------------------------------------------------ state
    @property
    def state(self):
        s = np.full(self.logodds.shape, UNKNOWN, dtype=np.uint8)
        s[self.logodds < self.T_FREE] = FREE
        s[self.logodds > self.T_OCC] = OCCUPIED
        return s

    def blocked(self, inflation_radius):
        """Cells the robot centre may not occupy: obstacles and forbidden
        areas grown by ``inflation_radius``."""
        return dilate(self.forbidden | (self.state == OCCUPIED),
                      int(np.ceil(inflation_radius / self.cell)))

    def traversable(self, inflation_radius):
        return (self.state == FREE) & ~self.blocked(inflation_radius)

    # --------------------------------------------------------------- updating
    def integrate_scan(self, x, y, theta, ranges, fov, max_range, stride=1):
        """Fuse one LiDAR scan taken at pose (x, y, theta).

        Webots orders the range image from left to right, so ray i points at
        ``theta + fov/2 - i*fov/(n-1)`` in the world frame.
        """
        ranges = np.asarray(ranges, dtype=float)
        n = len(ranges)
        idx = np.arange(0, n, stride)
        r = ranges[idx]
        angles = theta + fov / 2.0 - idx * fov / (n - 1)

        finite = np.isfinite(r)
        hit = finite & (r < 0.98 * max_range)
        dist = np.where(finite, np.minimum(r, max_range), max_range)

        # Cells traversed by each ray are free
        step = self.cell / 2.0
        t = np.arange(0.0, dist.max() + step, step)
        mask = t[None, :] < (dist[:, None] - self.cell / 2.0)
        px = x + np.outer(np.cos(angles), t)
        py = y + np.outer(np.sin(angles), t)
        free = self._flat_indices(px[mask], py[mask])

        # Ray end points are occupied
        hx = x + dist[hit] * np.cos(angles[hit])
        hy = y + dist[hit] * np.sin(angles[hit])
        occ = self._flat_indices(hx, hy)
        free = np.setdiff1d(free, occ, assume_unique=True)

        flat = self.logodds.ravel()
        flat[free] += self.L_FREE
        flat[occ] += self.L_OCC
        np.clip(self.logodds, self.L_MIN, self.L_MAX, out=self.logodds)

    def _flat_indices(self, xs, ys):
        cols = np.floor((xs - self.x_min) / self.cell).astype(int)
        rows = np.floor((ys - self.y_min) / self.cell).astype(int)
        ok = (rows >= 0) & (rows < self.rows) & (cols >= 0) & (cols < self.cols)
        return np.unique(rows[ok] * self.cols + cols[ok])

    def mark_forbidden(self, x, y, radius):
        """Mark a disc as a no-go area (used for the zone border)."""
        rr, cc = np.mgrid[0:self.rows, 0:self.cols]
        cx = self.x_min + (cc + 0.5) * self.cell
        cy = self.y_min + (rr + 0.5) * self.cell
        self.forbidden |= (cx - x) ** 2 + (cy - y) ** 2 <= radius ** 2

    # ------------------------------------------------------------- messaging
    def to_bytes(self):
        """Compact serialisation for the radio link (int8 log-odds)."""
        q = np.clip(np.round(self.logodds * 30), -127, 127).astype(np.int8)
        return q.tobytes() + np.packbits(self.forbidden).tobytes()

    def merge_bytes(self, data):
        """Merge a map received from another robot: keep, for each cell, the
        estimate with the higher confidence."""
        n = self.rows * self.cols
        other = np.frombuffer(data[:n], dtype=np.int8).reshape(self.rows, self.cols) / 30.0
        take = np.abs(other) > np.abs(self.logodds)
        self.logodds[take] = other[take]
        forb = np.unpackbits(np.frombuffer(data[n:], dtype=np.uint8))[:n]
        self.forbidden |= forb.reshape(self.rows, self.cols).astype(bool)


def dilate(mask, radius_cells):
    """Binary dilation with a disc structuring element (pure NumPy)."""
    if radius_cells <= 0:
        return mask.copy()
    out = mask.copy()
    rows, cols = mask.shape
    r = radius_cells
    for dr in range(-r, r + 1):
        for dc in range(-r, r + 1):
            if dr == 0 and dc == 0 or dr * dr + dc * dc > r * r:
                continue
            src = mask[max(0, -dr):rows - max(0, dr), max(0, -dc):cols - max(0, dc)]
            out[max(0, dr):rows - max(0, -dr), max(0, dc):cols - max(0, -dc)] |= src
    return out
