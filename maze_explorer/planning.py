"""Grid search: A* path planning, BFS distance maps and frontier selection."""

import heapq
import math
from collections import deque

import numpy as np

from .occupancy_grid import FREE, UNKNOWN, dilate

SQRT2 = math.sqrt(2.0)
NEIGHBOURS_8 = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                (-1, -1, SQRT2), (-1, 1, SQRT2), (1, -1, SQRT2), (1, 1, SQRT2)]


def astar(traversable, start, goal, penalty=None):
    """A* on an 8-connected grid with the octile-distance heuristic.

    Diagonal moves may not cut obstacle corners. ``penalty`` (optional, same
    shape, >= 0) adds a per-cell cost, e.g. to keep paths away from walls; the
    heuristic stays admissible because penalties are non-negative. The start
    cell is always allowed (the robot may be standing inside an inflated
    area). Returns the list of cells from start to goal, or None.
    """
    rows, cols = traversable.shape
    if not traversable[goal]:
        return None

    def h(cell):
        dr, dc = abs(cell[0] - goal[0]), abs(cell[1] - goal[1])
        return (dr + dc) + (SQRT2 - 2.0) * min(dr, dc)

    g = np.full((rows, cols), np.inf)
    parent = {}
    closed = np.zeros((rows, cols), dtype=bool)
    g[start] = 0.0
    open_heap = [(h(start), 0.0, start)]

    while open_heap:
        _, g_cur, cur = heapq.heappop(open_heap)
        if closed[cur]:
            continue
        if cur == goal:
            path = [cur]
            while cur in parent:
                cur = parent[cur]
                path.append(cur)
            return path[::-1]
        closed[cur] = True
        r, c = cur
        for dr, dc, cost in NEIGHBOURS_8:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols) or not traversable[nr, nc]:
                continue
            if dr and dc and not (traversable[r + dr, c] and traversable[r, c + dc]):
                continue                                   # no corner cutting
            ng = g_cur + cost * (1.0 + (penalty[nr, nc] if penalty is not None else 0.0))
            if ng < g[nr, nc]:
                g[nr, nc] = ng
                parent[(nr, nc)] = cur
                heapq.heappush(open_heap, (ng + h((nr, nc)), ng, (nr, nc)))
    return None


def line_of_sight(traversable, a, b):
    """True if the straight segment between two cells stays traversable."""
    n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1])) * 2) + 1
    rows = np.rint(np.linspace(a[0], b[0], n)).astype(int)
    cols = np.rint(np.linspace(a[1], b[1], n)).astype(int)
    return bool(traversable[rows, cols].all())


def simplify_path(path, traversable):
    """Remove intermediate cells that are in direct line of sight."""
    if not path or len(path) < 3:
        return path
    out, i = [path[0]], 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not line_of_sight(traversable, path[i], path[j]):
            j -= 1
        out.append(path[j])
        i = j
    return out


def clearance_penalty(blocked, traversable, width_cells, weight):
    """Cost that decays linearly over ``width_cells`` from the blocked area,
    so that A* prefers the middle of corridors."""
    penalty = np.zeros(blocked.shape)
    ring = blocked.copy()
    for k in range(width_cells, 0, -1):
        grown = dilate(ring, 1)
        penalty[grown & ~ring & traversable] = weight * k / width_cells
        ring = grown
    return penalty


def bfs_distances(passable, start):
    """Breadth-first (4-connected) step distance from ``start``; -1 = unreached."""
    rows, cols = passable.shape
    dist = np.full((rows, cols), -1, dtype=int)
    dist[start] = 0
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and passable[nr, nc] and dist[nr, nc] < 0:
                dist[nr, nc] = dist[r, c] + 1
                queue.append((nr, nc))
    return dist


def nearest_cell(mask_target, passable, start):
    """Nearest cell (BFS order through ``passable``) where mask_target is True."""
    rows, cols = passable.shape
    seen = np.zeros((rows, cols), dtype=bool)
    seen[start] = True
    queue = deque([start])
    while queue:
        cell = queue.popleft()
        if mask_target[cell]:
            return cell
        r, c = cell
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and passable[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                queue.append((nr, nc))
    return None


def frontier_mask(state):
    """Free cells with at least one unknown 4-neighbour."""
    unknown = np.pad(state == UNKNOWN, 1, constant_values=False)
    near_unknown = (unknown[:-2, 1:-1] | unknown[2:, 1:-1] |
                    unknown[1:-1, :-2] | unknown[1:-1, 2:])
    return (state == FREE) & near_unknown


def label_clusters(mask):
    """8-connected components of a boolean mask -> list of cell lists."""
    seen = np.zeros(mask.shape, dtype=bool)
    clusters = []
    for start in zip(*np.nonzero(mask)):
        if seen[start]:
            continue
        seen[start] = True
        cluster, queue = [], deque([start])
        while queue:
            r, c = queue.popleft()
            cluster.append((r, c))
            for dr, dc, _ in NEIGHBOURS_8:
                nr, nc = r + dr, c + dc
                if (0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1]
                        and mask[nr, nc] and not seen[nr, nc]):
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        clusters.append(cluster)
    return clusters


def select_target(state, traversable, robot_cell, approach_cells, min_size,
                  excluded=None, unswept=None):
    """Exploration target selection (frontier + floor coverage).

    Candidate goals are
    * cells within ``approach_cells`` of a frontier cluster (LiDAR mapping:
      frontier cells often lie inside the inflated wall margin, so the robot
      only needs to get close), and
    * free cells the ground sensors have not swept yet (``unswept``), which
      is what finds red tiles on the floor.
    Clusters smaller than ``min_size`` are treated as noise. The candidate
    with the shortest BFS path from the robot wins.
    """
    goals = np.zeros(state.shape, dtype=bool)
    for cluster in label_clusters(frontier_mask(state)):
        if len(cluster) >= min_size:
            goals[tuple(np.array(cluster).T)] = True
    goals = dilate(goals, approach_cells)
    if unswept is not None:
        for cluster in label_clusters(unswept & traversable):
            if len(cluster) >= min_size:
                goals[tuple(np.array(cluster).T)] = True

    dist = bfs_distances(traversable | _single(robot_cell, state.shape), robot_cell)
    goals &= traversable & (dist > 0)
    if excluded is not None:
        goals &= ~excluded
    if not goals.any():
        return None
    cand = np.where(goals, dist, np.iinfo(int).max)
    cell = np.unravel_index(np.argmin(cand), cand.shape)
    return int(cell[0]), int(cell[1])


def _single(cell, shape):
    m = np.zeros(shape, dtype=bool)
    m[cell] = True
    return m
