"""Unit tests for the hardware-independent algorithms.

    python -m unittest discover -s tests -v
"""

import math
import unittest

import numpy as np

from maze_explorer.occupancy_grid import FREE, OCCUPIED, UNKNOWN, OccupancyGrid, dilate
from maze_explorer.pid import PID
from maze_explorer.planning import astar, clearance_penalty, select_target, simplify_path


class TestPID(unittest.TestCase):
    def test_proportional_only(self):
        pid = PID(kp=2.0, ts=0.1)
        self.assertAlmostEqual(pid.update(1.5), 3.0)

    def test_trapezoidal_integral(self):
        pid = PID(kp=0.0, ki=1.0, ts=0.1)
        for _ in range(10):
            u = pid.update(1.0)
        self.assertAlmostEqual(u, 1.0)                 # integral of 1 over 1 s

    def test_closed_loop_zero_steady_state_error(self):
        pid, x, ts = PID(kp=2.0, ki=1.0, ts=0.01), 0.0, 0.01
        for _ in range(3000):                          # plant: dx/dt = u - 0.5 (disturbance)
            x += ts * (pid.update(1.0 - x) - 0.5)
        self.assertAlmostEqual(x, 1.0, places=3)

    def test_anti_windup(self):
        pid = PID(kp=1.0, ki=10.0, ts=0.01, output_limits=(-1.0, 1.0))
        for _ in range(1000):
            pid.update(5.0)                            # saturated for a long time
        self.assertLess(pid.integral, 1.0)
        self.assertLess(pid.update(-0.5), 0.5)         # recovers without a long overshoot


class TestOccupancyGrid(unittest.TestCase):
    def setUp(self):
        self.g = OccupancyGrid((-1.0, 1.0), (-1.0, 1.0), 0.05)

    def test_coordinate_round_trip(self):
        for x, y in [(0.0, 0.0), (-0.97, 0.42), (0.51, -0.88)]:
            wx, wy = self.g.grid_to_world(*self.g.world_to_grid(x, y))
            self.assertLessEqual(abs(wx - x), self.g.cell / 2 + 1e-9)
            self.assertLessEqual(abs(wy - y), self.g.cell / 2 + 1e-9)

    def test_scan_marks_free_and_occupied(self):
        # Robot at origin facing +x, every ray hits a wall 0.5 m away
        n, fov = 181, math.pi
        for _ in range(3):
            self.g.integrate_scan(0.0, 0.0, 0.0, [0.5] * n, fov, 4.0)
        s = self.g.state
        self.assertEqual(s[self.g.world_to_grid(0.25, 0.0)], FREE)
        self.assertEqual(s[self.g.world_to_grid(0.51, 0.0)], OCCUPIED)
        self.assertEqual(s[self.g.world_to_grid(0.0, 0.51)], OCCUPIED)   # leftmost ray (+y)
        self.assertEqual(s[self.g.world_to_grid(-0.4, 0.0)], UNKNOWN)    # behind the robot

    def test_map_serialisation_round_trip(self):
        self.g.integrate_scan(0.0, 0.0, 0.0, [0.5] * 91, math.pi, 4.0)
        self.g.mark_forbidden(0.5, 0.5, 0.1)
        other = OccupancyGrid((-1.0, 1.0), (-1.0, 1.0), 0.05)
        other.merge_bytes(self.g.to_bytes())
        np.testing.assert_array_equal(other.state, self.g.state)
        np.testing.assert_array_equal(other.forbidden, self.g.forbidden)

    def test_dilate(self):
        m = np.zeros((9, 9), dtype=bool)
        m[4, 4] = True
        d = dilate(m, 2)
        self.assertTrue(d[4, 2] and d[2, 4] and d[5, 5])
        self.assertFalse(d[2, 2])                       # outside the disc


class TestPlanning(unittest.TestCase):
    def test_astar_straight_line(self):
        grid = np.ones((10, 10), dtype=bool)
        path = astar(grid, (0, 0), (0, 9))
        self.assertEqual(len(path), 10)
        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[-1], (0, 9))

    def test_astar_through_gap(self):
        grid = np.ones((11, 11), dtype=bool)
        grid[:, 5] = False
        grid[8, 5] = True                               # single opening
        path = astar(grid, (0, 0), (0, 10))
        self.assertIn((8, 5), path)
        self.assertTrue(all(grid[c] for c in path))

    def test_astar_unreachable(self):
        grid = np.ones((5, 5), dtype=bool)
        grid[:, 2] = False
        self.assertIsNone(astar(grid, (0, 0), (0, 4)))

    def test_astar_no_corner_cutting(self):
        grid = np.ones((3, 3), dtype=bool)
        grid[0, 1] = grid[1, 0] = False
        self.assertIsNone(astar(grid, (0, 0), (1, 1)))

    def test_astar_optimal_cost(self):
        grid = np.ones((20, 20), dtype=bool)
        path = astar(grid, (0, 0), (5, 12))
        cost = sum(math.dist(path[i], path[i + 1]) for i in range(len(path) - 1))
        self.assertAlmostEqual(cost, 7 + 5 * math.sqrt(2))   # octile distance

    def test_clearance_penalty_centres_path(self):
        grid = np.ones((9, 30), dtype=bool)
        grid[0, :] = grid[-1, :] = False                # corridor walls
        pen = clearance_penalty(~grid, grid, 3, 5.0)
        path = astar(grid, (1, 0), (1, 29), pen)
        self.assertIn(4, {r for r, _ in path})          # moves to the centre line

    def test_simplify_path(self):
        grid = np.ones((10, 10), dtype=bool)
        path = astar(grid, (0, 0), (9, 9))
        self.assertEqual(simplify_path(path, grid), [(0, 0), (9, 9)])

    def test_select_target_picks_nearest_frontier(self):
        state = np.full((20, 20), UNKNOWN, dtype=np.uint8)
        state[5:15, 2:18] = FREE                        # known free rectangle
        trav = state == FREE
        target = select_target(state, trav, (10, 4), 0, 3)
        self.assertEqual(target[1], 2)                  # left edge is closest


if __name__ == "__main__":
    unittest.main()
