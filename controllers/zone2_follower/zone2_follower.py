"""Webots controller for the follower robot (see maze_explorer/explorer.py)."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from controller import Robot  # noqa: E402  (provided by Webots)

from maze_explorer import FOLLOWER, MazeExplorer  # noqa: E402

robot = Robot()
explorer = MazeExplorer(robot, FOLLOWER)
while robot.step(explorer.timestep) != -1:
    explorer.update()
