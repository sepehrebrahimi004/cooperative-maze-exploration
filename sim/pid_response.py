"""Closed-loop responses of the motion controller (docs/pid_response.png).

The robot starts at the origin facing +x and is asked to reach a waypoint
0.5 m away at 90 degrees to its left: it first turns on the spot (heading
loop), then drives forward (distance loop) while holding its heading.
"""

import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from maze_explorer.motion import PathFollower  # noqa: E402
from sim.fake_webots import Radio, SimRobot  # noqa: E402
from sim.run_simulation import DOCS  # noqa: E402
from sim.world import MazeWorld  # noqa: E402


def main():
    robot = SimRobot(MazeWorld(), Radio(), x=-1.2, y=-0.7, theta=0.0)
    dt = robot.timestep / 1000.0
    follower = PathFollower(dt)
    goal = (-1.2, -0.2)
    follower.set_path([goal])
    log = {"t": [], "heading": [], "dist": [], "wl": [], "wr": []}
    t = 0.0
    while t < 8.0:
        wl, wr = follower.update(robot.x, robot.y, robot.theta)
        robot.motors["left wheel motor"].setVelocity(wl)
        robot.motors["right wheel motor"].setVelocity(wr)
        robot.advance()
        log["t"].append(t)
        log["heading"].append(math.degrees(robot.theta))
        log["dist"].append(math.dist((robot.x, robot.y), goal))
        log["wl"].append(wl)
        log["wr"].append(wr)
        t += dt

    fig, axs = plt.subplots(3, 1, figsize=(7, 7), sharex=True)
    axs[0].plot(log["t"], log["heading"], lw=1.6)
    axs[0].axhline(90, color="k", ls="--", lw=0.8)
    axs[0].set_ylabel("heading (deg)")
    axs[0].set_title("Closed-loop response: 90° turn, then 0.5 m drive")
    axs[1].plot(log["t"], log["dist"], lw=1.6, color="C1")
    axs[1].set_ylabel("distance to goal (m)")
    axs[2].plot(log["t"], log["wl"], label="left wheel")
    axs[2].plot(log["t"], log["wr"], label="right wheel")
    axs[2].set_ylabel("wheel speed (rad/s)")
    axs[2].set_xlabel("time (s)")
    axs[2].legend()
    for ax in axs:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(DOCS, exist_ok=True)
    fig.savefig(os.path.join(DOCS, "pid_response.png"), dpi=130)
    print(f"final distance {log['dist'][-1] * 100:.1f} cm, final heading {log['heading'][-1]:.1f} deg")


if __name__ == "__main__":
    main()
