"""Run the full two-robot mission offline and produce the README figures.

    python -m sim.run_simulation            # figures + metrics
    python -m sim.run_simulation --gif      # also render docs/exploration.gif
"""

import argparse
import math
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from maze_explorer import FOLLOWER, LEADER, MazeExplorer  # noqa: E402
from maze_explorer.explorer import plot_map  # noqa: E402
from maze_explorer.occupancy_grid import FREE, OccupancyGrid  # noqa: E402
from sim.fake_webots import Radio, SimRobot  # noqa: E402
from sim.world import MazeWorld  # noqa: E402

DOCS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")
LEADER_START = (-1.3, -0.8, math.pi / 2)
FOLLOWER_START = (1.25, 0.8, -math.pi / 2)


def draw_world(ax, world, alpha=1.0):
    for a, b, c, d in world.walls:
        ax.add_patch(Rectangle((a, b), c - a, d - b, color="#2a4d8f", alpha=alpha, lw=0))
    for a, b, c, d in world.obstacles:
        ax.add_patch(Rectangle((a, b), c - a, d - b, color="#777777", alpha=alpha, lw=0))
    for a, b, c, d in world.red_tiles:
        ax.add_patch(Rectangle((a, b), c - a, d - b, color="#c0392b", alpha=0.5 * alpha, lw=0))
    a, b, c, d = world.border_tile
    ax.add_patch(Rectangle((a, b), c - a, d - b, color="#2e86de", alpha=0.6 * alpha, lw=0))


def merged_map(*grids):
    out = OccupancyGrid((grids[0].x_min, grids[0].x_max), (grids[0].y_min, grids[0].y_max), grids[0].cell)
    for g in grids:
        take = np.abs(g.logodds) > np.abs(out.logodds)
        out.logodds[take] = g.logodds[take]
    return out


def run(max_time=1800.0, gif=False, quiet=False, seed=0):
    world, radio = MazeWorld(), Radio()
    log = (lambda m: None) if quiet else print
    robots, explorers = {}, {}
    for role, start, k in ((LEADER, LEADER_START, 1), (FOLLOWER, FOLLOWER_START, 2)):
        robots[role] = SimRobot(world, radio, *start, seed=2 * seed + k)
        explorers[role] = MazeExplorer(robots[role], role, log=log)

    t, dt = 0.0, robots[LEADER].timestep / 1000.0
    phase_end, frames, crossed = {}, [], {LEADER: False, FOLLOWER: False}
    next_frame = 0.0
    wall_clock = time.time()
    while t < max_time and not all(e.finished for e in explorers.values()):
        for role in (LEADER, FOLLOWER):
            e = explorers[role]
            e.update()
            robots[role].advance()
            if e.finished and role not in phase_end:
                phase_end[role] = t
            x = robots[role].x
            crossed[role] |= (x > 0.0) if role == LEADER else (x < 0.0)
        if gif and t >= next_frame:
            frames.append(dict(
                map=merged_map(explorers[LEADER].map, explorers[FOLLOWER].map).state.copy(),
                traj={r: list(explorers[r].trajectory) for r in robots},
                pose={r: (robots[r].x, robots[r].y, robots[r].theta) for r in robots},
                t=t))
            next_frame += 4.0
        t += dt

    metrics = evaluate(world, robots, explorers, phase_end, crossed, t)
    metrics["wall_clock_s"] = time.time() - wall_clock
    return world, robots, explorers, metrics, frames


def evaluate(world, robots, explorers, phase_end, crossed, t_end):
    final = explorers[FOLLOWER].map            # follower holds the merged map
    truth = world.free_cells(final)
    rr, cc = np.mgrid[0:final.rows, 0:final.cols]
    cx = final.x_min + (cc + 0.5) * final.cell
    mapped = final.state == FREE
    m = {"mission_time_s": t_end}
    for role, zone in ((LEADER, cx < 0), (FOLLOWER, cx > 0)):
        own = explorers[role].map.state == FREE
        tiles_true = [tl for tl in world.red_tiles if (tl[0] < 0) == (role == LEADER)]
        found = explorers[role].red_tiles if role == LEADER else [
            p for p in explorers[role].red_tiles if p[0] > 0]
        m[role] = dict(
            finished_at_s=phase_end.get(role),
            distance_m=robots[role].distance,
            collisions=robots[role].collisions,
            crossed_border=crossed[role],
            zone_coverage=float((own & truth & zone).sum() / (truth & zone).sum()),
            red_tiles=f"{len(found)}/{len(tiles_true)}",
        )
    m["final_map_coverage"] = float((mapped & truth).sum() / truth.sum())
    return m


def save_figures(world, robots, explorers, frames, gif):
    os.makedirs(DOCS, exist_ok=True)
    colours = {LEADER: "#e67e22", FOLLOWER: "#16a085"}

    # 1. Ground truth with both trajectories
    fig, ax = plt.subplots(figsize=(9, 6))
    draw_world(ax, world)
    for role, e in explorers.items():
        tr = np.array(e.trajectory)
        ax.plot(tr[:, 0], tr[:, 1], color=colours[role], lw=1.4,
                label=f"{'Robot 1 (zone 1, leader)' if role == LEADER else 'Robot 2 (zone 2, follower)'}")
        ax.plot(*tr[0], "o", color=colours[role], ms=8, mec="k")
    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.set_title("Ground-truth maze and robot trajectories (o = start)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
    fig.savefig(os.path.join(DOCS, "trajectories.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    # 2. Final merged map as presented by the follower
    f = explorers[FOLLOWER]
    fig, ax = plt.subplots(figsize=(9, 6))
    plot_map(ax, f.map, f.red_tiles, f.border)
    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("Final occupancy map built from LiDAR by both robots")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
    fig.savefig(os.path.join(DOCS, "final_map.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    if gif and frames:
        from matplotlib.animation import FuncAnimation, PillowWriter
        from matplotlib.colors import ListedColormap
        g = f.map
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        cmap = ListedColormap(["#b8b8b8", "#ffffff", "#202020"])

        def draw(i):
            fr = frames[i]
            ax.clear()
            ax.imshow(fr["map"], cmap=cmap, vmin=0, vmax=2, origin="lower",
                      extent=(g.x_min, g.x_max, g.y_min, g.y_max), interpolation="nearest")
            for role in (LEADER, FOLLOWER):
                tr = np.array(fr["traj"][role])
                if len(tr) > 1:
                    ax.plot(tr[:, 0], tr[:, 1], color=colours[role], lw=1)
                x, y, th = fr["pose"][role]
                ax.plot(x, y, "o", color=colours[role], ms=6, mec="k")
                ax.plot([x, x + 0.08 * math.cos(th)], [y, y + 0.08 * math.sin(th)], "k-", lw=1)
            ax.set_xlim(-1.55, 1.55)
            ax.set_ylim(-1.05, 1.05)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"t = {fr['t']:5.0f} s")

        anim = FuncAnimation(fig, draw, frames=len(frames))
        anim.save(os.path.join(DOCS, "exploration.gif"), writer=PillowWriter(fps=8), dpi=80)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gif", action="store_true", help="also render an animated GIF")
    parser.add_argument("--quiet", action="store_true", help="hide the robot logs")
    parser.add_argument("--seed", type=int, default=0, help="sensor-noise seed")
    args = parser.parse_args()
    world, robots, explorers, metrics, frames = run(gif=args.gif, quiet=args.quiet, seed=args.seed)
    save_figures(world, robots, explorers, frames, args.gif)
    print("\n=== Mission summary ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
