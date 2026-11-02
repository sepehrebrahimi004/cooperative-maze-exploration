"""Cooperative maze exploration: the robot-side state machine.

The same class drives both e-puck robots:

* leader   (zone 1): EXPLORE -> GO_TO_BORDER -> signal follower -> RETURN_HOME -> DONE
* follower (zone 2): WAIT -> EXPLORE -> GO_TO_BORDER -> PRESENT_MAP -> DONE

It only uses the permitted sensors (LiDAR, GPS, compass, ground sensors) plus
an emitter/receiver pair for the hand-over message. All motion is closed-loop
(see motion.PathFollower); there are no delays or open-loop manoeuvres.
"""

import json
import math

import numpy as np

from . import config as cfg
from .motion import PathFollower
from .occupancy_grid import OCCUPIED, OccupancyGrid
from .planning import (astar, bfs_distances, clearance_penalty, nearest_cell, select_target,
                       simplify_path)

LEADER, FOLLOWER = "leader", "follower"


class MazeExplorer:
    def __init__(self, robot, role, log=print):
        if role not in (LEADER, FOLLOWER):
            raise ValueError(f"unknown role {role!r}")
        self.robot, self.role, self._log = robot, role, log
        self.timestep = int(robot.getBasicTimeStep())
        self.dt = self.timestep / 1000.0
        self.time = 0.0

        self._init_devices()
        self.map = OccupancyGrid(cfg.MAP_X_LIMITS, cfg.MAP_Y_LIMITS, cfg.CELL_SIZE)
        self.follower = PathFollower(self.dt)

        self.state = "EXPLORE" if role == LEADER else "WAIT"
        self.finished = False
        self.home = None              # start position (world)
        self.border = None            # border tile position (world)
        self.red_tiles = []           # detected red tiles (world)
        self.trajectory = []

        self.goal_cell = None
        self.next_replan = 0.0
        self.next_map_update = 0.0
        self.excluded = np.zeros((self.map.rows, self.map.cols), dtype=bool)
        self.swept = np.zeros((self.map.rows, self.map.cols), dtype=bool)
        r = int(np.ceil(cfg.SWEEP_RADIUS / self.map.cell))
        self._sweep_offsets = [(dr, dc) for dr in range(-r, r + 1) for dc in range(-r, r + 1)
                               if dr * dr + dc * dc <= r * r]
        self._progress_ref = None

    # ---------------------------------------------------------------- devices
    def _init_devices(self):
        ts, r = self.timestep, self.robot
        self.lidar = r.getDevice(cfg.LIDAR_NAME)
        self.lidar.enable(ts)
        self.gps = r.getDevice("gps")
        self.gps.enable(ts)
        self.compass = r.getDevice("compass")
        self.compass.enable(ts)
        self.ground = [r.getDevice(f"gs{i}") for i in range(3)]
        for s in self.ground:
            s.enable(ts)
        self.leds = [r.getDevice(f"led{i}") for i in range(8)]
        self.left_motor = r.getDevice("left wheel motor")
        self.right_motor = r.getDevice("right wheel motor")
        for m in (self.left_motor, self.right_motor):
            m.setPosition(float("inf"))
            m.setVelocity(0.0)
        if self.role == LEADER:
            self.emitter = r.getDevice("emitter")
            self.emitter.setChannel(cfg.COMM_CHANNEL)
        else:
            self.receiver = r.getDevice("receiver")
            self.receiver.enable(ts)
            self.receiver.setChannel(cfg.COMM_CHANNEL)

    def log(self, msg):
        self._log(f"[{self.role:8s} t={self.time:6.1f}s] {msg}")

    def pose(self):
        x, y = self.gps.getValues()[:2]
        c = self.compass.getValues()
        return x, y, math.atan2(c[0], c[1])   # heading of the robot's x-axis

    # -------------------------------------------------------------- main step
    def update(self):
        """Run one control period. Call once per ``robot.step()``."""
        self.time += self.dt
        x, y, th = self.pose()
        if self.home is None:
            self.home = (x, y)
        if not self.trajectory or math.dist(self.trajectory[-1], (x, y)) > 0.01:
            self.trajectory.append((x, y))

        ranges = self.lidar.getRangeImage()
        if self.time >= self.next_map_update:
            self.map.integrate_scan(x, y, th, ranges, self.lidar.getFov(),
                                    self.lidar.getMaxRange(), cfg.LIDAR_RAY_STRIDE)
            self.next_map_update = self.time + cfg.MAP_UPDATE_PERIOD

        self._check_floor(x, y, th)
        self._mark_swept(x + 0.03 * math.cos(th), y + 0.03 * math.sin(th))
        if self.role == FOLLOWER:
            self._check_radio()

        left, right = getattr(self, "_state_" + self.state.lower())(x, y, th)
        left, right = self._safety_filter(left, right, ranges)
        self.left_motor.setVelocity(left)
        self.right_motor.setVelocity(right)

    # ------------------------------------------------------------ perception
    def _check_floor(self, x, y, th):
        values = [s.getValue() for s in self.ground]
        on_red = sum(v > cfg.RED_THRESHOLD for v in values) >= 2
        on_border = sum(v < cfg.BORDER_THRESHOLD for v in values) >= 2

        for led in self.leds:
            led.set(1 if on_red else 0)
        if on_red:
            p = (x + 0.03 * math.cos(th), y + 0.03 * math.sin(th))
            if all(math.dist(p, t) > cfg.RED_TILE_MIN_SEPARATION for t in self.red_tiles):
                self.red_tiles.append(p)
                self.log(f"RED TILE #{len(self.red_tiles)} found at ({p[0]:+.2f}, {p[1]:+.2f}) - LEDs on")

        if on_border:
            # The ground sensors sit at the front: the tile centre is ahead.
            c = (x + 0.10 * math.cos(th), y + 0.10 * math.sin(th))
            if not self.map.forbidden[self.map.world_to_grid(*c)]:
                if self.border is None:
                    self.border = c
                    self.log(f"border tile detected at ({c[0]:+.2f}, {c[1]:+.2f}); marked as no-go area")
                self.map.mark_forbidden(*c, cfg.BORDER_FORBIDDEN_RADIUS)
                self.follower.set_path([])
                self.next_replan = 0.0

    def _mark_swept(self, x, y):
        r0, c0 = self.map.world_to_grid(x, y)
        for dr, dc in self._sweep_offsets:
            r, c = r0 + dr, c0 + dc
            if 0 <= r < self.map.rows and 0 <= c < self.map.cols:
                self.swept[r, c] = True

    def _check_radio(self):
        while self.receiver.getQueueLength() > 0:
            data = (self.receiver.getBytes() if hasattr(self.receiver, "getBytes")
                    else self.receiver.getData())
            self.receiver.nextPacket()
            header, _, payload = bytes(data).partition(b"\n")
            msg = json.loads(header.decode("utf-8"))
            if msg.get("cmd") != "start":
                continue
            self.map.merge_bytes(payload)
            if msg.get("border"):
                self.border = tuple(msg["border"])
                self.map.mark_forbidden(*self.border, cfg.BORDER_FORBIDDEN_RADIUS)
            self.red_tiles += [tuple(t) for t in msg.get("red_tiles", [])]
            self.log("start message and zone-1 map received from leader")
            if self.state == "WAIT":
                self.state = "EXPLORE"

    def _safety_filter(self, left, right, ranges):
        """Cancel forward motion if the LiDAR sees something right in front."""
        if left + right <= 0:
            return left, right
        n, fov = len(ranges), self.lidar.getFov()
        angles = fov / 2.0 - np.arange(n) * fov / (n - 1)
        front = np.asarray(ranges)[np.abs(angles) < 0.5]
        if np.min(front) < cfg.EMERGENCY_STOP_DISTANCE:
            mean = (left + right) / 2.0
            self.follower.set_path([])
            self.next_replan = 0.0
            return left - mean, right - mean
        return left, right

    # -------------------------------------------------------------- planning
    def _planning_start(self, x, y, trav):
        """Grid cell to plan from. If the robot stands inside an inflated or
        forbidden margin, this is the nearest safe cell (None if there is none)."""
        start = self.map.world_to_grid(x, y)
        if trav[start]:
            return start
        return nearest_cell(trav, self.map.state != OCCUPIED, start)

    def _plan_to(self, x, y, goal):
        """A* from the robot to ``goal`` (grid cell); loads the waypoints."""
        trav = self.map.traversable(cfg.INFLATION_RADIUS)
        start = self._planning_start(x, y, trav)
        if start is None:
            return False
        prefix = [] if start == self.map.world_to_grid(x, y) else [start]
        width = int(round(cfg.CLEARANCE_WIDTH / self.map.cell))
        penalty = clearance_penalty(self.map.blocked(cfg.INFLATION_RADIUS), trav, width,
                                    cfg.CLEARANCE_WEIGHT)
        path = astar(trav, start, goal, penalty)
        if path is None:
            return False
        # Shortcuts may only cross cells with full clearance
        cells = prefix + simplify_path(path, trav & (penalty == 0))[1:]
        self.follower.set_path([self.map.grid_to_world(*c) for c in cells] or
                               [self.map.grid_to_world(*goal)])
        self._progress_ref = (x, y, self.time)
        return True

    def _exclude(self, cell):
        r = int(np.ceil(cfg.BLACKLIST_RADIUS / self.map.cell))
        r0, c0 = cell
        self.excluded[max(0, r0 - r):r0 + r + 1, max(0, c0 - r):c0 + r + 1] = True

    def _stuck(self, x, y):
        if self._progress_ref is None or self.follower.done:
            return False
        px, py, t0 = self._progress_ref
        if math.dist((x, y), (px, py)) > 0.05:
            self._progress_ref = (x, y, self.time)
            return False
        return self.time - t0 > cfg.STUCK_TIMEOUT

    def _navigate(self, x, y, th, point):
        """Drive to the reachable cell closest to a world ``point`` with
        periodic replanning. Returns (left, right, arrived)."""
        if self.goal_cell is not None and self.follower.done and self.next_replan > 0:
            return 0.0, 0.0, True
        if self.goal_cell is None or self.time >= self.next_replan or self._stuck(x, y):
            self.next_replan = self.time + cfg.REPLAN_PERIOD
            for _ in range(2):                 # second try: re-pick the goal on the new map
                if self.goal_cell is not None and self._plan_to(x, y, self.goal_cell):
                    break
                self.goal_cell = self._closest_reachable(x, y, point)
            else:
                self.log("no path to goal")
                return 0.0, 0.0, True
        left, right = self.follower.update(x, y, th)
        return left, right, False

    # ---------------------------------------------------------------- states
    def _state_wait(self, x, y, th):
        return 0.0, 0.0

    def _state_explore(self, x, y, th):
        reached = self.follower.done and self.goal_cell is not None
        if reached or self.time >= self.next_replan or self._stuck(x, y):
            if self.goal_cell is not None and (reached or self._stuck(x, y)):
                self._exclude(self.goal_cell)          # visited or unreachable
            trav = self.map.traversable(cfg.INFLATION_RADIUS)
            start = self._planning_start(x, y, trav) or self.map.world_to_grid(x, y)
            target = select_target(self.map.state, trav, start,
                                   int(round(cfg.FRONTIER_APPROACH_RADIUS / self.map.cell)),
                                   cfg.MIN_FRONTIER_SIZE, self.excluded, ~self.swept)
            self.next_replan = self.time + cfg.REPLAN_PERIOD
            if target is None:
                if self.follower.done and self.time > 2.0:   # allow a few scans first
                    self.log(f"exploration complete ({len(self.red_tiles)} red tiles found)")
                    self.follower.set_path([])
                    self.state, self.goal_cell = "GO_TO_BORDER", None
                    self.next_replan = 0.0
                    return 0.0, 0.0
            elif target != self.goal_cell or self.follower.done:
                self.goal_cell = target
                if not self._plan_to(x, y, target):
                    self._exclude(target)
        return self.follower.update(x, y, th)

    def _closest_reachable(self, x, y, point):
        """Reachable cell closest to a world point (on our side of the border)."""
        trav = self.map.traversable(cfg.INFLATION_RADIUS)
        start = self._planning_start(x, y, trav) or self.map.world_to_grid(x, y)
        dist = bfs_distances(trav, start)
        rr, cc = np.nonzero(dist >= 0)
        wx = self.map.x_min + (cc + 0.5) * self.map.cell
        wy = self.map.y_min + (rr + 0.5) * self.map.cell
        k = np.argmin((wx - point[0]) ** 2 + (wy - point[1]) ** 2)
        return int(rr[k]), int(cc[k])

    def _state_go_to_border(self, x, y, th):
        if self.border is None:
            self.log("warning: border tile was never detected; stopping here")
            return self._at_border()
        left, right, arrived = self._navigate(x, y, th, self.border)
        return self._at_border() if arrived else (left, right)

    def _at_border(self):
        self.follower.set_path([])
        self.goal_cell, self.next_replan = None, 0.0
        if self.role == LEADER:
            header = json.dumps({"cmd": "start", "border": self.border,
                                 "red_tiles": self.red_tiles}).encode("utf-8")
            self.emitter.send(header + b"\n" + self.map.to_bytes())
            self.log("at the border: start signal and map sent to follower")
            self.state = "RETURN_HOME"
        else:
            self.log("back at the border - presenting the complete map")
            self.state = "PRESENT_MAP"
        return 0.0, 0.0

    def _state_return_home(self, x, y, th):
        left, right, arrived = self._navigate(x, y, th, self.home)
        if arrived:
            self.log(f"returned to start position ({self.home[0]:+.2f}, {self.home[1]:+.2f})")
            self.state, self.finished = "DONE", True
            return 0.0, 0.0
        return left, right

    def _state_present_map(self, x, y, th):
        self.present_map()
        self.state, self.finished = "DONE", True
        return 0.0, 0.0

    def _state_done(self, x, y, th):
        return 0.0, 0.0

    # ----------------------------------------------------------- presentation
    def present_map(self, path="final_map.png"):
        s = self.map.state
        known = np.count_nonzero(s != 0) * self.map.cell ** 2
        self.log(f"complete map: {known:.2f} m^2 observed, {len(self.red_tiles)} red tiles")
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return
        fig, ax = plt.subplots(figsize=(8, 5.5))
        plot_map(ax, self.map, self.red_tiles, self.border)
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)


def plot_map(ax, grid, red_tiles=(), border=None):
    """Draw an occupancy grid (white free, black occupied, grey unknown)."""
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#b8b8b8", "#ffffff", "#202020"])
    extent = (grid.x_min, grid.x_max, grid.y_min, grid.y_max)
    ax.imshow(grid.state, cmap=cmap, vmin=0, vmax=2, origin="lower", extent=extent,
              interpolation="nearest")
    for i, (tx, ty) in enumerate(red_tiles):
        ax.plot(tx, ty, "s", ms=9, mfc="#e03030", mec="k", label="red tile" if i == 0 else None)
    if border is not None:
        ax.plot(*border, "s", ms=11, mfc="#3070e0", mec="k", label="border tile")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
