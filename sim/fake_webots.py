"""Minimal stand-ins for the Webots devices used by the controllers.

Only the subset of the Webots Python API that ``maze_explorer`` uses is
implemented, with the same conventions (ENU world, compass returning the
north vector in the robot frame, LiDAR range image ordered left to right).
"""

import math

import numpy as np

from maze_explorer import config as cfg


class _Device:
    def enable(self, timestep):
        pass


class Lidar(_Device):
    def __init__(self, body, resolution=667, fov=math.radians(240), max_range=4.0, noise=0.003):
        self.body, self.n, self.fov, self.max_range, self.noise = body, resolution, fov, max_range, noise

    def getFov(self):
        return self.fov

    def getMaxRange(self):
        return self.max_range

    def getHorizontalResolution(self):
        return self.n

    def getRangeImage(self):
        b = self.body
        angles = b.theta + self.fov / 2 - np.arange(self.n) * self.fov / (self.n - 1)
        r = b.world.raycast(b.x, b.y, angles, self.max_range)
        r = r + b.rng.normal(0.0, self.noise, self.n)
        return list(r)


class GPS(_Device):
    def __init__(self, body):
        self.body = body

    def getValues(self):
        return [self.body.x, self.body.y, 0.0]


class Compass(_Device):
    def __init__(self, body):
        self.body = body

    def getValues(self):
        # North (+y) expressed in the robot frame (x forward, y left)
        return [math.sin(self.body.theta), math.cos(self.body.theta), 0.0]


class GroundSensor(_Device):
    OFFSETS = [(0.03, 0.01), (0.03, 0.0), (0.03, -0.01)]   # gs0 (left), gs1, gs2 (right)

    def __init__(self, body, index):
        self.body, self.dx, self.dy = body, *self.OFFSETS[index]

    def getValue(self):
        b, c, s = self.body, math.cos(self.body.theta), math.sin(self.body.theta)
        x = b.x + self.dx * c - self.dy * s
        y = b.y + self.dx * s + self.dy * c
        return b.world.floor_value(x, y) + b.rng.normal(0.0, 8.0)


class LED(_Device):
    def __init__(self):
        self.value = 0

    def set(self, value):
        self.value = value


class Motor(_Device):
    def __init__(self):
        self.velocity = 0.0

    def setPosition(self, position):
        pass

    def setVelocity(self, velocity):
        if abs(velocity) > cfg.MAX_WHEEL_SPEED + 1e-9:
            raise ValueError(f"requested velocity {velocity} exceeds the motor limit")
        self.velocity = velocity


class Emitter(_Device):
    def __init__(self, radio):
        self.radio, self.channel = radio, 0

    def setChannel(self, channel):
        self.channel = channel

    def send(self, data):
        self.radio.broadcast(self, data if isinstance(data, bytes) else data.encode())


class Receiver(_Device):
    def __init__(self, radio):
        self.radio, self.channel, self.queue = radio, 0, []
        radio.receivers.append(self)

    def setChannel(self, channel):
        self.channel = channel

    def getQueueLength(self):
        return len(self.queue)

    def getBytes(self):
        return self.queue[0]

    def nextPacket(self):
        self.queue.pop(0)


class Radio:
    def __init__(self):
        self.receivers = []

    def broadcast(self, emitter, data):
        for r in self.receivers:
            if r.channel == emitter.channel:
                r.queue.append(data)


class SimRobot:
    """A simulated e-puck: differential-drive kinematics plus devices."""

    def __init__(self, world, radio, x, y, theta, timestep=32, seed=0):
        self.world, self.x, self.y, self.theta = world, x, y, theta
        self.timestep = timestep
        self.rng = np.random.default_rng(seed)
        self.collisions = 0
        self._in_collision = False
        self.distance = 0.0
        self.motors = {"left wheel motor": Motor(), "right wheel motor": Motor()}
        self.leds = [LED() for _ in range(8)]
        self.devices = {
            cfg.LIDAR_NAME: Lidar(self), "gps": GPS(self), "compass": Compass(self),
            "emitter": Emitter(radio), "receiver": Receiver(radio),
            **{f"gs{i}": GroundSensor(self, i) for i in range(3)},
            **{f"led{i}": led for i, led in enumerate(self.leds)},
            **self.motors,
        }

    def getBasicTimeStep(self):
        return self.timestep

    def getDevice(self, name):
        return self.devices[name]

    def advance(self):
        """Integrate the kinematics over one time step."""
        dt = self.timestep / 1000.0
        wl = self.motors["left wheel motor"].velocity
        wr = self.motors["right wheel motor"].velocity
        v = cfg.WHEEL_RADIUS * (wl + wr) / 2.0
        w = cfg.WHEEL_RADIUS * (wr - wl) / cfg.AXLE_LENGTH
        theta_mid = self.theta + w * dt / 2.0
        nx = self.x + v * dt * math.cos(theta_mid)
        ny = self.y + v * dt * math.sin(theta_mid)
        self.theta = (self.theta + w * dt + math.pi) % (2 * math.pi) - math.pi
        if self.world.collides(nx, ny, cfg.ROBOT_RADIUS):
            if not self._in_collision:
                self.collisions += 1
            self._in_collision = True
        else:
            self._in_collision = False
            self.distance += math.hypot(nx - self.x, ny - self.y)
            self.x, self.y = nx, ny

    def crossed_border(self):
        a, b, c, d = self.world.border_tile
        return a < self.x < c and b < self.y < d
