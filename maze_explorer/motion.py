"""Closed-loop waypoint following for a differential-drive robot."""

import math

from . import config as cfg
from .pid import PID


def wrap_angle(a):
    """Wrap an angle to [-pi, pi)."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def unicycle_to_wheels(v, w):
    """Convert (linear, angular) velocity to wheel speeds (rad/s), scaling both
    wheels down together if either would exceed the motor limit."""
    left = (v - w * cfg.AXLE_LENGTH / 2.0) / cfg.WHEEL_RADIUS
    right = (v + w * cfg.AXLE_LENGTH / 2.0) / cfg.WHEEL_RADIUS
    scale = max(1.0, abs(left) / cfg.MAX_WHEEL_SPEED, abs(right) / cfg.MAX_WHEEL_SPEED)
    return left / scale, right / scale


class PathFollower:
    """Follows a list of world-frame waypoints with two PID loops:

    * heading loop  - error = bearing to the next waypoint - robot heading,
                      output = angular velocity
    * distance loop - error = remaining path length, output = linear velocity

    When the heading error is large the robot first turns on the spot, and the
    forward speed is scaled by cos(heading error) so it slows down in turns.
    """

    def __init__(self, ts):
        self.heading_pid = PID(ts=ts, output_limits=(-cfg.MAX_ANGULAR_SPEED, cfg.MAX_ANGULAR_SPEED),
                               **cfg.HEADING_PID)
        self.distance_pid = PID(ts=ts, output_limits=(0.0, cfg.MAX_LINEAR_SPEED),
                                **cfg.DISTANCE_PID)
        self.waypoints = []

    def set_path(self, waypoints):
        self.waypoints = list(waypoints)
        self.heading_pid.reset()
        self.distance_pid.reset()

    @property
    def done(self):
        return not self.waypoints

    def remaining_length(self, x, y):
        pts = [(x, y)] + self.waypoints
        return sum(math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))

    def update(self, x, y, theta):
        """Return (left, right) wheel speeds for the current pose."""
        while self.waypoints:
            tol = cfg.GOAL_TOLERANCE if len(self.waypoints) == 1 else cfg.WAYPOINT_TOLERANCE
            if math.dist((x, y), self.waypoints[0]) > tol:
                break
            self.waypoints.pop(0)
        if not self.waypoints:
            return 0.0, 0.0

        tx, ty = self.waypoints[0]
        heading_error = wrap_angle(math.atan2(ty - y, tx - x) - theta)
        w = self.heading_pid.update(heading_error)
        if abs(heading_error) > cfg.ROTATE_IN_PLACE_ANGLE:
            v = 0.0
        else:
            v = self.distance_pid.update(self.remaining_length(x, y)) * math.cos(heading_error)
        return unicycle_to_wheels(v, w)
