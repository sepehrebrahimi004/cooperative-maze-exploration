"""Tunable parameters shared by the controllers and the offline simulator."""

# --- Map -------------------------------------------------------------------
CELL_SIZE = 0.025                      # occupancy-grid resolution (m)
MAP_X_LIMITS = (-1.6, 1.6)             # world area covered by the map (m)
MAP_Y_LIMITS = (-1.1, 1.1)

# --- e-puck geometry -------------------------------------------------------
WHEEL_RADIUS = 0.0205                  # m
AXLE_LENGTH = 0.052                    # m
MAX_WHEEL_SPEED = 6.28                 # rad/s
ROBOT_RADIUS = 0.037                   # m
INFLATION_RADIUS = 0.045               # obstacles are grown by this for planning (m)
CLEARANCE_WIDTH = 0.075                # m; beyond the inflation, A* pays extra to stay this far from walls
CLEARANCE_WEIGHT = 3.0                 # extra cost per cell right next to the inflated area

# --- Motion control --------------------------------------------------------
MAX_LINEAR_SPEED = 0.10                # m/s
MAX_ANGULAR_SPEED = 2.5                # rad/s
ROTATE_IN_PLACE_ANGLE = 0.6            # rad; turn on the spot above this heading error
WAYPOINT_TOLERANCE = 0.03              # m
GOAL_TOLERANCE = 0.04                  # m
EMERGENCY_STOP_DISTANCE = 0.06         # m; LiDAR range in front that stops forward motion
HEADING_PID = dict(kp=4.0, ki=0.4, kd=0.05, tf=0.05)
DISTANCE_PID = dict(kp=1.2, ki=0.0, kd=0.0, tf=0.05)

# --- Exploration -----------------------------------------------------------
MAP_UPDATE_PERIOD = 0.128              # s between LiDAR map updates
REPLAN_PERIOD = 1.0                    # s between frontier re-selections
MIN_FRONTIER_SIZE = 4                  # cells; smaller frontier clusters are ignored
FRONTIER_APPROACH_RADIUS = 0.05        # m; robot only needs to get this close to a frontier
STUCK_TIMEOUT = 6.0                    # s without progress before a target is abandoned
BLACKLIST_RADIUS = 0.06                # m; area around an abandoned/visited target that is skipped
SWEEP_RADIUS = 0.07                    # m; floor area around the ground sensors counted as checked
LIDAR_RAY_STRIDE = 2                   # use every n-th LiDAR ray for mapping

# --- Ground sensors (thresholds calibrated in the course world) ------------
RED_THRESHOLD = 890                    # gs value above which the floor is a red tile
BORDER_THRESHOLD = 550                 # gs value below which the floor is the blue border tile
BORDER_FORBIDDEN_RADIUS = 0.15         # m; area around the border tile the robot may not enter
RED_TILE_MIN_SEPARATION = 0.25         # m; detections closer than this are the same tile

# --- Devices (names used in the Webots world) ------------------------------
LIDAR_NAME = "Hokuyo URG-04LX"
COMM_CHANNEL = 1
