"""
Configuration file for the data collection system.
Modify these parameters to customize the behavior.
"""

# Robot Configuration
ROBOT_IP = "172.16.0.2"  # IP address of the Franka robot

# Control Parameters
TRANSLATION_SCALE = 3.0    # Increased for more responsive SpaceMouse input
ROTATION_SCALE = 2.0       # Increased for more responsive rotation
LOOP_RATE_HZ = 10          # Data collection frequency (robot control runs at 1000Hz independently)

# Data Storage
DATA_ROOT_DIR = "./my_robot_data"  # Where to save collected data

# SpaceMouse Settings
SPACEMOUSE_DEADZONE = 0.02  # Further reduced deadzone for more sensitive zero detection

# Gripper Settings
GRIPPER_TOGGLE_DEBOUNCE = 0.5  # Minimum time between gripper toggles (seconds)
GRIPPER_OPEN_WIDTH = 0.08      # Target width when gripper is open
GRIPPER_CLOSED_WIDTH = 0.0     # Target width when gripper is closed

# Safety Limits (optional - for future use)
MAX_TRANSLATION_DELTA = 0.05   # Maximum translation per step (meters)
MAX_ROTATION_DELTA = 0.1       # Maximum rotation per step (radians)
