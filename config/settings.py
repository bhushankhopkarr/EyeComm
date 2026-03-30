import torch
from screeninfo import get_monitors

# --- Screen ---
_monitor = get_monitors()[0]
SCREEN_WIDTH = _monitor.width
SCREEN_HEIGHT = _monitor.height

# --- Camera ---
CAMERA_INDEX = 0
CAMERA_FPS = 30
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# --- Gaze Smoothing ---
SMOOTHING_FACTOR = 0.96
FAST_SMOOTHING_FACTOR = 0.75
SMALL_MOVEMENT_THRESHOLD = 0.02
KALMAN_PROCESS_NOISE = 1e-4
KALMAN_MEASUREMENT_NOISE = 5e-1
CURSOR_DEADZONE = 10

# --- Dwell ---
DWELL_TIME = 0.8
DWELL_COOLDOWN = 1.2
DWELL_RADIUS = 50

# --- Virtual Keyboard ---
KEY_WIDTH = 60
KEY_HEIGHT = 60
KEY_PADDING = 10
KEYBOARD_ALPHA = 220

# --- Device ---
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# --- Logging ---
LOGGING_ENABLED = False
LOG_LEVEL = "DEBUG"
LOG_FILE = "gaze_control.log"
