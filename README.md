# GazeControl

GazeControl is a real-time eye-gaze mouse and virtual keyboard system built with Python.
It combines camera-based gaze detection, smoothing, screen-point estimation, dwell clicking, and an on-screen keyboard with word suggestions.

## Features

- Real-time gaze tracking with MediaPipe Face Mesh
- Head-pose-aware gaze signal extraction (yaw and pitch included in debug info)
- Kalman + exponential smoothing for stable pointer movement
- Mouse movement driven by gaze
- Dwell click support with configurable radius, timing, and cooldown
- Full-width virtual keyboard window (optimized for macOS main-thread UI constraints)
- Word prediction and suggestion dwell selection
- Camera preview and debug overlay
- Automatic device selection: MPS, CUDA, or CPU

## Project Structure

- `main.py` - application entry point
- `config/` - global settings (camera, smoothing, dwell, keyboard, logging)
- `gaze/` - camera capture, detection, smoothing, and coordinate estimation
- `control/` - mouse movement and dwell click handling
- `keyboard/` - virtual keyboard, layout, and word prediction
- `ui/` - overlays and click feedback
- `utils/` - logging and screen utilities

## Requirements

- macOS (primary target in current implementation)
- Python 3.10+
- Webcam

Python packages used by this project:

- opencv-python
- mediapipe
- numpy
- torch
- screeninfo
- pygame
- pyautogui
- filterpy

## Setup

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install --upgrade pip
pip install opencv-python mediapipe numpy torch screeninfo pygame pyautogui filterpy
```

3. Run the app:

```bash
python main.py
```

## macOS Permissions

On first run, macOS may block mouse/keyboard control or camera access.
Grant the following permissions in System Settings:

- Privacy & Security -> Accessibility (for mouse/keyboard control)
- Privacy & Security -> Input Monitoring (may be required depending on OS setup)
- Privacy & Security -> Camera

If controls do not respond after enabling permissions, restart the terminal and run again.

## Controls

- Press `q` to quit from the keyboard window or OpenCV preview window.
- Gaze above the keyboard area enables dwell click behavior.
- Gaze over keys/suggestions to type/select via dwell.

## Configuration

Tune behavior in `config/settings.py`:

- Camera: index, FPS, frame size
- Smoothing: `SMOOTHING_FACTOR`, Kalman noise values
- Dwell: `DWELL_TIME`, `DWELL_COOLDOWN`, `DWELL_RADIUS`
- Gaze mapping ranges: `GAZE_X_MIN/MAX`, `GAZE_Y_MIN/MAX`
- Keyboard: key size and spacing
- Logging: level and output file

## Troubleshooting

- No camera feed:
  - Check camera permission and whether another app is using the webcam.
- Cursor movement is jittery:
  - Increase smoothing and/or adjust Kalman noise values.
- Dwell clicks trigger too early/late:
  - Adjust `DWELL_TIME`, `DWELL_RADIUS`, and `DWELL_COOLDOWN`.
- Mouse/typing does not work on macOS:
  - Re-check Accessibility/Input Monitoring permissions for your terminal and Python.

## Notes

- Logging output is written to `gaze_control.log`.
- The virtual keyboard and preview loops are intentionally coordinated for macOS threading behavior.
