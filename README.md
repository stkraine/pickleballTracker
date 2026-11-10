# Pickleball Tracker

A desktop app for tracking a pickleball's movement in recorded video.
<img width="1263" height="810" alt="image" src="https://github.com/user-attachments/assets/204a757c-9b90-4968-9546-32995552a0eb" />


## Plan

### MVP

Import a recorded video and play it back with controls. Detect the ball in each
frame and display its position and path as an overlay. Keep video analysis from
blocking the interface.

### Implementation

- Build the desktop interface with Python and PySide6.
- Use OpenCV to read and process video frames.
- Detect the ball by color and connect detections across frames to estimate its
  trajectory.
- Let users click the ball to seed or correct tracking when automatic detection
  loses it.
- Decode and track video frames in a background thread so playback controls
  remain responsive.

### Results and validation

- Let users review the tracked path.
- Export an annotated video and timestamped ball positions.
- Test video handling and tracking under varied lighting, court colors, motion
  blur, and occlusion.
- Package the app for the intended desktop operating system(s).

## Scope

Start with recorded-video tracking. Live camera tracking is a possible future
extension.

## Run

Python 3.10 or newer is required. The setup command installs the GUI and video
dependencies.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[test]"
python -m pickleball_tracker
```

Choose **Open video** to load a recording. Use **Play** to start playback, or
choose **Select ball** and click its position in the video to seed tracking.
The detector is an early color-based prototype and may need a manual seed or
correction, especially when the ball is small or blurred. Its tracking search
allows a minimum 120-pixel jump between frames and expands for longer frame
intervals; use **Select ball** to correct it if the ball is still lost.

Run the tests with `python -m pytest`.
