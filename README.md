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
use **Previous frame** and **Next frame** to inspect tracking one frame at a
time while paused. Forward stepping preserves tracker state; backward stepping
rebuilds it from up to two preceding frames. The confidence readout shows a
heuristic score, detector and motion components, the selection reason, and the
number of color candidates.

To constrain detection to a court area, pause and choose **Select ROI**, then
click at least three polygon vertices around the feasible ball area. This is a
positive ROI: candidate centers inside the green polygon are considered, and
all candidates outside it are excluded by either tracking method. Right-click
to undo the last point, then choose **Finish ROI** to apply it and reprocess
the current frame. Choose **Clear ROI** to return to full-frame detection; the
ROI is reset when you open another video.

Use the **Tracking method** menu to choose **Color (HSV)** or **Learned (COCO
sports ball)**. The learned method downloads the pinned 35.9 MB YOLOX model from
OpenCV Zoo the first time it is selected and caches it outside the project. Its
checksum is verified before use. The model is Apache-2.0 licensed and trained
on COCO's generic sports-ball class, not specifically on pickleballs; it can
miss small or motion-blurred balls and may need you to select the ball to seed
tracking. The model and license are from the
[pinned OpenCV Zoo YOLOX release](https://github.com/opencv/opencv_zoo/tree/0b263e423d012606b83d1f81238d11c177da2b9c/models/object_detection_yolox).
Inference uses OpenCV DNN on the CPU and can be slower than real-time.

Choose **Select ball** and click its position in the video to seed tracking.
The color tracker is an early prototype and may need a manual seed or
correction, especially when the ball is small or blurred. It accepts compact
yellow motion streaks as candidates in addition to round detections, and labels
streak matches in the frame diagnostics. Both methods predict the next
position from recent detections, accept clear direction reversals, and allow a
minimum 200-pixel jump between frames, expanding for longer frame intervals.
Use **Select ball** if the active tracker still loses the ball.

The confidence score is diagnostic, not a calibrated probability. Its separate
diagnostic fields are intended to make it possible to replace the color-based
detector with a learned detector without coupling model output to the UI.

Run the tests with `python -m pytest`.
