import cv2
import numpy as np

from pickleball_tracker.tracker import BallTracker


def _frame_with_ball(x: int, y: int) -> np.ndarray:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[:] = (45, 80, 35)
    cv2.circle(frame, (x, y), 6, (0, 255, 255), -1)
    return frame


def test_seeded_tracker_follows_nearest_yellow_object() -> None:
    tracker = BallTracker()
    tracker.seed(100, 100, 0.0)

    frame = _frame_with_ball(120, 100)
    cv2.circle(frame, (240, 180), 6, (0, 255, 255), -1)
    annotated, detection = tracker.update(frame, 1 / 60)

    assert detection is not None
    assert abs(detection.x - 120) <= 1
    assert abs(detection.y - 100) <= 1
    assert annotated.shape == frame.shape
    assert not np.array_equal(annotated, frame)


def test_seeded_tracker_recovers_after_fast_hit() -> None:
    tracker = BallTracker()
    tracker.seed(50, 100, 0.0)

    frame = _frame_with_ball(160, 100)
    cv2.circle(frame, (270, 100), 6, (0, 255, 255), -1)
    _, detection = tracker.update(frame, 1 / 60)

    assert detection is not None
    assert abs(detection.x - 160) <= 1


def test_unseeded_tracker_detects_yellow_ball() -> None:
    tracker = BallTracker()
    _, detection = tracker.update(_frame_with_ball(150, 90), 0.0)

    assert detection is not None
    assert abs(detection.x - 150) <= 1
    assert abs(detection.y - 90) <= 1


def test_reset_clears_previous_tracking_position() -> None:
    tracker = BallTracker()
    tracker.seed(20, 20, 0.0)
    tracker.reset()

    _, detection = tracker.update(_frame_with_ball(280, 200), 1 / 60)

    assert detection is not None
    assert abs(detection.x - 280) <= 1
    assert abs(detection.y - 200) <= 1


def test_seed_remains_visible_when_color_detection_misses() -> None:
    tracker = BallTracker()
    tracker.seed(100, 100, 0.0)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    annotated, detection = tracker.update(frame, 0.0)

    assert detection is None
    assert not np.array_equal(annotated, frame)
