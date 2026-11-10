import cv2
import numpy as np

from pickleball_tracker.tracker import BallDetection, BallTracker


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
    annotated, detection, diagnostics = tracker.update(frame, 1 / 60)

    assert detection is not None
    assert diagnostics.reason == "prediction_match"
    assert 0.0 <= diagnostics.confidence <= 1.0
    assert diagnostics.detection_confidence is not None
    assert diagnostics.motion_confidence is not None
    assert diagnostics.candidate_count == 2
    assert abs(detection.x - 120) <= 1
    assert abs(detection.y - 100) <= 1
    assert annotated.shape == frame.shape
    assert not np.array_equal(annotated, frame)


def test_seeded_tracker_recovers_after_fast_hit() -> None:
    tracker = BallTracker()
    tracker.seed(50, 100, 0.0)

    frame = _frame_with_ball(160, 100)
    cv2.circle(frame, (270, 100), 6, (0, 255, 255), -1)
    _, detection, _ = tracker.update(frame, 1 / 60)

    assert detection is not None
    assert abs(detection.x - 160) <= 1


def test_tracker_accepts_elongated_yellow_motion_streak() -> None:
    tracker = BallTracker()
    tracker.seed(50, 100, 0.0)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[:] = (45, 80, 35)
    cv2.line(frame, (100, 100), (155, 100), (0, 255, 255), 6)

    _, detection, diagnostics = tracker.update(frame, 1 / 60)

    assert detection is not None
    assert 120 <= detection.x <= 135
    assert detection.is_streak
    assert diagnostics.reason == "streak_nearest_reacquisition"
    assert diagnostics.detection_confidence is not None
    assert 0 < diagnostics.detection_confidence < 1


def test_tracker_uses_velocity_prediction_to_ignore_distractor() -> None:
    tracker = BallTracker()
    tracker.seed(50, 100, 0.0)
    _, first, _ = tracker.update(_frame_with_ball(80, 100), 1 / 60)
    assert first is not None

    frame = _frame_with_ball(110, 100)
    cv2.circle(frame, (85, 100), 6, (0, 255, 255), -1)
    _, detection, _ = tracker.update(frame, 2 / 60)

    assert detection is not None
    assert abs(detection.x - 110) <= 1


def test_tracker_accepts_direction_reversal_and_resets_prediction() -> None:
    tracker = BallTracker()
    tracker.seed(50, 100, 0.0)
    _, first, _ = tracker.update(_frame_with_ball(80, 100), 1 / 60)
    assert first is not None

    frame = _frame_with_ball(40, 100)
    cv2.circle(frame, (110, 100), 6, (0, 255, 255), -1)
    _, reversal, reversal_diagnostics = tracker.update(frame, 2 / 60)

    assert reversal is not None
    assert reversal_diagnostics.reason == "direction_reversal"
    assert reversal_diagnostics.motion_confidence is not None
    assert reversal_diagnostics.detection_confidence is not None
    assert (
        reversal_diagnostics.motion_confidence
        < reversal_diagnostics.detection_confidence
    )
    assert abs(reversal.x - 40) <= 1

    frame = _frame_with_ball(20, 100)
    cv2.circle(frame, (100, 100), 6, (0, 255, 255), -1)
    _, continued, continued_diagnostics = tracker.update(frame, 3 / 60)

    assert continued is not None
    assert continued_diagnostics.reason == "prediction_match"
    assert abs(continued.x - 20) <= 1


def test_unseeded_tracker_detects_yellow_ball() -> None:
    tracker = BallTracker()
    _, detection, diagnostics = tracker.update(_frame_with_ball(150, 90), 0.0)

    assert detection is not None
    assert diagnostics.reason == "initial_detection"
    assert diagnostics.motion_confidence is None
    assert diagnostics.candidate_count == 1
    assert abs(detection.x - 150) <= 1
    assert abs(detection.y - 90) <= 1


def test_reset_clears_previous_tracking_position() -> None:
    tracker = BallTracker()
    tracker.seed(20, 20, 0.0)
    tracker.reset()

    _, detection, _ = tracker.update(_frame_with_ball(280, 200), 1 / 60)

    assert detection is not None
    assert abs(detection.x - 280) <= 1
    assert abs(detection.y - 200) <= 1


def test_seed_remains_visible_when_color_detection_misses() -> None:
    tracker = BallTracker()
    tracker.seed(100, 100, 0.0)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    annotated, detection, diagnostics = tracker.update(frame, 0.0)

    assert detection is None
    assert diagnostics.confidence == 0
    assert diagnostics.reason == "no_color_candidates"
    assert not np.array_equal(annotated, frame)


def test_learned_candidate_provider_reports_generic_miss_reason() -> None:
    tracker = BallTracker(candidate_provider=lambda _frame: [])

    _, detection, diagnostics = tracker.update(_frame_with_ball(100, 100), 0.0)

    assert detection is None
    assert diagnostics.reason == "no_candidates"


def test_region_of_interest_filters_candidates_and_is_drawn_on_frame() -> None:
    positive_roi = (
        (20, 20),
        (100, 20),
        (100, 60),
        (60, 60),
        (60, 100),
        (20, 100),
    )
    tracker = BallTracker(
        candidate_provider=lambda _frame: [
            (BallDetection(40, 50, 5, 0.9), 80),
            (BallDetection(80, 80, 5, 0.95), 80),
            (BallDetection(240, 180, 5, 0.95), 80),
        ],
        region_of_interest=positive_roi,
    )
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    annotated, detection, diagnostics = tracker.update(frame, 0.0)

    assert detection is not None
    assert (detection.x, detection.y) == (40, 50)
    assert diagnostics.candidate_count == 1
    assert diagnostics.reason == "initial_detection"
    assert np.any(annotated[20, 20:100] != frame[20, 20:100])


def test_region_of_interest_reports_when_all_candidates_are_outside() -> None:
    tracker = BallTracker(
        candidate_provider=lambda _frame: [
            (BallDetection(240, 180, 5, 0.95), 80),
        ],
        region_of_interest=((20, 20), (100, 20), (100, 100), (20, 100)),
    )

    _, detection, diagnostics = tracker.update(
        np.zeros((240, 320, 3), dtype=np.uint8), 0.0
    )

    assert detection is None
    assert diagnostics.candidate_count == 0
    assert diagnostics.reason == "outside_roi"


def test_changing_region_of_interest_resets_previous_tracking_state() -> None:
    tracker = BallTracker(
        candidate_provider=lambda _frame: [
            (BallDetection(150, 150, 5, 0.9), 80),
        ]
    )
    tracker.seed(10, 10, 0.0)
    tracker.set_region_of_interest(
        ((100, 100), (200, 100), (200, 200), (100, 200))
    )

    _, detection, diagnostics = tracker.update(
        np.zeros((240, 320, 3), dtype=np.uint8), 1 / 60
    )

    assert detection is not None
    assert (detection.x, detection.y) == (150, 150)
    assert diagnostics.reason == "initial_detection"


def test_invalid_region_of_interest_is_rejected() -> None:
    try:
        BallTracker(region_of_interest=((20, 20), (50, 50)))
    except ValueError as error:
        assert "at least three points" in str(error)
    else:
        raise AssertionError("Invalid ROI bounds were accepted.")


def test_tracker_reports_when_candidates_are_outside_search() -> None:
    tracker = BallTracker()
    tracker.seed(10, 10, 0.0)

    _, detection, diagnostics = tracker.update(_frame_with_ball(300, 220), 1 / 60)

    assert detection is None
    assert diagnostics.confidence == 0
    assert diagnostics.candidate_count == 1
    assert diagnostics.reason == "outside_search"
