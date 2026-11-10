import queue

import cv2
import numpy as np
import pytest

from pickleball_tracker.app import VideoWorker


def test_video_worker_steps_forward_and_backward(tmp_path) -> None:
    video_path = tmp_path / "frame-step-test.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        30,
        (320, 240),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV MJPG video writer is unavailable.")

    for x in (60, 90, 120, 150):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        frame[:] = (45, 80, 35)
        cv2.circle(frame, (x, 100), 6, (0, 255, 255), -1)
        writer.write(frame)
    writer.release()

    messages = queue.Queue()
    frames = queue.Queue(maxsize=2)
    worker = VideoWorker(video_path, messages, frames)
    worker.start()

    try:
        assert messages.get(timeout=5).kind == "loaded"
        assert frames.get(timeout=5).frame_index == 0

        worker.step(1)
        worker.step(2)
        first_step = frames.get(timeout=5)
        assert first_step.frame_index == 1
        assert first_step.tracking_reason == "prediction_match"
        second_step = frames.get(timeout=5)
        assert second_step.frame_index == 2
        assert second_step.tracking_reason == "prediction_match"

        worker.step(1)
        previous_step = frames.get(timeout=5)
        assert previous_step.frame_index == 1
        assert previous_step.tracking_reason == "prediction_match"
    finally:
        worker.stop()
        worker.join(timeout=5)

    assert not worker.is_alive()


def test_video_worker_applies_and_clears_region_of_interest(tmp_path) -> None:
    video_path = tmp_path / "roi-test.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        30,
        (320, 240),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV MJPG video writer is unavailable.")

    for x in (60, 120, 180):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        frame[:] = (45, 80, 35)
        cv2.circle(frame, (x, 100), 6, (0, 255, 255), -1)
        writer.write(frame)
    writer.release()

    messages = queue.Queue()
    frames = queue.Queue(maxsize=2)
    worker = VideoWorker(video_path, messages, frames)
    worker.start()

    try:
        assert messages.get(timeout=5).kind == "loaded"
        assert frames.get(timeout=5).frame_index == 0

        worker.set_region_of_interest(
            ((100, 50), (150, 50), (150, 150), (100, 150))
        )
        worker.seek(1)
        filtered_frame = frames.get(timeout=5)

        assert filtered_frame.frame_index == 1
        assert filtered_frame.detected
        assert filtered_frame.candidate_count == 1
        assert filtered_frame.frame[50, 100, 1] > 200

        worker.set_region_of_interest(None)
        worker.seek(0)
        full_frame = frames.get(timeout=5)

        assert full_frame.frame_index == 0
        assert full_frame.detected
        assert full_frame.tracking_reason == "initial_detection"
        assert full_frame.frame[50, 100, 1] < 200
    finally:
        worker.stop()
        worker.join(timeout=5)

    assert not worker.is_alive()
