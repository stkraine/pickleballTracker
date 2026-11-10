from pathlib import Path

import numpy as np
import pytest

from pickleball_tracker.learned_detector import (
    MODEL_NAME,
    YoloXSportsBallDetector,
    ensure_pretrained_model,
)


class _FakeYoloXNetwork:
    def __init__(self) -> None:
        self.output = np.zeros((1, 8400, 85), dtype=np.float32)
        ball_cell = 9 * 80 + 40
        self.output[0, ball_cell, :6] = (0.5, 0.5, 0.0, 0.0, 0.9, 0.0)
        self.output[0, ball_cell, 5 + 32] = 0.8
        self.output[0, ball_cell + 1, :6] = (0.5, 0.5, 0.0, 0.0, 0.99, 0.0)
        self.output[0, ball_cell + 1, 5 + 0] = 0.99

    def setPreferableBackend(self, _backend: int) -> None:
        pass

    def setPreferableTarget(self, _target: int) -> None:
        pass

    def setInput(self, input_blob: np.ndarray) -> None:
        assert input_blob.shape == (1, 3, 640, 640)

    def forward(self) -> np.ndarray:
        return self.output


def test_yolox_returns_sports_ball_and_maps_letterboxed_coordinates() -> None:
    detector = YoloXSportsBallDetector(
        model_path=Path("unused.onnx"),
        network=_FakeYoloXNetwork(),
    )
    candidates = detector(np.zeros((1080, 1920, 3), dtype=np.uint8))

    assert len(candidates) == 1
    detection, _area = candidates[0]
    assert abs(detection.x - 972) < 1
    assert abs(detection.y - 228) < 1
    assert abs(detection.confidence - 0.72) < 0.001


def test_yolox_rejects_invalid_confidence_threshold() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        YoloXSportsBallDetector(
            model_path=Path("unused.onnx"),
            network=_FakeYoloXNetwork(),
            min_confidence=1.0,
        )


def test_cached_model_with_invalid_checksum_is_rejected(tmp_path: Path) -> None:
    (tmp_path / MODEL_NAME).write_bytes(b"not a model")

    with pytest.raises(ValueError, match="checksum"):
        ensure_pretrained_model(tmp_path)
