from __future__ import annotations

import hashlib
import os
import sys
import tempfile
from pathlib import Path
from typing import Protocol
from urllib.request import Request, urlopen

import cv2
import numpy as np
from numpy.typing import NDArray

from pickleball_tracker.tracker import BallDetection, Candidate, ColorFrame


MODEL_NAME = "object_detection_yolox_2022nov.onnx"
MODEL_SHA256 = "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063"
MODEL_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/"
    "0b263e423d012606b83d1f81238d11c177da2b9c/models/object_detection_yolox/"
    f"{MODEL_NAME}"
)
INPUT_SIZE = 640
SPORTS_BALL_CLASS_ID = 32
MIN_CONFIDENCE = 0.01
NMS_THRESHOLD = 0.5
STRIDES = (8, 16, 32)
ModelOutput = NDArray[np.float32]


class YoloXNetwork(Protocol):
    def setPreferableBackend(self, backend: int) -> None: ...

    def setPreferableTarget(self, target: int) -> None: ...

    def setInput(self, input_blob: NDArray[np.float32]) -> None: ...

    def forward(self) -> ModelOutput: ...


def _cache_directory() -> Path:
    if sys.platform == "win32":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local"))
    elif sys.platform == "darwin":
        root = Path.home() / "Library/Caches"
    else:
        root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return root / "pickleball-tracker"


def ensure_pretrained_model(cache_dir: Path | None = None) -> Path:
    """Download the pinned COCO YOLOX model once and validate its checksum."""
    directory = cache_dir if cache_dir is not None else _cache_directory()
    model_path = directory / MODEL_NAME
    if model_path.is_file():
        if _sha256(model_path) != MODEL_SHA256:
            raise ValueError(f"Cached model checksum does not match: {model_path}")
        return model_path

    directory.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        request = Request(MODEL_URL, headers={"User-Agent": "pickleball-tracker"})
        with urlopen(request, timeout=60) as response:
            with tempfile.NamedTemporaryFile(
                mode="wb", prefix=f"{MODEL_NAME}.", suffix=".part",
                dir=directory, delete=False,
            ) as output:
                temporary_path = Path(output.name)
                digest = hashlib.sha256()
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
                    digest.update(chunk)

        if digest.hexdigest() != MODEL_SHA256:
            raise ValueError("Downloaded YOLOX model failed SHA-256 verification.")
        os.replace(temporary_path, model_path)
        return model_path
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


class YoloXSportsBallDetector:
    """Detect COCO sports balls with OpenCV DNN and a pinned YOLOX model."""

    def __init__(
        self,
        model_path: Path | None = None,
        *,
        network: YoloXNetwork | None = None,
        min_confidence: float = MIN_CONFIDENCE,
    ) -> None:
        if not 0 < min_confidence < 1:
            raise ValueError("min_confidence must be between 0 and 1.")
        self._network = (
            network
            if network is not None
            else cv2.dnn.readNetFromONNX(
                str(model_path if model_path is not None else ensure_pretrained_model())
            )
        )
        self._network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self._network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self._min_confidence = min_confidence
        self._grids, self._expanded_strides = self._make_grids()

    def __call__(self, frame: ColorFrame) -> list[Candidate]:
        height, width = frame.shape[:2]
        scale = min(INPUT_SIZE / height, INPUT_SIZE / width)
        resized_width = max(1, round(width * scale))
        resized_height = max(1, round(height * scale))
        resized = cv2.resize(
            frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
        )
        letterboxed = np.full(
            (INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8
        )
        letterboxed[:resized_height, :resized_width] = resized
        rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB).astype(np.float32)
        self._network.setInput(np.transpose(rgb, (2, 0, 1))[None])
        output = np.asarray(self._network.forward(), dtype=np.float32).reshape(-1, 85)
        return self._decode_candidates(output, scale, width, height)

    def _decode_candidates(
        self, output: ModelOutput, scale: float, frame_width: int, frame_height: int
    ) -> list[Candidate]:
        predictions = output.copy()
        predictions[:, :2] = (
            predictions[:, :2] + self._grids[0]
        ) * self._expanded_strides[0]
        predictions[:, 2:4] = (
            np.exp(np.clip(predictions[:, 2:4], -10, 10))
            * self._expanded_strides[0]
        )
        scores = predictions[:, 4] * predictions[:, 5 + SPORTS_BALL_CLASS_ID]
        indices = np.flatnonzero(scores >= self._min_confidence)

        boxes: list[list[int]] = []
        confidences: list[float] = []
        detections: list[BallDetection] = []
        for index in indices:
            center_x, center_y, box_width, box_height = predictions[index, :4]
            left = max(0.0, (center_x - box_width / 2) / scale)
            top = max(0.0, (center_y - box_height / 2) / scale)
            right = min(frame_width, (center_x + box_width / 2) / scale)
            bottom = min(frame_height, (center_y + box_height / 2) / scale)
            actual_width = right - left
            actual_height = bottom - top
            if actual_width <= 0 or actual_height <= 0:
                continue

            confidence = float(scores[index])
            boxes.append(
                [
                    round(left),
                    round(top),
                    round(actual_width),
                    round(actual_height),
                ]
            )
            confidences.append(confidence)
            detections.append(
                BallDetection(
                    x=(left + right) / 2,
                    y=(top + bottom) / 2,
                    radius=max(actual_width, actual_height) / 2,
                    confidence=confidence,
                )
            )

        retained = cv2.dnn.NMSBoxes(
            boxes,
            confidences,
            self._min_confidence,
            NMS_THRESHOLD,
        )
        return [
            (detections[int(index)], boxes[int(index)][2] * boxes[int(index)][3])
            for index in np.asarray(retained).reshape(-1)
        ]

    @staticmethod
    def _make_grids() -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        grids: list[NDArray[np.float32]] = []
        expanded_strides: list[NDArray[np.float32]] = []
        for stride in STRIDES:
            grid_size = INPUT_SIZE // stride
            grid_x, grid_y = np.meshgrid(
                np.arange(grid_size), np.arange(grid_size)
            )
            grids.append(
                np.stack((grid_x, grid_y), axis=2)
                .reshape(1, -1, 2)
                .astype(np.float32)
            )
            expanded_strides.append(
                np.full((1, grid_size * grid_size, 1), stride, dtype=np.float32)
            )
        return np.concatenate(grids, axis=1), np.concatenate(
            expanded_strides, axis=1
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        while chunk := model_file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
