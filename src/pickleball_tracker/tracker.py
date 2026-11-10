from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import hypot, pi

import cv2
import numpy as np
from numpy.typing import NDArray


ColorFrame = NDArray[np.uint8]


@dataclass(frozen=True)
class BallDetection:
    x: float
    y: float
    radius: float
    confidence: float


class BallTracker:
    """Track small, bright-yellow objects and draw their recent path."""

    MIN_SEARCH_RADIUS = 120.0
    MAX_SEARCH_SPEED = 6000.0

    def __init__(self, history_seconds: float = 2.0) -> None:
        self._history_seconds = history_seconds
        self._trail: deque[tuple[float, BallDetection]] = deque()
        self._previous: tuple[float, BallDetection] | None = None

    def reset(self) -> None:
        self._trail.clear()
        self._previous = None

    def seed(self, x: float, y: float, timestamp: float) -> None:
        detection = BallDetection(x=x, y=y, radius=7.0, confidence=1.0)
        self._previous = (timestamp, detection)
        self._trail.append((timestamp, detection))

    def update(
        self, frame: ColorFrame, timestamp: float
    ) -> tuple[ColorFrame, BallDetection | None]:
        detection = self._find_detection(frame, timestamp)

        if detection is not None:
            self._previous = (timestamp, detection)
            self._trail.append((timestamp, detection))
        elif self._previous is not None and timestamp - self._previous[0] > 0.5:
            self._previous = None

        while self._trail and timestamp - self._trail[0][0] > self._history_seconds:
            self._trail.popleft()

        annotated = frame.copy()
        self._draw_trail(annotated, timestamp)
        visible_detection = detection
        if visible_detection is None and self._previous is not None:
            previous_time, previous_detection = self._previous
            if timestamp - previous_time <= 0.25:
                visible_detection = previous_detection
        if visible_detection is not None:
            center = (round(visible_detection.x), round(visible_detection.y))
            radius = max(5, round(visible_detection.radius) + 4)
            cv2.circle(annotated, center, radius, (0, 210, 255), 2, cv2.LINE_AA)
            cv2.drawMarker(
                annotated,
                center,
                (255, 255, 255),
                cv2.MARKER_CROSS,
                14,
                1,
                cv2.LINE_AA,
            )
        return annotated, detection

    def _find_detection(
        self, frame: ColorFrame, timestamp: float
    ) -> BallDetection | None:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array((16, 90, 130), dtype=np.uint8),
            np.array((48, 255, 255), dtype=np.uint8),
        )
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates: list[tuple[BallDetection, float]] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if not 4 <= area <= 650:
                continue

            x, y, width, height = cv2.boundingRect(contour)
            if max(width, height) > 48:
                continue
            aspect_ratio = width / height
            if not 0.25 <= aspect_ratio <= 4.0:
                continue

            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue

            center_x = moments["m10"] / moments["m00"]
            center_y = moments["m01"] / moments["m00"]
            perimeter = cv2.arcLength(contour, True)
            circularity = (
                min(1.0, 4 * pi * area / (perimeter * perimeter))
                if perimeter
                else 0.0
            )
            detection = BallDetection(
                x=center_x,
                y=center_y,
                radius=hypot(width, height) / 4,
                confidence=circularity,
            )
            candidates.append((detection, area))

        if not candidates:
            return None

        if self._previous is not None:
            previous_time, previous = self._previous
            elapsed = max(timestamp - previous_time, 1 / 120)
            max_distance = max(
                self.MIN_SEARCH_RADIUS, self.MAX_SEARCH_SPEED * elapsed
            )
            nearby = [
                (detection, area)
                for detection, area in candidates
                if hypot(detection.x - previous.x, detection.y - previous.y)
                <= max_distance
            ]
            if not nearby:
                return None

            return min(
                nearby,
                key=lambda candidate: (
                    hypot(
                        candidate[0].x - previous.x,
                        candidate[0].y - previous.y,
                    )
                    / max_distance
                    + (1.0 - candidate[0].confidence) * 0.35
                ),
            )[0]

        height, width = frame.shape[:2]
        center_x, center_y = width / 2, height / 2
        return min(
            candidates,
            key=lambda candidate: (
                1.0 - candidate[0].confidence
                + hypot(
                    (candidate[0].x - center_x) / width,
                    (candidate[0].y - center_y) / height,
                )
                * 0.2
                + abs(candidate[1] - 50) / 1000
            ),
        )[0]

    def _draw_trail(self, frame: ColorFrame, timestamp: float) -> None:
        points = [
            (round(detection.x), round(detection.y))
            for seen_at, detection in self._trail
            if timestamp - seen_at <= self._history_seconds
        ]
        if len(points) > 1:
            cv2.polylines(
                frame,
                [np.asarray(points, dtype=np.int32)],
                False,
                (0, 210, 255),
                2,
                cv2.LINE_AA,
            )
