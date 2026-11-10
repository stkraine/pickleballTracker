from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from math import hypot, pi

import cv2
import numpy as np
from numpy.typing import NDArray


ColorFrame = NDArray[np.uint8]
RegionOfInterest = tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class BallDetection:
    x: float
    y: float
    radius: float
    confidence: float
    is_streak: bool = False


Candidate = tuple[BallDetection, float]
CandidateProvider = Callable[[ColorFrame], list[Candidate]]


@dataclass(frozen=True)
class TrackingDiagnostics:
    confidence: float
    detection_confidence: float | None
    motion_confidence: float | None
    candidate_count: int
    reason: str


class BallTracker:
    """Track small, bright-yellow objects and draw their recent path."""

    MIN_SEARCH_RADIUS = 200.0
    MAX_SEARCH_SPEED = 3600.0
    MAX_PREDICTION_TIME = 0.1
    MIN_PREDICTION_TOLERANCE = 40.0
    MAX_PREDICTION_TOLERANCE = 120.0
    MIN_REVERSAL_DISTANCE = 20.0
    MIN_REVERSAL_ALIGNMENT = 0.5
    MAX_CANDIDATE_AREA = 2500.0
    MAX_CANDIDATE_DIMENSION = 120
    MAX_CANDIDATE_ASPECT_RATIO = 12.0
    STREAK_ASPECT_RATIO = 3.5

    def __init__(
        self,
        history_seconds: float = 2.0,
        *,
        candidate_provider: CandidateProvider | None = None,
        region_of_interest: RegionOfInterest | None = None,
    ) -> None:
        self._history_seconds = history_seconds
        self._trail: deque[tuple[float, BallDetection]] = deque()
        self._previous: tuple[float, BallDetection] | None = None
        self._prior: tuple[float, BallDetection] | None = None
        self._candidate_provider = candidate_provider
        self._region_of_interest = self.validate_region_of_interest(
            region_of_interest
        )

    def reset(self) -> None:
        self._trail.clear()
        self._previous = None
        self._prior = None

    def set_region_of_interest(
        self, region_of_interest: RegionOfInterest | None
    ) -> None:
        self._region_of_interest = self.validate_region_of_interest(
            region_of_interest
        )
        self.reset()

    def seed(self, x: float, y: float, timestamp: float) -> None:
        detection = BallDetection(x=x, y=y, radius=7.0, confidence=1.0)
        self._previous = (timestamp, detection)
        self._prior = None
        self._trail.append((timestamp, detection))

    def update(
        self, frame: ColorFrame, timestamp: float
    ) -> tuple[ColorFrame, BallDetection | None, TrackingDiagnostics]:
        detection, reset_velocity, diagnostics = self._find_detection(
            frame, timestamp
        )

        if detection is not None:
            if not reset_velocity:
                self._prior = self._previous
            else:
                self._prior = None
            self._previous = (timestamp, detection)
            self._trail.append((timestamp, detection))
        elif self._previous is not None and timestamp - self._previous[0] > 0.5:
            self._previous = None
            self._prior = None

        while self._trail and timestamp - self._trail[0][0] > self._history_seconds:
            self._trail.popleft()

        annotated = frame.copy()
        self._draw_region_of_interest(annotated)
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
        return annotated, detection, diagnostics

    def _find_detection(
        self, frame: ColorFrame, timestamp: float
    ) -> tuple[BallDetection | None, bool, TrackingDiagnostics]:
        candidates = (
            self._candidate_provider(frame)
            if self._candidate_provider is not None
            else self._find_color_candidates(frame)
        )
        candidate_count = len(candidates)
        if self._region_of_interest is not None:
            polygon = np.asarray(self._region_of_interest, dtype=np.float32)
            candidates = [
                candidate
                for candidate in candidates
                if cv2.pointPolygonTest(
                    polygon, (candidate[0].x, candidate[0].y), False
                )
                >= 0
            ]

        if not candidates:
            reason = (
                "outside_roi"
                if candidate_count and self._region_of_interest is not None
                else (
                    "no_candidates"
                    if self._candidate_provider is not None
                    else "no_color_candidates"
                )
            )
            return None, False, TrackingDiagnostics(
                confidence=0.0,
                detection_confidence=None,
                motion_confidence=None,
                candidate_count=0,
                reason=reason,
            )

        if self._previous is not None:
            return self._select_near_previous(candidates, timestamp)

        return self._select_initial_candidate(frame, candidates)

    def _find_color_candidates(
        self, frame: ColorFrame
    ) -> list[Candidate]:
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
            if not 4 <= area <= self.MAX_CANDIDATE_AREA:
                continue

            _, _, width, height = cv2.boundingRect(contour)
            if max(width, height) > self.MAX_CANDIDATE_DIMENSION:
                continue
            aspect_ratio = max(width / height, height / width)
            if aspect_ratio > self.MAX_CANDIDATE_ASPECT_RATIO:
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
            extent = min(1.0, area / (width * height))
            is_streak = aspect_ratio >= self.STREAK_ASPECT_RATIO
            detection_confidence = (
                min(0.65, 0.35 + 0.3 * extent) if is_streak else circularity
            )
            detection = BallDetection(
                x=center_x,
                y=center_y,
                radius=(
                    min(width, height) / 2
                    if is_streak
                    else hypot(width, height) / 4
                ),
                confidence=detection_confidence,
                is_streak=is_streak,
            )
            candidates.append((detection, area))

        return candidates

    def _select_near_previous(
        self, candidates: list[Candidate], timestamp: float
    ) -> tuple[BallDetection | None, bool, TrackingDiagnostics]:
        if self._previous is None:
            raise RuntimeError("Cannot select near a previous detection before tracking.")

        previous_time, previous = self._previous
        elapsed = max(timestamp - previous_time, 1 / 120)
        max_distance = max(
            self.MIN_SEARCH_RADIUS, self.MAX_SEARCH_SPEED * elapsed
        )
        nearby = [
            candidate
            for candidate in candidates
            if hypot(candidate[0].x - previous.x, candidate[0].y - previous.y)
            <= max_distance
        ]
        if not nearby:
            return None, False, TrackingDiagnostics(
                confidence=0.0,
                detection_confidence=None,
                motion_confidence=None,
                candidate_count=len(candidates),
                reason="outside_search",
            )

        velocity = self._estimate_velocity()
        if velocity is None:
            predicted_x, predicted_y = previous.x, previous.y
        else:
            prediction_time = min(elapsed, self.MAX_PREDICTION_TIME)
            predicted_x = previous.x + velocity[0] * prediction_time
            predicted_y = previous.y + velocity[1] * prediction_time

            reversals = [
                candidate
                for candidate in nearby
                if self._is_direction_reversal(candidate[0], previous, velocity)
            ]
            if reversals:
                detection = self._rank_by_previous(reversals, previous)
                return detection, True, self._diagnostics(
                    detection,
                    motion_confidence=0.35,
                    candidate_count=len(candidates),
                    reason=self._candidate_reason(detection, "direction_reversal"),
                )

        tolerance = max(
            self.MIN_PREDICTION_TOLERANCE,
            min(self.MAX_PREDICTION_TOLERANCE, max_distance * 0.25),
        )
        predicted_candidates = [
            candidate
            for candidate in nearby
            if hypot(
                candidate[0].x - predicted_x,
                candidate[0].y - predicted_y,
            )
            <= tolerance
        ]
        if predicted_candidates:
            detection = min(
                predicted_candidates,
                key=lambda candidate: (
                    hypot(
                        candidate[0].x - predicted_x,
                        candidate[0].y - predicted_y,
                    )
                    / tolerance
                    + (1.0 - candidate[0].confidence) * 0.35
                ),
            )[0]
            motion_confidence = max(
                0.0,
                1
                - hypot(
                    detection.x - predicted_x,
                    detection.y - predicted_y,
                )
                / tolerance,
            )
            return detection, False, self._diagnostics(
                detection,
                motion_confidence=motion_confidence,
                candidate_count=len(candidates),
                reason=self._candidate_reason(detection, "prediction_match"),
            )

        detection = self._rank_by_previous(nearby, previous)
        return detection, True, self._diagnostics(
            detection,
            motion_confidence=0.2,
            candidate_count=len(candidates),
            reason=self._candidate_reason(detection, "nearest_reacquisition"),
        )

    def _select_initial_candidate(
        self, frame: ColorFrame, candidates: list[Candidate]
    ) -> tuple[BallDetection, bool, TrackingDiagnostics]:
        height, width = frame.shape[:2]
        center_x, center_y = width / 2, height / 2
        detection = min(
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
        return detection, False, self._diagnostics(
            detection,
            motion_confidence=None,
            candidate_count=len(candidates),
            reason=self._candidate_reason(detection, "initial_detection"),
        )

    @staticmethod
    def _candidate_reason(detection: BallDetection, reason: str) -> str:
        return f"streak_{reason}" if detection.is_streak else reason

    @staticmethod
    def _diagnostics(
        detection: BallDetection,
        motion_confidence: float | None,
        candidate_count: int,
        reason: str,
    ) -> TrackingDiagnostics:
        detection_confidence = min(max(detection.confidence, 0.0), 1.0)
        if motion_confidence is not None:
            motion_confidence = min(max(motion_confidence, 0.0), 1.0)
            confidence = 0.6 * detection_confidence + 0.4 * motion_confidence
        else:
            confidence = detection_confidence
        return TrackingDiagnostics(
            confidence=confidence,
            detection_confidence=detection_confidence,
            motion_confidence=motion_confidence,
            candidate_count=candidate_count,
            reason=reason,
        )

    def _estimate_velocity(self) -> tuple[float, float] | None:
        if self._prior is None or self._previous is None:
            return None

        prior_time, prior = self._prior
        previous_time, previous = self._previous
        elapsed = previous_time - prior_time
        if elapsed <= 0 or elapsed > 0.25:
            return None

        velocity_x = (previous.x - prior.x) / elapsed
        velocity_y = (previous.y - prior.y) / elapsed
        speed = hypot(velocity_x, velocity_y)
        if speed > self.MAX_SEARCH_SPEED:
            scale = self.MAX_SEARCH_SPEED / speed
            velocity_x *= scale
            velocity_y *= scale
        return velocity_x, velocity_y

    @classmethod
    def _is_direction_reversal(
        cls,
        candidate: BallDetection,
        previous: BallDetection,
        velocity: tuple[float, float],
    ) -> bool:
        delta_x = candidate.x - previous.x
        delta_y = candidate.y - previous.y
        distance = hypot(delta_x, delta_y)
        if distance < cls.MIN_REVERSAL_DISTANCE:
            return False

        speed = hypot(*velocity)
        if speed == 0:
            return False
        alignment = (delta_x * velocity[0] + delta_y * velocity[1]) / (
            distance * speed
        )
        return alignment <= -cls.MIN_REVERSAL_ALIGNMENT

    @staticmethod
    def _rank_by_previous(
        candidates: list[tuple[BallDetection, float]],
        previous: BallDetection,
    ) -> BallDetection:
        return min(
            candidates,
            key=lambda candidate: (
                hypot(
                    candidate[0].x - previous.x,
                    candidate[0].y - previous.y,
                )
                + (1.0 - candidate[0].confidence) * 15
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

    def _draw_region_of_interest(self, frame: ColorFrame) -> None:
        if self._region_of_interest is None:
            return
        height, width = frame.shape[:2]
        polygon = np.asarray(
            [
                (min(max(x, 0), width - 1), min(max(y, 0), height - 1))
                for x, y in self._region_of_interest
            ],
            dtype=np.int32,
        )
        cv2.polylines(
            frame,
            [polygon],
            True,
            (60, 220, 120),
            2,
            cv2.LINE_AA,
        )

    @staticmethod
    def validate_region_of_interest(
        region_of_interest: RegionOfInterest | None,
    ) -> RegionOfInterest | None:
        if region_of_interest is None:
            return None
        if len(region_of_interest) < 3:
            raise ValueError("ROI polygon must contain at least three points.")
        if any(x < 0 or y < 0 for x, y in region_of_interest):
            raise ValueError("ROI polygon points must be non-negative.")
        if cv2.contourArea(np.asarray(region_of_interest, dtype=np.float32)) <= 0:
            raise ValueError("ROI polygon must enclose a positive area.")
        return region_of_interest
