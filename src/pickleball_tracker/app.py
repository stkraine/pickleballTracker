from __future__ import annotations

import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeGuard

import cv2
import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QCloseEvent,
    QColor,
    QImage,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
    QPixmap,
    QPolygonF,
    QResizeEvent,
)
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from pickleball_tracker.learned_detector import YoloXSportsBallDetector
from pickleball_tracker.tracker import BallTracker, RegionOfInterest


TrackingMethod = Literal["color", "learned"]
ColorFrame = NDArray[np.uint8]
PREVIEW_WIDTH = 1200
PREVIEW_HEIGHT = 675


def _is_tracking_method(value: str) -> TypeGuard[TrackingMethod]:
    return value in ("color", "learned")


@dataclass(frozen=True)
class FramePacket:
    frame_index: int
    fps: float
    source_width: int
    source_height: int
    frame: ColorFrame
    detected: bool
    confidence: float
    detection_confidence: float | None
    motion_confidence: float | None
    candidate_count: int
    tracking_reason: str


@dataclass(frozen=True)
class WorkerMessage:
    kind: str
    detail: str = ""
    fps: float = 0.0
    total_frames: int = 0


@dataclass(frozen=True)
class _PlayRequest:
    playing: bool


@dataclass(frozen=True)
class _SeekRequest:
    frame_index: int


@dataclass(frozen=True)
class _StepRequest:
    frame_index: int


@dataclass(frozen=True)
class _SeedRequest:
    frame_index: int
    x: float
    y: float


@dataclass(frozen=True)
class _SetTrackingMethodRequest:
    method: TrackingMethod


@dataclass(frozen=True)
class _SetRegionOfInterestRequest:
    region_of_interest: RegionOfInterest | None


@dataclass(frozen=True)
class _StopRequest:
    pass


_Command = (
    _PlayRequest
    | _SeekRequest
    | _StepRequest
    | _SeedRequest
    | _SetTrackingMethodRequest
    | _SetRegionOfInterestRequest
    | _StopRequest
)


class VideoWorker(threading.Thread):
    """Decode and track frames off the GUI event loop."""

    def __init__(
        self,
        path: Path,
        messages: queue.Queue[WorkerMessage],
        frames: queue.Queue[FramePacket],
        tracking_method: TrackingMethod = "color",
    ) -> None:
        super().__init__(name="video-reader", daemon=True)
        self._path = path
        self._messages = messages
        self._frames = frames
        self._tracking_method = tracking_method
        self._commands: queue.Queue[_Command] = queue.Queue()
        self._stop_requested = threading.Event()

    def set_playing(self, playing: bool) -> None:
        self._commands.put(_PlayRequest(playing))

    def seek(self, frame_index: int) -> None:
        self._commands.put(_SeekRequest(frame_index))

    def step(self, frame_index: int) -> None:
        if frame_index < 0:
            raise ValueError("Frame index cannot be negative.")
        self._commands.put(_StepRequest(frame_index))

    def seed(self, frame_index: int, x: float, y: float) -> None:
        self._commands.put(_SeedRequest(frame_index, x, y))

    def set_tracking_method(self, method: TrackingMethod) -> None:
        self._commands.put(_SetTrackingMethodRequest(method))

    def set_region_of_interest(
        self, region_of_interest: RegionOfInterest | None
    ) -> None:
        BallTracker.validate_region_of_interest(region_of_interest)
        self._commands.put(_SetRegionOfInterestRequest(region_of_interest))

    def stop(self) -> None:
        self._stop_requested.set()
        self._commands.put(_StopRequest())

    def run(self) -> None:
        capture = cv2.VideoCapture(str(self._path))
        if not capture.isOpened():
            capture.release()
            self._messages.put(
                WorkerMessage("error", f"Could not open video:\n{self._path}")
            )
            return

        fps = capture.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            fps = 30.0
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        try:
            tracker = self._make_tracker(self._tracking_method)
        except (OSError, ValueError, cv2.error) as error:
            capture.release()
            self._messages.put(WorkerMessage("error", str(error)))
            return

        self._messages.put(
            WorkerMessage(
                "loaded",
                detail=self._tracking_method,
                fps=fps,
                total_frames=total_frames,
            )
        )
        frame_index = -1
        playing = False
        region_of_interest: RegionOfInterest | None = None
        next_frame_at = time.monotonic()

        ok, frame = capture.read()
        if not ok:
            self._messages.put(
                WorkerMessage("error", "Could not read the first video frame.")
            )
            capture.release()
            return
        frame_index = 0
        self._publish_frame(
            tracker, frame, frame_index, fps, source_width, source_height
        )

        while not self._stop_requested.is_set():
            try:
                first_command = self._commands.get(timeout=0.01 if playing else 0.05)
            except queue.Empty:
                first_command = None

            if first_command is not None:
                commands = [first_command]
                while True:
                    try:
                        commands.append(self._commands.get_nowait())
                    except queue.Empty:
                        break

                latest_play_request: _PlayRequest | None = None
                frame_requests: list[_SeekRequest | _StepRequest | _SeedRequest] = []
                latest_method_request: _SetTrackingMethodRequest | None = None
                latest_roi_request: _SetRegionOfInterestRequest | None = None
                should_stop = False
                for command in commands:
                    if isinstance(command, _StopRequest):
                        should_stop = True
                    elif isinstance(command, _PlayRequest):
                        latest_play_request = command
                    elif isinstance(
                        command,
                        (
                            _SeekRequest,
                            _StepRequest,
                            _SeedRequest,
                            _SetTrackingMethodRequest,
                            _SetRegionOfInterestRequest,
                        ),
                    ):
                        if isinstance(command, _SetTrackingMethodRequest):
                            latest_method_request = command
                        elif isinstance(command, _SetRegionOfInterestRequest):
                            latest_roi_request = command
                        elif isinstance(command, _StepRequest):
                            frame_requests.append(command)
                        elif frame_requests and isinstance(
                            frame_requests[-1], _SeekRequest
                        ):
                            frame_requests[-1] = command
                        else:
                            frame_requests.append(command)

                if should_stop:
                    break
                if latest_roi_request is not None:
                    region_of_interest = latest_roi_request.region_of_interest
                    tracker.set_region_of_interest(
                        region_of_interest
                    )
                if (
                    latest_method_request is not None
                    and latest_method_request.method != self._tracking_method
                ):
                    self._messages.put(
                        WorkerMessage(
                            "method_loading",
                            detail=latest_method_request.method,
                        )
                    )
                    try:
                        tracker = self._make_tracker(
                            latest_method_request.method,
                            region_of_interest,
                        )
                    except (OSError, ValueError, cv2.error) as error:
                        self._messages.put(
                            WorkerMessage("method_error", detail=str(error))
                        )
                    else:
                        self._tracking_method = latest_method_request.method
                        self._messages.put(
                            WorkerMessage(
                                "method_loaded",
                                detail=self._tracking_method,
                            )
                        )
                if latest_play_request is not None:
                    playing = latest_play_request.playing
                    next_frame_at = time.monotonic()
                    if (
                        playing
                        and total_frames > 0
                        and frame_index >= total_frames - 1
                    ):
                        tracker.reset()
                        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ok, frame = capture.read()
                        if not ok:
                            self._messages.put(
                                WorkerMessage(
                                    "error",
                                    "Could not restart the video from its first frame.",
                                )
                            )
                            playing = False
                        else:
                            frame_index = 0
                            self._publish_frame(
                                tracker,
                                frame,
                                frame_index,
                                fps,
                                source_width,
                                source_height,
                            )

                for frame_request in frame_requests:
                    if isinstance(frame_request, _StepRequest):
                        playing = False
                        target_frame = min(
                            max(frame_request.frame_index, 0),
                            max(total_frames - 1, 0),
                        )
                        if target_frame == frame_index:
                            continue
                        if target_frame > frame_index:
                            while frame_index < target_frame:
                                ok, frame = capture.read()
                                if not ok:
                                    self._messages.put(
                                        WorkerMessage(
                                            "error",
                                            "Could not decode the next video frame.",
                                        )
                                    )
                                    break
                                frame_index += 1
                                if frame_index == target_frame:
                                    self._publish_frame(
                                        tracker,
                                        frame,
                                        frame_index,
                                        fps,
                                        source_width,
                                        source_height,
                                    )
                                else:
                                    tracker.update(frame, frame_index / fps)
                        else:
                            frame_index = target_frame
                            replay_start = max(target_frame - 2, 0)
                            tracker.reset()
                            capture.set(cv2.CAP_PROP_POS_FRAMES, replay_start)
                            for replay_index in range(replay_start, target_frame + 1):
                                ok, frame = capture.read()
                                if not ok:
                                    self._messages.put(
                                        WorkerMessage(
                                            "error",
                                            "Could not decode frames while "
                                            "stepping backward.",
                                        )
                                    )
                                    break
                                if replay_index == target_frame:
                                    self._publish_frame(
                                        tracker,
                                        frame,
                                        frame_index,
                                        fps,
                                        source_width,
                                        source_height,
                                    )
                                else:
                                    tracker.update(
                                        frame, replay_index / fps
                                    )
                    else:
                        tracker.reset()
                        frame_index = min(
                            max(frame_request.frame_index, 0),
                            max(total_frames - 1, 0),
                        )
                        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                        ok, frame = capture.read()
                        if not ok:
                            self._messages.put(
                                WorkerMessage(
                                    "error", "Could not seek to that video frame."
                                )
                            )
                            playing = False
                            continue

                        if isinstance(frame_request, _SeedRequest):
                            tracker.seed(
                                frame_request.x,
                                frame_request.y,
                                frame_index / fps,
                            )
                        self._publish_frame(
                            tracker,
                            frame,
                            frame_index,
                            fps,
                            source_width,
                            source_height,
                        )
                    next_frame_at = time.monotonic()

            if not playing or time.monotonic() < next_frame_at:
                continue

            ok, frame = capture.read()
            if not ok:
                playing = False
                if total_frames and frame_index + 1 < total_frames:
                    self._messages.put(
                        WorkerMessage(
                            "error",
                            f"Video decoding stopped at frame {frame_index + 1} "
                            f"of {total_frames}.",
                        )
                    )
                else:
                    self._messages.put(WorkerMessage("ended"))
                continue

            frame_index += 1
            self._publish_frame(
                tracker, frame, frame_index, fps, source_width, source_height
            )
            next_frame_at += 1 / fps
            now = time.monotonic()
            if next_frame_at < now - 1 / fps:
                next_frame_at = now

        capture.release()

    @staticmethod
    def _make_tracker(
        method: TrackingMethod,
        region_of_interest: RegionOfInterest | None = None,
    ) -> BallTracker:
        if method == "learned":
            return BallTracker(
                candidate_provider=YoloXSportsBallDetector(),
                region_of_interest=region_of_interest,
            )
        if method == "color":
            return BallTracker(region_of_interest=region_of_interest)
        raise ValueError(f"Unknown tracking method: {method}")

    def _publish_frame(
        self,
        tracker: BallTracker,
        frame: ColorFrame,
        frame_index: int,
        fps: float,
        source_width: int,
        source_height: int,
    ) -> None:
        annotated, detection, diagnostics = tracker.update(frame, frame_index / fps)
        height, width = annotated.shape[:2]
        scale = min(PREVIEW_WIDTH / width, PREVIEW_HEIGHT / height, 1.0)
        if scale < 1:
            annotated = cv2.resize(
                annotated,
                (round(width * scale), round(height * scale)),
                interpolation=cv2.INTER_AREA,
            )
        rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        packet = FramePacket(
            frame_index=frame_index,
            fps=fps,
            source_width=source_width,
            source_height=source_height,
            frame=rgb_frame,
            detected=detection is not None,
            confidence=diagnostics.confidence,
            detection_confidence=diagnostics.detection_confidence,
            motion_confidence=diagnostics.motion_confidence,
            candidate_count=diagnostics.candidate_count,
            tracking_reason=diagnostics.reason,
        )
        try:
            self._frames.put_nowait(packet)
        except queue.Full:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                pass
            self._frames.put_nowait(packet)


class VideoCanvas(QLabel):
    ball_clicked = Signal(float, float)
    roi_point_added = Signal(int, int)
    roi_point_removed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(640, 360)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: #080c10; border-radius: 6px;")
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self._original_pixmap = QPixmap()
        self._source_dimensions = (0, 0)
        self._rendered_dimensions = (0, 0)
        self._selecting_ball = False
        self._selecting_roi = False
        self._roi_preview_points: list[tuple[int, int]] = []

    def set_frame(self, packet: FramePacket) -> None:
        rgb_frame = np.ascontiguousarray(packet.frame)
        height, width = rgb_frame.shape[:2]
        image = QImage(
            rgb_frame.data,
            width,
            height,
            rgb_frame.strides[0],
            QImage.Format.Format_RGB888,
        ).copy()
        self._original_pixmap = QPixmap.fromImage(image)
        self._source_dimensions = (packet.source_width, packet.source_height)
        self._update_pixmap()

    def set_ball_selection(self, selecting: bool) -> None:
        self._selecting_ball = selecting
        self._update_cursor()

    def set_roi_selection(self, selecting: bool) -> None:
        self._selecting_roi = selecting
        self._update_cursor()

    def set_roi_preview_points(self, points: list[tuple[int, int]]) -> None:
        self._roi_preview_points = points.copy()
        self.update()

    def clear_frame(self) -> None:
        self._original_pixmap = QPixmap()
        self._source_dimensions = (0, 0)
        self._rendered_dimensions = (0, 0)
        self.clear()

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._update_pixmap()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        super().mousePressEvent(event)
        if self._selecting_roi:
            if event.button() == Qt.MouseButton.RightButton:
                self.roi_point_removed.emit()
                event.accept()
                return
            if event.button() != Qt.MouseButton.LeftButton:
                return
            point = self._map_position_to_source(event.position())
            if point is None:
                return
            self.roi_point_added.emit(round(point[0]), round(point[1]))
            event.accept()
            return

        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self._selecting_ball:
            point = self._map_position_to_source(event.position())
            if point is not None:
                self.ball_clicked.emit(*point)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        super().mouseMoveEvent(event)
        if self._selecting_roi:
            self.update()

    def _image_display_rect(self) -> QRectF:
        rendered_width, rendered_height = self._rendered_dimensions
        contents = self.contentsRect()
        offset_x = contents.x() + (contents.width() - rendered_width) / 2
        offset_y = contents.y() + (contents.height() - rendered_height) / 2
        return QRectF(offset_x, offset_y, rendered_width, rendered_height)

    def _map_position_to_source(self, position: QPointF) -> tuple[float, float] | None:
        source_width, source_height = self._source_dimensions
        rendered_width, rendered_height = self._rendered_dimensions
        image_rect = self._image_display_rect()
        image_x = position.x() - image_rect.x()
        image_y = position.y() - image_rect.y()
        if (
            source_width <= 0
            or source_height <= 0
            or rendered_width <= 0
            or rendered_height <= 0
            or image_x < 0
            or image_y < 0
            or image_x >= rendered_width
            or image_y >= rendered_height
        ):
            return None
        return (
            image_x * source_width / rendered_width,
            image_y * source_height / rendered_height,
        )

    def paintEvent(self, event: QPaintEvent) -> None:
        super().paintEvent(event)
        if not self._roi_preview_points:
            return
        image_rect = self._image_display_rect()
        source_width, source_height = self._source_dimensions
        if source_width <= 0 or source_height <= 0:
            return
        points = QPolygonF(
            [
                QPointF(
                    image_rect.x() + x * image_rect.width() / source_width,
                    image_rect.y() + y * image_rect.height() / source_height,
                )
                for x, y in self._roi_preview_points
            ]
        )
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor(60, 220, 120), 2))
        if len(points) >= 3:
            painter.setBrush(QColor(60, 220, 120, 45))
            painter.drawPolygon(points)
        else:
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPolyline(points)
        painter.setBrush(QColor(60, 220, 120))
        for point in points:
            painter.drawEllipse(point, 4, 4)

    def _update_cursor(self) -> None:
        cursor = (
            Qt.CursorShape.CrossCursor
            if self._selecting_ball or self._selecting_roi
            else Qt.CursorShape.ArrowCursor
        )
        self.setCursor(cursor)

    def _update_pixmap(self) -> None:
        if self._original_pixmap.isNull():
            return
        target_size = self.contentsRect().size()
        if target_size.isEmpty():
            return
        scaled = self._original_pixmap.scaled(
            target_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._rendered_dimensions = (scaled.width(), scaled.height())
        self.setPixmap(scaled)


class PickleballTrackerApp(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Pickleball Tracker")
        self.resize(1280, 820)
        self.setMinimumSize(900, 600)
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background-color: #111820; color: #e7edf3; }
            QLabel#title { color: #f2f6fa; font-size: 20px; font-weight: 700; }
            QLabel { color: #c6d0d8; font-size: 13px; }
            QPushButton {
                background-color: #263744; color: #f2f6fa; border: 0;
                border-radius: 5px; padding: 9px 14px; font-size: 13px;
            }
            QPushButton:hover { background-color: #355266; }
            QPushButton:disabled { background-color: #202a32; color: #88949e; }
            QSlider::groove:horizontal {
                height: 5px; background: #33434f; border-radius: 2px;
            }
            QSlider::sub-page:horizontal { background: #50c8a0; border-radius: 2px; }
            QSlider::handle:horizontal {
                width: 14px; margin: -5px 0; border-radius: 7px;
                background: #50c8a0;
            }
            """
        )

        self._messages: queue.Queue[WorkerMessage] = queue.Queue()
        self._frames: queue.Queue[FramePacket] = queue.Queue(maxsize=2)
        self._worker: VideoWorker | None = None
        self._current_frame = 0
        self._fps = 30.0
        self._total_frames = 0
        self._playing = False
        self._selecting_ball = False
        self._selecting_roi = False
        self._region_of_interest: RegionOfInterest | None = None
        self._roi_points: list[tuple[int, int]] = []
        self._updating_slider = False
        self._confidence_band: str | None = None
        self._active_tracking_method: TrackingMethod = "color"

        self._status = QLabel("Open a video to begin.")
        self._status.setWordWrap(True)
        self._time_text = QLabel("00:00.0 / 00:00.0")
        self._tracking_method_selector = QComboBox()
        self._tracking_method_selector.addItem("Color (HSV)", "color")
        self._tracking_method_selector.addItem("Learned (COCO sports ball)", "learned")
        self._tracking_method_selector.currentIndexChanged.connect(
            self._on_tracking_method_changed
        )
        self._play_button = QPushButton("Play")
        self._play_button.setEnabled(False)
        self._previous_button = QPushButton("Previous frame")
        self._previous_button.setEnabled(False)
        self._next_button = QPushButton("Next frame")
        self._next_button.setEnabled(False)
        self._select_button = QPushButton("Select ball")
        self._select_button.setEnabled(False)
        self._roi_button = QPushButton("Select ROI")
        self._roi_button.setEnabled(False)
        self._roi_button.setToolTip(
            "Draw a positive polygon: detections inside are allowed; "
            "detections outside are excluded."
        )
        self._finish_roi_button = QPushButton("Finish ROI")
        self._finish_roi_button.setEnabled(False)
        self._finish_roi_button.setToolTip(
            "Apply the positive ROI after adding at least three points."
        )
        self._clear_roi_button = QPushButton("Clear ROI")
        self._clear_roi_button.setEnabled(False)
        self._timeline = QSlider(Qt.Orientation.Horizontal)
        self._timeline.setEnabled(False)
        self._confidence_bar = QProgressBar()
        self._confidence_bar.setRange(0, 100)
        self._confidence_bar.setFormat("%p%")
        self._confidence_bar.setTextVisible(True)
        self._confidence_bar.setToolTip(
            "Heuristic confidence combines candidate shape and motion agreement. "
            "It is not a calibrated probability."
        )
        self._diagnostics_text = QLabel("No frame analyzed.")
        self._diagnostics_text.setWordWrap(True)
        self._canvas = VideoCanvas()

        self._build_ui()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll_worker)
        self._timer.start(20)

    def _build_ui(self) -> None:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(16, 16, 16, 14)
        layout.setSpacing(10)

        heading = QHBoxLayout()
        title = QLabel("Pickleball Tracker")
        title.setObjectName("title")
        heading.addWidget(title)
        heading.addStretch(1)
        heading.addWidget(QLabel("Tracking method"))
        heading.addWidget(self._tracking_method_selector)
        open_button = QPushButton("Open video")
        open_button.clicked.connect(self._open_video)
        heading.addWidget(open_button)
        layout.addLayout(heading)

        self._canvas.ball_clicked.connect(self._on_video_click)
        self._canvas.roi_point_added.connect(self._on_roi_point_added)
        self._canvas.roi_point_removed.connect(self._on_roi_point_removed)
        layout.addWidget(self._canvas, stretch=1)

        controls = QHBoxLayout()
        self._play_button.clicked.connect(self._toggle_play)
        controls.addWidget(self._play_button)
        self._previous_button.clicked.connect(lambda: self._step_frame(-1))
        controls.addWidget(self._previous_button)
        self._next_button.clicked.connect(lambda: self._step_frame(1))
        controls.addWidget(self._next_button)
        self._select_button.clicked.connect(self._begin_ball_selection)
        controls.addWidget(self._select_button)
        self._roi_button.clicked.connect(self._begin_roi_selection)
        controls.addWidget(self._roi_button)
        self._finish_roi_button.clicked.connect(self._finish_roi_selection)
        controls.addWidget(self._finish_roi_button)
        self._clear_roi_button.clicked.connect(self._clear_roi)
        controls.addWidget(self._clear_roi_button)
        controls.addStretch(1)
        controls.addWidget(self._time_text)
        layout.addLayout(controls)

        self._timeline.valueChanged.connect(self._on_timeline_change)
        layout.addWidget(self._timeline)
        confidence_row = QHBoxLayout()
        confidence_row.addWidget(QLabel("Tracking confidence (heuristic)"))
        confidence_row.addWidget(self._confidence_bar, stretch=1)
        layout.addLayout(confidence_row)
        layout.addWidget(self._diagnostics_text)
        layout.addWidget(self._status)
        self.setCentralWidget(container)

    def _open_video(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Open a pickleball video",
            str(Path.home() / "Videos"),
            "Video files (*.mp4 *.mov *.mkv *.avi *.webm);;All files (*)",
        )
        if not selected:
            return

        if self._worker is not None and self._worker.is_alive():
            self._worker.stop()
            self._worker.join(timeout=2)
            if self._worker.is_alive():
                QMessageBox.critical(
                    self,
                    "Video still closing",
                    "The current video is still closing. Please try again.",
                )
                return

        self._worker = None
        self._messages = queue.Queue()
        self._frames = queue.Queue(maxsize=2)
        self._canvas.clear_frame()
        self._total_frames = 0
        self._current_frame = 0
        self._playing = False
        self._selecting_ball = False
        self._selecting_roi = False
        self._region_of_interest = None
        self._roi_points = []
        self._canvas.set_ball_selection(False)
        self._canvas.set_roi_selection(False)
        self._canvas.set_roi_preview_points([])
        self._tracking_method_selector.setEnabled(False)
        self._play_button.setEnabled(False)
        self._previous_button.setEnabled(False)
        self._next_button.setEnabled(False)
        self._select_button.setEnabled(False)
        self._roi_button.setEnabled(False)
        self._finish_roi_button.setEnabled(False)
        self._clear_roi_button.setEnabled(False)
        self._timeline.setRange(0, 0)
        self._timeline.setEnabled(False)
        self._confidence_bar.setValue(0)
        self._diagnostics_text.setText("Waiting for the first frame.")
        self._play_button.setText("Play")
        self._time_text.setText("00:00.0 / 00:00.0")
        method = self._tracking_method_selector.currentData()
        if not _is_tracking_method(method):
            raise ValueError(f"Unknown tracking method selected: {method}")
        self._active_tracking_method = method
        suffix = " (the learned model downloads on first use)" if method == "learned" else ""
        self._status.setText(f"Opening {Path(selected).name}{suffix}...")

        self._worker = VideoWorker(
            Path(selected),
            self._messages,
            self._frames,
            tracking_method=method,
        )
        self._worker.start()

    def _on_tracking_method_changed(self, _index: int) -> None:
        method = self._tracking_method_selector.currentData()
        if not isinstance(method, str) or not _is_tracking_method(method):
            return
        if self._worker is None:
            self._active_tracking_method = method
            return
        if method == self._active_tracking_method:
            return

        self._tracking_method_selector.setEnabled(False)
        self._stop_playback_for_inspection()
        self._worker.set_tracking_method(method)
        self._worker.seek(self._current_frame)
        self._status.setText(
            "Loading learned detector and resetting tracking..."
            if method == "learned"
            else "Switching to color tracking..."
        )

    def _toggle_play(self) -> None:
        if self._worker is None:
            return
        self._playing = not self._playing
        self._worker.set_playing(self._playing)
        self._play_button.setText("Pause" if self._playing else "Play")
        self._selecting_ball = False
        self._selecting_roi = False
        self._canvas.set_ball_selection(False)
        self._canvas.set_roi_selection(False)
        self._roi_points = []
        self._canvas.set_roi_preview_points([])
        self._finish_roi_button.setEnabled(False)
        self._status.setText("Playing" if self._playing else "Paused")

    def _begin_ball_selection(self) -> None:
        if self._worker is None:
            return
        self._stop_playback_for_inspection()
        self._selecting_ball = True
        self._canvas.set_ball_selection(True)
        self._status.setText(
            "Click the center of the ball in the video to seed tracking."
        )

    def _begin_roi_selection(self) -> None:
        if self._worker is None:
            return
        self._stop_playback_for_inspection()
        self._selecting_roi = True
        self._roi_points = []
        self._canvas.set_roi_preview_points([])
        self._canvas.set_roi_selection(True)
        self._finish_roi_button.setEnabled(False)
        self._status.setText(
            "Click at least three points around the positive ROI; right-click "
            "to undo the last point."
        )

    def _on_timeline_change(self, frame_index: int) -> None:
        if self._updating_slider or self._worker is None:
            return
        self._stop_playback_for_inspection()
        self._current_frame = frame_index
        self._worker.seek(frame_index)
        self._time_text.setText(self._format_time(frame_index / self._fps))

    def _step_frame(self, delta: int) -> None:
        if self._worker is None:
            return
        self._stop_playback_for_inspection()
        self._current_frame = min(
            max(self._current_frame + delta, 0),
            max(self._total_frames - 1, 0),
        )
        self._worker.step(self._current_frame)

    def _stop_playback_for_inspection(self) -> None:
        self._playing = False
        if self._worker is not None:
            self._worker.set_playing(False)
        self._play_button.setText("Play")
        self._selecting_ball = False
        self._selecting_roi = False
        self._canvas.set_ball_selection(False)
        self._canvas.set_roi_selection(False)
        self._roi_points = []
        self._canvas.set_roi_preview_points([])
        self._finish_roi_button.setEnabled(False)

    def _on_video_click(self, x: float, y: float) -> None:
        if not self._selecting_ball or self._worker is None:
            return
        self._worker.seed(self._current_frame, x, y)
        self._selecting_ball = False
        self._canvas.set_ball_selection(False)
        self._status.setText("Tracking from selected point. Press Play to continue.")

    def _on_roi_point_added(self, x: int, y: int) -> None:
        if not self._selecting_roi:
            return
        if self._roi_points and self._roi_points[-1] == (x, y):
            return
        self._roi_points.append((x, y))
        self._canvas.set_roi_preview_points(self._roi_points)
        self._finish_roi_button.setEnabled(len(self._roi_points) >= 3)
        self._status.setText(
            f"Positive ROI: {len(self._roi_points)} point(s). "
            "Add more points or choose Finish ROI."
        )

    def _on_roi_point_removed(self) -> None:
        if not self._selecting_roi or not self._roi_points:
            return
        self._roi_points.pop()
        self._canvas.set_roi_preview_points(self._roi_points)
        self._finish_roi_button.setEnabled(len(self._roi_points) >= 3)
        self._status.setText(
            f"Positive ROI: {len(self._roi_points)} point(s). "
            "Right-click removes the last point."
        )

    def _finish_roi_selection(self) -> None:
        if not self._selecting_roi or self._worker is None:
            return
        if len(self._roi_points) < 3:
            self._status.setText("A positive ROI needs at least three points.")
            return
        region_of_interest = tuple(self._roi_points)
        try:
            BallTracker.validate_region_of_interest(region_of_interest)
        except ValueError as error:
            self._status.setText(f"Could not use that ROI: {error}")
            return
        self._region_of_interest = region_of_interest
        self._selecting_roi = False
        self._canvas.set_roi_selection(False)
        self._canvas.set_roi_preview_points([])
        self._roi_points = []
        self._finish_roi_button.setEnabled(False)
        self._clear_roi_button.setEnabled(True)
        self._worker.set_region_of_interest(region_of_interest)
        self._worker.seek(self._current_frame)
        self._status.setText(
            "Positive ROI applied. Only detections inside the polygon are considered."
        )

    def _clear_roi(self) -> None:
        if self._worker is None or self._region_of_interest is None:
            return
        self._region_of_interest = None
        self._clear_roi_button.setEnabled(False)
        self._worker.set_region_of_interest(None)
        self._worker.seek(self._current_frame)
        self._status.setText("ROI cleared. The full frame is used for detection.")

    def _poll_worker(self) -> None:
        while True:
            try:
                message = self._messages.get_nowait()
            except queue.Empty:
                break
            self._handle_message(message)

        latest_frame: FramePacket | None = None
        while True:
            try:
                latest_frame = self._frames.get_nowait()
            except queue.Empty:
                break
        if latest_frame is not None:
            self._display_frame(latest_frame)

    def _handle_message(self, message: WorkerMessage) -> None:
        if message.kind == "loaded":
            if not _is_tracking_method(message.detail):
                raise RuntimeError(
                    f"Worker reported an unknown tracking method: {message.detail}"
                )
            self._active_tracking_method = message.detail
            self._tracking_method_selector.setEnabled(True)
            self._fps = message.fps
            self._total_frames = message.total_frames
            self._timeline.setRange(0, max(message.total_frames - 1, 0))
            self._timeline.setEnabled(message.total_frames > 0)
            self._play_button.setEnabled(True)
            self._previous_button.setEnabled(True)
            self._next_button.setEnabled(True)
            self._select_button.setEnabled(True)
            self._roi_button.setEnabled(True)
            self._status.setText(
                "Learned COCO sports-ball tracker ready. "
                "Select the ball to seed it if detection is uncertain."
                if message.detail == "learned"
                else "Video loaded. Play, or select the ball to seed tracking."
            )
        elif message.kind == "method_loading":
            self._status.setText(
                "Loading learned detector and resetting tracking..."
                if message.detail == "learned"
                else "Switching to color tracking..."
            )
        elif message.kind == "method_loaded":
            if not _is_tracking_method(message.detail):
                raise RuntimeError(
                    f"Worker reported an unknown tracking method: {message.detail}"
                )
            self._active_tracking_method = message.detail
            self._tracking_method_selector.setEnabled(True)
            self._status.setText(
                "Learned COCO sports-ball tracker ready. "
                "Select the ball to seed it if detection is uncertain."
                if message.detail == "learned"
                else "Color tracker ready."
            )
        elif message.kind == "method_error":
            self._tracking_method_selector.blockSignals(True)
            self._tracking_method_selector.setCurrentIndex(
                self._tracking_method_selector.findData(
                    self._active_tracking_method
                )
            )
            self._tracking_method_selector.blockSignals(False)
            self._tracking_method_selector.setEnabled(True)
            self._status.setText("Could not load the selected tracking method.")
            QMessageBox.critical(self, "Tracking method error", message.detail)
        elif message.kind == "error":
            self._playing = False
            self._play_button.setText("Play")
            self._play_button.setEnabled(False)
            self._previous_button.setEnabled(False)
            self._next_button.setEnabled(False)
            self._select_button.setEnabled(False)
            self._timeline.setEnabled(False)
            self._tracking_method_selector.blockSignals(True)
            self._tracking_method_selector.setCurrentIndex(
                self._tracking_method_selector.findData("color")
            )
            self._tracking_method_selector.blockSignals(False)
            self._tracking_method_selector.setEnabled(True)
            self._active_tracking_method = "color"
            self._roi_button.setEnabled(False)
            self._clear_roi_button.setEnabled(False)
            self._status.setText("Could not load or read the video.")
            QMessageBox.critical(self, "Video error", message.detail)
        elif message.kind == "ended":
            self._playing = False
            self._play_button.setText("Play")
            self._status.setText("End of video.")

    def _display_frame(self, packet: FramePacket) -> None:
        self._current_frame = packet.frame_index
        self._canvas.set_frame(packet)
        self._updating_slider = True
        self._timeline.setValue(packet.frame_index)
        self._updating_slider = False
        self._time_text.setText(
            f"{self._format_time(packet.frame_index / packet.fps)} / "
            f"{self._format_time(self._total_frames / packet.fps)}"
        )
        self._confidence_bar.setValue(round(packet.confidence * 100))
        confidence_band = (
            "high"
            if packet.confidence >= 0.7
            else "medium"
            if packet.confidence >= 0.4
            else "low"
        )
        if confidence_band != self._confidence_band:
            confidence_color = {
                "high": "#50c8a0",
                "medium": "#f0b44d",
                "low": "#e16b6b",
            }[confidence_band]
            self._confidence_bar.setStyleSheet(
                "QProgressBar { border: 1px solid #33434f; border-radius: 3px; "
                "text-align: center; background: #202a32; color: #f2f6fa; }"
                f"QProgressBar::chunk {{ background-color: {confidence_color}; }}"
            )
            self._confidence_band = confidence_band
        detection_confidence = (
            f"{packet.detection_confidence:.0%}"
            if packet.detection_confidence is not None
            else "n/a"
        )
        motion_confidence = (
            f"{packet.motion_confidence:.0%}"
            if packet.motion_confidence is not None
            else "n/a"
        )
        reason = packet.tracking_reason.replace("_", " ")
        self._diagnostics_text.setText(
            f"Frame {packet.frame_index} • {reason}"
            f" • detector {detection_confidence}"
            f" • motion {motion_confidence}"
            f" • {packet.candidate_count} candidate(s)"
        )
        if self._selecting_ball:
            return
        if self._playing:
            self._status.setText(
                "Tracking ball..." if packet.detected else "Searching for ball..."
            )

    @staticmethod
    def _format_time(seconds: float) -> str:
        minutes, remaining_seconds = divmod(max(seconds, 0.0), 60)
        return f"{int(minutes):02d}:{remaining_seconds:04.1f}"

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._worker is not None and self._worker.is_alive():
            self._worker.stop()
            self._worker.join(timeout=2)
            if self._worker.is_alive():
                QMessageBox.warning(
                    self,
                    "Video still closing",
                    "The video reader is still stopping. Please close the app again.",
                )
                event.ignore()
                return
        event.accept()


def main() -> int:
    application = QApplication(sys.argv)
    window = PickleballTrackerApp()
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
