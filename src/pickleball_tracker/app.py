from __future__ import annotations

import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QCloseEvent, QImage, QMouseEvent, QPixmap, QResizeEvent
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from pickleball_tracker.tracker import BallTracker


ColorFrame = NDArray[np.uint8]
PREVIEW_WIDTH = 1200
PREVIEW_HEIGHT = 675


@dataclass(frozen=True)
class FramePacket:
    frame_index: int
    fps: float
    source_width: int
    source_height: int
    frame: ColorFrame
    detected: bool


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
class _SeedRequest:
    frame_index: int
    x: float
    y: float


@dataclass(frozen=True)
class _StopRequest:
    pass


_Command = _PlayRequest | _SeekRequest | _SeedRequest | _StopRequest


class VideoWorker(threading.Thread):
    """Decode and track frames off the GUI event loop."""

    def __init__(
        self,
        path: Path,
        messages: queue.Queue[WorkerMessage],
        frames: queue.Queue[FramePacket],
    ) -> None:
        super().__init__(name="video-reader", daemon=True)
        self._path = path
        self._messages = messages
        self._frames = frames
        self._commands: queue.Queue[_Command] = queue.Queue()
        self._stop_requested = threading.Event()

    def set_playing(self, playing: bool) -> None:
        self._commands.put(_PlayRequest(playing))

    def seek(self, frame_index: int) -> None:
        self._commands.put(_SeekRequest(frame_index))

    def seed(self, frame_index: int, x: float, y: float) -> None:
        self._commands.put(_SeedRequest(frame_index, x, y))

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
        self._messages.put(
            WorkerMessage("loaded", fps=fps, total_frames=total_frames)
        )

        tracker = BallTracker()
        frame_index = -1
        playing = False
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
                latest_frame_request: _SeekRequest | _SeedRequest | None = None
                should_stop = False
                for command in commands:
                    if isinstance(command, _StopRequest):
                        should_stop = True
                    elif isinstance(command, _PlayRequest):
                        latest_play_request = command
                    elif isinstance(command, (_SeekRequest, _SeedRequest)):
                        latest_frame_request = command

                if should_stop:
                    break
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

                if latest_frame_request is not None:
                    tracker.reset()
                    frame_index = min(
                        max(latest_frame_request.frame_index, 0),
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

                    if isinstance(latest_frame_request, _SeedRequest):
                        tracker.seed(
                            latest_frame_request.x,
                            latest_frame_request.y,
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

    def _publish_frame(
        self,
        tracker: BallTracker,
        frame: ColorFrame,
        frame_index: int,
        fps: float,
        source_width: int,
        source_height: int,
    ) -> None:
        annotated, detection = tracker.update(frame, frame_index / fps)
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

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(640, 360)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: #080c10; border-radius: 6px;")
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self._original_pixmap = QPixmap()
        self._source_dimensions = (0, 0)
        self._rendered_dimensions = (0, 0)

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
        cursor = (
            Qt.CursorShape.CrossCursor if selecting else Qt.CursorShape.ArrowCursor
        )
        self.setCursor(cursor)

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
        if event.button() != Qt.MouseButton.LeftButton:
            return

        rendered_width, rendered_height = self._rendered_dimensions
        source_width, source_height = self._source_dimensions
        if (
            not rendered_width
            or not rendered_height
            or not source_width
            or not source_height
        ):
            return

        contents = self.contentsRect()
        offset_x = contents.x() + (contents.width() - rendered_width) / 2
        offset_y = contents.y() + (contents.height() - rendered_height) / 2
        image_x = event.position().x() - offset_x
        image_y = event.position().y() - offset_y
        if not (0 <= image_x < rendered_width and 0 <= image_y < rendered_height):
            return

        self.ball_clicked.emit(
            image_x * source_width / rendered_width,
            image_y * source_height / rendered_height,
        )

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
        self._updating_slider = False

        self._status = QLabel("Open a video to begin.")
        self._status.setWordWrap(True)
        self._time_text = QLabel("00:00.0 / 00:00.0")
        self._play_button = QPushButton("Play")
        self._play_button.setEnabled(False)
        self._select_button = QPushButton("Select ball")
        self._select_button.setEnabled(False)
        self._timeline = QSlider(Qt.Orientation.Horizontal)
        self._timeline.setEnabled(False)
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
        open_button = QPushButton("Open video")
        open_button.clicked.connect(self._open_video)
        heading.addWidget(open_button)
        layout.addLayout(heading)

        self._canvas.ball_clicked.connect(self._on_video_click)
        layout.addWidget(self._canvas, stretch=1)

        controls = QHBoxLayout()
        self._play_button.clicked.connect(self._toggle_play)
        controls.addWidget(self._play_button)
        self._select_button.clicked.connect(self._begin_ball_selection)
        controls.addWidget(self._select_button)
        controls.addStretch(1)
        controls.addWidget(self._time_text)
        layout.addLayout(controls)

        self._timeline.valueChanged.connect(self._on_timeline_change)
        layout.addWidget(self._timeline)
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
        self._canvas.set_ball_selection(False)
        self._play_button.setEnabled(False)
        self._select_button.setEnabled(False)
        self._timeline.setRange(0, 0)
        self._timeline.setEnabled(False)
        self._play_button.setText("Play")
        self._time_text.setText("00:00.0 / 00:00.0")
        self._status.setText(f"Opening {Path(selected).name}...")

        self._worker = VideoWorker(Path(selected), self._messages, self._frames)
        self._worker.start()

    def _toggle_play(self) -> None:
        if self._worker is None:
            return
        self._playing = not self._playing
        self._worker.set_playing(self._playing)
        self._play_button.setText("Pause" if self._playing else "Play")
        self._selecting_ball = False
        self._canvas.set_ball_selection(False)
        self._status.setText("Playing" if self._playing else "Paused")

    def _begin_ball_selection(self) -> None:
        if self._worker is None:
            return
        self._selecting_ball = True
        self._playing = False
        self._worker.set_playing(False)
        self._play_button.setText("Play")
        self._canvas.set_ball_selection(True)
        self._status.setText(
            "Click the center of the ball in the video to seed tracking."
        )

    def _on_timeline_change(self, frame_index: int) -> None:
        if self._updating_slider or self._worker is None:
            return
        self._current_frame = frame_index
        self._worker.seek(frame_index)
        self._time_text.setText(self._format_time(frame_index / self._fps))

    def _on_video_click(self, x: float, y: float) -> None:
        if not self._selecting_ball or self._worker is None:
            return
        self._worker.seed(self._current_frame, x, y)
        self._selecting_ball = False
        self._canvas.set_ball_selection(False)
        self._status.setText("Tracking from selected point. Press Play to continue.")

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
            self._fps = message.fps
            self._total_frames = message.total_frames
            self._timeline.setRange(0, max(message.total_frames - 1, 0))
            self._timeline.setEnabled(message.total_frames > 0)
            self._play_button.setEnabled(True)
            self._select_button.setEnabled(True)
            self._status.setText(
                "Video loaded. Play, or select the ball to seed tracking."
            )
        elif message.kind == "error":
            self._playing = False
            self._play_button.setText("Play")
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
