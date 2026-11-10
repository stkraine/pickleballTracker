import os
import queue
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from pickleball_tracker.app import PickleballTrackerApp, VideoCanvas, VideoWorker

QT_APP = QApplication.instance() or QApplication([])


def test_tracking_method_selector_switches_paused_video_and_reprocesses() -> None:
    window = PickleballTrackerApp()
    worker = VideoWorker(Path("unused"), queue.Queue(), queue.Queue())
    window._worker = worker
    window._current_frame = 12

    window._tracking_method_selector.setCurrentIndex(1)

    assert worker._commands.get_nowait().playing is False
    assert worker._commands.get_nowait().method == "learned"
    assert worker._commands.get_nowait().frame_index == 12
    assert not window._tracking_method_selector.isEnabled()

    window._timer.stop()
    window.close()
    window.deleteLater()
    QT_APP.processEvents()


def test_roi_controls_send_polygon_and_clear_request_to_worker() -> None:
    window = PickleballTrackerApp()
    worker = VideoWorker(Path("unused"), queue.Queue(), queue.Queue())
    window._worker = worker
    window._current_frame = 12
    window._begin_roi_selection()

    window._on_roi_point_added(10, 20)
    window._on_roi_point_added(110, 20)
    window._on_roi_point_added(110, 120)
    assert window._finish_roi_button.isEnabled()
    window._finish_roi_selection()

    assert worker._commands.get_nowait().playing is False
    assert worker._commands.get_nowait().region_of_interest == (
        (10, 20),
        (110, 20),
        (110, 120),
    )
    assert worker._commands.get_nowait().frame_index == 12
    assert window._clear_roi_button.isEnabled()

    window._clear_roi()

    assert worker._commands.get_nowait().region_of_interest is None
    assert worker._commands.get_nowait().frame_index == 12
    assert not window._clear_roi_button.isEnabled()

    window._timer.stop()
    window.close()
    window.deleteLater()
    QT_APP.processEvents()


def test_canvas_roi_clicks_map_to_source_polygon_points() -> None:
    canvas = VideoCanvas()
    canvas.resize(1000, 600)
    canvas._source_dimensions = (1920, 1080)
    canvas._rendered_dimensions = (960, 540)
    canvas.set_roi_selection(True)
    canvas.show()
    QT_APP.processEvents()
    image_rect = canvas._image_display_rect()
    added: list[tuple[int, int]] = []
    canvas.roi_point_added.connect(lambda x, y: added.append((x, y)))
    for display_x, display_y in ((100, 60), (500, 60), (500, 300)):
        QTest.mouseClick(
            canvas,
            Qt.MouseButton.LeftButton,
            pos=QPoint(
                round(image_rect.x() + display_x),
                round(image_rect.y() + display_y),
            ),
        )

    assert added == [(200, 120), (1000, 120), (1000, 600)]
    removed: list[bool] = []
    canvas.roi_point_removed.connect(lambda: removed.append(True))
    canvas.set_roi_preview_points(added)
    QT_APP.processEvents()
    QTest.mouseClick(
        canvas,
        Qt.MouseButton.RightButton,
        pos=QPoint(round(image_rect.x() + 500), round(image_rect.y() + 300)),
    )
    assert removed == [True]
    canvas.deleteLater()
    QT_APP.processEvents()
