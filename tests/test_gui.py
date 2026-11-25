import os
from dataclasses import replace
from threading import Event
from time import monotonic, sleep

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QThreadPool  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog  # noqa: E402

from pollen_edf import ui  # noqa: E402
from pollen_edf.config import load_settings  # noqa: E402
from pollen_edf.evaluation import ROI  # noqa: E402
from pollen_edf.ui_workers import LatestTask, WorkSession  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def wait_for(condition, seconds=8):
    deadline = monotonic() + seconds
    while not condition() and monotonic() < deadline:
        QApplication.processEvents()
        sleep(0.015)
    assert condition()


def test_latest_only_and_worker_cancellation(app):
    pool = QThreadPool()
    latest = LatestTask(pool)
    first = Event()
    release = Event()
    results = []

    def delayed(_cancel, _progress):
        first.set()
        release.wait(3)
        return "stale"

    latest.complete.connect(lambda value, error: results.append((value, error)))
    latest.submit(delayed)
    wait_for(first.is_set)
    latest.submit(lambda _cancel, _progress: "also stale")
    latest.submit(lambda _cancel, _progress: "current")
    release.set()
    wait_for(lambda: results == [("current", None)])

    worker = WorkSession(pool)
    busy, finished = [], []
    worker.busy_changed.connect(busy.append)
    worker.complete.connect(lambda value, error: finished.append((value, error)))
    worker.start(lambda cancel, _progress: cancel.wait(2))
    assert worker.busy
    worker.stop()
    wait_for(lambda: bool(finished) and not worker.busy)
    assert busy == [True, False] and finished == [(True, None)]


def test_gui_starts_without_engine(app, tmp_path):
    settings = replace(load_settings(), engine=tmp_path / "missing.exe", artifacts_dir=tmp_path / "results")
    window = ui.MainWindow(settings)
    window.show()
    try:
        wait_for(lambda: window.engine_label.text() == "引擎不可用")
        assert window.a_folder.isEnabled() and not window.run_button.isEnabled()
        assert window.dataset is None and window.tree.topLevelItemCount() == 0
    finally:
        window.close()


def test_gui_import_raw_result_depth_and_roi(app, tmp_path, synthetic_root, stub_engine, monkeypatch):
    monkeypatch.setattr(ui, "FocusStackEngine", lambda _exe: stub_engine)
    monkeypatch.setattr("pollen_edf.experiments.machine_information", lambda: {"source": "GUI test fixture"})
    settings = replace(load_settings(), artifacts_dir=tmp_path / "results")
    window = ui.MainWindow(settings)
    window.show()
    try:
        wait_for(lambda: window.capabilities is not None)
        window.import_source(synthetic_root / "dataset.json")
        wait_for(lambda: window.raw.size_pixels is not None)
        assert len(window.dataset.stacks) == 3 and window.raw.source.name == "Z_001.jpg"
        for index in (6, 3, 2, 10, 11):
            window.slider.setValue(index)
        assert window.raw.source.name == "Z_012.jpg" and window.raw.size_pixels is None
        wait_for(lambda: window.raw.size_pixels is not None)
        window.raw.set_roi(ROI(10, 20, 40, 35))
        assert "ROI" in window.raw_label.text() and window.run_button.isEnabled()
        reference = synthetic_root / "XY_00_00" / "Z_006.jpg"
        monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *_args, **_kwargs: (str(reference), "JPEG"))
        monkeypatch.setattr(QDialog, "exec", lambda _dialog: QDialog.DialogCode.Accepted)
        window.import_reference()
        wait_for(lambda: window.compare.left.size_pixels is not None and window.compare.left.source == reference)
        assert window.references["XY_00_00"]["kind"] == "Helicon reference"
        window.parameters.depth.setChecked(True)
        window.run_selected()
        wait_for(lambda: window.compare.right.size_pixels is not None and not window.work.busy)
        assert window.selected_result.task["status"] == "success"
        assert "Laplacian Variance" in window.metrics.toPlainText()
        assert window.compare.left.size_pixels is not None
        window.compare.actual()
        assert window.compare.right.transform().m11() * window.compare.right.viewport().devicePixelRatioF() == 1
        window.tabs.setCurrentIndex(2)
        wait_for(lambda: window.depth.size_pixels is not None)
        assert window.depth.source.name == "depth.png"
        window.compare.right.set_roi(ROI(10, 20, 40, 35))
        window.compare_metrics()
        wait_for(lambda: not window.work.busy and "右图" in window.metrics.toPlainText())
        assert "ROI" in window.metrics.toPlainText()
    finally:
        window.close()
        QTest.qWait(100)
