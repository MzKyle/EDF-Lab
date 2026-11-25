from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from PySide6.QtCore import QThreadPool, QTimer, Qt, Slot
from PySide6.QtGui import QAction, QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QDialog, QDialogButtonBox,
                               QFileDialog, QFormLayout, QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit,
                               QListWidget, QMainWindow, QPlainTextEdit, QProgressBar, QScrollArea, QSlider,
                               QSplitter, QTabWidget, QToolBar, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from .config import Settings
from .dataset import Dataset, Stack, load_dataset, manual_dataset, validate_stack
from .engine import EngineCapabilities, FocusStackEngine
from .evaluation import METRIC_LABELS, focus_curve, measure_file
from .experiments import PRESETS, ExperimentRunner, ExperimentUpdate, ensure_separate_output
from .images import absolute_path, check_cancel, sha256
from .process_runner import log_tail
from .records import artifact_path, attach_reference, get_reference, read_json, save_review, write_report
from .ui_compare import Comparison, button
from .ui_parameters import Parameters
from .ui_viewer import ImageViewer, PreviewCache
from .ui_workers import WorkSession


STYLE = """
QWidget { background: #17222f; color: #e0e8f0; font-size: 12px; }
QMainWindow, QStatusBar { background: #131d28; }
QToolBar { border: 0; border-bottom: 1px solid #314052; background: #1a2837; spacing: 5px; padding: 6px 8px; }
QToolButton, QPushButton { background: #213143; border: 1px solid #395067; border-radius: 3px; padding: 5px 9px; }
QToolButton:hover, QPushButton:hover { background: #2d4359; border-color: #537189; }
QPushButton:checked { color: #acf3e7; background: #1b484b; border-color: #3e8f8a; }
QPushButton#runButton { color: #e4fffa; background: #216b65; border-color: #469891; font-weight: 600; }
QPushButton#runButton:hover { background: #278078; }
QToolButton:disabled, QPushButton:disabled, QPushButton#runButton:disabled, QCheckBox:disabled, QLabel:disabled {
    background: #192533; color: #758498; border-color: #2b3b4e; }
QTreeWidget, QPlainTextEdit, QGraphicsView, QListWidget { background: #101923; border: 1px solid #314257; }
QTreeWidget, QListWidget { alternate-background-color: #14202c; outline: 0; }
QTreeWidget::item, QListWidget::item { min-height: 25px; }
QTreeWidget::item:selected, QListWidget::item:selected { background: #244755; color: #eafffc; }
QHeaderView::section { background: #1d2d3e; border: 0; border-bottom: 1px solid #35485b; padding: 5px; color: #bacbdc; }
QLineEdit, QComboBox { background: #111b26; border: 1px solid #394f64; border-radius: 3px; padding: 4px 6px; }
QLineEdit:focus, QComboBox:focus { border-color: #50a49a; }
QComboBox QAbstractItemView { background: #182635; selection-background-color: #2b535f; }
QComboBox::drop-down { border: 0; width: 22px; }
QTabBar::tab { background: #1b2938; color: #afc0d2; padding: 8px 15px; border-right: 1px solid #314153; }
QTabBar::tab:selected { color: #d8fbf4; border-bottom: 2px solid #5bc5b6; background: #233647; }
QTabWidget::pane { border: 1px solid #314153; }
QGroupBox { border: 1px solid #33475b; border-radius: 3px; margin-top: 12px; padding: 6px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; color: #b9d1e4; }
QScrollArea { border: 0; }
QScrollBar:horizontal, QScrollBar:vertical { background: #111b26; border: 0; }
QScrollBar:horizontal { height: 8px; }
QScrollBar:vertical { width: 8px; }
QScrollBar::handle { background: #395269; border-radius: 2px; min-height: 18px; min-width: 18px; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }
QSlider::groove:horizontal { height: 4px; background: #304356; }
QSlider::handle:horizontal { background: #63bdb2; width: 13px; margin: -5px 0; border-radius: 2px; }
QSplitter::handle { background: #26384a; }
QProgressBar { background: #101923; border: 1px solid #3b5065; color: #e0e8f0; text-align: center; border-radius: 2px; }
QProgressBar::chunk { background: #2b857c; }
QLabel#subtleLabel { color: #9aafc2; }
"""


@dataclass(frozen=True)
class ShownResult:
    root: Path
    task: dict
    attempt: dict

    @property
    def label(self) -> str:
        selected = self.task.get("selection", {}).get("layer_count", "—")
        sampling = self.task["sampling"]
        if sampling["mode"] == "stride":
            detail = f" · 步长 {sampling['step']} / 偏移 {sampling['offset']}"
        elif sampling["mode"] == "range":
            detail = f" · Z {sampling['start']}–{sampling['end']}"
        else:
            detail = ""
        return f"{self.task['preset']} · {selected} 层{detail} · {self.task['status']}"

    def file(self, field: str) -> Path | None:
        value = self.attempt.get(field)
        return artifact_path(self.root, value) if value else None


class MainWindow(QMainWindow):
    def __init__(self, settings: Settings):
        super().__init__()
        self.settings = settings
        self.engine = FocusStackEngine(settings.engine)
        self.runner = ExperimentRunner(self.engine, settings.artifacts_dir)
        self.capabilities: EngineCapabilities | None = None
        self.dataset: Dataset | None = None
        self.stack: Stack | None = None
        self.runs: dict[Path, dict] = {}
        self.references: dict[str, dict] = {}
        self.results: list[ShownResult] = []
        self.selected_result: ShownResult | None = None
        self._on_done = None
        self._closing = False
        self._work_pool = QThreadPool(self)
        self._work_pool.setMaxThreadCount(1)
        self._preview_pool = QThreadPool(self)
        self._preview_pool.setMaxThreadCount(2)
        self.work = WorkSession(self._work_pool, self)
        self.work.progress.connect(self._progress)
        self.work.complete.connect(self._finished)
        self.work.busy_changed.connect(self._update_actions)
        self.cache = PreviewCache(settings.preview_cache_mib)
        self.setWindowTitle("Pollen-EDF-Lab 0.1 · 景深合成工作台")
        self.resize(1460, 920)
        self.setMinimumSize(1060, 680)
        self._build()
        self._update_actions()
        QTimer.singleShot(0, self.probe_engine)

    def _build(self) -> None:
        toolbar = QToolBar("实验操作", self)
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        self.a_folder = self._action(toolbar, "导入目录", self.import_folder, "Ctrl+O")
        self.a_index = self._action(toolbar, "导入索引", self.import_index)
        self.a_files = self._action(toolbar, "图片列表", self.import_files)
        self.a_reference = self._action(toolbar, "关联 Helicon", self.import_reference)
        toolbar.addSeparator()
        self.run_button = button("运行实验", self.run_selected)
        self.run_button.setObjectName("runButton")
        toolbar.addWidget(self.run_button)
        self.a_batch = self._action(toolbar, "批量对照", self.choose_batch)
        self.a_stop = self._action(toolbar, "停止", self.stop)
        toolbar.addSeparator()
        self.a_open = self._action(toolbar, "打开记录", self.open_experiment)
        self.a_resume = self._action(toolbar, "恢复与重试", self.resume_experiment)
        self.a_export = self._action(toolbar, "导出报告", self.export_report)
        self.a_review = self._action(toolbar, "记录审查", self.record_review)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(9, 8, 8, 8)
        self.dataset_label = QLabel("请选择本地焦栈或数据索引")
        self.dataset_label.setWordWrap(True)
        self.dataset_label.setObjectName("subtleLabel")
        left_layout.addWidget(self.dataset_label)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["XY 视野 / 实验", "层数 / 状态"])
        self.tree.header().setStretchLastSection(False)
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.tree.setAlternatingRowColors(True)
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.tree.currentItemChanged.connect(self._tree_changed)
        left_layout.addWidget(self.tree, 1)
        self.validate_button = button("校验所选视野", self.validate_view)
        left_layout.addWidget(self.validate_button)

        self.tabs = QTabWidget()
        self.compare = Comparison(self._preview_pool, self.cache)
        self.compare.message.connect(self._message)
        self.compare.metrics_requested.connect(self.compare_metrics)
        self.compare.left_choice.currentIndexChanged.connect(self._comparison_selected)
        self.compare.right_choice.currentIndexChanged.connect(self._comparison_selected)
        self.tabs.addTab(self.compare, "结果比对")
        original = QWidget()
        raw_layout = QVBoxLayout(original)
        raw_layout.setContentsMargins(8, 7, 8, 8)
        line = QHBoxLayout()
        self.raw = ImageViewer(self._preview_pool, self.cache, "导入数据后浏览原始焦层")
        self.raw.load_failed.connect(self._message)
        self.raw_label = QLabel("—")
        self.raw_label.setObjectName("subtleLabel")
        line.addWidget(button("适应", self.raw.fit_image))
        line.addWidget(button("1:1", self.raw.actual_size))
        line.addWidget(button("框选 ROI", self.raw.set_roi_mode, checkable=True))
        self.curve_button = button("ROI 清晰度曲线", self.show_curve)
        line.addWidget(self.curve_button)
        line.addStretch()
        line.addWidget(self.raw_label)
        raw_layout.addLayout(line)
        raw_layout.addWidget(self.raw, 1)
        slider_line = QHBoxLayout()
        slider_line.addWidget(QLabel("Z 焦层"))
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(5)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._z_changed)
        slider_line.addWidget(self.slider, 1)
        self.z_position = QLabel("—")
        self.z_position.setMinimumWidth(125)
        slider_line.addWidget(self.z_position)
        raw_layout.addLayout(slider_line)
        self.tabs.addTab(original, "原始焦层")
        self.depth = ImageViewer(self._preview_pool, self.cache, "选中带有焦层图的实验结果")
        self.depth.load_failed.connect(self._message)
        self.tabs.addTab(self.depth, "焦层图")
        self.tabs.currentChanged.connect(self._tab_changed)
        self.raw.image_changed.connect(self._raw_info)
        self.raw.roi_changed.connect(self._raw_info)
        self.slider_timer = QTimer(self, interval=65, singleShot=True)
        self.slider_timer.timeout.connect(self.raw.start_source)
        for key, delta in (("[", -1), ("]", 1)):
            shortcut = QShortcut(QKeySequence(key), original)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(lambda step=delta: self.slider.setValue(self.slider.value() + step))

        self.parameters = Parameters(self.settings.timeout_s)
        parameter_scroll = QScrollArea()
        parameter_scroll.setWidgetResizable(True)
        parameter_scroll.setWidget(self.parameters)
        parameter_scroll.setMinimumWidth(286)
        parameter_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        panels = QSplitter()
        panels.setChildrenCollapsible(False)
        panels.addWidget(left)
        panels.addWidget(self.tabs)
        panels.addWidget(parameter_scroll)
        panels.setSizes([243, 908, 306])
        panels.setStretchFactor(0, 0)
        panels.setStretchFactor(1, 1)
        panels.setStretchFactor(2, 0)

        self.bottom = QTabWidget()
        self.activity = QPlainTextEdit()
        self.activity.setReadOnly(True)
        self.activity.setMaximumBlockCount(900)
        self.metrics = QPlainTextEdit()
        self.metrics.setReadOnly(True)
        self.bottom.addTab(self.activity, "进度与日志")
        self.bottom.addTab(self.metrics, "耗时、内存与指标")
        split = QSplitter(Qt.Orientation.Vertical)
        split.setChildrenCollapsible(False)
        split.addWidget(panels)
        split.addWidget(self.bottom)
        split.setSizes([720, 140])
        split.setStretchFactor(0, 1)
        self.setCentralWidget(split)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setFixedWidth(102)
        self.progress.setFormat("%v/%m")
        self.engine_label = QLabel("正在检测引擎…")
        self.statusBar().addPermanentWidget(self.progress)
        self.statusBar().addPermanentWidget(self.engine_label)
        detect = button("检测引擎", self.probe_engine)
        self.statusBar().addPermanentWidget(detect)
        self.detect_button = detect
        self._rebuild_results()

    def _action(self, toolbar, name: str, callback, shortcut: str | None = None) -> QAction:
        action = QAction(name, self)
        action.triggered.connect(callback)
        if shortcut:
            action.setShortcut(shortcut)
        toolbar.addAction(action)
        return action

    def _update_actions(self, _busy=None) -> None:
        ready, view = not self.work.busy, self.stack is not None
        for action in (self.a_folder, self.a_index, self.a_files, self.a_open):
            action.setEnabled(ready)
        for action in (self.a_reference, self.validate_button, self.curve_button):
            action.setEnabled(ready and view)
        self.detect_button.setEnabled(ready)
        self.run_button.setEnabled(ready and view and self.capabilities is not None)
        self.a_batch.setEnabled(self.run_button.isEnabled())
        self.a_resume.setEnabled(ready and self.capabilities is not None)
        self.a_stop.setEnabled(not ready)
        self.a_export.setEnabled(ready and bool(self.runs))
        self.a_review.setEnabled(ready and self.selected_result is not None and self.selected_result.task["status"] == "success")

    def _log(self, message: str) -> None:
        self.activity.appendPlainText(message)

    def _message(self, message: str) -> None:
        self.statusBar().showMessage(message.splitlines()[0], 9000)
        self._log(message)

    def _start(self, label: str, operation, callback) -> None:
        if self.work.busy:
            self._message("已有后台任务，当前可以停止任务。")
            return
        self._on_done = callback
        self.progress.setRange(0, 0)
        self._log(label)
        self.statusBar().showMessage(label)
        self.work.start(operation)

    @Slot(object, object)
    def _finished(self, value, error) -> None:
        if self._closing:
            QTimer.singleShot(0, self.close)
            return
        callback, self._on_done = self._on_done, None
        if self.progress.maximum() == 0:
            self.progress.setRange(0, 1)
            self.progress.setValue(0 if error else 1)
        if error is not None:
            self._message(str(error))
            if callback == self._engine_ready:
                self._engine_ready(None)
        elif callback:
            self.statusBar().clearMessage()
            callback(value)

    @Slot(object)
    def _progress(self, update: ExperimentUpdate) -> None:
        self.progress.setRange(0, update.total)
        self.progress.setValue(update.finished)
        self.statusBar().showMessage(update.message)
        if any(update.message.endswith(suffix) for suffix in ("· 开始", "· success", "· failed", "· timed_out", "· cancelled")):
            self._log(update.message)

    def probe_engine(self) -> None:
        self._start("检测 focus-stack 引擎", lambda _cancel, _progress: self.engine.validate_environment(), self._engine_ready)

    def _engine_ready(self, capabilities: EngineCapabilities | None) -> None:
        self.capabilities = capabilities
        self.parameters.set_capabilities(capabilities)
        if capabilities:
            status = "可请求 OpenCL" if capabilities.opencl_built else "OpenCL 不可用"
            self.engine_label.setText(f"focus-stack {capabilities.release} · {status}")
            self.engine_label.setToolTip(f"{capabilities.version}\n{capabilities.executable}\n{capabilities.binary_sha256}")
            self._log(f"{capabilities.version} · {status}")
        else:
            self.engine_label.setText("引擎不可用")
        self._update_actions()

    def import_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择单视野或批量采集目录")
        if path:
            self.import_source(Path(path))

    def import_index(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择数据索引", filter="JSON 或 CSV (*.json *.csv)")
        if path:
            self.import_source(Path(path))

    def import_source(self, path: Path) -> None:
        self._import(lambda cancel: load_dataset(path, cancel=cancel))

    def import_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "选择同一视野的焦层图片",
                                                filter="图像 (*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.webp)")
        if not paths:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("确认图片顺序")
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("默认使用文件名中的 Z 编号。可拖动列表并确认手动排序。"))
        entries = QListWidget()
        entries.addItems(paths)
        entries.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        layout.addWidget(entries)
        explicit = QCheckBox("按此列表顺序生成 Z 编号")
        layout.addWidget(explicit)
        buttons = self._dialog_buttons(dialog)
        layout.addWidget(buttons)
        dialog.resize(690, 480)
        if dialog.exec():
            files = tuple(Path(entries.item(index).text()) for index in range(entries.count()))
            ordered = explicit.isChecked()
            self._import(lambda cancel: manual_dataset(files, explicit_order=ordered, cancel=cancel))

    @staticmethod
    def _dialog_buttons(dialog: QDialog) -> QDialogButtonBox:
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("确认")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        return buttons

    def _import(self, loader) -> None:
        def operation(cancel, _notify):
            dataset = loader(cancel)
            dataset_id = dataset.record()["id"]
            runs = {}
            for path in sorted(self.settings.artifacts_dir.glob("experiment_*/manifest.json")):
                check_cancel(cancel)
                try:
                    manifest = read_json(path)
                    if manifest.get("schema") == 1 and manifest["dataset"]["id"] == dataset_id:
                        runs[absolute_path(path.parent)] = manifest
                except (OSError, ValueError, KeyError):
                    continue
            refs = self._references_for(dataset, dataset_id, runs, cancel)
            return dataset, runs, refs

        self._start("导入数据与实验记录", operation, self._apply_import)

    def _references_for(self, dataset: Dataset, dataset_id: str, runs: dict, cancel) -> dict:
        refs = {key: value for manifest in runs.values() for key, value in manifest.get("helicon_references", {}).items()}
        for stack in dataset.stacks:
            check_cancel(cancel)
            if local := get_reference(self.settings.artifacts_dir, dataset_id, stack.id):
                refs[stack.id] = local
        return refs

    def _apply_import(self, value) -> None:
        self.dataset, self.runs, self.references = value
        self.stack = None
        self._rebuild_tree()
        self.tabs.setCurrentIndex(1)
        self._message(f"已导入 {len(self.dataset.stacks)} 个视野。")

    def _rebuild_tree(self) -> None:
        if not self.dataset:
            return
        selected_view = self.stack.id if self.stack else None
        chosen = None
        dataset = self.dataset
        total = sum(len(stack.frames) for stack in dataset.stacks)
        tag = " · SYNTHETIC" if dataset.provenance == "SYNTHETIC" and "SYNTHETIC" not in dataset.name else ""
        self.dataset_label.setText(f"{dataset.name}{tag}\n{len(dataset.stacks)} 视野 · {total} 张图像")
        self.tree.blockSignals(True)
        self.tree.clear()
        for stack in dataset.stacks:
            item = QTreeWidgetItem([stack.id, str(len(stack.frames))])
            item.setData(0, Qt.ItemDataRole.UserRole, stack.id)
            self.tree.addTopLevelItem(item)
            if stack.id == selected_view or chosen is None:
                chosen = item
            for root, manifest in self.runs.items():
                for task in manifest["tasks"]:
                    if task["xy"] != stack.id:
                        continue
                    child = QTreeWidgetItem([task["preset"], task["status"]])
                    child.setToolTip(0, f"{root.name} / {task['id']}")
                    child.setData(0, Qt.ItemDataRole.UserRole, stack.id)
                    child.setData(0, Qt.ItemDataRole.UserRole + 1, (str(root), task["id"]))
                    item.addChild(child)
        if chosen:
            self.tree.setCurrentItem(chosen)
            chosen.setExpanded(True)
        self.tree.blockSignals(False)
        self._tree_changed(chosen, None)

    def _tree_changed(self, current, _old) -> None:
        if not current or not self.dataset:
            return
        view = current.data(0, Qt.ItemDataRole.UserRole)
        stack = next(stack for stack in self.dataset.stacks if stack.id == view)
        if stack is not self.stack:
            self.stack = stack
            self.slider.blockSignals(True)
            self.slider.setRange(0, len(stack.frames) - 1)
            self.slider.setValue(0)
            self.slider.setEnabled(len(stack.frames) > 1)
            self.slider.blockSignals(False)
            self.raw.set_roi(None)
            self._z_changed(0, preserve=False)
            self.slider_timer.stop()
            self.raw.start_source()
        target = current.data(0, Qt.ItemDataRole.UserRole + 1)
        self._rebuild_results(target)
        if target:
            self.tabs.setCurrentIndex(0)
        self._update_actions()

    def _z_changed(self, index: int, preserve: bool = True) -> None:
        if self.stack:
            frame = self.stack.frames[index]
            physical = f" / {frame.z_um:g} µm" if frame.z_um is not None else ""
            self.z_position.setText(f"Z {frame.z}{physical}    {index + 1}/{len(self.stack.frames)}")
            self.raw.set_source(frame.path, original=True, preserve=preserve, deferred=True)
            self.slider_timer.start()

    def _raw_info(self, _value=None) -> None:
        size, roi = self.raw.size_pixels, self.raw.roi
        dimensions = f"{size[0]} × {size[1]}" if size else "—"
        self.raw_label.setText(f"{dimensions}   ·   ROI ({roi.x}, {roi.y}, {roi.width}, {roi.height})" if roi else dimensions)

    def _tab_changed(self, index: int) -> None:
        if index == 2:
            self.depth.set_source(self.selected_result.file("depth") if self.selected_result else None)

    def _rebuild_results(self, target=None) -> None:
        view = self.stack.id if self.stack else None
        self.results = [ShownResult(root, task, task["attempts"][-1] if task["attempts"] else {})
                        for root, manifest in sorted(self.runs.items(), reverse=True)
                        for task in manifest["tasks"] if task["xy"] == view]
        left, right = self.compare.left_choice, self.compare.right_choice
        for choice in (left, right):
            choice.blockSignals(True)
            choice.clear()
        left.addItem("选择左侧图像", None)
        right.addItem("选择实验结果", None)
        reference = self.references.get(view)
        if reference:
            left.addItem("Helicon 参考" + (f" · {reference['version']}" if reference["version"] else ""), "helicon")
        successful = []
        for index, entry in enumerate(self.results):
            tooltip = f"{entry.root.name} / {entry.task['id']}\n{entry.label}"
            if entry.file("output") and entry.task["status"] == "success":
                left.addItem(entry.label, index)
                left.setItemData(left.count() - 1, tooltip, Qt.ItemDataRole.ToolTipRole)
                successful.append(index)
            right.addItem(entry.label, index)
            right.setItemData(right.count() - 1, tooltip, Qt.ItemDataRole.ToolTipRole)
        selected = next((index for index, item in enumerate(self.results)
                         if target and (str(item.root), item.task["id"]) == tuple(target)), None)
        if selected is None and self.results:
            selected = successful[0] if successful else 0
        right.setCurrentIndex(right.findData(selected) if selected is not None else 0)
        initial = next((index for index in successful if index != selected), None)
        left.setCurrentIndex(1 if reference else left.findData(initial) if initial is not None else 0)
        for choice in (left, right):
            choice.blockSignals(False)
        self._comparison_selected()

    def _comparison_selected(self, _index=None) -> None:
        if not hasattr(self, "metrics"):
            return
        selected = self.compare.right_choice.currentData()
        self.selected_result = self.results[selected] if isinstance(selected, int) else None
        left_index = self.compare.left_choice.currentData()
        view = self.stack.id if self.stack else None
        if left_index == "helicon" and view in self.references:
            left = Path(self.references[view]["path"])
        else:
            left = self.results[left_index].file("output") if isinstance(left_index, int) else None
        right = self.selected_result.file("output") if self.selected_result and self.selected_result.task["status"] == "success" else None
        self.compare.set_sources(left, right)
        depth = self.selected_result.file("depth") if right else None
        self.depth.set_source(depth if self.tabs.currentIndex() == 2 else None)
        self._result_metrics()
        self._update_actions()

    def _result_metrics(self) -> None:
        entry = self.selected_result
        if not entry:
            self.metrics.clear()
            return
        attempt = entry.attempt
        elapsed, rss = attempt.get("elapsed_s"), attempt.get("peak_rss_bytes")
        elapsed_text = f"{elapsed:.3f} s" if elapsed is not None else "—"
        rss_text = f"{rss / 1048576:.1f} MiB" if rss is not None else "—"
        text = (f"{entry.task['xy']}  ·  {entry.label}\n"
                f"原生进程 {elapsed_text}    采样峰值 RSS {rss_text}\n"
                f"执行证据 {attempt.get('acceleration', '—')}\n"
                f"{entry.root.name}\n")
        if attempt.get("metrics"):
            text += "\n" + self._format_metrics(attempt["metrics"])
        if attempt.get("diagnostic"):
            text += "\n" + attempt["diagnostic"]
        self.metrics.setPlainText(text)
        result_log = entry.file("log")
        if result_log and (tail := log_tail(result_log, 6200)):
            self._log(f"{entry.label} · 原生日志末尾\n{tail.rstrip()}")

    @staticmethod
    def _format_metrics(metrics: dict) -> str:
        label = f"{metrics['width']} × {metrics['height']} · "
        label += f"ROI {metrics['roi']}" if metrics.get("roi") else "全图"
        lines = [label]
        for key, name in METRIC_LABELS.items():
            value = metrics.get(key)
            display = "—" if value is None else f"{value:.6g}"
            lines.append(f"{name}   {display}")
        return "\n".join(lines)

    def validate_view(self) -> None:
        stack = self.stack
        if not stack:
            return

        def display(result) -> None:
            size = f"{result.size[0]} × {result.size[1]}" if result.size else "无可读图像"
            info = [f"{stack.id} · {len(stack.frames)} 层 · {size}", str(result.formats), *result.warnings, *result.errors]
            self._message("\n".join(info))

        self._start(f"校验 {stack.id}", lambda cancel, _notify: validate_stack(stack, cancel=cancel), display)

    def import_reference(self) -> None:
        if not self.dataset or not self.stack:
            return
        path, _ = QFileDialog.getOpenFileName(self, "选择已有 Helicon 合成图",
                                              filter="图像 (*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.webp)")
        if not path:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("记录 Helicon 参考来源")
        form = QFormLayout(dialog)
        version, parameters = QLineEdit(), QPlainTextEdit()
        parameters.setPlaceholderText("方法、半径、平滑度或其他已知参数")
        parameters.setFixedHeight(74)
        form.addRow("版本（可选）", version)
        form.addRow("参数（可选）", parameters)
        form.addRow(self._dialog_buttons(dialog))
        if not dialog.exec():
            return
        dataset, view = self.dataset, self.stack.id
        metadata = version.text().strip(), parameters.toPlainText().strip()

        def operation(cancel, _notify):
            check_cancel(cancel)
            ensure_separate_output(self.settings.artifacts_dir, dataset)
            return attach_reference(self.settings.artifacts_dir, dataset.record()["id"], view, Path(path), *metadata)

        def done(reference) -> None:
            self.references[view] = reference
            self._rebuild_results()
            self.tabs.setCurrentIndex(0)
            self._message(f"已关联 {view} 的 Helicon 参考图。")

        self._start(f"读取 {view} 的参考图", operation, done)

    def run_selected(self) -> None:
        if self.stack:
            self._run_experiment([self.stack.id])

    def choose_batch(self) -> None:
        if not self.dataset:
            return
        all_views = [stack.id for stack in self.dataset.stacks]
        selected = list(dict.fromkeys(item.data(0, Qt.ItemDataRole.UserRole) for item in self.tree.selectedItems()))
        dialog = QDialog(self)
        dialog.setWindowTitle("批量参数对照")
        layout = QVBoxLayout(dialog)
        include_all = QCheckBox(f"处理全部 {len(all_views)} 个视野")
        include_all.setChecked(len(selected) < 2)
        layout.addWidget(include_all)
        for index, preset in enumerate(PRESETS):
            option = QCheckBox(preset)
            option.setEnabled(self.parameters.preset.model().item(index).isEnabled())
            option.setChecked(preset == self.parameters.preset.currentData())
            layout.addWidget(option)
        layout.addWidget(QLabel("抽层策略和手动调整将应用到所选预设。"))
        layout.addWidget(self._dialog_buttons(dialog))
        if dialog.exec():
            presets = [option.text() for option in dialog.findChildren(QCheckBox) if option is not include_all and option.isChecked()]
            if presets:
                self._run_experiment(all_views if include_all.isChecked() else selected, presets)
            else:
                self._message("请选择至少一个预设。")

    def _run_experiment(self, views: list[str], presets: list[str] | None = None) -> None:
        dataset = self.dataset
        if not dataset or not views:
            return
        try:
            trials = self.parameters.trials(presets)
            timeout = float(self.parameters.timeout.text())
        except ValueError as error:
            self._message(str(error))
            return
        self._start(f"准备实验 · {len(views)} 视野 × {len(trials)} 组合",
                    lambda cancel, notify: self._read_run(self.runner.run(dataset, views, trials, timeout, cancel, notify)),
                    self._add_run)

    def stop(self) -> None:
        self.work.stop()
        self.statusBar().showMessage("正在取消并回收任务…")

    @staticmethod
    def _read_run(root: Path) -> tuple[Path, dict]:
        manifest = read_json(root / "manifest.json")
        if manifest.get("schema") != 1 or "dataset" not in manifest:
            raise ValueError("实验记录格式不受支持。")
        return root, manifest

    def _add_run(self, value: tuple[Path, dict]) -> None:
        root, manifest = value
        if not self.dataset or self.dataset.record()["id"] != manifest["dataset"]["id"]:
            self.dataset = Dataset.from_record(manifest["dataset"])
            self.stack, self.runs, self.references = None, {}, {}
        self.runs[root] = manifest
        for view, ref in manifest.get("helicon_references", {}).items():
            self.references.setdefault(view, ref)
        self._rebuild_tree()
        if self.stack:
            matching = [task for task in manifest["tasks"] if task["xy"] == self.stack.id]
            if matching:
                self._rebuild_results((str(root), matching[0]["id"]))
        self.tabs.setCurrentIndex(0)
        self._message(f"实验 {manifest['status']} · {root}")

    def open_experiment(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "打开已有实验记录", str(self.settings.artifacts_dir), "manifest.json (manifest.json)")
        if path:
            root = absolute_path(Path(path).parent)
            self._start("打开实验记录", lambda _cancel, _notify: self._read_run(root), self._add_run)

    def resume_experiment(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择需恢复的实验目录", str(self.settings.artifacts_dir))
        if path:
            self._start("恢复实验", lambda cancel, notify: self._read_run(self.runner.resume(Path(path), cancel, notify)), self._add_run)

    def export_report(self) -> None:
        root = self.selected_result.root if self.selected_result else next(reversed(self.runs), None)
        if root is None or not self.dataset:
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出离线 HTML 报告", str(root / "report.html"), "HTML (*.html)")
        if not path:
            return
        destination = Path(path)
        if destination.suffix.lower() != ".html":
            destination = Path(str(destination) + ".html")
        dataset = self.dataset
        manifest = self.runs[root]

        def export(_cancel, _notify):
            ensure_separate_output(destination, dataset)
            return write_report(root, manifest, destination)

        self._start("导出报告", export, lambda saved: self._message(f"报告已保存 · {saved}"))

    def compare_metrics(self) -> None:
        selected = [(side, viewer.source, viewer.roi) for side, viewer in (("左图", self.compare.left), ("右图", self.compare.right))
                    if viewer.source and viewer.size_pixels]
        if not selected:
            self._message("请先载入比较图像。")
            return

        def operation(cancel, _notify):
            result = []
            for side, path, roi in selected:
                check_cancel(cancel)
                result.append(f"{side} · {path.name}\n{self._format_metrics(measure_file(path, roi))}")
            return "\n\n".join(result)

        def show(text) -> None:
            self.metrics.setPlainText(text)
            self.bottom.setCurrentWidget(self.metrics)

        self._start("计算真实像素的图像指标", operation, show)

    def show_curve(self) -> None:
        stack, roi = self.stack, self.raw.roi
        if not stack or not roi:
            self._message("请先在原始焦层中框选 ROI。")
            return
        self._start(f"计算 {stack.id} 的 ROI 清晰度曲线", lambda cancel, _notify: focus_curve(stack, roi, cancel),
                    lambda samples: self._display_curve(stack, roi, samples))

    def _display_curve(self, stack, roi, samples) -> None:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        physical = all(point["z_um"] is not None for point in samples)
        x = [point["z_um"] if physical else point["z"] for point in samples]
        dialog = QDialog(self)
        dialog.setWindowTitle(f"{stack.id} · ROI 清晰度曲线")
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel(f"ROI ({roi.x}, {roi.y}) · {roi.width} × {roi.height} px"))
        figure = Figure(figsize=(8, 6), layout="constrained")
        axes = figure.subplots(3, 1, sharex=True)
        for axis, metric in zip(axes, ("laplacian_variance", "tenengrad", "sml")):
            axis.plot(x, [point[metric] for point in samples], color="#176c72", linewidth=1.45, marker=".")
            axis.set_ylabel(METRIC_LABELS[metric])
            axis.grid(alpha=0.2)
        axes[-1].set_xlabel("Z (um)" if physical else "Z index")
        layout.addWidget(FigureCanvasQTAgg(figure))
        dialog.resize(780, 650)
        dialog.exec()

    def record_review(self) -> None:
        entry = self.selected_result
        if not entry:
            return
        note, accepted = QInputDialog.getMultiLineText(self, "记录人工 ROI 审查", "观察到的纹理、形态或边缘特征")
        if not accepted or not note.strip():
            return
        cited = [(side, viewer.source, viewer.roi) for side, viewer in (("left", self.compare.left), ("right", self.compare.right))
                 if viewer.source and viewer.size_pixels]
        mapping = asdict(self.compare.mapping) if self.compare.mapping else None

        def operation(cancel, _notify):
            images = {}
            for side, path, roi in cited:
                images[side] = {"path": str(path), "sha256": sha256(path, cancel),
                                "roi": asdict(roi) if roi else None, "metrics": measure_file(path, roi)}
            path = save_review(entry.root, {"task": entry.task["id"], "xy": entry.task["xy"], "note": note,
                                           "images": images, "roi_mapping": mapping})
            write_report(entry.root, read_json(entry.root / "manifest.json"))
            return path

        self._start("保存人工 ROI 审查", operation, lambda path: self._message(f"ROI 审查已保存 · {path}"))

    def closeEvent(self, event) -> None:
        if self.work.busy:
            self._closing = True
            self.stop()
            event.ignore()
            return
        self.slider_timer.stop()
        for viewer in (self.raw, self.compare.left, self.compare.right, self.depth):
            viewer.finish()
        super().closeEvent(event)


def launch(settings: Settings) -> int:
    application = QApplication([])
    application.setApplicationName("Pollen-EDF-Lab")
    application.setStyle("Fusion")
    application.setFont(QFont("Microsoft YaHei UI", 9))
    application.setStyleSheet(STYLE)
    window = MainWindow(settings)
    window.show()
    return application.exec()
