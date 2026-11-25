from __future__ import annotations

import math
from pathlib import Path

from PySide6.QtCore import QThreadPool, Signal
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout,
                               QLabel, QLineEdit, QPushButton, QSplitter, QVBoxLayout, QWidget)

from .evaluation import ROI, ROIMapping
from .ui_viewer import ImageViewer, PreviewCache


def button(text: str, callback, *, checkable=False) -> QPushButton:
    widget = QPushButton(text)
    widget.setCheckable(checkable)
    widget.clicked.connect(callback)
    return widget


class Comparison(QWidget):
    message = Signal(str)
    metrics_requested = Signal()

    def __init__(self, pool: QThreadPool, cache: PreviewCache, parent=None):
        super().__init__(parent)
        self.left = ImageViewer(pool, cache, "关联 Helicon 或打开历史融合图")
        self.right = ImageViewer(pool, cache, "在此查看融合图")
        self.left_choice = QComboBox()
        self.right_choice = QComboBox()
        for choice in (self.left_choice, self.right_choice):
            choice.setMinimumContentsLength(14)
            choice.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.left_dimensions = QLabel()
        self.right_dimensions = QLabel()
        self.roi_label = QLabel("")
        self.mapping: ROIMapping | None = None
        self._moving = False
        self._selecting_roi = False
        self._mapping_roi = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 7, 8, 8)
        tools = QHBoxLayout()
        self.sync = button("同步视图", self._toggle_sync, checkable=True)
        self.sync.setChecked(True)
        tools.addWidget(self.sync)
        tools.addWidget(button("适应", self.fit))
        tools.addWidget(button("1:1", self.actual))
        tools.addWidget(button("框选 ROI", self._roi_mode, checkable=True))
        tools.addWidget(button("放大 ROI", self.fit_roi))
        tools.addWidget(button("ROI 对应", self.configure_mapping))
        tools.addStretch()
        tools.addWidget(button("读取指标", lambda: self.metrics_requested.emit()))
        layout.addLayout(tools)
        split = QSplitter()
        split.setChildrenCollapsible(False)
        split.addWidget(self._pane(self.left_choice, self.left_dimensions, self.left))
        split.addWidget(self._pane(self.right_choice, self.right_dimensions, self.right))
        split.setSizes([450, 450])
        layout.addWidget(split, 1)
        layout.addWidget(self.roi_label)
        self._show_roi()

        self.left.camera_changed.connect(lambda camera: self._sync_camera(self.right, camera))
        self.right.camera_changed.connect(lambda camera: self._sync_camera(self.left, camera))
        self.left.image_changed.connect(lambda: self._image_label(self.left_dimensions, self.left))
        self.right.image_changed.connect(lambda: self._image_label(self.right_dimensions, self.right))
        self.left.roi_changed.connect(self._left_roi)
        self.right.roi_changed.connect(self._right_roi)
        self.left.load_failed.connect(self.message)
        self.right.load_failed.connect(self.message)

    @staticmethod
    def _pane(choice, dimensions, viewer) -> QWidget:
        pane = QWidget()
        layout = QVBoxLayout(pane)
        layout.setContentsMargins(0, 0, 0, 0)
        line = QHBoxLayout()
        line.addWidget(choice, 1)
        line.addWidget(dimensions)
        layout.addLayout(line)
        layout.addWidget(viewer, 1)
        return pane

    @staticmethod
    def _image_label(label: QLabel, viewer: ImageViewer) -> None:
        size = viewer.size_pixels
        label.setText(f"{size[0]} × {size[1]}" if size else "")

    def set_sources(self, left: Path | None, right: Path | None) -> None:
        if left != self.left.source or right != self.right.source:
            self.mapping = None
            self._roi_mode(self._selecting_roi)
        self.left.set_source(left)
        self.right.set_source(right)
        self._show_roi()

    def _sync_camera(self, other: ImageViewer, camera: tuple[float, float, float]) -> None:
        if self.sync.isChecked() and not self._moving:
            self._moving = True
            try:
                other.apply_camera(camera)
            finally:
                self._moving = False

    def _toggle_sync(self, enabled: bool) -> None:
        if enabled and self.right._camera:
            self._sync_camera(self.left, self.right._camera)

    def fit(self) -> None:
        self._moving = True
        try:
            self.left.fit_image()
            self.right.fit_image()
        finally:
            self._moving = False

    def actual(self) -> None:
        self._moving = True
        try:
            self.left.actual_size()
            self.right.actual_size()
        finally:
            self._moving = False

    def fit_roi(self) -> None:
        self._moving = True
        try:
            self.left.fit_roi()
            self.right.fit_roi()
        finally:
            self._moving = False

    def _roi_mode(self, enabled: bool) -> None:
        self._selecting_roi = enabled
        self.left.set_roi_mode(enabled and self.mapping is None)
        self.right.set_roi_mode(enabled)

    def _left_roi(self, _roi: ROI | None) -> None:
        if self.mapping and not self._mapping_roi:
            self.mapping = None
            self._roi_mode(self._selecting_roi)
        self._show_roi()

    def _right_roi(self, roi: ROI | None) -> None:
        if self.mapping:
            self._mapping_roi = True
            try:
                self.left.set_roi(self.mapping.to_left(roi) if roi else None)
            except ValueError as error:
                self.left.set_roi(None)
                self.message.emit(str(error))
            finally:
                self._mapping_roi = False
        self._show_roi()

    def _show_roi(self, _unused=None) -> None:
        def format_roi(roi: ROI | None) -> str:
            return f"({roi.x}, {roi.y})  {roi.width} × {roi.height}" if roi else "全图"

        relation = "已指定 ROI 对应" if self.mapping else "独立 ROI"
        self.roi_label.setText(f"{relation}   ·   左 {format_roi(self.left.roi)}   ·   右 {format_roi(self.right.roi)}")

    def configure_mapping(self) -> None:
        if not self.left.size_pixels or not self.right.size_pixels:
            self.message.emit("载入两张图后可填写已知的像素对应关系。")
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("ROI 像素对应")
        form = QFormLayout(dialog)
        instruction = QLabel("从右图映射到左图。请按实际裁剪、偏移、比例和有效区域填写。")
        instruction.setWordWrap(True)
        form.addRow(instruction)
        mapping = self.mapping

        def roi_text(roi: ROI | None, size: tuple[int, int]) -> str:
            return f"{roi.x}, {roi.y}, {roi.width}, {roi.height}" if roi else f"0, 0, {size[0]}, {size[1]}"

        scale = QLineEdit(f"{mapping.scale_x}, {mapping.scale_y}" if mapping else "1, 1")
        offset = QLineEdit(f"{mapping.offset_x}, {mapping.offset_y}" if mapping else "0, 0")
        valid_right = QLineEdit(roi_text(mapping.valid_right if mapping else None, self.right.size_pixels))
        valid_left = QLineEdit(roi_text(mapping.valid_left if mapping else None, self.left.size_pixels))
        for label, entry in (("XY 比例", scale), ("XY 偏移 (px)", offset), ("右有效区 x,y,w,h", valid_right),
                             ("左有效区 x,y,w,h", valid_left)):
            form.addRow(label, entry)
        status = QLabel("")
        status.setWordWrap(True)
        form.addRow(status)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        clear = buttons.addButton("独立框选", QDialogButtonBox.ButtonRole.ResetRole)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("应用")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        form.addRow(buttons)

        def parse(text: str, convert):
            return [convert(value) for value in text.replace("，", ",").split(",")]

        def apply() -> None:
            try:
                sx, sy = parse(scale.text(), float)
                dx, dy = parse(offset.text(), float)
                right = ROI(*parse(valid_right.text(), int))
                left = ROI(*parse(valid_left.text(), int))
                if not right.fits(*self.right.size_pixels) or not left.fits(*self.left.size_pixels):
                    raise ValueError("有效区域必须在原图内。")
                if not all(math.isfinite(value) for value in (sx, sy, dx, dy)) or min(sx, sy) <= 0:
                    raise ValueError("请输入有效比例及偏移。")
                self.mapping = ROIMapping(sx, sy, dx, dy, right, left)
                self._roi_mode(self._selecting_roi)
                self._right_roi(self.right.roi)
            except (TypeError, ValueError) as error:
                status.setText(str(error))
                return
            dialog.accept()

        def reset() -> None:
            self.mapping = None
            self._roi_mode(self._selecting_roi)
            self._show_roi()
            dialog.accept()

        clear.clicked.connect(reset)
        buttons.accepted.connect(apply)
        buttons.rejected.connect(dialog.reject)
        dialog.setMinimumWidth(410)
        dialog.exec()
