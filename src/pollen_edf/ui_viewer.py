from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QRectF, QThreadPool, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QTransform
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsScene, QGraphicsView, QLabel

from .evaluation import ROI
from .images import absolute_path, check_cancel, normalized_image, read_image
from .ui_workers import LatestTask


@dataclass(frozen=True)
class Preview:
    image: QImage
    width: int
    height: int
    full: bool


class PreviewCache:
    def __init__(self, capacity_mib: int):
        capacity = capacity_mib * 1024 * 1024
        self.limits = {False: capacity // 5, True: capacity * 4 // 5}
        self.images = {False: OrderedDict(), True: OrderedDict()}
        self.used = {False: 0, True: 0}
        self.lock = Lock()

    def decode(self, path: Path, full: bool, original: bool, cancel: Event) -> Preview:
        check_cancel(cancel)
        path = absolute_path(path)
        stat = path.stat()
        key = (path, original, stat.st_size, stat.st_mtime_ns)
        cache = self.images[full]
        with self.lock:
            previous = cache.get(key)
            if previous:
                cache.move_to_end(key)
                return previous
        image = read_image(path, engine_input=original)
        height, width = image.shape[:2]
        if not full and max(width, height) > 1400:
            factor = 1400 / max(width, height)
            image = cv2.resize(image, (max(1, round(width * factor)), max(1, round(height * factor))),
                               interpolation=cv2.INTER_AREA)
        if image.dtype != np.uint8:
            image = normalized_image(image).astype(np.uint8)
        if image.ndim == 2:
            mode = QImage.Format.Format_Grayscale8
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB if image.shape[2] == 4 else cv2.COLOR_BGR2RGB)
            mode = QImage.Format.Format_RGB888
        image = np.ascontiguousarray(image)
        qimage = QImage(image.data, image.shape[1], image.shape[0], image.strides[0], mode).copy()
        preview = Preview(qimage, width, height, full)
        check_cancel(cancel)
        cost = qimage.sizeInBytes()
        if cost <= self.limits[full]:
            with self.lock:
                previous = cache.pop(key, None)
                if previous:
                    self.used[full] -= previous.image.sizeInBytes()
                while self.used[full] + cost > self.limits[full]:
                    _, removed = cache.popitem(last=False)
                    self.used[full] -= removed.image.sizeInBytes()
                cache[key] = preview
                self.used[full] += cost
        return preview


class ImageViewer(QGraphicsView):
    camera_changed = Signal(object)
    roi_changed = Signal(object)
    image_changed = Signal()
    load_failed = Signal(str)

    def __init__(self, pool: QThreadPool, cache: PreviewCache, placeholder: str, parent=None):
        super().__init__(parent)
        self.cache = cache
        self.placeholder = placeholder
        self.source: Path | None = None
        self.original = False
        self.size_pixels: tuple[int, int] | None = None
        self.roi: ROI | None = None
        self._roi_mode = False
        self._anchor = None
        self._installing = False
        self._preview: Preview | None = None
        self._wanted_full = False
        self._load_pending = False
        self._saved_camera = None
        self._camera = None
        self.setScene(QGraphicsScene(self))
        self.pixels = QGraphicsPixmapItem()
        self.scene().addItem(self.pixels)
        outline = QPen(QColor("#41d9bf"), 1.5)
        outline.setCosmetic(True)
        self.roi_item = QGraphicsRectItem()
        self.roi_item.setPen(outline)
        self.roi_item.setZValue(1)
        self.roi_item.hide()
        self.scene().addItem(self.roi_item)
        self.label = QLabel(placeholder, self.viewport())
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.label.setStyleSheet("color: #8597aa; background: transparent; font-size: 13px;")
        self.setBackgroundBrush(QColor("#0b1119"))
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setRenderHints(QPainter.RenderHint.SmoothPixmapTransform)
        self.setMinimumSize(160, 150)
        self.task = LatestTask(pool, self)
        self.task.complete.connect(self._loaded)
        self.zoom_timer = QTimer(self, singleShot=True, interval=100)
        self.zoom_timer.timeout.connect(self._maybe_full)

    def set_source(self, path: Path | None, *, original: bool = False, preserve: bool = False,
                   deferred: bool = False) -> None:
        if self.source == path and self.original == original:
            if not deferred:
                self.start_source()
            return
        self.task.invalidate()
        self.zoom_timer.stop()
        self._saved_camera = self._camera if preserve else None
        self._camera = self._saved_camera
        self.source, self.original = path, original
        self.size_pixels = None
        self._preview = None
        self._wanted_full = False
        self._load_pending = path is not None
        self._anchor = None
        self.pixels.setPixmap(QPixmap())
        self.roi_item.hide()
        if not preserve:
            self.set_roi(None)
        self.label.setText("正在加载图像…" if path else self.placeholder)
        self.label.show()
        self.image_changed.emit()
        if not deferred:
            self.start_source()

    def start_source(self) -> None:
        if self._load_pending:
            self._load_pending = False
            self._request(full=False)

    def _request(self, full: bool) -> None:
        source, original = self.source, self.original
        self.task.submit(lambda cancel, _notify: self.cache.decode(source, full, original, cancel))

    @Slot(object, object)
    def _loaded(self, preview: Preview | None, error: Exception | None) -> None:
        if error:
            if self._preview is None:
                self.label.setText("图像加载失败")
            self.load_failed.emit(str(error))
            return
        assert preview is not None
        before = self._camera or self._saved_camera
        self._installing = True
        try:
            self._preview = preview
            self.size_pixels = preview.width, preview.height
            pixmap = QPixmap.fromImage(preview.image)
            self.pixels.setPixmap(pixmap)
            self.pixels.setTransform(QTransform.fromScale(preview.width / pixmap.width(), preview.height / pixmap.height()))
            self.setSceneRect(0, 0, preview.width, preview.height)
            if before:
                self.apply_camera(before)
            else:
                self.fit_image()
            if self.roi and not self.roi.fits(preview.width, preview.height):
                self.set_roi(None)
            else:
                self._draw_roi()
        finally:
            self._installing = False
        self.label.hide()
        self._capture_camera(emit=False)
        self.image_changed.emit()

    def _maybe_full(self) -> None:
        preview = self._preview
        if (preview and not preview.full and not self._wanted_full
                and preview.image.width() != preview.width
                and self.transform().m11() * self.viewport().devicePixelRatioF() > 1.06 * preview.image.width() / preview.width):
            self._wanted_full = True
            self._request(full=True)

    def finish(self) -> None:
        self.zoom_timer.stop()
        self.task.invalidate()

    def fit_image(self) -> None:
        if self.size_pixels:
            self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self._capture_camera()

    def actual_size(self) -> None:
        if self.size_pixels:
            x, y, _ = self._camera or (0.5, 0.5, 1)
            self.apply_camera((x, y, 1 / self.viewport().devicePixelRatioF()))
            self._capture_camera()

    def apply_camera(self, camera: tuple[float, float, float]) -> None:
        if self.size_pixels:
            x, y, scale = camera
            previous, self._installing = self._installing, True
            try:
                self.setTransform(QTransform.fromScale(scale, scale))
                self.centerOn(x * self.size_pixels[0], y * self.size_pixels[1])
            finally:
                self._installing = previous
            self._capture_camera(emit=False)

    def _capture_camera(self, *, emit: bool = True) -> None:
        if self.size_pixels:
            center = self.mapToScene(self.viewport().rect().center())
            x = min(1.0, max(0.0, center.x() / self.size_pixels[0]))
            y = min(1.0, max(0.0, center.y() / self.size_pixels[1]))
            self._camera = (x, y, self.transform().m11())
            if emit and not self._installing:
                self.camera_changed.emit(self._camera)
            self.zoom_timer.start()

    def _draw_roi(self) -> None:
        if self.roi and self.size_pixels:
            self.roi_item.setRect(self.roi.x, self.roi.y, self.roi.width, self.roi.height)
            self.roi_item.show()
        else:
            self.roi_item.hide()

    def set_roi(self, roi: ROI | None) -> None:
        if roi and (self.size_pixels is None or not roi.fits(*self.size_pixels)):
            raise ValueError("ROI 超出了当前原图有效区域。")
        self.roi = roi
        self._draw_roi()
        self.roi_changed.emit(roi)

    def set_roi_mode(self, enabled: bool) -> None:
        self._roi_mode = enabled
        self.setDragMode(QGraphicsView.DragMode.NoDrag if enabled else QGraphicsView.DragMode.ScrollHandDrag)
        self.viewport().setCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.OpenHandCursor)

    def fit_roi(self) -> None:
        if self.roi:
            self.fitInView(self.roi_item.rect(), Qt.AspectRatioMode.KeepAspectRatio)
            self._capture_camera()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.label.setGeometry(self.viewport().rect())

    def scrollContentsBy(self, dx: int, dy: int) -> None:
        super().scrollContentsBy(dx, dy)
        if hasattr(self, "zoom_timer") and not self._installing:
            self._capture_camera()

    def wheelEvent(self, event) -> None:
        if self.size_pixels and event.angleDelta().y():
            factor = 1.2 ** (event.angleDelta().y() / 120)
            current = self.transform().m11()
            target = min(16, max(0.025, current * factor))
            self.scale(target / current, target / current)
            self._capture_camera()
            event.accept()
            return
        super().wheelEvent(event)

    def mousePressEvent(self, event) -> None:
        if self._roi_mode and self.size_pixels and event.button() == Qt.MouseButton.LeftButton:
            self._anchor = self.mapToScene(event.position().toPoint())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._anchor is not None:
            self.roi_item.setRect(self._drag_rect(event.position().toPoint()))
            self.roi_item.show()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._anchor is not None:
            rect = self._drag_rect(event.position().toPoint())
            self._anchor = None
            x, y = math.floor(rect.left()), math.floor(rect.top())
            roi = ROI(x, y, math.ceil(rect.right()) - x, math.ceil(rect.bottom()) - y)
            if roi.fits(*self.size_pixels):
                self.set_roi(roi)
            else:
                self._draw_roi()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _drag_rect(self, point: QPoint) -> QRectF:
        return QRectF(self._anchor, self.mapToScene(point)).normalized().intersected(self.sceneRect())
