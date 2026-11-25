from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from threading import Event

import cv2
import numpy as np

from .dataset import Stack
from .images import check_cancel, normalized_image, read_image


@dataclass(frozen=True)
class ROI:
    x: int
    y: int
    width: int
    height: int

    def fits(self, width: int, height: int) -> bool:
        return (self.x >= 0 and self.y >= 0 and self.width >= 3 and self.height >= 3
                and self.x + self.width <= width and self.y + self.height <= height)

    def crop(self, image: np.ndarray) -> np.ndarray:
        if not self.fits(image.shape[1], image.shape[0]):
            raise ValueError("ROI 必须在原图有效区域内且至少为 3 × 3 像素。")
        return image[self.y:self.y + self.height, self.x:self.x + self.width]


@dataclass(frozen=True)
class ROIMapping:
    """User-supplied pixel map from the right image to the left image."""

    scale_x: float
    scale_y: float
    offset_x: float
    offset_y: float
    valid_right: ROI
    valid_left: ROI

    def to_left(self, roi: ROI) -> ROI:
        if (not all(math.isfinite(value) for value in (self.scale_x, self.scale_y, self.offset_x, self.offset_y))
                or self.scale_x <= 0 or self.scale_y <= 0):
            raise ValueError("ROI 映射需要大于零的有限比例及有限偏移。")
        if not self._contains(self.valid_right, roi):
            raise ValueError("ROI 超出了右图指定的有效区域。")
        x = math.floor(self.scale_x * roi.x + self.offset_x)
        y = math.floor(self.scale_y * roi.y + self.offset_y)
        x2 = math.ceil(self.scale_x * (roi.x + roi.width) + self.offset_x)
        y2 = math.ceil(self.scale_y * (roi.y + roi.height) + self.offset_y)
        mapped = ROI(x, y, x2 - x, y2 - y)
        if not self._contains(self.valid_left, mapped) or min(mapped.width, mapped.height) < 3:
            raise ValueError("映射后的左图 ROI 超出了指定有效区域或小于 3 × 3 像素。")
        return mapped

    @staticmethod
    def _contains(area: ROI, region: ROI) -> bool:
        return ROI(region.x - area.x, region.y - area.y, region.width, region.height).fits(area.width, area.height)


METRIC_LABELS = {
    "laplacian_variance": "Laplacian Variance",
    "tenengrad": "Tenengrad",
    "sml": "SML",
    "sml_per_pixel": "SML / px",
    "brightness_mean": "亮度均值",
    "brightness_std": "亮度标准差",
    "saturated_ratio": "饱和比例",
    "background_noise_estimate": "背景噪声估计",
}


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    conversion = cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY
    return cv2.cvtColor(image, conversion)


def _sharpness(gray: np.ndarray) -> dict[str, float]:
    laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=1)
    _, lap_std = cv2.meanStdDev(laplacian)
    del laplacian
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    cv2.multiply(gx, gx, dst=gx)
    cv2.multiply(gy, gy, dst=gy)
    tenengrad = cv2.mean(gx)[0] + cv2.mean(gy)[0]
    del gx, gy
    difference = np.array([[-1.0, 2.0, -1.0]], dtype=np.float32)
    dx = cv2.filter2D(gray, cv2.CV_32F, difference)
    dy = cv2.filter2D(gray, cv2.CV_32F, difference.T)
    np.abs(dx, out=dx)
    np.abs(dy, out=dy)
    modified_laplacian = cv2.mean(dx)[0] + cv2.mean(dy)[0]
    return {"laplacian_variance": float(lap_std[0, 0] ** 2), "tenengrad": float(tenengrad),
            "sml": float(modified_laplacian * gray.size), "sml_per_pixel": float(modified_laplacian)}


def _noise_estimate(gray: np.ndarray) -> tuple[float | None, list[list[int]]]:
    height, width = gray.shape
    side = min(32, height, width)
    if side < 8:
        return None, []
    candidates = []
    xs = np.unique(np.linspace(0, width - side, 5, dtype=int))
    ys = np.unique(np.linspace(0, height - side, 5, dtype=int))
    for y in ys:
        for x in xs:
            tile = gray[y:y + side, x:x + side].astype(np.float32)
            activity = np.mean(np.abs(np.diff(tile, axis=0))) + np.mean(np.abs(np.diff(tile, axis=1)))
            candidates.append((float(activity), int(y), int(x), tile))
    estimates, windows = [], []
    for _, y, x, tile in sorted(candidates, key=lambda entry: entry[:3])[:5]:
        residual = cv2.Laplacian(tile, cv2.CV_32F)[1:-1, 1:-1]
        estimates.append(float(np.median(np.abs(residual - np.median(residual))) * 1.4826 / math.sqrt(20)))
        windows.append([x, y, side, side])
    return float(np.median(estimates)), windows


def measure(image: np.ndarray, roi: ROI | None = None) -> dict:
    height, width = image.shape[:2]
    color = normalized_image(roi.crop(image) if roi else image)
    if color.ndim == 3:
        color = color[:, :, :3]
    gray = _gray(color)
    mean, std = cv2.meanStdDev(gray)
    extremes = (color <= 0) | (color >= 255)
    if color.ndim == 3:
        extremes = np.any(extremes, axis=2)
    noise, windows = _noise_estimate(gray)
    sharpness = _sharpness(gray) if min(gray.shape) >= 3 else dict.fromkeys(tuple(METRIC_LABELS)[:4])
    return {"width": width, "height": height, "roi": [roi.x, roi.y, roi.width, roi.height] if roi else None,
            **sharpness, "brightness_mean": float(mean[0, 0]), "brightness_std": float(std[0, 0]),
            "saturated_ratio": float(np.mean(extremes)), "background_noise_estimate": noise,
            "noise_windows_in_measured_region": windows}


def measure_file(path: Path, roi: ROI | None = None) -> dict:
    return measure(read_image(path), roi)


def focus_curve(stack: Stack, roi: ROI, cancel: Event | None = None) -> list[dict]:
    samples = []
    for frame in stack.frames:
        check_cancel(cancel)
        image = read_image(frame.path, engine_input=True, preserve_depth=True)
        gray = _gray(normalized_image(roi.crop(image)))
        samples.append({"z": frame.z, "z_um": frame.z_um, **_sharpness(gray)})
        del image
    return samples
