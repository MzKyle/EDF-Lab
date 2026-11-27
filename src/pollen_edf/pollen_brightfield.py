"""Continuous focal-plane weights for reflected-light and bright-field captures."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class FocalWeights:
    layers: np.ndarray
    total: np.ndarray

    def weight(self, index: int) -> np.ndarray:
        return self.layers[index].astype(np.float32) / self.total


def brightfield_score(luma: np.ndarray, noise_aware: bool, context_sigma: float = 0.0) -> tuple[np.ndarray, float]:
    """Measure local in-band contrast and discount low-frequency defocus."""
    fine = cv2.GaussianBlur(luma, (0, 0), 0.58)
    medium = cv2.GaussianBlur(luma, (0, 0), 1.15)
    coarse = cv2.GaussianBlur(luma, (0, 0), 2.45)
    fine_band = fine - medium
    coarse_band = medium - coarse
    local = cv2.sqrt(cv2.GaussianBlur(fine_band * fine_band, (0, 0), 1.25))
    broad = cv2.sqrt(cv2.GaussianBlur(coarse_band * coarse_band, (0, 0), 1.25))
    score = np.maximum(local - 0.16 * broad, 0.0)
    floor = 0.0
    if noise_aware:
        floor = float(np.quantile(score[::5, ::5], 0.15))
        np.maximum(score - 0.85 * floor, 0, out=score)
    if context_sigma:
        surrounding = cv2.GaussianBlur(score, (0, 0), context_sigma)
        score = cv2.addWeighted(score, 0.38, surrounding, 0.62, 0)
    return score, floor


def smooth_in_focus_order(volume: np.ndarray) -> None:
    """Discount a one-layer spike without keeping a second score volume."""
    previous = volume[0].astype(np.float32)
    for index, row in enumerate(volume):
        current = row.astype(np.float32)
        following = volume[min(index + 1, len(volume) - 1)].astype(np.float32)
        row[:] = 0.20 * previous + 0.60 * current + 0.20 * following
        previous = current


def focal_weights(volume: np.ndarray, labels: np.ndarray, strongest: np.ndarray) -> FocalWeights:
    """Reuse the score volume for a soft distribution over actual source layers."""
    total = np.zeros(strongest.shape, np.float32)
    reference = labels.astype(np.float32)
    limit = 0.58 * strongest
    span = 0.42 * strongest + 0.015
    for index, row in enumerate(volume):
        relative = np.maximum(row.astype(np.float32) - limit, 0.0) / span
        distance = (index - reference) / 4.0
        axial = 0.24 + 0.76 * np.exp(-0.5 * distance * distance)
        row[:] = relative * relative * axial + 0.0003
        total += row.astype(np.float32)
    return FocalWeights(volume, total)
