"""Distributions over observed layers for low-confidence pixels."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RegionalWeights:
    mass: np.ndarray
    candidates: np.ndarray
    candidate_count: np.ndarray

    def weight(self, index: int) -> np.ndarray:
        return self.mass * self.candidates[index] / self.candidate_count


def identify_halos(luminance: np.ndarray, foreground: np.ndarray, confidence: np.ndarray,
                   hard_threshold: float) -> RegionalWeights:
    """Prefer observed low-luminance phases when both ends show diffuse light."""
    count = luminance.shape[0]
    order = int(0.20 * (count - 1))
    low = np.partition(luminance, order, axis=0)[order].astype(np.float32)
    tail = max(1, count // 8)
    early = np.mean(luminance[:tail], axis=0, dtype=np.float32)
    late = np.mean(luminance[-tail:], axis=0, dtype=np.float32)
    raised_ends = np.minimum(early, late) - low
    weak = np.clip((hard_threshold - confidence) / 0.29, 0, 1)
    low_support = np.clip((24.0 - low) / 12.0, 0, 1)
    temporal_trough = np.clip((raised_ends - 4.0) / 9.0, 0, 1)
    mass = (foreground * weak * low_support * temporal_trough).astype(np.float32)
    candidates = luminance <= low + 2.0
    candidate_count = np.count_nonzero(candidates, axis=0).astype(np.float32)
    return RegionalWeights(mass, candidates, candidate_count)


def pool_broad_responses(scores: np.ndarray, strongest: np.ndarray, foreground: np.ndarray,
                         confidence: np.ndarray, hard_threshold: float) -> RegionalWeights:
    """Keep broad, weak Z responses from switching abruptly between observed layers."""
    candidates = scores >= (0.80 * strongest)[None, ...]
    candidate_count = np.count_nonzero(candidates, axis=0).astype(np.float32)
    share = candidate_count / len(scores)
    breadth = np.clip((share - 0.22) / 0.48, 0, 1)
    weak = np.clip((hard_threshold - confidence) / 0.36, 0, 1)
    mass = (0.92 * foreground * breadth * weak).astype(np.float32)
    return RegionalWeights(mass, candidates, candidate_count)


def anchor_weak_stack(scores: np.ndarray, foreground: np.ndarray, remaining: np.ndarray,
                      confidence: np.ndarray, hard_threshold: float) -> tuple[RegionalWeights | None, int | None]:
    core = foreground >= 0.90
    count = np.count_nonzero(core)
    confident = np.count_nonzero(core & (confidence >= hard_threshold))
    if count < 256 or confident / count >= 0.003:
        return None, None
    strengths = [float(np.sum(plane, where=core, dtype=np.float64)) for plane in scores]
    selected = int(np.argmax(strengths))
    middle = len(scores) // 2
    if strengths[middle] >= 0.90 * strengths[selected]:
        selected = middle
    candidates = np.zeros(scores.shape, dtype=bool)
    candidates[selected] = True
    weak = np.clip((hard_threshold - confidence) / 0.22, 0, 1)
    return RegionalWeights(remaining * weak, candidates, np.ones_like(remaining)), selected
