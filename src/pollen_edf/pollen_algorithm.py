"""Focus scoring, depth selection and pixel fusion in camera coordinates."""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil, sqrt
from typing import Callable

import cv2
import numpy as np

from .pollen_brightfield import FocalWeights, brightfield_score, focal_weights, smooth_in_focus_order
from .pollen_weights import RegionalWeights, anchor_weak_stack, identify_halos, pool_broad_responses


SCORE_BYTES = 192 * 1024**2
HARD_CONFIDENCE = 0.60


@dataclass(frozen=True)
class PollenEDFOptions:
    scorer: str = "multiscale"
    score_scale: int = 2
    noise_aware: bool = True
    refine: bool = True
    background: bool = True
    defects: bool = True
    soft: bool = True
    halo_guard: bool = False
    broad_pool: bool = False
    weak_stack_anchor: bool = False
    pyramid_levels: int = 0
    threads: int = 8
    depth: bool = True
    intermediate: bool = False
    brightfield: bool = False
    focus_context_px: int = 0

    def validate(self, count: int) -> None:
        if not 2 <= count <= 65535:
            raise ValueError("PollenEDF 需要 2 至 65535 个焦层。")
        if self.scorer not in {"sml", "tenengrad", "multiscale"}:
            raise ValueError("scorer 必须是 sml、tenengrad 或 multiscale。")
        for name in ("noise_aware", "refine", "background", "defects", "soft", "halo_guard",
                     "broad_pool", "weak_stack_anchor", "depth", "intermediate", "brightfield"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} 必须为布尔值。")
        for name, low, high in (("score_scale", 1, 32), ("pyramid_levels", 0, 4), ("threads", 1, 128),
                                ("focus_context_px", 0, 64)):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} 必须为 [{low}, {high}] 内的整数。")
        if (self.pyramid_levels or self.brightfield) and not self.soft:
            raise ValueError("金字塔或明场融合需要启用软融合。")
        if self.focus_context_px and not self.brightfield:
            raise ValueError("焦点邻域参数仅用于明场预设。")

    def output_names(self) -> tuple[str, ...]:
        names = ("fused.png",)
        if self.depth:
            names += ("depth.png", "depth-index.png", "confidence.png")
        if self.intermediate:
            names += ("foreground.png", "alternate.png", "stages.json")
            if self.halo_guard:
                names += ("halo.png",)
            if self.broad_pool:
                names += ("broad.png",)
            if self.weak_stack_anchor:
                names += ("anchor.png",)
        return names


@dataclass(frozen=True)
class FocusSelection:
    layer: np.ndarray
    alternate: np.ndarray
    confidence: np.ndarray
    foreground: np.ndarray
    alternate_weight: np.ndarray
    stages: dict
    defects: np.ndarray | None = None
    halo: RegionalWeights | None = None
    broad: RegionalWeights | None = None
    anchor: RegionalWeights | None = None
    focal_weights: FocalWeights | None = None


def _sml(gray: np.ndarray) -> np.ndarray:
    kernel = np.array([[1.0, -2.0, 1.0]], dtype=np.float32)
    gx = cv2.filter2D(gray, cv2.CV_32F, kernel)
    gy = cv2.filter2D(gray, cv2.CV_32F, kernel.T)
    np.abs(gx, out=gx)
    np.abs(gy, out=gy)
    return cv2.boxFilter(gx + gy, -1, (5, 5))


def focus_score(gray: np.ndarray, scorer: str, noise_aware: bool) -> tuple[np.ndarray, float]:
    """Score one area-reduced layer, in grayscale intensity units."""
    working = cv2.GaussianBlur(gray, (0, 0), 0.62) if noise_aware else gray
    if scorer == "tenengrad":
        dx = cv2.Sobel(working, cv2.CV_32F, 1, 0, ksize=3, scale=0.125)
        dy = cv2.Sobel(working, cv2.CV_32F, 0, 1, ksize=3, scale=0.125)
        score = cv2.sqrt(cv2.boxFilter(dx * dx + dy * dy, -1, (5, 5)))
    elif scorer == "multiscale":
        fine = _sml(working)
        medium = _sml(cv2.GaussianBlur(working, (0, 0), 1.35))
        broad = _sml(cv2.GaussianBlur(working, (0, 0), 2.6))
        score = np.maximum(fine - 0.85 * broad, 0.0) + 0.35 * medium
    else:
        score = _sml(working)
    if not noise_aware:
        return score, 0.0

    # Use the dim half of a subsample to estimate the measured response floor.
    # The luminance term tempers camera noise on bright, nearly flat surfaces.
    probe = working[::4, ::4]
    dim = probe <= np.median(probe)
    floor = float(np.quantile(score[::4, ::4][dim], 0.70))
    level = cv2.boxFilter(working, -1, (5, 5))
    np.maximum(score - (floor + 0.045 * cv2.sqrt(level)), 0, out=score)
    score /= 1.0 + 0.006 * level
    return score, floor


def _rank(volume: np.ndarray, order: list[int], include: Callable | None = None) -> tuple[np.ndarray, np.ndarray]:
    layer = np.full(volume.shape[1:], order[0], dtype=np.uint16)
    best = np.full(volume.shape[1:], -1.0, dtype=np.float32)
    for index in order:
        score = volume[index]
        use = score > best
        if include is not None:
            use &= include(index)
        best[use] = score[use]
        layer[use] = index
    return layer, best


def _selected_score(volume: np.ndarray, label: np.ndarray) -> np.ndarray:
    return np.take_along_axis(volume, label[None, ...], axis=0)[0]


def refine_labels(volume: np.ndarray, label: np.ndarray, confidence: np.ndarray, guide: np.ndarray) -> np.ndarray:
    """Vote with axial neighbours without averaging their focal indices."""
    h, w = label.shape
    score = _selected_score(volume, label)
    label_pad = np.pad(label, 1, mode="edge")
    guide_pad = np.pad(guide, 1, mode="edge")
    conf_pad = np.pad(confidence, 1, mode="edge")
    updated = label.copy()
    merit = np.full(label.shape, 1.0, dtype=np.float32)
    can_change = (confidence < HARD_CONFIDENCE) & (score > 0.015)
    for y, x in ((0, 1), (1, 0), (1, 2), (2, 1)):
        candidate = label_pad[y:y+h, x:x+w]
        level = guide_pad[y:y+h, x:x+w]
        neighbour_confidence = conf_pad[y:y+h, x:x+w]
        relative = _selected_score(volume, candidate) / np.maximum(score, 0.015)
        edge_weight = np.exp(-np.abs(level - guide) / 12.0)
        candidate_merit = relative + 0.23 * edge_weight * neighbour_confidence
        accept = can_change & (relative >= 0.82) & (candidate_merit > merit) & (candidate != label)
        updated[accept] = candidate[accept]
        merit[accept] = candidate_merit[accept]
    return updated


def _foreground(guide: np.ndarray) -> tuple[np.ndarray, dict]:
    # Bright-field or evenly illuminated stacks get no dark-background treatment.
    low = float(np.quantile(guide[::4, ::4], 0.12))
    if low > 20.0:
        return np.ones_like(guide), {"dark_field": False, "level": low}
    threshold = max(4.5, low + 3.5)
    alpha = np.clip((guide - threshold) / 5.0, 0.0, 1.0)
    return alpha, {"dark_field": True, "level": low, "threshold": threshold}


def select_focus(load: Callable[[int], np.ndarray], size: tuple[int, int], count: int,
                 options: PollenEDFOptions, checkpoint: Callable[[str], None]) -> FocusSelection:
    w, h = size
    dtype = np.dtype(np.float16 if options.brightfield else np.float32)
    itemsize = dtype.itemsize
    scale = max(options.score_scale, ceil(sqrt(count * w * h * itemsize / SCORE_BYTES)))
    while count * ceil(w / scale) * ceil(h / scale) * itemsize > SCORE_BYTES:
        scale += 1
    reduced = ceil(w / scale), ceil(h / scale)
    volume = np.empty((count, reduced[1], reduced[0]), dtype=dtype)
    luminance = np.empty(volume.shape, dtype=np.uint8) if options.background and options.halo_guard else None
    guide = np.zeros(volume.shape[1:], dtype=np.float32)
    point_votes = np.zeros((h, w), dtype=np.uint16) if options.background and options.defects else None
    floors = []
    for index in range(count):
        checkpoint(f"score {index + 1}/{count}")
        image = load(index)
        if image.shape != (h, w, 3):
            raise ValueError("所有输入应为同尺寸 BGR 图像。")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        del image
        if point_votes is not None:
            local_median = cv2.medianBlur(gray, 3)
            point_votes += (local_median <= 8) & (cv2.subtract(gray, local_median) > 10)
        small = cv2.resize(gray, reduced, interpolation=cv2.INTER_AREA).astype(np.float32)
        if options.brightfield:
            local, floor = brightfield_score(small, options.noise_aware, options.focus_context_px / scale)
        else:
            local, floor = focus_score(small, options.scorer, options.noise_aware)
        volume[index] = local
        smooth_luma = cv2.GaussianBlur(small, (0, 0), 0.85)
        np.maximum(guide, smooth_luma, out=guide)
        if luminance is not None:
            luminance[index] = np.rint(smooth_luma).astype(np.uint8)
        floors.append(floor)

    checkpoint("select focus")
    if options.brightfield:
        smooth_in_focus_order(volume)
    midpoint = count // 2
    order = sorted(range(count), key=lambda index: (abs(index - midpoint), index))
    primary, strongest = _rank(volume, order)
    # Adjacent planes can legitimately show almost equal focus. Competing
    # evidence is evaluated outside the two-layer neighbourhood.
    primary_int = primary.astype(np.int32)
    _, distant_score = _rank(volume, order, lambda index: np.abs(primary_int - index) > 2)
    margin = np.clip((strongest - np.maximum(distant_score, 0)) / (strongest + 0.12), 0, 1)
    strength = strongest / (strongest + 0.35)
    confidence = (strength * (0.30 + 0.70 * margin)).astype(np.float32)
    confidence[(primary == 0) | (primary == count - 1)] *= 0.80
    labels = refine_labels(volume, primary, confidence, guide) if options.refine else primary
    alternate, alternate_score = _rank(volume, order, lambda index: labels != index)
    if options.brightfield:
        ones = np.ones_like(guide)
        stages = {"score_grid": list(reduced), "effective_scale": scale, "score_volume_bytes": volume.nbytes,
                  "reference_index": midpoint, "noise_floors": floors,
                  "label_changes": int(np.count_nonzero(labels != primary)),
                  "foreground_area": 1.0, "high_confidence_area": float(np.mean(confidence >= HARD_CONFIDENCE)),
                  "background_model": {"brightfield_weights": True}, "focus_context_px": options.focus_context_px}
        return FocusSelection(labels, alternate, confidence, ones, np.zeros_like(guide), stages,
                              focal_weights=focal_weights(volume, labels, strongest))
    selected = _selected_score(volume, labels)
    relative = np.clip(alternate_score / np.maximum(selected, 0.015), 0, 1)
    weight = (0.35 * (1.0 - confidence) * np.clip((relative - 0.58) / 0.42, 0, 1)).astype(np.float32)
    weight[confidence >= HARD_CONFIDENCE] = 0.0
    if not options.soft:
        weight.fill(0)
    foreground, background = _foreground(guide) if options.background else (np.ones_like(guide), {})
    halo = (identify_halos(luminance, foreground, confidence, HARD_CONFIDENCE)
            if luminance is not None and background["dark_field"] else None)
    remaining = foreground - halo.mass if halo else foreground
    anchor, anchor_index = (anchor_weak_stack(volume, foreground, remaining, confidence, HARD_CONFIDENCE)
                            if options.soft and options.weak_stack_anchor else (None, None))
    broad = None
    if options.soft and options.broad_pool:
        broad = pool_broad_responses(volume, strongest, remaining - anchor.mass if anchor else remaining,
                                     confidence, HARD_CONFIDENCE)
    defects = None
    if point_votes is not None and background["dark_field"]:
        # The same isolated response in most focal planes is evidence of a
        # camera-coordinate impulse. Require dark spatial support as well.
        recurring = np.uint8(point_votes >= ceil(0.65 * count))
        isolated = cv2.boxFilter(recurring, cv2.CV_16U, (3, 3), normalize=False) == 1
        dark_support = cv2.resize(cv2.medianBlur(foreground, 3), size) < 0.05
        defects = (recurring != 0) & isolated & dark_support
    stages = {"score_grid": [reduced[0], reduced[1]], "effective_scale": scale,
              "score_volume_bytes": volume.nbytes, "reference_index": midpoint,
              "noise_floors": floors, "label_changes": int(np.count_nonzero(labels != primary)),
              "foreground_area": float(np.mean(foreground >= 0.5)), "background_model": background,
              "recurring_dark_impulses": int(np.count_nonzero(defects)) if defects is not None else 0,
              "halo_sample_weight": float(halo.mass.mean()) if halo else 0.0,
              "broad_focus_weight": float(broad.mass.mean()) if broad else 0.0,
              "single_plane_fallback_index": anchor_index,
              "single_plane_fallback_weight": float(anchor.mass.mean()) if anchor else 0.0,
              "high_confidence_area": float(np.mean((confidence >= HARD_CONFIDENCE) & (foreground >= 0.98))),
              "endpoint_fraction_in_foreground": float(np.count_nonzero(
                   ((labels == 0) | (labels == count - 1)) & (foreground >= 0.5)
              ) / max(1, np.count_nonzero(foreground >= 0.5)))}
    return FocusSelection(labels, alternate, confidence, foreground, weight, stages, defects, halo, broad, anchor)


def _gaussian_pyramid(image: np.ndarray, levels: int) -> list[np.ndarray]:
    result = [image]
    for _ in range(levels):
        result.append(cv2.pyrDown(result[-1]))
    return result


def _pyramid_add(bands: list[np.ndarray], image: np.ndarray, weights: np.ndarray) -> None:
    levels = len(bands) - 1
    gaussian = _gaussian_pyramid(image.astype(np.float32), levels)
    weight_pyramid = _gaussian_pyramid(weights, levels)
    for index, (band, layer, factor) in enumerate(zip(bands, gaussian, weight_pyramid)):
        if index < levels:
            layer -= cv2.pyrUp(gaussian[index + 1], dstsize=(layer.shape[1], layer.shape[0]))
        layer *= factor[..., None]
        np.add(band, layer, out=band)


def _pyramid_merge(bands: list[np.ndarray]) -> np.ndarray:
    result = bands[-1]
    for band in reversed(bands[:-1]):
        result = cv2.pyrUp(result, dstsize=(band.shape[1], band.shape[0])) + band
    return result


def render_fusion(load: Callable[[int], np.ndarray], size: tuple[int, int], count: int,
                  selection: FocusSelection, options: PollenEDFOptions,
                  checkpoint: Callable[[str], None]) -> np.ndarray:
    w, h = size
    labels = cv2.resize(selection.layer, size, interpolation=cv2.INTER_NEAREST)
    focal = selection.focal_weights
    hard = np.empty((h, w, 3), dtype=np.uint8) if focal is None else None
    blend = options.soft or options.background or focal is not None
    if blend:
        if hard is not None:
            hard_mask = (cv2.resize(selection.confidence, size) >= HARD_CONFIDENCE)
            hard_mask &= cv2.resize(selection.foreground, size) >= 0.98
        levels = min(options.pyramid_levels, max(0, min(w, h).bit_length() - 2))
        if levels:
            dims = [(ceil(h / 2**i), ceil(w / 2**i), 3) for i in range(levels + 1)]
            bands = [np.zeros(shape, dtype=np.float32) for shape in dims]
        else:
            accumulation = np.zeros((h, w, 3), dtype=np.float32)
            weighted = np.empty_like(accumulation)
        if focal is None:
            regional = tuple(distribution for distribution in (selection.halo, selection.broad, selection.anchor)
                             if distribution is not None)
            foreground = selection.foreground.copy()
            for distribution in regional:
                foreground -= distribution.mass
            primary_weight = (1.0 - selection.alternate_weight) * foreground
            alternate_weight = selection.alternate_weight * foreground
            dark_weight = (1.0 - selection.foreground) / count

    active = set(np.unique(selection.layer).tolist())
    for index in range(count):
        checkpoint(f"render {index + 1}/{count}")
        if not blend and index not in active:
            continue
        image = load(index)
        if hard is not None and index in active:
            cv2.copyTo(image, np.uint8(labels == index), hard)
        if blend:
            if focal is None:
                weight = (dark_weight + primary_weight * (selection.layer == index)
                          + alternate_weight * (selection.alternate == index))
                for distribution in regional:
                    weight += distribution.weight(index)
            else:
                weight = focal.weight(index)
            full = cv2.resize(weight, size, interpolation=cv2.INTER_LINEAR)
            if levels:
                _pyramid_add(bands, image, full)
            else:
                np.multiply(image, full[..., None], out=weighted)
                np.add(accumulation, weighted, out=accumulation)
    if not blend:
        return hard
    checkpoint("finalize fusion")
    output = _pyramid_merge(bands) if levels else accumulation
    np.maximum(output, 0, out=output)
    np.minimum(output, 255, out=output)
    np.rint(output, out=output)
    result = output.astype(np.uint8)
    if hard is not None:
        cv2.copyTo(hard, hard_mask.astype(np.uint8), result)
    if selection.defects is not None and selection.defects.any():
        selected = selection.defects & ~hard_mask
        result[selected] = cv2.medianBlur(result, 3)[selected]
    return result


def diagnostic_images(selection: FocusSelection, size: tuple[int, int], count: int,
                      options: PollenEDFOptions) -> dict[str, np.ndarray]:
    images = {}
    if options.depth:
        labels = cv2.resize(selection.layer, size, interpolation=cv2.INTER_NEAREST)
        images["depth-index.png"] = labels + 1
        images["depth.png"] = np.round(labels.astype(np.float32) * (255.0 / (count - 1))).astype(np.uint8)
        images["confidence.png"] = np.round(cv2.resize(selection.confidence, size) * 255.0).astype(np.uint8)
    if options.intermediate:
        images["foreground.png"] = np.round(cv2.resize(selection.foreground, size) * 255.0).astype(np.uint8)
        alternate = cv2.resize(selection.alternate, size, interpolation=cv2.INTER_NEAREST)
        images["alternate.png"] = np.round(alternate.astype(np.float32) * (255.0 / (count - 1))).astype(np.uint8)
        for enabled, name, weights in ((options.halo_guard, "halo.png", selection.halo),
                                       (options.broad_pool, "broad.png", selection.broad),
                                       (options.weak_stack_anchor, "anchor.png", selection.anchor)):
            if enabled:
                mass = weights.mass if weights else np.zeros_like(selection.foreground)
                images[name] = np.round(cv2.resize(mass, size) * 255.0).astype(np.uint8)
    return images
