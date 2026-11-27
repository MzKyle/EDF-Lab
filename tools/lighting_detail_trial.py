"""Test tighter high-frequency weights on the same measured development crops."""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from lighting_benchmark import CONTEXT
from lighting_measure import compare_roi
from pollen_edf.dataset import load_dataset
from pollen_edf.experiments import trials_for
from pollen_edf.images import read_image
from pollen_edf.pollen_algorithm import _gaussian_pyramid, _pyramid_merge, render_fusion, select_focus
from pollen_edf.records import read_json, write_json
from v02_measure import ROI_SIZE, png, tile


def render_detail(frames, selection, size, levels):
    """Tighten the layer distribution of the two finest bands."""
    pool = selection.focal_weights
    power_total = np.zeros_like(pool.total)
    for index in range(len(frames)):
        weight = pool.weight(index)
        power_total += weight * weight
    bands = None
    for index, image in enumerate(frames):
        weight = pool.weight(index)
        context = _gaussian_pyramid(cv2.resize(weight, size), levels)
        detail = _gaussian_pyramid(cv2.resize(weight * weight / power_total, size), 1)
        planes = _gaussian_pyramid(image.astype(np.float32), levels)
        if bands is None:
            bands = [np.zeros_like(plane) for plane in planes]
        for k, plane in enumerate(planes):
            if k < levels:
                plane -= cv2.pyrUp(planes[k+1], dstsize=(plane.shape[1], plane.shape[0]))
            plane *= (detail[k] if k < len(detail) else context[k])[..., None]
            bands[k] += plane
    return np.rint(np.clip(_pyramid_merge(bands), 0.0, 255.0)).astype(np.uint8)


def evaluate(root: Path) -> None:
    cv2.setNumThreads(4)
    rois = read_json(root / "dev-lighting-rois.json")
    options = trials_for([CONTEXT])[0].options
    checkpoint = lambda _stage: None
    results = []
    for stack in load_dataset(root / "inputs" / "dev.json").stacks:
        original_size = stack.metadata["source_size"]
        w, h = ROI_SIZE
        windows = {}
        for kind, region in rois[stack.id]["regions"].items():
            if kind == "quiet":
                continue
            x, y = region["xy"]
            left, top = max(x-64, 0), max(y-64, 0)
            right = min(x+w+64, original_size[0])
            bottom = min(y+h+64, original_size[1])
            windows[kind] = ((left, top), (right-left, bottom-top), (x-left, y-top), [])
        for frame in stack.frames:
            image = read_image(frame.path, engine_input=True)
            for origin, size, _, stored in windows.values():
                stored.append(tile(image, origin, size).copy())
        for kind, (_, size, offset, frames) in windows.items():
            region = rois[stack.id]["regions"][kind]
            scale = 2 if len(frames) == 29 else 3
            config = replace(options, score_scale=scale)
            selection = select_focus(frames.__getitem__, size, len(frames), config, checkpoint)
            original = render_fusion(frames.__getitem__, size, len(frames), selection, config, checkpoint)
            detail = render_detail(frames, selection, size, config.pyramid_levels)
            folder = root / "detail-crop-trial" / stack.id
            row = {"view": stack.id, "region": kind, "original_roi": region["xy"], "score_scale": scale}
            for name, image in (("context", original), ("detail", detail)):
                patch = tile(image, offset)
                png(folder / f"{kind}-{name}.png", patch)
                row[name] = compare_roi(root / region["folder"], patch, region)
            results.append(row)
        print(f"{stack.id}: detail crop trial", flush=True)
    write_json(root / "detail-crop-trial" / "measurements.json", {
        "scope": "development crops with padding; no holdout tuning",
        "change": "squared and normalized high-band focal weights; unchanged low bands",
        "results": results})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    evaluate(parser.parse_args().root.resolve())
