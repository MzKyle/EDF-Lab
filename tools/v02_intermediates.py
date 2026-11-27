"""Export real, native-pixel diagnostic crops and inspect the native wavelet stages."""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import re
from threading import Event

import cv2
import numpy as np

from pollen_edf.config import load_settings
from pollen_edf.dataset import load_dataset
from pollen_edf.engine import FocusStackEngine
from pollen_edf.experiments import ExperimentRunner, Trial
from pollen_edf.images import read_image, sha256
from pollen_edf.records import read_json, write_json
from v02_benchmark import BASE, show_progress
from v02_measure import ROI_SIZE, checked_tile, gray, matches, png, tile


VIEW = "20261106_163002_v016"
ANCHOR = "dark"
PAD = 96


def inspect_stages(root, experiment, source, anchor):
    crop_offset = (PAD, PAD)
    raw_tiles = source.stacks[0].frames
    bg = read_image(root / anchor["folder"] / "dark-mask.png") > 0
    template = read_image(root / anchor["folder"] / "source.png")
    reports = {}
    for task in read_json(experiment / "manifest.json")["tasks"]:
        failure = task["status"] != "success"
        attempt = task["attempts"][-1]
        if failure:
            raise RuntimeError(attempt.get("diagnostic", task["status"]))
        result = read_image(experiment / attempt["output"])
        folder = experiment / attempt["dir"] / "steps"
        gray_stage = next(path for path in folder.glob("*merge_result.*") if path.is_file())
        inverse = read_image(gray_stage)
        if result.shape[:2] != inverse.shape[:2]:
            raise ValueError("Inverse-wavelet stage and final result differ in size.")
        aligned = task["options"]["alignment"] != "none"
        xy, ncc = matches(template, result, crop_offset, 24) if aligned else (crop_offset, None)
        inverse_patch, final_patch = (checked_tile(image, xy) for image in (inverse, result))
        if inverse_patch is None or final_patch is None:
            raise ValueError("Common diagnostic ROI is outside the result.")
        source_patch = tile(read_image(raw_tiles[anchor["best"]["sn"]-1].path), crop_offset)
        diagnostics = root / "native-diagnostics" / task["preset"]
        png(diagnostics / "source.png", source_patch)
        png(diagnostics / "inverse-wavelet.png", inverse_patch)
        png(diagnostics / "final.png", final_patch)

        def dark_detail(image):
            luma = gray(image).astype(np.float32)
            residual = luma - cv2.GaussianBlur(luma, (0, 0), 1.15)
            return {"rms": float(np.sqrt(np.mean(residual[bg]**2))), "p95": float(np.percentile(luma[bg], 95))}

        fields = {label: dark_detail(image) for label, image in (
            ("raw_standard_luma", source_patch), ("inverse_wavelet_pca_gray", inverse_patch),
            ("final_standard_luma", final_patch)
        )}
        support = None
        if not aligned:
            have_raw_color = np.zeros(final_patch.shape[:2], dtype=bool)
            for frame in raw_tiles:
                patch = tile(read_image(frame.path), crop_offset)
                have_raw_color |= np.all(patch == final_patch, axis=2)
            support = float(have_raw_color.mean())
        reports[task["preset"]] = {"xy": list(xy), "ncc": ncc, "dark_metrics": fields,
                                    "observed_color_fraction": support, "inverse_debug_file": gray_stage.name,
                                    "inverse_debug_codec": gray_stage.suffix.lower(),
                                    "folder": diagnostics.relative_to(root).as_posix()}
    write_json(root / "native-diagnostics.json", {"view": VIEW, "region": ANCHOR,
               "experiment": str(experiment), "native_pixel_crop": True, "reports": reports})


def trace_camera_points(root, origin):
    experiment = Path(read_json(root / "dev-native-ofat.json")["path"])
    tasks = {task["preset"]: task["attempts"][-1] for task in read_json(experiment / "manifest.json")["tasks"]
             if task["xy"] == VIEW and task["status"] == "success"}
    default, no_align = (tasks[name] for name in ("FS_Default", "FS_NoAlign"))
    log = (experiment / default["log"]).read_text(encoding="utf-8")
    numeric = r"(-?\d+(?:\.\d+)?)"
    triple = r"\(" + numeric + "," + numeric + r"\)"
    corners = re.findall(r"frame_\d+\.jpg transformed corners TL " + triple + r", TR " + triple
                         + r", BL " + triple, log)
    regions = [tuple(map(int, row)) for row in re.findall(r"valid area X (\d+), Y (\d+), W (\d+), H (\d+)", log)]
    crop_x, crop_y = (max(row[i] for row in regions) for i in (0, 1))
    base = read_image(experiment / default["output"])
    control = read_image(experiment / no_align["output"])
    w = min(x + width for x, _, width, _ in regions) - crop_x
    h = min(y + height for _, y, _, height in regions) - crop_y
    if base.shape[:2] != (h, w):
        raise ValueError("The logged valid rectangle does not match the baseline output.")
    transforms = np.array(corners, np.float32).reshape(-1, 3, 2)
    hot = read_image(root / origin["folder"] / "recurring-points.png")
    yx = np.argwhere(hot != 0)
    coords = yx[:, ::-1] + np.array(origin["xy"])
    height, width = control.shape[:2]
    relative = coords / (width, height)
    path = (transforms[:, 0, None] + (transforms[:, 1] - transforms[:, 0])[:, None] * relative[None, :, 0, None]
            + (transforms[:, 2] - transforms[:, 0])[:, None] * relative[None, :, 1, None]
            - (crop_x, crop_y))
    travel = np.linalg.norm(np.ptp(path, axis=0), axis=1)

    rows = []
    for item in np.argsort(travel)[-6:][::-1]:
        raw_center = tuple(map(int, coords[item]))
        warped = path[:, item]
        center = np.round(warped.mean(axis=0)).astype(int)
        left, top = int(center[0] - 34), int(center[1] - 34)
        tile_warped = gray(base[top:top+68, left:left+68]).astype(np.float32)
        tile_fixed = gray(control[raw_center[1]-34:raw_center[1]+34,
                                  raw_center[0]-34:raw_center[0]+34]).astype(np.float32)
        observed = []
        for array in (tile_warped, tile_fixed):
            residual = array - cv2.GaussianBlur(array, (0, 0), 1.15)
            observed.append(residual)
        mask = np.zeros_like(tile_warped, np.uint8)
        for x, y in np.round(warped - (left, top)).astype(int):
            cv2.circle(mask, (x, y), 1, 1, -1)
        along_path = np.argwhere((observed[0] > 2.5) & (mask != 0))
        fixed_core = np.argwhere(observed[1][31:38, 31:38] > 2.5)
        span = [float(np.linalg.norm(np.ptp(points, axis=0))) if len(points) > 1 else 0.0
                for points in (along_path, fixed_core)]
        label = f"point-{len(rows)+1}"
        sub = root / "alignment-points"
        for name, image in (("native", tile_warped), ("no-align", tile_fixed)):
            # Identical display gain and nearest-neighbour magnification for both dark crops.
            linear = np.uint8(np.clip(image * 8.0, 0, 255))
            png(sub / f"{label}-{name}.png", cv2.resize(linear, (272, 272), interpolation=cv2.INTER_NEAREST))
        png(sub / f"{label}-predicted.png", cv2.resize(mask * 255, (272, 272), interpolation=cv2.INTER_NEAREST))
        rows.append({"location_in_camera": raw_center, "axis_aligned_displacement_px": float(travel[item]),
                     "default_bright_pixels_on_predicted_path": len(along_path),
                     "default_residual_span_px": span[0],
                     "no_alignment_central_7px_residual_span_px": span[1],
                     "derived_display_multiplier": 8,
                     "native": f"alignment-points/{label}-native.png",
                     "no_align": f"alignment-points/{label}-no-align.png",
                     "predicted": f"alignment-points/{label}-predicted.png"})
    write_json(root / "alignment-points.json", {"view": VIEW, "signed_transforms": len(corners),
               "crop_origin": [crop_x, crop_y], "recurring_points": len(coords), "observations": rows})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    raw = load_dataset(root / "inputs" / "dev.json")
    stack = next(stack for stack in raw.stacks if stack.id == VIEW)
    anchor = read_json(root / "dev-roi-spec.json")[VIEW]["regions"][ANCHOR]
    directory = root / "diagnostic-inputs"
    rows = []
    w, h = ROI_SIZE
    xy = anchor["xy"][0]-PAD, anchor["xy"][1]-PAD
    for frame in stack.frames:
        original = read_image(frame.path, engine_input=True)
        path = directory / f"Z_{frame.z:03}.png"
        png(path, tile(original, xy, (w+2*PAD, h+2*PAD)))
        rows.append({"xy": VIEW, "z": frame.z, "path": path.name, "sha256": sha256(path),
                     "metadata": {"source": str(frame.path), "source_sha256": sha256(frame.path),
                                  "xyz_px_xy": xy, "archive_z": frame.metadata["archive_z"]}})
    index = directory / "dataset.json"
    write_json(index, {"name": "Real pollen native-pixel crop for wavelet diagnostics",
                       "metadata": {"provenance": "real_crop", "crop_wh": [w+2*PAD, h+2*PAD]}, "frames": rows})
    dataset = load_dataset(index)
    variants = (
        ("FS_Steps_Default", {}),
        ("FS_Steps_NoAlign", {"alignment": "none"}),
        ("FS_Steps_NoAlign_Denoise0", {"alignment": "none", "denoise": 0.0}),
        ("FS_Steps_NoAlign_Denoise64", {"alignment": "none", "denoise": 64.0}),
    )
    trials = [Trial(name, replace(BASE, intermediate=True, acceleration="cpu", **values))
              for name, values in variants]
    runner = ExperimentRunner(FocusStackEngine(load_settings().engine), root / "native-steps")
    experiment = runner.run(dataset, [VIEW], trials, 300.0, Event(), show_progress)
    inspect_stages(root, experiment, dataset, anchor)
    trace_camera_points(root, read_json(root / "dev-roi-spec.json")[VIEW]["regions"]["dark"])
    print(experiment)


if __name__ == "__main__":
    main()
