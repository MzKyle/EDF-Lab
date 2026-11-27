"""Bright-background ROI measurements against observed focal planes."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import median

import cv2
import numpy as np

from pollen_edf.dataset import load_dataset
from pollen_edf.images import read_image, sha256
from pollen_edf.records import read_json, write_json
from v02_measure import (MIN_MATCH, ROI_SIZE, blurred, collect, edge_response, focus_plot, focus_row,
                         gray, matches, png, tile)


FOCAL_PATCH = 48
CENTRES = [(x, y) for y in (48, 112, 176, 240) for x in (48, 120, 192, 264, 336)]
DEV_WINDOWS = {
    "20250423_000246_v004": {"wall": [1456, 1948]},
    "20250423_000246_v008": {"wall": [64, 972]},
}


def raw_locations(stack) -> dict:
    projection = contrast = None
    for frame in stack.frames:
        reduced = cv2.imdecode(np.fromfile(frame.path, np.uint8), cv2.IMREAD_REDUCED_GRAYSCALE_4)
        if reduced is None:
            raise ValueError(f"Cannot read {frame.path}")
        fine = edge_response(blurred(reduced, 0.8))
        broad = cv2.GaussianBlur(reduced.astype(np.float32), (0, 0), 1.6)
        relief = np.abs(broad - cv2.GaussianBlur(broad, (0, 0), 12.0))
        if projection is None:
            projection, contrast = fine, relief
        else:
            np.maximum(projection, fine, out=projection)
            np.maximum(contrast, relief, out=contrast)
    w, h = (side // 4 for side in ROI_SIZE)
    allowed = np.zeros(projection.shape, bool)
    pad = 8
    allowed[h//2+pad:-(h//2+pad), w//2+pad:-(w//2+pad)] = True
    window = (w, h)
    detail = cv2.boxFilter(np.minimum(projection, 5.0), -1, window)
    structure = cv2.boxFilter(np.minimum(contrast, 28.0), -1, window)
    wall_score = detail + 0.26 * structure
    wall_score[~allowed] = -np.inf
    y, x = np.unravel_index(wall_score.argmax(), wall_score.shape)
    loc = {"wall": [int(x-w//2)*4, int(y-h//2)*4]}
    loc.update(DEV_WINDOWS.get(stack.id, {}))
    centre_wall = [loc["wall"][0]//4+w//2, loc["wall"][1]//4+h//2]
    _, _, stats, centroids = cv2.connectedComponentsWithStats(np.uint8(contrast > 13.0))
    small = []
    for box, centre in zip(stats[1:], centroids[1:]):
        bx, by, bw, bh, area = map(int, box)
        xx, yy = map(int, np.rint(centre))
        if (allowed[yy, xx] and 9 <= area <= 185 and 3 <= min(bw, bh) <= max(bw, bh) <= 30
                and (abs(xx-centre_wall[0]) >= w or abs(yy-centre_wall[1]) >= h)):
            strength = float(contrast[by:by+bh, bx:bx+bw].max())
            if strength >= 16:
                small.append((abs(np.log(area/42)) - 0.03 * min(strength, 45.0), xx, yy))
    if small:
        _, xx, yy = min(small)
        loc["small"] = [(xx-w//2)*4, (yy-h//2)*4]
    quiet = detail + 0.22 * structure
    quiet[~allowed] = np.inf
    yy, xx = np.unravel_index(quiet.argmin(), quiet.shape)
    loc["quiet"] = [int(xx-w//2)*4, int(yy-h//2)*4]
    return loc


def anchor_tiles(image: np.ndarray) -> list[np.ndarray]:
    r = FOCAL_PATCH // 2
    return [image[y-r:y+r, x-r:x+r] for x, y in CENTRES]


def focus_anchors(patches: list[np.ndarray]) -> tuple[list, list]:
    details = []
    for patch in patches:
        focus = edge_response(blurred(patch, 0.95))
        observations = [float(np.mean(part)) for part in anchor_tiles(focus)]
        details.append(observations)
    curve = np.asarray(details, np.float32)
    choices = []
    pool = np.arange(len(patches))
    for index, sn in enumerate(curve.argmax(axis=0)):
        xy = CENTRES[index]
        source = anchor_tiles(patches[sn])[index]
        span = float(np.ptp(np.percentile(blurred(source, 1.2), [10, 90])))
        strongest = float(curve[sn, index])
        distant = float(curve[np.abs(pool-sn) > 3, index].max(initial=0.0))
        if span >= 9.0 and strongest >= 0.25 and strongest >= 1.10 * distant:
            choices.append({"index": index, "centre": xy, "plane": int(sn), "contrast": span,
                            "peak": strongest, "remote_peak": distant,
                            "at_endpoint": bool(sn == 0 or sn == len(patches)-1)})
    return choices, details


def inspect_raw(root: Path, split: str) -> dict:
    path = root / f"{split}-lighting-rois.json"
    if path.is_file():
        return read_json(path)
    results = {}
    for stack in load_dataset(root / "inputs" / f"{split}.json").stacks:
        locations = raw_locations(stack)
        patches = {name: [] for name in locations}
        original_hashes = []
        for frame in stack.frames:
            original_hashes.append(sha256(frame.path))
            if original_hashes[-1] != frame.expected_sha256:
                raise ValueError(f"Source has changed since indexing: {frame.path}")
            image = read_image(frame.path, engine_input=True)
            if list(image.shape[1::-1]) != stack.metadata["source_size"]:
                raise ValueError(f"Unexpected decoded size for {frame.path}")
            for name, xy in locations.items():
                patches[name].append(tile(image, xy).copy())
        view = {"meta": stack.metadata, "hashes": original_hashes, "regions": {}}
        for name, xy in locations.items():
            stack_patches = patches[name]
            folder = root / "raw" / split / stack.id / name
            rows = [focus_row(patch, frame) for patch, frame in zip(stack_patches, stack.frames)]
            anchors, sharpness = focus_anchors(stack_patches)
            for anchor in anchors:
                sn, index = anchor["plane"], anchor["index"]
                anchor["z"] = rows[sn]["z"]
                png(folder / f"source-anchor-{index}.png", anchor_tiles(stack_patches[sn])[index])
            if name == "small":
                key = "center_score"
            else:
                key = "score"
            best = max(range(len(rows)), key=lambda index: rows[index][key])
            max_detail = np.maximum.reduce([edge_response(blurred(patch, 1.25)) for patch in stack_patches])
            no_clip = np.logical_and.reduce([np.max(patch, axis=2) < 251 for patch in stack_patches])
            quiet = (max_detail <= np.percentile(max_detail, 40)) & no_clip
            clipped = [np.min(patch, axis=2) >= 250 for patch in stack_patches]
            png(folder / "source.png", stack_patches[best])
            png(folder / "quiet-mask.png", np.uint8(quiet)*255)
            focus_plot(folder, rows)
            view["regions"][name] = {"xy": xy, "best": rows[best], "curve": rows, "anchor_curve": sharpness,
                                     "anchors": anchors, "quiet_pixels": int(quiet.sum()),
                                     "white_fraction_best": float(clipped[best].mean()),
                                     "white_fraction_every_layer": float(np.logical_and.reduce(clipped).mean()),
                                     "folder": folder.relative_to(root).as_posix()}
        results[stack.id] = view
        print(f"{split} {stack.id}: raw reference prepared", flush=True)
    write_json(path, results)
    return results


def compare_roi(raw_folder: Path, patch: np.ndarray, region: dict) -> dict:
    quiet_mask = read_image(raw_folder / "quiet-mask.png") != 0
    luma = gray(patch).astype(np.float32)
    residual = luma - cv2.GaussianBlur(luma, (0, 0), 1.15)
    correlations, fixed_correlations, transfers, shifts = [], [], [], []
    diagnostics = {"quiet_rms": None, "edge_correlation": None, "edge_fixed_correlation": None,
                   "edge_transfer": None, "anchor_shift_px": None, "anchors": 0,
                   "white_fraction": float(np.mean(np.min(patch, axis=2) >= 250))}
    if np.count_nonzero(quiet_mask) >= 64:
        diagnostics["quiet_rms"] = float(np.sqrt(np.mean(residual[quiet_mask]**2)))
    outputs = anchor_tiles(patch)
    margin = 8
    for anchor in region["anchors"]:
        output = outputs[anchor["index"]]
        source = read_image(raw_folder / f"source-anchor-{anchor['index']}.png")
        template = blurred(source, 1.5)[margin:-margin, margin:-margin]
        corr = cv2.matchTemplate(blurred(output, 1.5), template, cv2.TM_CCOEFF_NORMED)
        _, score, _, (x, y) = cv2.minMaxLoc(corr)
        if score < 0.66:
            x = y = margin
        ref = blurred(source, .85) - blurred(source, 2.5)
        result = blurred(output, .85) - blurred(output, 2.5)
        span = FOCAL_PATCH - 2 * margin
        reference = ref[margin:-margin, margin:-margin].ravel()
        reference = reference - reference.mean()
        energy = max(1e-6, float(np.dot(reference, reference)))
        for xx, yy, collection in ((margin, margin, fixed_correlations), (x, y, correlations)):
            observed = result[yy:yy+span, xx:xx+span].ravel()
            observed = observed - observed.mean()
            signed = float(np.dot(reference, observed))
            collection.append(signed / max(1e-6, float(np.sqrt(energy*np.dot(observed, observed)))))
        transfers.append(signed / energy)
        shifts.append(float(np.hypot(x-margin, y-margin)))
    if correlations:
        diagnostics.update(edge_correlation=median(correlations), edge_fixed_correlation=median(fixed_correlations),
                           edge_transfer=median(transfers), anchor_shift_px=median(shifts), anchors=len(correlations))
    return diagnostics


def compare(root: Path, split: str, specs: dict) -> None:
    results, rows = {}, []
    canvas = ROI_SIZE
    for view, raw in specs.items():
        methods = {}
        for method, path, run, stage, camera in collect(root, split, view):
            if [item["sha256"] for item in run["inputs"]] != raw["hashes"]:
                raise ValueError(f"Input order or hashes differ in {view} {method}")
            full = read_image(path)
            h, w = full.shape[:2]
            sw, sh = raw["meta"]["source_size"]
            shift = (w-sw)//2, (h-sh)//2
            if camera and (w, h) != (sw, sh):
                raise ValueError(f"Unwarped output size changed in {view} {method}")
            region_results = {}
            folder = root / "comparisons" / split / view / method
            png(folder / "field.png", cv2.resize(full, (756, round(756*h/w)), interpolation=cv2.INTER_AREA))
            for name, region in raw["regions"].items():
                raw_folder = root / region["folder"]
                source = read_image(raw_folder / "source.png")
                x, y = region["xy"]
                expected = x+shift[0], y+shift[1]
                xy, ncc = expected, None
                if not camera and name != "quiet":
                    match, ncc = matches(source, full, expected, 45)
                    if ncc is not None and ncc >= MIN_MATCH:
                        xy = match
                        if name == "wall":
                            shift = xy[0]-x, xy[1]-y
                patch = tile(full, xy)
                if xy[0] < 0 or xy[1] < 0 or patch.shape[:2] != canvas[::-1]:
                    continue
                usable = camera or name == "quiet" or (ncc is not None and ncc >= MIN_MATCH)
                png(folder / f"{name}.png", patch)
                for extra in ("depth", "confidence"):
                    source_map = path.parent / f"{extra}.png"
                    if source_map.is_file():
                        png(folder / f"{name}-{extra}.png", tile(read_image(source_map), xy))
                data = {"camera_xy": [x, y], "output_xy": list(xy), "ncc": ncc, "usable": usable}
                data.update(compare_roi(raw_folder, patch, region) if usable else {})
                region_results[name] = data
                rows.append({"split": split, "view": view, "method": method, "roi": name, **data})
            methods[method] = {"regions": region_results, "elapsed_s": run["elapsed_s"],
                               "rss_mib": run["peak_rss_bytes"]/2**20, "path": path.relative_to(root).as_posix(),
                               "folder": folder.relative_to(root).as_posix(),
                               "sha256": sha256(path), "stage": stage}
        results[view] = methods
        print(f"{split} {view}: {len(methods)} methods compared", flush=True)
    write_json(root / f"{split}-lighting-metrics.json", results)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (root / f"{split}-lighting-metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--split", choices=("dev", "holdout", "coverage"), default="dev")
    args = parser.parse_args()
    cv2.setNumThreads(2)
    root = args.root.resolve()
    compare(root, args.split, inspect_raw(root, args.split))


if __name__ == "__main__":
    main()
