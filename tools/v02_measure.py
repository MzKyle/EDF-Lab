"""Measure V0.2 results against independently selected locations in the raw stacks."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from pollen_edf.dataset import load_dataset
from pollen_edf.evaluation import measure
from pollen_edf.config import ENGINE_PIN
from pollen_edf.images import read_image, sha256
from pollen_edf.records import read_json, write_json


ROI_SIZE = (384, 288)
MIN_MATCH = 0.62
PROBES = ((96, 72), (288, 72), (96, 216), (288, 216))
PRIOR_ROIS = {
    "20261030_140410_v004": [1662, 1033],
    "20261106_163002_v016": [1920, 1928],
}


def png(path: Path, array: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, output = cv2.imencode(".png", array, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise ValueError(f"Cannot encode {path}")
    path.write_bytes(output.tobytes())
    return path.name


def gray(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr


def blurred(bgr, sigma=1.7):
    return cv2.GaussianBlur(gray(bgr).astype(np.float32), (0, 0), sigma)


def tile(image, xy, size=ROI_SIZE):
    x, y = xy
    w, h = size
    return image[y:y+h, x:x+w]


def edge_response(luma):
    weights = np.array([[1, -2, 1]], np.float32)
    x = cv2.filter2D(luma, cv2.CV_32F, weights)
    y = cv2.filter2D(luma, cv2.CV_32F, weights.T)
    return np.abs(x) + np.abs(y)


def locate_regions(stack):
    """Set all ROI locations using raw images alone before viewing outputs."""
    peak = edges = None
    for frame in stack.frames:
        image = cv2.imdecode(np.fromfile(frame.path, np.uint8), cv2.IMREAD_REDUCED_COLOR_4)
        if image is None:
            raise ValueError(f"Unreadable raw thumbnail {frame.path}")
        source = blurred(image, 0.85)
        detail = edge_response(source)
        if peak is None:
            peak, edges = source, detail
        else:
            np.maximum(peak, source, out=peak)
            np.maximum(edges, detail, out=edges)
    height, width = peak.shape
    w, h = (value // 4 for value in ROI_SIZE)
    limits = (w // 2 + 16, h // 2 + 16)
    border = np.zeros_like(peak, dtype=bool)
    border[limits[1]:height-limits[1], limits[0]:width-limits[0]] = True
    # Clipping isolates points of unusually high intensity from the ROI selection.
    energy = cv2.boxFilter(np.minimum(edges, 5.0), -1, (w, h))
    energy[~border] = -1.0
    iy, ix = np.unravel_index(energy.argmax(), energy.shape)
    locations = {}
    if float(energy[iy, ix]) > 0.16 and "near-empty" not in stack.metadata["scene"]:
        locations["detail"] = (int(ix), int(iy))
        # A lower-area component gives a known compact feature to follow across methods.
        threshold = max(6.5, float(np.percentile(peak, 15)) + 4.0)
        _, _, boxes, centroids = cv2.connectedComponentsWithStats(np.uint8(peak >= threshold))
        candidates = []
        for index, box in enumerate(boxes[1:], 1):
            bx, by, bw, bh, area = map(int, box)
            xx, yy = (int(round(value)) for value in centroids[index])
            if not (border[yy, xx] and 16 <= area <= 650 and bw >= 4 and bh >= 4):
                continue
            if abs(xx - ix) < w and abs(yy - iy) < h:
                continue
            maximum = float(peak[by:by+bh, bx:bx+bw].max())
            if maximum >= threshold + 5.0:
                rank = abs(np.log(area / 65.0)) + 0.018 * abs(maximum - 24.0)
                candidates.append((rank, xx, yy))
        if candidates:
            _, xx, yy = min(candidates)
            locations["small"] = xx, yy
        else:
            energy[max(0, iy-h):iy+h, max(0, ix-w):ix+w] = -1.0
            yy, xx = np.unravel_index(energy.argmax(), energy.shape)
            if energy[yy, xx] > 0.08:
                locations["second"] = int(xx), int(yy)
    darkness = cv2.boxFilter(peak, -1, (w, h))
    darkness[~border] = np.inf
    # A dark window near the centre has more stable crop coordinates than a corner.
    gy, gx = np.ogrid[:height, :width]
    darkness += 0.8 * ((gx / width - 0.5)**2 + (gy / height - 0.5)**2)
    yy, xx = np.unravel_index(darkness.argmin(), darkness.shape)
    locations["dark"] = int(xx), int(yy)
    regions = {name: [x * 4 - ROI_SIZE[0] // 2, y * 4 - ROI_SIZE[1] // 2]
               for name, (x, y) in locations.items()}
    if stack.id in PRIOR_ROIS:
        regions["prior"] = PRIOR_ROIS[stack.id]
    return regions


def focus_row(patch, frame):
    g = gray(patch)
    detail = edge_response(blurred(patch, 1.15))
    mean = float(np.mean(detail))
    probes = [float(np.mean(detail[y-32:y+32, x-32:x+32])) for x, y in PROBES]
    w, h = ROI_SIZE
    center = g[h//2-32:h//2+32, w//2-32:w//2+32]
    center_detail = detail[h//2-32:h//2+32, w//2-32:w//2+32]
    return {"sn": frame.z, "z": frame.metadata["archive_z"], "score": mean,
            "luma": float(g.mean()), "probes": probes, "center_score": float(center_detail.mean()),
            "center_p95": float(np.percentile(center, 95))}


def peaks(values):
    upper = max(values)
    if upper < 0.10:
        return []
    hits = [i for i in range(1, len(values)-1) if values[i-1] < values[i] >= values[i+1]
            and values[i] >= upper * 0.56]
    chosen = []
    for index in sorted(hits, key=lambda i: values[i], reverse=True):
        if all(abs(i - index) >= 3 and min(values[min(i, index):max(i, index)+1])
               <= 0.89 * min(values[index], values[i]) for i in chosen):
            chosen.append(index)
    return sorted(chosen)


def focus_plot(folder, rows):
    fig, ax = plt.subplots(figsize=(6.2, 2.7), layout="constrained")
    z = [p["z"] for p in rows]
    ax.plot(z, [p["score"] for p in rows], label="entire ROI", color="#142b3e", lw=2)
    colors = ("#bf682c", "#17877b", "#8453a4", "#4672a5")
    for i, color in enumerate(colors):
        ax.plot(z, [p["probes"][i] for p in rows], label=f"64 px window {i+1}", color=color, alpha=.85)
    ax.plot(z, [p["center_score"] for p in rows], label="centre 64 px", ls="--", color="#667a24", alpha=.8)
    ax.set(xlabel="Z token in filename", ylabel="Smoothed SML / px")
    ax.grid(alpha=.2)
    ax.legend(loc="best", fontsize=7, ncol=2)
    fig.savefig(folder / "focus.png", dpi=135)
    plt.close(fig)


def raw_regions(root, split, dataset):
    results = {}
    for stack in dataset.stacks:
        locations = locate_regions(stack)
        shape = ROI_SIZE[1], ROI_SIZE[0]
        states = {key: {"rows": [], "peak": np.zeros(shape, np.float32), "hits": np.zeros(shape, np.uint16),
                        "best_patch": None, "best_score": -1.0} for key in locations}
        hashes = []
        for frame in stack.frames:
            image = read_image(frame.path, engine_input=True)
            hashes.append(sha256(frame.path))
            for key, xy in locations.items():
                patch = tile(image, xy)
                state = states[key]
                row = focus_row(patch, frame)
                state["rows"].append(row)
                np.maximum(state["peak"], blurred(patch, 1.35), out=state["peak"])
                g = gray(patch)
                base = cv2.medianBlur(g, 3)
                state["hits"] += (base <= 8) & (g.astype(np.int16) - base > 10)
                score = (row["center_score"] + 0.02 * row["center_p95"]) if key == "small" else row["score"]
                if score > state["best_score"]:
                    state.update(best_score=score, best_patch=patch.copy(), best=row)
        view = {"meta": stack.metadata, "raw_hashes": hashes, "regions": {}}
        for key, xy in locations.items():
            state = states[key]
            folder = root / "raw" / split / stack.id / key
            background = cv2.erode(np.uint8(state["peak"] <= 7.0), np.ones((5, 5), np.uint8))
            hot = state["hits"] >= int(np.ceil(0.65 * len(stack.frames)))
            best = state["best_patch"]
            rows = state["rows"]
            signal = blurred(best, 1.0)
            foreground = signal >= max(8.0, float(np.percentile(signal, 98)) * 0.24)
            png(folder / "source.png", best)
            png(folder / "dark-mask.png", background * 255)
            png(folder / "foreground-mask.png", np.uint8(foreground) * 255)
            png(folder / "recurring-points.png", np.uint8(hot) * 255)
            focus_plot(folder, rows)
            by_probe = [[row["probes"][i] for row in rows] for i in range(len(PROBES))]
            probe_peaks = [[rows[i]["z"] for i in peaks(values)] for values in by_probe]
            probe_contrast = [int(np.percentile(signal[y-32:y+32, x-32:x+32], 95)
                                 - np.percentile(signal[y-32:y+32, x-32:x+32], 15)) for x, y in PROBES]
            best_probes = [rows[int(np.argmax(values))]["sn"] for values in by_probe]
            data = {"xy": xy, "best": state["best"], "curve": rows, "probe_peaks": probe_peaks,
                    "probe_contrast": probe_contrast,
                    "boundary_probes": sum(sn in (1, len(stack.frames)) and contrast >= 7
                                           for sn, contrast in zip(best_probes, probe_contrast)),
                    "dark_pixels": int(background.sum()), "recurring_points": int(hot.sum()),
                    "folder": folder.relative_to(root).as_posix()}
            view["regions"][key] = data
        results[stack.id] = view
        print(f"{split} {stack.id} raw ROIs {', '.join(locations)}", flush=True)
    write_json(root / f"{split}-roi-spec.json", results)


def matches(patch, image, expected, margin):
    template = blurred(patch)
    w, h = ROI_SIZE
    if template.shape != (h, w) or min(image.shape[:2]) < min(h, w):
        return expected, None
    tw, th = image.shape[1], image.shape[0]
    x, y = expected
    left, top = max(0, x-margin), max(0, y-margin)
    right, bottom = min(tw, x+w+margin), min(th, y+h+margin)
    if right-left < w or bottom-top < h or template.std() < 1.0:
        return expected, None
    probe = blurred(image[top:bottom, left:right])
    _, score, _, (dx, dy) = cv2.minMaxLoc(cv2.matchTemplate(probe, template, cv2.TM_CCOEFF_NORMED))
    return (left + dx, top + dy), float(score)


def masked(patch, background, foreground, hot, source):
    luma = gray(patch).astype(np.float32)
    values = luma[background]
    m = measure(patch)
    result = {name: m[name] for name in ("tenengrad", "sml_per_pixel", "brightness_mean", "laplacian_variance")}
    result.update(dark_rms=None, dark_p95=None, dark_mean=None, halo_p95=None, halo_mean=None,
                  fg_luma=None, fg_bgr=None, source_edge_corr=None, trail_pixels=None)
    residual = luma - cv2.GaussianBlur(luma, (0, 0), 1.15)
    if len(values) >= 64:
        result.update(dark_rms=float(np.sqrt(np.mean(residual[background]**2))),
                      dark_p95=float(np.percentile(values, 95)), dark_mean=float(values.mean()))
    near_particles = cv2.dilate(foreground.astype(np.uint8), np.ones((41, 41), np.uint8)) != 0
    ring = near_particles & background
    if ring.sum() >= 64:
        result.update(halo_p95=float(np.percentile(luma[ring], 95)), halo_mean=float(np.mean(luma[ring])))
    if foreground.sum() >= 64:
        src = edge_response(blurred(source))[foreground]
        out = edge_response(blurred(patch))[foreground]
        corr = float(np.corrcoef(src, out)[0, 1]) if src.std() > 1e-4 and out.std() > 1e-4 else None
        result.update(fg_luma=float(luma[foreground].mean()), fg_bgr=patch[foreground].mean(axis=0).tolist(),
                      source_edge_corr=corr)
    if hot.any():
        collar = cv2.dilate(hot.astype(np.uint8), np.ones((17, 17), np.uint8)).astype(bool) & ~hot
        result["trail_pixels"] = int(np.count_nonzero(collar & (residual > 2.5)))
    return result


def collect(root, split, view):
    records = []
    reference = root / f"{split}-helicon" / view
    if (reference / "run.json").is_file():
        run = read_json(reference / "run.json")
        if run["status"] == "success" and "png" in run:
            records.append(("Helicon-B", reference / "fused.png", run, "helicon", False))
    for stage in ("native-ofat", "native-locked", "pollen-ofat", "pollen-locked", "pollen-revised", "pollen-fallback"):
        pointer = root / f"{split}-{stage}.json"
        if not pointer.is_file():
            continue
        experiment = Path(read_json(pointer)["path"])
        log = read_json(experiment / "manifest.json")
        for task in log["tasks"]:
            if task["xy"] != view or task["status"] != "success":
                continue
            attempt = task["attempts"][-1]
            if stage == "native-locked" and any(x[0] == task["preset"] for x in records):
                continue
            in_camera_coords = stage.startswith("pollen-") or task["options"].get("alignment") == "none"
            records.append((task["preset"], experiment / attempt["output"], attempt, stage, in_camera_coords))
    return records


def checked_tile(image, xy):
    w, h = ROI_SIZE
    if not (0 <= xy[0] <= image.shape[1]-w and 0 <= xy[1] <= image.shape[0]-h):
        return None
    return tile(image, xy)


def edge_profile(root, split, view, region, methods):
    folder = root / region["folder"]
    raw = read_image(folder / "source.png")
    luma = blurred(raw, 1.5)
    dx = cv2.Sobel(luma, cv2.CV_32F, 1, 0)
    dy = cv2.Sobel(luma, cv2.CV_32F, 0, 1)
    energy = dx**2 + dy**2
    h, w = energy.shape
    ys = np.arange(42, h-42, 4)
    xs = np.arange(42, w-42, 4)
    ygrid, xgrid = np.meshgrid(ys, xs, indexing="ij")
    scores = energy[ygrid, xgrid]
    candidates = np.argsort(scores, axis=None)[-160:][::-1]
    offsets = np.arange(-33, 34, dtype=np.float32)[None, :]
    tangent = np.arange(-4, 5, dtype=np.float32)[:, None]
    for index in candidates:
        iy, ix = np.unravel_index(index, scores.shape)
        x, y = int(xs[ix]), int(ys[iy])
        norm = np.sqrt(energy[y, x])
        if norm < 8.0:
            break
        ux, uy = dx[y, x] / norm, dy[y, x] / norm
        mx = np.float32(x + ux * offsets - uy * tangent)
        my = np.float32(y + uy * offsets + ux * tangent)
        reference = cv2.remap(luma, mx, my, cv2.INTER_LINEAR).mean(axis=0)
        if reference[:10].mean() < reference[-10:].mean() * 0.52 and reference[-10:].mean() > 12.0:
            break
    else:
        return
    if norm < 8.0:
        return
    colors = {"Raw": "#1a283a", "Helicon-B": "#258173", "FS_Default": "#c86a33", "FS_Tuned": "#a48d1c",
              "PollenEDF-Adaptive": "#a23b52"}
    if "PollenEDF-Adaptive-GuardFallback" in methods:
        colors["PollenEDF-Adaptive-GuardFallback"] = "#5068a8"
    profiles = {"Raw": cv2.remap(gray(raw).astype(np.float32), mx, my, cv2.INTER_LINEAR).mean(axis=0).tolist()}
    for method in colors:
        if method not in methods:
            continue
        entry = methods[method]
        filename = f'{region["kind"]}.png'
        if not entry["regions"][region["kind"]]["usable"] or not (root / entry["folder"] / filename).is_file():
            continue
        image = gray(read_image(root / entry["folder"] / filename)).astype(np.float32)
        profiles[method] = cv2.remap(image, mx, my, cv2.INTER_LINEAR).mean(axis=0).tolist()
    fig, ax = plt.subplots(figsize=(6.5, 2.9), layout="constrained")
    for method, samples in profiles.items():
        ax.plot(offsets.ravel(), samples, color=colors[method], label=method, lw=2 if method == "Raw" else 1.4)
    ax.axvline(0, color="grey", alpha=.25)
    ax.set(xlabel="Distance along the raw-image edge normal (px)", ylabel="Mean grayscale")
    ax.grid(alpha=.2)
    ax.legend(loc="best", fontsize=7, ncol=2)
    fig.savefig(folder / "edge-profile.png", dpi=140)
    plt.close(fig)
    overlay = raw.copy()
    cv2.line(overlay, (round(x-ux*33), round(y-uy*33)), (round(x+ux*33), round(y+uy*33)), (18, 176, 235), 1)
    png(folder / "edge-profile-location.png", overlay)
    write_json(folder / "edge-profile.json", {"split": split, "view": view, "xy_in_roi": [x, y],
               "direction": [float(ux), float(uy)], "distances_px": offsets.ravel().tolist(), "profiles": profiles})


def measure_gap(root, data, specs):
    view = "20261029_144803_v061"
    if "PollenEDF-Adaptive" not in data.get(view, {}):
        return
    points = {"gap": [(208, 150), (255, 269), (192, 127)],
              "grain": [(107, 75), (290, 99), (330, 175)]}
    region = specs[view]["regions"]["detail"]
    source = read_image(root / region["folder"] / "source.png")
    stack = next(stack for stack in load_dataset(root / "inputs" / "dev.json").stacks if stack.id == view)

    def probes(image):
        g = gray(image)
        return {name: [float(np.median(g[y-3:y+4, x-3:x+4])) for x, y in coords]
                for name, coords in points.items()}

    curves = [probes(tile(read_image(frame.path, engine_input=True), region["xy"])) for frame in stack.frames]
    z = [frame.metadata["archive_z"] for frame in stack.frames]
    samples = {"raw_anchor": probes(source)}
    for name in ("Helicon-B", "FS_Default", "FS_Tuned", "PollenEDF-Adaptive", "PollenEDF-Adaptive-Guard",
                 "PollenEDF-Adaptive-GuardPool", "PollenEDF-Adaptive-GuardFallback"):
        if name in data[view]:
            samples[name] = probes(read_image(root / data[view][name]["folder"] / "detail.png"))
    directory = root / "gap-review"
    directory.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 3.3), layout="constrained")
    palette = {"gap": "#a23f39", "grain": "#21836c"}
    for name, coords in points.items():
        for k, (x, y) in enumerate(coords, 1):
            ax.plot(z, [curve[name][k-1] for curve in curves], label=f"{name} {k}", color=palette[name],
                    alpha=1.0-0.19*(k-1))
            color = (44, 148, 251) if name == "gap" else (115, 210, 40)
            cv2.circle(source, (x, y), 6, color, 1)
            cv2.putText(source, f"{name[0]}{k}", (x+8, y-6), cv2.FONT_HERSHEY_SIMPLEX, .41, color, 1, cv2.LINE_AA)
    ax.grid(alpha=.24)
    ax.set(xlabel="Original focal plane Z token", ylabel="Median grayscale in 7 × 7 px")
    ax.legend(ncol=3, fontsize=8)
    fig.savefig(directory / "curves.png", dpi=150)
    plt.close(fig)
    png(directory / "locations.png", source)
    write_json(directory / "measurement.json", {"view": view, "positions": points, "z": z,
                                                "luminance_curves": curves, "samples": samples})


def measure_holdout_failure(root, data, specs):
    """Inspect the two visibly mottled windows in the held-out low-contrast field."""
    view = "20261103_160942_v061"
    fused = data.get(view, {}).get("PollenEDF-Adaptive")
    if fused is None:
        return
    region = specs[view]["regions"]["detail"]
    windows = {"grain_surface": [30, 174, 60, 58], "intergrain_gap": [94, 166, 50, 64]}
    output = (root / fused["path"]).parent
    state = read_json(output / "stages.json")
    records = {}
    overlay = read_image(root / fused["folder"] / "detail.png")
    layers = {name: tile(read_image(output / f"{name}.png"), region["xy"])
              for name in ("depth-index", "confidence", "halo", "anchor")}
    for index, (name, xywh) in enumerate(windows.items(), 1):
        x, y, w, h = xywh
        values = {layer: tile(array, (x, y), (w, h)) for layer, array in layers.items()}
        records[name] = {"coords_in_roi": xywh,
                         "label_index_p10_p90": np.percentile(values["depth-index"], [10, 90]).tolist(),
                         **{field: float(np.median(values[layer]) / 255.0) for field, layer in
                            (("local_confidence_median", "confidence"), ("halo_mass_median", "halo"),
                             ("anchor_mass_median", "anchor"))}}
        color = (75, 201, 255) if index == 1 else (214, 220, 43)
        cv2.rectangle(overlay, (x, y), (x+w-1, y+h-1), color, 1)
        cv2.putText(overlay, str(index), (x+3, y-4), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1, cv2.LINE_AA)
    directory = root / "holdout-review"
    png(directory / "locations.png", overlay)
    stack = next(item for item in load_dataset(root / "inputs" / "holdout.json").stacks if item.id == view)
    curves = []
    for frame in stack.frames:
        patch = tile(read_image(frame.path, engine_input=True), region["xy"])
        edges = edge_response(blurred(patch, 1.15))
        luma = gray(patch)
        curves.append({"z": frame.metadata["archive_z"], **{
            name: {"score": float(tile(edges, (x, y), (w, h)).mean()),
                   "luma": float(np.median(tile(luma, (x, y), (w, h))))}
            for name, (x, y, w, h) in windows.items()}})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), layout="constrained")
    for index, (name, color) in enumerate(zip(windows, ("#99712f", "#21788f")), 1):
        for axis, metric in zip(axes, ("score", "luma")):
            axis.plot([row["z"] for row in curves], [row[name][metric] for row in curves],
                      label=f"Window {index}", color=color)
    for axis, label in zip(axes, ("Smoothed SML / px", "Median grayscale")):
        axis.set(xlabel="Original focal plane Z token", ylabel=label)
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    fig.savefig(directory / "curves.png", dpi=140)
    plt.close(fig)
    write_json(root / "holdout-low-contrast-failure.json", {
        "view": view, "foreground_area": state["foreground_area"], "high_confidence_area": state["high_confidence_area"],
        "single_plane_fallback_index": state["single_plane_fallback_index"],
        "observations": records, "raw_z_curves": curves})


def compare(root, split):
    specs = read_json(root / f"{split}-roi-spec.json")
    result, table = {}, []
    for view, source_state in specs.items():
        regions = source_state["regions"]
        w, h = source_state["meta"]["source_size"]
        fixed = {}
        for key, region in regions.items():
            raw = root / region["folder"]
            fixed[key] = (read_image(raw / "source.png"),
                          read_image(raw / "dark-mask.png") > 0,
                          read_image(raw / "foreground-mask.png") > 0,
                          read_image(raw / "recurring-points.png") > 0)
        view_rows = {}
        for method, file, run, stage, camera_coords in collect(root, split, view):
            supplied = [item["sha256"] for item in run["inputs"]]
            if supplied != source_state["raw_hashes"]:
                raise ValueError(f"Ordered inputs differ for {view} {method}")
            output = read_image(file)
            height, width = output.shape[:2]
            if camera_coords and (width, height) != (w, h):
                raise ValueError(f"Unwarped output dimensions differ from camera coordinates for {view} {method}")
            shift = ((width-w)//2, (height-h)//2)
            local = {}
            elapsed, rss = run["elapsed_s"], run["peak_rss_bytes"]
            folder = root / "comparisons" / split / view / method
            preview = cv2.resize(output, (756, round(height * 756 / width)), interpolation=cv2.INTER_AREA)
            png(folder / "field.png", preview)
            maps = {name: read_image(file.parent / f"{name}.png")
                    for name in ("depth", "confidence", "foreground", "alternate", "halo", "broad", "anchor")
                    if (file.parent / f"{name}.png").is_file()}
            for key, region in regions.items():
                raw, bg, fg, hot = fixed[key]
                x, y = region["xy"]
                expected = x + shift[0], y + shift[1]
                xy, ncc = expected, None
                if camera_coords:
                    coordinate_mode = "camera"
                elif key == "dark":
                    coordinate_mode = "field_offset"
                else:
                    matched, ncc = matches(raw, output, expected, 46)
                    if ncc is not None and ncc < MIN_MATCH:
                        matched, ncc = matches(raw, output, expected, 130)
                    coordinate_mode = "weak_match"
                    if ncc is not None and ncc >= MIN_MATCH:
                        xy, coordinate_mode = matched, "integer_match"
                        if key == "detail":
                            shift = xy[0] - x, xy[1] - y
                patch = checked_tile(output, xy)
                record = {"camera_xy": [x, y], "xy": list(xy), "coordinate_mode": coordinate_mode,
                          "ncc": ncc, "usable": False, "metrics": {}}
                if patch is not None:
                    png(folder / f"{key}.png", patch)
                    record.update(usable=coordinate_mode != "weak_match", metrics=masked(patch, bg, fg, hot, raw))
                    for name, map_image in maps.items():
                        layer = checked_tile(map_image, xy)
                        if layer is not None:
                            png(folder / f"{key}-{name}.png", layer)
                local[key] = record
                table.append({"split": split, "view": view, "method": method, "region": key, "usable": record["usable"],
                              "camera_x": x, "camera_y": y, "output_x": xy[0], "output_y": xy[1],
                              "coordinate_mode": coordinate_mode, "ncc": ncc, "elapsed_s": elapsed, "rss_mib": rss / 2**20,
                              "width": width, "height": height, **record["metrics"]})
            view_rows[method] = {"regions": local, "elapsed_s": elapsed, "rss_mib": rss / 2**20,
                                 "width": width, "height": height, "path": file.relative_to(root).as_posix(),
                                 "folder": folder.relative_to(root).as_posix(), "stage": stage,
                                 "sha256": sha256(file)}
        result[view] = view_rows
        for key in ("detail", "prior"):
            if key in regions and view_rows:
                edge_profile(root, split, view, {**regions[key], "kind": key}, view_rows)
        print(f"{split} {view}: {len(view_rows)} methods", flush=True)
    write_json(root / f"{split}-metrics.json", result)
    if split == "dev":
        measure_gap(root, result, specs)
    else:
        measure_holdout_failure(root, result, specs)
    columns = list(dict.fromkeys(key for row in table for key in row))
    with (root / f"{split}-measurements.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        exporter = csv.DictWriter(stream, fieldnames=columns)
        exporter.writeheader()
        exporter.writerows(table)


def audit(root):
    lock = read_json(root / "locked.json")
    if sha256(root / "inputs" / "holdout.json") != lock["holdout_index_sha256"]:
        raise ValueError("The holdout index differs from the parameter lock.")
    data = {split: load_dataset(root / "inputs" / f"{split}.json") for split in ("dev", "holdout")}
    dates = {split: {stack.metadata["date"] for stack in item.stacks} for split, item in data.items()}
    if dates["dev"] & dates["holdout"]:
        raise ValueError("The development and holdout acquisition dates overlap.")
    spec = {split: read_json(root / f"{split}-roi-spec.json") for split in data}
    ordered = {}
    for split, dataset in data.items():
        for stack in dataset.stacks:
            ordered[stack.id] = [sha256(frame.path) for frame in stack.frames]
            if ordered[stack.id] != spec[split][stack.id]["raw_hashes"]:
                raise ValueError(f"Original frames differ from the evaluation record for {stack.id}")

    pointers = {
        "dev-native-ofat": 70, "dev-native-locked": 5, "dev-pollen-ofat": 55, "dev-pollen-revised": 10,
        "dev-pollen-fallback": 5,
        "holdout-native-locked": 8, "holdout-pollen-locked": 16,
    }
    evidence = {}
    for name, expected in pointers.items():
        experiment = Path(read_json(root / f"{name}.json")["path"])
        manifest = read_json(experiment / "manifest.json")
        engine = manifest["engine"]
        if len(manifest["tasks"]) != expected or manifest["status"] != "completed":
            raise ValueError(f"Incomplete comparison {name}")
        if "native" in name and engine["binary_sha256"] != ENGINE_PIN["binary_sha256"]:
            raise ValueError(f"The native executable differs from the official baseline in {name}")
        if name.startswith("holdout"):
            if manifest["created_at"] < lock["frozen_at"]:
                raise ValueError(f"Run predates the parameter lock {name}")
            if "pollen" in name and engine["identity_sha256"] != lock["pollen_identity_sha256"]:
                raise ValueError("The holdout PollenEDF identity differs from the lock.")
        for task in manifest["tasks"]:
            latest = task["attempts"][-1]
            if name == "holdout-native-locked" and task["preset"] == "FS_Tuned":
                if any(task["options"][key] != value for key, value in lock["focus_stack"].items()):
                    raise ValueError("Frozen native options differ in the holdout run.")
            if name == "holdout-pollen-locked" and task["preset"] == "PollenEDF-Adaptive":
                if task["options"] != lock["frozen_adaptive_options"]:
                    raise ValueError("Frozen adaptive options differ in the holdout run.")
            if [item["sha256"] for item in latest["inputs"]] != ordered[task["xy"]]:
                raise ValueError(f"Input mismatch {name} {task['xy']} {task['preset']}")
            for file, metadata in latest["files"].items():
                path = experiment / latest["dir"] / file
                if sha256(path) != metadata["sha256"]:
                    raise ValueError(f"Invalid result hash {path}")
        evidence[name] = {"tasks": expected, "manifest": str(experiment / "manifest.json")}

    diagnostics = read_json(root / "native-diagnostics.json")
    diagnostics_root = Path(diagnostics["experiment"])
    actual = read_json(diagnostics_root / "manifest.json")
    if actual["status"] != "completed" or actual["engine"]["binary_sha256"] != ENGINE_PIN["binary_sha256"]:
        raise ValueError("The native stage diagnostics are incomplete or use a different binary.")

    exact_pixels = {}
    holdout = read_json(root / "holdout-metrics.json")
    for split, dataset in data.items():
        for stack in dataset.stacks:
            helicon = root / f"{split}-helicon" / stack.id
            reference = read_json(helicon / "run.json")
            if ([row["sha256"] for row in reference["inputs"]] != ordered[stack.id]
                    or sha256(helicon / "fused.png") != reference["png"]["sha256"]
                    or sha256(helicon / "ordered.tiff") != reference["tiff_sha256"]
                    or reference["tiff_compression"] != 1):
                raise ValueError(f"Helicon reference differs for {stack.id}")
            if split != "holdout":
                continue
            if reference["started_at"] < lock["frozen_at"]:
                raise ValueError(f"Helicon reference predates the holdout lock for {stack.id}")
            if set(holdout[stack.id]) != {"Helicon-B", "FS_Default", "FS_Tuned", "PollenEDF-SML",
                                         "PollenEDF-Tenengrad", "PollenEDF-DepthMap", "PollenEDF-Adaptive"}:
                raise ValueError(f"Required holdout methods missing on {stack.id}")
            file = root / holdout[stack.id]["PollenEDF-Adaptive"]["path"]
            result = read_image(file)
            labels = read_image(file.with_name("depth-index.png"))
            high = ((read_image(file.with_name("confidence.png")) >= 156)
                    & (read_image(file.with_name("foreground.png")) >= 252))
            checked = 0
            for index, frame in enumerate(stack.frames, 1):
                selected = high & (labels == index)
                if not selected.any():
                    continue
                original = read_image(frame.path, engine_input=True)
                if not np.array_equal(result[selected], original[selected]):
                    raise ValueError(f"High-confidence pixels differ from their selected input in {stack.id}")
                checked += int(selected.sum())
            if checked != int(high.sum()):
                raise ValueError(f"Invalid layer indices for {stack.id}")
            exact_pixels[stack.id] = checked
    write_json(root / "holdout-checks.json", {
        "archived_originals_verified": sum(len(values) for values in ordered.values()),
        "acquisition_dates": {key: sorted(values) for key, values in dates.items()},
        "input_order_and_output_hashes": "verified", "runs": evidence,
        "helicon_verified": len(ordered), "diagnostic_native_jobs": len(actual["tasks"]),
        "exact_high_confidence_bgr_pixels_by_holdout_view": exact_pixels,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("raw", "compare", "audit"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--split", choices=("dev", "holdout"), default="dev")
    args = parser.parse_args()
    cv2.setNumThreads(2)
    root = args.root.resolve()
    if args.stage == "raw":
        raw_regions(root, args.split, load_dataset(root / "inputs" / f"{args.split}.json"))
    elif args.stage == "audit":
        audit(root)
    else:
        compare(root, args.split)


if __name__ == "__main__":
    main()
