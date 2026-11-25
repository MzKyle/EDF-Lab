from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

from .images import absolute_path


def generate(output: Path, *, width: int = 384, height: int = 256, layers: int = 12,
             views: int = 2, seed: int = 2026, shift: float = 0, scale: float = 0,
             brightness: float = 0, noise: float = 1.5, faults: bool = False) -> Path:
    if min(width, height) < 64 or layers < 2 or views < 1 or seed < 0:
        raise ValueError("模拟图像至少为 64 × 64、两层和一个视野，随机种子必须非负。")
    if not all(math.isfinite(value) and value >= 0 for value in (shift, scale, brightness, noise)) or scale >= 1:
        raise ValueError("模拟扰动必须为非负有限数，倍率扰动小于 1。")
    root = absolute_path(output)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError(f"模拟数据输出目录必须为空 ({root})")
    metadata = {"provenance": "SYNTHETIC", "generator": "pollen-edf-lab/0.1.0", "seed": seed,
                "width": width, "height": height, "layers": layers, "views": views,
                "shift_px": shift, "scale_fraction": scale, "brightness_fraction": brightness,
                "noise_sigma": noise, "faults": faults}
    (root / "SYNTHETIC.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    rng = np.random.default_rng(seed)
    rows = []
    expected = {}
    for view_number in range(views):
        blank = views > 1 and view_number == views - 1
        view = f"XY_00_{view_number:02d}"
        directory = root / view
        directory.mkdir()
        expected[view] = {"first_z": 1, "last_z": layers, "blank": blank}
        for index in range(layers):
            z = index + 1
            name = f"Z_{z:03d}.jpg"
            image = _frame(width, height, index / (layers - 1), blank, rng, shift, scale, brightness, noise)
            encoded, payload = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 94])
            if not encoded:
                raise OSError("模拟 JPEG 编码失败。")
            (directory / name).write_bytes(payload.tobytes())
            rows.append({"xy": view, "z": z, "z_um": index * 0.75, "path": f"{view}/{name}"})
    if faults:
        view = "XY_faults"
        directory = root / view
        directory.mkdir()
        first_frame = (root / rows[0]["path"]).read_bytes()
        for number, raw in ((1, first_frame), (3, b"SYNTHETIC deliberately damaged JPEG\n")):
            name = f"Z_{number:03d}.jpg"
            (directory / name).write_bytes(raw)
            rows.append({"xy": view, "z": number, "path": f"{view}/{name}"})
        expected[view] = {"first_z": 1, "last_z": 3}
    name = root.name if root.name.upper().startswith("SYNTHETIC") else f"SYNTHETIC {root.name}"
    index = {"name": name, "metadata": metadata, "views": expected, "frames": rows}
    (root / "dataset.json").write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    return root


def _frame(width: int, height: int, z: float, blank: bool, rng: np.random.Generator,
           shift: float, scale: float, brightness: float, noise: float) -> np.ndarray:
    field = np.full((height, width), 126.0, dtype=np.float32)
    if not blank:
        for cx, cy, rx, ry, target, strength, period in (
            (0.26, 0.29, 0.16, 0.12, 0.20, 54, 5), (0.63, 0.64, 0.21, 0.18, 0.51, 27, 9),
            (0.73, 0.27, 0.09, 0.09, 0.78, 13, 6),
        ):
            _paint(field, cx, cy, rx, ry, abs(z - target), strength, period)
    if shift or scale:
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 0, 1 + scale * (2 * z - 1))
        matrix[0, 2] += shift * (2 * z - 1)
        matrix[1, 2] += shift * math.sin(z * math.pi) / 2
        field = cv2.warpAffine(field, matrix, (width, height), borderMode=cv2.BORDER_REFLECT)
    field *= 1 + brightness * (2 * z - 1)
    if noise:
        field += rng.standard_normal(field.shape, dtype=np.float32) * noise
    np.clip(field, 0, 247, out=field)
    gray = field.astype(np.uint8)
    return cv2.merge((gray - np.minimum(gray, 7), gray, gray + 8))


def _paint(field: np.ndarray, cx: float, cy: float, rx: float, ry: float,
           distance: float, strength: int, period: int) -> None:
    height, width = field.shape
    radius_x, radius_y = int(width * rx), int(height * ry)
    padding = 28
    box_width, box_height = (radius_x + padding) * 2 + 1, (radius_y + padding) * 2 + 1
    x, y = np.meshgrid(np.arange(box_width, dtype=np.float32) - box_width // 2,
                       np.arange(box_height, dtype=np.float32) - box_height // 2, sparse=True)
    gate = (x / radius_x) ** 2 + (y / radius_y) ** 2 <= 1
    bars = np.sin(x * (math.tau / period)) * np.cos(y * (math.tau / (period + 2)))
    patch = (gate * (16 + strength * bars)).astype(np.float32)
    cv2.GaussianBlur(patch, (0, 0), 0.35 + 8 * distance, dst=patch)
    left, top = int(width * cx) - box_width // 2, int(height * cy) - box_height // 2
    x0, y0 = max(left, 0), max(top, 0)
    x1, y1 = min(left + box_width, width), min(top + box_height, height)
    field[y0:y1, x0:x1] += patch[y0 - top:y1 - top, x0 - left:x1 - left]
