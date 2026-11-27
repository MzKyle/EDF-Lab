"""Subprocess entry point for the project's Python fusion algorithm."""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import cv2
import numpy as np

from .images import absolute_path, decode
from .pollen_algorithm import PollenEDFOptions, diagnostic_images, render_fusion, select_focus
from .records import read_json, write_json


def _write_png(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not ok:
        raise ValueError(f"无法保存 PNG ({path})")
    path.write_bytes(encoded.tobytes())


def _load(path: Path, expected_hash: str | None) -> np.ndarray:
    data = path.read_bytes()
    if expected_hash and hashlib.sha256(data).hexdigest() != expected_hash:
        raise ValueError(f"预检后输入文件发生变化 ({path})")
    image = decode(data, path, engine_input=True, preserve_depth=True)
    if image.dtype == np.uint16:
        image = np.rint(image.astype(np.float32) * (255.0 / 65535)).astype(np.uint8)
    if image.dtype != np.uint8:
        raise ValueError(f"PollenEDF 需要 8 位或 16 位无符号输入 ({path})")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def run(path: Path) -> None:
    spec = read_json(path)
    paths = [absolute_path(name) for name in spec["paths"]]
    hashes = spec["hashes"] or [None] * len(paths)
    if len(hashes) != len(paths):
        raise ValueError("输入哈希数量与输入路径数量不匹配。")
    options = PollenEDFOptions(**spec["options"])
    count = len(paths)
    options.validate(count)
    cv2.ocl.setUseOpenCL(False)
    cv2.setNumThreads(options.threads)
    print(f"PollenEDF · CPU · OpenCV {cv2.__version__} · {cv2.getNumThreads()} OpenCV threads", flush=True)

    def load(index: int) -> np.ndarray:
        return _load(paths[index], hashes[index])

    def progress(message: str) -> None:
        print(message, flush=True)

    first = load(0)
    size = (int(first.shape[1]), int(first.shape[0]))
    del first
    selection = select_focus(load, size, count, options, progress)
    output = render_fusion(load, size, count, selection, options, progress)
    _write_png(path.parent / "fused.png", output)
    del output
    if options.intermediate:
        write_json(path.parent / "stages.json", selection.stages)
    for name, image in diagnostic_images(selection, size, count, options).items():
        _write_png(path.parent / name, image)
    progress("saved " + ", ".join(options.output_names()))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("pollen_worker requires one request JSON path")
    run(Path(sys.argv[1]))
