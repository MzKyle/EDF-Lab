"""PollenEDF backend with the shared process lifecycle and artifact checks."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
import shutil
import sys
import tempfile
from pathlib import Path
from threading import Event, Lock
from typing import ClassVar

import cv2
import numpy as np

from .engine import FusionRequest, FusionResult
from .images import FORMATS, Cancelled, absolute_path, check_cancel, png_info, sha256
from .pollen_algorithm import PollenEDFOptions
from .process_runner import run_process
from .records import read_json, write_json


@dataclass(frozen=True)
class PollenEDFCapabilities:
    backend: ClassVar[str] = "pollen-edf"
    version: str
    interpreter: str
    identity_sha256: str
    source_hashes: dict[str, str]
    python: str
    opencv: str
    numpy: str

    def record(self) -> dict:
        return {"backend": self.backend, **asdict(self)}


class PollenEDFEngine:
    def __init__(self) -> None:
        self._capabilities: PollenEDFCapabilities | None = None
        self._active: Event | None = None
        self._lock = Lock()

    def validate_environment(self) -> PollenEDFCapabilities:
        package = Path(__file__).parent
        names = ("pollen_algorithm.py", "pollen_weights.py", "pollen_brightfield.py",
                 "pollen_engine.py", "pollen_worker.py", "images.py")
        hashes = {name: sha256(package / name) for name in names}
        runtime = (sys.version.split()[0], cv2.__version__, np.__version__)
        fingerprint = hashlib.sha256("|".join([*hashes.values(), *runtime]).encode("ascii")).hexdigest()
        self._capabilities = PollenEDFCapabilities("PollenEDF 0.2", sys.executable,
                                                 fingerprint, hashes, *runtime)
        return self._capabilities

    def get_capabilities(self) -> PollenEDFCapabilities:
        return self._capabilities or self.validate_environment()

    def cancel(self) -> None:
        with self._lock:
            if self._active is not None:
                self._active.set()

    def fuse(self, request: FusionRequest) -> FusionResult:
        with self._lock:
            if self._active is not None:
                raise ValueError("同一引擎实例正在运行融合任务。")
            self._active = request.cancel
        try:
            return self._fuse(request)
        finally:
            with self._lock:
                self._active = None

    def _fuse(self, request: FusionRequest) -> FusionResult:
        options = request.options
        if not isinstance(options, PollenEDFOptions):
            raise ValueError("PollenEDF 需要自研引擎选项。")
        options.validate(len(request.inputs))
        if not math.isfinite(request.timeout_s) or request.timeout_s <= 0:
            raise ValueError("融合超时必须是正的有限秒数。")
        if request.input_hashes and len(request.input_hashes) != len(request.inputs):
            raise ValueError("输入哈希数量与输入路径数量不匹配。")
        if any(path.suffix.lower() not in FORMATS for path in request.inputs):
            raise ValueError("输入文件扩展名与支持的格式不匹配。")

        capabilities = self.get_capabilities()
        destination = absolute_path(request.output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        if any(destination.iterdir()):
            raise ValueError(f"融合输出目录必须为空 ({destination})")
        log = destination / "execution.log"
        names = options.output_names()
        command: tuple[str, ...] = ()
        process = None
        files = {}
        try:
            with tempfile.TemporaryDirectory(prefix="pollen-edf-python-") as temporary:
                scratch = Path(temporary)
                worker_input = scratch / "request.json"
                write_json(worker_input, {
                    "paths": [str(absolute_path(path)) for path in request.inputs],
                    "hashes": request.input_hashes, "options": asdict(options),
                })
                command = (sys.executable, "-m", "pollen_edf.pollen_worker", str(worker_input))
                process = run_process(list(command), cwd=scratch, log=log, timeout_s=request.timeout_s,
                                      cancel=request.cancel, progress=request.progress)
                status, diagnostic = process.status, process.diagnostic
                if status == "success":
                    check_cancel(request.cancel)
                    files = self._verify_files(scratch, names)
                    for name in names:
                        check_cancel(request.cancel)
                        partial = destination / f"{name}.part"
                        shutil.copyfile(scratch / name, partial)
                        partial.replace(destination / name)
        except Cancelled as error:
            status, diagnostic = "cancelled", str(error)
        except (OSError, ValueError) as error:
            status, diagnostic = "failed", str(error)
        if status != "success":
            for name in names:
                (destination / name).unlink(missing_ok=True)
                (destination / f"{name}.part").unlink(missing_ok=True)
            files = {}
            with log.open("a", encoding="utf-8") as stream:
                stream.write(f"\n[PollenEDF] {status} | {diagnostic}\n")
        return FusionResult(status, diagnostic, destination / "fused.png" if files else None,
                            destination / "depth.png" if "depth.png" in files else None,
                            log, process.elapsed_s if process else None, process.peak_rss_bytes if process else None,
                            command, capabilities.version, process.returncode if process else None,
                            "cpu" if process else "unknown", ("OpenCV ndarray CPU pipeline",) if process else (), files)

    @staticmethod
    def _verify_files(path: Path, names: tuple[str, ...]) -> dict:
        files = {}
        for name in names:
            incoming = path / name
            if name.endswith(".png"):
                info = png_info(incoming)
                if name != "fused.png" and any(info[key] != files["fused.png"][key] for key in ("width", "height")):
                    raise ValueError(f"{name} 的尺寸与融合图不一致。")
            else:
                read_json(incoming)
                info = {"sha256": sha256(incoming), "size_bytes": incoming.stat().st_size}
            files[name] = info
        return files
