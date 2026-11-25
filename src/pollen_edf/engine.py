from __future__ import annotations

import hashlib
import math
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Event, Lock
from typing import Protocol

from .config import ENGINE_PIN
from .images import FORMATS, Cancelled, absolute_path, check_cancel, png_info, sha256
from .process_runner import Progress, log_tail, run_process


@dataclass(frozen=True)
class EngineCapabilities:
    executable: Path
    release: str
    commit: str
    version: str
    binary_sha256: str
    source: str
    options: frozenset[str]
    help_text: str
    opencv_build: str
    opencl_built: bool

    def record(self) -> dict:
        return {**asdict(self), "executable": str(self.executable), "options": sorted(self.options)}


@dataclass(frozen=True)
class EngineOptions:
    alignment: str = "default"
    whitebalance: bool = True
    contrast: bool = True
    transform: bool = True
    full_resolution_align: bool = False
    consistency: int | None = None
    denoise: float | None = None
    threads: int | None = None
    batch_size: int | None = None
    reference: int | None = None
    crop: str = "auto"
    depth: bool = False
    intermediate: bool = False
    acceleration: str = "auto"

    def arguments(self, count: int, capabilities: EngineCapabilities) -> list[str]:
        if count < 2:
            raise ValueError("focus-stack 至少需要两层输入。")
        if self.alignment not in {"default", "none", "global"}:
            raise ValueError("alignment 必须为 default、none 或 global。")
        if self.crop not in {"auto", "none", "keep-size"}:
            raise ValueError("crop 必须为 auto、none 或 keep-size。")
        if self.acceleration not in {"auto", "cpu", "opencl"}:
            raise ValueError("acceleration 必须为 auto、cpu 或 opencl。")
        if self.acceleration == "opencl" and not capabilities.opencl_built:
            raise ValueError("当前引擎的 OpenCV 构建未声明 OpenCL 支持。")
        for name in ("whitebalance", "contrast", "transform", "full_resolution_align", "depth", "intermediate"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} 必须为布尔值。")
        if self.alignment == "none" and (not (self.whitebalance and self.contrast and self.transform)
                                        or self.full_resolution_align or self.crop == "keep-size"):
            raise ValueError("关闭配准时不能同时设置几何或光度配准选项。")
        if self.full_resolution_align and not self.transform:
            raise ValueError("全分辨率几何配准需要启用几何变换。")
        args = ["--output=fused.png", "--verbose"]
        flags = {
            "--no-align": self.alignment == "none", "--global-align": self.alignment == "global",
            "--no-whitebalance": not self.whitebalance, "--no-contrast": not self.contrast,
            "--no-transform": not self.transform, "--full-resolution-align": self.full_resolution_align,
            "--nocrop": self.crop == "none", "--align-keep-size": self.crop == "keep-size",
            "--depthmap=depth.png": self.depth, "--save-steps": self.intermediate,
            "--no-opencl": self.acceleration == "cpu",
        }
        args.extend(flag for flag, enabled in flags.items() if enabled)
        integers = (("consistency", "--consistency", 0, 2), ("threads", "--threads", 1, 2**31 - 1),
                    ("batch_size", "--batchsize", 1, 2**31 - 1), ("reference", "--reference", 0, count - 1))
        for name, flag, minimum, maximum in integers:
            value = getattr(self, name)
            if value is not None:
                if type(value) is not int or not minimum <= value <= maximum:
                    raise ValueError(f"{name} 必须是 [{minimum}, {maximum}] 内的整数。")
                args.append(f"{flag}={value}")
        if self.denoise is not None:
            if (type(self.denoise) not in (int, float) or not 0 <= abs(self.denoise) <= 3.4028234e38 or
                    0 < abs(self.denoise) < 1.1754944e-38):
                raise ValueError("denoise 必须可由 CLI 原生 float 解析为有限数。")
            args.append(f"--denoise={self.denoise:.9g}")
        unsupported = sorted({arg.split("=", 1)[0] for arg in args} - capabilities.options)
        if unsupported:
            raise ValueError(f"当前引擎 --help 未提供参数 {', '.join(unsupported)}")
        return args


@dataclass(frozen=True)
class FusionRequest:
    inputs: tuple[Path, ...]
    output_dir: Path
    options: EngineOptions = field(default_factory=EngineOptions)
    timeout_s: float = 1200.0
    cancel: Event = field(default_factory=Event)
    progress: Progress | None = None
    input_hashes: tuple[str, ...] = ()


@dataclass(frozen=True)
class FusionResult:
    status: str
    diagnostic: str
    output: Path | None
    depth: Path | None
    log: Path
    elapsed_s: float | None
    peak_rss_bytes: int | None
    command: tuple[str, ...]
    engine_version: str
    exit_code: int | None
    acceleration: str
    acceleration_evidence: tuple[str, ...]
    files: dict = field(default_factory=dict)


class FusionEngine(Protocol):
    def validate_environment(self) -> EngineCapabilities: ...
    def get_capabilities(self) -> EngineCapabilities: ...
    def fuse(self, request: FusionRequest) -> FusionResult: ...
    def cancel(self) -> None: ...


def _probe(exe: Path, parameter: str) -> tuple[int, str]:
    options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    process = subprocess.run([str(exe), parameter], executable=str(exe), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, timeout=10, check=False, **options)
    return process.returncode, process.stdout.decode("utf-8", errors="replace").strip()


def _stage(source: Path, target: Path, expected: str | None, cancel: Event) -> None:
    digest = hashlib.sha256()
    with absolute_path(source).open("rb") as original, target.open("xb") as output:
        for block in iter(lambda: original.read(1024 * 1024), b""):
            check_cancel(cancel)
            digest.update(block)
            output.write(block)
    if expected is not None and digest.hexdigest() != expected:
        raise ValueError(f"预检后输入文件发生变化 ({source})")


def _opencl_evidence(log: Path) -> tuple[str, ...]:
    evidence = []
    with log.open("rb") as stream:
        for line in stream:
            for part in line.split(b"\r"):
                if b"OpenCL" in part:
                    evidence.append(part.strip().decode("utf-8", errors="replace")[:400])
                    if len(evidence) >= 30:
                        return tuple(evidence)
    return tuple(evidence)


class FocusStackEngine:
    def __init__(self, executable: Path):
        self.executable = absolute_path(executable)
        self._capabilities: EngineCapabilities | None = None
        self._active: Event | None = None
        self._lock = Lock()

    def validate_environment(self) -> EngineCapabilities:
        exe = self.executable
        if not exe.is_file():
            raise ValueError(f"未找到 focus-stack ({exe})。请运行 scripts/setup_focus_stack.ps1 或设置 POLLEN_EDF_ENGINE。")
        digest = sha256(exe)
        version_code, version_output = _probe(exe, "--version")
        match = re.search(r"^focus-stack\s+(\S+?),?\s+built\b", version_output, re.M)
        if version_code or not match:
            raise ValueError(f"focus-stack --version 执行失败，退出码 {version_code}。{version_output}")
        version = match[1].rstrip(",")
        if digest != ENGINE_PIN["binary_sha256"] and ENGINE_PIN["commit"][:7] not in version:
            raise ValueError(f"可执行文件版本 {version} 与固定提交 {ENGINE_PIN['commit']} 不匹配。")
        help_code, help_text = _probe(exe, "--help")
        options = frozenset(re.findall(r"^\s+(--[a-z0-9-]+)(?=[=\s]|$)", help_text, re.M))
        if help_code not in (0, 1) or "Usage:" not in help_text or "--output" not in options:
            raise ValueError(f"无法读取 focus-stack --help，退出码 {help_code}。{help_text}")
        build_code, build = _probe(exe, "--opencv-version") if "--opencv-version" in options else (1, "")
        if build_code:
            build = ""
        official = digest == ENGINE_PIN["binary_sha256"]
        self._capabilities = EngineCapabilities(
            exe, ENGINE_PIN["release"], ENGINE_PIN["commit"], version, digest,
            ENGINE_PIN["archive_url"] if official else "https://github.com/PetteriAimonen/focus-stack/tree/" + ENGINE_PIN["commit"],
            options, help_text, build, bool(re.search(r"^\s*OpenCL:\s+YES\b", build, re.M)),
        )
        return self._capabilities

    def get_capabilities(self) -> EngineCapabilities:
        return self._capabilities or self.validate_environment()

    def cancel(self) -> None:
        with self._lock:
            if self._active:
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
        capabilities = self.get_capabilities()
        count = len(request.inputs)
        args = request.options.arguments(count, capabilities)
        if not math.isfinite(request.timeout_s) or request.timeout_s <= 0:
            raise ValueError("融合超时必须是正的有限数。")
        if request.input_hashes and len(request.input_hashes) != count:
            raise ValueError("输入哈希数量与输入路径数量不匹配。")
        if any(path.suffix.lower() not in FORMATS for path in request.inputs):
            raise ValueError("输入文件扩展名与支持的格式不匹配。")
        output_dir = absolute_path(request.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if any(output_dir.iterdir()):
            raise ValueError(f"融合输出目录必须为空 ({output_dir})")
        log = output_dir / "execution.log"
        log.touch(exist_ok=False)
        names = [f"frame_{index:06d}{path.suffix.lower()}" for index, path in enumerate(request.inputs)]
        command: tuple[str, ...] = ()
        result = None
        evidence: tuple[str, ...] = ()
        acceleration = "unknown"
        output, depth = None, None
        files = {}
        try:
            with tempfile.TemporaryDirectory(prefix="pollen-edf-") as temporary:
                scratch = Path(temporary)
                for index, (source, name) in enumerate(zip(request.inputs, names)):
                    check_cancel(request.cancel)
                    if request.progress:
                        request.progress(f"暂存输入 {index + 1}/{count}")
                    expected = request.input_hashes[index] if request.input_hashes else None
                    _stage(source, scratch / name, expected, request.cancel)
                command = (str(self.executable), *args, *names)
                result = run_process(list(command), cwd=scratch, log=log, timeout_s=request.timeout_s,
                                     cancel=request.cancel, progress=request.progress)
                evidence = _opencl_evidence(log)
                if any("OpenCL disabled" in item for item in evidence):
                    acceleration = "cpu_forced"
                elif any("OpenCL device:" in item for item in evidence):
                    acceleration = "opencl_device_reported"
                elif any("OpenCL not available" in item or "no devices available" in item for item in evidence):
                    acceleration = "opencl_unavailable"
                if result.status != "success":
                    status, diagnostic = result.status, result.diagnostic
                else:
                    check_cancel(request.cancel)
                    if request.options.acceleration == "opencl" and acceleration != "opencl_device_reported":
                        raise ValueError("请求 OpenCL，但原生日志没有报告可用设备。")
                    selected = ["fused.png", "depth.png"] if request.options.depth else ["fused.png"]
                    for name in selected:
                        files[name] = png_info(scratch / name)
                    if request.options.depth and any(files["fused.png"][key] != files["depth.png"][key]
                                                     for key in ("width", "height")):
                        raise ValueError("焦层图与融合图尺寸不同。")
                    if request.options.intermediate:
                        steps = output_dir / "steps"
                        steps.mkdir()
                        for item in sorted(scratch.iterdir()):
                            check_cancel(request.cancel)
                            if item.is_file() and item.name not in names and item.name not in selected:
                                shutil.copyfile(item, steps / item.name)
                    for name in selected:
                        check_cancel(request.cancel)
                        partial = output_dir / f"{name}.part"
                        shutil.copyfile(scratch / name, partial)
                        partial.replace(output_dir / name)
                    output = output_dir / "fused.png"
                    depth = output_dir / "depth.png" if request.options.depth else None
                    status, diagnostic = "success", ""
        except Cancelled as error:
            status, diagnostic = "cancelled", str(error)
        except (OSError, ValueError) as error:
            status, diagnostic = "failed", str(error)
        if status != "success":
            files = {}
            output, depth = None, None
            for name in ("fused.png", "depth.png", "fused.png.part", "depth.png.part"):
                (output_dir / name).unlink(missing_ok=True)
            with log.open("ab") as stream:
                stream.write(f"\n[Pollen-EDF-Lab] {status} | {diagnostic}\n".encode("utf-8"))
        return FusionResult(status, diagnostic, output, depth, log, result.elapsed_s if result else None,
                            result.peak_rss_bytes if result else None, command, capabilities.version,
                            result.returncode if result else None, acceleration, evidence, files)


def diagnostics(result: FusionResult) -> str:
    return log_tail(result.log)
