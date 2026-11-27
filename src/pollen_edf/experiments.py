from __future__ import annotations

import math
import os
import platform
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from typing import Callable, Iterable
from uuid import uuid4

import cv2
import psutil
import yaml

from . import __version__
from .config import RESOURCES
from .dataset import Dataset, StackCheck, validate_stack
from .engine import EngineOptions, FusionEngine, FusionRequest, FusionResult
from .evaluation import measure_file
from .images import Cancelled, absolute_path, canonical_path, check_cancel, sha256
from .pollen_algorithm import PollenEDFOptions
from .records import (artifact_path, experiment_lock, get_reference, read_json, relative_artifact, timestamp,
                      write_json, write_report, write_summary, write_text)
from .sampling import SampleSpec


PRESETS = yaml.safe_load((RESOURCES / "experiment_presets.yaml").read_text(encoding="utf-8"))
_HARD = {"background": False, "soft": False}
POLLEN_PRESETS = {
    "PollenEDF-SML": {**_HARD, "scorer": "sml", "noise_aware": False, "refine": False},
    "PollenEDF-Tenengrad": {**_HARD, "scorer": "tenengrad", "noise_aware": False, "refine": False},
    "PollenEDF-DepthMap": _HARD,
    "PollenEDF-Adaptive": {"halo_guard": True, "weak_stack_anchor": True},
    "PollenEDF-Brightfield": {"brightfield": True, "background": False, "defects": False},
    "PollenEDF-Brightfield-Context": {"brightfield": True, "background": False, "defects": False,
                                    "focus_context_px": 24, "pyramid_levels": 3},
}


def manifest_backend(manifest: dict) -> str:
    return manifest["engine"].get("backend", "focus-stack")


def _options_type(backend: str) -> type[EngineOptions] | type[PollenEDFOptions]:
    if backend == "pollen-edf":
        return PollenEDFOptions
    if backend == "focus-stack":
        return EngineOptions
    raise ValueError(f"未知融合引擎 {backend}")


@dataclass(frozen=True)
class Trial:
    preset: str
    options: EngineOptions | PollenEDFOptions
    sampling: SampleSpec = SampleSpec()

    @property
    def backend(self) -> str:
        return "pollen-edf" if isinstance(self.options, PollenEDFOptions) else "focus-stack"


def trials_for(names: Iterable[str], samples: Iterable[SampleSpec] = (SampleSpec(),),
               overrides: dict | None = None) -> list[Trial]:
    trials = []
    samples = tuple(samples)
    for name in names:
        if name in PRESETS:
            cls, values = EngineOptions, PRESETS[name]
        elif name in POLLEN_PRESETS:
            cls, values = PollenEDFOptions, POLLEN_PRESETS[name]
        else:
            raise ValueError(f"未知实验预设 {name}")
        extra = set(overrides or ()) - {field.name for field in fields(cls)}
        if extra:
            raise ValueError(f"{name} 不支持参数 {', '.join(sorted(extra))}")
        options = cls(**{**values, **(overrides or {})})
        trials.extend(Trial(name, options, sample) for sample in samples)
    return trials


def machine_information() -> dict:
    cpu = platform.processor()
    if os.name == "nt":
        import winreg

        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    gpus = []
    gpu_probe = "not_available"
    if os.name == "nt":
        import json

        try:
            system_root = Path(os.environ["SystemRoot"])
            shell = system_root / "System32/WindowsPowerShell/v1.0/powershell.exe"
            cmd = ("[Console]::OutputEncoding = [Text.Encoding]::UTF8\n"
                   "Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion | ConvertTo-Json -Compress")
            completed = subprocess.run([str(shell), "-NoProfile", "-NonInteractive", "-Command", cmd],
                                       capture_output=True, encoding="utf-8-sig", timeout=8,
                                       creationflags=subprocess.CREATE_NO_WINDOW, check=True)
            data = json.loads(completed.stdout) if completed.stdout.strip() else []
            gpus = data if isinstance(data, list) else [data]
            gpu_probe = "Win32_VideoController"
        except (OSError, ValueError, KeyError, subprocess.SubprocessError):
            pass
    return {"python": sys.version, "os": platform.platform(), "architecture": platform.machine(), "cpu": cpu,
            "physical_cores": psutil.cpu_count(logical=False), "logical_cores": psutil.cpu_count(),
            "memory_bytes": psutil.virtual_memory().total, "lab_opencv": cv2.__version__,
            "gpu_inventory_source": gpu_probe, "display_adapters": gpus}


def ensure_separate_output(path: Path, dataset: Dataset) -> None:
    resolved = canonical_path(path)
    if any(resolved.is_relative_to(canonical_path(root)) for root in dataset.protected_roots):
        raise ValueError("实验或参考记录的输出目录必须位于导入数据目录之外。")


def _slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name).strip("._-")[:36] or "xy"


@dataclass(frozen=True)
class ExperimentUpdate:
    message: str
    finished: int
    total: int


Notify = Callable[[ExperimentUpdate], None]


class ExperimentRunner:
    def __init__(self, engine: FusionEngine, artifacts: Path):
        self.engine = engine
        self.artifacts = absolute_path(artifacts)

    def run(self, dataset: Dataset, views: Iterable[str], trials: Iterable[Trial], timeout_s: float,
            cancel: Event, notify: Notify | None = None) -> Path:
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("超时必须是正的有限秒数。")
        ensure_separate_output(self.artifacts, dataset)
        capabilities = self.engine.validate_environment()
        by_view = {stack.id: stack for stack in dataset.stacks}
        view_list = list(dict.fromkeys(views))
        trial_list = list(trials)
        if not view_list or not trial_list or any(view not in by_view for view in view_list):
            raise ValueError("实验需要有效的视野和至少一个参数组合。")
        if any(trial.backend != capabilities.backend for trial in trial_list):
            raise ValueError("同一实验的预设必须使用同一种融合引擎。")
        experiment_id = ("experiment_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8])
        root = absolute_path(self.artifacts / experiment_id)
        root.mkdir(parents=True, exist_ok=False)
        record = dataset.record()
        tasks = []
        for view_no, view in enumerate(view_list, 1):
            for trial_no, trial in enumerate(trial_list, 1):
                task_id = f"{view_no:03d}-{trial_no:03d}"
                directory = f"view_{view_no:03d}_{_slug(view)}/{trial_no:03d}_{_slug(trial.preset)}"
                tasks.append({"id": task_id, "xy": view, "preset": trial.preset, "dir": directory,
                              "options": asdict(trial.options), "sampling": asdict(trial.sampling),
                              "status": "pending", "attempts": []})
        references = {view: reference for view in view_list
                      if (reference := get_reference(self.artifacts, record["id"], view)) is not None}
        manifest = {"schema": 1, "lab_version": __version__, "id": experiment_id, "created_at": timestamp(),
                    "status": "pending", "timeout_s": timeout_s, "dataset": record, "tasks": tasks,
                    "resource_sampling": {"rss": "peak sampled aggregate process-tree RSS", "interval_s": 0.08},
                    "metrics_recipe": "pollen-edf/0.1",
                    "engine": capabilities.record(), "environments": [{"recorded_at": timestamp(), **machine_information()}],
                    "helicon_references": references, "source_hashes": {}, "validation": {}}
        with experiment_lock(root):
            snapshot = {"dataset_id": record["id"], "timeout_s": timeout_s, "engine": capabilities.identity_sha256,
                        "backend": capabilities.backend,
                        "trials": [{"xy": task["xy"], "preset": task["preset"], "options": task["options"],
                                    "sampling": task["sampling"]} for task in tasks]}
            write_text(root / "config.yaml", yaml.safe_dump(snapshot, allow_unicode=True, sort_keys=False))
            self._execute(root, manifest, dataset, cancel, notify)
        return root

    def resume(self, root: Path, cancel: Event, notify: Notify | None = None) -> Path:
        root = absolute_path(root)
        with experiment_lock(root):
            manifest = read_json(root / "manifest.json")
            if manifest.get("schema") != 1 or manifest.get("id") != root.name:
                raise ValueError("实验目录或记录版本无效。")
            dataset = Dataset.from_record(manifest["dataset"])
            ensure_separate_output(root, dataset)
            capabilities = self.engine.validate_environment()
            stored = manifest["engine"]
            digest = stored.get("identity_sha256") or stored.get("binary_sha256")
            if capabilities.backend != manifest_backend(manifest) or capabilities.identity_sha256 != digest:
                raise ValueError("引擎哈希与实验记录不同。请使用原引擎恢复，或开始新实验。")
            for view, inputs in manifest["source_hashes"].items():
                for path, previous_hash in inputs.items():
                    check_cancel(cancel)
                    if sha256(Path(path), cancel) != previous_hash:
                        raise ValueError(f"视野 {view} 的输入哈希已变化，请作为新数据集实验 ({path})")
            for task in manifest["tasks"]:
                if task["status"] == "success" and self._outputs_match(root, task, cancel, capabilities.backend):
                    continue
                if task["attempts"]:
                    previous = task["attempts"][-1]
                    if previous["status"] == "running":
                        previous.update(status="interrupted", ended_at=timestamp(), diagnostic="上次运行中断。")
                    if previous["status"] == "success":
                        task["verification_issue"] = "上次结果缺失或输出哈希不匹配，已安排重试。"
                task["status"] = "pending"
            manifest["environments"].append({"recorded_at": timestamp(), **machine_information()})
            self._execute(root, manifest, dataset, cancel, notify)
        return root

    @staticmethod
    def _outputs_match(root: Path, task: dict, cancel: Event, backend: str = "focus-stack") -> bool:
        attempt = task["attempts"][-1]
        if backend == "pollen-edf":
            expected = set(PollenEDFOptions(**task["options"]).output_names())
        else:
            expected = {"fused.png", "depth.png"} if task["options"]["depth"] else {"fused.png"}
        if set(attempt.get("files", {})) != expected:
            return False
        if attempt.get("output") != f"{attempt['dir']}/fused.png":
            return False
        if "depth.png" in expected and attempt.get("depth") != f"{attempt['dir']}/depth.png":
            return False
        if "confidence.png" in expected and attempt.get("confidence") != f"{attempt['dir']}/confidence.png":
            return False
        try:
            return all(sha256(artifact_path(root, f"{attempt['dir']}/{name}"), cancel) == file["sha256"]
                       for name, file in attempt["files"].items())
        except OSError:
            return False

    def _execute(self, root: Path, manifest: dict, dataset: Dataset, cancel: Event, notify: Notify | None) -> None:
        tasks = manifest["tasks"]
        by_view = {stack.id: stack for stack in dataset.stacks}
        checks: dict[str, StackCheck] = {}
        options_type = _options_type(manifest_backend(manifest))
        total = len(tasks)
        finished = sum(task["status"] == "success" for task in tasks)
        manifest["status"] = "running"
        manifest["updated_at"] = timestamp()
        write_json(root / "manifest.json", manifest)

        def emit(message: str) -> None:
            if notify:
                notify(ExperimentUpdate(message, finished, total))

        for task in tasks:
            if task["status"] == "success":
                continue
            if cancel.is_set():
                break
            stack = by_view[task["xy"]]
            serial = len(task["attempts"]) + 1
            directory = f"{task['dir']}/attempt_{serial:03d}"
            out = artifact_path(root, directory)
            attempt = {"dir": directory, "started_at": timestamp(), "status": "running",
                       "environment": len(manifest["environments"]) - 1}
            task["attempts"].append(attempt)
            task["status"] = "running"
            write_json(root / "manifest.json", manifest)
            out.mkdir(parents=True, exist_ok=False)
            label = f"{stack.id} · {task['preset']}"
            emit(f"{label} · 开始")
            try:
                if stack.id not in checks:
                    checks[stack.id] = validate_stack(stack, hash_files=True, cancel=cancel,
                                                       progress=lambda i, n: emit(f"{label} · 校验 {i}/{n}"))
                check = checks[stack.id]
                manifest["validation"][stack.id] = {key: value for key, value in asdict(check).items() if key != "hashes"}
                if check.errors:
                    raise ValueError("\n".join(check.errors))
                previous = manifest["source_hashes"].setdefault(stack.id, check.hashes)
                if previous != check.hashes:
                    raise ValueError(f"视野 {stack.id} 的源文件与实验记录不匹配。")
                frames, task["selection"] = SampleSpec(**task["sampling"]).select(stack)
                attempt["inputs"] = [{**frame.record(), "sha256": check.hashes[str(frame.path)]} for frame in frames]
                write_json(root / "manifest.json", manifest)
                result = self.engine.fuse(FusionRequest(tuple(frame.path for frame in frames), out,
                                                         options_type(**task["options"]), manifest["timeout_s"], cancel,
                                                         lambda message: emit(f"{label} · {message}"),
                                                         tuple(item["sha256"] for item in attempt["inputs"])))
                self._record_result(root, out, attempt, result, cancel)
            except (Cancelled, OSError, ValueError, RuntimeError) as error:
                status = "cancelled" if isinstance(error, Cancelled) else "failed"
                log = out / "execution.log"
                with log.open("a", encoding="utf-8") as stream:
                    stream.write(f"\n[Pollen-EDF-Lab] {status} | {error}\n")
                attempt.update(status=status, diagnostic=str(error), log=relative_artifact(log, root),
                               output=None, depth=None, files={}, metrics=None)
            attempt["ended_at"] = timestamp()
            task["status"] = attempt["status"]
            finished += 1
            write_json(root / "manifest.json", manifest)
            emit(f"{label} · {task['status']}")
        if all(task["status"] == "success" for task in tasks):
            manifest["status"] = "completed"
        elif cancel.is_set() or any(task["status"] == "cancelled" for task in tasks):
            manifest["status"] = "cancelled"
        else:
            manifest["status"] = "failed"
        manifest["updated_at"] = timestamp()
        write_json(root / "manifest.json", manifest)
        write_summary(root, manifest)
        write_report(root, manifest)
        emit(f"实验 {manifest['status']}")

    @staticmethod
    def _record_result(root: Path, out: Path, attempt: dict, result: FusionResult, cancel: Event) -> None:
        attempt.update(status=result.status, diagnostic=result.diagnostic, command=list(result.command),
                       engine_version=result.engine_version, exit_code=result.exit_code, elapsed_s=result.elapsed_s,
                       peak_rss_bytes=result.peak_rss_bytes, acceleration=result.acceleration,
                       acceleration_evidence=list(result.acceleration_evidence), log=relative_artifact(result.log, root),
                       output=None, depth=None, files={})
        if result.status == "success" and result.output:
            check_cancel(cancel)
            metrics = measure_file(result.output)
            check_cancel(cancel)
            write_json(out / "metrics.json", metrics)
            attempt.update(output=relative_artifact(result.output, root), files=result.files, metrics=metrics,
                           depth=relative_artifact(result.depth, root) if result.depth else None)
            if "confidence.png" in result.files:
                attempt["confidence"] = relative_artifact(out / "confidence.png", root)
        elif result.status == "success":
            raise ValueError("引擎报告成功但没有给出融合图路径。")
