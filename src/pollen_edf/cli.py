from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
from dataclasses import asdict, replace
from importlib.metadata import version
from pathlib import Path
from threading import Event

import yaml

from .config import load_settings
from .dataset import Dataset, load_dataset, manual_dataset, validate_stack
from .engine import FocusStackEngine
from .experiments import PRESETS, ExperimentRunner, ensure_separate_output, machine_information, trials_for
from .images import Cancelled, absolute_path, check_cancel, display_path
from .records import attach_reference, read_json, write_report
from .sampling import SampleSpec
from .synthetic import generate


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=argparse.SUPPRESS, help="本地 YAML 设置")
    parser.add_argument("--engine", type=Path, default=argparse.SUPPRESS, help="固定版本 focus-stack 路径")


def _add_input(parser: argparse.ArgumentParser, required: bool = True) -> None:
    sources = parser.add_mutually_exclusive_group(required=required)
    sources.add_argument("--input", type=Path, help="焦栈目录、批量目录或 JSON/CSV 索引")
    sources.add_argument("--files", type=Path, nargs="+", help="手动列出图片")
    parser.add_argument("--explicit-order", action="store_true", help="以 --files 的参数顺序确定编号")
    parser.add_argument("--view", action="append", help="指定 XY 视野，可重复")


def _add_fusion(parser: argparse.ArgumentParser, batch: bool) -> None:
    _add_input(parser, required=not batch)
    if batch:
        parser.add_argument("--resume", type=Path, help="恢复已有实验目录")
    parser.add_argument("--preset", action="append", choices=PRESETS, help="选择预设，可重复")
    parser.add_argument("--output", type=Path, help="实验记录父目录")
    parser.add_argument("--timeout", type=float, help="单任务原生进程超时秒数")
    parser.add_argument("--alignment", choices=("default", "global", "none"))
    parser.add_argument("--crop", choices=("auto", "keep-size", "none"))
    parser.add_argument("--acceleration", choices=("auto", "cpu", "opencl"))
    for flag in ("whitebalance", "contrast", "transform", "full-resolution-align", "depth", "intermediate"):
        parser.add_argument(f"--{flag}", action=argparse.BooleanOptionalAction, default=None)
    for flag in ("consistency", "threads", "batch-size", "reference"):
        parser.add_argument(f"--{flag}", type=int)
    parser.add_argument("--denoise", type=float)
    parser.add_argument("--sample", choices=("all", "stride", "range", "list", "uniform"), default="all")
    parser.add_argument("--step", type=int, help="每隔多少个已采集焦层抽取")
    parser.add_argument("--offset", type=int, action="append", help="stride 的起始位置，可重复")
    parser.add_argument("--range", type=int, nargs=2, metavar=("FIRST_Z", "LAST_Z"))
    parser.add_argument("--layers", nargs="+", type=int, help="显式指定 Z 编号")
    parser.add_argument("--count", type=int, action="append", help="uniform 目标层数，可重复")
    parser.add_argument("--keep-endpoints", action="store_true", help="强制保留源焦栈的第一层和最后一层")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pollen_edf", description="Pollen-EDF-Lab · 离线景深合成")
    _add_common(parser)
    subparsers = parser.add_subparsers(dest="action", required=True)
    doctor = subparsers.add_parser("doctor", help="检测环境和真实 native engine")
    doctor.add_argument("--json", action="store_true", help="输出完整探测记录")
    gui = subparsers.add_parser("gui", help="启动 PySide6 工作台")
    inspect = subparsers.add_parser("inspect", help="导入并逐帧校验")
    _add_input(inspect)
    inspect.add_argument("--hash", action="store_true", help="计算文件 SHA-256")
    fuse = subparsers.add_parser("fuse", help="在一个视野运行融合与参数对照")
    _add_fusion(fuse, False)
    benchmark = subparsers.add_parser("benchmark", help="串行批量实验或恢复")
    _add_fusion(benchmark, True)
    generate_data = subparsers.add_parser("generate-test-data", help="生成标注为 SYNTHETIC 的数据")
    generate_data.add_argument("--output", type=Path, required=True)
    generate_data.add_argument("--stress", action="store_true", help="默认改为 4096 × 3000，50 层，一个视野")
    for flag in ("width", "height", "layers", "views", "seed"):
        generate_data.add_argument(f"--{flag}", type=int)
    for flag in ("shift", "scale", "brightness", "noise"):
        generate_data.add_argument(f"--{flag}", type=float)
    generate_data.add_argument("--faults", action="store_true")
    reference = subparsers.add_parser("reference", help="关联一张已有 Helicon 合成图")
    _add_input(reference)
    reference.add_argument("--image", type=Path, required=True)
    reference.add_argument("--version", default="")
    reference.add_argument("--parameters", default="")
    report = subparsers.add_parser("report", help="从实验记录生成离线 HTML 报告")
    report.add_argument("--experiment", required=True, type=Path)
    report.add_argument("--output", type=Path)
    for child in (doctor, gui, inspect, fuse, benchmark, generate_data, reference, report):
        _add_common(child)
    return parser


def _dataset(args: argparse.Namespace, cancel: Event) -> Dataset:
    if args.files:
        return manual_dataset(args.files, explicit_order=args.explicit_order, cancel=cancel)
    if args.input is None or args.explicit_order:
        raise ValueError("请指定 --input 或 --files。--explicit-order 只适用于 --files。")
    return load_dataset(args.input, cancel=cancel)


def _views(dataset: Dataset, requested: list[str] | None, single: bool = False) -> list[str]:
    available = [stack.id for stack in dataset.stacks]
    views = requested or available
    unknown = set(views) - set(available)
    if unknown:
        raise ValueError(f"找不到视野 {sorted(unknown)}。可用视野为 {available}。")
    if single and len(set(views)) != 1:
        raise ValueError(f"请选择一个 --view。可用视野为 {available}。")
    return views


def _samples(args: argparse.Namespace) -> list[SampleSpec]:
    values = {"stride": args.step is not None or args.offset is not None, "range": args.range is not None,
              "list": args.layers is not None, "uniform": args.count is not None}
    if any(enabled and args.sample != mode for mode, enabled in values.items()):
        raise ValueError("抽层参数与 --sample 策略不匹配。")
    endpoints = args.keep_endpoints
    if args.sample == "all":
        return [SampleSpec()]
    if args.sample == "stride" and args.step is not None:
        return [SampleSpec("stride", step=args.step, offset=offset, keep_endpoints=endpoints)
                for offset in dict.fromkeys(args.offset or [0])]
    if args.sample == "range" and args.range:
        return [SampleSpec("range", start=args.range[0], end=args.range[1], keep_endpoints=endpoints)]
    if args.sample == "list" and args.layers:
        return [SampleSpec("list", layers=tuple(args.layers), keep_endpoints=endpoints)]
    if args.sample == "uniform" and args.count:
        return [SampleSpec("uniform", count=count, keep_endpoints=endpoints) for count in dict.fromkeys(args.count)]
    raise ValueError("抽层策略缺少所需参数。请查看 --help。")


ENGINE_FIELDS = ("alignment", "whitebalance", "contrast", "transform", "full_resolution_align", "consistency",
                 "denoise", "threads", "batch_size", "reference", "crop", "depth", "intermediate", "acceleration")


def _run_command(args: argparse.Namespace, cancel: Event) -> int:
    settings = load_settings(getattr(args, "config", None))
    if hasattr(args, "engine"):
        settings = replace(settings, engine=absolute_path(args.engine))
    if args.action == "gui":
        from .ui import launch

        return launch(settings)
    if args.action == "generate-test-data":
        options = {key: getattr(args, key) for key in ("width", "height", "layers", "views", "seed", "shift",
                                                       "scale", "brightness", "noise") if getattr(args, key) is not None}
        defaults = {"width": 4096, "height": 3000, "layers": 50, "views": 1} if args.stress else {}
        print(display_path(generate(args.output, **{**defaults, **options}, faults=args.faults)))
        return 0
    if args.action == "report":
        root = absolute_path(args.experiment)
        manifest = read_json(root / "manifest.json")
        if args.output is not None:
            ensure_separate_output(args.output, Dataset.from_record(manifest["dataset"]))
            if args.output.suffix.lower() != ".html":
                raise ValueError("报告输出文件必须以 .html 结尾。")
        print(display_path(write_report(root, manifest, args.output)))
        return 0
    engine = FocusStackEngine(settings.engine)
    if args.action == "doctor":
        return _doctor(engine, args.json)
    if args.action == "inspect":
        dataset = _dataset(args, cancel)
        checks = {}
        for identifier in _views(dataset, args.view):
            check_cancel(cancel)
            stack = next(stack for stack in dataset.stacks if stack.id == identifier)
            checks[identifier] = {"images": len(stack.frames), "z": [f.z for f in stack.frames],
                                  "z_um": [f.z_um for f in stack.frames],
                                  **asdict(validate_stack(stack, hash_files=args.hash, cancel=cancel))}
        print(json.dumps({"name": dataset.name, "dataset_id": dataset.record()["id"], "source": dataset.source,
                          "provenance": dataset.provenance, "metadata": dataset.metadata,
                          "images": sum(value["images"] for value in checks.values()), "stacks": checks},
                         ensure_ascii=False, indent=2))
        return 1 if any(check["errors"] for check in checks.values()) else 0
    if args.action == "reference":
        dataset = _dataset(args, cancel)
        ensure_separate_output(settings.artifacts_dir, dataset)
        reference = attach_reference(settings.artifacts_dir, dataset.record()["id"], _views(dataset, args.view, True)[0],
                                     args.image, args.version, args.parameters)
        print(json.dumps(reference, ensure_ascii=False, indent=2))
        return 0
    overrides = {key: getattr(args, key) for key in ENGINE_FIELDS if getattr(args, key) is not None}
    engine_runner = ExperimentRunner(engine, args.output or settings.artifacts_dir)

    def progress(update) -> None:
        if any(token in update.message for token in (" · 开始", " · success", " · failed", " · timed_out", " · cancelled", "实验 ")):
            print(f"[{update.finished}/{update.total}] {update.message}", file=sys.stderr, flush=True)

    if args.action == "benchmark" and args.resume:
        if (args.input or args.files or args.explicit_order or args.view or args.preset or args.output or
                args.timeout is not None or overrides or _samples(args) != [SampleSpec()]):
            raise ValueError("恢复时请沿用记录内的数据集和参数组合，使用 --resume 指定目录即可。")
        root = engine_runner.resume(args.resume, cancel, progress)
    else:
        dataset = _dataset(args, cancel)
        presets = args.preset or (["FS_Default"] if args.action == "fuse" else list(PRESETS))
        views = _views(dataset, args.view, single=args.action == "fuse")
        trials = trials_for(dict.fromkeys(presets), _samples(args), overrides)
        timeout = args.timeout if args.timeout is not None else settings.timeout_s
        root = engine_runner.run(dataset, views, trials, timeout, cancel, progress)
    manifest = read_json(root / "manifest.json")
    print(display_path(root))
    return {"completed": 0, "cancelled": 130}.get(manifest["status"], 1)


def _doctor(engine: FocusStackEngine, json_output: bool) -> int:
    packages = {pkg: version(pkg) for pkg in ("pollen-edf-lab", "PySide6-Essentials", "opencv-python-headless",
                                             "numpy", "matplotlib", "psutil", "PyYAML")}
    machine = machine_information()
    try:
        capabilities = engine.validate_environment()
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(json.dumps({"environment": machine, "packages": packages, "error": str(error)}, ensure_ascii=False, indent=2)
              if json_output else f"引擎检测失败。{error}")
        return 1
    presets = {}
    for trial in trials_for(PRESETS):
        try:
            trial.options.arguments(2, capabilities)
            presets[trial.preset] = "available"
        except ValueError as error:
            presets[trial.preset] = str(error)
    if json_output:
        print(json.dumps({"environment": machine, "packages": packages, "engine": capabilities.record(),
                          "presets": presets}, ensure_ascii=False, indent=2))
    else:
        print(f"Python {sys.version.split()[0]} · {machine['os']} · {machine['cpu']}")
        print(f"focus-stack release {capabilities.release} · {capabilities.version} · {capabilities.executable}")
        print(f"commit {capabilities.commit}\nSHA-256 {capabilities.binary_sha256}")
        print(f"OpenCV 构建声明 OpenCL 支持 {'是' if capabilities.opencl_built else '否'}")
        print("\n".join(f"{key}  {status}" for key, status in presets.items()))
    return 0


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if not stream.isatty() and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = make_parser().parse_args(argv)
    cancel = Event()
    previous = signal.signal(signal.SIGINT, lambda *_: cancel.set()) if args.action != "gui" else None
    try:
        return _run_command(args, cancel)
    except (Cancelled, KeyboardInterrupt):
        print("操作已取消。", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError, yaml.YAMLError) as error:
        print(str(error), file=sys.stderr)
        return 1
    finally:
        if previous is not None:
            signal.signal(signal.SIGINT, previous)
