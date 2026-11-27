"""Run the V0.2 real-capture experiment serially, with a date-disjoint split."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from threading import Event

from pollen_edf.config import load_settings
from pollen_edf.dataset import load_dataset
from pollen_edf.engine import EngineOptions, FocusStackEngine
import cv2
from PIL import Image

from pollen_edf.experiments import ExperimentRunner, Trial, trials_for
from pollen_edf.images import image_format, png_info, read_image, sha256
from pollen_edf.pollen_engine import PollenEDFEngine
from pollen_edf.process_runner import run_process
from pollen_edf.records import read_json, timestamp, write_json

DEV = (
    ("2026-10-29", "144803", 61, "dense, bright yellow-green"),
    ("2026-10-29", "102002", 1, "near-empty dark control"),
    ("2026-10-30", "140410", 4, "short stack, focus close to the first plane"),
    ("2026-11-05", "163242", 61, "sparse dim particles"),
    ("2026-11-06", "163002", 16, "blue-green rings and bright-pixel trails"),
)
HOLDOUT = (
    ("2026-11-03", "160942", 61, "dense, low contrast"),
    ("2026-11-04", "085918", 16, "numerous bright particles"),
    ("2026-11-11", "145648", 31, "bright rings, mixed texture"),
    ("2026-11-13", "104731", 106, "very sparse dim particles near the edge"),
)
FRAME = re.compile(r"SN_(\d+)\+[0-9_]+\+Z_(-?\d+)\+\((\d+)_(\d+)\)\+[^+]+\+[\d.]+\.jpg", re.I)
BASE = EngineOptions(threads=8, batch_size=8)
NATIVE_TUNING = {
    "FS_Default": {},
    "FS_NoAlign": {"alignment": "none"},
    "FS_Global": {"alignment": "global"},
    "FS_FullRes": {"full_resolution_align": True},
    "FS_WhiteBalanceOff": {"whitebalance": False},
    "FS_ContrastOff": {"contrast": False},
    "FS_Consistency0": {"consistency": 0},
    "FS_Consistency1": {"consistency": 1},
    "FS_Consistency2": {"consistency": 2},
    "FS_Denoise0": {"denoise": 0.0},
    "FS_Denoise4": {"denoise": 4.0},
    "FS_Denoise16": {"denoise": 16.0},
    "FS_Denoise64": {"denoise": 64.0},
    "FS_Reference6": {"reference": 6},
}
NATIVE_CANDIDATE = {"alignment": "none", "denoise": 64.0}
POLLEN_NAMES = ("PollenEDF-SML", "PollenEDF-Tenengrad", "PollenEDF-DepthMap", "PollenEDF-Adaptive")
POLLEN_ABLATIONS = {
    "PollenEDF-SingleScale": {"scorer": "sml"},
    "PollenEDF-NoiseOff": {"noise_aware": False},
    "PollenEDF-RefineOff": {"refine": False},
    "PollenEDF-BackgroundOff": {"background": False},
    "PollenEDF-DefectsOff": {"defects": False},
    "PollenEDF-SoftOff": {"soft": False},
    "PollenEDF-Pyramid2": {"pyramid_levels": 2},
}


def make_index(archive: Path, root: Path, split: str) -> Path:
    archive = archive.resolve()
    tables = {"dev": DEV, "holdout": HOLDOUT}
    views, rows = {}, []
    for date, time, index, scene in tables[split]:
        capture = archive / date / f"Images_{time}"
        directory, = capture.glob(f"{index}_(*)")
        view = f"{date.replace('-', '')}_{time}_v{index:03}"
        frames = []
        for file in directory.glob("*.jpg"):
            match = FRAME.fullmatch(file.name)
            if match is None:
                raise ValueError(f"Unexpected filename {file}")
            sn, z, w, h = map(int, match.groups())
            frames.append((sn, z, w, h, file))
        frames.sort()
        n = len(frames)
        if n < 2 or [entry[0] for entry in frames] != list(range(1, n + 1)):
            raise ValueError(f"Invalid capture sequence {directory}")
        if any(first[1] >= second[1] for first, second in zip(frames, frames[1:])):
            raise ValueError(f"The archive Z values do not follow SN order in {directory}")
        views[view] = {"date": date, "capture": capture.name, "source_view": directory.name,
                       "scene": scene, "first_z": 1, "last_z": n,
                       "source_size": list(frames[0][2:4])}
        rows.extend({"xy": view, "z": sn, "path": str(file), "metadata": {"archive_z": z}}
                    for sn, z, _, _, file in frames)
    target = root / "inputs" / f"{split}.json"
    index = {"name": f"V0.2 pollen captures · {split}",
             "metadata": {"provenance": "real_microscopy_capture", "split": split, "archive": str(archive),
                          "coordinates": "logical SN; archive Z token retained; physical unit unknown"},
             "views": views, "frames": rows}
    if target.exists():
        if read_json(target) != index:
            raise ValueError(f"The frozen split differs from {target}")
    else:
        write_json(target, index)
    return target


def show_progress(update) -> None:
    if any(word in update.message for word in (" · 开始", " · success", " · failed", " · timed_out", "实验 ")):
        print(f"[{update.finished}/{update.total}] {update.message}", flush=True)


def native_candidate(root: Path) -> dict:
    path = root / "native-candidate.json"
    if path.is_file():
        return read_json(path)
    ofat = Path(read_json(root / "dev-native-ofat.json")["path"]) / "manifest.json"
    if read_json(ofat)["status"] != "completed":
        raise ValueError("Complete the native development sweep before the tuned comparison.")
    choice = {"selected_at": timestamp(), "focus_stack": NATIVE_CANDIDATE, "development_manifest": str(ofat),
              "basis": "Disable frame transforms after tracing recurring camera points. "
                       "Level 64 reduced dark residual in the development sweep and the no-alignment crop."}
    write_json(path, choice)
    return choice


def freeze(root: Path) -> Path:
    target = root / "locked.json"
    if target.is_file():
        return target
    paths = {stage: Path(read_json(root / f"dev-{stage}.json")["path"]) / "manifest.json"
             for stage in ("native-locked", "pollen-fallback")}
    native = read_json(paths["native-locked"])
    pollen = read_json(paths["pollen-fallback"])
    if any(manifest["status"] != "completed" for manifest in (native, pollen)):
        raise ValueError("Complete both selected development runs before freezing the holdout parameters.")
    options = pollen["tasks"][0]["options"]
    focus_stack = native_candidate(root)["focus_stack"]
    if any(task["options"] != options for task in pollen["tasks"]):
        raise ValueError("Development adaptive options differ between views.")
    if any(task["options"][key] != value for task in native["tasks"] for key, value in focus_stack.items()):
        raise ValueError("The native development run differs from the selected tuning.")
    write_json(target, {
        "frozen_at": timestamp(),
        "pollen_identity_sha256": pollen["engine"]["identity_sha256"],
        "pollen_edf": {key: options[key] for key in ("halo_guard", "weak_stack_anchor")},
        "frozen_adaptive_options": options,
        "focus_stack": focus_stack,
        "dev_candidate_manifest": str(paths["pollen-fallback"]),
        "holdout_index_sha256": sha256(root / "inputs" / "holdout.json"),
        "development_basis": "Preserve the measured inter-grain gaps and retain a representative source "
                             "plane when focused foreground evidence is too scarce.",
    })
    return target


def native_trials(stage: str, locked: dict | None, split: str) -> list[Trial]:
    if stage == "native-ofat":
        return [Trial(name, replace(BASE, **values)) for name, values in NATIVE_TUNING.items()]
    tuned = Trial("FS_Tuned", replace(BASE, **locked["focus_stack"]))
    return [tuned] if split == "dev" else [Trial("FS_Default", BASE), tuned]


def pollen_trials(stage: str, locked: dict | None) -> list[Trial]:
    trials = trials_for(POLLEN_NAMES, overrides={"intermediate": True, "halo_guard": False,
                                                "broad_pool": False, "weak_stack_anchor": False})
    if stage == "pollen-revised":
        guard = replace(trials[-1].options, halo_guard=True)
        return [Trial("PollenEDF-Adaptive-Guard", guard),
                Trial("PollenEDF-Adaptive-GuardPool", replace(guard, broad_pool=True))]
    if stage == "pollen-fallback":
        return [Trial("PollenEDF-Adaptive-GuardFallback",
                      replace(trials[-1].options, halo_guard=True, weak_stack_anchor=True))]
    if stage == "pollen-ofat":
        adaptive = trials[-1].options
        trials.extend(Trial(name, replace(adaptive, **values)) for name, values in POLLEN_ABLATIONS.items())
    elif stage == "pollen-locked":
        adaptive = trials[-1].options
        trials[-1] = Trial(POLLEN_NAMES[-1], replace(adaptive, **locked["frozen_adaptive_options"]))
    return trials


def write_png(path: Path, image) -> dict:
    ok, data = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not ok:
        raise ValueError(f"Cannot encode {path}")
    path.write_bytes(data.tobytes())
    return png_info(path)


def helicon_run(exe: Path, stack, output: Path, cancel: Event) -> None:
    existing = output / "run.json"
    if existing.exists():
        previous = read_json(existing)
        if (previous["status"] == "success" and "png" in previous and previous["exe_sha256"] == sha256(exe)
                and previous["png"]["sha256"] == sha256(output / "fused.png")
                and previous["tiff_sha256"] == sha256(output / "ordered.tiff")
                and all(sha256(frame.path) == row["sha256"] for frame, row in zip(stack.frames, previous["inputs"]))
                and len(previous["inputs"]) == len(stack.frames)):
            print(f"{stack.id} · existing Helicon verified", flush=True)
            return
        if previous["status"] == "success" and "png" in previous:
            raise ValueError(f"Existing Helicon record differs: {existing}")
        history = output / f"retry-{len(list(output.glob('retry-*'))) + 1:03d}"
        history.mkdir()
        for name in ("run.json", "execution.log", "application.log", "ordered.tiff", "fused.png"):
            evidence = output / name
            if evidence.is_file():
                evidence.replace(history / name)
    output.mkdir(parents=True, exist_ok=True)
    run = {"view": stack.id, "version": "Helicon Focus 8.1.0 x64", "exe": str(exe), "started_at": timestamp(),
           "exe_sha256": sha256(exe), "settings": "Method B / radius 8 / smoothing 4 / TIFF compression 1"}
    with tempfile.TemporaryDirectory(prefix="pollen-v02-helicon-") as folder:
        work = Path(folder)
        staged, inputs = [], []
        for i, frame in enumerate(stack.frames):
            data = frame.path.read_bytes()
            if image_format(data) != "JPEG" or data.rfind(b"\xff\xd9") < 2:
                raise ValueError(f"Invalid JPEG {frame.path}")
            staged_file = work / f"frame_{i:04}.jpg"
            staged_file.write_bytes(data)
            staged.append(str(staged_file))
            inputs.append({"path": str(frame.path), "sn": frame.z, "sha256": hashlib.sha256(data).hexdigest()})
        listing = work / "inputs.txt"
        listing.write_text("\n".join(staged) + "\n", encoding="utf-8")
        tiff = work / "ordered.tiff"
        command = [str(exe), "-silent", "-mp:1", "-rp:8", "-sp:4", "-tif:u", "-i", str(listing), f"-save:{tiff}"]
        result = run_process(command, cwd=work, log=output / "execution.log", timeout_s=180.0, cancel=cancel)
        run.update(status=result.status, exit_code=result.returncode, elapsed_s=result.elapsed_s,
                   peak_rss_bytes=result.peak_rss_bytes, inputs=inputs, command=command)
        write_json(existing, run)
        if result.status != "success":
            raise RuntimeError(f"{stack.id} Helicon {result.status}: {result.diagnostic}")
        with tiff.open("rb") as header, Image.open(tiff) as image:
            if image_format(header.read(16)) != "TIFF" or image.tag_v2[259] != 1:
                raise ValueError(f"Helicon did not write a lossless uncompressed TIFF ({stack.id})")
        log = (output / "execution.log").read_text(encoding="utf-8", errors="replace")
        app_log = Path(os.environ["LOCALAPPDATA"]) / "HeliconFocus" / "HeliconFocus.log"
        if app_log.is_file():
            recorded = app_log.read_text(encoding="utf-8", errors="replace")
            if work.name in recorded:
                verified = [line for line in recorded.splitlines()
                            if re.search(r"Helicon Focus 8\.1\.0|OpenCL|Processing source|ordered\.tiff", line)
                            and not re.search(r"activation|serial key|email|password", line, re.I)]
                (output / "application.log").write_text("\n".join(verified) + "\n", encoding="utf-8")
                log = "\n".join(verified)
        if "8.1.0" not in log or "OpenCL device not selected" not in log:
            raise ValueError(f"Expected Helicon version/CPU evidence missing in {output}")
        processed = [int(index) for index in re.findall(r"Processing source .*?frame_(\d+)\.jpg", log)]
        if processed[:len(stack.frames)] != list(range(len(stack.frames))):
            raise ValueError(f"Helicon processing order differs for {stack.id}")
        run["tiff_sha256"] = sha256(tiff)
        run["tiff_compression"] = 1
        shutil.copyfile(tiff, output / "ordered.tiff")
        run["png"] = write_png(output / "fused.png", read_image(output / "ordered.tiff"))
        run["finished_at"] = timestamp()
        write_json(existing, run)
    print(f"{stack.id} · Helicon B · {run['elapsed_s']:.2f} s", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "freeze", "native-ofat", "native-locked", "helicon",
                                         "pollen-ofat", "pollen-revised", "pollen-fallback", "pollen-locked"))
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "holdout"), default="dev")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--helicon", type=Path, help="Helicon Focus 8.1.0 signed executable")
    args = parser.parse_args()
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")
    root = args.output.resolve()
    if root.is_relative_to(args.archive.resolve()):
        parser.error("The experiment output must be outside the read-only archive.")
    if args.stage == "prepare":
        for split in ("dev", "holdout"):
            path = make_index(args.archive, root, split)
            print(f"{split} {path}")
        return 0
    if args.stage == "freeze":
        print(freeze(root))
        return 0
    if (args.stage.endswith("-ofat") or args.stage in {"pollen-revised", "pollen-fallback"}) and args.split != "dev":
        parser.error("Candidate development and OFAT apply only to the development set.")
    held_out = read_json(root / "locked.json") if args.split == "holdout" else None
    if held_out and sha256(root / "inputs" / "holdout.json") != held_out["holdout_index_sha256"]:
        raise ValueError("The holdout index differs from the parameter lock.")
    cancel = Event()
    signal.signal(signal.SIGINT, lambda *_: cancel.set())
    root.mkdir(parents=True, exist_ok=True)
    if args.stage == "helicon":
        if args.helicon is None:
            parser.error("--helicon is required for the Helicon stage.")
        data = load_dataset(make_index(args.archive, root, args.split))
        for stack in data.stacks:
            helicon_run(args.helicon.resolve(), stack, root / f"{args.split}-helicon" / stack.id, cancel)
        return 0
    settings = load_settings()
    pollen = args.stage.startswith("pollen-")
    engine = PollenEDFEngine() if pollen else FocusStackEngine(settings.engine)
    caps = engine.validate_environment()
    write_json(root / ("pollen-capabilities.json" if pollen else "native-capabilities.json"), caps.record())
    if args.stage == "native-ofat":
        write_json(root / "native-ofat-plan.json", {"baseline_options": BASE.__dict__, "changes": NATIVE_TUNING,
                                                   "same_views_for_all_variants": True,
                                                   "opencl_available_in_pinned_binary": caps.opencl_built})
    runner = ExperimentRunner(engine, root / f"{args.split}-{args.stage}")
    if args.resume:
        run = runner.resume(args.resume, cancel, show_progress)
    else:
        if args.stage == "native-locked" and args.split == "dev":
            locked = native_candidate(root)
        elif args.split == "holdout":
            locked = held_out
        else:
            locked = read_json(root / "locked.json") if args.stage.endswith("-locked") else None
        if pollen and locked and caps.identity_sha256 != locked["pollen_identity_sha256"]:
            raise ValueError("The frozen PollenEDF source or runtime changed after development.")
        data = load_dataset(make_index(args.archive, root, args.split))
        trials = pollen_trials(args.stage, locked) if pollen else native_trials(args.stage, locked, args.split)
        run = runner.run(data, [stack.id for stack in data.stacks], trials,
                         settings.timeout_s, cancel, show_progress)
    status = read_json(run / "manifest.json")["status"]
    write_json(root / f"{args.split}-{args.stage}.json", {"path": str(run), "status": status})
    print(json.dumps({"experiment": str(run), "status": status}, ensure_ascii=False))
    return int(status != "completed")


if __name__ == "__main__":
    raise SystemExit(main())
