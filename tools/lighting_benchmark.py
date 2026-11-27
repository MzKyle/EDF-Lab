"""Run matched real-capture comparisons for the new lighting archive."""
from __future__ import annotations

import argparse
import re
import signal
import sys
from dataclasses import replace
from pathlib import Path
from threading import Event

from pollen_edf.config import load_settings
from pollen_edf.dataset import load_dataset
from pollen_edf.engine import FocusStackEngine
from pollen_edf.experiments import ExperimentRunner, Trial, trials_for
from pollen_edf.images import sha256
from pollen_edf.pollen_algorithm import PollenEDFOptions
from pollen_edf.pollen_engine import PollenEDFEngine
from pollen_edf.records import read_json, timestamp, write_json
from v02_benchmark import BASE, FRAME, NATIVE_CANDIDATE, helicon_run, show_progress


DEV_VIEWS = {"000246": {1, 4, 8}, "000907": {4, 9, 15}}
HOLDOUT_CAPTURE = "004157"
BRIGHTFIELD = "PollenEDF-Brightfield"
PYRAMID = "PollenEDF-Brightfield-Pyramid"
CONTEXT = "PollenEDF-Brightfield-Context"


def prepare(archive: Path, root: Path) -> None:
    groups = {split: ({}, []) for split in ("dev", "holdout", "coverage")}
    for date in sorted(archive.iterdir()):
        if not date.is_dir():
            continue
        for capture in sorted(date.glob("Images_*")):
            match = re.fullmatch(r"Images_(\d{6}).*", capture.name)
            if not capture.is_dir() or match is None:
                continue
            take = match[1]
            for directory in sorted(capture.iterdir()):
                match = re.fullmatch(r"(\d+)_\((-?\d+)_(-?\d+)\)", directory.name)
                if not directory.is_dir() or match is None:
                    continue
                number, x, y = map(int, match.groups())
                view = f"{date.name.replace('-', '')}_{take}_v{number:03d}"
                files = []
                for path in directory.glob("*.jpg"):
                    match = FRAME.fullmatch(path.name)
                    if match is None:
                        raise ValueError(f"Unexpected image filename in {view}: {path.name}")
                    files.append((*map(int, match.groups()), path))
                files.sort()
                if [row[0] for row in files] != list(range(1, len(files) + 1)) or len(files) < 2:
                    raise ValueError(f"Invalid SN sequence for {view}")
                if any(old[1] >= new[1] for old, new in zip(files, files[1:])):
                    raise ValueError(f"Non-increasing Z sequence for {view}")
                sizes = {(w, h) for _, _, w, h, _ in files}
                if len(sizes) != 1:
                    raise ValueError(f"Mixed sizes for {view}")
                split = "holdout" if take == HOLDOUT_CAPTURE else "dev" if number in DEV_VIEWS.get(take, ()) else "coverage"
                views, rows = groups[split]
                views[view] = {"date": date.name, "capture": capture.name, "source_view": directory.name,
                               "scene": "new illumination", "source_size": list(sizes.pop()), "xy_tokens": [x, y],
                               "first_z": files[0][0], "last_z": files[-1][0],
                               "first_archive_z": files[0][1], "last_archive_z": files[-1][1]}
                rows.extend({"xy": view, "z": sn, "path": str(path), "sha256": sha256(path),
                             "metadata": {"archive_z": z}} for sn, z, _, _, path in files)
    for split, (views, rows) in groups.items():
        path = root / "inputs" / f"{split}.json"
        value = {"name": f"New lighting · {split}", "views": views, "frames": rows,
                 "metadata": {"provenance": "real_microscopy_capture", "archive": str(archive),
                              "split": split, "focus_order": "SN ascending; Z units uncalibrated"}}
        if path.is_file() and read_json(path) != value:
            raise ValueError(f"The archive differs from the indexed {split} input.")
        if not rows:
            raise ValueError(f"The {split} split contains no images.")
        write_json(path, value)
        print(f"{split}: {len(views)} views / {len(rows)} frames", flush=True)


def run_stage(root: Path, split: str, stage: str, heli: Path | None, cancel: Event) -> None:
    dataset = load_dataset(root / "inputs" / f"{split}.json")
    if stage == "helicon":
        if heli is None:
            raise ValueError("Pass the verified Helicon 8.1.0 executable using --helicon.")
        for stack in dataset.stacks:
            helicon_run(heli, stack, root / f"{split}-helicon" / stack.id, cancel)
        return
    settings = load_settings()
    if stage == "native-locked":
        engine = FocusStackEngine(settings.engine)
        trials = [Trial("FS_Tuned", replace(BASE, **NATIVE_CANDIDATE))]
        if split == "dev":
            trials.insert(0, Trial("FS_Default", BASE))
    else:
        engine = PollenEDFEngine()
        if stage == "pollen-revised":
            trials = trials_for([BRIGHTFIELD], overrides={"intermediate": True})
            trials.append(Trial(PYRAMID, replace(trials[0].options, pyramid_levels=2)))
        elif stage == "pollen-fallback" and split == "dev":
            trials = trials_for([CONTEXT], overrides={"intermediate": True})
        elif stage == "pollen-fallback":
            lock = read_json(root / "locked.json")
            if engine.get_capabilities().identity_sha256 != lock["identity_sha256"]:
                raise ValueError("The current engine differs from the frozen development result.")
            if sha256(root / "inputs" / f"{split}.json") != lock["index_hashes"][split]:
                raise ValueError(f"The frozen {split} input index differs.")
            trials = [Trial(lock["method"], PollenEDFOptions(**lock["options"]))]
        else:
            trials = trials_for(["PollenEDF-Adaptive"], overrides={"intermediate": True})
    runner = ExperimentRunner(engine, root / f"{split}-{stage}")
    pointer = root / f"{split}-{stage}.json"
    if pointer.is_file():
        previous = read_json(pointer)
        if previous["status"] == "completed":
            print(f"{split} {stage}: completed, see {previous['path']}", flush=True)
            return
        run = runner.resume(Path(previous["path"]), cancel, show_progress)
    else:
        run = runner.run(dataset, [stack.id for stack in dataset.stacks], trials, settings.timeout_s,
                         cancel, show_progress)
    state = read_json(run / "manifest.json")["status"]
    write_json(pointer, {"path": str(run), "status": state, "recorded_at": timestamp()})
    if state != "completed":
        raise RuntimeError(f"{split} {stage}: {state}; see {run / 'manifest.json'}")


def freeze(root: Path, name: str) -> None:
    path = root / "locked.json"
    if path.is_file():
        if read_json(path)["method"] != name:
            raise ValueError("This experiment already has a frozen candidate.")
        return
    stage = "pollen-fallback" if name == CONTEXT else "pollen-revised"
    manifest_path = Path(read_json(root / f"dev-{stage}.json")["path"]) / "manifest.json"
    data = read_json(manifest_path)
    tasks = [task for task in data["tasks"] if task["preset"] == name]
    if data["status"] != "completed" or not tasks or any(task["options"] != tasks[0]["options"] for task in tasks):
        raise ValueError("Finish the development comparison before freezing its candidate.")
    write_json(path, {"method": name, "options": tasks[0]["options"], "frozen_at": timestamp(),
                      "identity_sha256": data["engine"]["identity_sha256"], "development_manifest": str(manifest_path),
                      "index_hashes": {split: sha256(root / "inputs" / f"{split}.json")
                                       for split in ("dev", "holdout", "coverage")}})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "baseline", "freeze", "validate", "helicon", "native-locked",
                                          "pollen-locked", "pollen-revised", "pollen-fallback"))
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "holdout", "coverage"), default="dev")
    parser.add_argument("--helicon", type=Path)
    parser.add_argument("--method", choices=(BRIGHTFIELD, PYRAMID, CONTEXT))
    args = parser.parse_args()
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")
    root = args.output.resolve()
    if args.stage == "prepare":
        if args.archive is None or root.is_relative_to(args.archive.resolve()):
            parser.error("Pass --archive and choose an output directory outside the archive.")
        prepare(args.archive.resolve(), root)
        return
    if args.stage == "freeze":
        if args.method is None:
            parser.error("Pass the selected development candidate using --method.")
        freeze(root, args.method)
        return
    cancel = Event()
    signal.signal(signal.SIGINT, lambda *_: cancel.set())
    stages = {"baseline": ("helicon", "pollen-locked", "native-locked"),
              "validate": ("helicon", "pollen-fallback")}.get(args.stage, (args.stage,))
    for stage in stages:
        run_stage(root, args.split, stage, args.helicon, cancel)


if __name__ == "__main__":
    main()
