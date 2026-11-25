import csv
from pathlib import Path
from threading import Event

import pytest

from pollen_edf.dataset import load_dataset
from pollen_edf.experiments import ExperimentRunner, trials_for
from pollen_edf.images import sha256
from pollen_edf.records import attach_reference, read_json, write_json
from pollen_edf.sampling import SampleSpec


@pytest.fixture(autouse=True)
def stub_machine_probe(monkeypatch):
    monkeypatch.setattr("pollen_edf.experiments.machine_information", lambda: {"source": "test fixture"})


def test_batch_failure_isolation_and_resume(synthetic_root, tmp_path, stub_engine):
    dataset = load_dataset(synthetic_root / "dataset.json")
    artifacts = tmp_path / "独立 输出"
    focal_plane = dataset.stacks[0].frames[0].path
    attach_reference(artifacts, dataset.record()["id"], "XY_00_00", focal_plane, "TEST", "Synthetic fixture")
    runner = ExperimentRunner(stub_engine, artifacts)
    root = runner.run(dataset, ["XY_faults", "XY_00_00"], trials_for(["FS_Default"],
                      [SampleSpec("stride", step=2, keep_endpoints=True)], {"depth": True}), 10, Event())
    manifest = read_json(root / "manifest.json")
    failed, success = manifest["tasks"]
    assert manifest["status"] == "failed"
    assert failed["status"] == "failed" and not failed["attempts"][-1]["output"]
    assert success["status"] == "success" and len(stub_engine.requests) == 1
    attempt = success["attempts"][0]
    assert [item["z"] for item in attempt["inputs"]] == [1, 3, 5, 7, 9, 11, 12]
    assert all(item["sha256"] == sha256(Path(item["path"])) for item in attempt["inputs"])
    assert sha256(root / attempt["output"]) == attempt["files"]["fused.png"]["sha256"]
    assert (root / attempt["depth"]).is_file()
    assert manifest["helicon_references"]["XY_00_00"]["sha256"] == sha256(focal_plane)
    with (root / "summary.csv").open(encoding="utf-8-sig", newline="") as stream:
        assert [row["status"] for row in csv.DictReader(stream)] == ["failed", "success"]
    assert "SYNTHETIC" in (root / "report.html").read_text(encoding="utf-8")
    assert (root / "config.yaml").is_file()

    fault = dataset.stacks[-1].frames[-1].path
    fault.write_bytes(focal_plane.read_bytes())
    runner.resume(root, Event())
    after = read_json(root / "manifest.json")
    assert after["status"] == "completed"
    assert [len(task["attempts"]) for task in after["tasks"]] == [2, 1]
    assert after["tasks"][0]["attempts"][0]["status"] == "failed"
    assert len(stub_engine.requests) == 2 and after["tasks"][1]["attempts"][0] == attempt


def test_skip_completed_or_retry_bad_output_and_refuse_changed_source(synthetic_root, tmp_path, stub_engine):
    dataset = load_dataset(synthetic_root)
    runner = ExperimentRunner(stub_engine, tmp_path / "results")
    root = runner.run(dataset, ["XY_00_00"], trials_for(["FS_Default"]), 10, Event())
    output = read_json(root / "manifest.json")["tasks"][0]["attempts"][0]["output"]
    runner.resume(root, Event())
    assert len(stub_engine.requests) == 1
    (root / output).write_bytes(b"damaged by test")
    runner.resume(root, Event())
    task = read_json(root / "manifest.json")["tasks"][0]
    assert len(task["attempts"]) == 2 and task["status"] == "success"
    assert task["attempts"][1]["output"] != output
    source = dataset.stacks[0].frames[0].path
    source.write_bytes(dataset.stacks[0].frames[2].path.read_bytes())
    with pytest.raises(ValueError, match="哈希已变化"):
        runner.resume(root, Event())


def test_cancelled_batch_resumes_and_never_exposes_partial_success(synthetic_root, tmp_path, stub_engine):
    dataset = load_dataset(synthetic_root)
    runner = ExperimentRunner(stub_engine, tmp_path / "results")
    cancellation = Event()

    def stop(update):
        if update.message.endswith("· 开始"):
            cancellation.set()

    root = runner.run(dataset, ["XY_00_00", "XY_00_01"], trials_for(["FS_Default"]), 10, cancellation, stop)
    manifest = read_json(root / "manifest.json")
    assert manifest["status"] == "cancelled"
    assert [task["status"] for task in manifest["tasks"]] == ["cancelled", "pending"]
    assert manifest["tasks"][0]["attempts"][0]["output"] is None
    # Simulate a machine interruption after reserving a later attempt.
    task = manifest["tasks"][1]
    task["status"] = "running"
    task["attempts"].append({"status": "running", "dir": task["dir"] + "/attempt_001"})
    write_json(root / "manifest.json", manifest)
    runner.resume(root, Event())
    recovered = read_json(root / "manifest.json")
    assert recovered["status"] == "completed"
    assert [attempt["status"] for attempt in recovered["tasks"][1]["attempts"]] == ["interrupted", "success"]


def test_output_below_input_folder_rejected(synthetic_root, stub_engine):
    dataset = load_dataset(synthetic_root)
    output = synthetic_root / "实验输出"
    runner = ExperimentRunner(stub_engine, output)
    with pytest.raises(ValueError, match="导入数据目录之外"):
        runner.run(dataset, ["XY_00_00"], trials_for(["FS_Default"]), 10, Event())
    assert not output.exists()
