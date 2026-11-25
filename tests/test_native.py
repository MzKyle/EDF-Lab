import os
import shutil
from threading import Event

import pytest

from pollen_edf.config import ENGINE_PIN, load_settings
from pollen_edf.dataset import load_dataset
from pollen_edf.engine import FocusStackEngine
from pollen_edf.experiments import ExperimentRunner, trials_for
from pollen_edf.images import absolute_path, png_info, sha256
from pollen_edf.records import artifact_path, read_json
from pollen_edf.synthetic import generate


@pytest.mark.native
def test_official_native_fusion_with_chinese_spaces_and_long_paths(tmp_path):
    exe = load_settings().engine
    if os.name != "nt" or not exe.is_file():
        pytest.skip("install the pinned Windows x64 native focus-stack release to run this test")
    engine = FocusStackEngine(exe)
    caps = engine.validate_environment()
    assert caps.binary_sha256 == ENGINE_PIN["binary_sha256"]
    assert caps.version == ENGINE_PIN["binary_version"]
    assert "--no-align" in caps.help_text
    test_data = absolute_path(tmp_path / "中文和 空格" / ("焦层实验" * 15) / ("原始图像" * 24) / ("路径" * 24))
    output = absolute_path(tmp_path / "独立 输出" / ("融合结果" * 24) / ("研究记录" * 25))
    try:
        dataset_dir = generate(test_data, width=192, height=128, layers=8, views=1, shift=1)
        dataset = load_dataset(dataset_dir / "dataset.json")
        stack = dataset.stacks[0]
        assert len(str(stack.frames[0].path)) > 260
        loaded = {frame.path: sha256(frame.path) for frame in stack.frames}
        runner = ExperimentRunner(engine, output)
        root = runner.run(dataset, [stack.id], trials_for(["FS_CPU"], overrides={"depth": True, "threads": 3}), 40, Event())
        manifest = read_json(root / "manifest.json")
        attempt = manifest["tasks"][0]["attempts"][0]
        assert manifest["status"] == "completed", attempt
        assert len(attempt["inputs"]) == 8 and attempt["exit_code"] == 0
        assert attempt["acceleration"] == "cpu_forced" and "--no-opencl" in attempt["command"]
        assert attempt["elapsed_s"] > 0 and attempt["peak_rss_bytes"] > 0
        for label in ("output", "depth"):
            path = artifact_path(root, attempt[label])
            assert png_info(path) == attempt["files"][path.name]
        assert b"Saved to fused.png" in artifact_path(root, attempt["log"]).read_bytes()
        assert all(sha256(path) == digest for path, digest in loaded.items())
        runner.resume(root, Event())
        assert len(read_json(root / "manifest.json")["tasks"][0]["attempts"]) == 1
    finally:
        # Pytest's regular temporary cleanup can hit the legacy Win32 path limit.
        for name in ("中文和 空格", "独立 输出"):
            path = absolute_path(str(tmp_path / name).removeprefix("\\\\?\\"))
            extended = f"\\\\?\\{path}"
            if path.is_dir():
                shutil.rmtree(extended)
