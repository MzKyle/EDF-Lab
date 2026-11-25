import sys
from dataclasses import replace
from pathlib import Path
from threading import Event, Thread
from time import monotonic, sleep

import psutil
import pytest

from pollen_edf import engine as engine_module
from pollen_edf.dataset import load_dataset
from pollen_edf.engine import EngineOptions, FocusStackEngine, FusionRequest
from pollen_edf.process_runner import run_process


def test_verified_option_mapping_and_limits(stub_capabilities):
    options = EngineOptions(alignment="global", contrast=False, whitebalance=False, consistency=1,
                            denoise=0, reference=2, threads=3, batch_size=2, depth=True, acceleration="cpu")
    assert options.arguments(6, stub_capabilities) == [
        "--output=fused.png", "--verbose", "--global-align", "--no-whitebalance", "--no-contrast",
        "--depthmap=depth.png", "--no-opencl", "--consistency=1", "--threads=3", "--batchsize=2",
        "--reference=2", "--denoise=0",
    ]
    photometric = EngineOptions(alignment="global", transform=False, denoise=-1)
    assert photometric.arguments(6, stub_capabilities)[2:] == ["--global-align", "--no-transform", "--denoise=-1"]
    for invalid in (EngineOptions(consistency=3), EngineOptions(denoise=float("nan")), EngineOptions(threads=0),
                    EngineOptions(reference=6), EngineOptions(alignment="none", full_resolution_align=True),
                    EngineOptions(batch_size=1.5), EngineOptions(acceleration="opencl")):
        with pytest.raises(ValueError):
            invalid.arguments(6, stub_capabilities)
    with pytest.raises(ValueError, match="--depthmap"):
        options.arguments(6, replace(stub_capabilities, options=stub_capabilities.options - {"--depthmap"}))


def test_process_nonzero_exit_and_both_log_streams(tmp_path):
    log = tmp_path / "exec.log"
    code = "import sys; print('out-' * 100000); print('why failure', file=sys.stderr); sys.exit(7)"
    run = run_process([sys.executable, "-c", code], cwd=tmp_path, log=log, timeout_s=6, cancel=Event())
    assert run.status == "failed" and run.returncode == 7
    assert b"why failure" in log.read_bytes() and log.stat().st_size > 200_000


@pytest.mark.parametrize("method", ["timed_out", "cancelled"])
def test_process_tree_is_reaped(tmp_path, method):
    pid_file = tmp_path / "child.pid"
    code = ("from pathlib import Path\nimport subprocess, sys, time\n"
            "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
            "Path('child.pid').write_text(str(p.pid))\ntime.sleep(60)\n")
    cancel = Event()

    def request_stop():
        limit = monotonic() + 5
        while not pid_file.exists() and monotonic() < limit:
            sleep(0.02)
        cancel.set()

    thread = Thread(target=request_stop) if method == "cancelled" else None
    if thread:
        thread.start()
    result = run_process([sys.executable, "-c", code], cwd=tmp_path, log=tmp_path / "output.log",
                         timeout_s=8 if thread else 1.5, cancel=cancel)
    if thread:
        thread.join()
    assert result.status == method
    assert result.peak_rss_bytes and result.elapsed_s > 0
    pid = int(pid_file.read_text())
    limit = monotonic() + 4
    while psutil.pid_exists(pid) and monotonic() < limit:
        try:
            if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                break
        except psutil.NoSuchProcess:
            break
        sleep(0.02)
    try:
        assert psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        pass


@pytest.mark.parametrize("mode", ["missing", "bad_png", "missing_depth", "nonzero"])
def test_wrapper_rejects_mock_process_output(tmp_path, synthetic_root, stub_capabilities, monkeypatch, mode):
    stack = load_dataset(synthetic_root).stacks[0]
    engine = FocusStackEngine(Path("fixture.exe"))
    monkeypatch.setattr(engine, "get_capabilities", lambda: stub_capabilities)
    output_code = {
        "missing": "pass",
        "bad_png": "from pathlib import Path; Path('fused.png').write_bytes(b'not an image')",
        "missing_depth": "import cv2, numpy as np; cv2.imwrite('fused.png', np.zeros((20, 20), np.uint8))",
        "nonzero": "import sys; print('error'); sys.exit(9)",
    }[mode]

    def run_mock(_command, **kwargs):
        return run_process([sys.executable, "-c", output_code], **kwargs)

    monkeypatch.setattr(engine_module, "run_process", run_mock)
    out = tmp_path / "out"
    result = engine.fuse(FusionRequest(tuple(frame.path for frame in stack.frames[:2]), out,
                                       EngineOptions(depth=mode == "missing_depth"), timeout_s=10))
    assert result.status == "failed" and result.diagnostic
    assert result.output is None and result.depth is None and not result.files
    assert result.log.is_file() and not (out / "fused.png").exists()
