from pathlib import Path

import pytest

from pollen_edf.engine import EngineCapabilities, FusionResult
from pollen_edf.images import png_info, read_image, sha256
from pollen_edf.synthetic import generate


TEST_OPTIONS = frozenset({"--output", "--verbose", "--no-align", "--global-align", "--no-whitebalance",
                          "--no-contrast", "--no-transform", "--full-resolution-align", "--nocrop",
                          "--align-keep-size", "--depthmap", "--save-steps", "--no-opencl", "--consistency",
                          "--threads", "--batchsize", "--reference", "--denoise"})


@pytest.fixture
def synthetic_root(tmp_path):
    return generate(tmp_path / "SYNTHETIC fixtures", width=160, height=112, faults=True)


@pytest.fixture
def stub_capabilities():
    return EngineCapabilities(Path("fixture-engine"), "test", "synthetic-test", "test-double", "0" * 64,
                              "test fixture", TEST_OPTIONS, "", "", False)


class StubEngine:
    def __init__(self, capabilities):
        self.capabilities = capabilities
        self.requests = []
        self.fail_next = False

    def validate_environment(self):
        return self.capabilities

    get_capabilities = validate_environment

    def fuse(self, request):
        import cv2

        self.requests.append(request)
        args = ("test-double", *request.options.arguments(len(request.inputs), self.capabilities))
        out = request.output_dir
        log = out / "execution.log"
        log.write_text("TEST DOUBLE only\n", encoding="utf-8")
        status = "cancelled" if request.cancel.is_set() else "failed" if self.fail_next else "success"
        self.fail_next = False
        if status != "success":
            return FusionResult(status, "test failure", None, None, log, 0.02, 5000, args,
                                self.capabilities.version, 1, "test", ())
        assert all(sha256(path) == digest for path, digest in zip(request.inputs, request.input_hashes))
        output, depth = out / "fused.png", out / "depth.png" if request.options.depth else None
        _, payload = cv2.imencode(".png", read_image(request.inputs[0]))
        files = {}
        for path in (output, depth):
            if path is not None:
                path.write_bytes(payload.tobytes())
                files[path.name] = png_info(path)
        return FusionResult(status, "", output, depth, log, 0.02, 5000, args,
                            self.capabilities.version, 0, "test", (), files)

    def cancel(self):
        if self.requests:
            self.requests[-1].cancel.set()


@pytest.fixture
def stub_engine(stub_capabilities):
    return StubEngine(stub_capabilities)
