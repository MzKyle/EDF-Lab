from pathlib import Path

import numpy as np
import pytest

from pollen_edf.dataset import Frame, Stack, load_dataset
from pollen_edf.evaluation import ROI, ROIMapping, focus_curve, measure
from pollen_edf.sampling import SampleSpec


def test_sampling_records_actual_spacings_and_bounds():
    stack = Stack("xy", tuple(Frame(Path(f"{z}.jpg"), z, z * 0.5) for z in range(1, 51)))
    keep = SampleSpec("stride", step=4, offset=2, keep_endpoints=True).select(stack)[1]
    assert keep["z"][:3] == [1, 3, 7] and keep["z"][-1] == 50
    assert keep["z_intervals"][:2] == [2, 4]
    assert keep["z_um_intervals"][:2] == [1.0, 2.0]
    assert keep["variable"] == "spacing_with_full_coverage"
    range_result = SampleSpec("range", start=12, end=20).select(stack)[1]
    assert range_result["variable"] == "coverage_at_original_spacing"
    assert range_result["z_range"] == [12, 20]
    different_both = SampleSpec("stride", step=4, offset=2).select(stack)[1]
    assert different_both["variable"] == "spacing_and_coverage"
    assert SampleSpec("list", layers=(9, 1, 4)).select(stack)[1]["z"] == [1, 4, 9]
    assert SampleSpec().select(stack)[1]["variable"] == "all_layers"
    for count in (2, 10, 13, 17, 25, 50):
        info = SampleSpec("uniform", count=count).select(stack)[1]
        assert len(set(info["z"])) == count and info["z_range"] == [1, 50]
    with pytest.raises(ValueError):
        SampleSpec("list", layers=(1, 9, 9)).select(stack)
    with pytest.raises(ValueError):
        SampleSpec("stride", step=2, offset=2).select(stack)


def test_metrics_and_roi_correspondence(synthetic_root):
    flat = np.full((40, 50, 3), 90, dtype=np.uint8)
    flat[:, :10] = 255
    roi = ROI(15, 7, 25, 20)
    numbers = measure(flat, roi)
    assert numbers["brightness_mean"] == 90 and numbers["brightness_std"] == 0
    for name in ("tenengrad", "sml", "laplacian_variance", "saturated_ratio", "background_noise_estimate"):
        assert numbers[name] == 0
    assert measure(flat)["saturated_ratio"] == pytest.approx(0.2)
    with pytest.raises(ValueError):
        measure(flat, ROI(48, 0, 5, 10))
    mapping = ROIMapping(1.5, 2, -10, -3, ROI(12, 2, 90, 90), ROI(0, 0, 180, 220))
    assert mapping.to_left(ROI(15, 20, 10, 5)) == ROI(12, 37, 16, 10)
    with pytest.raises(ValueError):
        mapping.to_left(ROI(11, 40, 10, 10))
    stack = load_dataset(synthetic_root / "dataset.json").stacks[0]
    curve = focus_curve(stack, ROI(20, 19, 47, 33))
    assert [item["z"] for item in curve] == list(range(1, 13))
    assert [item["z_um"] for item in curve[-2:]] == [7.5, 8.25]
    assert max(item["sml"] for item in curve) > min(item["sml"] for item in curve)
    assert curve[-1]["sml"] == pytest.approx(curve[-1]["sml_per_pixel"] * 47 * 33)
