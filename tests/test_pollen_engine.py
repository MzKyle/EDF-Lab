from dataclasses import replace
from threading import Event

import cv2
import numpy as np

from pollen_edf.dataset import load_dataset
from pollen_edf.experiments import ExperimentRunner, trials_for
from pollen_edf.images import read_image, sha256
from pollen_edf.pollen_algorithm import FocusSelection, PollenEDFOptions, render_fusion, select_focus
from pollen_edf.pollen_brightfield import focal_weights, smooth_in_focus_order
from pollen_edf.pollen_engine import PollenEDFEngine
from pollen_edf.pollen_weights import anchor_weak_stack, identify_halos, pool_broad_responses
from pollen_edf.records import read_json


def test_hard_selection_reuses_observed_bgr_pixels():
    # Synthetic complementary texture and differing colors give an exact pixel oracle.
    strip = np.tile(np.array([20, 120, 20, 120], dtype=np.uint8), (16, 6))
    focused = np.stack((strip, strip, strip), axis=-1)
    flat = np.full_like(focused, (18, 36, 55))
    frames = [np.hstack((focused, flat)), np.hstack((flat, focused))]
    options = trials_for(["PollenEDF-SML"], overrides={"score_scale": 1})[0].options
    mark = lambda _stage: None
    selection = select_focus(frames.__getitem__, (48, 16), 2, options, mark)
    fused = render_fusion(frames.__getitem__, (48, 16), 2, selection, options, mark)
    assert np.all(selection.layer[:, 4:18] == 0)
    assert np.all(selection.layer[:, 30:44] == 1)
    for index, image in enumerate(frames):
        chosen = selection.layer == index
        assert np.array_equal(fused[chosen], image[chosen])


def test_dark_stack_averages_and_pyramid_keeps_confident_pixels():
    size = (48, 24)
    frames = [np.full((24, 48, 3), value, np.uint8) for value in (0, 2, 4)]
    options = PollenEDFOptions(score_scale=1)
    mark = lambda _stage: None
    selection = select_focus(frames.__getitem__, size, 3, options, mark)
    assert not selection.confidence.any()
    assert not selection.foreground.any()
    assert np.all(render_fusion(frames.__getitem__, size, 3, selection, options, mark) == 2)

    label = np.zeros((24, 48), np.uint16)
    ones = np.ones((24, 48), np.float32)
    confidence = ones.copy()
    confidence[:, :24] = 0.0
    selection = FocusSelection(label, label + 2, confidence, ones, (1.0 - confidence) * 0.3, {})
    result = render_fusion(frames.__getitem__, size, 3, selection, replace(options, pyramid_levels=2), mark)
    assert np.all(result[:, 24:] == 0)
    assert result[12, 8, 0] == 1


def test_python_worker_runs_and_revalidates_all_diagnostics(synthetic_root, tmp_path, monkeypatch):
    monkeypatch.setattr("pollen_edf.experiments.machine_information", lambda: {"source": "test fixture"})
    data = load_dataset(synthetic_root / "dataset.json")
    view = data.stacks[0]
    runner = ExperimentRunner(PollenEDFEngine(), tmp_path / "独立结果")
    original = [sha256(frame.path) for frame in view.frames]
    settings = {"intermediate": True, "threads": 2, "halo_guard": True, "broad_pool": True,
                "weak_stack_anchor": True}
    root = runner.run(data, [view.id], trials_for(["PollenEDF-Adaptive"], overrides=settings), 20.0, Event())
    run = read_json(root / "manifest.json")
    attempt = run["tasks"][0]["attempts"][0]
    assert run["status"] == "completed", attempt
    assert run["engine"]["backend"] == "pollen-edf" and attempt["acceleration"] == "cpu"
    assert attempt["peak_rss_bytes"] > 0 and attempt["elapsed_s"] > 0
    assert set(attempt["files"]) == set(PollenEDFOptions(**settings).output_names())
    labels = read_image(root / attempt["dir"] / "depth-index.png")
    assert labels.dtype == np.uint16 and 1 <= labels.min() <= labels.max() <= len(view.frames)
    assert all(sha256(frame.path) == value for frame, value in zip(view.frames, original))
    assert "置信度" in (root / "report.html").read_text(encoding="utf-8")
    runner.resume(root, Event())
    assert len(read_json(root / "manifest.json")["tasks"][0]["attempts"]) == 1
    (root / attempt["confidence"]).write_bytes(b"damaged by test")
    runner.resume(root, Event())
    assert len(read_json(root / "manifest.json")["tasks"][0]["attempts"]) == 2


def test_repeated_dark_impulse_is_distinct_from_a_small_resolved_object():
    image = np.zeros((42, 48, 3), np.uint8)
    image[8, 8] = 30
    cv2.circle(image, (31, 26), 3, (60, 90, 75), -1)
    frames = [image.copy() for _ in range(6)]
    options = PollenEDFOptions(score_scale=1)
    mark = lambda _stage: None
    selection = select_focus(frames.__getitem__, (48, 42), len(frames), options, mark)
    result = render_fusion(frames.__getitem__, (48, 42), len(frames), selection, options, mark)
    assert selection.defects[8, 8] and np.count_nonzero(selection.defects) == 1
    assert not result[8, 8].any()
    assert np.array_equal(result[26, 31], image[26, 31])


def test_observed_layer_weights_for_troughs_and_broad_responses():
    # Synthetic Z profiles. The central high peak models an isolated resolved structure.
    luma = np.array([[35, 3, 35], [21, 4, 21], [10, 18, 10], [6, 44, 6],
                     [7, 12, 7], [13, 3, 13], [29, 2, 29], [47, 2, 47]], np.uint8)[:, None, :]
    confidence = np.array([[0.18, 0.22, 0.7]], np.float32)
    foreground = np.ones_like(confidence)
    result = identify_halos(luma, foreground, confidence, 0.60)
    assert result.mass[0, 0] > 0.9 and not result.mass[0, 1:].any()
    assert not result.weight(0).any() and result.weight(3)[0, 0] > 0.0
    assert np.allclose(sum(result.weight(index) for index in range(len(luma))), result.mass)

    scores = np.ones_like(luma, dtype=np.float32)
    scores[:, 0, 1] = [0, 0, 0, 1, 0.1, 0, 0, 0]
    pooled = pool_broad_responses(scores, scores.max(axis=0), foreground, confidence, 0.60)
    assert pooled.mass[0, 0] > 0.85 and not pooled.mass[0, 1:].any()
    assert np.allclose(sum(pooled.weight(index) for index in range(len(scores))), pooled.mass)


def test_weak_stack_anchor_uses_an_observed_plane_and_respects_supported_fields():
    scores = np.full((4, 20, 20), 0.94, dtype=np.float32)
    scores[0] = 1.0
    foreground = np.ones_like(scores[0])
    confidence = foreground * 0.1
    weight, index = anchor_weak_stack(scores, foreground, foreground, confidence, 0.6)
    assert index == 2
    assert np.array_equal(weight.weight(index), foreground)
    assert not weight.weight(1).any()
    confidence[0, :3] = 0.8
    assert anchor_weak_stack(scores, foreground, foreground, confidence, 0.6) == (None, None)


def test_brightfield_weights_normalize_and_preserve_a_flat_stack():
    # Synthetic local score profile with one supported focus and one flat location.
    scores = np.array([[0.0, 0.0], [1.2, 0.0], [2.0, 0.0], [1.2, 0.0], [0.0, 0.0]], np.float16)[:, None, :]
    smooth_in_focus_order(scores)
    weights = focal_weights(scores, scores.argmax(axis=0), scores.max(axis=0).astype(np.float32))
    mass = np.stack([weights.weight(index) for index in range(len(scores))])
    assert np.allclose(mass.sum(axis=0), 1.0)
    assert mass[2, 0, 0] > mass[0, 0, 0]
    assert np.allclose(mass[:, 0, 1], 0.2)

    size = (48, 24)
    images = [np.full((24, 48, 3), level, np.uint8) for level in (190, 200, 210)]
    options = trials_for(["PollenEDF-Brightfield"], overrides={"pyramid_levels": 2})[0].options
    checkpoint = lambda _stage: None
    selection = select_focus(images.__getitem__, size, 3, options, checkpoint)
    result = render_fusion(images.__getitem__, size, 3, selection, options, checkpoint)
    assert selection.focal_weights is not None
    assert np.all(result == 200)
