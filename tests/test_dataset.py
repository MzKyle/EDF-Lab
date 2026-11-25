import json

import pytest

from pollen_edf.dataset import load_dataset, manual_dataset, parse_z, validate_stack
from pollen_edf.images import sha256
from pollen_edf.synthetic import generate


def test_numeric_order_and_grouping(synthetic_root):
    dataset = load_dataset(synthetic_root)
    assert dataset.provenance == "SYNTHETIC"
    assert [stack.id for stack in dataset.stacks] == ["XY_00_00", "XY_00_01", "XY_faults"]
    assert [frame.z for frame in dataset.stacks[0].frames] == list(range(1, 13))
    assert [parse_z(name) for name in ("Z-2.jpg", "视野_z_004.png", "9.bmp")] == [-2, 4, 9]
    with pytest.raises(ValueError):
        parse_z("MP100_img_005.jpg")
    with pytest.raises(ValueError):
        parse_z("z1_Z2.jpg")


def test_missing_corruption_duplicate_and_dimensions(synthetic_root, tmp_path):
    dataset = load_dataset(synthetic_root / "dataset.json")
    fault = validate_stack(dataset.stacks[-1], hash_files=True)
    assert fault.missing_ranges == [(2, 2)]
    assert fault.errors and fault.warnings
    orig = dataset.stacks[0].frames[0].path
    orig2 = dataset.stacks[0].frames[1].path
    clone = tmp_path / "中文 Z_1.jpg"
    clone.write_bytes(orig.read_bytes())
    different = generate(tmp_path / "SYNTHETIC small", width=64, height=64, layers=2, views=1)
    other = different / "XY_00_00" / "Z_001.jpg"
    indexed = tmp_path / "acquisition.json"
    indexed.write_text(json.dumps({"views": {"位置 A": {"first_z": 0, "last_z": 6}}, "frames": [
        {"xy": "位置 A", "z": 1, "path": str(orig), "z_um": 4},
        {"xy": "位置 A", "z": 1, "path": str(clone), "z_um": 4},
        {"xy": "位置 A", "z": 3, "path": str(other), "z_um": 7},
        {"xy": "位置 A", "z": 4, "path": str(orig2), "z_um": 8, "sha256": "a" * 64},
    ]}, ensure_ascii=False), encoding="utf-8")
    validation = validate_stack(load_dataset(indexed).stacks[0])
    assert validation.duplicate_z == [1]
    assert validation.missing_ranges == [(0, 0), (2, 2), (5, 6)]
    assert any("分辨率不一致" in error for error in validation.errors)
    assert any("SHA-256" in error for error in validation.errors)


def test_manual_order_and_csv_metadata(synthetic_root, tmp_path):
    paths = list((synthetic_root / "XY_00_00").glob("*.jpg"))
    ambiguous = tmp_path / "任意 名称.jpg"
    ambiguous.write_bytes(paths[0].read_bytes())
    with pytest.raises(ValueError):
        manual_dataset([ambiguous, paths[1]])
    explicit = manual_dataset([paths[10], ambiguous], explicit_order=True)
    assert [frame.path.name for frame in explicit.stacks[0].frames] == [paths[10].name, ambiguous.name]
    csv_file = tmp_path / "索引 测试.csv"
    csv_file.write_text(f"xy,z,path,z_um,sha256,exposure_ms\n视野,10,{ambiguous.name},0.25,{sha256(ambiguous)},12\n"
                        f"视野,2,{paths[1]},-0.5,,13\n", encoding="utf-8-sig")
    stack = load_dataset(csv_file).stacks[0]
    assert [frame.z for frame in stack.frames] == [2, 10]
    assert stack.frames[1].metadata["exposure_ms"] == "12"
    assert stack.frames[0].z_um == -0.5
    assert not validate_stack(stack).errors


def test_path_extension_is_verified(synthetic_root, tmp_path):
    source = synthetic_root / "XY_00_00" / "Z_001.jpg"
    wrong = tmp_path / "Z_2.png"
    wrong.write_bytes(source.read_bytes())
    truncated = tmp_path / "Z_3.jpg"
    truncated.write_bytes(source.read_bytes()[:-8])
    stack = manual_dataset([wrong, truncated]).stacks[0]
    assert len(validate_stack(stack).errors) == 2


def test_synthetic_is_deterministic(tmp_path):
    options = dict(width=64, height=64, layers=3, seed=81, shift=2.0, scale=0.04, brightness=0.1)
    first = generate(tmp_path / "A", **options)
    second = generate(tmp_path / "B", **options)
    assert (first / "SYNTHETIC.json").read_bytes() == (second / "SYNTHETIC.json").read_bytes()
    assert all(sha256(path) == sha256(second / path.relative_to(first)) for path in first.rglob("*.jpg"))
    with pytest.raises(ValueError):
        generate(first, **options)
