from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Event
from typing import Callable, Iterable

from .images import FORMATS, absolute_path, canonical_path, check_cancel, decode, image_format


@dataclass(frozen=True)
class Frame:
    path: Path
    z: int
    z_um: float | None = None
    expected_sha256: str | None = None
    metadata: dict = field(default_factory=dict)

    def record(self) -> dict:
        return {**asdict(self), "path": str(self.path)}


@dataclass(frozen=True)
class Stack:
    id: str
    frames: tuple[Frame, ...]
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Dataset:
    name: str
    stacks: tuple[Stack, ...]
    source: str
    protected_roots: tuple[Path, ...]
    provenance: str = "unclassified"
    metadata: dict = field(default_factory=dict)

    def record(self) -> dict:
        value = {"name": self.name, "source": self.source, "provenance": self.provenance,
                 "metadata": self.metadata, "protected_roots": list(map(str, self.protected_roots)),
                 "stacks": [{"id": stack.id, "metadata": stack.metadata,
                             "frames": [f.record() for f in stack.frames]} for stack in self.stacks]}
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")
        return {**value, "id": hashlib.sha256(encoded).hexdigest()}

    @classmethod
    def from_record(cls, value: dict) -> Dataset:
        stacks = []
        for entry in value["stacks"]:
            frames = tuple(Frame(**{**frame, "path": absolute_path(frame["path"])}) for frame in entry["frames"])
            stacks.append(Stack(entry["id"], frames, entry["metadata"]))
        return cls(value["name"], tuple(stacks), value["source"],
                   tuple(map(absolute_path, value["protected_roots"])), value["provenance"], value["metadata"])


@dataclass(frozen=True)
class StackCheck:
    errors: list[str]
    warnings: list[str]
    size: tuple[int, int] | None
    formats: dict[str, int]
    missing_ranges: list[tuple[int, int]]
    duplicate_z: list[int]
    hashes: dict[str, str]


Z_TOKEN = re.compile(r"(?:^|[\s._-])z[_\s]?(-?\d+)(?=$|[\s._-])", re.I | re.ASCII)
INT_TEXT = re.compile(r"-?[0-9]+\Z")
SHA256_TEXT = re.compile(r"[0-9a-fA-F]{64}\Z")


def parse_z(name: str) -> int:
    stem = Path(name).stem
    if INT_TEXT.fullmatch(stem):
        return int(stem)
    tokens = Z_TOKEN.findall(stem)
    if len(tokens) != 1:
        raise ValueError(f"无法唯一确定焦层编号 ({name})。请使用 JSON/CSV 索引或手动顺序。")
    return int(tokens[0])


def sorted_stack(view: str, frames: Iterable[Frame], metadata: dict | None = None) -> Stack:
    view = view.strip()
    ordered = tuple(sorted(frames, key=lambda frame: (frame.z, str(frame.path))))
    if not view or not ordered:
        raise ValueError("每个视野都必须具有名称和至少一帧。")
    return Stack(view, ordered, metadata or {})


def _image_paths(directory: Path) -> list[Path]:
    candidates = (absolute_path(path) for path in directory.iterdir() if path.suffix.lower() in FORMATS)
    return [path for path in candidates if path.is_file()]


def parse_directory(path: Path, cancel: Event | None = None) -> tuple[Stack, ...]:
    root_files = _image_paths(path)
    children = []
    for directory in sorted(path.iterdir()):
        check_cancel(cancel)
        directory = absolute_path(directory)
        if directory.is_dir() and (images := _image_paths(directory)):
            children.append((directory.name, images))
    if root_files and children:
        raise ValueError("根目录和子目录均包含图像。请导入指定视野目录或提供索引。")
    groups = [(path.name or "XY_00_00", root_files)] if root_files else children
    if not groups:
        raise ValueError(f"目录中没有受支持的焦层图片 ({path})")
    return tuple(sorted_stack(view, (Frame(file, parse_z(file.name)) for file in images))
                 for view, images in groups)


def _synthetic_metadata(candidates: Iterable[Path]) -> tuple[str, dict]:
    for directory in dict.fromkeys(candidates):
        marker = directory / "SYNTHETIC.json"
        if marker.is_file():
            metadata = json.loads(marker.read_text(encoding="utf-8-sig"))
            if not isinstance(metadata, dict) or metadata.get("provenance") != "SYNTHETIC":
                raise ValueError(f"无效的模拟数据标记 ({marker})")
            return "SYNTHETIC", metadata
    return "unclassified", {}


def _json_object(value: object, field_name: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} 必须为对象。")
    return value


def _parse_index(path: Path, cancel: Event | None) -> Dataset:
    metadata = {}
    view_metadata = {}
    name = path.stem
    with path.open(encoding="utf-8-sig", newline="") as stream:
        if path.suffix.lower() == ".csv":
            rows = list(csv.DictReader(stream))
        else:
            payload = _json_object(json.load(stream), "JSON 索引")
            rows = payload.get("frames")
            name = payload.get("name", name)
            metadata = _json_object(payload.get("metadata", {}), "metadata")
            view_metadata = _json_object(payload.get("views", {}), "views")
    if not isinstance(rows, list) or not rows or not isinstance(name, str) or not name.strip():
        raise ValueError("索引必须提供数据集名称和非空 frames 记录。")
    groups: dict[str, list[Frame]] = defaultdict(list)
    for row in rows:
        check_cancel(cancel)
        row = _json_object(row, "焦层记录")
        view, z, file = row.get("xy", "XY_00_00"), row.get("z"), row.get("path")
        if not isinstance(view, str) or not view.strip() or not isinstance(file, str) or not file.strip():
            raise ValueError("索引中的 xy 和 path 必须为非空字符串。")
        if type(z) is not int and not (isinstance(z, str) and INT_TEXT.fullmatch(z.strip())):
            raise ValueError(f"索引焦层 z 必须是整数 ({file})")
        raw_z_um = row.get("z_um")
        try:
            z_um = None if raw_z_um in (None, "") else float(raw_z_um)
        except (TypeError, ValueError) as error:
            raise ValueError(f"z_um 必须为有限数 ({file})") from error
        if z_um is not None and (type(raw_z_um) is bool or not math.isfinite(z_um)):
            raise ValueError(f"z_um 必须为有限数 ({file})")
        digest = row.get("sha256") or None
        if digest is not None and (not isinstance(digest, str) or not SHA256_TEXT.fullmatch(digest)):
            raise ValueError(f"SHA-256 必须为 64 位十六进制字符串 ({file})")
        extra = {key: value for key, value in row.items()
                 if key not in {"xy", "z", "path", "z_um", "sha256", "metadata"}}
        if None in extra:
            raise ValueError("CSV 记录含有多余的单元格。")
        extra.update(_json_object(row.get("metadata") or {}, "焦层 metadata"))
        groups[view.strip()].append(Frame(absolute_path(path.parent / file), int(z), z_um,
                                          digest.lower() if digest else None, extra))
    stacks = tuple(sorted_stack(view, frames, _json_object(view_metadata.get(view, {}), "视野元数据"))
                   for view, frames in sorted(groups.items()))
    roots = tuple(dict.fromkeys([path.parent, *(frame.path.parent for stack in stacks for frame in stack.frames)]))
    provenance, marker = _synthetic_metadata([*roots, *(root.parent for root in roots)])
    if metadata.get("provenance") == "SYNTHETIC":
        provenance = "SYNTHETIC"
    return Dataset(name.strip(), stacks, str(path), roots, provenance, {**metadata, **marker})


DirectoryParser = Callable[[Path, Event | None], tuple[Stack, ...]]


def load_dataset(source: str | Path, *, parser: DirectoryParser = parse_directory,
                 cancel: Event | None = None) -> Dataset:
    path = absolute_path(source)
    if path.is_dir():
        stacks = parser(path, cancel)
        identifiers = [stack.id for stack in stacks]
        if not identifiers or len(identifiers) != len(set(identifiers)):
            raise ValueError("目录解析器必须返回非空且名称唯一的视野。")
        provenance, metadata = _synthetic_metadata((path, path.parent))
        return Dataset(path.name or "Dataset", stacks, str(path), (path,), provenance, metadata)
    if path.is_file() and path.suffix.lower() in {".csv", ".json"}:
        return _parse_index(path, cancel)
    raise ValueError(f"请选择图像目录、JSON 索引或 CSV 索引 ({path})")


def manual_dataset(files: Iterable[str | Path], *, explicit_order: bool = False,
                   cancel: Event | None = None) -> Dataset:
    frames = []
    for index, file in enumerate(files):
        check_cancel(cancel)
        path = absolute_path(file)
        z = index + 1 if explicit_order else parse_z(path.name)
        frames.append(Frame(path, z))
    stack = sorted_stack("XY_00_00", frames)
    roots = tuple(dict.fromkeys(frame.path.parent for frame in frames))
    provenance, metadata = _synthetic_metadata([*roots, *(root.parent for root in roots)])
    return Dataset("手动图片列表", (stack,), "manual", roots, provenance,
                   {**metadata, "order": "explicit-list" if explicit_order else "filename-z"})


def validate_stack(stack: Stack, *, hash_files: bool = False, cancel: Event | None = None,
                   progress: Callable[[int, int], None] | None = None) -> StackCheck:
    errors: list[str] = []
    warnings: list[str] = []
    numbers = sorted(set(frame.z for frame in stack.frames))
    duplicates = sorted(z for z, count in Counter(f.z for f in stack.frames).items() if count > 1)
    if duplicates:
        errors.append(f"焦层编号重复 {duplicates}")
    first, last = stack.metadata.get("first_z"), stack.metadata.get("last_z")
    missing = [(a + 1, b - 1) for a, b in zip(numbers, numbers[1:]) if b > a + 1]
    if any(key in stack.metadata for key in ("first_z", "last_z")):
        if type(first) is not int or type(last) is not int or first > min(numbers) or last < max(numbers):
            errors.append("first_z 和 last_z 必须是覆盖有效编号的整数范围。")
        else:
            if first < min(numbers):
                missing.insert(0, (first, min(numbers) - 1))
            if last > max(numbers):
                missing.append((max(numbers) + 1, last))
    if missing:
        warnings.append(f"缺少焦层区间 {missing}")
    positions = [frame.z_um for frame in stack.frames if frame.z_um is not None]
    if positions and len(positions) != len(stack.frames):
        warnings.append("部分物理 Z 坐标缺失，焦层仍按编号排序。")
    if len(positions) > 1 and not (all(a < b for a, b in zip(positions, positions[1:])) or
                                  all(a > b for a, b in zip(positions, positions[1:]))):
        warnings.append("物理 Z 坐标随编号变化不严格单调。")
    formats: Counter = Counter()
    hashes = {}
    size = None
    seen = set()
    for index, frame in enumerate(stack.frames, 1):
        check_cancel(cancel)
        try:
            canonical = canonical_path(frame.path)
            if canonical in seen:
                errors.append(f"同一文件被重复引用 ({frame.path})")
            seen.add(canonical)
            raw = frame.path.read_bytes()
            expected = FORMATS.get(frame.path.suffix.lower())
            detected = image_format(raw)
            if expected is None or expected != detected:
                raise ValueError(f"扩展名与支持的文件格式不匹配 ({frame.path}, {detected})")
            if detected == "JPEG" and raw.rfind(b"\xff\xd9", 2) == -1:
                raise ValueError(f"JPEG 缺少结束标记 ({frame.path})")
            image = decode(raw, frame.path, engine_input=True)
            incoming_size = (int(image.shape[1]), int(image.shape[0]))
            del image
            formats[detected] += 1
            if size is None:
                size = incoming_size
            elif incoming_size != size:
                errors.append(f"分辨率不一致 ({frame.path}, {incoming_size}, 应为 {size})")
            if hash_files or frame.expected_sha256:
                digest = hashlib.sha256(raw).hexdigest()
                hashes[str(frame.path)] = digest
                if frame.expected_sha256 and frame.expected_sha256 != digest:
                    errors.append(f"SHA-256 与索引不符 ({frame.path})")
        except (OSError, ValueError, RuntimeError) as error:
            errors.append(str(error))
        if progress:
            progress(index, len(stack.frames))
    return StackCheck(errors, warnings, size, dict(formats), missing, duplicates, hashes)
