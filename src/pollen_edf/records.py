from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

from .evaluation import METRIC_LABELS
from .images import absolute_path, canonical_path, display_path, read_image, sha256


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    path = absolute_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding=encoding, dir=path.parent,
                                     newline="", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(content)
        except BaseException:
            stream.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, data: object) -> None:
    write_text(path, json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2) + "\n")


def read_json(path: Path) -> dict:
    value = json.loads(absolute_path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"记录必须是 JSON 对象 ({path})")
    return value


@contextmanager
def experiment_lock(root: Path):
    with (root / ".lock").open("a+b") as stream:
        try:
            if os.name == "nt":
                import msvcrt

                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise ValueError(f"实验正在被另一个进程使用 ({root})") from error
        yield


def _reference_path(artifacts: Path, dataset_id: str, view: str) -> Path:
    if len(dataset_id) != 64 or any(ch not in "0123456789abcdef" for ch in dataset_id):
        raise ValueError("数据集标识无效。")
    key = hashlib.sha256(view.encode("utf-8")).hexdigest()
    return absolute_path(artifacts / ".references" / dataset_id / f"{key}.json")


def attach_reference(artifacts: Path, dataset_id: str, view: str, image_path: Path,
                     version: str, parameters: str) -> dict:
    image_path = absolute_path(image_path)
    image = read_image(image_path)
    reference = {"kind": "Helicon reference", "xy": view, "path": str(image_path),
                 "version": version, "parameters": parameters, "stored_at": timestamp(),
                 "width": image.shape[1], "height": image.shape[0], "sha256": sha256(image_path)}
    write_json(_reference_path(artifacts, dataset_id, view), reference)
    return reference


def get_reference(artifacts: Path, dataset_id: str, view: str) -> dict | None:
    path = _reference_path(artifacts, dataset_id, view)
    return read_json(path) if path.is_file() else None


def artifact_path(root: Path, relative: str) -> Path:
    path = absolute_path(root / relative)
    if not canonical_path(path).is_relative_to(canonical_path(root)):
        raise ValueError(f"记录中的路径超出实验目录 ({relative})")
    return path


def relative_artifact(path: Path, root: Path) -> str:
    return display_path(path).relative_to(display_path(root)).as_posix()


def save_review(root: Path, review: dict) -> Path:
    path = absolute_path(root / "reviews" / f"review_{uuid4().hex[:12]}.json")
    write_json(path, {**review, "created_at": timestamp()})
    return path


def write_summary(root: Path, manifest: dict) -> None:
    columns = ["task", "xy", "preset", "sampling", "layers", "sampling_variable", "status", "attempts", "elapsed_s",
               "peak_rss_bytes", "acceleration", "width", "height", *METRIC_LABELS, "diagnostic", "output", "log"]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    for task in manifest["tasks"]:
        last = task["attempts"][-1] if task["attempts"] else {}
        selection = task.get("selection", {})
        metrics = last.get("metrics") or {}
        writer.writerow({"task": task["id"], "xy": task["xy"], "preset": task["preset"],
                         "sampling": task["sampling"]["mode"], "layers": selection.get("layer_count"),
                         "sampling_variable": selection.get("variable"), "attempts": len(task["attempts"]),
                         "status": task["status"], **{key: metrics.get(key) for key in ("width", "height", *METRIC_LABELS)},
                         **{key: last.get(key) for key in ("elapsed_s", "peak_rss_bytes", "acceleration", "diagnostic", "output", "log")}})
    write_text(root / "summary.csv", buffer.getvalue(), encoding="utf-8-sig")


def write_report(root: Path, manifest: dict, destination: Path | None = None) -> Path:
    destination = absolute_path(destination or root / "report.html")
    local = canonical_path(destination.parent) == canonical_path(root)
    esc = html.escape

    def link(relative: str | None, title: str) -> str:
        if not relative:
            return "—"
        file = artifact_path(root, relative)
        uri = quote(relative) if local else canonical_path(file).as_uri()
        return f'<a href="{esc(uri, quote=True)}">{title}</a>'

    rows, details = [], []
    for task in manifest["tasks"]:
        last = task["attempts"][-1] if task["attempts"] else {}
        metrics = last.get("metrics") or {}
        selection = task.get("selection", {})
        elapsed, rss = last.get("elapsed_s"), last.get("peak_rss_bytes")
        pair = f"{metrics['width']} × {metrics['height']}" if metrics else "—"
        cells = (esc(task["xy"]), esc(task["preset"]), str(selection.get("layer_count", "—")),
                 esc(task["status"]), f"{elapsed:.3f}" if elapsed is not None else "—",
                 f"{rss / 1048576:.1f}" if rss is not None else "—", pair,
                 link(last.get("output"), "融合图"),
                 link(last.get("depth"), "焦层图")
                 + (f" · {link(last['confidence'], '置信度')}" if last.get("confidence") else ""),
                 link(last.get("log"), "日志"))
        rows.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
        info = {"selection": selection, "parameters": task["options"], "command": last.get("command"),
                "acceleration": last.get("acceleration"), "acceleration_evidence": last.get("acceleration_evidence"),
                "metrics": metrics, "diagnostic": last.get("diagnostic"),
                "attempts": [{"status": entry["status"], "dir": entry["dir"], "diagnostic": entry.get("diagnostic")}
                             for entry in task["attempts"]]}
        details.append(f"<details><summary>{esc(task['xy'])} / {esc(task['preset'])} · {esc(task['id'])}</summary>"
                       f"<pre>{esc(json.dumps(info, ensure_ascii=False, indent=2))}</pre></details>")
    reviews = []
    for path in sorted((root / "reviews").glob("review_*.json")):
        reviews.append(link(relative_artifact(path, root), esc(path.stem)))
    references = manifest.get("helicon_references", {})
    ref_info = ""
    if references:
        ref_info = "<h2>Helicon 参考</h2><pre>" + esc(json.dumps(references, ensure_ascii=False, indent=2)) + "</pre>"
    reviews_html = "<h2>人工 ROI 记录</h2><p>" + " · ".join(reviews) + "</p>" if reviews else ""
    rows_html, details_html = "".join(rows), "".join(details)
    engine = esc(manifest["engine"]["version"])
    if manifest["engine"].get("backend", "focus-stack") == "focus-stack":
        engine = f"focus-stack {engine}"
    provenance = esc(manifest["dataset"]["provenance"])
    content = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pollen-EDF-Lab · 实验报告</title>
<style>body{{font:15px system-ui,sans-serif;margin:36px auto;padding:0 22px;max-width:1120px;color:#213140}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}th,td{{text-align:left;padding:9px;border-bottom:1px solid #cbd5e1}}
th{{background:#edf2f6}}a{{color:#176481}}pre{{background:#f0f4f7;padding:14px;white-space:pre-wrap;overflow-wrap:anywhere}}details{{padding:8px 0}}</style>
</head><body><h1>景深合成实验</h1><p>{esc(manifest['id'])} · {esc(manifest['status'])} · {esc(manifest['created_at'])}</p>
<p>{esc(manifest['dataset']['name'])} · {provenance} · {engine}</p>
<p>{link('manifest.json', '完整记录')} · {link('config.yaml', '配置快照')} · {link('summary.csv', 'CSV 汇总')}</p>
<table><thead><tr><th>视野</th><th>预设</th><th>层数</th><th>状态</th><th>耗时 (s)</th><th>峰值 RSS (MiB)</th>
<th>输出尺寸</th><th>结果</th><th>深度</th><th>日志</th></tr></thead><tbody>{rows_html}</tbody></table>
<h2>参数与测量</h2>{details_html}{ref_info}{reviews_html}</body></html>"""
    write_text(destination, content)
    return destination
