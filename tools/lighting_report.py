"""Generate the complete matched-input report for the new lighting trial."""
from __future__ import annotations

import argparse
import html
from pathlib import Path
from statistics import median

import cv2
import numpy as np

from pollen_edf.dataset import load_dataset
from pollen_edf.images import read_image, sha256
from pollen_edf.records import read_json, write_json, write_text
from lighting_benchmark import CONTEXT
from v02_measure import gray, plt, tile
from v02_report import CSS, fig, fmt, link


SPLITS = {"dev": "开发", "holdout": "独立采集留出", "coverage": "同批其它视野"}
REGIONS = {"wall": "外壁和表面结构", "small": "小颗粒", "quiet": "低结构背景"}


def value(method: dict, area: str, key: str):
    return method["regions"].get(area, {}).get(key)


def centre(numbers) -> float | None:
    values = [number for number in numbers if number is not None]
    return median(values) if values else None


def summarize(split: str, data: dict) -> str:
    methods = {}
    for results in data.values():
        for name, result in results.items():
            methods.setdefault(name, []).append(result)
    rows = []
    for name, runs in methods.items():
        values = [name, str(len(runs)), fmt(centre(r["elapsed_s"] for r in runs)),
                  fmt(centre(r["rss_mib"] for r in runs), 0)]
        for region, key in (("wall", "edge_correlation"), ("small", "edge_correlation"),
                            ("wall", "edge_transfer"), ("quiet", "quiet_rms")):
            metric = [value(r, region, key) for r in runs]
            count = sum(q is not None for q in metric)
            values.append(f"{fmt(centre(metric), 3)} ({count})")
        rows.append("<tr>" + "".join(f"<td>{html.escape(val)}</td>" for val in values) + "</tr>")
    return (f"<h3>{SPLITS[split]} · {len(data)} 个视野</h3><div class='table'><table>"
            "<thead><tr><th>方法</th><th>视野数</th><th>秒</th><th>峰值 RSS MiB</th>"
            "<th>外壁带通相关</th><th>小颗粒带通相关</th><th>外壁带通相对幅度</th>"
            "<th>背景残差 RMS</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")


def render_views(root: Path, split: str, data: dict, specs: dict, notes: dict) -> str:
    sections = []
    for view, methods in data.items():
        spec = specs[view]
        meta = spec["meta"]
        full_outputs = " · ".join(link(item["path"], name) for name, item in methods.items())
        logs = []
        for name, item in methods.items():
            folder = Path(item["path"]).parent
            filename = "run.json" if name == "Helicon-B" else "execution.log"
            target = (folder / filename).as_posix()
            if (root / target).is_file():
                logs.append(link(target, name))
        annotation = notes.get(view)
        if annotation:
            annotation = f"<blockquote>{html.escape(annotation)}</blockquote>"
        else:
            annotation = ""
        panels = []
        for name, raw in spec["regions"].items():
            path = raw["folder"]
            items = [fig(f"{path}/source.png", f'参考原片 Z={raw["best"]["z"]:+d}')]
            extras = []
            for method, result in methods.items():
                if name not in result["regions"]:
                    continue
                entry = result["regions"][name]
                label = method if entry["usable"] else f"{method} · 低可信配对"
                items.append(fig(f'{result["folder"]}/{name}.png', label))
                for suffix, desc in (("depth", "焦层"), ("confidence", "置信度")):
                    file = f'{result["folder"]}/{name}-{suffix}.png'
                    if (root / file).is_file():
                        extras.append(fig(file, f"{method} · {desc}"))
            results = "".join(items)
            diagnostics = ("<details><summary>焦层图和置信度图</summary><div class='grid'>"
                           + "".join(extras) + "</div></details>") if extras else ""
            x, y = raw["xy"]
            white_pct = raw["white_fraction_every_layer"] * 100.0
            panels.append(f"<div class='roi'><h3>{REGIONS[name]} · ({x}, {y}) · 384 × 288</h3>"
                          f"<p>原片结构锚点 {len(raw['anchors'])} 个 · 各层均近白 {white_pct:.1f}%</p>"
                          f"<div class='grid'>{results}</div><p>"
                          f"{link(f'{path}/focus.png','原始焦点评分曲线')} · "
                          f"{link(f'{path}/quiet-mask.png','原片背景计量掩膜')}</p>{diagnostics}</div>")
        fields = "".join(fig(f'{item["folder"]}/field.png', name) for name, item in methods.items())
        sections.append(f"<section id='{view}'><h2>{view}</h2>"
                        f"<p>{SPLITS[split]} · {html.escape(meta['capture'])} · {int(meta['last_z'])} 层</p>"
                        f"<p>全分辨率成片　{full_outputs}</p><p>执行记录　{' · '.join(logs)}</p>{annotation}"
                        f"<details><summary>整视野缩览</summary><div class='grid'>{fields}</div></details>"
                        + "".join(panels) + "</section>")
    return "".join(sections)


def native_cases(root: Path, dev: dict, specs: dict) -> str:
    """Link the optics to the output using the raw Z traces at two fixed observables."""
    small_view = "20250423_000246_v004"
    rim_view = "20250423_000246_v008"
    region = "wall"
    stacks = {stack.id: stack for stack in load_dataset(root / "inputs" / "dev.json").stacks}
    x, y = specs[rim_view]["regions"][region]["xy"]
    samples = {}
    output = {}
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 3.5), layout="constrained")
    detail = slice(245, 369)
    for frame in stacks[rim_view].frames:
        z = frame.metadata["archive_z"]
        if z in (-32, -16, -8, 0, 8):
            crop = tile(read_image(frame.path, engine_input=True), (x, y))
            line = gray(crop)[117:124].mean(axis=0)
            samples[z] = line.tolist()
            axes[0].plot(np.arange(384)[detail], line[detail], label=f"Raw Z {z:+d}")
    for method in ("Helicon-B", "PollenEDF-Adaptive", "PollenEDF-Brightfield-Pyramid", CONTEXT):
        if method in dev[rim_view]:
            path = root / dev[rim_view][method]["folder"] / f"{region}.png"
            line = gray(read_image(path))[117:124].mean(axis=0)
            output[method] = line.tolist()
            axes[1].plot(np.arange(384)[detail], line[detail], label=method)
    for ax in axes:
        ax.set(xlabel="Horizontal position in ROI (px)", ylabel="Mean gray value")
        ax.legend(fontsize=7)
        ax.grid(alpha=.2)
    folder = root / "case-measurements"
    folder.mkdir(exist_ok=True)
    figure.savefig(folder / "rim-profiles.png", dpi=150)
    plt.close(figure)
    xy = specs[small_view]["regions"][region]["xy"]
    def spot(array):
        return float(gray(array)[37:50, 45:60].mean())
    signal = [{"z": frame.metadata["archive_z"], "mean_luma": spot(tile(read_image(frame.path, engine_input=True), xy))}
              for frame in stacks[small_view].frames]
    final = {method: spot(read_image(root / run["folder"] / f"{region}.png"))
             for method, run in dev[small_view].items()}
    write_json(folder / "case-profiles.json", {"rim": {"view": rim_view, "roi_camera_xy": [x, y],
                                                      "roi_rows": [117, 124], "raw": samples, "fused": output},
                                             "spot": {"view": small_view, "roi_camera_xy": xy,
                                                      "roi_xywh": [45, 37, 15, 13], "raw": signal, "fused": final}})
    return ('<section id="optical-cases"><h2>原片中的离焦现象</h2>'
            '<p>左图是 29 层短栈中同一条右侧外缘的不同源层，右图是各方法的同一剖面。'
            '原始焦层另有一个明暗随焦点反转的小区域。</p><div class="curve">'
            + fig("case-measurements/rim-profiles.png", "同一外缘的原片与融合亮度剖面")
            + f'</div><p>{link("case-measurements/case-profiles.json","逐层数值和坐标")} · '
            f'{link("#"+rim_view,"短栈 ROI")} · {link("#"+small_view,"有亮芯的 ROI")}</p></section>')


def write_audit(root: Path, groups: dict, specs: dict) -> None:
    states = {}
    lock_path = root / "locked.json"
    lock = read_json(lock_path) if lock_path.is_file() else {}
    for split, data in groups.items():
        source_hash = sha256(root / "inputs" / f"{split}.json")
        if lock and source_hash != lock["index_hashes"][split]:
            raise ValueError(f"The frozen {split} source index has changed.")
        if split != "dev" and lock and any(set(methods) != {"Helicon-B", lock["method"]} for methods in data.values()):
            raise ValueError(f"Unmatched Helicon and candidate outputs in {split}.")
        if split != "dev" and lock:
            path = Path(read_json(root / f"{split}-pollen-fallback.json")["path"]) / "manifest.json"
            run = read_json(path)
            if (run["engine"]["identity_sha256"] != lock["identity_sha256"] or run["created_at"] < lock["frozen_at"]
                    or any(task["options"] != lock["options"] for task in run["tasks"])):
                raise ValueError(f"The {split} run differs from its freeze record.")
        for view, methods in data.items():
            for name, fused in methods.items():
                if sha256(root / fused["path"]) != fused["sha256"]:
                    raise ValueError(f"The output has changed: {fused['path']}")
                if name == "Helicon-B":
                    folder = (root / fused["path"]).parent
                    reference = read_json(folder / "run.json")
                    if (reference["tiff_compression"] != 1 or reference["png"]["sha256"] != fused["sha256"]
                            or [frame["sha256"] for frame in reference["inputs"]] != specs[split][view]["hashes"]
                            or (split != "dev" and lock and reference["started_at"] < lock["frozen_at"])):
                        raise ValueError(f"The Helicon reference differs from the measurement for {view}.")
        states[split] = {"views": len(data), "frames_verified_during_roi_measurement": sum(len(v["hashes"])
                        for v in specs[split].values()), "results_checked": sum(map(len, data.values())),
                        "input_index_sha256": source_hash}
    write_json(root / "audit.json", states)


def render(root: Path) -> None:
    cv2.setNumThreads(2)
    groups = {split: read_json(root / f"{split}-lighting-metrics.json") for split in SPLITS
              if (root / f"{split}-lighting-metrics.json").is_file()}
    specs = {split: read_json(root / f"{split}-lighting-rois.json") for split in groups}
    notes = read_json(root / "review.json") if (root / "review.json").is_file() else {"summary": [], "views": {}}
    write_audit(root, groups, specs)
    index = " · ".join(link("#"+view, view[9:]) for split in groups for view in groups[split])
    summary = "".join(f"<li>{html.escape(line)}</li>" for line in notes["summary"])
    tables = "".join(summarize(split, records) for split, records in reversed(groups.items()))
    case = native_cases(root, groups["dev"], specs["dev"]) if "dev" in groups else ""
    results = "".join(render_views(root, split, records, specs[split], notes["views"])
                      for split, records in groups.items())
    paths = ["audit.json", "locked.json", "detail-crop-trial/measurements.json", "pytest.txt"]
    for split in groups:
        paths.extend([f"inputs/{split}.json", f"{split}-lighting-metrics.csv", f"{split}-lighting-rois.json"])
        for stage in ("native-locked", "pollen-locked", "pollen-revised", "pollen-fallback"):
            pointer = root / f"{split}-{stage}.json"
            if pointer.is_file():
                manifest = Path(read_json(pointer)["path"]) / "manifest.json"
                paths.append(manifest.relative_to(root).as_posix())
    artifacts = " · ".join(link(path, path) for path in paths if (root / path).is_file())
    report = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>新打光花粉景深融合对照</title><style>{CSS}</style></head><body><main>
<header><h1>新打光花粉景深融合对照</h1>
<p>同一原始焦栈 · Helicon Focus 8.1.0 方法 B，半径 8，平滑 4 · 2026 年 11 月 27 日</p></header>
<section class="findings"><h2>实拍结果</h2><ul>{summary}</ul></section>
<section><h2>测量对照</h2><p>各数值为对应组的视野中位数，括号是有效视野数。结构相关使用原始焦层中预选的 48 px 结构锚点，
成片锚点允许至多 8 px 的轴向平移。逐行 CSV 同时保留固定坐标相关和匹配量。相关用于核对结构的位置与极性，仍需检查光晕和实际纹理。
背景 RMS 只使用原片定义的低结构、未近白像素。近白掩膜以三个通道都至少 250 计，近白区域的原片纹理可见性有限。
Helicon 耗时包括保存无压缩 TIFF，转换对照 PNG 另计。</p>{tables}</section>
{case}
<nav>{index}</nav>
{results}
<section><h2>原始索引、执行记录与测量</h2><p>{artifacts}</p></section></main></body></html>"""
    write_text(root / "report.html", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    render(parser.parse_args().root.resolve())
