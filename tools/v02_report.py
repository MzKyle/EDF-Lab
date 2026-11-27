"""Build the offline V0.2 scientific report from the measured artifacts."""
from __future__ import annotations

import argparse
import html
from pathlib import Path
from statistics import median
from urllib.parse import quote

import numpy as np

from pollen_edf.images import read_image
from pollen_edf.records import read_json, write_text
from v02_benchmark import NATIVE_TUNING, POLLEN_ABLATIONS, POLLEN_NAMES
from v02_measure import png


BASELINES = ("Helicon-B", "FS_Default", "FS_Tuned", *POLLEN_NAMES, "PollenEDF-Adaptive-Guard",
             "PollenEDF-Adaptive-GuardPool", "PollenEDF-Adaptive-GuardFallback")
CSS = """
:root{color-scheme:light;--ink:#17251f;--muted:#52645b;--line:#d9e2d9;--link:#0b6351}
*{box-sizing:border-box}
body{margin:0;background:#f2f5f1;color:var(--ink);font:15px/1.58 "Segoe UI","Microsoft YaHei",sans-serif}
main{max-width:1460px;margin:auto;padding:28px 24px 50px}
h1,h2,h3,p,figure{margin:0}
h1{font-size:clamp(25px,4vw,42px);line-height:1.22}
header p,.muted,figcaption{color:var(--muted)}
header p,section p{margin-top:8px}
a{color:var(--link);text-underline-offset:3px}
nav{display:flex;gap:8px 16px;flex-wrap:wrap;padding:14px 0}
section{border:1px solid var(--line);background:#fff;padding:18px 21px;margin:16px 0;border-radius:8px}
.findings{background:#e8f2e9;border-left:5px solid var(--link)}
.findings ul{margin-bottom:0;padding-left:22px}
.findings li+li{margin-top:7px}
.table{overflow:auto}
table{border-collapse:collapse;width:100%;margin:10px 0 0;font-variant-numeric:tabular-nums}
td,th{padding:8px 9px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}
thead{background:#eff4ee}
.roi{margin:16px 0 0;padding-top:12px;border-top:1px solid var(--line)}
.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:8px 0}
figure img{width:100%;height:auto;display:block;border:1px solid var(--line);background:#060907;border-radius:3px}
figcaption{padding-top:3px;font-size:13px}
.curve{max-width:720px}
blockquote{margin:8px 0;padding:6px 13px;background:#f3f6f1;border-left:3px solid var(--line)}
details{margin:9px 0}
summary{color:var(--link);cursor:pointer}
@media(max-width:732px){main{padding:16px 10px}section{padding:12px 10px}.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
"""


def link(path, label):
    return f'<a href="{quote(str(path), safe="/#")}">{html.escape(str(label))}</a>'


def fig(path, label):
    name = html.escape(str(label))
    uri = quote(str(path), safe="/")
    return (f'<figure><a href="{uri}"><img src="{uri}" alt="{name}" loading="lazy" '
            f'decoding="async"></a><figcaption>{name}</figcaption></figure>')


def optional(path, root, label):
    return link(path, label) if (root / path).is_file() else ""


def fmt(number, decimals=2):
    return "—" if number is None else f"{number:.{decimals}f}"


def centre(values):
    complete = [value for value in values if value is not None]
    return median(complete) if complete else None


def condition(case, method):
    return [view[method] for view in case.values() if method in view]


def dark(method, key):
    roi = method["regions"].get("dark", {})
    return roi.get("metrics", {}).get(key) if roi.get("usable") else None


def texture(method, key):
    roi = method["regions"].get("detail", {})
    return roi.get("metrics", {}).get(key) if roi.get("usable") else None


def method_label(split, method):
    if method == "PollenEDF-Adaptive-GuardFallback":
        return "PollenEDF-Adaptive · 冻结"
    if method == "PollenEDF-Adaptive":
        return method + (" · 冻结" if split == "holdout" else " · 首轮")
    return method


def method_table(groups, methods):
    rows = []
    for split, data in groups:
        for method in methods:
            variants = condition(data, method)
            if not variants:
                continue
            stats = [fmt(centre(details[key] for details in variants), precision)
                     for key, precision in (("elapsed_s", 2), ("rss_mib", 0))]
            stats.extend([fmt(centre(dark(item, "dark_rms") for item in variants), 3),
                          fmt(centre(dark(item, "trail_pixels") for item in variants), 1),
                          fmt(centre(texture(item, "fg_luma") for item in variants), 2),
                          fmt(centre(texture(item, "source_edge_corr") for item in variants), 3)])
            values = [split, method_label(split, method), f"{len(variants)}/{len(data)}", *stats]
            rows.append("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in values) + "</tr>")
    return ('<div class="table"><table><thead><tr><th>集</th><th>方法</th><th>完成视野</th>'
            '<th>秒</th><th>峰值 RSS MiB</th><th>暗区高通 RMS</th><th>孤立点邻域亮残差像素</th>'
            '<th>前景灰度</th><th>与原焦层边缘相关</th></tr></thead><tbody>'
            + "".join(rows) + '</tbody></table></div>')


def render_views(root, split, data, specs, annotations):
    sections = []
    for view, methods in data.items():
        spec = specs[view]
        label = f'{spec["meta"]["date"]} · {spec["meta"]["capture"]} · {spec["meta"]["source_view"]}'
        ordered = [method for method in BASELINES if method in methods]
        roi_sections = []
        for name, region in spec["regions"].items():
            raw = region["folder"]
            title = {"detail": "结构窗口", "small": "紧凑小颗粒窗口", "second": "第二结构窗口",
                     "dark": "独立暗区", "prior": "V0.1 重点窗口"}[name]
            cards = [fig(f"{raw}/source.png", f'原始焦层 Z={region["best"]["z"]:+d}')]
            extra = []
            for method, version in methods.items():
                xy = version["regions"][name]
                item = f'{version["folder"]}/{name}.png'
                if (root / item).is_file():
                    text = method_label(split, method) + (" · 低匹配" if not xy["usable"] else "")
                    (cards if method in ordered else extra).append(fig(item, text))
            chosen = methods.get("PollenEDF-Adaptive-GuardFallback", methods.get("PollenEDF-Adaptive"))
            maps = []
            if chosen:
                for name_map, label_map in (("depth", "焦层图"), ("confidence", "置信度"),
                                            ("foreground", "前景权重"), ("alternate", "竞争焦层"),
                                            ("halo", "离焦间隙修正权重"), ("broad", "宽峰混合权重"),
                                            ("anchor", "单层回退权重")):
                    item = f'{chosen["folder"]}/{name}-{name_map}.png'
                    if (root / item).is_file():
                        maps.append(fig(item, label_map))
            more = f'<details><summary>本窗口的消融输出</summary><div class="grid">{"".join(extra)}</div></details>' if extra else ""
            evidence = [link(f"{raw}/focus.png", "完整焦点曲线"), link(f"{raw}/dark-mask.png", "原片暗区掩膜"),
                        link(f"{raw}/recurring-points.png", "原片重复亮点掩膜")]
            profile = ""
            if (root / raw / "edge-profile.png").is_file():
                profile = ('<details><summary>原片边缘法向亮度曲线</summary><div class="grid">'
                           + fig(f"{raw}/edge-profile-location.png", "原片选定位置")
                           + fig(f"{raw}/edge-profile.png", "同坐标亮度曲线") + '</div></details>')
            prefix = f'{title} · 相机像素 ({region["xy"][0]}, {region["xy"][1]}) · 384 × 288'
            roi_sections.append(f'<div class="roi"><h3>{prefix}</h3><div class="grid">{"".join(cards)}</div>'
                                f'<div class="grid">{"".join(maps)}</div><p>{" · ".join(evidence)}</p>'
                                f'<details><summary>焦点评分曲线</summary><div class="curve">{fig(f"{raw}/focus.png","原始焦层")}</div></details>'
                                f'{profile}{more}</div>')
        full = " · ".join(link(methods[m]["path"], method_label(split, m)) for m in ordered)
        logs = []
        for method in ordered:
            path = Path(methods[method]["path"]).with_name(
                "application.log" if method == "Helicon-B" else "execution.log"
            )
            if (root / path).is_file():
                logs.append(link(path.as_posix(), method_label(split, method)))
        annotation = annotations.get(view, "")
        note = f'<blockquote>{html.escape(annotation)}</blockquote>' if annotation else ""
        sections.append(f'<section id="{view}"><h2>{html.escape(label)}</h2>'
                        f'<p class="muted">{html.escape(split)} · {html.escape(spec["meta"]["scene"])}</p>'
                        f'<p>全分辨率成片　{full}</p><p>执行日志　{" · ".join(logs)}</p>'
                        f'{note}{"".join(roi_sections)}</section>')
    return "".join(sections)


def render_native(root):
    path = root / "native-diagnostics.json"
    if not path.is_file():
        return ""
    data = read_json(path)
    panels = []
    for label, report in data["reports"].items():
        figures = "".join(fig(f'{report["folder"]}/{file}.png', name) for file, name in (
            ("source", "最佳源焦层"), ("inverse-wavelet", "逆小波灰度"), ("final", "颜色重建后")))
        metric = report["dark_metrics"]
        levels = [fmt(metric[key]["rms"], 3) for key in (
            "raw_standard_luma", "inverse_wavelet_pca_gray", "final_standard_luma")]
        panels.append(f'<h3>{html.escape(label)}</h3><div class="grid">{figures}</div>'
                      f'<p>同窗口暗区高通 RMS　{" / ".join(levels)}</p>')
    traces = root / "alignment-points.json"
    if traces.is_file():
        report = read_json(traces)
        panels.append('<h3>配准与重复亮点</h3><p>用执行日志中的每层仿射角点预测相机固定亮点的轨迹。'
                      '两幅暗区局部用相同的 8 倍显示增益。</p>')
        for row in report["observations"]:
            panels.append(f'<p>相机位置 {row["location_in_camera"]} · 预测位移范围 '
                          f'{row["axis_aligned_displacement_px"]:.1f} px</p><div class="grid">'
                          + fig(row["native"], "默认配准") + fig(row["predicted"], "预测占位")
                          + fig(row["no_align"], "关闭配准") + '</div>')
        panels.append(f'<p>{link("alignment-points.json", "轨迹与残差记录")}</p>')
    return ('<section id="native-detail"><h2>原生中间图</h2>'
            '<p>同一真实像素区域分离保存逆小波灰度和颜色重建后的结果。灰度投影及调试 JPEG 编码会影响数值。</p>'
            + "".join(panels) + f'<p>{link("native-diagnostics.json", "完整测量")}</p></section>')


def render_gap(root):
    report = root / "gap-review" / "measurement.json"
    if not report.is_file():
        return ""
    samples = read_json(report)["samples"]
    rows = "".join(f'<tr><td>{html.escape(name)}</td><td>{html.escape(str(values["gap"]))}</td>'
                   f'<td>{html.escape(str(values["grain"]))}</td></tr>' for name, values in samples.items())
    return ('<section><h2>第一次自适应结果的间隙误选</h2>'
            '<p>三处颗粒间隙在原片中呈两端亮、中央暗的曲线。三处有结构的颗粒提供同区域对照。'
            '记录 7×7 像素中位亮度和修订前后的变化。</p><div class="grid">'
            + fig("gap-review/locations.png", "原片中的六个观察点")
            + fig("gap-review/curves.png", "六处原片 Z 曲线")
            + '</div><div class="table"><table><thead><tr><th>结果</th><th>三个间隙</th>'
            '<th>三个颗粒</th></tr></thead><tbody>' + rows
            + f'</tbody></table></div><p>{link("gap-review/measurement.json", "完整数据")}</p></section>')


def render_dark_example(root, development, reference):
    view = "20261029_102002_v001"
    methods = development.get(view, {})
    keys = ("Helicon-B", "FS_Default", "FS_Tuned", "PollenEDF-Adaptive-GuardFallback")
    if not all(key in methods for key in keys):
        return ""
    raw = f'{reference[view]["regions"]["dark"]["folder"]}/source.png'
    source = [("原始焦层", raw)] + [(method_label("dev", key), f'{methods[key]["folder"]}/dark.png') for key in keys]
    panels = []
    for i, (method, name) in enumerate(source, 1):
        image = read_image(root / name)
        gain = np.minimum(image.astype(np.uint16) * 8, 255).astype(np.uint8)
        file = f"dark-review/panel-{i}.png"
        png(root / file, gain)
        panels.append(fig(file, method))
    return ('<section><h2>近空白视野中的亮点轨迹</h2><p>10-29 近空白视野的同一暗区以固定 8 倍增益显示。'
            '数值比较仍使用原始亮度。</p><div class="grid">' + "".join(panels) + '</div></section>')


def render_holdout_failure(root, results):
    entry = "holdout-low-contrast-failure.json"
    if not results or not (root / "holdout-review" / "curves.png").is_file():
        return ""
    data = read_json(root / entry)
    view = data["view"]
    labels = ("1 · 颗粒表面", "2 · 颗粒间隙")
    rows = []
    for label, window in zip(labels, data["observations"].values()):
        layers = "–".join(f"{value:g}" for value in window["label_index_p10_p90"])
        values = [label, layers, fmt(window["local_confidence_median"], 3),
                  fmt(window["halo_mass_median"], 3), fmt(window["anchor_mass_median"], 3)]
        rows.append("<tr>" + "".join(f"<td>{value}</td>" for value in values) + "</tr>")
    panels = [fig(f'{results[view][m]["folder"]}/detail.png', m) for m in ("Helicon-B", "FS_Tuned")]
    panels.extend([fig("holdout-review/locations.png", "Adaptive 的块状误选位置"),
                   fig(f'{results[view]["PollenEDF-Adaptive"]["folder"]}/detail-confidence.png', "同区焦点置信度")])
    return ('<section id="holdout-failure"><h2>留出失败案例 · 11 月 3 日密集低对比视野</h2>'
            '<p>两处块状误选均位于局部低置信区域。焦层索引为输入顺序的 1 起始编号。'
            '全图有效前景内仍有足量高置信像素，因此该视野未触发整栈回退。</p>'
            f'<div class="grid">{"".join(panels)}</div><div class="curve">'
            + fig("holdout-review/curves.png", "两个窗口的原始焦点评分和亮度")
            + '</div><div class="table"><table><thead><tr><th>窗口</th><th>焦层索引 P10–P90</th>'
            '<th>置信度中位数</th><th>间隙修正权重</th><th>单层回退权重</th>'
            f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
            f'<p>{link(entry,"逐层数据和坐标")} · {link("#"+view,"同视野完整对照")}</p></section>')


def run_report(root):
    notes = read_json(root / "review.json") if (root / "review.json").exists() else {"findings": [], "views": {}}
    groups = [(split, read_json(root / f"{split}-metrics.json")) for split in ("dev", "holdout")
              if (root / f"{split}-metrics.json").is_file()]
    if not groups:
        raise ValueError("Generate the measurement records before the report.")
    navigation = " ".join(link("#"+view, view) for _, results in groups for view in results)
    failure = render_holdout_failure(root, dict(groups).get("holdout", {}))
    if failure:
        navigation = link("#holdout-failure", "留出失败案例") + " " + navigation
    observations = "".join(f"<li>{html.escape(text)}</li>" for text in notes["findings"])
    method_html = method_table(reversed(groups), BASELINES)
    ablations = method_table(groups[:1], (*NATIVE_TUNING, *POLLEN_NAMES, *POLLEN_ABLATIONS,
                                         "PollenEDF-Adaptive-Guard", "PollenEDF-Adaptive-GuardPool",
                                         "PollenEDF-Adaptive-GuardFallback"))
    views = "".join(render_views(root, title, data, read_json(root / f"{title}-roi-spec.json"), notes["views"])
                    for title, data in groups)
    dark_example = render_dark_example(root, groups[0][1], read_json(root / "dev-roi-spec.json"))
    paths = ["locked.json", "native-candidate.json", "native-ofat-plan.json", "native-capabilities.json",
             "pollen-capabilities.json", "dev-roi-spec.json", "holdout-roi-spec.json",
             "dev-measurements.csv", "holdout-measurements.csv", "dev-metrics.json", "holdout-metrics.json",
             "dev-native-ofat.json", "dev-native-locked.json", "dev-pollen-ofat.json", "holdout-native-locked.json",
             "holdout-pollen-locked.json", "dev-pollen-revised.json", "dev-pollen-fallback.json", "pytest.txt",
             "holdout-checks.json", "holdout-low-contrast-failure.json"]
    evidence = " · ".join(value for name in paths if (value := optional(name, root, name)))
    manifests = []
    for file in paths:
        full = root / file
        if full.is_file() and any(part in file for part in ("-ofat.json", "-locked.json", "-revised.json", "-fallback.json")):
            run = read_json(full)
            experiment = Path(run["path"]).relative_to(root)
            manifests.append(link((experiment / "report.html").as_posix(), f"{full.stem} 运行详情"))
    content = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PollenEDF V0.2 实拍实验</title><style>{CSS}</style></head><body><main>
<header><h1>PollenEDF V0.2 实拍实验</h1><p>2026 年 11 月 27 日 · 原生像素与固定焦层顺序 · 日期拆分</p></header>
<nav>{navigation}</nav>
<section class="findings"><h2>实验判断</h2><ul>{observations}</ul></section>
<section><h2>基线与冻结方案</h2><p>表内各计量为对应集内的视野中位数。前景数值取可信匹配的结构窗口。
边缘相关衡量一张原始焦层的局部相似性，适合辅助看图。Helicon 耗时包含 TIFF 写盘，转换 PNG 另计。</p>{method_html}</section>
{failure}
<section><h2>开发集单因素比较</h2><p>每个变体运行同一组 5 个开发视野。暗区高通 RMS 和孤立亮点邻域残差采用原片定义的窗口。
逐行坐标和置信匹配状态保存在 CSV。</p>{ablations}</section>
{render_native(root)}
{dark_example}
{render_gap(root)}
{views}
<section><h2>数据与记录</h2><p>原片确定观察窗口。每个方法的输出以相机坐标或整数相关匹配定位到相同的内部区域。
低匹配窗保留视野偏移，仅供看图。背景掩膜使用跨焦层亮度上界，孤立点要求在多数源焦层的同一位置出现。
固定版二进制与自研源码哈希分别由运行记录保存。</p><p>{evidence}</p><p>{" · ".join(manifests)}</p></section>
</main></body></html>"""
    write_text(root / "report.html", content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    run_report(args.root.resolve())


if __name__ == "__main__":
    main()
