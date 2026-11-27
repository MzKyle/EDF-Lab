# 新打光花粉景深融合验证

本次从用户提供的各焦层图片目录读取 40 个视野、1,979 张原片。解码尺寸为 4096 × 3000，39 个视野为 50 层，00:02:46 的第 8 个视野为 29 层。焦层使用 SN 数字升序，保留文件名的原始 Z 标记和每张原片的 SHA-256。1,979 个文件哈希均唯一。文件目录日期为 2025-04-23。

6 个开发视野取自 00:02:46 和 00:09:07，包含 29 层短栈。00:41:57 的 16 个视野整次采集留出。其他 18 个同批视野用于扩大覆盖，00:02:46 和 00:09:07 中可见重复的颗粒布置。结果可解读为这批数据的对照。 [原片缩览](reports/2026-11-27-new-lighting-edf/pilot/capture-overlap.png)

## 明场融合

明场候选使用细尺度带通评价，并扣减较宽离焦结构的响应。带通评分存储为半精度，使 50 层输入保有更密的三倍降采样评分网格。每个位置从连续焦层分配软权重，成片仍使用原始像素分辨率。开发试验比较原位加权、两级金字塔和带邻域约束的三级金字塔。邻域高斯的标准差相当于 24 个原图像素，这组参数对应 GUI 和 CLI 中的 `PollenEDF-Brightfield-Context` 预设。

## 开发证据

- 29 层视野的右外壁随焦层位置变化。相同横向剖面中，Adaptive 在 x≈311 和 327 处形成两个暗谷，首轮明场金字塔在 x≈307 和 333 处形成两个暗谷。邻域版在 x≈313 处保留一个主暗谷，Helicon B 的暗谷在 x≈312。 [源焦层和融合剖面](reports/2026-11-27-new-lighting-edf/report.html#optical-cases)
- 另一视野的 15 × 13 px 小区域在原片 Z=+2 的均值约为 249.3，Z=+30 降至 102.7。Helicon B、旧 Adaptive、明场邻域版分别为 106.7、211.6、181.9。邻域版减轻了被离焦亮斑抬高的程度，中心仍偏亮。 [小颗粒和光学层数据](reports/2026-11-27-new-lighting-edf/case-measurements/case-profiles.json)
- 三个高亮背景窗口中，所有原始焦层的 BGR 都至少为 250 的像素占 16.2%–42.2%。该像素区缺少足够的可见强度余量。背景残差只比较原片中低结构且未达到近白阈值的像素。 [相同原始窗口与掩膜](reports/2026-11-27-new-lighting-edf/dev-lighting-rois.json)

## 整次采集留出结果

下表只统计 00:41:57 的 16 个留出视野。窗口匹配后计算同一批原片结构锚点的带通相关和幅度。小颗粒行使用双方均有可信匹配的 14 个视野。

| 指标中位数 | Helicon B | 明场邻域版 |
| --- | ---: | ---: |
| 外壁结构带通相关 | 0.866 | 0.907 |
| 外壁细节相对幅度 | 0.883 | 0.818 |
| 小颗粒结构带通相关 | 0.968 | 0.972 |
| 小颗粒细节相对幅度 | 1.014 | 0.879 |
| 低结构背景残差 RMS | 0.655 | 0.445 |
| 峰值 RSS（MiB） | 3116 | 851 |
| 单视野耗时（秒） | 4.55 | 20.64 |

14/16 个外壁窗口的结构相关较高，16/16 个背景窗口的残差较低。Helicon 的小颗粒相对幅度在 14/14 个有效窗口中更高，通常也更接近原片。第 5 个视野左侧浅色颗粒的局部亮度纹理在明场邻域版中较连贯，第 14 个视野的两枚暗色颗粒在 Helicon 中保留较鲜明的细纹。新方案改善了这批数据的部分轮廓和背景，外壁与小颗粒的细节对比仍不足以支持整体超过 Helicon B。 [留出 CSV](reports/2026-11-27-new-lighting-edf/holdout-lighting-metrics.csv) · [外壁对照](reports/2026-11-27-new-lighting-edf/pilot/holdout-walls.png) · [小颗粒对照](reports/2026-11-27-new-lighting-edf/pilot/holdout-small-particles.png)

同批的其余 18 个视野也使用冻结参数。17/18 的外壁相关更高，13 个可用低结构窗口的背景残差均更低。18 个外壁窗口的细节幅度均低于 Helicon。三组共核验 1,979 张原片和 110 次全视野结果。 [同批其余视野 CSV](reports/2026-11-27-new-lighting-edf/coverage-lighting-metrics.csv) · [输入与结果审计](reports/2026-11-27-new-lighting-edf/audit.json)

## 细节对比的裁片复验

在 6 个开发视野的外壁与小颗粒裁片中，保留标准邻域版的低频权重，只将最细两级频带的焦层权重平方后重新归一化。12 个窗口的细节幅度均提高。外壁幅度中位数从 0.843 到 0.893，小颗粒从 0.833 到 0.887。短栈右外壁仍在 x≈313 处保持单一暗谷。低结构像素的高通残差也升高，这组结果适用于有边距的开发裁片。冻结的全视野结果使用标准邻域版。 [裁片测量及输入条件](reports/2026-11-27-new-lighting-edf/detail-crop-trial/measurements.json) · [短栈标准裁片](reports/2026-11-27-new-lighting-edf/detail-crop-trial/20250423_000246_v008/wall-context.png) · [短栈细频带裁片](reports/2026-11-27-new-lighting-edf/detail-crop-trial/20250423_000246_v008/wall-detail.png)

## 复跑

阶段按视野串行。明场邻域版从开发数据选出。运行留出前固化参数、引擎身份及输入索引哈希。各方法都读同一组原图。Helicon Focus 8.1.0 以方法 B、半径 8、平滑 4 生成无压缩 TIFF，再无损转成比较用 PNG。

~~~powershell
$lab = '.\.venv\Scripts\python.exe'
$out = '.\artifacts\new-light-20261127'
$archive = 'E:\新打光方案\各焦层图片'
$helicon = 'D:\Helicon Focus 8\_HeliconFocus.exe'
& $lab .\tools\lighting_benchmark.py prepare --archive $archive --output $out
& $lab .\tools\lighting_benchmark.py baseline --output $out --helicon $helicon
& $lab .\tools\lighting_benchmark.py pollen-revised --output $out
& $lab .\tools\lighting_benchmark.py pollen-fallback --output $out
& $lab .\tools\lighting_measure.py $out
& $lab .\tools\lighting_benchmark.py freeze --output $out --method PollenEDF-Brightfield-Context
foreach ($split in @('holdout', 'coverage')) {
    & $lab .\tools\lighting_benchmark.py validate --output $out --split $split --helicon $helicon
    & $lab .\tools\lighting_measure.py $out --split $split
}
& $lab .\tools\lighting_detail_trial.py $out
& $lab .\tools\lighting_report.py $out
~~~

外壁和小颗粒的带通相关检验原始焦层可见结构、方向与位置。结构锚点仅依据原片选定。测量保留固定坐标和小范围校对两种相关结果。细节幅度、轮廓、光晕和小颗粒可见性还需从同视野对照图核对。 [HTML 报告](reports/2026-11-27-new-lighting-edf/report.html) · [参数冻结记录](reports/2026-11-27-new-lighting-edf/locked.json)

项目测试通过 30 项，包含固定发布版 focus-stack 的原生集成测试。 [测试日志](reports/2026-11-27-new-lighting-edf/pytest.txt)
