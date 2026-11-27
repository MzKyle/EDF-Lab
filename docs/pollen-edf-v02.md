# PollenEDF V0.2 算法与验证方法

PollenEDF 使用 CPU 上的 NumPy 和 OpenCV，在原始相机像素坐标上选择和融合焦层。四种预设共用导入、抽层、进程生命周期和实验记录。官方 focus-stack 1.5 的二进制和预设保留为基线。

| 预设 | 评分及像素来源 |
| --- | --- |
| PollenEDF-SML | 两轴二阶差分绝对值的局部均值，每像素选一个实际焦层 |
| PollenEDF-Tenengrad | 局部 Sobel 能量的平方根，每像素选一个实际焦层 |
| PollenEDF-DepthMap | 多尺度评分、噪声门限、焦层置信度与边缘感知优化，硬选择 |
| PollenEDF-Adaptive | DepthMap 的选择图加暗背景平均、重复孤立亮点处理、低置信度软融合、离焦间隙修正和单层回退 |

## 可单独验证的阶段

1. 逐张解码。用面积插值生成计算网格，按焦栈体积自动增大降采样倍数，浮点评分体积上限 192 MiB。最终像素保持原始分辨率。
2. 多尺度 SML 以细尺度为主，抑制粗尺度上响应相当的缓慢亮度过渡。噪声感知评分先低通，再扣除该帧低亮采样估算的响应下限与随局部亮度变化的项。
3. 保存最大评分的实际输入索引。以该分数的绝对强度，以及与距离超过两层的最强竞争响应的差值估算置信度。源栈端点获胜时降低置信度。
4. 低置信度处允许边缘感知邻域投票。投票只采用邻域实际选中的层，要求该层在当前像素仍有足够评分，邻域亮度差增大时权重衰减。
5. 背景依据跨层的低频亮度上界估计，低亮变化小的区域用跨层原色平均。孤立点处理要求暗区中同一相机位置在至少 65% 焦层有局部高通尖峰，并检查邻域候选密度。候选只在受控暗区以邻域中值代换。
6. 前景高置信度区写入选层的完整 BGR 像素。低置信度处以最多 35% 权重混合一个竞争层。可选的拉普拉斯金字塔用各尺度的高斯权重合成，在高置信度区恢复选层原像素。
7. 离焦间隙修正检查 Z 序列两端是否比低分位亮度显著更高。在低置信度暗间隙中，从接近低分位的已拍焦层取样，逐渐替换误选的亮层。
8. 宽峰软融合统计有多少层的评分达到局部最高分的 80%。低置信且大量焦层评分接近时，提高这些候选层的混合占比，减少逐块跳层。狭窄峰与高置信像素保持原选择路径。
9. 单层回退用于高置信像素少于有效前景的 0.3% 的视野。在足量前景存在的情况下，从源栈选一张代表性焦层，让低置信前景退回可观察的原始结构。中间层的总体评分接近最佳时优先使用中间层。

各阶段可通过 `--scorer`、`--score-scale`、`--no-noise-aware`、`--no-refine`、`--no-background`、`--no-defects`、`--no-soft`、`--halo-guard`、`--broad-pool` 和 `--pyramid-levels` 做对照。`depth-index.png` 的无符号 16 位值从 1 起，按本次选层列表查回实际 Z。`depth.png` 是 8 位显示图，`confidence.png` 是 0 至 255 的相对置信度。背景上的索引保留评分获胜层，合成像素的来源仍取决于背景掩膜、置信度及混合权重。`--intermediate` 保留前景权重图、竞争层图、启用的修订权重图和阶段统计。

这版面向相机坐标稳定的焦栈。存在视野平移、旋转或尺度变化时，光学结构在相机坐标上的对应可能随 Z 改变。多峰评分表示局部存在可竞争焦点，适合结合原片核验。同一位置的真实亚像素颗粒与重复的相机亮点也可能难以区分。

## 实拍复跑

`tools/v02_benchmark.py` 固定开发和留出索引。开发集为 5 个视野，留出集为 4 个 V0.1 未参与合成对照的视野，采集日期与开发集互斥，均来自本机真实备份。选层按 SN 整数顺序，原名 Z 保存为相对位置标记。原始备份路径只读，运行和中间图写入独立实验目录。

按顺序准备索引、提取开发 ROI、跑原生和自研单因素实验并生成测量结果。Helicon 只作为参考运行，使用 B、半径 8、平滑 4 和经文件签名及 TIFF 压缩标签核对的无损输出。离线比较使用从该 TIFF 无损转换的 PNG。各原生变体沿用 8 线程、批次 8、其它选项与基线相同。官方构建缺少 OpenCL 支持时，保留实际 CPU 日志。

~~~powershell
$lab = '.\.venv\Scripts\python.exe'
$archive = 'E:\原图备份'
$output = '.\artifacts\v02-real-20261127'
$helicon = '<签名版 Helicon Focus 8.1.0 路径>'
& $lab .\tools\v02_benchmark.py prepare --archive $archive --output $output
& $lab .\tools\v02_measure.py raw $output
& $lab .\tools\v02_benchmark.py native-ofat --archive $archive --output $output
& $lab .\tools\v02_benchmark.py helicon --archive $archive --output $output --helicon $helicon
& $lab .\tools\v02_intermediates.py $output
& $lab .\tools\v02_benchmark.py native-locked --archive $archive --output $output
& $lab .\tools\v02_benchmark.py pollen-ofat --archive $archive --output $output
& $lab .\tools\v02_benchmark.py pollen-revised --archive $archive --output $output
& $lab .\tools\v02_benchmark.py pollen-fallback --archive $archive --output $output
& $lab .\tools\v02_measure.py compare $output
~~~

开发集确认后，用 `freeze` 固化留出方案。记录包含自研实现与运行库的身份哈希、选定的自主参数及 focus-stack 调优选项。当前冻结选择使用间隙修正与单层回退，宽峰增强保持可选。`--weak-stack-anchor` 控制单层回退。运行目录保存排序后的原图哈希、算法选项、原生和 Python 执行日志、耗时和进程树采样峰值 RSS。

~~~powershell
& $lab .\tools\v02_benchmark.py freeze --archive $archive --output $output
& $lab .\tools\v02_measure.py raw $output --split holdout
& $lab .\tools\v02_benchmark.py helicon --archive $archive --output $output --helicon $helicon --split holdout
& $lab .\tools\v02_benchmark.py native-locked --archive $archive --output $output --split holdout
& $lab .\tools\v02_benchmark.py pollen-locked --archive $archive --output $output --split holdout
& $lab .\tools\v02_measure.py compare $output --split holdout
& $lab .\tools\v02_measure.py audit $output
& $lab .\tools\v02_report.py $output
~~~

无效边缘按各方法共同可测的内区处理。结构 ROI 在原片上预定。关闭配准的原生和自研成片使用相机坐标，有配准的输出做整数像素配对，全程保持原始采样。CSV 保留相机及输出坐标、匹配相关系数和低匹配标志。低匹配窗口沿用视野偏移供观察，缺少可用于对照的精确对应。暗区掩膜取自全部原片的平滑亮度上界。指标结合真实原片、清晰度曲线、轮廓和纹理解读，Helicon 是参考，源焦层本身也有景深和噪声限制。

上游小波流程为同色阶映射、配准、复小波分解、按系数幅度挑选、一致性投票、软阈值、逆变换与颜色映射。`v02_intermediates.py` 用真实原生像素 ROI 检查逆变换灰度与最终颜色输出。调试 JPEG 与最终无损 PNG 的度量应结合编码精度及灰度投影差异解释。 [固定版配准](https://github.com/PetteriAimonen/focus-stack/blob/0cd289e224fe24aead5b146faeaa46950775f346/src/task_align.cc) · [系数选择](https://github.com/PetteriAimonen/focus-stack/blob/0cd289e224fe24aead5b146faeaa46950775f346/src/task_merge.cc) · [降噪](https://github.com/PetteriAimonen/focus-stack/blob/0cd289e224fe24aead5b146faeaa46950775f346/src/task_denoise.cc) · [颜色重建](https://github.com/PetteriAimonen/focus-stack/blob/0cd289e224fe24aead5b146faeaa46950775f346/src/task_reassign.cc) · [Helicon 命令行](https://www.heliconsoft.com/focus/help/english/HeliconFocus.html#HF_COM_LINE)
