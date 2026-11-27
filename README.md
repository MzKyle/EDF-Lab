# Pollen-EDF-Lab 0.2

Windows x64 上运行的离线景深合成实验工作台。用 PySide6 浏览显微镜 Z-stack 和对照结果。V0.2 新增使用 NumPy 和 OpenCV 的 PollenEDF，保留固定版本的原生 focus-stack 和 Helicon 参考。GUI 和 CLI 共用数据校验、实验编排、指标和记录模块。

## 安装与启动

需要 Windows x64 和 `uv`。依赖通过 `uv.lock` 固定，本机已验证 CPython 3.11.16。可以先用 `winget install --id astral-sh.uv -e` 安装 `uv`。 [uv 官方安装文档](https://docs.astral.sh/uv/getting-started/installation/)

在项目根目录运行以下命令。

```powershell
.\scripts\setup.ps1
.\scripts\check_environment.ps1
.\scripts\run.ps1
```

`setup.ps1` 建立 `.venv`，并安装带 SHA-256 校验的官方 focus-stack 1.5 Windows 归档。使用手动带入的归档时，运行 `.\scripts\setup.ps1 -EngineArchive 'D:\离线软件\focus-stack_Windows.zip'`。离线安装可在 Python 3.11 和锁定依赖已进入本机 `uv` 缓存时加 `-Offline`。安装后，浏览、校验、融合、指标和 HTML 报告均可离线运行。

使用 `-SkipEngine` 可先部署 Python 和 GUI。引擎可以稍后用 `.\scripts\setup_focus_stack.ps1` 安装，工作台上的 **检测引擎** 可刷新状态。`.\scripts\run.ps1 doctor --json` 提供完整探测记录。自定义固定提交构建可通过 `POLLEN_EDF_ENGINE`、`--engine` 或本地 YAML 的 `engine` 指定。 [引擎集成与参数](docs/focus-stack-integration.md)

若新电脑提示缺少 MSVC 运行时 DLL，可安装 Microsoft Visual C++ v14 Redistributable 的 x64 版本，然后重新检测引擎。 [Microsoft 下载说明](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170)

## 用模拟数据上手

```powershell
.\scripts\run.ps1 generate-test-data --output '.\datasets\SYNTHETIC_demo'
.\scripts\run.ps1 inspect --input '.\datasets\SYNTHETIC_demo\dataset.json' --hash
.\scripts\run.ps1 fuse --input '.\datasets\SYNTHETIC_demo\dataset.json' --view XY_00_00 --preset FS_Default --depth
```

用 GUI 导入 `dataset.json` 或模拟数据目录，选择 `XY_00_00`。在 **原始焦层** 用滑块、左右方向键或 `[`、`]` 浏览，在右栏配置预设与焦层选择。点 **运行实验** 后到 **结果比对** 查看输出。**同步视图** 联动缩放和平移，**1:1** 按物理屏幕像素显示，**框选 ROI** 与 **放大 ROI** 用于局部检查。原始焦层 ROI 可绘制清晰度曲线，结果页可测量两图各自 ROI，并保存人工观察。

模拟生成器提供确定性纹理、离焦、噪声、平移、倍率和亮度变化。默认包含一个纹理视野和一个空白视野。`--faults` 加入缺层和损坏样本，`--stress` 生成一个 4096 × 3000、50 层视野。`--help` 列出生成参数。

## 导入真实采集

1. 单视野可选原图目录，批量采集可选其父目录。目录解析识别显式 `Z_12`、`Z12` 等编号。文件名另有规则时使用 JSON/CSV 索引，或在 **图片列表** 中按已知采集顺序明确排列。
2. 在 GUI 校验所选视野，也可以运行 `inspect --input <数据路径> --hash` 校验整个数据集。用采集信息复核 XY 分组、Z 方向、缺层和 µm 坐标。
3. 先在代表性视野运行 `FS_Default`，再选择 `FS_NoAlign`、`FS_GeometryOnly` 等对照。选中对应视野后，用 **关联 Helicon** 打开已有合成图并填写已知版本、方法及参数。Helicon 图作为人工审查参考。
4. 从 **结果比对** 查看同一标本区域，从 **原始焦层** 查阅对应细节。用 **记录审查** 保存 ROI 与观察，用 **导出报告** 生成离线 HTML。批量任务在 **恢复与重试** 继续已有实验目录。

真实文件可保存在任意目录。Lab 读取源图，实验写入独立的 `artifacts/experiment_*` 目录。 [数据索引格式](docs/dataset-format.md) · [实验方法与指标](docs/experiment-protocol.md)

## CLI 示例

按预设选择自研引擎。焦层图、原始选择索引和置信度图随默认结果保存。

~~~powershell
.\scripts\run.ps1 fuse --input '.\datasets\dataset.json' --view XY_00_00 --preset PollenEDF-Adaptive --intermediate
.\scripts\run.ps1 benchmark --input '.\datasets\dataset.json' --backend pollen-edf
.\scripts\run.ps1 doctor --backend pollen-edf
~~~

PollenEDF 提供 SML、Tenengrad 硬选择、多尺度深度图和自适应四种预设。CLI 参数可逐项控制噪声感知评分、焦层优化、背景融合、孤立亮点处理、低置信度软融合和拉普拉斯金字塔。GUI 的预设列表可运行以上四种方法，结果页可打开焦层图和置信度图。 [V0.2 算法与实拍复跑方法](docs/pollen-edf-v02.md)

新打光的明背景焦栈可试用 `PollenEDF-Brightfield` 和 `PollenEDF-Brightfield-Context`。两种实验预设以连续焦层权重合成，后者加入邻域评分和金字塔融合。 [新打光试验及复跑方法](docs/new-lighting-edf.md)

以下命令使用固定的原生 focus-stack。

```powershell
.\scripts\run.ps1 benchmark --input '.\datasets\SYNTHETIC_demo\dataset.json' --view XY_00_00 --preset FS_Default --preset FS_NoAlign
.\scripts\run.ps1 benchmark --input '.\datasets\SYNTHETIC_demo\dataset.json' --view XY_00_00 --preset FS_CPU --sample stride --step 3 --offset 0 --offset 1 --keep-endpoints
.\scripts\run.ps1 fuse --input '.\datasets\SYNTHETIC_demo\dataset.json' --view XY_00_00 --sample range --range 4 9
```

`fuse` 接受多个 `--preset`，在一个视野串行执行。`benchmark` 可选择多个 `--view`，缺省处理全部视野和全部预设。恢复用 `benchmark --resume <experiment_目录>`，报告重新生成用 `report --experiment <experiment_目录>`。融合时按 `Ctrl+C` 可取消。

外部配置示例

```yaml
artifacts_dir: D:/EDF/Results
timeout_s: 1200
preview_cache_mib: 112
```

用 `.\scripts\run.ps1 gui --config 'D:\EDF\lab.yaml'` 加载。外部 YAML 的相对路径以该文件所在目录为基准。实验目录包含 `manifest.json`、`config.yaml`、`summary.csv`、`report.html` 和各视野的运行目录。成功任务可查看 PNG、指标和执行日志。重试产生新的 `attempt_*`。

## 验证与项目文件

```powershell
.\.venv\Scripts\python.exe -m pytest -m 'not native'
.\.venv\Scripts\python.exe -m pytest -m native
```

普通测试覆盖数据校验、抽层、模拟进程、异常输出、实验恢复与 Qt 任务状态。`native` 标记使用确切的官方 Windows 二进制，包含超长中文和空格路径实测。

- [工程架构](docs/architecture.md) 列出核心模块与记录目录。
- [第三方许可证](THIRD_PARTY_NOTICES.md) 记录原生归档、Wheel 与版权说明。
- [验收记录](docs/verification.md) 提供实际运行、报告和日志路径。
- [实拍花粉景深融合对照](docs/reports/2026-11-26-pollen-edf/README.md) 收录 10 个真实视野的 focus-stack 与 Helicon 对照图、指标和运行证据。
- V0.2 实拍研发报告收录失效分析、单因素实验和日期留出验证。 [V0.2 实拍报告](docs/reports/2026-11-27-pollen-edf-v02/README.md)
- 新打光报告保留 40 个视野的明场融合对照、外壁剖面和裁片复验。 [新打光测试报告](docs/reports/2026-11-27-new-lighting-edf/README.md)
