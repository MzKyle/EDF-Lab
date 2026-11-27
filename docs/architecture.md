# 工程架构

```text
scripts/run.ps1 → pollen_edf.cli
                       ├── inspect / reference / report / generate-test-data
                       ├── GUI (PySide6)
                       └── fuse / benchmark
                               ↓
             dataset → sampling → ExperimentRunner ┬→ FocusStackEngine → 官方原生进程
                              └── records/evaluation └→ PollenEDFEngine → Python 工作进程
```

| 文件 | 职责 |
| --- | --- |
| `config.py`、`resources/` | 默认配置、预设和官方制品锁定信息 |
| `dataset.py`、`images.py` | 目录与索引导入、逐帧验证、Unicode 和长路径 I/O、PNG 校验 |
| `sampling.py`、`experiments.py` | 可复现抽层、格式化待运行计划、串行执行、超时与取消状态、恢复 |
| `engine.py`、`process_runner.py` | 原生能力探测、明确参数映射、输入暂存、Windows Job Object、日志和 RSS 监控 |
| `pollen_algorithm.py`、`pollen_engine.py`、`pollen_worker.py` | 自研清晰度评分、焦层选择和融合，复用可取消的工作进程与产物校验 |
| `evaluation.py` | 真实像素 ROI、像素对应映射、辅助指标和焦层曲线 |
| `records.py` | 原子更新、独占实验锁、Helicon 参考、人工审查、CSV 和 HTML |
| `ui.py`、`ui_compare.py`、`ui_parameters.py` | 工作台、比较器、参数和抽层控件 |
| `ui_viewer.py`、`ui_workers.py` | 预览、缓存、工作线程与最新任务合并 |
| `synthetic.py`、`tools/generate_synthetic_zstack.py` | 可重复的 `SYNTHETIC` 测试输入 |
| `tests/` | 后端、模拟进程、Qt 状态和真实原生引擎集成 |

批处理按视野及参数组合依次启动融合进程。预检逐帧解码并释放，只保留文件信息和 SHA-256。原生引擎用经哈希核对的短 ASCII 路径副本向窄字符 CLI 传入源图。PollenEDF 的工作进程根据同一记录逐张读原图，使用受内存上限约束的评分体积，再逐张读入涉及的焦层完成融合。实验输出以 PNG 校验后写入单独的尝试目录。Windows 进程在恢复运行前加入 Job Object，结束、取消或超时都关闭 Job Object 以回收整棵进程树。

GUI 最多运行两个同时解码的预览任务。缩略图和全尺寸图分别进入按字节限制的 LRU，默认合计 112 MiB。场景坐标使用原始像素，放大后才请求全尺寸图。切换焦层时会使旧预览任务失效。ROI 指标由 OpenCV 在后台读取实际像素。单个预览窗口只保留当前图，处理进程和当前显示图各自占用额外内存。

## 记录目录

```text
artifacts/experiment_<UTC>_<id>/
├── manifest.json
├── config.yaml
├── summary.csv
├── report.html
├── reviews/                            # 选择记录人工审查时创建
└── view_001_XY_00_00/001_FS_Default/
    ├── attempt_001/
    │   ├── execution.log
    │   ├── fused.png
    │   ├── depth.png                    # 可选
    │   ├── confidence.png               # PollenEDF，可选
    │   ├── depth-index.png              # PollenEDF，16 位序号，可选
    │   ├── metrics.json
    │   └── steps/                       # 可选中间图
    └── attempt_002/                     # 重试时创建
```

Manifest 保存原始有序焦层、实际选层和哈希、引擎版本与二进制 SHA-256、实际参数数组、硬件探测、输出哈希及不同尝试的状态。`success` 结果来自退出码 0 且通过完整性检查的 PNG。外部程序强制结束后，`running` 尝试可在恢复时转为 `interrupted`。恢复会复核已有输入及成功输出哈希，保留校验通过的任务并重试其余项。实验目录内报告使用相对链接，可在本地直接打开。另存到其他目录时报告引用原实验文件的本地路径。

Helicon 关联记录以数据索引标识和 XY 视野为键存入 `artifacts/.references`。关联保存图片路径、SHA-256、尺寸、版本和已知参数，后续实验会把当前关联带入 Manifest。ROI 审查记录所看图像的路径与哈希、对应坐标、辅助指标和人工观察。
