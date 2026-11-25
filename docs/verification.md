# V0.1 本机验收记录

验证日期为 2026 年 11 月 25 日。环境使用 Windows 11 x64、CPython 3.11.16、Core i7-14700K 和官方 focus-stack 1.5 Windows Release。已验证从本地 ZIP 的全新安装，以及依赖缓存准备好后的离线安装检查。 [固定引擎来源与 SHA-256](focus-stack-integration.md)

下面的报告、图片和日志是验收机器 `artifacts/` 下的本地文件。每次运行生成自己的实验目录。环境探测输出保存在 `artifacts/verification/doctor.json`，也可通过 `doctor --json` 重现。 [复跑命令](../README.md)

## 自动化测试

`python -m pytest -ra` 执行 23 个测试，其中 22 个为非原生测试，1 个直接执行官方引擎。全部通过。测试覆盖数据校验、抽层、ROI 度量、参数合法性、模拟子进程的超时和取消、错误退出及回收、异常 PNG、失败重试、源哈希变化、Manifest 和 Qt 后台状态。原生项使用中文、空格和超过 260 字符的输入及输出路径，核验融合图、焦层图、日志和原图哈希。本机测试输出为 `artifacts/verification/pytest.txt`。

## 实际原生融合

| 场景 | 本机结果 | 记录 |
| --- | --- | --- |
| 384 × 256、12 层模拟单视野 | 八项预设全部成功，各输出融合图和焦层图 | `artifacts/experiment_20261125T034957Z_e516b331/report.html` |
| 两个 384 × 256 模拟视野 | 每视野用两个起始偏移抽取 5 层和 6 层，四项成功。恢复时复核后复用结果 | `artifacts/experiment_20261125T035000Z_5208fe80/report.html` |
| 4096 × 3000、50 层模拟单视野 | `FS_CPU`、4 线程、批次 4，约 46.72 s，采样峰值 RSS 2544.3 MiB，得到 4085 × 2991 PNG | `artifacts/experiment_20261125T035001Z_c6c933e8/report.html` |

额外的原生参数检查使用零基参考层 2，按目标层数抽取 `1, 4, 7, 9, 12`，关闭几何变换并保留输入尺寸，保存了 11 张中间图。报告为 `artifacts/verification/engine-parameters/experiment_20261125T035954Z_6333ec26/report.html`。

对同一官方引擎触发了真实超时和执行中取消。两种状态都保留诊断记录，报告位于 `artifacts/verification/lifecycle/` 的实验目录 `experiment_20261125T040225Z_79b3d8dd` 和 `experiment_20261125T040225Z_e9c9ab5a`。

GUI 在 Windows Qt 平台检查了空状态、模拟数据导入、图像切换、已生成的 PNG、双图 ROI 框选、指标、焦层图和 12 点清晰度曲线。`artifacts/verification/workbench.png` 展示同一模拟视野的两个原生预设结果。

压力记录用于核对本机处理资源。临时目录容量应覆盖本次任务的压缩输入、输出及所选中间图。用 `--threads` 与 `--batch-size` 可显式控制原生任务配置。官方 Windows 构建的实际日志报告 OpenCL 不可用。真实 M-P100 格式、121 视野整轮规模、花粉光学表现、实拍参考图与 OpenCL 构建的设备执行待后续数据和环境验证。 [真实数据实验方法](experiment-protocol.md)

花粉专项融合、深度学习和 YOLO 评价、生产软件集成、自适应 Z 采集、原生内核改动及 GPU 调优属于后续开发范围。
