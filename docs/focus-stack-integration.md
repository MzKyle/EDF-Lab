# 固定引擎与配置

本项目复用 Petteri Aimonen 的 focus-stack 1.5 官方 Windows x64 Release。上游管线使用 ECC 及光度配准、复小波融合、一致性过滤、可选降噪和可选深度估计。Python 负责输入、编排、校验和显示。 [固定提交 README](https://github.com/PetteriAimonen/focus-stack/blob/0cd289e224fe24aead5b146faeaa46950775f346/README.md) · [算法说明](https://github.com/PetteriAimonen/focus-stack/blob/0cd289e224fe24aead5b146faeaa46950775f346/docs/Algorithms.md) · [CLI 源码](https://github.com/PetteriAimonen/focus-stack/blob/0cd289e224fe24aead5b146faeaa46950775f346/src/main.cc)

| 项目 | 已核实值 |
| --- | --- |
| Release / commit | `1.5` · `0cd289e224fe24aead5b146faeaa46950775f346` |
| `--version` | `1.3-42-g0cd289e` · OpenCV `4.12.0` |
| Windows ZIP SHA-256 | `0104c863e1fc961cd87520c87d41c927d2968822d48fd37e76da00e1dfb6c7c9` |
| `focus-stack.exe` SHA-256 | `e256176d70bec54bb51a448b2adc452e7f49d21bc5042b40689a3257bbfdee9e` |
| 安装位置 | `third_party/focus-stack/1.5/focus-stack/focus-stack.exe` |

归档 SHA-256 与官方发布页面一致。安装脚本展开原始发布包，校验 EXE 和实际 `--help` 输出。本版本帮助按上游约定返回退出码 1。`doctor --json` 和 Manifest 保存实测帮助和 OpenCV 构建信息。 [官方 Release](https://github.com/PetteriAimonen/focus-stack/releases/tag/1.5) · [官方归档哈希](https://github.com/PetteriAimonen/focus-stack/releases/expanded_assets/1.5)

## 已核对的参数

| 用途 | 上游参数或行为 |
| --- | --- |
| 融合输出 | `--output=fused.png`，Lab 使用 PNG |
| 配准 | 默认邻帧，`--global-align` 直接对参考帧，`--no-align` 完全跳过 |
| 几何变换 | 默认启用，`--no-transform` 关闭，`--full-resolution-align` 解除默认 2048 px 配准分辨率限制 |
| 光度补偿 | 白平衡和曝光/对比度默认启用，分别由 `--no-whitebalance`、`--no-contrast` 关闭 |
| 一致性 | `--consistency=0`、`1`、`2`，默认 2 |
| 小波降噪 | `--denoise=1.0` 为默认，上游仅在数值为正时进入降噪分支，关闭预设使用 0。Lab 校验 CLI 原生 `float` 可解析的有限值 |
| 线程 | `--threads=N`，默认逻辑 CPU 数 + 1，Lab 接受正 `int` |
| 合并批次 | `--batchsize=N`，默认 8，Lab 接受正 `int` |
| 参考焦层 | `--reference=N`，未指定时用输入列表的 `count / 2`，Lab 要求 `0 ≤ N < count` |
| 裁剪 | 默认裁掉无效边界，`--align-keep-size` 保留配准前尺寸，`--nocrop` 保留扩展边界 |
| 焦层图 | `--depthmap=depth.png`，默认关闭 |
| 中间产物 | `--save-steps`，默认关闭，开启时保存至本次 `steps/` |
| 计算方式 | 默认允许上游尝试 OpenCL，`--no-opencl` 强制 CPU |
| 诊断 | Lab 加 `--verbose` 记录步骤、裁剪及计算方式的实际证据 |

`FS_Default` 保留上游算法参数。`FS_GeometryOnly` 保持几何配准并关闭两个光度补偿项。`FS_Consistency0/1/2` 精确传入对应档位。其他预设分别关闭配准、降噪或 OpenCL。若探测结果中缺失某个选项，则该参数组合会显示明确的不可用原因。`--reference` 在上游源码按抽层后列表的 `0` 到 `N-1` 处理。

## OpenCL 与自定义构建

本机官方 Windows 包的 `--opencv-version` 无 OpenCL 支持项，实际运行 `--verbose` 报告 `OpenCL not available`。`FS_CPU` 实际日志报告 `OpenCL disabled`。对一个在 OpenCV 构建信息中声明 OpenCL 的固定提交构建，`--acceleration opencl` 请求上游默认路径，并要求它报告设备。设备存在的日志条目记为 `opencl_device_reported`，单个任务到底在何处执行需要更细的上游剖析证据。

Windows 源码构建入口是上游固定提交的 `Makefile.windows` 和发布 workflow。它使用 MSVC x64、`nmake` 及 OpenCV，官方制品通过 vcpkg 构建。自定义构建要保留引擎运行所需的 DLL，并让 `--version` 包含固定提交缩写。可以使用拉取固定提交且不下载 tags 的 Git 工作副本，使 `git describe` 显示提交缩写。用 `POLLEN_EDF_ENGINE` 指向可执行文件后运行 `doctor --json`，Manifest 将记录该二进制自己的 SHA-256。 [Windows 构建文件](https://github.com/PetteriAimonen/focus-stack/blob/0cd289e224fe24aead5b146faeaa46950775f346/Makefile.windows) · [发布 workflow](https://github.com/PetteriAimonen/focus-stack/blob/0cd289e224fe24aead5b146faeaa46950775f346/.github/workflows/release.yml)
