# 第三方组件与许可证

实验室源码与原创文档采用仓库根目录的 MIT 许可证。运行时依赖由锁定的 Wheel 和官方引擎归档分别安装，保留各组件的版权和许可证。 [项目许可证](LICENSE)

## focus-stack 1.5 Windows x64

Petteri Aimonen 的上游程序采用 MIT，版权为 © 2019 Petteri Aimonen。安装包内保留原始 `LICENSE.txt`，仓库额外保存许可证副本。Windows 发布由 MSVC 和 vcpkg 构建。此处根据官方 ZIP 内容、PE 版本资源和实测 OpenCV 构建信息核对组件。 [上游许可证](https://github.com/PetteriAimonen/focus-stack/blob/0cd289e224fe24aead5b146faeaa46950775f346/LICENSE.md) · [本地 MIT 副本](third_party/focus-stack/licenses/focus-stack-MIT.txt)

| 制品或内置组件 | 已核对版本 | 许可证与原文 |
| --- | --- | --- |
| `opencv_*.dll` | 4.12.0 | Apache-2.0 · [OpenCV 许可证](third_party/focus-stack/licenses/OpenCV-4.12.0.txt) |
| `jpeg62.dll`、`turbojpeg.dll` | libjpeg-turbo 3.1.3 | IJG、BSD-3-Clause、Zlib，依所含代码区分 · [许可证](third_party/focus-stack/licenses/libjpeg-turbo-3.1.3-LICENSE.md) · [IJG 声明](third_party/focus-stack/licenses/libjpeg-turbo-3.1.3-README.ijg) |
| `libpng16.dll` | 1.6.53 | PNG Reference Library License v2，原声明也收录旧版许可证 · [libpng 声明](third_party/focus-stack/licenses/libpng-1.6.53.txt) |
| `tiff.dll` | libtiff 4.7.1 | libtiff 许可证及附带声明 · [libtiff 声明](third_party/focus-stack/licenses/libtiff-4.7.1.md) |
| `libwebp*.dll`、`libsharpyuv.dll` | libwebp 1.6.0 | BSD-3-Clause · [许可证](third_party/focus-stack/licenses/libwebp-1.6.0.txt) · [专利声明](third_party/focus-stack/licenses/libwebp-1.6.0-PATENTS.txt) |
| `zlib1.dll` | 1.3.1 | Zlib · [zlib 声明](third_party/focus-stack/licenses/zlib-1.3.1.txt) |
| `liblzma.dll` | XZ Utils 5.8.2 | 此 DLL 为 0BSD · [组件说明](third_party/focus-stack/licenses/liblzma-5.8.2.txt) · [许可证](third_party/focus-stack/licenses/liblzma-5.8.2-0BSD.txt) |
| `libprotobuf*.dll`、`libprotoc.dll` | Protobuf 29.5.0 | BSD-3-Clause · [Protobuf 声明](third_party/focus-stack/licenses/protobuf-v29.5.txt) |
| `abseil_dll.dll` | 导出符号标记 LTS 20250814 | Apache-2.0 · [Abseil 许可证](third_party/focus-stack/licenses/Abseil-20250814.txt) |
| OpenCV 内置 FlatBuffers | 25.9.23 | Apache-2.0 · [FlatBuffers 许可证](third_party/focus-stack/licenses/flatbuffers-v25.9.23.txt) |

该制品依赖 Windows 系统组件及 MSVC 运行库。Lab 也记录实际引擎构建与使用情况，随实验保留核对依据。第三方组件文本取自相应上游版本。 [发布构建流程](https://github.com/PetteriAimonen/focus-stack/blob/0cd289e224fe24aead5b146faeaa46950775f346/.github/workflows/release.yml)

## Python 环境

| 依赖 | 本机锁定版本 | 许可证 |
| --- | --- | --- |
| CPython | 3.11.16 | PSF 及随运行时附带的组件许可证 |
| PySide6 Essentials、Shiboken6 | 6.12.0 | PyPI Wheel 提供 LGPL-3.0-only、GPL-2.0-only 或 GPL-3.0-only 选项。本项目通过 LGPL 路径动态使用 Qt Core、Gui、Widgets，测试使用 Qt Test |
| `opencv-python-headless` | 4.14.0.94 | OpenCV 和 Python 打包 Apache-2.0。Wheel 内附 `LICENSE.txt` 及 `LICENSE-3RD-PARTY.txt` |
| NumPy | 2.4.6 | BSD-3-Clause 为主，Wheel 的 `LICENSE.txt` 及 `numpy.libs` 保留捆绑库声明 |
| Matplotlib | 3.11.2 | PSF 衍生的 Matplotlib 许可证，Wheel 保留附带组件条款 |
| psutil | 7.2.2 | BSD-3-Clause |
| PyYAML | 6.0.4 | MIT |
| pytest | 9.2.0 | MIT，开发测试依赖 |
| setuptools | 80.9.0 | MIT，构建依赖 |
| uv | 0.12.7 | MIT 或 Apache-2.0，独立的安装工具 |

传递依赖的固定版本与 Wheel SHA-256 见 `uv.lock`。这些依赖的原始许可随 Wheels 保存在环境中，主要包括 contourpy、cycler、kiwisolver 和 python-dateutil 的 BSD 条款，fonttools、iniconfig、pluggy、pyparsing 的 MIT 条款，Pillow 的 MIT-CMU，Pygments 的 BSD-2-Clause，packaging 的 Apache-2.0 或 BSD-2-Clause，以及 tzdata 的 Apache-2.0 和来源数据说明。 [依赖锁](uv.lock)

Qt 的原文包含随附第三方组件说明和相应源代码来源。项目保存 LGPLv3 及其引用的 GPLv3 文本。Python 虚拟环境与引擎安装目录为本地生成内容，项目源目录中的清单和许可证供查阅。 [PySide6 Essentials 许可证声明](https://pypi.org/project/PySide6-Essentials/6.12.0/#description) · [Qt 第三方组件](https://doc.qt.io/qt-6/licenses-used-in-qt.html) · [Qt 源代码归档](https://download.qt.io/official_releases/qt/) · [LGPLv3](third_party/qt/LGPL-3.0.txt) · [GPLv3](third_party/qt/GPL-3.0.txt)

Helicon Focus 是外部参考图的来源。Lab 记录用户提供的图片及已知设置，关联功能通过本地文件工作。
