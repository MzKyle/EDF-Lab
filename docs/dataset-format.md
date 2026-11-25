# 数据导入与焦层顺序

## 目录

选择单视野目录可直接读取其中的图像。选择采集父目录时，以它的各个直接子目录为视野。

```text
Capture/
├── XY_00_00/
│   ├── Z_1.jpg
│   ├── Z_2.jpg
│   └── Z_10.jpg
└── XY_00_01/
    ├── Z_1.jpg
    └── Z_2.jpg
```

识别 `Z_12`、`z12`、`Z-2`（负数）及完全由整数构成的文件名。`capture_Z_3_rgb.jpg` 含一个可识别的 Z 标记。每个视野按整数 `1, 2, 10` 排序。根目录和子目录同时存有受支持图像、文件名存在多个 Z 标记或编号语义无法判断时，请选择单一目录或提供显式索引。日后可为 M-P100 的真实命名实现 `dataset.load_dataset(..., parser=...)` 目录解析器。

当前支持 JPEG、PNG、TIFF、BMP、WebP 的单帧读取。校验以一帧为单位核对文件签名、可解码性和同视野尺寸。编号重复和分辨率不一致视为错误。编号内部的缺层区间以警告记录，可用于主动稀疏采集。索引内提供预期起止编号时，校验也会报告两端缺层。

## JSON 索引

用 UTF-8 编码，图像路径可以绝对指定，也可相对索引文件所在目录。`z` 是唯一焦层编号。`z_um` 为可选的实际物理坐标，按微米记录。正向或反向的单调坐标均可，焦层顺序始终取 `z` 的整数顺序。

```json
{
  "name": "M-P100 采集归档",
  "metadata": {"instrument": "M-P100", "objective": "20x"},
  "views": {
    "XY_00_00": {"first_z": 1, "last_z": 2, "x_um": 0, "y_um": 0}
  },
  "frames": [
    {"xy": "XY_00_00", "z": 2, "z_um": 0.8, "path": "raw/IMG_B.jpg"},
    {"xy": "XY_00_00", "z": 1, "z_um": 0.0, "path": "raw/IMG_A.jpg", "metadata": {"exposure_ms": 12}}
  ]
}
```

`name`、`metadata`、`views`、`z_um`、`sha256` 可省略。无 `xy` 时使用 `XY_00_00`。提供 `sha256` 会触发校验，也可用 CLI 的 `--hash` 计算全部输入文件的 SHA-256。JSON 还可保留未识别的帧属性。JSON 中的 Windows 绝对路径可写成 `D:/采集/IMG_A.jpg`。

## CSV 或手动列表

CSV 的有效表头为 `xy,z,path,z_um,sha256`，需提供 `path` 和 `z`。字节以 UTF-8 或 UTF-8 BOM 读取。可加 `exposure_ms` 等列记录元数据。

```csv
xy,z,path,z_um,exposure_ms
XY_00_00,2,raw/IMG_B.jpg,0.8,12
XY_00_00,1,raw/IMG_A.jpg,0.0,12
```

手动列表可按文件名解析 Z，也可明确以列表次序生成从 1 开始的 Z 编号。GUI 提供拖动重排与确认选项。CLI 使用以下形式。

```powershell
.\scripts\run.ps1 inspect --files 'D:\采集\IMG_A.jpg' 'D:\采集\IMG_B.jpg' --explicit-order --hash
```

手动顺序生成的是逻辑编号。实际坐标和采集元数据适合通过索引传入。已导入数据的原图路径会记入 Manifest，恢复实验时使用该路径复核文件哈希。
