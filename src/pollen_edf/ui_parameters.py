from __future__ import annotations

from dataclasses import asdict

from PySide6.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                               QLineEdit, QVBoxLayout, QWidget)

from .engine import EngineCapabilities, EngineOptions
from .experiments import POLLEN_PRESETS, PRESETS, trials_for
from .pollen_algorithm import PollenEDFOptions
from .sampling import SampleSpec


def combo(options: list[tuple[str, object]]) -> QComboBox:
    widget = QComboBox()
    for label, data in options:
        widget.addItem(label, data)
    return widget


class Parameters(QWidget):
    def __init__(self, timeout_s: float, parent=None):
        super().__init__(parent)
        self.capabilities: EngineCapabilities | None = None
        self.preset = combo([(key, key) for key in (*PRESETS, *POLLEN_PRESETS)])
        self.alignment = combo([("相邻帧", "default"), ("全局参考帧", "global"), ("关闭", "none")])
        self.crop = combo([("自动裁剪", "auto"), ("保留输入大小", "keep-size"), ("保留外推边界", "none")])
        self.acceleration = combo([("引擎默认", "auto"), ("强制 CPU", "cpu"), ("请求 OpenCL", "opencl")])
        self.consistency = combo([("引擎默认 (2)", None), ("0 · 关闭", 0), ("1 · 中", 1), ("2 · 高", 2)])
        self.whitebalance = QCheckBox("白平衡")
        self.contrast = QCheckBox("曝光与对比度")
        self.transform = QCheckBox("几何变换")
        self.full_resolution_align = QCheckBox("全分辨率配准")
        self.depth = QCheckBox("输出焦层图")
        self.intermediate = QCheckBox("保留中间结果")
        self.denoise = self._entry("默认 1.0")
        self.threads = self._entry("默认 CPU 数 + 1")
        self.batch_size = self._entry("默认 8")
        self.reference = self._entry("默认中间层，索引从 0 开始")
        self.timeout = QLineEdit(f"{timeout_s:g}")
        self.sample_mode = combo([("全部焦层", "all"), ("间隔抽取", "stride"), ("连续范围", "range"),
                                  ("指定编号", "list"), ("目标层数", "uniform")])
        self.sample_values = self._entry("")
        self.endpoints = QCheckBox("保留源焦栈首尾")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(9, 6, 9, 8)
        layout.setSpacing(10)
        algorithms, form = self._group("算法参数")
        for label, field in (("预设", self.preset), ("配准", self.alignment), ("裁剪", self.crop),
                             ("一致性过滤", self.consistency), ("小波降噪", self.denoise), ("参考层索引", self.reference)):
            form.addRow(label, field)
        form.addRow(self._pair(self.transform, self.full_resolution_align))
        form.addRow(self._pair(self.whitebalance, self.contrast))
        form.addRow(self._pair(self.depth, self.intermediate))
        layout.addWidget(algorithms)
        resources, form = self._group("运行资源")
        for label, field in (("计算模式", self.acceleration), ("线程数", self.threads),
                             ("合并批次", self.batch_size), ("超时 (s)", self.timeout)):
            form.addRow(label, field)
        layout.addWidget(resources)
        sampling, form = self._group("焦层选择")
        form.addRow("抽层策略", self.sample_mode)
        self.sample_label = QLabel()
        form.addRow(self.sample_label, self.sample_values)
        form.addRow(self.endpoints)
        layout.addWidget(sampling)
        layout.addStretch()

        self.preset.currentIndexChanged.connect(self._apply_preset)
        self.alignment.currentIndexChanged.connect(self._align_fields)
        self.transform.toggled.connect(self._align_fields)
        self.sample_mode.currentIndexChanged.connect(self._sample_fields)
        self._apply_preset()
        self._sample_fields()

    @staticmethod
    def _entry(placeholder: str) -> QLineEdit:
        widget = QLineEdit()
        widget.setPlaceholderText(placeholder)
        widget.setToolTip(placeholder)
        return widget

    @staticmethod
    def _group(title: str) -> tuple[QGroupBox, QFormLayout]:
        group = QGroupBox(title)
        form = QFormLayout(group)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setVerticalSpacing(6)
        return group, form

    @staticmethod
    def _pair(first: QWidget, second: QWidget) -> QWidget:
        parent = QWidget()
        layout = QHBoxLayout(parent)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(first)
        layout.addWidget(second)
        layout.addStretch()
        return parent

    def set_capabilities(self, capabilities: EngineCapabilities | None) -> None:
        self.capabilities = capabilities
        for index, key in enumerate(PRESETS):
            item = self.preset.model().item(index)
            try:
                if capabilities:
                    trials_for([key])[0].options.arguments(2, capabilities)
            except ValueError as error:
                item.setEnabled(False)
                item.setToolTip(str(error))
            else:
                item.setEnabled(True)
                item.setToolTip("")
        opencl = self.acceleration.model().item(2)
        opencl.setEnabled(bool(capabilities and capabilities.opencl_built))
        opencl.setToolTip("需要引擎 OpenCV 构建及可用设备。")
        if not opencl.isEnabled() and self.acceleration.currentData() == "opencl":
            self.acceleration.setCurrentIndex(0)

    def _apply_preset(self) -> None:
        name = self.preset.currentData()
        pollen = name in POLLEN_PRESETS
        base = EngineOptions(**PRESETS.get(name, {}))
        selected = PollenEDFOptions(**POLLEN_PRESETS[name]) if pollen else base
        for name in ("alignment", "crop", "acceleration", "consistency"):
            widget = getattr(self, name)
            widget.setCurrentIndex(widget.findData(getattr(base, name)))
        for name in ("transform", "full_resolution_align", "whitebalance", "contrast"):
            getattr(self, name).setChecked(getattr(base, name))
        for name in ("depth", "intermediate"):
            getattr(self, name).setChecked(getattr(selected, name))
        for name in ("denoise", "batch_size", "reference"):
            value = getattr(base, name)
            getattr(self, name).setText("" if value is None else f"{value:g}")
        self.threads.setText("" if selected.threads is None else str(selected.threads))
        for name in ("alignment", "crop", "acceleration", "consistency", "denoise", "batch_size", "reference"):
            getattr(self, name).setEnabled(not pollen)
        self._align_fields()

    def is_pollen(self) -> bool:
        return self.preset.currentData() in POLLEN_PRESETS

    def _align_fields(self) -> None:
        enabled = not self.is_pollen() and self.alignment.currentData() != "none"
        for widget in (self.whitebalance, self.contrast, self.transform):
            widget.setEnabled(enabled)
        self.full_resolution_align.setEnabled(enabled and self.transform.isChecked())
        item = self.crop.model().item(1)
        item.setEnabled(enabled)
        if not enabled and self.crop.currentData() == "keep-size":
            self.crop.setCurrentIndex(0)

    def _sample_fields(self) -> None:
        mode = self.sample_mode.currentData()
        label, example = {"all": ("", ""), "stride": ("步长 / 偏移", "3 / 0,1,2"),
                          "range": ("起止 Z", "10,30"), "list": ("Z 编号", "1,5,9,12"),
                          "uniform": ("目标层数", "50,25,17,13,10")}[mode]
        self.sample_values.clear()
        self.sample_label.setText(label)
        self.sample_values.setPlaceholderText(example)
        self.sample_label.setVisible(mode != "all")
        self.sample_values.setVisible(mode != "all")
        self.endpoints.setEnabled(mode in {"stride", "range", "list"})

    def samples(self) -> list[SampleSpec]:
        mode = self.sample_mode.currentData()
        if mode == "all":
            return [SampleSpec()]
        text = self.sample_values.text().strip().replace("，", ",")
        endpoints = self.endpoints.isChecked() and self.endpoints.isEnabled()
        try:
            if mode == "stride":
                step, separator, offsets = text.partition("/")
                starts = [int(value) for value in offsets.split(",")] if separator else [0]
                return [SampleSpec(mode, step=int(step), offset=start, keep_endpoints=endpoints)
                        for start in dict.fromkeys(starts)]
            values = [int(value) for value in text.split(",")]
            if mode == "range":
                start, end = values
                return [SampleSpec(mode, start=start, end=end, keep_endpoints=endpoints)]
            if mode == "list":
                return [SampleSpec(mode, layers=tuple(values), keep_endpoints=endpoints)]
            return [SampleSpec(mode, count=value) for value in dict.fromkeys(values)]
        except ValueError as error:
            raise ValueError("请按抽层输入框内的示例填写整数。") from error

    def selected_options(self) -> EngineOptions | PollenEDFOptions:
        if self.is_pollen():
            base = POLLEN_PRESETS[self.preset.currentData()]
            try:
                threads = int(self.threads.text().strip() or "8")
            except ValueError as error:
                raise ValueError("threads 参数格式无效。") from error
            return PollenEDFOptions(**{**base, "threads": threads, "depth": self.depth.isChecked(),
                                       "intermediate": self.intermediate.isChecked()})
        values = {name: getattr(self, name).currentData() for name in ("alignment", "crop", "acceleration", "consistency")}
        values.update({name: getattr(self, name).isChecked()
                       for name in ("transform", "full_resolution_align", "whitebalance", "contrast", "depth", "intermediate")})
        if values["alignment"] == "none":
            values.update(transform=True, whitebalance=True, contrast=True, full_resolution_align=False)
        elif not values["transform"]:
            values["full_resolution_align"] = False
        for name in ("denoise", "threads", "batch_size", "reference"):
            text = getattr(self, name).text().strip()
            try:
                values[name] = (float(text) if name == "denoise" else int(text)) if text else None
            except ValueError as error:
                raise ValueError(f"{name} 参数格式无效。") from error
        return EngineOptions(**values)

    def trials(self, names: list[str] | None = None):
        base_name = self.preset.currentData()
        base = asdict(trials_for([base_name])[0].options)
        current = asdict(self.selected_options())
        overrides = {name: value for name, value in current.items() if value != base[name]}
        return trials_for(names or [base_name], self.samples(), overrides)
