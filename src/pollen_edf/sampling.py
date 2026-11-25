from __future__ import annotations

from dataclasses import dataclass

from .dataset import Frame, Stack


@dataclass(frozen=True)
class SampleSpec:
    mode: str = "all"
    step: int | None = None
    offset: int = 0
    start: int | None = None
    end: int | None = None
    layers: tuple[int, ...] = ()
    count: int | None = None
    keep_endpoints: bool = False

    def select(self, stack: Stack) -> tuple[tuple[Frame, ...], dict]:
        frames = stack.frames
        if self.mode == "all":
            positions = range(len(frames))
        elif self.mode == "stride":
            if (type(self.step) is not int or self.step < 1 or type(self.offset) is not int
                    or not 0 <= self.offset < self.step):
                raise ValueError("间隔抽样需要正整数步长及小于步长的非负偏移。")
            positions = range(self.offset, len(frames), self.step)
        elif self.mode == "range":
            if type(self.start) is not int or type(self.end) is not int or self.start > self.end:
                raise ValueError("范围抽样需要递增的整数焦层区间。")
            if self.start < frames[0].z or self.end > frames[-1].z:
                raise ValueError("焦层范围超出该视野的已采集编号。")
            positions = [i for i, frame in enumerate(frames) if self.start <= frame.z <= self.end]
        elif self.mode == "list":
            if any(type(number) is not int for number in self.layers):
                raise ValueError("指定焦层列表的元素必须是整数。")
            available = {frame.z for frame in frames}
            requested = set(self.layers)
            if not requested or len(requested) != len(self.layers) or requested - available:
                raise ValueError(f"指定焦层列表为空、包含重复值或不可用编号 {sorted(requested - available)}")
            positions = [i for i, frame in enumerate(frames) if frame.z in requested]
        elif self.mode == "uniform":
            if type(self.count) is not int or not 2 <= self.count <= len(frames):
                raise ValueError(f"均匀抽样目标应为 [2, {len(frames)}] 内的整数。")
            denominator = self.count - 1
            positions = [(part * (len(frames) - 1) + denominator // 2) // denominator for part in range(self.count)]
        else:
            raise ValueError(f"未知抽层策略 {self.mode}")
        indices = list(positions)
        if self.keep_endpoints:
            indices = sorted(set(indices) | {0, len(frames) - 1})
        if len(indices) < 2:
            raise ValueError("抽层至少应选中两帧。")
        selected = tuple(frames[index] for index in indices)
        z = [frame.z for frame in selected]
        z_um = [frame.z_um for frame in selected]
        covers = indices[0] == 0 and indices[-1] == len(frames) - 1
        adjacent = all(b == a + 1 for a, b in zip(indices, indices[1:]))
        if covers:
            variable = "all_layers" if adjacent else "spacing_with_full_coverage"
        else:
            variable = "coverage_at_original_spacing" if adjacent else "spacing_and_coverage"
        physical = all(value is not None for value in z_um)
        return selected, {"variable": variable, "layer_count": len(z), "available_layers": len(frames), "z": z,
                          "z_intervals": [b - a for a, b in zip(z, z[1:])], "z_range": [z[0], z[-1]],
                          "available_z_range": [frames[0].z, frames[-1].z], "z_um": z_um,
                          "z_um_intervals": [b - a for a, b in zip(z_um, z_um[1:])] if physical else None,
                          "z_um_range": [min(z_um), max(z_um)] if physical else None}
