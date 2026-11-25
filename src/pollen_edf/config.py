from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import yaml


RESOURCES = Path(__file__).with_name("resources")
ENGINE_PIN = json.loads((RESOURCES / "engine.json").read_text(encoding="utf-8"))
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if not (PROJECT_ROOT / "pyproject.toml").is_file():
    PROJECT_ROOT = Path.cwd()


@dataclass(frozen=True)
class Settings:
    engine: Path
    artifacts_dir: Path
    timeout_s: float
    preview_cache_mib: int


def load_settings(path: str | Path | None = None) -> Settings:
    defaults = yaml.safe_load((RESOURCES / "default.yaml").read_text(encoding="utf-8"))
    base = PROJECT_ROOT
    if path is not None:
        path = Path(path).resolve()
        override = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
        if override is None:
            override = {}
        if not isinstance(override, dict):
            raise ValueError("配置文件必须是 YAML 映射。")
        unknown = override.keys() - (defaults.keys() | {"engine"})
        if unknown:
            raise ValueError(f"未知配置项 {sorted(map(str, unknown))}")
        base = path.parent
        defaults.update(override)
    engine = Path(os.environ.get("POLLEN_EDF_ENGINE") or defaults.get("engine") or
                  PROJECT_ROOT / ENGINE_PIN["installed_path"])
    timeout = defaults["timeout_s"]
    cache = defaults["preview_cache_mib"]
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout_s 必须是正的有限数。")
    if type(cache) is not int or cache < 8:
        raise ValueError("preview_cache_mib 必须是至少为 8 的整数。")
    return Settings(base / engine, base / Path(defaults["artifacts_dir"]), float(timeout), cache)
