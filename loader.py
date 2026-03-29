from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

from schema import PipelineConfig


def load_config(config: PipelineConfig | dict[str, Any] | str | Path | None = None) -> PipelineConfig:
    if config is None:
        return PipelineConfig()
    if isinstance(config, PipelineConfig):
        return config
    if isinstance(config, dict):
        return PipelineConfig.from_dict(config)

    path = Path(config)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".toml":
        with path.open("rb") as f:
            raw = tomllib.load(f)
        return PipelineConfig.from_dict(raw)
    if suffix in {".json"}:
        raw = json.loads(path.read_text())
        return PipelineConfig.from_dict(raw)

    raise ValueError(f"Unsupported config format: {suffix}. Use .toml or .json")
