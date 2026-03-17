from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    category: str
    inputs: tuple[str, ...]
    params: dict[str, Any]
    warmup: int
    formula_id: str
    leakage_safe: bool = True


@dataclass
class FeatureFrame:
    data: pd.DataFrame
    feature_columns: list[str]
    target_column: str


@dataclass
class SignalFrame:
    data: pd.DataFrame
    probability_column: str
    signal_column: str


@dataclass
class BacktestResult:
    frame: pd.DataFrame
    metrics: dict[str, float]
    benchmark_metrics: dict[str, float]


@dataclass
class ValidationReport:
    summary: dict[str, Any]
    fold_metrics: list[dict[str, Any]]
    governance: dict[str, Any]
    regime_metrics: dict[str, dict[str, float]]
    sensitivity: dict[str, Any]
    artifacts_dir: Path | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def _sanitize(value: Any) -> Any:
        if isinstance(value, pd.DataFrame):
            return {"type": "DataFrame", "rows": int(len(value)), "columns": list(value.columns)}
        if isinstance(value, pd.Series):
            return {"type": "Series", "rows": int(len(value)), "name": value.name}
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {str(k): ValidationReport._sanitize(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [ValidationReport._sanitize(v) for v in value]
        if isinstance(value, Path):
            return str(value)
        return value

    def to_dict(self) -> dict[str, Any]:
        extras: dict[str, Any] = {key: self._sanitize(value) for key, value in self.extras.items()}

        return {
            "summary": self._sanitize(self.summary),
            "fold_metrics": self._sanitize(self.fold_metrics),
            "governance": self._sanitize(self.governance),
            "regime_metrics": self._sanitize(self.regime_metrics),
            "sensitivity": self._sanitize(self.sensitivity),
            "artifacts_dir": str(self.artifacts_dir) if self.artifacts_dir else None,
            "extras": extras,
        }
