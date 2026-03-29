from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np
import pandas as pd

from amce_types import FeatureFrame, FeatureSpec

from formulas import compute_formula
from specs import get_feature_specs


def feature_registry_version(specs: Iterable[FeatureSpec] | None = None) -> str:
    specs = list(specs) if specs is not None else get_feature_specs()
    payload = "|".join(f"{s.name}:{s.formula_id}:{sorted(s.params.items())}" for s in specs)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def compute_feature_frame(df: pd.DataFrame, include_categories: list[str] | None = None) -> FeatureFrame:
    base = df.copy()
    specs = get_feature_specs()
    if include_categories:
        specs = [s for s in specs if s.category in include_categories]

    for spec in specs:
        base[spec.name] = compute_formula(base, spec.formula_id, spec.params)

    feature_cols = [s.name for s in specs]
    base = base.replace([np.inf, -np.inf], np.nan)
    return FeatureFrame(data=base, feature_columns=feature_cols, target_column="Target")


def validate_feature_frame(frame: FeatureFrame, specs: list[FeatureSpec] | None = None) -> dict[str, object]:
    if specs is None:
        specs = [s for s in get_feature_specs() if s.name in frame.feature_columns]
    missing = [s.name for s in specs if s.name not in frame.data.columns]
    if missing:
        raise ValueError(f"Missing computed features: {missing[:5]}")

    inf_counts = np.isinf(frame.data[frame.feature_columns].values).sum()
    max_warmup = max(s.warmup for s in specs)
    post_warm = frame.data.iloc[max_warmup:]
    nan_rate = float(post_warm[frame.feature_columns].isna().mean().mean())

    return {
        "feature_count": len(frame.feature_columns),
        "max_warmup": max_warmup,
        "inf_count": int(inf_counts),
        "post_warmup_nan_rate": nan_rate,
    }
