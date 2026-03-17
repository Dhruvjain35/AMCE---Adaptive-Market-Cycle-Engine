from __future__ import annotations

import pandas as pd


def add_targets(df: pd.DataFrame, horizon_days: int = 1) -> pd.DataFrame:
    """Add forward-return labels for classification and diagnostics."""
    out = df.copy()
    out["Fwd_Ret"] = out["Risk"].shift(-horizon_days) / out["Risk"] - 1.0
    out["Target"] = (out["Fwd_Ret"] > 0).astype(int)
    return out
