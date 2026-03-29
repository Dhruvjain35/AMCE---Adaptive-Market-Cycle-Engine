"""Legacy compatibility wrapper for data loading."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Tuple

import pandas as pd

from schema import DataConfig
from providers import YFinanceDataProvider, load_base_frame


@dataclass
class DataLoader:
    start_date: str = "1993-01-01"
    end_date: str | None = None

    def __post_init__(self) -> None:
        self.provider = YFinanceDataProvider(auto_adjust=True)

    def load_trading_data(self, risk_ticker: str, safe_ticker: str) -> pd.DataFrame:
        cfg = DataConfig(
            risk_ticker=risk_ticker,
            safe_ticker=safe_ticker,
            start_date=self.start_date,
            end_date=self.end_date,
        )
        return load_base_frame(self.provider, cfg)

    def get_date_range(self, data: pd.DataFrame) -> Tuple[datetime, datetime]:
        return data.index[0], data.index[-1]

    def split_data(self, data: pd.DataFrame, train_pct: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if not 0.0 < train_pct < 1.0:
            raise ValueError(f"train_pct must be between 0 and 1, got {train_pct}")
        split_idx = int(len(data) * train_pct)
        return data.iloc[:split_idx], data.iloc[split_idx:]
