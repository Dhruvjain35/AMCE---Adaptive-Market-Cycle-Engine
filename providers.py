from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import pandas as pd
import yfinance as yf

from amce.config.schema import DataConfig

logger = logging.getLogger(__name__)


class DataProvider(Protocol):
    def load_prices(self, universe: list[str], start: str, end: str | None = None) -> pd.DataFrame:
        ...

    def load_macro(self, series: list[str], start: str, end: str | None = None) -> pd.DataFrame:
        ...

    def load_ohlcv(self, ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
        ...


@dataclass
class YFinanceDataProvider:
    auto_adjust: bool = True

    def _download(self, tickers: list[str], start: str, end: str | None) -> pd.DataFrame:
        kwargs = {
            "start": start,
            "progress": False,
            "auto_adjust": self.auto_adjust,
        }
        if end:
            kwargs["end"] = end
        raw = yf.download(tickers=tickers, **kwargs)

        if raw.empty:
            raise ValueError(f"No data returned for tickers={tickers}")

        if isinstance(raw.columns, pd.MultiIndex):
            if "Close" in raw.columns.get_level_values(0):
                close = raw["Close"]
            else:
                close = raw.xs(raw.columns.levels[0][0], level=0, axis=1)
        else:
            close = raw

        close = close.ffill().dropna(how="all")
        return close

    def load_prices(self, universe: list[str], start: str, end: str | None = None) -> pd.DataFrame:
        return self._download(universe, start, end)

    def load_macro(self, series: list[str], start: str, end: str | None = None) -> pd.DataFrame:
        return self._download(series, start, end)

    def load_ohlcv(self, ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
        kwargs = {"start": start, "progress": False, "auto_adjust": self.auto_adjust}
        if end:
            kwargs["end"] = end
        raw = yf.download(tickers=ticker, **kwargs)
        if raw.empty:
            return pd.DataFrame()
        if isinstance(raw.columns, pd.MultiIndex):
            raw = raw.xs(ticker, level=1, axis=1)
        return raw.ffill()


@dataclass
class InMemoryDataProvider:
    frame: pd.DataFrame

    def load_prices(self, universe: list[str], start: str, end: str | None = None) -> pd.DataFrame:
        cols = [c for c in universe if c in self.frame.columns]
        out = self.frame.loc[:, cols]
        return _slice_dates(out, start, end)

    def load_macro(self, series: list[str], start: str, end: str | None = None) -> pd.DataFrame:
        cols = [c for c in series if c in self.frame.columns]
        out = self.frame.loc[:, cols]
        return _slice_dates(out, start, end)

    def load_ohlcv(self, ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
        # In-memory fixtures can provide `Volume` (and optional OHLC columns) directly.
        candidates = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in self.frame.columns]
        out = self.frame.loc[:, candidates] if candidates else pd.DataFrame(index=self.frame.index)
        return _slice_dates(out, start, end)


def _slice_dates(df: pd.DataFrame, start: str, end: str | None) -> pd.DataFrame:
    out = df.copy()
    out = out.loc[out.index >= pd.Timestamp(start)]
    if end is not None:
        out = out.loc[out.index <= pd.Timestamp(end)]
    return out


def load_base_frame(provider: DataProvider, cfg: DataConfig) -> pd.DataFrame:
    price_cols = [cfg.risk_ticker, cfg.safe_ticker]
    macro_cols = [cfg.vix_ticker, cfg.yield_ticker]

    prices = provider.load_prices(price_cols, cfg.start_date, cfg.end_date)
    macro = provider.load_macro(macro_cols, cfg.start_date, cfg.end_date)

    frame = pd.concat([prices, macro], axis=1).ffill().dropna()
    rename_map = {
        cfg.risk_ticker: "Risk",
        cfg.safe_ticker: "Safe",
        cfg.vix_ticker: "VIX",
        cfg.yield_ticker: "Yield",
    }
    frame = frame.rename(columns=rename_map)

    required = ["Risk", "Safe", "VIX", "Yield"]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing required fields after load: {missing}")

    frame = frame[required].dropna()
    if cfg.include_risk_volume:
        try:
            ohlcv = provider.load_ohlcv(cfg.risk_ticker, cfg.start_date, cfg.end_date)
            if "Volume" in ohlcv.columns:
                frame["Volume"] = ohlcv["Volume"].reindex(frame.index).ffill()
        except Exception:
            pass

    if len(frame) < 252:
        raise ValueError(f"Insufficient data rows: {len(frame)}")

    logger.info("Loaded base frame rows=%s range=%s->%s", len(frame), frame.index.min(), frame.index.max())
    return frame
