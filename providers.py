from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import numpy as np
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

    @staticmethod
    def _extract_close_frame(raw: pd.DataFrame, expected_tickers: list[str]) -> pd.DataFrame:
        if raw.empty:
            return pd.DataFrame()

        if isinstance(raw.columns, pd.MultiIndex):
            level0 = raw.columns.get_level_values(0)
            if "Close" in level0:
                close = raw["Close"]
            elif "Adj Close" in level0:
                close = raw["Adj Close"]
            else:
                close = raw.xs(level0[0], level=0, axis=1)
        else:
            if "Close" in raw.columns:
                close = raw[["Close"]]
            elif "Adj Close" in raw.columns:
                close = raw[["Adj Close"]]
            else:
                close = raw.copy()

        if isinstance(close, pd.Series):
            close = close.to_frame()

        if isinstance(close.columns, pd.MultiIndex):
            close.columns = [str(c[-1]) for c in close.columns]

        close = close.copy()
        if close.shape[1] == 1 and len(expected_tickers) == 1:
            close.columns = [expected_tickers[0]]
        return close

    def _download(self, tickers: list[str], start: str, end: str | None) -> pd.DataFrame:
        ticker_list = list(tickers)
        kwargs = {
            "start": start,
            "progress": False,
            "auto_adjust": self.auto_adjust,
        }
        if end:
            kwargs["end"] = end
        raw = yf.download(tickers=ticker_list, **kwargs)
        close = self._extract_close_frame(raw, ticker_list)

        missing = [t for t in ticker_list if t not in close.columns]
        for ticker in missing:
            try:
                single_raw = yf.download(tickers=ticker, **kwargs)
                single_close = self._extract_close_frame(single_raw, [ticker])
                if ticker in single_close.columns:
                    close = pd.concat([close, single_close[[ticker]]], axis=1)
            except Exception as exc:
                logger.warning("Fallback download failed for ticker=%s: %s", ticker, exc)

        close = close.loc[:, ~close.columns.duplicated()]
        close = close.ffill().dropna(how="all")
        if close.empty:
            raise ValueError(f"No data returned for tickers={ticker_list}")
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
        candidates = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in self.frame.columns]
        out = self.frame.loc[:, candidates] if candidates else pd.DataFrame(index=self.frame.index)
        return _slice_dates(out, start, end)


def _slice_dates(df: pd.DataFrame, start: str, end: str | None) -> pd.DataFrame:
    out = df.copy()
    out = out.loc[out.index >= pd.Timestamp(start)]
    if end is not None:
        out = out.loc[out.index <= pd.Timestamp(end)]
    return out


def _normalise_universe(primary: str, extras: list[str] | None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in [primary, *(extras or [])]:
        ticker = str(raw).strip()
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        out.append(ticker)
    return out


def _softmax_rows(values: pd.DataFrame, temperature: float = 8.0) -> pd.DataFrame:
    cleaned = values.replace([np.inf, -np.inf], np.nan)
    row_max = cleaned.max(axis=1)
    shifted = cleaned.sub(row_max, axis=0)
    scaled = (shifted * temperature).clip(-50, 50)
    exp_scores = np.exp(scaled)
    exp_scores = exp_scores.where(cleaned.notna(), 0.0)
    denom = exp_scores.sum(axis=1).replace(0.0, np.nan)
    return exp_scores.div(denom, axis=0)


def _slug_ticker(ticker: str) -> str:
    slug = "".join(ch if ch.isalnum() else "_" for ch in str(ticker))
    slug = slug.strip("_")
    return slug or "ticker"


def _build_routed_composite(prices: pd.DataFrame, router: str, column_name: str) -> tuple[pd.Series, pd.DataFrame]:
    px = prices.ffill().dropna(how="all")
    if px.empty:
        return pd.Series(dtype=float, name=column_name), pd.DataFrame(index=prices.index)

    px = px.loc[:, ~px.columns.duplicated()]
    if px.shape[1] == 1:
        series = px.iloc[:, 0].rename(column_name)
        weights = pd.DataFrame(1.0, index=px.index, columns=px.columns)
        return series, weights

    rets = px.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    n_assets = px.shape[1]
    eq_weights = pd.DataFrame(1.0 / n_assets, index=px.index, columns=px.columns)

    mode = str(router).strip().lower()
    if mode == "equal":
        weights_now = eq_weights
    else:
        mom_21 = px.pct_change(21).clip(-1.0, 1.0)
        mom_63 = px.pct_change(63).clip(-1.0, 1.0)
        vol_21 = rets.rolling(21).std() * np.sqrt(252)
        draw_63 = px.divide(px.rolling(63).max()).subtract(1.0)

        # A lightweight scenario-probability score: favor persistent momentum and lower-vol names.
        scenario_score = 0.50 * mom_63 + 0.30 * mom_21 + 0.20 * draw_63 - 0.15 * vol_21
        weights_now = _softmax_rows(scenario_score).fillna(eq_weights)

    # One-day lag avoids look-ahead bias.
    weights = weights_now.shift(1).fillna(eq_weights)
    composite_ret = (weights * rets).sum(axis=1)

    anchor = float(px.iloc[0].mean())
    if not np.isfinite(anchor) or anchor <= 0:
        anchor = 1.0
    composite = (1.0 + composite_ret).cumprod() * anchor
    composite.name = column_name
    return composite, weights


def load_base_frame(provider: DataProvider, cfg: DataConfig) -> pd.DataFrame:
    risk_universe = _normalise_universe(cfg.risk_ticker, cfg.risk_tickers)
    safe_universe = _normalise_universe(cfg.safe_ticker, cfg.safe_tickers)
    if not risk_universe:
        raise ValueError("No risk tickers configured.")
    if not safe_universe:
        raise ValueError("No safe tickers configured.")

    benchmark = str(cfg.benchmark_ticker).strip()
    price_cols = list(dict.fromkeys([*risk_universe, *safe_universe, benchmark] if benchmark else [*risk_universe, *safe_universe]))
    macro_cols = [cfg.vix_ticker, cfg.yield_ticker]

    try:
        prices = provider.load_prices(price_cols, cfg.start_date, cfg.end_date)
    except Exception:
        fallback_cols = list(dict.fromkeys([*risk_universe, *safe_universe]))
        prices = provider.load_prices(fallback_cols, cfg.start_date, cfg.end_date)
    macro = provider.load_macro(macro_cols, cfg.start_date, cfg.end_date)

    risk_prices = prices.loc[:, [c for c in risk_universe if c in prices.columns]]
    safe_prices = prices.loc[:, [c for c in safe_universe if c in prices.columns]]
    if risk_prices.empty:
        raise ValueError(f"Missing required fields after load: {risk_universe}")
    if safe_prices.empty:
        raise ValueError(f"Missing required fields after load: {safe_universe}")

    risk_series, risk_weights = _build_routed_composite(risk_prices, cfg.basket_router, "Risk")
    safe_series, safe_weights = _build_routed_composite(safe_prices, cfg.basket_router, "Safe")

    base = pd.DataFrame(index=prices.index)
    base["Risk"] = risk_series.reindex(base.index).ffill()
    base["Safe"] = safe_series.reindex(base.index).ffill()
    if benchmark and benchmark in prices.columns:
        base["SPX"] = prices[benchmark]

    if risk_weights.shape[1] > 1:
        for ticker in risk_weights.columns:
            base[f"RiskW_{_slug_ticker(ticker)}"] = risk_weights[ticker].reindex(base.index)
    if safe_weights.shape[1] > 1:
        for ticker in safe_weights.columns:
            base[f"SafeW_{_slug_ticker(ticker)}"] = safe_weights[ticker].reindex(base.index)

    frame = pd.concat([base, macro], axis=1).ffill()
    rename_map = {
        cfg.vix_ticker: "VIX",
        cfg.yield_ticker: "Yield",
    }
    frame = frame.rename(columns=rename_map)

    required = ["Risk", "Safe", "VIX", "Yield"]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing required fields after load: {missing}")

    frame = frame.dropna(subset=required)
    if cfg.include_risk_volume:
        volume_parts: list[pd.Series] = []
        for ticker in risk_universe:
            try:
                ohlcv = provider.load_ohlcv(ticker, cfg.start_date, cfg.end_date)
                if "Volume" in ohlcv.columns:
                    volume_parts.append(ohlcv["Volume"].rename(ticker))
            except Exception:
                continue
        if volume_parts:
            volume_df = pd.concat(volume_parts, axis=1).reindex(frame.index).ffill()
            frame["Volume"] = volume_df.sum(axis=1, min_count=1).ffill()

    if len(frame) < 252:
        raise ValueError(f"Insufficient data rows: {len(frame)}")

    logger.info(
        "Loaded base frame rows=%s range=%s->%s risk=%s safe=%s router=%s benchmark=%s",
        len(frame),
        frame.index.min(),
        frame.index.max(),
        risk_universe,
        safe_universe,
        cfg.basket_router,
        benchmark,
    )
    return frame
