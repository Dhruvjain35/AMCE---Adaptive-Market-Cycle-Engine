from __future__ import annotations

from amce.types import FeatureSpec


def _spec(name: str, category: str, formula_id: str, warmup: int, **params: object) -> FeatureSpec:
    inputs = tuple(params.get("inputs", [params.get("source", "Risk")]))
    return FeatureSpec(
        name=name,
        category=category,
        inputs=tuple(str(x) for x in inputs),
        params={k: v for k, v in params.items() if k != "inputs"},
        warmup=warmup,
        formula_id=formula_id,
        leakage_safe=True,
    )


def get_feature_specs() -> list[FeatureSpec]:
    specs: list[FeatureSpec] = []

    # Trend (10)
    specs += [
        _spec("MA_10_Dist", "trend", "ma_dist", 10, source="Risk", window=10),
        _spec("MA_20_Dist", "trend", "ma_dist", 20, source="Risk", window=20),
        _spec("MA_50_Dist", "trend", "ma_dist", 50, source="Risk", window=50),
        _spec("MA_100_Dist", "trend", "ma_dist", 100, source="Risk", window=100),
        _spec("MA_200_Dist", "trend", "ma_dist", 200, source="Risk", window=200),
        _spec("EMA_21_Dist", "trend", "ema_dist", 21, source="Risk", window=21),
        _spec("EMA_63_Dist", "trend", "ema_dist", 63, source="Risk", window=63),
        _spec("Trend_Slope_20", "trend", "rolling_slope", 20, source="Risk", window=20),
        _spec("Trend_Slope_63", "trend", "rolling_slope", 63, source="Risk", window=63),
        _spec("ADX_14_Proxy", "trend", "adx_proxy", 28, source="Risk", window=14),
    ]

    # Momentum (11)
    specs += [
        _spec("Mom_5D", "momentum", "pct_change", 5, source="Risk", window=5),
        _spec("Mom_21D", "momentum", "pct_change", 21, source="Risk", window=21),
        _spec("Mom_63D", "momentum", "pct_change", 63, source="Risk", window=63),
        _spec("Mom_126D", "momentum", "pct_change", 126, source="Risk", window=126),
        _spec("Mom_252D", "momentum", "pct_change", 252, source="Risk", window=252),
        _spec("Residual_Mom_63", "momentum", "residual_mom", 63, source="Risk", window=63),
        _spec("Mom_Accel_21_63", "momentum", "mom_accel", 63, source="Risk", short=21, long=63),
        _spec("Reversal_5_21", "momentum", "reversal", 21, source="Risk", short=5, long=21),
        _spec("RSI_14", "momentum", "rsi", 14, source="Risk", window=14),
        _spec("RSI_28", "momentum", "rsi", 28, source="Risk", window=28),
        _spec("Stoch_K_14", "momentum", "stoch_k", 14, source="Risk", window=14),
    ]

    # Volatility / Risk (10)
    specs += [
        _spec("Vol_10", "volatility_risk", "rolling_vol", 11, source="Risk", window=10),
        _spec("Vol_21", "volatility_risk", "rolling_vol", 22, source="Risk", window=21),
        _spec("Vol_63", "volatility_risk", "rolling_vol", 64, source="Risk", window=63),
        _spec("Downside_Vol_21", "volatility_risk", "downside_vol", 22, source="Risk", window=21),
        _spec("Upside_Vol_21", "volatility_risk", "upside_vol", 22, source="Risk", window=21),
        _spec("UpDown_Vol_Ratio_21", "volatility_risk", "updown_vol_ratio", 22, source="Risk", window=21),
        _spec("ATR_14_Proxy", "volatility_risk", "atr_proxy", 15, source="Risk", window=14),
        _spec("Vol_of_Vol_21", "volatility_risk", "vol_of_vol", 43, source="Risk", window=21),
        _spec("Risk_Safe_Vol_Ratio_21", "volatility_risk", "risk_safe_vol_ratio", 22, source="Risk", source2="Safe", window=21),
        _spec("Drawdown_63", "volatility_risk", "drawdown", 63, source="Risk", window=63),
    ]

    # Cross Asset / Macro (10)
    specs += [
        _spec("Yield_Level", "cross_asset_macro", "level", 1, source="Yield"),
        _spec("Yield_Change_5D", "cross_asset_macro", "diff", 6, source="Yield", window=5),
        _spec("Yield_Change_21D", "cross_asset_macro", "diff", 22, source="Yield", window=21),
        _spec("Yield_Change_63D", "cross_asset_macro", "diff", 64, source="Yield", window=63),
        _spec("Yield_Trend_63", "cross_asset_macro", "trend_deviation", 63, source="Yield", window=63),
        _spec("Yield_ZScore_63", "cross_asset_macro", "zscore", 63, source="Yield", window=63),
        _spec("VIX_Level", "cross_asset_macro", "level", 1, source="VIX"),
        _spec("VIX_Change_5D", "cross_asset_macro", "pct_change", 5, source="VIX", window=5),
        _spec("VIX_Change_21D", "cross_asset_macro", "pct_change", 21, source="VIX", window=21),
        _spec("VIX_Term_Proxy_21", "cross_asset_macro", "vix_term_proxy", 21, source="VIX", window=21),
    ]

    # Breadth / Participation (8)
    specs += [
        _spec("Risk_vs_Safe_RS_21", "breadth_participation", "relative_strength", 21, source="Risk", source2="Safe", window=21),
        _spec("Risk_vs_Safe_RS_63", "breadth_participation", "relative_strength", 63, source="Risk", source2="Safe", window=63),
        _spec("Risk_vs_Safe_RS_126", "breadth_participation", "relative_strength", 126, source="Risk", source2="Safe", window=126),
        _spec("Positive_Days_21", "breadth_participation", "positive_day_ratio", 21, source="Risk", window=21),
        _spec("Positive_Days_63", "breadth_participation", "positive_day_ratio", 63, source="Risk", window=63),
        _spec("Dispersion_Proxy_21", "breadth_participation", "dispersion_proxy", 21, source="Risk", source2="Safe", window=21),
        _spec("Risk_Safe_Corr_63", "breadth_participation", "rolling_corr", 63, source="Risk", source2="Safe", window=63),
        _spec("Return_Autocorr_21", "breadth_participation", "rolling_autocorr", 21, source="Risk", window=21, lag=1),
    ]

    # Carry / Defensive (8)
    specs += [
        _spec("Carry_Proxy_21", "carry_defensive", "carry_proxy", 21, source="Yield", source2="Safe", window=21),
        _spec("Cash_vs_Safe_Adv_21", "carry_defensive", "cash_vs_safe_adv", 21, source="Yield", source2="Safe", window=21),
        _spec("Cash_vs_Safe_Adv_63", "carry_defensive", "cash_vs_safe_adv", 63, source="Yield", source2="Safe", window=63),
        _spec("Defensive_Spread_21", "carry_defensive", "defensive_spread", 21, source="Risk", source2="Safe", window=21),
        _spec("Defensive_Spread_63", "carry_defensive", "defensive_spread", 63, source="Risk", source2="Safe", window=63),
        _spec("Risk_Off_Pressure", "carry_defensive", "risk_off_pressure", 63, source="Risk", source2="VIX", window=63),
        _spec("Yield_Vol_Interaction_21", "carry_defensive", "yield_vol_interaction", 21, source="Yield", source2="Risk", window=21),
        _spec("Flight_to_Quality_21", "carry_defensive", "flight_to_quality", 21, source="Safe", source2="Risk", window=21),
    ]

    # Extra quality controls / tails (7)
    specs += [
        _spec("Donchian_20_Pos", "trend", "donchian_position", 20, source="Risk", window=20),
        _spec("Donchian_63_Pos", "trend", "donchian_position", 63, source="Risk", window=63),
        _spec("Stoch_D_14", "momentum", "stoch_d", 16, source="Risk", window=14),
        _spec("VaR_5pct_63", "volatility_risk", "rolling_quantile", 64, source="Risk", window=63, quantile=0.05),
        _spec("ExpectedShortfall_63", "volatility_risk", "expected_shortfall", 64, source="Risk", window=63, quantile=0.05),
        _spec("Skew_63", "volatility_risk", "rolling_skew", 64, source="Risk", window=63),
        _spec("Kurtosis_63", "volatility_risk", "rolling_kurt", 64, source="Risk", window=63),
    ]

    return specs
