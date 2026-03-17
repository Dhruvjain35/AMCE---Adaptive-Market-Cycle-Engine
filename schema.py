from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class DataConfig:
    risk_ticker: str = "QQQ"
    safe_ticker: str = "IEF"
    vix_ticker: str = "^VIX"
    yield_ticker: str = "^TNX"
    start_date: str = "2003-01-01"
    end_date: str | None = None
    auto_adjust: bool = True
    include_risk_volume: bool = True


@dataclass
class FeatureConfig:
    include_categories: list[str] = field(
        default_factory=lambda: [
            "trend",
            "momentum",
            "volatility_risk",
            "cross_asset_macro",
            "breadth_participation",
            "carry_defensive",
        ]
    )
    min_history_days: int = 252 * 3
    target_horizon_days: int = 1
    use_expanded_alpha_library: bool = True
    enable_lagged_context: bool = True
    lag_days: list[int] = field(default_factory=lambda: [1, 2, 5])
    enable_rolling_context: bool = True
    rolling_windows: list[int] = field(default_factory=lambda: [5, 21, 63])
    enable_feature_selection: bool = True
    feature_selection_top_k: int = 90
    min_feature_non_null_ratio: float = 0.70


@dataclass
class ModelConfig:
    random_state: int = 42
    logistic_c: float = 0.5
    logistic_l1_c: float = 0.20
    ridge_alpha: float = 1.0
    rf_estimators: int = 300
    rf_max_depth: int = 5
    rf_min_samples_leaf: int = 20
    gb_estimators: int = 200
    gb_max_depth: int = 3
    gb_learning_rate: float = 0.03
    use_lightgbm_if_available: bool = True
    use_xgboost_if_available: bool = True
    threshold_grid_low: float = 0.35
    threshold_grid_high: float = 0.65
    threshold_grid_step: float = 0.01
    enable_regime_expert: bool = True
    regime_expert_weight: float = 0.65
    min_stress_train_samples: int = 120
    uncertainty_penalty: float = 0.15


@dataclass
class RiskConfig:
    threshold: float | None = None
    threshold_band: float = 0.06
    target_annual_vol: float = 0.12
    max_exposure: float = 1.0
    max_daily_turnover: float = 0.30
    drawdown_de_risk_at: float = -0.12
    drawdown_de_risk_factor: float = 0.5
    stop_loss_sigma: float = 2.75
    tail_var_window: int = 63
    tail_var_floor: float = -0.03
    tail_risk_min_scale: float = 0.30


@dataclass
class BacktestConfig:
    transaction_cost_bps: float = 3.0
    slippage_bps: float = 5.0
    execution_lag_days: int = 1
    rebalance_frequency: str = "daily"  # daily|weekly|monthly
    benchmark_apply_costs: bool = False
    max_adv_participation: float = 0.15
    liquidity_lookback_days: int = 21
    allow_partial_fills: bool = True


@dataclass
class ValidationConfig:
    n_splits: int = 5
    embargo_days: int = 21
    min_train_days: int = 252 * 3
    rolling_window_days: int = 252
    permutation_trials: int = 1000
    significance_level: float = 0.05
    max_oos_drawdown: float = -0.25
    max_tercile_sharpe_dispersion: float = 0.90
    institutional_baseline_enabled: bool = True
    institutional_baseline_threshold: float = 0.50
    require_institutional_outperformance: bool = True
    min_institutional_sharpe_uplift: float = 0.35
    min_institutional_sortino_uplift: float = 0.35
    min_institutional_drawdown_uplift: float = 0.00
    peer_suite_enabled: bool = True
    peer_model_names: list[str] = field(
        default_factory=lambda: [
            "doubleensemble",
            "lightgbm",
            "mlp",
            "tft",
            "xgboost",
            "catboost",
            "tra",
            "linear",
            "gats",
            "alstm",
        ]
    )
    peer_probability_threshold: float = 0.50
    require_beat_all_peers: bool = True
    min_peer_sharpe_uplift_macro: float = 0.20
    min_peer_sortino_uplift_macro: float = 0.20
    min_peer_drawdown_uplift_macro: float = -0.015
    min_peer_sharpe_uplift_equity: float = 0.10
    min_peer_sortino_uplift_equity: float = 0.10
    min_peer_drawdown_uplift_equity: float = -0.015
    superiority_bootstrap_trials: int = 300
    superiority_confidence: float = 0.95
    dual_track_enabled: bool = True
    equity_track_rebalance_frequency: str = "weekly"
    equity_track_extra_cost_bps: float = 5.0
    regime_method: str = "gmm"  # gmm|kmeans|hmm|volatility
    regime_states: int = 4


@dataclass
class ReportingConfig:
    output_dir: str = "outputs/runs"
    persist_artifacts: bool = True


@dataclass
class PipelineConfig:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PipelineConfig":
        return cls(
            data=DataConfig(**raw.get("data", {})),
            features=FeatureConfig(**raw.get("features", {})),
            model=ModelConfig(**raw.get("model", {})),
            risk=RiskConfig(**raw.get("risk", {})),
            backtest=BacktestConfig(**raw.get("backtest", {})),
            validation=ValidationConfig(**raw.get("validation", {})),
            reporting=ReportingConfig(**raw.get("reporting", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
