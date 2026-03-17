from __future__ import annotations

from amce.config.schema import ValidationConfig


def evaluate_governance(
    strategy_metrics: dict[str, float],
    benchmark_metrics: dict[str, float],
    p_value: float,
    tercile_metrics: list[dict[str, float]],
    institutional_metrics: dict[str, float] | None,
    cfg: ValidationConfig,
    peer_head_to_head: dict[str, object] | None = None,
) -> dict[str, object]:
    sharpe_vals = [m.get("sharpe", 0.0) for m in tercile_metrics] if tercile_metrics else [0.0]
    sharpe_dispersion = max(sharpe_vals) - min(sharpe_vals) if sharpe_vals else 0.0

    institutional_gate = True
    institutional_uplift: dict[str, float] = {}
    if cfg.require_institutional_outperformance and institutional_metrics:
        sharpe_uplift = strategy_metrics.get("sharpe", 0.0) - institutional_metrics.get("sharpe", 0.0)
        sortino_uplift = strategy_metrics.get("sortino", 0.0) - institutional_metrics.get("sortino", 0.0)
        drawdown_uplift = strategy_metrics.get("max_drawdown", 0.0) - institutional_metrics.get("max_drawdown", 0.0)
        institutional_uplift = {
            "sharpe_uplift": float(sharpe_uplift),
            "sortino_uplift": float(sortino_uplift),
            "drawdown_uplift": float(drawdown_uplift),
        }
        institutional_gate = (
            sharpe_uplift >= cfg.min_institutional_sharpe_uplift
            and sortino_uplift >= cfg.min_institutional_sortino_uplift
            and drawdown_uplift >= cfg.min_institutional_drawdown_uplift
        )

    checks = {
        "max_drawdown_gate": strategy_metrics.get("max_drawdown", 0.0) >= cfg.max_oos_drawdown,
        "significance_gate": p_value < cfg.significance_level,
        "stability_gate": sharpe_dispersion <= cfg.max_tercile_sharpe_dispersion,
        "benchmark_risk_adjusted_gate": strategy_metrics.get("sharpe", 0.0) > benchmark_metrics.get("sharpe", 0.0),
        "institutional_uplift_gate": institutional_gate,
    }

    peer_gate = True
    failing_models: list[str] = []
    if cfg.peer_suite_enabled and cfg.require_beat_all_peers:
        if peer_head_to_head is None:
            peer_gate = False
        else:
            peer_gate = bool(peer_head_to_head.get("passed_all_peers", False))
            failing_models = list(peer_head_to_head.get("failing_models", []))
    checks["peer_league_gate"] = peer_gate

    return {
        "checks": checks,
        "passed": all(checks.values()),
        "p_value": p_value,
        "tercile_sharpe_dispersion": sharpe_dispersion,
        "drawdown_limit": cfg.max_oos_drawdown,
        "significance_level": cfg.significance_level,
        "institutional_uplift": institutional_uplift,
        "institutional_thresholds": {
            "min_sharpe_uplift": cfg.min_institutional_sharpe_uplift,
            "min_sortino_uplift": cfg.min_institutional_sortino_uplift,
            "min_drawdown_uplift": cfg.min_institutional_drawdown_uplift,
        },
        "peer_suite": peer_head_to_head or {},
        "peer_failing_models": failing_models,
    }
