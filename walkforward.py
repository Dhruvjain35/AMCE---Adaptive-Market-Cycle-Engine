from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from metrics import compute_metrics, compute_tercile_metrics, rolling_sharpe, run_backtest
from peers import build_peer_model, default_peer_model_names, get_top_peer_specs
from schema import BacktestConfig, PipelineConfig
from selection import select_features
from ensemble import EnsembleStack, learn_threshold
from institutional_baseline import QlibStyleBaseline
from construction import probabilities_to_exposure
from detection import fit_regime_classifier
from amce_types import FeatureFrame, ValidationReport

from governance import evaluate_governance
from stats import bootstrap_alpha_ci, bootstrap_superiority_probability, permutation_test_sharpe


@dataclass
class FoldSlice:
    train_start: int
    train_end: int
    test_start: int
    test_end: int


def _make_folds(n: int, min_train: int, n_splits: int, embargo_days: int) -> list[FoldSlice]:
    if n <= min_train:
        return []

    test_size = max(60, (n - min_train) // n_splits)
    folds: list[FoldSlice] = []

    for i in range(n_splits):
        test_start = min_train + i * test_size
        test_end = n if i == n_splits - 1 else min(n, test_start + test_size)
        train_end = max(min_train, test_start - embargo_days)

        if test_start >= n or test_end - test_start < 30:
            continue
        folds.append(FoldSlice(0, train_end, test_start, test_end))

    return folds


def _prepare_fold_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    cfg: PipelineConfig,
    fold_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, object], StandardScaler]:
    tr = train[feature_cols].replace([np.inf, -np.inf], np.nan).copy()
    te = test[feature_cols].replace([np.inf, -np.inf], np.nan).copy()

    non_null = tr.notna().mean()
    valid_cols = [c for c in feature_cols if non_null.get(c, 0.0) >= cfg.features.min_feature_non_null_ratio]
    if len(valid_cols) < 10:
        valid_cols = non_null.sort_values(ascending=False).head(min(20, len(non_null))).index.tolist()
    if not valid_cols:
        valid_cols = feature_cols[: min(20, len(feature_cols))]

    tr = tr[valid_cols]
    te = te[valid_cols]

    med = tr.median().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    tr = tr.fillna(med)
    te = te.fillna(med)

    std = tr.std(ddof=0).replace([np.inf, -np.inf], np.nan)
    usable = [c for c in valid_cols if float(std.get(c, 0.0)) > 1e-12]
    if not usable:
        usable = valid_cols
    if not usable:
        raise ValueError("No usable features available for fold after filtering.")

    tr = tr[usable]
    te = te[usable]
    selected = usable
    selection_meta: dict[str, object] = {
        "candidate_features": len(feature_cols),
        "non_null_features": len(valid_cols),
        "usable_features": len(usable),
    }

    if cfg.features.enable_feature_selection and len(usable) > cfg.features.feature_selection_top_k:
        fs = select_features(
            tr,
            train["Target"],
            top_k=cfg.features.feature_selection_top_k,
            random_state=fold_seed,
        )
        selected = [c for c in fs.selected_list if c in tr.columns]
        tr = tr[selected]
        te = te[selected]
        selection_meta["selected_features"] = len(selected)
        selection_meta["top_features"] = selected[:12]
    else:
        selection_meta["selected_features"] = len(selected)
        selection_meta["top_features"] = selected[:12]

    scaler = StandardScaler()
    x_train_df = pd.DataFrame(scaler.fit_transform(tr), index=tr.index, columns=selected)
    x_test_df = pd.DataFrame(scaler.transform(te), index=te.index, columns=selected)
    return x_train_df, x_test_df, selected, selection_meta, scaler


def _safe_returns(df: pd.DataFrame) -> pd.Series:
    cash = (df["Yield"] / 100) / 252
    trend = df.get("Yield_Trend_63", pd.Series(False, index=df.index)) > 0
    safe = df["Safe"].pct_change().fillna(0.0)
    return pd.Series(np.where(trend, cash, safe), index=df.index)


def _apply_uncertainty_penalty(probs: np.ndarray, uncertainty: np.ndarray, penalty: float) -> np.ndarray:
    if penalty <= 0:
        return np.clip(probs, 0.0, 1.0)
    return np.clip(probs - penalty * uncertainty, 0.0, 1.0)


def _blend_regime_expert(
    train: pd.DataFrame,
    train_regime: pd.Series,
    test_regime: pd.Series,
    x_train_df: pd.DataFrame,
    x_test_df: pd.DataFrame,
    base_probs: np.ndarray,
    base_uncertainty: np.ndarray,
    cfg: PipelineConfig,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    if not cfg.model.enable_regime_expert:
        return base_probs, base_uncertainty, {"used": 0.0}

    stress_states = {"HighVol", "MeanReverting"}
    stress_idx = train_regime[train_regime.isin(stress_states)].index
    normal_idx = train_regime[~train_regime.isin(stress_states)].index

    if len(stress_idx) < cfg.model.min_stress_train_samples or len(normal_idx) < cfg.model.min_stress_train_samples:
        return base_probs, base_uncertainty, {"used": 0.0}

    rng = np.random.default_rng(seed)
    sample_n = min(len(normal_idx), len(stress_idx) * 2)
    sampled_normal = rng.choice(normal_idx.to_numpy(), size=sample_n, replace=False)

    train_idx = stress_idx.union(pd.Index(sampled_normal))
    y_sub = train.loc[train_idx, "Target"]
    if y_sub.nunique() < 2:
        return base_probs, base_uncertainty, {"used": 0.0}

    stress_model = EnsembleStack(cfg.model)
    stress_model.fit(x_train_df.loc[train_idx], y_sub)
    stress_probs, stress_unc = stress_model.predict_proba_with_uncertainty(x_test_df)

    out_probs = base_probs.copy()
    out_unc = base_uncertainty.copy()
    stress_mask = test_regime.isin(stress_states).to_numpy()

    w = float(np.clip(cfg.model.regime_expert_weight, 0.0, 1.0))
    out_probs[stress_mask] = (1 - w) * out_probs[stress_mask] + w * stress_probs[stress_mask]
    out_unc[stress_mask] = (1 - w) * out_unc[stress_mask] + w * stress_unc[stress_mask]

    return out_probs, out_unc, {"used": 1.0, "stress_share": float(stress_mask.mean()), "weight": w}


def _build_equity_track_backtest_cfg(cfg: PipelineConfig) -> BacktestConfig:
    extra = max(0.0, cfg.validation.equity_track_extra_cost_bps)
    return replace(
        cfg.backtest,
        transaction_cost_bps=cfg.backtest.transaction_cost_bps + extra,
        rebalance_frequency=cfg.validation.equity_track_rebalance_frequency,
        benchmark_apply_costs=True,
    )


def _extract_positive_class_proba(pred: np.ndarray) -> np.ndarray:
    arr = np.asarray(pred)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return arr[:, 1]
    raise ValueError(f"Unsupported probability array shape: {arr.shape}")


def _sensitivity_analysis(oos: pd.DataFrame, cfg: PipelineConfig) -> dict[str, object]:
    if len(oos) == 0:
        return {}

    probs = oos["Prob"]
    risk_ret = oos["R_ret"]
    safe_ret = oos["Defensive_ret"]

    threshold_center = float(cfg.risk.threshold if cfg.risk.threshold is not None else 0.5)
    threshold_grid = [threshold_center - 0.03, threshold_center, threshold_center + 0.03]
    threshold_results = []
    for th in threshold_grid:
        exp = probabilities_to_exposure(probs.to_numpy(), risk_ret, safe_ret, cfg.risk, threshold=float(np.clip(th, 0.05, 0.95)))
        gross = exp * risk_ret + (1 - exp) * safe_ret
        threshold_results.append({"threshold": float(th), "sharpe": compute_metrics(gross)["sharpe"]})

    cost_levels = [0.0, cfg.backtest.transaction_cost_bps, 10.0, 15.0]
    cost_results = []
    for c in cost_levels:
        friction = (c + cfg.backtest.slippage_bps) / 10000
        net = oos["Gross"] - oos["Turnover"] * friction
        cost_results.append({"total_cost_bps": c + cfg.backtest.slippage_bps, "sharpe": compute_metrics(net)["sharpe"]})

    return {"threshold_sweep": threshold_results, "cost_sweep": cost_results}


def _drawdown_crisis_table(oos: pd.DataFrame) -> pd.DataFrame:
    if "DD_Strat" not in oos.columns or len(oos) == 0:
        return pd.DataFrame(
            columns=[
                "start_date",
                "trough_date",
                "recovery_date",
                "max_drawdown",
                "decline_days",
                "recovery_days",
                "total_days",
                "regime_at_trough",
                "status",
            ]
        )

    dd = oos["DD_Strat"].fillna(0.0)
    idx = oos.index
    records: list[dict[str, object]] = []

    in_episode = False
    start_i = 0
    trough_i = 0
    trough_val = 0.0

    for i, value in enumerate(dd.to_numpy()):
        if not in_episode and value < 0:
            in_episode = True
            start_i = i
            trough_i = i
            trough_val = float(value)
            continue

        if not in_episode:
            continue

        if value < trough_val:
            trough_val = float(value)
            trough_i = i

        recovered = value >= -1e-10
        at_end = i == len(dd) - 1
        if recovered or at_end:
            end_i = i
            start_dt = idx[start_i]
            trough_dt = idx[trough_i]
            recovery_dt = idx[end_i] if recovered else pd.NaT
            decline_days = trough_i - start_i
            recovery_days = (end_i - trough_i) if recovered else np.nan
            total_days = end_i - start_i
            regime = oos.iloc[trough_i]["Regime"] if "Regime" in oos.columns else "Unknown"

            records.append(
                {
                    "start_date": str(start_dt.date()),
                    "trough_date": str(trough_dt.date()),
                    "recovery_date": str(recovery_dt.date()) if recovered else "",
                    "max_drawdown": trough_val,
                    "decline_days": int(decline_days),
                    "recovery_days": int(recovery_days) if recovered else np.nan,
                    "total_days": int(total_days),
                    "regime_at_trough": str(regime),
                    "status": "Recovered" if recovered else "Open",
                }
            )
            in_episode = False

    table = pd.DataFrame(records)
    if table.empty:
        return table
    return table.sort_values("max_drawdown").reset_index(drop=True)


def _overfitting_diagnostics(fold_metrics: list[dict[str, object]]) -> dict[str, object]:
    if not fold_metrics:
        return {}

    df = pd.DataFrame(fold_metrics)
    if not {"train_sharpe", "sharpe", "train_auc", "test_auc"}.issubset(df.columns):
        return {}

    df = df.replace([np.inf, -np.inf], np.nan)
    sharpe_gap = df["train_sharpe"] - df["sharpe"]
    auc_gap = df["train_auc"] - df["test_auc"]
    brier_gap = df["test_brier"] - df["train_brier"]

    avg_sharpe_gap = float(sharpe_gap.mean())
    avg_auc_gap = float(auc_gap.mean())
    avg_brier_gap = float(brier_gap.mean())
    elevated_folds = int(((sharpe_gap > 0.5) | (auc_gap > 0.10) | (brier_gap > 0.03)).sum())

    status = "Acceptable"
    if avg_sharpe_gap > 0.35 or avg_auc_gap > 0.08 or avg_brier_gap > 0.02 or elevated_folds >= max(2, len(df) // 2):
        status = "Elevated"

    return {
        "status": status,
        "avg_train_sharpe": float(df["train_sharpe"].mean()),
        "avg_test_sharpe": float(df["sharpe"].mean()),
        "avg_sharpe_gap": avg_sharpe_gap,
        "avg_train_auc": float(df["train_auc"].mean()),
        "avg_test_auc": float(df["test_auc"].mean()),
        "avg_auc_gap": avg_auc_gap,
        "avg_train_brier": float(df["train_brier"].mean()),
        "avg_test_brier": float(df["test_brier"].mean()),
        "avg_brier_gap": avg_brier_gap,
        "elevated_folds": elevated_folds,
        "total_folds": int(len(df)),
    }


def _build_institutional_comparison(
    strategy_metrics: dict[str, float],
    institutional_metrics: dict[str, float] | None,
) -> dict[str, object]:
    if not institutional_metrics:
        return {}

    return {
        "reference_model": "Qlib LightGBM Alpha158-style baseline",
        "reference_origin": "Microsoft Qlib public benchmark workflow",
        "strategy_vs_reference": {
            "sharpe_uplift": strategy_metrics.get("sharpe", 0.0) - institutional_metrics.get("sharpe", 0.0),
            "sortino_uplift": strategy_metrics.get("sortino", 0.0) - institutional_metrics.get("sortino", 0.0),
            "drawdown_uplift": strategy_metrics.get("max_drawdown", 0.0) - institutional_metrics.get("max_drawdown", 0.0),
            "annual_return_uplift": strategy_metrics.get("annual_return", 0.0) - institutional_metrics.get("annual_return", 0.0),
        },
    }


def _build_peer_league(
    cfg: PipelineConfig,
    amce_macro_metrics: dict[str, float],
    amce_equity_metrics: dict[str, float],
    amce_macro_returns: pd.Series,
    amce_equity_returns: pd.Series,
    peer_macro_metrics: dict[str, dict[str, float]],
    peer_equity_metrics: dict[str, dict[str, float]],
    peer_macro_returns: dict[str, pd.Series],
    peer_equity_returns: dict[str, pd.Series],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    specs = get_top_peer_specs()
    league: list[dict[str, object]] = []

    for name, m in peer_macro_metrics.items():
        em = peer_equity_metrics.get(name, {"sharpe": 0.0, "sortino": 0.0, "max_drawdown": 0.0})
        macro_sharpe_uplift = amce_macro_metrics["sharpe"] - m["sharpe"]
        macro_sortino_uplift = amce_macro_metrics["sortino"] - m["sortino"]
        macro_dd_uplift = amce_macro_metrics["max_drawdown"] - m["max_drawdown"]

        eq_sharpe_uplift = amce_equity_metrics["sharpe"] - em["sharpe"]
        eq_sortino_uplift = amce_equity_metrics["sortino"] - em["sortino"]
        eq_dd_uplift = amce_equity_metrics["max_drawdown"] - em["max_drawdown"]

        superior_macro = bootstrap_superiority_probability(
            amce_macro_returns,
            peer_macro_returns[name],
            n_boot=cfg.validation.superiority_bootstrap_trials,
            seed=cfg.model.random_state + 100,
        )
        superior_eq = bootstrap_superiority_probability(
            amce_equity_returns,
            peer_equity_returns[name],
            n_boot=cfg.validation.superiority_bootstrap_trials,
            seed=cfg.model.random_state + 200,
        )

        passes_macro = (
            macro_sharpe_uplift >= cfg.validation.min_peer_sharpe_uplift_macro
            and macro_sortino_uplift >= cfg.validation.min_peer_sortino_uplift_macro
            and macro_dd_uplift >= cfg.validation.min_peer_drawdown_uplift_macro
            and superior_macro >= cfg.validation.superiority_confidence
        )

        passes_equity = (
            eq_sharpe_uplift >= cfg.validation.min_peer_sharpe_uplift_equity
            and eq_sortino_uplift >= cfg.validation.min_peer_sortino_uplift_equity
            and eq_dd_uplift >= cfg.validation.min_peer_drawdown_uplift_equity
            and superior_eq >= cfg.validation.superiority_confidence
        )

        peer_info = specs.get(name)
        league.append(
            {
                "model_key": name,
                "model_label": peer_info.label if peer_info else name,
                "model_proxy": peer_info.proxy_label if peer_info else name,
                "source_url": peer_info.source_url if peer_info else "",
                "qlib_annualized_return": peer_info.qlib_annualized_return if peer_info else None,
                "qlib_information_ratio": peer_info.qlib_information_ratio if peer_info else None,
                "qlib_max_drawdown": peer_info.qlib_max_drawdown if peer_info else None,
                "is_surrogate": peer_info.is_surrogate if peer_info else True,
                "peer_macro_sharpe": m["sharpe"],
                "peer_equity_sharpe": em["sharpe"],
                "macro_sharpe_uplift": macro_sharpe_uplift,
                "macro_sortino_uplift": macro_sortino_uplift,
                "macro_drawdown_uplift": macro_dd_uplift,
                "equity_sharpe_uplift": eq_sharpe_uplift,
                "equity_sortino_uplift": eq_sortino_uplift,
                "equity_drawdown_uplift": eq_dd_uplift,
                "superiority_prob_macro": superior_macro,
                "superiority_prob_equity": superior_eq,
                "passes_macro": passes_macro,
                "passes_equity": passes_equity,
                "passes_all": passes_macro and (passes_equity if cfg.validation.dual_track_enabled else True),
            }
        )

    league.sort(key=lambda r: r["macro_sharpe_uplift"])
    failing = [row["model_key"] for row in league if not row["passes_all"]]

    summary = {
        "enabled": bool(cfg.validation.peer_suite_enabled),
        "models_tested": len(league),
        "passed_all_peers": len(failing) == 0,
        "failing_models": failing,
        "thresholds": {
            "macro": {
                "min_sharpe_uplift": cfg.validation.min_peer_sharpe_uplift_macro,
                "min_sortino_uplift": cfg.validation.min_peer_sortino_uplift_macro,
                "min_drawdown_uplift": cfg.validation.min_peer_drawdown_uplift_macro,
            },
            "equity": {
                "min_sharpe_uplift": cfg.validation.min_peer_sharpe_uplift_equity,
                "min_sortino_uplift": cfg.validation.min_peer_sortino_uplift_equity,
                "min_drawdown_uplift": cfg.validation.min_peer_drawdown_uplift_equity,
            },
            "superiority_confidence": cfg.validation.superiority_confidence,
        },
    }

    return league, summary


def run_walk_forward_validation(feature_frame: FeatureFrame, cfg: PipelineConfig) -> ValidationReport:
    df = feature_frame.data.copy()
    cols = feature_frame.feature_columns

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["Target", "Risk", "Safe", "Yield"])

    folds = _make_folds(
        n=len(df),
        min_train=cfg.validation.min_train_days,
        n_splits=cfg.validation.n_splits,
        embargo_days=cfg.validation.embargo_days,
    )
    if not folds:
        raise ValueError("Unable to create walk-forward folds. Increase history or lower min_train_days.")

    peer_names = cfg.validation.peer_model_names or default_peer_model_names()
    peer_names = [n.lower().strip() for n in peer_names]

    fold_metrics: list[dict[str, object]] = []
    oos_frames: list[pd.DataFrame] = []
    oos_equity_frames: list[pd.DataFrame] = []
    baseline_frames: list[pd.DataFrame] = []
    fold_feature_selections: list[dict[str, object]] = []
    fold_feature_importances: list[pd.Series] = []

    peer_macro_frames: dict[str, list[pd.DataFrame]] = {name: [] for name in peer_names}
    peer_equity_frames: dict[str, list[pd.DataFrame]] = {name: [] for name in peer_names}

    equity_bt_cfg = _build_equity_track_backtest_cfg(cfg)

    for i, fold in enumerate(folds, start=1):
        train = df.iloc[fold.train_start : fold.train_end].copy()
        test = df.iloc[fold.test_start : fold.test_end].copy()

        x_train_df, x_test_df, selected_cols, selection_meta, _ = _prepare_fold_features(
            train=train,
            test=test,
            feature_cols=cols,
            cfg=cfg,
            fold_seed=cfg.model.random_state + i,
        )
        fold_feature_selections.append({"fold": i, **selection_meta})

        regime_model = fit_regime_classifier(
            train,
            method=cfg.validation.regime_method,
            n_states=cfg.validation.regime_states,
            random_state=cfg.model.random_state + i,
        )
        train_regime = regime_model.predict(train)
        test_regime = regime_model.predict(test)

        model = EnsembleStack(cfg.model)
        model.fit(x_train_df, train["Target"])
        imp = model.feature_importance(selected_cols)
        if len(imp):
            fold_feature_importances.append(imp)

        train_probs_raw, train_unc = model.predict_proba_with_uncertainty(x_train_df)
        test_probs_raw, test_unc = model.predict_proba_with_uncertainty(x_test_df)
        test_component_probs = model.predict_component_probas(x_test_df)

        test_probs_raw, test_unc, expert_info = _blend_regime_expert(
            train=train,
            train_regime=train_regime,
            test_regime=test_regime,
            x_train_df=x_train_df,
            x_test_df=x_test_df,
            base_probs=test_probs_raw,
            base_uncertainty=test_unc,
            cfg=cfg,
            seed=cfg.model.random_state + i,
        )

        train_probs = _apply_uncertainty_penalty(train_probs_raw, train_unc, cfg.model.uncertainty_penalty)
        test_probs = _apply_uncertainty_penalty(test_probs_raw, test_unc, cfg.model.uncertainty_penalty)

        learned_threshold = cfg.risk.threshold
        if learned_threshold is None:
            learned_threshold = learn_threshold(
                train_probs,
                train["Risk"].pct_change().fillna(0.0),
                _safe_returns(train),
                cfg.model,
            )

        train_exposure = probabilities_to_exposure(
            train_probs,
            train["Risk"].pct_change().fillna(0.0),
            _safe_returns(train),
            cfg.risk,
            threshold=float(learned_threshold),
        )
        exposure = probabilities_to_exposure(
            test_probs,
            test["Risk"].pct_change().fillna(0.0),
            _safe_returns(test),
            cfg.risk,
            threshold=float(learned_threshold),
        )

        bt_train = run_backtest(train, train_exposure, cfg.backtest)
        bt = run_backtest(test, exposure, cfg.backtest)
        bt_equity = run_backtest(test, exposure, equity_bt_cfg)

        fold_frame = bt.frame.copy()
        fold_frame["Fold"] = i
        fold_frame["Prob_Raw"] = test_probs_raw
        fold_frame["Prob_Uncertainty"] = test_unc
        fold_frame["Prob"] = test_probs
        fold_frame["Threshold"] = float(learned_threshold)
        fold_frame["Regime"] = test_regime
        if not test_component_probs.empty:
            comp_cols = list(test_component_probs.columns[:8])
            for col in comp_cols:
                fold_frame[col] = test_component_probs[col]
        oos_frames.append(fold_frame)

        eq_frame = bt_equity.frame.copy()
        eq_frame["Fold"] = i
        eq_frame["Prob"] = test_probs
        eq_frame["Regime"] = test_regime
        oos_equity_frames.append(eq_frame)

        fm = {
            "fold": i,
            "train_rows": len(train),
            "test_rows": len(test),
            "train_start": str(train.index.min().date()),
            "train_end": str(train.index.max().date()),
            "test_start": str(test.index.min().date()),
            "test_end": str(test.index.max().date()),
            "threshold": float(learned_threshold),
            "regime_expert_used": expert_info.get("used", 0.0),
            "stress_share": expert_info.get("stress_share", 0.0),
            "regime_method": cfg.validation.regime_method,
            "selected_features": len(selected_cols),
            "top_features": ",".join(selected_cols[:8]),
            "model_count": len(model.base_models),
            "train_sharpe": bt_train.metrics["sharpe"],
            "train_sortino": bt_train.metrics["sortino"],
            "train_max_drawdown": bt_train.metrics["max_drawdown"],
            "test_sharpe": bt.metrics["sharpe"],
            "test_sortino": bt.metrics["sortino"],
            "test_max_drawdown": bt.metrics["max_drawdown"],
        }
        try:
            fm["train_auc"] = float(roc_auc_score(train["Target"], train_probs)) if train["Target"].nunique() > 1 else 0.5
            fm["test_auc"] = float(roc_auc_score(test["Target"], test_probs)) if test["Target"].nunique() > 1 else 0.5
        except Exception:
            fm["train_auc"] = 0.5
            fm["test_auc"] = 0.5
        try:
            fm["train_brier"] = float(brier_score_loss(train["Target"], np.clip(train_probs, 1e-6, 1 - 1e-6)))
            fm["test_brier"] = float(brier_score_loss(test["Target"], np.clip(test_probs, 1e-6, 1 - 1e-6)))
        except Exception:
            fm["train_brier"] = 0.25
            fm["test_brier"] = 0.25
        fm.update(bt.metrics)

        if cfg.validation.institutional_baseline_enabled:
            baseline = QlibStyleBaseline(random_state=cfg.model.random_state + i)
            baseline.fit(x_train_df, train["Target"])
            baseline_probs = baseline.predict_proba(x_test_df)
            baseline_exposure = pd.Series(
                (baseline_probs >= cfg.validation.institutional_baseline_threshold).astype(float),
                index=test.index,
                name="Exposure",
            )
            bt_base = run_backtest(test, baseline_exposure, cfg.backtest)

            base_frame = bt_base.frame.copy()
            base_frame["Fold"] = i
            base_frame["Prob_Baseline"] = baseline_probs
            baseline_frames.append(base_frame)

            fm["institutional_baseline_sharpe"] = bt_base.metrics["sharpe"]
            fm["institutional_baseline_sortino"] = bt_base.metrics["sortino"]
            fm["institutional_baseline_max_drawdown"] = bt_base.metrics["max_drawdown"]

        if cfg.validation.peer_suite_enabled:
            for model_name in peer_names:
                peer_model = build_peer_model(model_name, random_state=cfg.model.random_state + i * 11)
                peer_model.fit(x_train_df, train["Target"])  # type: ignore[attr-defined]
                peer_pred = peer_model.predict_proba(x_test_df)  # type: ignore[attr-defined]
                peer_probs = _extract_positive_class_proba(peer_pred)
                peer_exposure = pd.Series(
                    (peer_probs >= cfg.validation.peer_probability_threshold).astype(float),
                    index=test.index,
                    name="Exposure",
                )

                peer_bt = run_backtest(test, peer_exposure, cfg.backtest)
                peer_bt_eq = run_backtest(test, peer_exposure, equity_bt_cfg)

                pf = peer_bt.frame.copy()
                pf["Fold"] = i
                pf["PeerModel"] = model_name
                pf["PeerProb"] = peer_probs
                peer_macro_frames[model_name].append(pf)

                pfe = peer_bt_eq.frame.copy()
                pfe["Fold"] = i
                pfe["PeerModel"] = model_name
                pfe["PeerProb"] = peer_probs
                peer_equity_frames[model_name].append(pfe)

        fold_metrics.append(fm)

    oos = pd.concat(oos_frames).sort_index()
    oos_equity = pd.concat(oos_equity_frames).sort_index()
    oos["RollingSharpe126"] = rolling_sharpe(oos["Net"], window=126)

    strategy_metrics = compute_metrics(oos["Net"], benchmark_returns=oos["Benchmark_ret"], turnover=oos["Turnover"])
    strategy_metrics_equity = compute_metrics(
        oos_equity["Net"],
        benchmark_returns=oos_equity["Benchmark_ret"],
        turnover=oos_equity["Turnover"],
    )
    benchmark_metrics = compute_metrics(oos["Benchmark_ret"])
    benchmark_metrics_equity = compute_metrics(oos_equity["Benchmark_ret"])

    institutional_metrics: dict[str, float] | None = None
    baseline_oos: pd.DataFrame | None = None
    if baseline_frames:
        baseline_oos = pd.concat(baseline_frames).sort_index()
        institutional_metrics = compute_metrics(
            baseline_oos["Net"],
            benchmark_returns=baseline_oos["Benchmark_ret"],
            turnover=baseline_oos["Turnover"],
        )

    peer_macro_metrics: dict[str, dict[str, float]] = {}
    peer_equity_metrics: dict[str, dict[str, float]] = {}
    peer_macro_returns: dict[str, pd.Series] = {}
    peer_equity_returns: dict[str, pd.Series] = {}
    peer_oos_concat: dict[str, pd.DataFrame] = {}
    peer_oos_equity_concat: dict[str, pd.DataFrame] = {}

    if cfg.validation.peer_suite_enabled:
        for name in peer_names:
            if not peer_macro_frames[name]:
                continue
            peer_macro_df = pd.concat(peer_macro_frames[name]).sort_index()
            peer_equity_df = pd.concat(peer_equity_frames[name]).sort_index()
            peer_oos_concat[name] = peer_macro_df
            peer_oos_equity_concat[name] = peer_equity_df
            peer_macro_metrics[name] = compute_metrics(
                peer_macro_df["Net"],
                benchmark_returns=peer_macro_df["Benchmark_ret"],
                turnover=peer_macro_df["Turnover"],
            )
            peer_equity_metrics[name] = compute_metrics(
                peer_equity_df["Net"],
                benchmark_returns=peer_equity_df["Benchmark_ret"],
                turnover=peer_equity_df["Turnover"],
            )
            peer_macro_returns[name] = peer_macro_df["Net"]
            peer_equity_returns[name] = peer_equity_df["Net"]

    peer_league_table: list[dict[str, object]] = []
    peer_head_to_head_summary: dict[str, object] = {"enabled": False, "models_tested": 0, "passed_all_peers": True, "failing_models": []}
    if peer_macro_metrics:
        peer_league_table, peer_head_to_head_summary = _build_peer_league(
            cfg=cfg,
            amce_macro_metrics=strategy_metrics,
            amce_equity_metrics=strategy_metrics_equity,
            amce_macro_returns=oos["Net"],
            amce_equity_returns=oos_equity["Net"],
            peer_macro_metrics=peer_macro_metrics,
            peer_equity_metrics=peer_equity_metrics,
            peer_macro_returns=peer_macro_returns,
            peer_equity_returns=peer_equity_returns,
        )

    regime_metrics: dict[str, dict[str, float]] = {}
    for reg in sorted(oos["Regime"].dropna().unique().tolist()):
        sub = oos.loc[oos["Regime"] == reg, "Net"]
        if len(sub) >= 20:
            regime_metrics[reg] = compute_metrics(sub)

    terciles = compute_tercile_metrics(oos["Net"])

    p_val, perm_sharpes = permutation_test_sharpe(
        oos["Exposure"].to_numpy(),
        oos["R_ret"].to_numpy(),
        oos["Defensive_ret"].to_numpy(),
        strategy_metrics["sharpe"],
        n_trials=cfg.validation.permutation_trials,
        seed=cfg.model.random_state,
    )
    alpha_stats = bootstrap_alpha_ci(oos["Net"], oos["Benchmark_ret"], seed=cfg.model.random_state)

    governance = evaluate_governance(
        strategy_metrics,
        benchmark_metrics,
        p_val,
        terciles,
        institutional_metrics,
        cfg.validation,
        peer_head_to_head_summary,
    )
    sensitivity = _sensitivity_analysis(oos, cfg)
    institutional_comparison = _build_institutional_comparison(strategy_metrics, institutional_metrics)

    feature_importance_top: list[dict[str, object]] = []
    if fold_feature_importances:
        imp_df = pd.concat(fold_feature_importances, axis=1).fillna(0.0)
        imp_avg = imp_df.mean(axis=1).sort_values(ascending=False)
        feature_importance_top = [{"feature": k, "importance": float(v)} for k, v in imp_avg.head(40).items()]

    regime_counts = oos["Regime"].value_counts(dropna=False).to_dict()
    regime_counts = {str(k): int(v) for k, v in regime_counts.items()}
    overfit_diag = _overfitting_diagnostics(fold_metrics)
    crisis_table = _drawdown_crisis_table(oos)
    ensemble_component_columns = [c for c in oos.columns if c.startswith("Ens_")]

    summary = {
        "rows_oos": int(len(oos)),
        "n_folds": int(len(fold_metrics)),
        "strategy_metrics": strategy_metrics,
        "strategy_metrics_equity_track": strategy_metrics_equity,
        "benchmark_metrics": benchmark_metrics,
        "benchmark_metrics_equity_track": benchmark_metrics_equity,
        "institutional_baseline_metrics": institutional_metrics,
        "institutional_comparison": institutional_comparison,
        "peer_league_table": peer_league_table,
        "peer_head_to_head_summary": peer_head_to_head_summary,
        "feature_importance_top": feature_importance_top,
        "regime_distribution": regime_counts,
        "overfitting_diagnostics": overfit_diag,
        "crisis_episode_count": int(len(crisis_table)),
        "ensemble_component_columns": ensemble_component_columns,
        "institutional_reference_urls": [
            "https://github.com/microsoft/qlib/blob/main/examples/benchmarks/README.md",
            "https://github.com/microsoft/qlib/blob/main/examples/benchmarks/DoubleEnsemble/workflow_config_doubleensemble_Alpha158.yaml",
            "https://github.com/microsoft/qlib/blob/main/examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml",
            "https://arxiv.org/abs/2009.11189",
        ],
        "alpha_stats": alpha_stats,
        "permutation_p_value": p_val,
    }

    extras: dict[str, object] = {
        "oos_frame": oos,
        "oos_equity_track_frame": oos_equity,
        "perm_sharpes": perm_sharpes.tolist(),
        "tercile_metrics": terciles,
        "peer_oos_frames": peer_oos_concat,
        "peer_oos_equity_frames": peer_oos_equity_concat,
        "fold_feature_selection": fold_feature_selections,
        "crisis_table": crisis_table,
    }
    if baseline_oos is not None:
        extras["institutional_baseline_oos"] = baseline_oos

    return ValidationReport(
        summary=summary,
        fold_metrics=fold_metrics,
        governance=governance,
        regime_metrics=regime_metrics,
        sensitivity=sensitivity,
        extras=extras,
    )
