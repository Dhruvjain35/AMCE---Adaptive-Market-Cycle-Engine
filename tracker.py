from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd

from schema import PipelineConfig
from amce_types import ValidationReport


def dataset_fingerprint(df: pd.DataFrame) -> str:
    payload = df.to_csv(index=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def persist_run_artifacts(
    report: ValidationReport,
    base_frame: pd.DataFrame,
    config: PipelineConfig,
    feature_version: str,
) -> Path:
    out_root = Path(config.reporting.output_dir)
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Config snapshot
    (run_dir / "config_snapshot.json").write_text(json.dumps(config.to_dict(), indent=2))

    metadata = {
        "dataset_fingerprint": dataset_fingerprint(base_frame),
        "feature_registry_version": feature_version,
        "model_version": "ensemble_v1",
        "created_utc": datetime.utcnow().isoformat(),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # Persist fold metrics and OOS frame
    pd.DataFrame(report.fold_metrics).to_csv(run_dir / "fold_metrics.csv", index=False)

    oos = report.extras.get("oos_frame")
    if isinstance(oos, pd.DataFrame):
        oos.to_csv(run_dir / "oos_backtest.csv")

    crisis = report.extras.get("crisis_table")
    if isinstance(crisis, pd.DataFrame):
        crisis.to_csv(run_dir / "crisis_table.csv", index=False)

    summary_payload = report.to_dict()
    summary_payload["extras"].pop("oos_frame", None)
    (run_dir / "validation_report.json").write_text(json.dumps(summary_payload, indent=2))

    return run_dir
