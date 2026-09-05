#!/usr/bin/env python3
"""Assemble machine-readable v1.1.0 held-out summary and integrity checks."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


TARGETS = [f"Y{i:02d}" for i in range(1, 8)]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    clean = frame.replace({np.nan: None})
    return clean.to_dict(orient="records")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.root.resolve()
    validation = root / "validation"
    model_path = root / "models" / "SAF_Predict_public_model_bundle_v1.joblib"
    data_path = root / "data" / "held_out_test.csv"
    development_path = root / "data" / "development.csv"
    wide_path = validation / "held_out_predictions.csv"
    long_path = validation / "held_out_test_predictions_long.csv"
    metrics_path = validation / "held_out_test_metrics.csv"
    source_metrics_path = validation / "held_out_source_supported_metrics.csv"
    reproduction_path = validation / "held_out_reproduction_check.json"
    sanitization_path = root / "models" / "public_artifact_sanitization_checks.json"
    required = [
        model_path,
        data_path,
        development_path,
        wide_path,
        long_path,
        metrics_path,
        source_metrics_path,
        reproduction_path,
        sanitization_path,
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    bundle = joblib.load(model_path)
    heldout = pd.read_csv(data_path)
    development = pd.read_csv(development_path)
    wide = pd.read_csv(wide_path)
    long = pd.read_csv(long_path)
    metrics = pd.read_csv(metrics_path)
    source_metrics = pd.read_csv(source_metrics_path)
    reproduction = json.loads(reproduction_path.read_text(encoding="utf-8"))
    sanitization = json.loads(sanitization_path.read_text(encoding="utf-8"))

    if bundle.get("schema_version") != 2 or set(bundle.get("models", {})) != set(TARGETS):
        raise AssertionError("Expected a schema-v2 bundle with seven separately fitted models")
    if len(heldout) != 10 or len(wide) != 10 or len(long) != 70 or len(metrics) != 7:
        raise AssertionError("Unexpected held-out artifact row count")
    available = long.groupby("target")["label_available"].sum().astype(int).to_dict()
    expected_available = {target: (8 if target == "Y02" else 10) for target in TARGETS}
    if available != expected_available:
        raise AssertionError(f"Unexpected label availability: {available}")
    if not np.isfinite(long["predicted"].to_numpy(float)).all():
        raise AssertionError("One or more held-out predictions are non-finite")

    formulae_in_development = set(development["molecular_formula"].astype(str))
    formula_seen = heldout["molecular_formula"].astype(str).isin(formulae_in_development)
    metric_by_target = metrics.set_index("target")
    y04 = metric_by_target.loc["Y04"]
    y02_difference = np.abs(
        wide["Y02_pred"].to_numpy(float)
        - wide["Y01_pred"].to_numpy(float) * wide["Y03_pred"].to_numpy(float)
    )
    direct_canary = bool(float(np.max(y02_difference)) > 1e-6)
    if not direct_canary:
        raise AssertionError("Y02 predictions still satisfy the deprecated Y01 x Y03 identity")

    public_hashes = {
        "joblib_path": "models/SAF_Predict_public_model_bundle_v1.joblib",
        "joblib_sha256": sha256(model_path),
        "wide_holdout_predictions_path": "validation/held_out_predictions.csv",
        "wide_holdout_predictions_sha256": sha256(wide_path),
        "long_holdout_predictions_path": "validation/held_out_test_predictions_long.csv",
        "long_holdout_predictions_sha256": sha256(long_path),
        "held_out_test_metrics_path": "validation/held_out_test_metrics.csv",
        "held_out_test_metrics_sha256": sha256(metrics_path),
        "source_supported_metrics_path": "validation/held_out_source_supported_metrics.csv",
        "source_supported_metrics_sha256": sha256(source_metrics_path),
    }
    summary = {
        "platform": "SAF-Predict",
        "version": str(bundle["version"]),
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_set_name": "structurally stratified internal held-out test set",
        "independent_external_validation": False,
        "experimental_validation": False,
        "n_records": len(heldout),
        "Y02_reference_labels": int(heldout["Y02"].notna().sum()),
        "unique_formulae": int(heldout["molecular_formula"].nunique()),
        "records_with_formula_seen_in_development": int(formula_seen.sum()),
        "unique_formulae_seen_in_development": int(
            heldout.loc[formula_seen, "molecular_formula"].nunique()
        ),
        "recorded_private_joblib_sha256": reproduction.get("model_sha256"),
        "public_deidentified_artifacts": public_hashes,
        "metrics": json_records(metrics),
        "source_supported_metrics": json_records(source_metrics),
        "ood": {
            "median_distance_percentile": float(wide["ood_distance_percentile"].median()),
            "high_distance_warnings_at_or_above_95th_reference": int(
                (wide["ood_distance_percentile"] >= 95).sum()
            ),
        },
        "Y02_policy": (
            "Y02 is predicted by a separately fitted RF model trained on 162 identity-resolved "
            "independently collected responses in 66 molecular-formula groups. It is not calculated "
            "from the predicted Y01 and Y03 values."
        ),
        "Y04_limitation": (
            f"Y04 internal-holdout performance was R2 = {float(y04['r2']):.3f}; "
            f"MAE = {float(y04['mae']):.1f} degC; n = {int(y04['n'])}. Experimental "
            "confirmation is required for decision use."
        ),
        "interpretation_boundary": (
            "This small test set was held out within the same database-building workflow. It does "
            "not establish independent external, prospective, new-chemical-space, OOD, or "
            "experimental validation."
        ),
    }
    checks = {
        "check_scope": (
            "Self-generated released-file integrity and internal-consistency checks; not an "
            "independent audit or proof of analysis chronology."
        ),
        "version": str(bundle["version"]),
        "all_listed_checks_passed": True,
        "bundle_schema_version": int(bundle["schema_version"]),
        "separately_fitted_model_targets": sorted(bundle["models"]),
        "deployment_models": bundle["deployment_models"],
        "held_out_prediction_rows": len(wide),
        "record_property_rows": len(long),
        "metric_rows": len(metrics),
        "label_available_by_target": available,
        "all_predictions_finite": True,
        "Y02_separately_modelled_canary": direct_canary,
        "maximum_abs_Y02_prediction_minus_Y01_times_Y03": float(np.max(y02_difference)),
        "source_private_joblib_sha256": reproduction.get("model_sha256"),
        "public_deidentified_artifacts": public_hashes,
        "sanitization_checks_sha256": sha256(sanitization_path),
        "sanitization_semantic_equivalence_passed": bool(
            sanitization["semantic_equivalence"]["canonical_scientific_payload_equal"]
            and sanitization["semantic_equivalence"]["predictions_exact_for_all_checked_rows"]
        ),
        "author_reported_workflow": {
            "held_out_set_excluded_from_model_selection_and_tuning": True,
            "post_test_refitting_performed": False,
            "independently_timestamped_pre_evaluation_release_present_in_repository": False,
        },
        "claim_boundary": summary["interpretation_boundary"],
    }

    summary_path = validation / "held_out_test_summary.json"
    checks_path = validation / "held_out_evaluation_checks.json"
    with summary_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    with checks_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(checks, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"summary": str(summary_path), "checks": str(checks_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
