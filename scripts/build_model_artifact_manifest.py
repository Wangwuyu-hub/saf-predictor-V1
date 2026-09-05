#!/usr/bin/env python3
"""Build the v1.1.0 model-artifact manifest after public sanitization."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--private-joblib", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    private_joblib = args.private_joblib.resolve()
    model_dir = root / "models"
    validation = root / "validation"

    paths = {
        "joblib_path": model_dir / "SAF_Predict_public_model_bundle_v1.joblib",
        "browser_model_bundle_js_path": root / "model_bundle.js",
        "model_bundle_json_path": model_dir / "model_bundle.json",
        "wide_holdout_predictions_path": validation / "held_out_predictions.csv",
        "long_holdout_predictions_path": validation / "held_out_test_predictions_long.csv",
        "sanitization_check_path": model_dir / "public_artifact_sanitization_checks.json",
    }
    for path in [private_joblib, *paths.values(), model_dir / "model_training_summary.csv"]:
        if not path.is_file():
            raise FileNotFoundError(path)

    summary = pd.read_csv(model_dir / "model_training_summary.csv")
    if set(summary["target"]) != {f"Y{i:02d}" for i in range(1, 8)}:
        raise AssertionError("Training summary does not cover Y01-Y07")
    deployment = {
        row.target: str(row.deployment_model)
        for row in summary.itertuples(index=False)
    }
    best_parameters = {
        row.target: json.loads(row.best_params)
        for row in summary.itertuples(index=False)
    }
    public = {
        key: value.relative_to(root).as_posix()
        for key, value in paths.items()
    }
    public.update(
        {
            key.replace("_path", "_sha256"): sha256(value)
            for key, value in paths.items()
        }
    )
    public["deidentification_boundary"] = (
        "Molecule names, molecular formulae, original development-record identifiers, and "
        "bilingual display metadata are absent from public model artifacts. X01-X07 vectors "
        "remain for applicability-domain calculations and may be linkable to a known dataset."
    )

    artifact = {
        "platform_name": "SAF-Predict",
        "version": "1.1.0",
        "manifest_scope": (
            "Self-generated analysis metadata and content identifiers; not an independent audit "
            "or evidence of pre-evaluation chronology."
        ),
        "development_records": 300,
        "development_formula_groups": 73,
        "Y02_training_records": 162,
        "Y02_training_formula_groups": 66,
        "held_out_records": 10,
        "held_out_Y02_reference_labels": 8,
        "held_out_role": "structurally stratified internal held-out test set",
        "deployment_mapping": deployment,
        "recorded_hyperparameter_selection": (
            "Five-fold GroupKFold grid search on target-specific development complete cases, "
            "scored by negative RMSE."
        ),
        "recorded_uncertainty_method": (
            "Symmetric 90% and 95% empirical intervals from molecular-formula-group maxima of "
            "nested-CV out-of-fold absolute residuals."
        ),
        "recorded_descriptor_distance_method": (
            "Continuous nearest-neighbour seven-descriptor-space distance percentile relative "
            "to development leave-one-formula-group-out distances; the 95th percentile is a "
            "high-distance warning, not an externally validated OOD boundary."
        ),
        "Y02_policy": (
            "Y02 is a separately fitted random-forest output trained only on 162 identity-resolved "
            "independently collected labels. Missing Y02 labels are not filled with Y01 x Y03."
        ),
        "recorded_test_policy": (
            "The held-out table is not read during model selection or fitting. The repository does "
            "not independently establish when its reference values were first accessed."
        ),
        "claim_boundary": (
            "Research-stage physicochemical prescreening of pure hydrocarbon molecules; not "
            "independent external validation, fuel certification, sustainability assessment, or "
            "engine-performance verification."
        ),
        "artifact_generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_artifact_identifiers": {
            "private_joblib_sha256": sha256(private_joblib),
            "development_csv_sha256": sha256(root / "data" / "development.csv"),
            "nested_cv_oof_sha256": sha256(root / "artifacts" / "cv" / "oof_predictions.csv"),
        },
        "public_deidentified_artifacts": public,
        "recorded_best_parameters": best_parameters,
    }
    output = model_dir / "model_artifact_manifest.json"
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"path": str(output), "sha256": sha256(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
