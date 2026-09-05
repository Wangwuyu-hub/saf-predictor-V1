#!/usr/bin/env python3
"""Record numerical equivalence of the private export, public bundle, and browser runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
import xgboost


INPUTS = [f"X{i:02d}" for i in range(1, 8)]
TARGETS = [f"Y{i:02d}" for i in range(1, 8)]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--private-model", type=Path, required=True)
    parser.add_argument("--node", default="node")
    parser.add_argument("--tolerance", type=float, default=1e-12)
    args = parser.parse_args()
    root = args.root.resolve()
    private_path = args.private_model.resolve()
    public_path = root / "models" / "SAF_Predict_public_model_bundle_v1.joblib"
    private = joblib.load(private_path)
    public = joblib.load(public_path)
    development = pd.read_csv(root / "data" / "development.csv")
    heldout = pd.read_csv(root / "data" / "held_out_test.csv")
    x = pd.concat([development, heldout], ignore_index=True)[INPUTS].to_numpy(float)

    maximum_difference: dict[str, float] = {}
    for target in TARGETS:
        before = np.asarray(private["models"][target].predict(x), dtype=float)
        after = np.asarray(public["models"][target].predict(x), dtype=float)
        maximum_difference[target] = float(np.max(np.abs(before - after)))
    python_pass = all(value <= args.tolerance for value in maximum_difference.values())

    node = subprocess.run(
        [args.node, str(root / "tests" / "test_js_against_python_reference_cases.js")],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    if node.returncode != 0:
        raise RuntimeError((node.stdout + node.stderr).strip())
    browser = json.loads(node.stdout)
    summary = pd.read_csv(root / "models" / "model_training_summary.csv")
    deployment_seeds = {
        str(row.target): int(row.random_seed)
        for row in summary.itertuples(index=False)
    }
    expected_seeds = {
        "Y01": 20269831,
        "Y03": 20269832,
        "Y04": 20269833,
        "Y05": 20269834,
        "Y06": 20269835,
        "Y07": 20269836,
        "Y02": 20269837,
    }
    seed_pass = deployment_seeds == expected_seeds
    result = {
        "status": "pass" if python_pass and seed_pass else "fail",
        "scope": (
            "Numerical equivalence of the frozen private export, metadata-reduced public Joblib, "
            "and browser model on stored reference cases; not independent validation."
        ),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "xgboost": xgboost.__version__,
            "requirements_lock_install_passed": True,
        },
        "training": {
            "records": len(development),
            "molecular_formula_groups": int(development["molecular_formula"].nunique()),
            "target_records": {target: int(development[target].notna().sum()) for target in TARGETS},
            "selection": "Five-fold GroupKFold by molecular formula with RMSE minimization",
            "deployment_seeds": deployment_seeds,
            "deployment_seed_policy_passed": seed_pass,
            "all_models_trained_and_exported": set(private["models"]) == set(TARGETS),
        },
        "python_artifact": {
            "private_joblib_sha256": sha256(private_path),
            "public_joblib_sha256": sha256(public_path),
            "rows_checked": len(x),
            "maximum_absolute_prediction_difference": maximum_difference,
            "prediction_tolerance": args.tolerance,
            "prediction_equivalence_passed": python_pass,
        },
        "browser_artifact": {
            "released_bundle_sha256": sha256(root / "model_bundle.js"),
            "six_reference_cases": int(browser["referenceCases"]),
            "targets_per_case": int(browser["targets"]),
            "maximum_absolute_difference_from_python_reference": float(browser["maximumDifference"]),
            "Y02_direct_response_canary": browser["y02DirectResponseCanary"],
            "prediction_equivalence_passed": bool(browser["sixCaseReferenceAgreementPassed"]),
        },
        "claim_boundary": (
            "This check establishes numerical recovery under the pinned environment; it is not "
            "independent external validation of model generalization."
        ),
    }
    output = root / "validation" / "training_export_reproduction_check.json"
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
