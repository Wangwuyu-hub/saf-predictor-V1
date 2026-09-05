#!/usr/bin/env python3
"""Regression-check unchanged targets across the v1.0.0 to v1.1.0 Y02 migration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


INPUTS = [f"X{i:02d}" for i in range(1, 8)]
UNCHANGED_TARGETS = ["Y01", "Y03", "Y04", "Y05", "Y06", "Y07"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-model", type=Path, required=True)
    parser.add_argument("--new-model", type=Path, required=True)
    parser.add_argument("--development", type=Path, default=Path("data/development.csv"))
    parser.add_argument("--heldout", type=Path, default=Path("data/held_out_test.csv"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("validation/v1_0_to_v1_1_six_target_regression_check.json"),
    )
    parser.add_argument("--tolerance", type=float, default=1e-9)
    args = parser.parse_args()

    old = joblib.load(args.old_model)
    new = joblib.load(args.new_model)
    frames = [pd.read_csv(args.development), pd.read_csv(args.heldout)]
    x = pd.concat(frames, ignore_index=True)[INPUTS].to_numpy(float)
    maximum_differences: dict[str, float] = {}
    prediction_equal: dict[str, bool] = {}
    interval_equal: dict[str, bool] = {}
    deployment_equal: dict[str, bool] = {}
    for target in UNCHANGED_TARGETS:
        before = np.asarray(old["models"][target].predict(x), dtype=float)
        after = np.asarray(new["models"][target].predict(x), dtype=float)
        difference = float(np.max(np.abs(before - after)))
        maximum_differences[target] = difference
        prediction_equal[target] = difference <= args.tolerance
        interval_equal[target] = old["intervals"][target] == new["intervals"][target]
        deployment_equal[target] = old["deployment_models"][target] == new["deployment_models"][target]

    new_predictions = {
        target: np.asarray(new["models"][target].predict(x), dtype=float)
        for target in ["Y01", "Y02", "Y03"]
    }
    y02_difference = np.abs(
        new_predictions["Y02"] - new_predictions["Y01"] * new_predictions["Y03"]
    )
    result = {
        "record_type": "self-generated numerical regression check",
        "scope": (
            "Checks that the v1.1.0 Y02 semantic migration did not change predictions for "
            "Y01 and Y03-Y07 on the 300 development plus 10 held-out descriptor vectors."
        ),
        "old_model_sha256": sha256(args.old_model),
        "new_model_sha256": sha256(args.new_model),
        "rows_checked": len(x),
        "tolerance": args.tolerance,
        "maximum_absolute_prediction_difference": maximum_differences,
        "unchanged_target_predictions_within_tolerance": prediction_equal,
        "unchanged_target_intervals_exact": interval_equal,
        "unchanged_target_deployment_mapping_exact": deployment_equal,
        "v1_1_Y02_separately_modelled_canary": bool(float(np.max(y02_difference)) > 1e-6),
        "maximum_abs_v1_1_Y02_minus_Y01_times_Y03": float(np.max(y02_difference)),
    }
    result["all_checks_passed"] = bool(
        all(prediction_equal.values())
        and all(interval_equal.values())
        and all(deployment_equal.values())
        and result["v1_1_Y02_separately_modelled_canary"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
