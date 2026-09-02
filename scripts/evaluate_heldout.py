#!/usr/bin/env python3
"""Reproduce the archived evaluation on the 10-record internal held-out test set."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score


INPUTS = [f"X{i:02d}" for i in range(1, 8)]
TARGETS = [f"Y{i:02d}" for i in range(1, 8)]
OUTPUT_META = {
    "Y01": ("Mass-based net heat of combustion", "MJ kg^-1"),
    "Y02": ("Volumetric net heat of combustion", "MJ L^-1"),
    "Y03": ("Density", "g cm^-3"),
    "Y04": ("Solid-liquid phase-transition temperature", "degC"),
    "Y05": ("Boiling point", "degC"),
    "Y06": ("Flash point", "degC"),
    "Y07": ("Kinematic viscosity at 40 degC", "mm^2 s^-1"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def predict(bundle: dict[str, Any], x: np.ndarray) -> dict[str, np.ndarray]:
    predictions = {
        target: np.asarray(model.predict(x), dtype=float).reshape(-1)
        for target, model in bundle["models"].items()
    }
    predictions["Y02"] = predictions["Y01"] * predictions["Y03"]
    return {target: predictions[target] for target in TARGETS}


def ood(bundle: dict[str, Any], x: np.ndarray) -> list[dict[str, float | str]]:
    state = bundle["ood"]
    median = np.asarray(state["feature_median"], dtype=float)
    scale = np.asarray(state["feature_scale"], dtype=float)
    reference = np.asarray(state["group_reference_distances_sorted"], dtype=float)
    training_records = state["training_records"]
    train_x = np.asarray([row["x"] for row in training_records], dtype=float)
    scaled_train = (train_x - median) / scale
    results: list[dict[str, float | str]] = []
    for row in x:
        distance = np.abs(scaled_train - (row - median) / scale).mean(axis=1)
        nearest_index = int(np.argmin(distance))
        nearest_distance = float(distance[nearest_index])
        percentile = 100.0 * (
            np.searchsorted(reference, nearest_distance, side="right") + 1
        ) / (len(reference) + 1)
        results.append(
            {
                "distance": nearest_distance,
                "percentile": float(min(percentile, 100.0)),
                "nearest_record": str(training_records[nearest_index]["record_id"]),
            }
        )
    return results


def clopper_pearson(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, n - successes + 1))
    upper = 1.0 if successes == n else float(beta.ppf(1 - alpha / 2, successes + 1, n - successes))
    return lower, upper


def compare_archived(
    generated_predictions: pd.DataFrame,
    generated_metrics: pd.DataFrame,
    reference_dir: Path,
    tolerance: float,
) -> dict[str, Any]:
    archived_predictions = pd.read_csv(reference_dir / "held_out_predictions.csv")
    archived_metrics = pd.read_csv(reference_dir / "held_out_test_metrics.csv")
    if list(generated_predictions["record_id"]) != list(archived_predictions["record_id"]):
        raise ValueError("Archived and regenerated held-out record orders differ")
    prediction_columns = [f"{target}_pred" for target in TARGETS]
    prediction_difference = np.max(
        np.abs(
            generated_predictions[prediction_columns].to_numpy(float)
            - archived_predictions[prediction_columns].to_numpy(float)
        )
    )
    aligned = generated_metrics.set_index("target").loc[TARGETS]
    archived = archived_metrics.set_index("target").loc[TARGETS]
    metric_columns = ["r2", "rmse", "mae"]
    metric_difference = np.max(
        np.abs(aligned[metric_columns].to_numpy(float) - archived[metric_columns].to_numpy(float))
    )
    return {
        "reference_directory": str(reference_dir),
        "prediction_columns_compared": prediction_columns,
        "metric_columns_compared": metric_columns,
        "maximum_absolute_prediction_difference": float(prediction_difference),
        "maximum_absolute_metric_difference": float(metric_difference),
        "tolerance": tolerance,
        "pass": bool(prediction_difference <= tolerance and metric_difference <= tolerance),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/held_out_test.csv"))
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/SAF_Predict_public_model_bundle_v1.joblib"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("run_outputs/heldout"))
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=Path("validation"),
        help="Directory containing the archived v1.0.0 prediction and metric tables",
    )
    parser.add_argument("--tolerance", type=float, default=1e-9)
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = ["record_id", "molecule_name", "molecular_formula", "partition", *INPUTS, *TARGETS]
    missing = [column for column in required if column not in data]
    if missing:
        raise ValueError(f"Held-out CSV is missing columns: {missing}")
    if len(data) != 10 or set(data["partition"]) != {"internal_heldout_test"}:
        raise ValueError("Expected exactly 10 records labelled internal_heldout_test")
    numeric = data[[*INPUTS, *TARGETS]].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(float)).all():
        raise ValueError("Input or target matrix contains non-finite values")

    bundle = joblib.load(args.model)
    x = numeric[INPUTS].to_numpy(float)
    predictions = predict(bundle, x)
    domain = ood(bundle, x)

    wide_rows = []
    long_rows = []
    for index, row in data.iterrows():
        wide = {
            "record_id": row["record_id"],
            "molecule": row["molecule_name"],
            "molecular_formula": row["molecular_formula"],
            **{code: float(row[code]) for code in INPUTS},
            "ood_distance": domain[index]["distance"],
            "ood_distance_percentile": domain[index]["percentile"],
            "nearest_development_record": domain[index]["nearest_record"],
        }
        for target in TARGETS:
            actual = float(row[target])
            point = float(predictions[target][index])
            interval = bundle["intervals"][target]
            lower90 = point - float(interval["half_width_90"])
            upper90 = point + float(interval["half_width_90"])
            lower95 = point - float(interval["half_width_95"])
            upper95 = point + float(interval["half_width_95"])
            wide.update(
                {
                    f"{target}_pred": point,
                    f"{target}_lower90": lower90,
                    f"{target}_upper90": upper90,
                    f"{target}_lower95": lower95,
                    f"{target}_upper95": upper95,
                }
            )
            long_rows.append(
                {
                    "record_id": row["record_id"],
                    "molecule": row["molecule_name"],
                    "molecular_formula": row["molecular_formula"],
                    "target": target,
                    "property": OUTPUT_META[target][0],
                    "unit": OUTPUT_META[target][1],
                    "actual": actual,
                    "predicted": point,
                    "residual_actual_minus_predicted": actual - point,
                    "absolute_error": abs(actual - point),
                    "lower90": lower90,
                    "upper90": upper90,
                    "covered90": lower90 <= actual <= upper90,
                    "lower95": lower95,
                    "upper95": upper95,
                    "covered95": lower95 <= actual <= upper95,
                    "ood_distance": domain[index]["distance"],
                    "ood_distance_percentile": domain[index]["percentile"],
                    "nearest_development_record": domain[index]["nearest_record"],
                }
            )
        wide_rows.append(wide)

    wide = pd.DataFrame(wide_rows)
    long = pd.DataFrame(long_rows)
    metric_rows = []
    for target in TARGETS:
        block = long[long["target"] == target]
        actual = block["actual"].to_numpy(float)
        predicted = block["predicted"].to_numpy(float)
        covered90 = int(block["covered90"].sum())
        covered95 = int(block["covered95"].sum())
        ci90 = clopper_pearson(covered90, len(block))
        ci95 = clopper_pearson(covered95, len(block))
        metric_rows.append(
            {
                "target": target,
                "property": OUTPUT_META[target][0],
                "unit": OUTPUT_META[target][1],
                "n": len(block),
                "r2": float(r2_score(actual, predicted)),
                "rmse": float(math.sqrt(mean_squared_error(actual, predicted))),
                "mae": float(mean_absolute_error(actual, predicted)),
                "median_absolute_error": float(median_absolute_error(actual, predicted)),
                "coverage90_count": covered90,
                "coverage90_fraction": covered90 / len(block),
                "coverage90_exact95_lower": ci90[0],
                "coverage90_exact95_upper": ci90[1],
                "coverage95_count": covered95,
                "coverage95_fraction": covered95 / len(block),
                "coverage95_exact95_lower": ci95[0],
                "coverage95_exact95_upper": ci95[1],
                "interval90_width": 2 * bundle["intervals"][target]["half_width_90"],
                "interval95_width": 2 * bundle["intervals"][target]["half_width_95"],
            }
        )
    metrics = pd.DataFrame(metric_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "held_out_predictions.csv"
    long_path = args.output_dir / "held_out_predictions_long.csv"
    metrics_path = args.output_dir / "held_out_metrics.csv"
    wide.to_csv(predictions_path, index=False, lineterminator="\n")
    long.to_csv(long_path, index=False, lineterminator="\n")
    metrics.to_csv(metrics_path, index=False, lineterminator="\n")

    comparison = compare_archived(wide, metrics, args.reference_dir, args.tolerance)
    y02_exact = np.allclose(
        wide["Y02_pred"].to_numpy(float),
        wide["Y01_pred"].to_numpy(float) * wide["Y03_pred"].to_numpy(float),
        rtol=0,
        atol=1e-12,
    )
    report = {
        "status": "pass" if comparison["pass"] and y02_exact else "fail",
        "scope": "deterministic reproduction of the archived v1.0.0 internal held-out evaluation",
        "test_set": "structurally stratified internal held-out test set",
        "independent_external_validation": False,
        "records": len(data),
        "unique_formulae": int(data["molecular_formula"].nunique()),
        "model_sha256": sha256(args.model),
        "data_sha256": sha256(args.data),
        "Y02_exactly_derived_from_predicted_Y01_and_Y03": bool(y02_exact),
        "archived_result_comparison": comparison,
        "metrics": metric_rows,
        "Y04_limitation": (
            "Y04 performed poorly on the internal holdout (R² = −0.021; MAE = 25.2 °C; n = 10). "
            "Do not use Y04 alone to rank or select compounds; experimental confirmation is required."
        ),
    }
    report_path = args.output_dir / "held_out_reproduction_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
