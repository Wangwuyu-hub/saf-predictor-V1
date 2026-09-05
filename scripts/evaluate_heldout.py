#!/usr/bin/env python3
"""Evaluate the frozen SAF-Predict model on the internal held-out test set."""

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


def grouped_bootstrap_intervals(
    block: pd.DataFrame,
    *,
    iterations: int = 10_000,
    seed: int = 20260905,
) -> dict[str, float | int]:
    """Cluster bootstrap R2/RMSE/MAE by molecular-formula group."""
    groups = list(dict.fromkeys(block["molecular_formula"].astype(str)))
    # Each bootstrap draw repeats whole formula groups.  Precompute the
    # additive sufficient statistics once instead of rebuilding a DataFrame
    # thousands of times; this preserves the cluster-bootstrap definition and
    # makes the release audit practical on modest computers.
    group_stats = []
    formula = block["molecular_formula"].astype(str)
    for group in groups:
        group_block = block.loc[formula.eq(group)]
        actual = group_block["actual"].to_numpy(float)
        predicted = group_block["predicted"].to_numpy(float)
        group_stats.append(
            (
                float(len(actual)),
                float(actual.sum()),
                float(np.square(actual).sum()),
                float(np.square(actual - predicted).sum()),
                float(np.abs(actual - predicted).sum()),
            )
        )
    stats = np.asarray(group_stats, dtype=float)
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {"r2": [], "rmse": [], "mae": []}
    for _ in range(iterations):
        sampled_indices = rng.integers(0, len(groups), size=len(groups))
        n, sum_y, sum_y2, sum_squared_error, sum_absolute_error = stats[
            sampled_indices
        ].sum(axis=0)
        total_sum_squares = sum_y2 - (sum_y * sum_y / n)
        if n >= 2 and total_sum_squares > 0:
            values["r2"].append(float(1.0 - sum_squared_error / total_sum_squares))
        values["rmse"].append(float(math.sqrt(sum_squared_error / n)))
        values["mae"].append(float(sum_absolute_error / n))

    result: dict[str, float | int] = {
        "bootstrap_iterations": iterations,
        "bootstrap_formula_groups": len(groups),
    }
    for metric, samples in values.items():
        finite = np.asarray(samples, dtype=float)
        finite = finite[np.isfinite(finite)]
        result[f"{metric}_group_bootstrap_valid_draws"] = int(len(finite))
        result[f"{metric}_group_bootstrap_ci95_lower"] = float(np.quantile(finite, 0.025))
        result[f"{metric}_group_bootstrap_ci95_upper"] = float(np.quantile(finite, 0.975))
    return result


def summarize_block(
    block: pd.DataFrame,
    target: str,
    bundle: dict[str, Any],
    *,
    bootstrap_seed: int,
    development_iqr: float,
) -> dict[str, Any]:
    actual = block["actual"].to_numpy(float)
    predicted = block["predicted"].to_numpy(float)
    covered90 = int(block["covered90"].sum())
    covered95 = int(block["covered95"].sum())
    ci90 = clopper_pearson(covered90, len(block))
    ci95 = clopper_pearson(covered95, len(block))
    row: dict[str, Any] = {
        "target": target,
        "property": OUTPUT_META[target][0],
        "unit": OUTPUT_META[target][1],
        "n": len(block),
        "formula_groups": int(block["molecular_formula"].nunique()),
        "r2": float(r2_score(actual, predicted)) if len(actual) >= 2 and float(np.var(actual)) > 0 else np.nan,
        "rmse": float(math.sqrt(mean_squared_error(actual, predicted))),
        "mae": float(mean_absolute_error(actual, predicted)),
        "median_absolute_error": float(median_absolute_error(actual, predicted)),
        "mean_signed_error_actual_minus_predicted": float(np.mean(actual - predicted)),
        "normalized_mae_over_development_iqr": (
            float(mean_absolute_error(actual, predicted) / development_iqr)
            if development_iqr > 0
            else np.nan
        ),
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
    row.update(grouped_bootstrap_intervals(block, seed=bootstrap_seed))
    return row


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
        "--development",
        type=Path,
        default=Path("data/development.csv"),
        help="Development table used only to normalize MAE by each target's IQR.",
    )
    parser.add_argument(
        "--provenance",
        type=Path,
        default=Path("data/provenance_long.csv"),
        help="Long-form provenance table used to annotate each held-out property label.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/SAF_Predict_public_model_bundle_v1.joblib"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("run_outputs/heldout"))
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=None,
        help="Optional directory containing frozen v1.1.0 outputs for regression QA",
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
    if not np.isfinite(numeric[INPUTS].to_numpy(float)).all():
        raise ValueError("Input matrix contains non-finite values")
    complete_targets = [target for target in TARGETS if target != "Y02"]
    if not np.isfinite(numeric[complete_targets].to_numpy(float)).all():
        raise ValueError("Y01 and Y03-Y07 must be complete and finite")

    bundle = joblib.load(args.model)
    development = pd.read_csv(args.development)
    development_iqrs = {
        target: float(
            development[target].dropna().quantile(0.75)
            - development[target].dropna().quantile(0.25)
        )
        for target in TARGETS
    }
    provenance = pd.read_csv(args.provenance)
    provenance = provenance.loc[
        provenance["partition"].eq("internal_heldout_test"),
        [
            "record_id",
            "property_code",
            "value_origin_class",
            "high_confidence_source_subset",
            "primary_reference_id",
            "source_category",
            "conditions_status",
            "quality_grade",
        ],
    ].copy()
    if provenance.duplicated(["record_id", "property_code"]).any():
        raise ValueError("Held-out provenance contains duplicate record-property rows")
    provenance_lookup = provenance.set_index(["record_id", "property_code"]).to_dict("index")
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
            actual = float(row[target]) if pd.notna(row[target]) else np.nan
            point = float(predictions[target][index])
            interval = bundle["intervals"][target]
            provenance_row = provenance_lookup.get((str(row["record_id"]), target))
            if provenance_row is None:
                raise ValueError(f"Missing provenance row for {row['record_id']} / {target}")
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
                    "label_available": bool(np.isfinite(actual)),
                    "value_origin_class": provenance_row["value_origin_class"],
                    "high_confidence_source_subset": bool(
                        provenance_row["high_confidence_source_subset"]
                    ),
                    "primary_reference_id": provenance_row["primary_reference_id"],
                    "source_category": provenance_row["source_category"],
                    "conditions_status": provenance_row["conditions_status"],
                    "quality_grade": provenance_row["quality_grade"],
                    "actual": actual,
                    "predicted": point,
                    "residual_actual_minus_predicted": actual - point if np.isfinite(actual) else np.nan,
                    "absolute_error": abs(actual - point) if np.isfinite(actual) else np.nan,
                    "lower90": lower90,
                    "upper90": upper90,
                    "covered90": lower90 <= actual <= upper90 if np.isfinite(actual) else np.nan,
                    "lower95": lower95,
                    "upper95": upper95,
                    "covered95": lower95 <= actual <= upper95 if np.isfinite(actual) else np.nan,
                    "ood_distance": domain[index]["distance"],
                    "ood_distance_percentile": domain[index]["percentile"],
                    "nearest_development_record": domain[index]["nearest_record"],
                }
            )
        wide_rows.append(wide)

    wide = pd.DataFrame(wide_rows)
    long = pd.DataFrame(long_rows)
    metric_rows = []
    for target_index, target in enumerate(TARGETS):
        block = long[(long["target"] == target) & long["label_available"]].copy()
        metric_rows.append(
            summarize_block(
                block,
                target,
                bundle,
                bootstrap_seed=20260905 + target_index,
                development_iqr=development_iqrs[target],
            )
        )
    metrics = pd.DataFrame(metric_rows)

    source_supported_rows = []
    for target_index, target in enumerate(TARGETS):
        block = long[
            (long["target"] == target)
            & long["label_available"]
            & long["high_confidence_source_subset"]
        ].copy()
        if block.empty:
            source_supported_rows.append(
                {
                    "target": target,
                    "property": OUTPUT_META[target][0],
                    "unit": OUTPUT_META[target][1],
                    "n": 0,
                    "formula_groups": 0,
                    "estimable": False,
                    "reason": "No held-out labels are flagged as high-confidence source-supported records.",
                }
            )
            continue
        result = summarize_block(
            block,
            target,
            bundle,
            bootstrap_seed=20261905 + target_index,
            development_iqr=development_iqrs[target],
        )
        result["estimable"] = len(block) >= 2 and float(np.var(block["actual"].to_numpy(float))) > 0
        result["reason"] = "" if result["estimable"] else "Too few distinct reference values for R2."
        source_supported_rows.append(result)
    source_supported_metrics = pd.DataFrame(source_supported_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "held_out_predictions.csv"
    long_path = args.output_dir / "held_out_test_predictions_long.csv"
    metrics_path = args.output_dir / "held_out_test_metrics.csv"
    source_metrics_path = args.output_dir / "held_out_source_supported_metrics.csv"
    wide.to_csv(predictions_path, index=False, lineterminator="\n")
    long.to_csv(long_path, index=False, lineterminator="\n")
    metrics.to_csv(metrics_path, index=False, lineterminator="\n")
    source_supported_metrics.to_csv(source_metrics_path, index=False, lineterminator="\n")

    comparison = (
        compare_archived(wide, metrics, args.reference_dir, args.tolerance)
        if args.reference_dir is not None
        else {"performed": False, "pass": True}
    )
    y02_difference = np.abs(
        wide["Y02_pred"].to_numpy(float)
        - wide["Y01_pred"].to_numpy(float) * wide["Y03_pred"].to_numpy(float)
    )
    y02_direct_canary = bool(np.max(y02_difference) > 1e-6)
    y04_metric = metrics.set_index("target").loc["Y04"]
    report = {
        "status": "pass" if comparison["pass"] and y02_direct_canary else "fail",
        "scope": "v1.1.0 frozen-model evaluation on the internal held-out set",
        "test_set": "structurally stratified internal held-out test set",
        "independent_external_validation": False,
        "records": len(data),
        "unique_formulae": int(data["molecular_formula"].nunique()),
        "model_sha256": sha256(args.model),
        "data_sha256": sha256(args.data),
        "Y02_available_reference_labels": int(data["Y02"].notna().sum()),
        "Y02_separately_modelled_canary": y02_direct_canary,
        "maximum_abs_Y02_prediction_minus_Y01_times_Y03": float(np.max(y02_difference)),
        "archived_result_comparison": comparison,
        "metrics": metric_rows,
        "source_supported_metrics": source_supported_rows,
        "uncertainty_note": (
            "R2, RMSE and MAE confidence intervals use 10,000 deterministic molecular-formula-group "
            "bootstrap draws. Coverage confidence intervals are exact binomial intervals; all held-out "
            "estimates remain imprecise because the test set contains only 10 records."
        ),
        "Y04_limitation": (
            f"Y04 internal-holdout performance was R² = {y04_metric['r2']:.3f}; "
            f"MAE = {y04_metric['mae']:.1f} °C; n = {int(y04_metric['n'])}. "
            "Do not use Y04 alone to rank or select compounds; experimental confirmation is required."
        ),
    }
    report_path = args.output_dir / "held_out_reproduction_check.json"
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
