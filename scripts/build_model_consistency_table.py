#!/usr/bin/env python3
"""Compare XGBoost reference-model and frozen deployment-model OOF predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


TARGETS = [f"Y{i:02d}" for i in range(1, 8)]


def grouped_bootstrap(
    aligned: pd.DataFrame,
    deployment_model: str,
    *,
    iterations: int,
    seed: int,
) -> dict[str, float | int]:
    groups = list(dict.fromkeys(aligned["molecular_formula"].astype(str)))
    blocks = {
        group: aligned.loc[aligned["molecular_formula"].astype(str).eq(group)]
        for group in groups
    }
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    correlations: list[float] = []
    for _ in range(iterations):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        draw = pd.concat([blocks[group] for group in sampled], ignore_index=True)
        truth = draw["y_true"].to_numpy(float)
        reference = draw["XGBoost"].to_numpy(float)
        deployed = draw[deployment_model].to_numpy(float)
        if len(truth) >= 2 and float(np.var(truth)) > 0:
            deltas.append(float(r2_score(truth, deployed) - r2_score(truth, reference)))
        if float(np.std(reference)) > 0 and float(np.std(deployed)) > 0:
            correlations.append(float(np.corrcoef(reference, deployed)[0, 1]))
    delta_array = np.asarray(deltas, dtype=float)
    correlation_array = np.asarray(correlations, dtype=float)
    return {
        "bootstrap_iterations": iterations,
        "bootstrap_formula_groups": len(groups),
        "delta_r2_ci95_lower": float(np.quantile(delta_array, 0.025)),
        "delta_r2_ci95_upper": float(np.quantile(delta_array, 0.975)),
        "bootstrap_probability_delta_r2_gt_zero": float(np.mean(delta_array > 0)),
        "prediction_r_ci95_lower": float(np.quantile(correlation_array, 0.025)),
        "prediction_r_ci95_upper": float(np.quantile(correlation_array, 0.975)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", type=Path, default=Path("artifacts/cv/oof_predictions.csv"))
    parser.add_argument("--config", type=Path, default=Path("training/config.json"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/cv/xgboost_reference_vs_deployment_consistency.csv"),
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args()

    oof = pd.read_csv(args.oof)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for target_index, target in enumerate(TARGETS):
        block = oof.loc[oof["target"].eq(target)].copy()
        deployment_model = str(config["deployment_models"][target])
        required_models = {"XGBoost", deployment_model}
        if not required_models.issubset(set(block["model"])):
            raise ValueError(f"Missing required OOF predictions for {target}: {required_models}")
        truth = (
            block.loc[block["model"].eq("XGBoost"), ["record_id", "molecular_formula", "y_true"]]
            .drop_duplicates("record_id")
            .set_index("record_id")
        )
        predictions = block.pivot(index="record_id", columns="model", values="y_pred")
        aligned = truth.join(predictions, how="inner").reset_index()
        if len(aligned) != int(block.loc[block["model"].eq("XGBoost"), "record_id"].nunique()):
            raise AssertionError(f"OOF alignment failed for {target}")
        reference = aligned["XGBoost"].to_numpy(float)
        deployed = aligned[deployment_model].to_numpy(float)
        y_true = aligned["y_true"].to_numpy(float)
        prediction_r = 1.0 if deployment_model == "XGBoost" else float(np.corrcoef(reference, deployed)[0, 1])
        delta_r2 = float(r2_score(y_true, deployed) - r2_score(y_true, reference))
        row: dict[str, object] = {
            "target": target,
            "deployment_model": deployment_model,
            "n": len(aligned),
            "formula_groups": int(aligned["molecular_formula"].nunique()),
            "prediction_pearson_r_xgboost_vs_deployment": prediction_r,
            "xgboost_oof_r2": float(r2_score(y_true, reference)),
            "deployment_oof_r2": float(r2_score(y_true, deployed)),
            "delta_r2_deployment_minus_xgboost": delta_r2,
        }
        if deployment_model == "XGBoost":
            row.update(
                {
                    "bootstrap_iterations": args.bootstrap_iterations,
                    "bootstrap_formula_groups": int(aligned["molecular_formula"].nunique()),
                    "delta_r2_ci95_lower": 0.0,
                    "delta_r2_ci95_upper": 0.0,
                    "bootstrap_probability_delta_r2_gt_zero": 0.0,
                    "prediction_r_ci95_lower": 1.0,
                    "prediction_r_ci95_upper": 1.0,
                }
            )
        else:
            row.update(
                grouped_bootstrap(
                    aligned,
                    deployment_model,
                    iterations=args.bootstrap_iterations,
                    seed=args.seed + target_index,
                )
            )
        rows.append(row)

    output = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False, encoding="utf-8-sig", lineterminator="\n")
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
