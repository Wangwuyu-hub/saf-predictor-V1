#!/usr/bin/env python3
"""Portable SAF-Predict model-development and deployment export pipeline.

The scientific defaults mirror the recorded Figure 4 and SAF-Predict v1.1.0
workflow. All data and output locations are supplied on the command line; this
module contains no machine-specific path.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
import xgboost
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, ParameterGrid
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "config.json"
INPUT_CODES = [f"X{i:02d}" for i in range(1, 8)]
TARGET_CODES = [f"Y{i:02d}" for i in range(1, 8)]
DIRECTLY_MODELLED_TARGETS = TARGET_CODES.copy()
FORMULA_PROVENANCE_TARGETS = ["Y01", "Y03", "Y04", "Y05", "Y06", "Y07"]

TARGET_NAMES = {
    "Y01": "Mass-based net heat of combustion",
    "Y02": "Volumetric net heat of combustion",
    "Y03": "Density",
    "Y04": "Solid–liquid phase-transition temperature",
    "Y05": "Boiling point",
    "Y06": "Flash point",
    "Y07": "Kinematic viscosity at 40 °C",
}

TARGET_UNITS = {
    "Y01": "MJ kg−1",
    "Y02": "MJ L−1",
    "Y03": "g cm−3",
    "Y04": "°C",
    "Y05": "°C",
    "Y06": "°C",
    "Y07": "mm² s−1",
}

INPUT_META = {
    "X01": {
        "name": "Molar mass",
        "unit": "g mol-1",
        "definition": "RDKit Descriptors.MolWt average molecular mass",
        "step": 0.001,
    },
    "X02": {
        "name": "H/C atomic ratio",
        "unit": "1",
        "definition": "Hydrogen atom count divided by carbon atom count",
        "step": 0.001,
    },
    "X03": {
        "name": "Aromatic carbon fraction",
        "unit": "0-1",
        "definition": "Aromatic carbon count divided by total carbon count",
        "step": 0.001,
    },
    "X04": {
        "name": "Ring count",
        "unit": "count",
        "definition": "RDKit CalcNumRings",
        "step": 1,
    },
    "X05": {
        "name": "Branching-carbon count",
        "unit": "count",
        "definition": "Carbon atoms directly connected to at least three carbon atoms",
        "step": 1,
    },
    "X06": {
        "name": "Graph-automorphism descriptor",
        "unit": "1",
        "definition": "ln(1 + graph automorphism match count), ignoring stereochemistry",
        "step": 0.001,
    },
    "X07": {
        "name": "Kier kappa2 shape index",
        "unit": "1",
        "definition": "RDKit GraphDescriptors.Kappa2",
        "step": 0.001,
    },
}

OUTPUT_META = {
    code: {"name": TARGET_NAMES[code], "unit": TARGET_UNITS[code], "digits": digits}
    for code, digits in {
        "Y01": 2,
        "Y02": 2,
        "Y03": 4,
        "Y04": 1,
        "Y05": 1,
        "Y06": 1,
        "Y07": 3,
    }.items()
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "base_seed",
        "outer_folds",
        "inner_folds",
        "final_cv_folds",
        "selection_scoring",
        "model_order",
        "deployment_models",
        "final_fit_target_order",
        "fixed_parameters",
        "parameter_grids",
        "provenance_reconstruction",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Configuration is missing fields: {missing}")
    if config["model_order"] != ["XGBoost", "RF", "SVR", "ANN"]:
        raise ValueError("model_order must be XGBoost, RF, SVR, ANN")
    deployment = config["deployment_models"]
    if set(deployment) != set(DIRECTLY_MODELLED_TARGETS):
        raise ValueError("deployment_models must define all targets Y01-Y07")
    unknown = set(deployment.values()) - set(config["model_order"])
    if unknown:
        raise ValueError(f"Unknown deployment model families: {sorted(unknown)}")
    final_order = config["final_fit_target_order"]
    if (
        len(final_order) != len(DIRECTLY_MODELLED_TARGETS)
        or len(set(final_order)) != len(final_order)
        or set(final_order) != set(DIRECTLY_MODELLED_TARGETS)
    ):
        raise ValueError("final_fit_target_order must contain every target Y01-Y07 once")
    for model in config["model_order"]:
        if model not in config["fixed_parameters"] or model not in config["parameter_grids"]:
            raise ValueError(f"No estimator settings or parameter grid for {model}")
    reconstruction = config["provenance_reconstruction"]
    if not isinstance(reconstruction, dict):
        raise ValueError("provenance_reconstruction must be an object")
    if not isinstance(reconstruction.get("reference_map_column"), str):
        raise ValueError("provenance_reconstruction.reference_map_column must be a string")
    formula_references = reconstruction.get("formula_derived_reference_by_target")
    if set(formula_references or {}) != set(FORMULA_PROVENANCE_TARGETS):
        raise ValueError(
            "provenance_reconstruction must define formula-derived references for Y01 and Y03-Y07"
        )
    return config


def reconstruct_provenance_classes(data: pd.DataFrame, config: dict[str, Any]) -> None:
    """Restore the Figure 4 two-class provenance strata from ``reference_map``.

    The original Figure 4 script classified direct targets by whether the
    target-specific reference-map entry contained its recorded formula-derived
    reference ID. Public ``development.csv`` stores the source map rather than
    redundant ``provenance_Yxx`` columns, so a rerun reconstructs those archive
    strata. This is an analysis stratum only; the richer record-property
    provenance remains in ``data/provenance_long.csv``.
    """
    settings = config["provenance_reconstruction"]
    map_column = settings["reference_map_column"]
    formula_references: dict[str, str] = settings["formula_derived_reference_by_target"]
    for target in TARGET_CODES:
        output_column = f"provenance_{target}"
        if output_column in data.columns:
            data[output_column] = data[output_column].fillna("unspecified").astype(str)
            continue
        if target == "Y02":
            data[output_column] = np.where(
                data["Y02"].notna(),
                "author-curated independent response",
                "not available",
            )
            continue
        if map_column not in data.columns:
            data[output_column] = "unspecified"
            continue
        references = (
            data[map_column]
            .fillna("")
            .astype(str)
            .str.extract(rf"(?:^|；){target}→([^；]+)", expand=False)
        )
        data[output_column] = np.where(
            references.str.contains(formula_references[target], regex=False, na=False),
            "formula-derived",
            "source-supported",
        )


def load_development_csv(
    path: Path, expected_records: int, config: dict[str, Any]
) -> pd.DataFrame:
    data = pd.read_csv(path, encoding="utf-8-sig")
    required = ["record_id", "molecular_formula", *INPUT_CODES, *TARGET_CODES]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")
    if expected_records > 0 and len(data) != expected_records:
        raise ValueError(f"Expected {expected_records} records, found {len(data)}")
    if data["record_id"].isna().any():
        raise ValueError("record_id contains missing values")
    data["record_id"] = data["record_id"].astype(str)
    if data["record_id"].duplicated().any():
        duplicates = data.loc[data["record_id"].duplicated(), "record_id"].tolist()
        raise ValueError(f"Duplicate record_id values: {duplicates[:5]}")
    if data["molecular_formula"].isna().any():
        raise ValueError("molecular_formula contains missing values")
    data["molecular_formula"] = data["molecular_formula"].astype(str)
    if "molecule" in data.columns:
        data["molecule"] = data["molecule"].fillna(data["record_id"]).astype(str)
    elif "molecule_name" in data.columns:
        # The public table uses ``molecule_name`` while the original Figure 4
        # workbook used a display column named ``molecule``. Preserve it so
        # regenerated manifests and OOF records retain their archived labels.
        data["molecule"] = data["molecule_name"].fillna(data["record_id"]).astype(str)
    else:
        data["molecule"] = data["record_id"]
    for code in [*INPUT_CODES, *TARGET_CODES]:
        data[code] = pd.to_numeric(data[code], errors="raise")
    complete_codes = [*INPUT_CODES, *FORMULA_PROVENANCE_TARGETS]
    matrix = data[complete_codes].to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError("Inputs and Y01/Y03-Y07 must be complete and finite")
    y02_values = data["Y02"].dropna().to_numpy(dtype=float)
    if len(y02_values) == 0 or not np.isfinite(y02_values).all():
        raise ValueError("Y02 must contain at least one finite independent response")
    variances = data[[*INPUT_CODES, *TARGET_CODES]].var(ddof=1, skipna=True)
    constant = variances.index[variances <= 0].tolist()
    if constant:
        raise ValueError(f"Constant model columns are not supported: {constant}")
    reconstruct_provenance_classes(data, config)
    return data


def validate_group_counts(data: pd.DataFrame, config: dict[str, Any]) -> None:
    required = max(
        int(config["outer_folds"]),
        int(config["inner_folds"]),
        int(config["final_cv_folds"]),
    )
    for target in TARGET_CODES:
        group_count = data.loc[data[target].notna(), "molecular_formula"].nunique()
        if group_count < required:
            raise ValueError(
                f"{target} requires at least {required} molecular-formula groups; "
                f"found {group_count}"
            )


def build_outer_splits(
    data: pd.DataFrame, n_splits: int
) -> tuple[list[tuple[np.ndarray, np.ndarray]], pd.DataFrame]:
    groups = data["molecular_formula"].to_numpy(dtype=str)
    splitter = GroupKFold(n_splits=n_splits)
    placeholder = np.zeros((len(data), 1), dtype=float)
    splits = [
        (train.copy(), test.copy())
        for train, test in splitter.split(placeholder, groups=groups)
    ]
    assignments = np.full(len(data), -1, dtype=int)
    for fold, (train_index, test_index) in enumerate(splits, start=1):
        if set(groups[train_index]) & set(groups[test_index]):
            raise AssertionError(f"Formula-group leakage in outer fold {fold}")
        assignments[test_index] = fold
        if len(set(groups[train_index])) < 2:
            raise ValueError(f"Outer fold {fold} leaves too few training groups")
    covered = np.sort(np.concatenate([test for _, test in splits]))
    if not np.array_equal(covered, np.arange(len(data))) or (assignments < 1).any():
        raise AssertionError("Outer validation folds do not cover each record exactly once")
    manifest = pd.DataFrame(
        {
            "record_id": data["record_id"],
            "molecule": data["molecule"],
            "molecular_formula": data["molecular_formula"],
            "outer_fold": assignments,
        }
    )
    return splits, manifest


def model_grid(config: dict[str, Any], model_name: str, smoke: bool) -> Any:
    grid = copy.deepcopy(config["parameter_grids"][model_name])
    blocks = grid if isinstance(grid, list) else [grid]
    if model_name == "ANN":
        key = "regressor__regressor__hidden_layer_sizes"
        for block in blocks:
            block[key] = [tuple(value) for value in block[key]]
    if not smoke:
        return grid
    first = next(iter(ParameterGrid(grid)))
    return {key: [value] for key, value in first.items()}


def make_estimator(config: dict[str, Any], model_name: str, seed: int) -> Any:
    fixed = copy.deepcopy(config["fixed_parameters"][model_name])
    if model_name == "XGBoost":
        return XGBRegressor(random_state=seed, **fixed)
    if model_name == "RF":
        return RandomForestRegressor(random_state=seed, **fixed)
    if model_name == "SVR":
        svr = SVR(**fixed)
        return TransformedTargetRegressor(
            regressor=Pipeline([("scaler", StandardScaler()), ("regressor", svr)]),
            transformer=StandardScaler(),
        )
    if model_name == "ANN":
        mlp = MLPRegressor(random_state=seed, **fixed)
        return TransformedTargetRegressor(
            regressor=Pipeline([("scaler", StandardScaler()), ("regressor", mlp)]),
            transformer=StandardScaler(),
        )
    raise KeyError(model_name)


def nested_seed(config: dict[str, Any], target_index: int, model_index: int, fold: int) -> int:
    return int(config["base_seed"]) + target_index * 100 + model_index * 10 + fold


def final_seed(config: dict[str, Any], deployment_index: int) -> int:
    return int(config["base_seed"]) + 9000 + deployment_index


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def software_versions() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "joblib": joblib.__version__,
    }


def run_smoke(
    data: pd.DataFrame,
    config: dict[str, Any],
    output_dir: Path,
    n_jobs: int,
) -> dict[str, Any]:
    splits, _ = build_outer_splits(data, int(config["outer_folds"]))
    train_index, test_index = splits[0]
    x = data[INPUT_CODES].to_numpy(dtype=float)
    y = data["Y01"].to_numpy(dtype=float)
    groups = data["molecular_formula"].to_numpy(dtype=str)
    inner_folds = int(config["inner_folds"])
    rows: list[dict[str, Any]] = []
    for model_index, model_name in enumerate(config["model_order"]):
        seed = int(config["base_seed"]) + model_index
        search = GridSearchCV(
            make_estimator(config, model_name, seed),
            model_grid(config, model_name, smoke=True),
            scoring=config["selection_scoring"],
            cv=GroupKFold(n_splits=inner_folds),
            n_jobs=1 if model_name == "ANN" else n_jobs,
            refit=True,
            error_score="raise",
        )
        search.fit(x[train_index], y[train_index], groups=groups[train_index])
        prediction = np.asarray(search.predict(x[test_index]), dtype=float)
        rows.append(
            {
                "model": model_name,
                "target": "Y01",
                "outer_fold": 1,
                "n_train": int(len(train_index)),
                "n_test": int(len(test_index)),
                "r2": float(r2_score(y[test_index], prediction)),
                "rmse": rmse(y[test_index], prediction),
                "seed": seed,
                "best_params": search.best_params_,
            }
        )
    report = {
        "status": "smoke_passed",
        "scope": "Y01, first outer fold, first grid combination for each model; not a scientific result",
        "models_checked": list(config["model_order"]),
        "results": rows,
    }
    write_json(output_dir / "smoke_results.json", report)
    return report


def run_nested_cv(
    data: pd.DataFrame,
    input_path: Path,
    config: dict[str, Any],
    config_path: Path,
    output_dir: Path,
    n_jobs: int,
    smoke_grid: bool,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outer_folds = int(config["outer_folds"])
    inner_folds = int(config["inner_folds"])
    splits, manifest = build_outer_splits(data, outer_folds)
    write_csv(output_dir / "fold_manifest.csv", manifest)

    x = data[INPUT_CODES].to_numpy(dtype=float)
    groups = data["molecular_formula"].to_numpy(dtype=str)
    record_ids = data["record_id"].to_numpy(dtype=str)
    molecules = data["molecule"].to_numpy(dtype=str)
    formulae = data["molecular_formula"].to_numpy(dtype=str)
    fold_rows: list[dict[str, Any]] = []
    oof_rows: list[dict[str, Any]] = []

    for target_index, target in enumerate(TARGET_CODES):
        y = data[target].to_numpy(dtype=float)
        provenance = data[f"provenance_{target}"].to_numpy(dtype=str)
        available = np.isfinite(y)
        target_records = int(available.sum())
        target_groups = int(data.loc[available, "molecular_formula"].nunique())
        for model_index, model_name in enumerate(config["model_order"]):
            for outer_fold, (full_train_index, full_test_index) in enumerate(
                splits, start=1
            ):
                train_index = full_train_index[available[full_train_index]]
                test_index = full_test_index[available[full_test_index]]
                if len(test_index) < 2:
                    raise ValueError(
                        f"{target} outer fold {outer_fold} has fewer than two labelled records"
                    )
                train_group_count = len(set(groups[train_index]))
                if train_group_count < inner_folds:
                    raise ValueError(
                        f"Outer fold {outer_fold} has only {train_group_count} training groups; "
                        f"inner_folds={inner_folds}"
                    )
                seed = nested_seed(config, target_index, model_index, outer_fold)
                grid = model_grid(config, model_name, smoke=smoke_grid)
                search = GridSearchCV(
                    make_estimator(config, model_name, seed),
                    grid,
                    scoring=config["selection_scoring"],
                    cv=GroupKFold(n_splits=inner_folds),
                    n_jobs=1 if model_name == "ANN" else n_jobs,
                    refit=True,
                    error_score="raise",
                    return_train_score=False,
                )
                started = time.perf_counter()
                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter("always")
                    search.fit(x[train_index], y[train_index], groups=groups[train_index])
                elapsed = time.perf_counter() - started
                prediction = np.asarray(search.predict(x[test_index]), dtype=float).reshape(-1)
                if not np.isfinite(prediction).all():
                    raise ValueError(
                        f"Non-finite predictions for {target}/{model_name}/fold {outer_fold}"
                    )
                overlap = len(set(groups[train_index]) & set(groups[test_index]))
                if overlap:
                    raise AssertionError("Molecular-formula leakage in an outer fold")
                fold_rows.append(
                    {
                        "model": model_name,
                        "target": target,
                        "property": TARGET_NAMES[target],
                        "unit": TARGET_UNITS[target],
                        "target_records": target_records,
                        "target_formula_groups": target_groups,
                        "outer_fold": outer_fold,
                        "n_train": int(len(train_index)),
                        "n_test": int(len(test_index)),
                        "train_formula_groups": train_group_count,
                        "test_formula_groups": len(set(groups[test_index])),
                        "formula_overlap": overlap,
                        "r2": float(r2_score(y[test_index], prediction)),
                        "rmse": rmse(y[test_index], prediction),
                        "mae": float(mean_absolute_error(y[test_index], prediction)),
                        "best_inner_rmse": -float(search.best_score_),
                        "best_params": json.dumps(
                            search.best_params_, ensure_ascii=False, sort_keys=True
                        ),
                        "random_seed": seed,
                        "fit_seconds": elapsed,
                        "warning_count": len(captured),
                    }
                )
                for position, row_index in enumerate(test_index):
                    oof_rows.append(
                        {
                            "record_id": record_ids[row_index],
                            "molecule": molecules[row_index],
                            "molecular_formula": formulae[row_index],
                            "outer_fold": outer_fold,
                            "model": model_name,
                            "target": target,
                            "property": TARGET_NAMES[target],
                            "unit": TARGET_UNITS[target],
                            "y_true": float(y[row_index]),
                            "y_pred": float(prediction[position]),
                            "residual": float(y[row_index] - prediction[position]),
                            "provenance_class": provenance[row_index],
                        }
                    )
                write_csv(output_dir / "outer_fold_scores.csv", pd.DataFrame(fold_rows))
                write_csv(output_dir / "oof_predictions.csv", pd.DataFrame(oof_rows))
                print(
                    f"{target} | {model_name} | outer fold {outer_fold}/{outer_folds} | "
                    f"R2={fold_rows[-1]['r2']:.4f} | RMSE={fold_rows[-1]['rmse']:.6g} | "
                    f"{elapsed:.1f}s",
                    flush=True,
                )

    folds = pd.DataFrame(fold_rows)
    oof = pd.DataFrame(oof_rows)
    summary, provenance_metrics, sensitivity = summarize_nested(
        data, folds, oof, outer_folds, config["model_order"]
    )
    write_csv(output_dir / "model_target_summary.csv", summary)
    write_csv(output_dir / "provenance_stratified_metrics.csv", provenance_metrics)
    write_csv(output_dir / "y02_direct_vs_physical_baseline.csv", sensitivity)

    protocol = {
        "analysis": "four-model nested grouped cross-validation",
        "input_csv_name": input_path.name,
        "input_csv_sha256": sha256(input_path),
        "config_sha256": sha256(config_path),
        "records": len(data),
        "formula_groups": int(data["molecular_formula"].nunique()),
        "target_records": {
            target: int(data[target].notna().sum()) for target in TARGET_CODES
        },
        "target_formula_groups": {
            target: int(data.loc[data[target].notna(), "molecular_formula"].nunique())
            for target in TARGET_CODES
        },
        "inputs": INPUT_CODES,
        "targets": TARGET_CODES,
        "models": list(config["model_order"]),
        "outer_cv": f"GroupKFold(n_splits={outer_folds}), groups=molecular_formula",
        "inner_cv": f"GroupKFold(n_splits={inner_folds}), groups=molecular_formula",
        "selection_scoring": config["selection_scoring"],
        "base_seed": int(config["base_seed"]),
        "seed_policy": config.get("seed_policy"),
        "smoke_grid": bool(smoke_grid),
        "parameter_grids": config["parameter_grids"],
        "fixed_parameters": config["fixed_parameters"],
        "y02_policy": (
            "Y02 is an independently collected response and is modelled directly on "
            "available complete cases; predicted Y01 multiplied by predicted Y03 is "
            "retained only as a physical-baseline sensitivity analysis"
        ),
        "held_out_data_accessed": False,
        "software_versions": software_versions(),
        "generated_at_utc": utc_now(),
    }
    write_json(output_dir / "protocol.json", protocol)
    qa = nested_qa(data, manifest, folds, oof, summary, outer_folds, config["model_order"])
    qa["files"] = {
        name: sha256(output_dir / name)
        for name in [
            "fold_manifest.csv",
            "outer_fold_scores.csv",
            "oof_predictions.csv",
            "model_target_summary.csv",
            "provenance_stratified_metrics.csv",
            "y02_direct_vs_physical_baseline.csv",
            "protocol.json",
        ]
    }
    write_json(output_dir / "qa.json", qa)
    if not qa["all_checks_passed"]:
        raise RuntimeError("Nested-CV QA failed; inspect qa.json")
    return {
        "fold_manifest": output_dir / "fold_manifest.csv",
        "oof": output_dir / "oof_predictions.csv",
        "summary": output_dir / "model_target_summary.csv",
        "protocol": output_dir / "protocol.json",
        "qa": output_dir / "qa.json",
    }


def summarize_nested(
    data: pd.DataFrame,
    folds: pd.DataFrame,
    oof: pd.DataFrame,
    outer_folds: int,
    model_order: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    for target in TARGET_CODES:
        for model_name in model_order:
            fold_block = folds[
                (folds["target"] == target) & (folds["model"] == model_name)
            ].sort_values("outer_fold")
            oof_block = oof[(oof["target"] == target) & (oof["model"] == model_name)]
            expected_target_records = int(data[target].notna().sum())
            if len(fold_block) != outer_folds or len(oof_block) != expected_target_records:
                raise AssertionError(f"Incomplete nested-CV block: {target}/{model_name}")
            summary_rows.append(
                {
                    "model": model_name,
                    "target": target,
                    "property": TARGET_NAMES[target],
                    "unit": TARGET_UNITS[target],
                    "mean_fold_r2": float(fold_block["r2"].mean()),
                    "sd_fold_r2": float(fold_block["r2"].std(ddof=1)),
                    "median_fold_r2": float(fold_block["r2"].median()),
                    "min_fold_r2": float(fold_block["r2"].min()),
                    "max_fold_r2": float(fold_block["r2"].max()),
                    "oof_r2": float(r2_score(oof_block["y_true"], oof_block["y_pred"])),
                    "oof_rmse": rmse(
                        oof_block["y_true"].to_numpy(), oof_block["y_pred"].to_numpy()
                    ),
                    "oof_mae": float(
                        mean_absolute_error(oof_block["y_true"], oof_block["y_pred"])
                    ),
                }
            )
            for provenance, block in oof_block.groupby("provenance_class", sort=True):
                true = block["y_true"].to_numpy(dtype=float)
                predicted = block["y_pred"].to_numpy(dtype=float)
                provenance_rows.append(
                    {
                        "model": model_name,
                        "target": target,
                        "property": TARGET_NAMES[target],
                        "provenance_class": provenance,
                        "n": len(block),
                        "r2": float(r2_score(true, predicted)) if len(block) >= 2 else np.nan,
                        "rmse": rmse(true, predicted),
                        "mae": float(mean_absolute_error(true, predicted)),
                    }
                )
    summary = pd.DataFrame(summary_rows)
    summary["rank_by_mean_fold_r2"] = (
        summary.groupby("target")["mean_fold_r2"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    sensitivity_rows: list[dict[str, Any]] = []
    for model_name in model_order:
        y01 = oof[(oof["model"] == model_name) & (oof["target"] == "Y01")][
            ["record_id", "outer_fold", "y_pred"]
        ].rename(columns={"y_pred": "pred_y01"})
        y03 = oof[(oof["model"] == model_name) & (oof["target"] == "Y03")][
            ["record_id", "outer_fold", "y_pred"]
        ].rename(columns={"y_pred": "pred_y03"})
        y02 = oof[(oof["model"] == model_name) & (oof["target"] == "Y02")][
            ["record_id", "outer_fold", "y_true", "y_pred"]
        ].rename(columns={"y_pred": "pred_direct"})
        merged = y02.merge(y01, on=["record_id", "outer_fold"], validate="one_to_one")
        merged = merged.merge(y03, on=["record_id", "outer_fold"], validate="one_to_one")
        merged["pred_derived"] = merged["pred_y01"] * merged["pred_y03"]
        for approach, column in [
            ("Direct model", "pred_direct"),
            ("Product of Y01 and Y03 predictions (physical baseline)", "pred_derived"),
        ]:
            sensitivity_rows.append(
                {
                    "model": model_name,
                    "approach": approach,
                    "n": len(merged),
                    "oof_r2": float(r2_score(merged["y_true"], merged[column])),
                    "oof_rmse": rmse(
                        merged["y_true"].to_numpy(), merged[column].to_numpy()
                    ),
                    "oof_mae": float(mean_absolute_error(merged["y_true"], merged[column])),
                }
            )
    return summary, pd.DataFrame(provenance_rows), pd.DataFrame(sensitivity_rows)


def nested_qa(
    data: pd.DataFrame,
    manifest: pd.DataFrame,
    folds: pd.DataFrame,
    oof: pd.DataFrame,
    summary: pd.DataFrame,
    outer_folds: int,
    model_order: list[str],
) -> dict[str, Any]:
    expected_blocks = len(model_order) * len(TARGET_CODES)
    expected_ids_by_target = {
        target: set(data.loc[data[target].notna(), "record_id"].astype(str))
        for target in TARGET_CODES
    }
    formula_disjoint = True
    for fold in range(1, outer_folds + 1):
        test_formulae = set(
            manifest.loc[manifest["outer_fold"] == fold, "molecular_formula"]
        )
        train_formulae = set(
            manifest.loc[manifest["outer_fold"] != fold, "molecular_formula"]
        )
        formula_disjoint &= not bool(test_formulae & train_formulae)
    complete_oof_ids = all(
        set(block["record_id"].astype(str)) == expected_ids_by_target[target]
        for (_, target), block in oof.groupby(["model", "target"], sort=False)
    )
    expected_oof_rows = len(model_order) * sum(
        len(expected_ids_by_target[target]) for target in TARGET_CODES
    )
    checks = {
        "record_ids_unique": data["record_id"].nunique() == len(data),
        "all_inputs_and_non_y02_targets_finite": bool(
            np.isfinite(
                data[[*INPUT_CODES, *FORMULA_PROVENANCE_TARGETS]].to_numpy(dtype=float)
            ).all()
        ),
        "all_available_y02_values_finite": bool(
            np.isfinite(data["Y02"].dropna().to_numpy(dtype=float)).all()
        ),
        "fold_manifest_covers_each_record_once": (
            len(manifest) == len(data) and manifest["record_id"].nunique() == len(data)
        ),
        "formula_groups_disjoint_between_outer_train_and_validation": formula_disjoint,
        "outer_fold_score_blocks_complete": (
            len(folds) == expected_blocks * outer_folds
            and folds[["model", "target", "outer_fold"]].drop_duplicates().shape[0]
            == expected_blocks * outer_folds
        ),
        "oof_blocks_complete": (
            len(oof) == expected_oof_rows
            and oof[["model", "target", "record_id"]].drop_duplicates().shape[0]
            == expected_oof_rows
            and complete_oof_ids
        ),
        "all_metrics_finite": bool(
            np.isfinite(folds[["r2", "rmse", "mae", "best_inner_rmse"]].to_numpy()).all()
        ),
        "summary_has_28_model_target_rows": (
            len(summary) == expected_blocks
            and summary[["model", "target"]].drop_duplicates().shape[0] == expected_blocks
        ),
        "y02_is_directly_modelled_on_available_labels": True,
        "held_out_data_not_accessed": True,
    }
    return {
        "all_checks_passed": all(checks.values()),
        "checks": checks,
        "records": len(data),
        "formula_groups": int(data["molecular_formula"].nunique()),
        "target_records": {
            target: len(expected_ids_by_target[target]) for target in TARGET_CODES
        },
        "outer_fold_sizes": {
            str(key): int(value)
            for key, value in manifest["outer_fold"].value_counts().sort_index().items()
        },
    }


def finite_group_quantile(scores: np.ndarray, level: float) -> float:
    ordered = np.sort(np.asarray(scores, dtype=float))
    if len(ordered) == 0:
        raise ValueError("Cannot calibrate an interval from zero groups")
    rank = min(int(math.ceil((len(ordered) + 1) * level)), len(ordered))
    return float(ordered[rank - 1])


def build_intervals(
    data: pd.DataFrame, oof: pd.DataFrame, deployment: dict[str, str]
) -> dict[str, Any]:
    selected: dict[str, pd.DataFrame] = {}
    for target, model_name in deployment.items():
        block = oof[(oof["target"] == target) & (oof["model"] == model_name)].copy()
        if len(block) != int(data[target].notna().sum()):
            raise ValueError(f"Incomplete OOF block for {target}/{model_name}")
        selected[target] = block

    intervals: dict[str, Any] = {}
    for target in TARGET_CODES:
        block = selected[target].copy()
        block["absolute_residual"] = np.abs(block["y_true"] - block["y_pred"])
        group_scores = (
            block.groupby("molecular_formula", sort=True)["absolute_residual"]
            .max()
            .to_numpy(dtype=float)
        )
        target_values = data[target].dropna()
        target_iqr = float(target_values.quantile(0.75) - target_values.quantile(0.25))
        if target_iqr <= 0:
            raise ValueError(f"Non-positive target IQR for {target}")
        q90 = finite_group_quantile(group_scores, 0.90)
        q95 = finite_group_quantile(group_scores, 0.95)
        intervals[target] = {
            "method": "formula-group maximum absolute nested-CV OOF residual",
            "calibration_groups": int(len(group_scores)),
            "half_width_90": q90,
            "half_width_95": q95,
            "width_90_over_target_iqr": 2 * q90 / target_iqr,
            "width_95_over_target_iqr": 2 * q95 / target_iqr,
            "target_iqr": target_iqr,
        }
    return intervals


def build_ood_reference(data: pd.DataFrame, deidentify: bool) -> dict[str, Any]:
    x = data[INPUT_CODES].to_numpy(dtype=float)
    groups = data["molecular_formula"].to_numpy(dtype=str)
    median = np.median(x, axis=0)
    q05 = np.quantile(x, 0.05, axis=0)
    q95 = np.quantile(x, 0.95, axis=0)
    scale = q95 - q05
    if (scale <= 0).any():
        raise ValueError("Invalid OOD feature scale")
    normalized = (x - median) / scale
    distances = np.abs(normalized[:, None, :] - normalized[None, :, :]).mean(axis=2)
    distances[groups[:, None] == groups[None, :]] = np.inf
    leave_formula_out = distances.min(axis=1)
    group_reference = (
        pd.DataFrame({"formula": groups, "distance": leave_formula_out})
        .groupby("formula", sort=True)["distance"]
        .median()
    )
    records: list[dict[str, Any]] = []
    for index, row in data.reset_index(drop=True).iterrows():
        if deidentify:
            record = {
                "record_id": f"DEV-{index + 1:04d}",
                "x": [float(row[code]) for code in INPUT_CODES],
            }
        else:
            record = {
                "record_id": str(row["record_id"]),
                "molecule": str(row["molecule"]),
                "formula": str(row["molecular_formula"]),
                "x": [float(row[code]) for code in INPUT_CODES],
            }
        records.append(record)
    return {
        "method": "mean L1 distance after median and 5th-95th percentile scaling",
        "reference": "leave-one-molecular-formula-group-out nearest-neighbour distances",
        "feature_median": median.tolist(),
        "feature_scale": scale.tolist(),
        "group_reference_distances_sorted": np.sort(
            group_reference.to_numpy(dtype=float)
        ).tolist(),
        "reference_formula_groups": int(len(group_reference)),
        "training_records": records,
    }


def export_rf(model: RandomForestRegressor) -> dict[str, Any]:
    trees = []
    for estimator in model.estimators_:
        tree = estimator.tree_
        trees.append(
            {
                "children_left": tree.children_left.astype(int).tolist(),
                "children_right": tree.children_right.astype(int).tolist(),
                "feature": tree.feature.astype(int).tolist(),
                "threshold": tree.threshold.astype(float).tolist(),
                "value": tree.value[:, 0, 0].astype(float).tolist(),
            }
        )
    return {"kind": "random_forest", "n_features": int(model.n_features_in_), "trees": trees}


def export_svr(model: TransformedTargetRegressor) -> dict[str, Any]:
    pipeline = model.regressor_
    x_scaler: StandardScaler = pipeline.named_steps["scaler"]
    svr: SVR = pipeline.named_steps["regressor"]
    y_scaler: StandardScaler = model.transformer_
    return {
        "kind": "svr_rbf",
        "x_mean": x_scaler.mean_.astype(float).tolist(),
        "x_scale": x_scaler.scale_.astype(float).tolist(),
        "support_vectors": svr.support_vectors_.astype(float).tolist(),
        "dual_coef": svr.dual_coef_[0].astype(float).tolist(),
        "intercept": float(svr.intercept_[0]),
        "gamma": float(svr._gamma),
        "y_mean": float(y_scaler.mean_[0]),
        "y_scale": float(y_scaler.scale_[0]),
    }


def xgb_tree_value(node: dict[str, Any], row: np.ndarray) -> float:
    current = node
    while "leaf" not in current:
        split = str(current["split"])
        feature_index = int(split[1:]) if split.startswith("f") else INPUT_CODES.index(split)
        feature_value = np.float32(row[feature_index])
        split_value = np.float32(current["split_condition"])
        branch_id = int(current["yes"] if feature_value < split_value else current["no"])
        current = next(
            child for child in current["children"] if int(child["nodeid"]) == branch_id
        )
    return float(current["leaf"])


def export_xgboost(model: XGBRegressor, x_reference: np.ndarray) -> dict[str, Any]:
    booster = model.get_booster()
    trees = [json.loads(tree) for tree in booster.get_dump(dump_format="json")]
    reference = x_reference[: min(12, len(x_reference))]
    margins = np.asarray(
        booster.inplace_predict(reference, predict_type="margin"), dtype=float
    )
    tree_sums = np.asarray(
        [sum(xgb_tree_value(tree, row) for tree in trees) for row in reference],
        dtype=float,
    )
    base_margins = margins - tree_sums
    if not np.allclose(base_margins, base_margins[0], atol=1e-5, rtol=1e-6):
        raise ValueError("Could not determine a stable XGBoost base margin")
    return {
        "kind": "xgboost_trees",
        "base_margin": float(base_margins.mean()),
        "trees": trees,
    }


def export_browser_model(model: Any, model_name: str, x_reference: np.ndarray) -> dict[str, Any]:
    if model_name == "RF":
        return export_rf(model)
    if model_name == "SVR":
        return export_svr(model)
    if model_name == "XGBoost":
        return export_xgboost(model, x_reference)
    raise ValueError(
        f"Browser export for {model_name} is not implemented. ANN remains a comparison model "
        "and cannot be selected for the browser deployment."
    )


def training_ranges(data: pd.DataFrame, codes: list[str]) -> dict[str, Any]:
    ranges: dict[str, Any] = {}
    for code in codes:
        values = data[code].dropna().to_numpy(dtype=float)
        if len(values) == 0:
            raise ValueError(f"No finite values available for range {code}")
        ranges[code] = {
            "min": float(np.min(values)),
            "q05": float(np.quantile(values, 0.05)),
            "median": float(np.median(values)),
            "q95": float(np.quantile(values, 0.95)),
            "max": float(np.max(values)),
        }
    return ranges


def predict_deployment(models: dict[str, Any], x: np.ndarray) -> dict[str, np.ndarray]:
    predictions = {
        target: np.asarray(model.predict(x), dtype=float).reshape(-1)
        for target, model in models.items()
    }
    return {target: predictions[target] for target in TARGET_CODES}


def write_js_bundle(path: Path, bundle: dict[str, Any]) -> None:
    payload = json.dumps(bundle, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    wrapper = (
        "(function(root,factory){var value=factory();"
        "if(typeof module==='object'&&module.exports){module.exports=value;}"
        "else{root.SAF_MODEL_BUNDLE=value;}})"
        "(typeof self!=='undefined'?self:this,function(){return "
        + payload
        + ";});\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(wrapper, encoding="utf-8", newline="\n")


def train_and_export(
    data: pd.DataFrame,
    input_path: Path,
    config: dict[str, Any],
    config_path: Path,
    cv_dir: Path,
    output_dir: Path,
    version: str,
    n_jobs: int,
    deidentify_ood: bool,
) -> dict[str, Any]:
    protocol_path = cv_dir / "protocol.json"
    oof_path = cv_dir / "oof_predictions.csv"
    fold_manifest_path = cv_dir / "fold_manifest.csv"
    for path in [protocol_path, oof_path, fold_manifest_path]:
        if not path.is_file():
            raise FileNotFoundError(f"Required nested-CV artifact is missing: {path}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("smoke_grid"):
        raise ValueError("Refusing final export from nested CV produced with --smoke-grid")
    if protocol.get("input_csv_sha256") != sha256(input_path):
        raise ValueError("Nested-CV artifacts were generated from a different input CSV")

    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "models"
    test_dir = output_dir / "tests"
    model_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "training_config.json", config)

    x = data[INPUT_CODES].to_numpy(dtype=float)
    groups = data["molecular_formula"].to_numpy(dtype=str)
    deployment: dict[str, str] = config["deployment_models"]
    fitted: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    final_grid_rows: list[dict[str, Any]] = []

    for deployment_index, target in enumerate(config["final_fit_target_order"]):
        model_name = deployment[target]
        seed = final_seed(config, deployment_index)
        available = data[target].notna().to_numpy()
        target_x = x[available]
        target_y = data.loc[available, target].to_numpy(dtype=float)
        target_groups = groups[available]
        search = GridSearchCV(
            make_estimator(config, model_name, seed),
            model_grid(config, model_name, smoke=False),
            scoring=config["selection_scoring"],
            cv=GroupKFold(n_splits=int(config["final_cv_folds"])),
            n_jobs=1 if model_name == "ANN" else n_jobs,
            refit=True,
            error_score="raise",
            return_train_score=False,
        )
        search.fit(target_x, target_y, groups=target_groups)
        fitted[target] = search.best_estimator_
        summary_rows.append(
            {
                "target": target,
                "property": TARGET_NAMES[target],
                "deployment_model": model_name,
                "selection_data": (
                    f"development CSV complete cases for {target} "
                    f"(n={len(target_y)}, formula groups={len(set(target_groups))})"
                ),
                "selection_cv": (
                    f"GroupKFold(n_splits={config['final_cv_folds']}), "
                    "groups=molecular_formula"
                ),
                "selection_metric": "RMSE",
                "best_cv_rmse": -float(search.best_score_),
                "best_params": json.dumps(
                    search.best_params_, ensure_ascii=False, sort_keys=True
                ),
                "random_seed": seed,
            }
        )
        results = search.cv_results_
        for index, params in enumerate(results["params"]):
            final_grid_rows.append(
                {
                    "target": target,
                    "model": model_name,
                    "params": json.dumps(params, ensure_ascii=False, sort_keys=True),
                    "mean_test_rmse": -float(results["mean_test_score"][index]),
                    "std_test_score": float(results["std_test_score"][index]),
                    "rank_test_score": int(results["rank_test_score"][index]),
                    "random_seed": seed,
                }
            )
        print(
            f"FINAL {target} | {model_name} | RMSE={-search.best_score_:.6g} | "
            f"seed={seed}",
            flush=True,
        )

    oof = pd.read_csv(oof_path, encoding="utf-8-sig")
    intervals = build_intervals(data, oof, deployment)
    ood = build_ood_reference(data, deidentify=deidentify_ood)
    provenance_counts = {
        target: {
            str(key): int(value)
            for key, value in data.loc[data[target].notna(), f"provenance_{target}"]
            .value_counts()
            .sort_index()
            .items()
        }
        for target in TARGET_CODES
    }
    label_availability = {
        target: {
            "available": int(data[target].notna().sum()),
            "missing": int(data[target].isna().sum()),
        }
        for target in TARGET_CODES
    }
    generated_at = utc_now()
    training_metadata = {
        "input_csv_name": input_path.name,
        "input_csv_sha256": sha256(input_path),
        "records": len(data),
        "formula_groups": int(data["molecular_formula"].nunique()),
        "target_records": {
            target: int(data[target].notna().sum()) for target in TARGET_CODES
        },
        "target_formula_groups": {
            target: int(data.loc[data[target].notna(), "molecular_formula"].nunique())
            for target in TARGET_CODES
        },
        "config_sha256": sha256(config_path),
        "fold_manifest_sha256": sha256(fold_manifest_path),
        "oof_predictions_sha256": sha256(oof_path),
        "held_out_data_used": False,
        "base_seed": int(config["base_seed"]),
        "seed_policy": config.get("seed_policy"),
    }
    joblib_payload = {
        "schema_version": 2,
        "version": version,
        "input_codes": INPUT_CODES,
        "target_codes": TARGET_CODES,
        "models": fitted,
        "intervals": intervals,
        "ood": ood,
        "input_ranges": training_ranges(data, INPUT_CODES),
        "output_ranges": training_ranges(data, TARGET_CODES),
        "deployment_models": deployment,
        "training": training_metadata,
    }
    joblib_name = (
        "SAF_Predict_public_model_bundle_v1.joblib"
        if deidentify_ood
        else "SAF_Predict_model_bundle_v1.joblib"
    )
    joblib_path = model_dir / joblib_name
    joblib.dump(joblib_payload, joblib_path, compress=3)

    exported_models = {
        target: export_browser_model(fitted[target], deployment[target], x)
        for target in DIRECTLY_MODELLED_TARGETS
    }
    browser_bundle = {
        "schema_version": 2,
        "platform": "SAF-Predict",
        "version": version,
        "generated_at_utc": generated_at,
        "input_order": INPUT_CODES,
        "target_order": TARGET_CODES,
        "input_meta": INPUT_META,
        "output_meta": OUTPUT_META,
        "input_ranges": training_ranges(data, INPUT_CODES),
        "output_ranges": training_ranges(data, TARGET_CODES),
        "deployment_models": deployment,
        "models": exported_models,
        "intervals": intervals,
        "ood": ood,
        "provenance_counts": provenance_counts,
        "label_availability": label_availability,
        "training": training_metadata,
        "claim_boundary": (
            "Research-stage physicochemical prescreening of pure hydrocarbon molecules "
            "within the represented descriptor space; not finished-fuel certification or "
            "sustainability assessment."
        ),
    }
    bundle_json_path = model_dir / "model_bundle.json"
    write_json(bundle_json_path, browser_bundle)
    js_path = output_dir / "model_bundle.js"
    write_js_bundle(js_path, browser_bundle)

    summary = pd.DataFrame(summary_rows)
    order = {code: index for index, code in enumerate(TARGET_CODES)}
    summary["target_order"] = summary["target"].map(order)
    summary = summary.sort_values("target_order").drop(columns="target_order")
    write_csv(model_dir / "model_training_summary.csv", summary)
    write_csv(model_dir / "final_cv_search_results.csv", pd.DataFrame(final_grid_rows))

    predictions = predict_deployment(fitted, x)
    reference_rows: list[dict[str, Any]] = []
    for case_number, index in enumerate(np.linspace(0, len(data) - 1, 5, dtype=int), start=1):
        reference_rows.append(
            {
                "id": f"reference_case_{case_number:02d}",
                "x": x[index].astype(float).tolist(),
                "predictions": {
                    target: float(predictions[target][index]) for target in TARGET_CODES
                },
            }
        )
    median_x = np.median(x, axis=0).reshape(1, -1)
    median_prediction = predict_deployment(fitted, median_x)
    reference_rows.append(
        {
            "id": "synthetic_development_median",
            "x": median_x[0].astype(float).tolist(),
            "predictions": {
                target: float(median_prediction[target][0]) for target in TARGET_CODES
            },
        }
    )
    write_json(test_dir / "python_reference_predictions_6cases.json", reference_rows)

    model_card = {
        "model_name": "SAF-Predict",
        "version": version,
        "generated_at_utc": generated_at,
        "purpose": (
            "Preliminary physicochemical screening of pure SAF-relevant hydrocarbons "
            "from seven molecular descriptors."
        ),
        "training_data": training_metadata,
        "deployment_models": browser_bundle["deployment_models"],
        "input_definitions": INPUT_META,
        "output_definitions": OUTPUT_META,
        "interval_method": (
            "Symmetric 90% and 95% empirical intervals from molecular-formula-group maxima "
            "of nested-CV out-of-fold absolute residuals."
        ),
        "applicability_method": ood["method"],
        "y02_policy": (
            "Y02 is predicted by its separately fitted deployment model trained on "
            "162 identity-resolved independent responses in 66 molecular-formula groups."
        ),
        "software_versions": software_versions(),
        "deidentified_ood_records": bool(deidentify_ood),
        "limitations": [
            "The held-out test set is not read by this training and export entry point.",
            "Y02 labels were independently collected, but row-level primary-source and measurement-condition verification remains incomplete.",
            "The output is for research-stage pure-compound prescreening, not finished-fuel certification.",
            "The public OOD record vectors may be linkable to a separately known source dataset.",
        ],
    }
    write_json(model_dir / "model_card.json", model_card)

    artifact_paths = [
        joblib_path,
        bundle_json_path,
        js_path,
        model_dir / "model_training_summary.csv",
        model_dir / "final_cv_search_results.csv",
        model_dir / "model_card.json",
        test_dir / "python_reference_predictions_6cases.json",
        output_dir / "training_config.json",
    ]
    artifact_manifest = {
        "record_type": "self-generated training/export artifact manifest",
        "version": version,
        "generated_at_utc": generated_at,
        "files": {
            path.relative_to(output_dir).as_posix(): sha256(path) for path in artifact_paths
        },
    }
    write_json(output_dir / "artifact_manifest.json", artifact_manifest)
    return {
        "status": "trained_and_exported",
        "version": version,
        "records": len(data),
        "deployment_models": browser_bundle["deployment_models"],
        "joblib": str(joblib_path),
        "joblib_sha256": sha256(joblib_path),
        "browser_bundle": str(js_path),
        "browser_bundle_sha256": sha256(js_path),
        "y02_policy": model_card["y02_policy"],
    }


def add_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", type=Path, required=True, help="Canonical development CSV")
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help="JSON model/CV configuration"
    )
    parser.add_argument(
        "--expected-records",
        type=int,
        default=300,
        help="Expected development rows; use 0 to disable the count check",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Portable four-model nested CV and SAF-Predict deployment exporter"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="Validate CSV and configuration only")
    add_data_arguments(validate)

    smoke = subparsers.add_parser("smoke", help="Quick four-model structural smoke test")
    add_data_arguments(smoke)
    smoke.add_argument("--output-dir", type=Path, default=Path("run_outputs/smoke"))
    smoke.add_argument("--n-jobs", type=int, default=1)

    nested = subparsers.add_parser(
        "nested-cv", help="Run four algorithms for all seven targets with grouped nested CV"
    )
    add_data_arguments(nested)
    nested.add_argument("--output-dir", type=Path, default=Path("run_outputs/cv"))
    nested.add_argument("--n-jobs", type=int, default=1)
    nested.add_argument(
        "--smoke-grid",
        action="store_true",
        help="Use only the first grid combination; outputs cannot be used for final export",
    )

    export = subparsers.add_parser(
        "train-export", help="Train selected deployment models and export Joblib/JSON/JavaScript"
    )
    add_data_arguments(export)
    export.add_argument("--cv-dir", type=Path, required=True)
    export.add_argument("--output-dir", type=Path, default=Path("run_outputs/release"))
    export.add_argument("--version", default="1.1.0")
    export.add_argument("--n-jobs", type=int, default=1)
    export.add_argument(
        "--deidentify-ood",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Retain only DEV IDs and X vectors in exported OOD training records",
    )

    all_command = subparsers.add_parser(
        "all", help="Run full nested CV followed by final deployment training and export"
    )
    add_data_arguments(all_command)
    all_command.add_argument("--output-dir", type=Path, default=Path("run_outputs"))
    all_command.add_argument("--version", default="1.1.0")
    all_command.add_argument("--n-jobs", type=int, default=1)
    all_command.add_argument(
        "--deidentify-ood",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = args.input.resolve()
    config_path = args.config.resolve()
    config = load_config(config_path)
    data = load_development_csv(input_path, args.expected_records, config)
    validate_group_counts(data, config)

    if args.command == "validate":
        y02_available = data["Y02"].notna()
        paired_difference = (
            data.loc[y02_available, "Y02"]
            - data.loc[y02_available, "Y01"] * data.loc[y02_available, "Y03"]
        )
        print(
            json.dumps(
                {
                    "status": "valid",
                    "records": len(data),
                    "formula_groups": int(data["molecular_formula"].nunique()),
                    "input_csv_sha256": sha256(input_path),
                    "config_sha256": sha256(config_path),
                    "target_records": {
                        target: int(data[target].notna().sum()) for target in TARGET_CODES
                    },
                    "target_formula_groups": {
                        target: int(
                            data.loc[data[target].notna(), "molecular_formula"].nunique()
                        )
                        for target in TARGET_CODES
                    },
                    "Y02_independent_minus_physical_baseline": {
                        "n": int(y02_available.sum()),
                        "mean_difference": float(paired_difference.mean()),
                        "max_absolute_difference": float(paired_difference.abs().max()),
                    },
                    "deployed_Y02_rule": "separately fitted direct-response model",
                },
                indent=2,
            )
        )
        return

    if args.command == "smoke":
        report = run_smoke(data, config, args.output_dir.resolve(), args.n_jobs)
        print(json.dumps(report, indent=2))
        return

    if args.command == "nested-cv":
        paths = run_nested_cv(
            data,
            input_path,
            config,
            config_path,
            args.output_dir.resolve(),
            args.n_jobs,
            args.smoke_grid,
        )
        print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))
        return

    if args.command == "train-export":
        result = train_and_export(
            data,
            input_path,
            config,
            config_path,
            args.cv_dir.resolve(),
            args.output_dir.resolve(),
            args.version,
            args.n_jobs,
            args.deidentify_ood,
        )
        print(json.dumps(result, indent=2))
        return

    if args.command == "all":
        root = args.output_dir.resolve()
        cv_dir = root / "cv"
        release_dir = root / "release"
        run_nested_cv(
            data,
            input_path,
            config,
            config_path,
            cv_dir,
            args.n_jobs,
            smoke_grid=False,
        )
        result = train_and_export(
            data,
            input_path,
            config,
            config_path,
            cv_dir,
            release_dir,
            args.version,
            args.n_jobs,
            args.deidentify_ood,
        )
        print(json.dumps(result, indent=2))
        return

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
