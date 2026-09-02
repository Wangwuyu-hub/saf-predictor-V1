#!/usr/bin/env python3
"""Compare a nested-CV rerun with the archived SAF-Predict CV artifacts.

The comparison is intentionally row-order independent. It joins CSV tables on
their scientific keys, requires textual/categorical fields to match exactly,
and compares numerical fields with an absolute tolerance. Runtime durations,
timestamps, hashes, and other run-instance metadata are not comparison inputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = ROOT / "artifacts" / "cv"


TABLE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "name": "fold_manifest.csv",
        "keys": ("record_id",),
        "categorical": ("molecule", "molecular_formula", "outer_fold"),
        "numeric": (),
    },
    {
        "name": "outer_fold_scores.csv",
        "keys": ("model", "target", "outer_fold"),
        "categorical": (
            "property",
            "unit",
            "n_train",
            "n_test",
            "train_formula_groups",
            "test_formula_groups",
            "formula_overlap",
            "best_params",
            "random_seed",
            "warning_count",
        ),
        "numeric": ("r2", "rmse", "mae", "best_inner_rmse"),
        "ignored": ("fit_seconds",),
    },
    {
        "name": "oof_predictions.csv",
        "keys": ("record_id", "model", "target"),
        "categorical": (
            "molecule",
            "molecular_formula",
            "outer_fold",
            "property",
            "unit",
            "provenance_class",
        ),
        "numeric": ("y_true", "y_pred", "residual"),
    },
    {
        "name": "model_target_summary.csv",
        "keys": ("model", "target"),
        "categorical": ("property", "unit", "rank_by_mean_fold_r2"),
        "numeric": (
            "mean_fold_r2",
            "sd_fold_r2",
            "median_fold_r2",
            "min_fold_r2",
            "max_fold_r2",
            "oof_r2",
            "oof_rmse",
            "oof_mae",
        ),
    },
    {
        "name": "provenance_stratified_metrics.csv",
        "keys": ("model", "target", "provenance_class"),
        "categorical": ("property", "n"),
        "numeric": ("r2", "rmse", "mae"),
    },
    {
        "name": "y02_direct_vs_derived_sensitivity.csv",
        "keys": ("model", "approach"),
        "categorical": ("n",),
        "numeric": ("oof_r2", "oof_rmse", "oof_mae"),
    },
)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def key_text(row: dict[str, str], keys: tuple[str, ...]) -> str:
    return " | ".join(f"{field}={row.get(field, '')}" for field in keys)


def index_rows(
    rows: list[dict[str, str]], keys: tuple[str, ...]
) -> tuple[dict[tuple[str, ...], dict[str, str]], list[str]]:
    result: dict[tuple[str, ...], dict[str, str]] = {}
    duplicates: list[str] = []
    for row in rows:
        key = tuple(row.get(field, "") for field in keys)
        if key in result:
            duplicates.append(key_text(row, keys))
        result[key] = row
    return result, duplicates


def is_finite_number(value: str) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def sample_append(samples: list[dict[str, Any]], item: dict[str, Any], limit: int = 12) -> None:
    if len(samples) < limit:
        samples.append(item)


def compare_table(
    generated_dir: Path, archive_dir: Path, spec: dict[str, Any], tolerance: float
) -> dict[str, Any]:
    name = str(spec["name"])
    generated_path = generated_dir / name
    archive_path = archive_dir / name
    result: dict[str, Any] = {
        "artifact": name,
        "keys": list(spec["keys"]),
        "categorical_fields": list(spec.get("categorical", ())),
        "numeric_fields": list(spec.get("numeric", ())),
        "ignored_fields": list(spec.get("ignored", ())),
        "generated_artifact": f"generated-dir/{name}",
        "archived_artifact": f"archive-dir/{name}",
        "pass": False,
    }
    if not generated_path.is_file() or not archive_path.is_file():
        result["missing_files"] = [
            label
            for path, label in (
                (generated_path, f"generated-dir/{name}"),
                (archive_path, f"archive-dir/{name}"),
            )
            if not path.is_file()
        ]
        return result

    generated_rows = read_csv(generated_path)
    archived_rows = read_csv(archive_path)
    generated, generated_duplicates = index_rows(generated_rows, spec["keys"])
    archived, archived_duplicates = index_rows(archived_rows, spec["keys"])
    missing_from_generated = sorted(set(archived) - set(generated))
    unexpected_in_generated = sorted(set(generated) - set(archived))
    categorical_mismatches: list[dict[str, Any]] = []
    numeric_mismatches: list[dict[str, Any]] = []
    nonnumeric_values: list[dict[str, Any]] = []
    max_abs_difference = 0.0

    for key in sorted(set(generated) & set(archived)):
        generated_row = generated[key]
        archived_row = archived[key]
        identifier = key_text(archived_row, spec["keys"])
        for field in spec.get("categorical", ()):
            if generated_row.get(field) != archived_row.get(field):
                sample_append(
                    categorical_mismatches,
                    {
                        "key": identifier,
                        "field": field,
                        "generated": generated_row.get(field),
                        "archived": archived_row.get(field),
                    },
                )
        for field in spec.get("numeric", ()):
            generated_value = generated_row.get(field, "")
            archived_value = archived_row.get(field, "")
            if not is_finite_number(generated_value) or not is_finite_number(archived_value):
                sample_append(
                    nonnumeric_values,
                    {
                        "key": identifier,
                        "field": field,
                        "generated": generated_value,
                        "archived": archived_value,
                    },
                )
                continue
            difference = abs(float(generated_value) - float(archived_value))
            max_abs_difference = max(max_abs_difference, difference)
            if difference > tolerance:
                sample_append(
                    numeric_mismatches,
                    {
                        "key": identifier,
                        "field": field,
                        "generated": float(generated_value),
                        "archived": float(archived_value),
                        "absolute_difference": difference,
                    },
                )

    result.update(
        {
            "generated_rows": len(generated_rows),
            "archived_rows": len(archived_rows),
            "duplicate_generated_keys": generated_duplicates[:12],
            "duplicate_archived_keys": archived_duplicates[:12],
            "missing_from_generated_count": len(missing_from_generated),
            "missing_from_generated_sample": [" | ".join(key) for key in missing_from_generated[:12]],
            "unexpected_in_generated_count": len(unexpected_in_generated),
            "unexpected_in_generated_sample": [" | ".join(key) for key in unexpected_in_generated[:12]],
            "categorical_mismatch_count": len(categorical_mismatches),
            "categorical_mismatch_sample": categorical_mismatches,
            "nonnumeric_value_count": len(nonnumeric_values),
            "nonnumeric_value_sample": nonnumeric_values,
            "numeric_mismatch_count": len(numeric_mismatches),
            "numeric_mismatch_sample": numeric_mismatches,
            "maximum_absolute_numeric_difference": max_abs_difference,
        }
    )
    result["pass"] = not any(
        (
            generated_duplicates,
            archived_duplicates,
            missing_from_generated,
            unexpected_in_generated,
            categorical_mismatches,
            nonnumeric_values,
            numeric_mismatches,
        )
    )
    return result


def parse_splits(text: object) -> int | None:
    match = re.search(r"n_splits=(\d+)", str(text))
    return int(match.group(1)) if match else None


def values_equal(generated: Any, archived: Any, tolerance: float) -> bool:
    if isinstance(generated, bool) or isinstance(archived, bool):
        return generated is archived
    if isinstance(generated, (int, float)) and isinstance(archived, (int, float)):
        return math.isfinite(float(generated)) and math.isfinite(float(archived)) and abs(
            float(generated) - float(archived)
        ) <= tolerance
    if isinstance(generated, list) and isinstance(archived, list):
        return len(generated) == len(archived) and all(
            values_equal(left, right, tolerance) for left, right in zip(generated, archived)
        )
    if isinstance(generated, dict) and isinstance(archived, dict):
        return set(generated) == set(archived) and all(
            values_equal(generated[key], archived[key], tolerance) for key in generated
        )
    return generated == archived


def compare_value(
    generated: Any, archived: Any, tolerance: float, path: str, mismatches: list[dict[str, Any]]
) -> None:
    if not values_equal(generated, archived, tolerance):
        sample_append(mismatches, {"field": path, "generated": generated, "archived": archived})


def compare_protocol(generated_dir: Path, archive_dir: Path, tolerance: float) -> dict[str, Any]:
    generated_path = generated_dir / "protocol.json"
    archived_path = archive_dir / "protocol.json"
    result: dict[str, Any] = {
        "artifact": "protocol.json",
        "comparison": "normalized stable analysis contract; timestamps, hashes, paths, and run metadata ignored",
        "pass": False,
    }
    if not generated_path.is_file() or not archived_path.is_file():
        result["missing_files"] = [
            label
            for path, label in (
                (generated_path, "generated-dir/protocol.json"),
                (archived_path, "archive-dir/protocol.json"),
            )
            if not path.is_file()
        ]
        return result
    generated = json.loads(generated_path.read_text(encoding="utf-8"))
    archived = json.loads(archived_path.read_text(encoding="utf-8"))
    archive_models = archived.get("models", {})
    comparisons = {
        "records": (generated.get("records"), archived.get("n_records")),
        "inputs": (generated.get("inputs"), archived.get("inputs")),
        "targets": (generated.get("targets"), archived.get("targets")),
        "models": (generated.get("models"), list(archive_models)),
        "outer_folds": (parse_splits(generated.get("outer_cv")), parse_splits(archived.get("outer_cv"))),
        "inner_folds": (parse_splits(generated.get("inner_cv")), parse_splits(archived.get("inner_cv"))),
        "base_seed": (generated.get("base_seed"), archived.get("base_seed")),
    }
    mismatches: list[dict[str, Any]] = []
    for name, (generated_value, archived_value) in comparisons.items():
        compare_value(generated_value, archived_value, tolerance, name, mismatches)
    for model, archived_settings in archive_models.items():
        generated_grid = generated.get("parameter_grids", {}).get(model)
        compare_value(
            generated_grid,
            archived_settings.get("parameter_grid"),
            tolerance,
            f"parameter_grid.{model}",
            mismatches,
        )
    result.update(
        {
            "compared_fields": list(comparisons) + [f"parameter_grid.{model}" for model in archive_models],
            "mismatch_count": len(mismatches),
            "mismatch_sample": mismatches,
            "pass": not mismatches,
        }
    )
    return result


def compare_qa(generated_dir: Path, archive_dir: Path) -> dict[str, Any]:
    generated_path = generated_dir / "qa.json"
    archived_path = archive_dir / "qa.json"
    result: dict[str, Any] = {
        "artifact": "qa.json",
        "comparison": "stable completion status only; file hashes and timing-dependent details ignored",
        "pass": False,
    }
    if not generated_path.is_file() or not archived_path.is_file():
        result["missing_files"] = [
            label
            for path, label in (
                (generated_path, "generated-dir/qa.json"),
                (archived_path, "archive-dir/qa.json"),
            )
            if not path.is_file()
        ]
        return result
    generated = json.loads(generated_path.read_text(encoding="utf-8"))
    archived = json.loads(archived_path.read_text(encoding="utf-8"))
    generated_pass = generated.get("all_checks_passed")
    archived_pass = archived.get("pass")
    result.update(
        {
            "generated_all_checks_passed": generated_pass,
            "archived_pass": archived_pass,
            "pass": generated_pass is True and archived_pass is True,
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, required=True, help="JSON comparison report")
    parser.add_argument("--tolerance", type=float, default=1e-12)
    args = parser.parse_args()
    if args.tolerance < 0 or not math.isfinite(args.tolerance):
        parser.error("--tolerance must be a finite non-negative number")

    table_results = [
        compare_table(args.generated_dir, args.archive_dir, spec, args.tolerance)
        for spec in TABLE_SPECS
    ]
    json_results = [
        compare_protocol(args.generated_dir, args.archive_dir, args.tolerance),
        compare_qa(args.generated_dir, args.archive_dir),
    ]
    passed = all(result["pass"] for result in [*table_results, *json_results])
    report = {
        "status": "pass" if passed else "fail",
        "scope": "stable-key nested-CV reproduction comparison; not a byte-for-byte file comparison",
        "generated_directory_label": args.generated_dir.name,
        "archived_directory_label": "artifacts/cv" if args.archive_dir.resolve() == DEFAULT_ARCHIVE.resolve() else args.archive_dir.name,
        "numeric_tolerance": args.tolerance,
        "ignored_run_variant_fields": [
            "outer_fold_scores.csv.fit_seconds",
            "protocol.json timestamps, hashes, paths, and run metadata",
            "qa.json hashes and run-instance details",
        ],
        "table_comparisons": table_results,
        "json_comparisons": json_results,
        "all_comparisons_passed": passed,
    }
    write_json(args.output, report)
    print(json.dumps({"status": report["status"], "output": str(args.output)}))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
