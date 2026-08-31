#!/usr/bin/env python3
"""Create metadata-reduced public model and validation artifacts.

The private source joblib remains unchanged. The public copy retains the
trained estimators and all numerical applicability-domain data, but replaces
development-record metadata with stable public surrogate identifiers.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
VALIDATION_DIR = ROOT / "validation"

DEFAULT_SOURCE_JOBLIB = MODEL_DIR / "SAF_Predict_model_bundle_v1.joblib"
PUBLIC_JOBLIB = MODEL_DIR / "SAF_Predict_public_model_bundle_v1.joblib"
CHECKS_PATH = MODEL_DIR / "public_artifact_sanitization_checks.json"
BROWSER_JSON = MODEL_DIR / "model_bundle.json"
BROWSER_JS = ROOT / "model_bundle.js"
WIDE_CSV = VALIDATION_DIR / "held_out_predictions.csv"
LONG_CSV = VALIDATION_DIR / "held_out_test_predictions_long.csv"

EXPECTED_SOURCE_JOBLIB_SHA256 = "e5a648cba47e0ba98e902144e45b937610f82c0d96719b4ef248cf3ab903ef73"
EXPECTED_SOURCE_CSV_SHA256 = {
    WIDE_CSV.name: "040a212e0ff390af7d7ec8a0e68e7c77c5c19b09d2bbd3c6b274a5b245e19018",
    LONG_CSV.name: "a9132b32e99880d21371a000b4cd2cf6b924fa28c3a6044945ed1abc02a47581",
}
EXPECTED_PUBLIC_NEAREST_IDS = {
    "DEV-0004", "DEV-0032", "DEV-0047", "DEV-0075", "DEV-0079",
    "DEV-0101", "DEV-0153", "DEV-0171", "DEV-0184", "DEV-0236",
}
INPUT_CODES = [f"X{i:02d}" for i in range(1, 8)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_dump_joblib(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        joblib.dump(value, temporary_path, compress=3)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def builtin_without_name_zh(value: Any) -> Any:
    """Copy builtin containers while removing bilingual display metadata."""
    if isinstance(value, dict):
        return {
            key: builtin_without_name_zh(item)
            for key, item in value.items()
            if key != "name_zh"
        }
    if isinstance(value, list):
        return [builtin_without_name_zh(item) for item in value]
    if isinstance(value, tuple):
        return tuple(builtin_without_name_zh(item) for item in value)
    return value


def contains_key(value: Any, forbidden_key: str) -> bool:
    if isinstance(value, dict):
        return forbidden_key in value or any(contains_key(item, forbidden_key) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(contains_key(item, forbidden_key) for item in value)
    return False


def source_record_map(records: list[dict[str, Any]]) -> dict[str, str]:
    if len(records) != 300:
        raise ValueError(f"Expected 300 development records, found {len(records)}")
    mapping: dict[str, str] = {}
    for index, record in enumerate(records, start=1):
        source_id = str(record.get("record_id", ""))
        vector = record.get("x")
        if not source_id or source_id in mapping:
            raise ValueError(f"Missing or duplicate source record identifier at position {index}")
        if not isinstance(vector, list) or len(vector) != 7 or not all(math.isfinite(float(item)) for item in vector):
            raise ValueError(f"Invalid seven-descriptor vector for {source_id}")
        mapping[source_id] = f"DEV-{index:04d}"
    return mapping


def array_signature(value: Any) -> dict[str, Any]:
    array = np.asarray(value)
    contiguous = np.ascontiguousarray(array)
    return {
        "dtype": str(contiguous.dtype),
        "shape": list(contiguous.shape),
        "sha256": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
    }


def scalar_parameters(estimator: Any) -> dict[str, Any]:
    parameters: dict[str, Any] = {}
    for key, value in estimator.get_params(deep=True).items():
        if value is None or isinstance(value, (str, int, bool)):
            parameters[key] = value
        elif isinstance(value, float):
            parameters[key] = value if math.isfinite(value) else {"nonfinite_float": str(value)}
    return parameters


def estimator_signature(estimator: Any) -> dict[str, Any]:
    """Return a stable signature of fitted numerical estimator state."""
    class_name = type(estimator).__name__
    signature: dict[str, Any] = {
        "class": f"{type(estimator).__module__}.{class_name}",
        "scalar_parameters": scalar_parameters(estimator),
    }
    if class_name == "RandomForestRegressor":
        signature["trees"] = []
        for tree_estimator in estimator.estimators_:
            tree_state = tree_estimator.tree_.__getstate__()
            signature["trees"].append({
                "max_depth": int(tree_state["max_depth"]),
                "node_count": int(tree_state["node_count"]),
                "node_fields": {
                    field: array_signature(tree_state["nodes"][field])
                    for field in tree_state["nodes"].dtype.names
                },
                "values": array_signature(tree_state["values"]),
            })
        return signature
    if class_name == "TransformedTargetRegressor":
        signature["regressor"] = estimator_signature(estimator.regressor_)
        signature["transformer"] = estimator_signature(estimator.transformer_)
        return signature
    if class_name == "Pipeline":
        signature["steps"] = [
            {"name": name, "estimator": estimator_signature(step)}
            for name, step in estimator.steps
        ]
        return signature
    if class_name == "StandardScaler":
        signature["fitted_arrays"] = {
            name: array_signature(getattr(estimator, name))
            for name in ("mean_", "var_", "scale_", "n_samples_seen_")
            if hasattr(estimator, name)
        }
        signature["n_features_in"] = int(estimator.n_features_in_)
        return signature
    if class_name == "SVR":
        signature["fitted_arrays"] = {
            name: array_signature(getattr(estimator, name))
            for name in (
                "class_weight_", "support_", "support_vectors_", "_n_support",
                "dual_coef_", "intercept_", "_probA", "_probB",
            )
            if hasattr(estimator, name)
        }
        signature["fit_status"] = int(estimator.fit_status_)
        signature["gamma_fitted"] = float(estimator._gamma)
        signature["shape_fit"] = list(estimator.shape_fit_)
        return signature
    if class_name == "XGBRegressor":
        booster_bytes = bytes(estimator.get_booster().save_raw(raw_format="json"))
        signature["booster_json_sha256"] = hashlib.sha256(booster_bytes).hexdigest()
        signature["n_features_in"] = int(estimator.n_features_in_)
        return signature
    raise TypeError(f"Unsupported estimator type for stable signature: {signature['class']}")


def estimator_hashes(bundle: dict[str, Any]) -> dict[str, str]:
    return {
        target: canonical_sha256(estimator_signature(model))
        for target, model in bundle["models"].items()
    }


def scientific_payload(bundle: dict[str, Any]) -> dict[str, Any]:
    ood = bundle["ood"]
    return {
        "version": bundle["version"],
        "input_codes": bundle["input_codes"],
        "target_codes": bundle["target_codes"],
        "model_hashes": estimator_hashes(bundle),
        "intervals": bundle["intervals"],
        "input_ranges": bundle["input_ranges"],
        "output_ranges": bundle["output_ranges"],
        "deployment_models": bundle["deployment_models"],
        "ood": {key: value for key, value in ood.items() if key != "training_records"},
        "development_x_vectors": [record["x"] for record in ood["training_records"]],
    }


def public_joblib_from_source(source: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    records = source["ood"]["training_records"]
    mapping = source_record_map(records)
    public = dict(source)
    public["ood"] = dict(source["ood"])
    public["ood"]["training_records"] = [
        {"record_id": f"DEV-{index:04d}", "x": list(record["x"])}
        for index, record in enumerate(records, start=1)
    ]
    public = builtin_without_name_zh(public)
    public["public_release"] = {
        "development_record_metadata_removed": True,
        "public_surrogate_ids_assigned": True,
        "retained_for_applicability_domain": [
            "public surrogate record_id",
            "X01-X07 descriptor vector",
        ],
        "removed_from_public_artifact": [
            "source record_id",
            "molecule name",
            "molecular formula",
            "name_zh metadata",
        ],
        "linkability_caveat": (
            "Descriptor vectors are retained for applicability-domain calculations and may be linkable "
            "to a separately available source dataset; de-identification is not a guarantee of anonymity."
        ),
    }
    return public, mapping


def sanitize_browser_bundle(expected_vectors: list[list[float]]) -> dict[str, Any]:
    """Regenerate the public JSON and UMD bundle with metadata-only changes."""
    bundle = json.loads(BROWSER_JSON.read_text(encoding="utf-8"))
    bundle = builtin_without_name_zh(bundle)
    records = bundle["ood"]["training_records"]
    if len(records) != 300:
        raise ValueError(f"Expected 300 browser development records, found {len(records)}")
    vectors = [list(record["x"]) for record in records]
    if vectors != expected_vectors:
        raise AssertionError("Browser development vectors differ from the private source joblib")
    bundle["ood"]["training_records"] = [
        {"record_id": f"DEV-{index:04d}", "x": vector}
        for index, vector in enumerate(vectors, start=1)
    ]
    training = bundle.get("training", {})
    if "held_out_sheet_used" in training:
        training["author_reported_held_out_sheet_used"] = training.pop("held_out_sheet_used")
    bundle["public_release"] = {
        "development_record_metadata_removed": True,
        "public_surrogate_ids_assigned": True,
        "retained_for_applicability_domain": [
            "public surrogate record_id",
            "X01-X07 descriptor vector",
        ],
        "removed_from_public_artifact": [
            "source record_id",
            "molecule name",
            "molecular formula",
            "name_zh metadata",
        ],
        "linkability_caveat": (
            "Descriptor vectors are retained for applicability-domain calculations and may be linkable "
            "to a separately available source dataset; metadata removal is not a guarantee of anonymity."
        ),
    }

    json_payload = (json.dumps(bundle, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8")
    compact = json.dumps(bundle, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    wrapper = (
        "(function(root,factory){var value=factory();"
        "if(typeof module==='object'&&module.exports){module.exports=value;}"
        "else{root.SAF_MODEL_BUNDLE=value;}})"
        "(typeof self!=='undefined'?self:this,function(){return "
        + compact
        + ";});\n"
    ).encode("utf-8")
    atomic_write_bytes(BROWSER_JSON, json_payload)
    atomic_write_bytes(BROWSER_JS, wrapper)

    reloaded = json.loads(BROWSER_JSON.read_text(encoding="utf-8"))
    public_records = reloaded["ood"]["training_records"]
    schema_valid = all(
        list(record.keys()) == ["record_id", "x"]
        and record["record_id"] == f"DEV-{index:04d}"
        and len(record["x"]) == 7
        and all(math.isfinite(float(item)) for item in record["x"])
        for index, record in enumerate(public_records, start=1)
    )
    if not schema_valid or contains_key(reloaded, "name_zh"):
        raise AssertionError("Public browser-bundle metadata reduction failed verification")
    return {
        "json_path": "models/model_bundle.json",
        "json_sha256": sha256(BROWSER_JSON),
        "javascript_path": "model_bundle.js",
        "javascript_sha256": sha256(BROWSER_JS),
        "record_count": len(public_records),
        "record_schema_valid": schema_valid,
        "development_x_vectors_equal_private_source": vectors == expected_vectors,
        "name_zh_absent": not contains_key(reloaded, "name_zh"),
        "author_reported_held_out_field_scoped": (
            "held_out_sheet_used" not in reloaded.get("training", {})
            and "author_reported_held_out_sheet_used" in reloaded.get("training", {})
        ),
    }


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header: {path}")
        return list(reader.fieldnames), list(reader)


def csv_bytes(fieldnames: list[str], rows: list[dict[str, str]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8-sig")


def sanitize_validation_csv(
    path: Path,
    mapping: dict[str, str],
    remove_fields: set[str],
) -> dict[str, Any]:
    before_hash = sha256(path)
    before_fields, before_rows = read_csv(path)
    source_id_field = (
        "nearest_training_record"
        if "nearest_training_record" in before_fields
        else "nearest_development_record"
        if "nearest_development_record" in before_fields
        else None
    )
    if source_id_field is None:
        raise ValueError(f"No nearest-development-record field in {path.name}")

    after_fields: list[str] = []
    for field in before_fields:
        if field in remove_fields:
            continue
        after_fields.append("nearest_development_record" if field == source_id_field else field)
    if len(after_fields) != len(set(after_fields)):
        raise ValueError(f"Duplicate field generated while sanitizing {path.name}")

    valid_public_ids = set(mapping.values())
    after_rows: list[dict[str, str]] = []
    for before in before_rows:
        nearest = before[source_id_field]
        if nearest in mapping:
            public_nearest = mapping[nearest]
        elif nearest in valid_public_ids:
            public_nearest = nearest
        else:
            raise ValueError(f"Unknown nearest development record {nearest!r} in {path.name}")
        after: dict[str, str] = {}
        for field in before_fields:
            if field in remove_fields:
                continue
            output_field = "nearest_development_record" if field == source_id_field else field
            after[output_field] = public_nearest if field == source_id_field else before[field]
        after_rows.append(after)

    if len(before_rows) != len(after_rows):
        raise AssertionError(f"Row count changed for {path.name}")
    observed_public_nearest_ids = {row["nearest_development_record"] for row in after_rows}
    if observed_public_nearest_ids != EXPECTED_PUBLIC_NEAREST_IDS:
        raise AssertionError(f"Unexpected nearest-development surrogate IDs in {path.name}")
    identity_fields = [field for field in ("record_id", "molecule", "molecular_formula") if field in before_fields]
    preserved_fields = [
        field for field in before_fields
        if field not in remove_fields and field != source_id_field
    ]
    for row_index, (before, after) in enumerate(zip(before_rows, after_rows), start=2):
        for field in preserved_fields:
            if before[field] != after[field]:
                raise AssertionError(f"Unexpected change in {path.name}, row {row_index}, field {field}")
        for field in identity_fields:
            if before[field] != after[field]:
                raise AssertionError(f"Holdout identity changed in {path.name}, row {row_index}, field {field}")

    atomic_write_bytes(path, csv_bytes(after_fields, after_rows))
    written_fields, written_rows = read_csv(path)
    if written_fields != after_fields or written_rows != after_rows:
        raise AssertionError(f"CSV write verification failed for {path.name}")
    return {
        "path": f"validation/{path.name}",
        "expected_private_source_sha256": EXPECTED_SOURCE_CSV_SHA256[path.name],
        "input_sha256": before_hash,
        "public_sha256": sha256(path),
        "row_count": len(after_rows),
        "holdout_identity_fields_preserved": identity_fields,
        "all_other_field_values_preserved": True,
        "nearest_id_field": "nearest_development_record",
        "removed_fields": sorted(remove_fields),
    }


def prediction_equivalence(
    private: dict[str, Any],
    public: dict[str, Any],
    holdout_rows: list[dict[str, str]],
) -> tuple[dict[str, bool], dict[str, float], int]:
    vectors = [record["x"] for record in private["ood"]["training_records"]]
    vectors.extend([[float(row[code]) for code in INPUT_CODES] for row in holdout_rows])
    matrix = np.asarray(vectors, dtype=float)
    exact: dict[str, bool] = {}
    maximum_difference: dict[str, float] = {}
    for target in private["models"]:
        before = np.asarray(private["models"][target].predict(matrix), dtype=float)
        after = np.asarray(public["models"][target].predict(matrix), dtype=float)
        difference = np.abs(before - after)
        maximum_difference[target] = float(np.max(difference))
        exact[target] = bool(np.array_equal(before, after))
    return exact, maximum_difference, len(matrix)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build metadata-reduced public artifacts from the private source Joblib."
    )
    parser.add_argument(
        "--source-joblib",
        default=str(DEFAULT_SOURCE_JOBLIB),
        help="Path to the private source Joblib (not distributed in the public release).",
    )
    args = parser.parse_args()
    source_joblib = Path(args.source_joblib).expanduser().resolve()
    if not source_joblib.is_file():
        raise FileNotFoundError(
            "Private source Joblib not found. Supply it with --source-joblib; it is not distributed "
            "in the public release."
        )
    source_hash = sha256(source_joblib)
    if source_hash != EXPECTED_SOURCE_JOBLIB_SHA256:
        raise ValueError(
            f"Private source joblib hash mismatch: expected {EXPECTED_SOURCE_JOBLIB_SHA256}, found {source_hash}"
        )

    private = joblib.load(source_joblib)
    private_payload = scientific_payload(private)
    public, mapping = public_joblib_from_source(private)
    public_payload = scientific_payload(public)
    if private_payload != public_payload:
        raise AssertionError("Scientific payload changed before public joblib serialization")
    if contains_key(public, "name_zh"):
        raise AssertionError("name_zh remains in the public joblib")

    expected_vectors = [list(record["x"]) for record in private["ood"]["training_records"]]
    browser_bundle_check = sanitize_browser_bundle(expected_vectors)

    _, wide_rows_before = read_csv(WIDE_CSV)
    wide_csv_check = sanitize_validation_csv(
        WIDE_CSV,
        mapping,
        {"nearest_training_molecule", "nearest_training_formula"},
    )
    long_csv_check = sanitize_validation_csv(
        LONG_CSV,
        mapping,
        {"nearest_training_molecule", "nearest_training_formula"},
    )

    reuse_existing_public_joblib = False
    if PUBLIC_JOBLIB.is_file():
        try:
            existing_public = joblib.load(PUBLIC_JOBLIB)
            reuse_existing_public_joblib = (
                scientific_payload(existing_public) == public_payload
                and existing_public.get("public_release") == public.get("public_release")
                and not contains_key(existing_public, "name_zh")
            )
        except Exception:
            reuse_existing_public_joblib = False
    if not reuse_existing_public_joblib:
        atomic_dump_joblib(PUBLIC_JOBLIB, public)
    reloaded = joblib.load(PUBLIC_JOBLIB)
    reloaded_payload = scientific_payload(reloaded)
    payload_equal = private_payload == reloaded_payload
    if not payload_equal:
        raise AssertionError("Public joblib scientific payload changed after serialization")

    private_model_hashes = estimator_hashes(private)
    public_model_hashes = estimator_hashes(reloaded)
    model_hashes_equal = private_model_hashes == public_model_hashes
    if not model_hashes_equal:
        raise AssertionError("One or more trained estimators changed in the public joblib")

    exact_predictions, maximum_difference, prediction_rows = prediction_equivalence(
        private,
        reloaded,
        wide_rows_before,
    )
    if not all(exact_predictions.values()):
        raise AssertionError(f"Public joblib predictions changed: {maximum_difference}")

    private_records = private["ood"]["training_records"]
    public_records = reloaded["ood"]["training_records"]
    record_schema_valid = (
        len(public_records) == 300
        and all(
            list(record.keys()) == ["record_id", "x"]
            and record["record_id"] == f"DEV-{index:04d}"
            and len(record["x"]) == 7
            and all(math.isfinite(float(item)) for item in record["x"])
            for index, record in enumerate(public_records, start=1)
        )
    )
    x_vectors_equal = [record["x"] for record in private_records] == [record["x"] for record in public_records]
    ood_reference_equal = {
        key: value for key, value in private["ood"].items() if key != "training_records"
    } == {
        key: value for key, value in reloaded["ood"].items() if key != "training_records"
    }
    if not (record_schema_valid and x_vectors_equal and ood_reference_equal):
        raise AssertionError("Public applicability-domain payload failed verification")

    checks = {
        "record_type": "self-generated release-integrity record; not an independent audit",
        "source_private_joblib": {
            "distributed_in_public_release": False,
            "sha256": source_hash,
        },
        "public_joblib": {
            "path": "models/SAF_Predict_public_model_bundle_v1.joblib",
            "sha256": sha256(PUBLIC_JOBLIB),
        },
        "public_browser_bundle": browser_bundle_check,
        "development_record_count": len(public_records),
        "approved_transformations": [
            "Removed name_zh metadata from builtin public-artifact metadata containers.",
            "Replaced each OOD development record, in original stored order, with record_id DEV-0001 through DEV-0300 and the unchanged X01-X07 vector.",
            "Added factual public-release metadata-reduction notes and a descriptor-vector linkability caveat.",
            "Mapped validation nearest-development identifiers to DEV identifiers and removed nearest-development molecule/formula columns.",
        ],
        "canonical_scientific_payload_sha256": canonical_sha256(private_payload),
        "semantic_equivalence": {
            "canonical_scientific_payload_equal": payload_equal,
            "trained_estimator_joblib_hashes_equal": model_hashes_equal,
            "trained_estimator_hashes": private_model_hashes,
            "predictions_exact_for_all_checked_rows": all(exact_predictions.values()),
            "prediction_rows_checked": prediction_rows,
            "prediction_check_scope": (
                "Maintainer-generated semantic equivalence check on 300 development vectors and "
                "10 held-out descriptor rows; not an independent validation."
            ),
            "prediction_targets_exact": exact_predictions,
            "maximum_absolute_prediction_difference": maximum_difference,
            "intervals_equal": private["intervals"] == reloaded["intervals"],
            "input_ranges_equal": private["input_ranges"] == reloaded["input_ranges"],
            "output_ranges_equal": private["output_ranges"] == reloaded["output_ranges"],
            "deployment_mapping_equal": private["deployment_models"] == reloaded["deployment_models"],
            "ood_reference_arrays_equal": ood_reference_equal,
            "development_x_vectors_equal": x_vectors_equal,
            "public_record_schema_valid": record_schema_valid,
            "name_zh_absent": not contains_key(reloaded, "name_zh"),
        },
        "validation_csvs": {
            WIDE_CSV.name: wide_csv_check,
            LONG_CSV.name: long_csv_check,
        },
    }
    atomic_write_bytes(
        CHECKS_PATH,
        (json.dumps(checks, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8"),
    )
    print(json.dumps({
        "public_joblib_sha256": checks["public_joblib"]["sha256"],
        "checks_sha256": sha256(CHECKS_PATH),
        "canonical_scientific_payload_sha256": checks["canonical_scientific_payload_sha256"],
        "prediction_rows_checked": prediction_rows,
        "all_checks_passed": all([
            payload_equal,
            model_hashes_equal,
            all(exact_predictions.values()),
            record_schema_valid,
            x_vectors_equal,
            ood_reference_equal,
        ]),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
