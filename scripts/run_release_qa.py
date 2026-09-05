#!/usr/bin/env python3
"""Run scoped v1.1.0 release-integrity and numerical-agreement checks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import subprocess
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

import joblib


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_MANIFEST = ROOT / "models" / "model_artifact_manifest.json"
RELEASE_CHECKS = ROOT / "RELEASE_CHECKS.json"
RELEASE_MANIFEST = ROOT / "release_manifest.json"
CHECKSUMS = ROOT / "checksums.sha256"
VERSION = "1.1.0"
TARGETS = [f"Y{index:02d}" for index in range(1, 8)]
EXCLUDED_PUBLIC_PATHS = {"models/SAF_Predict_model_bundle_v1.joblib"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def truth(value: object) -> bool | None:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no", ""}:
        return False
    return None


def hash_matches(path: Path, expected: object) -> bool:
    return (
        path.is_file()
        and isinstance(expected, str)
        and bool(re.fullmatch(r"[0-9a-f]{64}", expected))
        and sha256(path) == expected
    )


class LocalReferenceParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.references: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attribute = "src" if tag in {"script", "img"} else "href" if tag in {"a", "link"} else None
        if attribute is None:
            return
        for name, value in attrs:
            if name == attribute and value:
                self.references.append(value)


def check_html_references() -> tuple[int, list[str]]:
    checked = 0
    missing: list[str] = []
    ignored_parts = {".git", "run_outputs", ".venv", "venv", "node_modules"}
    for html_path in sorted(ROOT.rglob("*.html")):
        if ignored_parts.intersection(html_path.parts):
            continue
        parser = LocalReferenceParser()
        parser.feed(html_path.read_text(encoding="utf-8"))
        for raw in parser.references:
            parsed = urlsplit(raw)
            if parsed.scheme or raw.startswith(("#", "javascript:", "mailto:", "data:")):
                continue
            checked += 1
            target = (html_path.parent / unquote(parsed.path)).resolve()
            if not target.exists():
                missing.append(f"{html_path.relative_to(ROOT).as_posix()} -> {raw}")
    return checked, missing


def public_release_files() -> list[Path]:
    """Return the finite set content-addressed by build_release_manifest.py."""
    excluded_names = {RELEASE_MANIFEST.name, CHECKSUMS.name}
    excluded_parts = {".git", "__pycache__", ".venv", "venv", "node_modules", "run_outputs"}
    return [
        path
        for path in sorted(ROOT.rglob("*"))
        if path.is_file()
        and path.name not in excluded_names
        and not excluded_parts.intersection(path.parts)
        and path.relative_to(ROOT).as_posix() not in EXCLUDED_PUBLIC_PATHS
    ]


def verify_release_manifest_and_checksums() -> dict[str, object]:
    """Verify the final content-addressed release without rewriting any file."""
    details: dict[str, object] = {
        "manifest_present": RELEASE_MANIFEST.is_file(),
        "checksums_present": CHECKSUMS.is_file(),
        "manifest_json_valid": False,
        "manifest_version_is_1_1_0": False,
        "manifest_file_count_matches": False,
        "manifest_paths_complete": False,
        "manifest_hashes_match": False,
        "checksum_paths_complete": False,
        "checksum_hashes_match": False,
        "missing_manifest_paths": [],
        "unexpected_manifest_paths": [],
        "manifest_hash_mismatches": [],
        "malformed_checksum_lines": [],
        "duplicate_checksum_paths": [],
        "missing_checksum_paths": [],
        "unexpected_checksum_paths": [],
        "checksum_hash_mismatches": [],
    }
    if not RELEASE_MANIFEST.is_file() or not CHECKSUMS.is_file():
        details["passed"] = False
        return details
    try:
        manifest = json.loads(RELEASE_MANIFEST.read_text(encoding="utf-8"))
        all_files = manifest["all_files"]
        if not isinstance(all_files, dict):
            raise TypeError("release_manifest.json all_files must be an object")
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        details["manifest_error"] = str(error)
        details["passed"] = False
        return details

    details["manifest_json_valid"] = True
    details["manifest_version_is_1_1_0"] = manifest.get("version") == VERSION
    actual_paths = {path.relative_to(ROOT).as_posix(): path for path in public_release_files()}
    manifest_paths = set(all_files)
    missing_manifest_paths = sorted(set(actual_paths) - manifest_paths)
    unexpected_manifest_paths = sorted(manifest_paths - set(actual_paths))
    manifest_hash_mismatches = sorted(
        name
        for name in manifest_paths & set(actual_paths)
        if not isinstance(all_files[name], str)
        or not re.fullmatch(r"[0-9a-f]{64}", all_files[name])
        or sha256(actual_paths[name]) != all_files[name]
    )
    details["missing_manifest_paths"] = missing_manifest_paths
    details["unexpected_manifest_paths"] = unexpected_manifest_paths
    details["manifest_hash_mismatches"] = manifest_hash_mismatches
    details["manifest_paths_complete"] = not missing_manifest_paths and not unexpected_manifest_paths
    details["manifest_hashes_match"] = not manifest_hash_mismatches
    details["manifest_file_count_matches"] = (
        manifest.get("file_count_excluding_manifest_and_checksum_list") == len(actual_paths)
    )

    checksum_entries: dict[str, str] = {}
    malformed_lines: list[str] = []
    duplicate_paths: list[str] = []
    for line_number, line in enumerate(CHECKSUMS.read_text(encoding="utf-8").splitlines(), start=1):
        match = re.fullmatch(r"([0-9a-f]{64}) \*(.+)", line)
        if not match:
            malformed_lines.append(str(line_number))
            continue
        digest, name = match.groups()
        if name in checksum_entries:
            duplicate_paths.append(name)
        checksum_entries[name] = digest
    expected_checksum_paths = set(actual_paths) | {RELEASE_MANIFEST.name}
    checksum_paths = set(checksum_entries)
    missing_checksum_paths = sorted(expected_checksum_paths - checksum_paths)
    unexpected_checksum_paths = sorted(checksum_paths - expected_checksum_paths)
    checksum_hash_mismatches = sorted(
        name
        for name, digest in checksum_entries.items()
        if name in expected_checksum_paths
        and sha256(RELEASE_MANIFEST if name == RELEASE_MANIFEST.name else actual_paths[name]) != digest
    )
    details["malformed_checksum_lines"] = malformed_lines
    details["duplicate_checksum_paths"] = sorted(set(duplicate_paths))
    details["missing_checksum_paths"] = missing_checksum_paths
    details["unexpected_checksum_paths"] = unexpected_checksum_paths
    details["checksum_hash_mismatches"] = checksum_hash_mismatches
    details["checksum_paths_complete"] = not any(
        (malformed_lines, duplicate_paths, missing_checksum_paths, unexpected_checksum_paths)
    )
    details["checksum_hashes_match"] = not checksum_hash_mismatches
    details["passed"] = all(
        bool(details[name])
        for name in (
            "manifest_json_valid",
            "manifest_version_is_1_1_0",
            "manifest_file_count_matches",
            "manifest_paths_complete",
            "manifest_hashes_match",
            "checksum_paths_complete",
            "checksum_hashes_match",
        )
    )
    return details


def scan_public_text() -> tuple[list[str], list[str]]:
    suffixes = {".html", ".md", ".json", ".js", ".css", ".py", ".txt", ".csv", ".sha256", ".yml", ".yaml"}
    special_names = {".gitattributes", ".gitignore", ".nojekyll", ".zenodo.json"}
    marker_pattern = re.compile(
        r"C:\\Users\\|pending\s+author\s+confirmation|\bTODO\b|\bFIXME\b|lorem\s+ipsum",
        re.IGNORECASE,
    )
    cjk_pattern = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
    marker_hits: list[str] = []
    cjk_hits: list[str] = []
    ignored_parts = {".git", "run_outputs", ".venv", "venv", "node_modules", "__pycache__"}
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or ignored_parts.intersection(path.parts):
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        if path.suffix.lower() not in suffixes and path.name not in special_names:
            continue
        text = path.read_text(encoding="utf-8-sig")
        relative = path.relative_to(ROOT).as_posix()
        if marker_pattern.search(text):
            marker_hits.append(relative)
        if cjk_pattern.search(text):
            cjk_hits.append(relative)
    return marker_hits, cjk_hits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", default=shutil.which("node"), help="Path to a Node.js executable")
    parser.add_argument(
        "--verify-release-manifest",
        action="store_true",
        help=(
            "Read-only verification of release_manifest.json and checksums.sha256. "
            "Run after build_release_manifest.py; this mode never rewrites RELEASE_CHECKS.json."
        ),
    )
    args = parser.parse_args()

    if not RELEASE_CHECKS.is_file() and not args.verify_release_manifest:
        RELEASE_CHECKS.write_text("{}\n", encoding="utf-8", newline="\n")

    required = [
        "index.html", "styles.css", "app.js", "predictor_core.js", "model_bundle.js",
        "held_out_test_data.js", "README.md", "README.html", "Start_SAF_Predictor.bat",
        "VERSION.txt", "RELEASE_NOTES_v1.1.0.md", "REPRODUCIBILITY.md",
        "models/model_bundle.json", "models/SAF_Predict_public_model_bundle_v1.joblib",
        "models/model_card.json", "models/model_artifact_manifest.json", "models/MODEL_CARD.md",
        "models/model_training_summary.csv",
        "models/public_artifact_sanitization_checks.json", "scripts/sanitize_public_bundle.py",
        "validation/held_out_predictions.csv", "validation/held_out_test_predictions_long.csv",
        "validation/held_out_test_metrics.csv", "validation/held_out_source_supported_metrics.csv",
        "validation/held_out_test_summary.json", "validation/held_out_evaluation_checks.json",
        "validation/held_out_reproduction_check.json", "validation/held_out_test_report.html",
        "examples/batch_input_template.csv", "tests/test_js_against_python_reference_cases.js",
        "tests/python_reference_predictions_6cases.json", "LICENSE", "DATA_LICENSE.md",
        "CITATION.cff", ".zenodo.json", "requirements.txt", "requirements-lock.txt",
        "environment.yml", "data/README.md", "data/development.csv", "data/held_out_test.csv",
        "data/provenance_long.csv", "data/references.csv", "data/data_dictionary.csv",
        "data/cv_fold_manifest.csv", "data/data_release_checks.json",
        "data/y02_direct_response_reconciliation.csv", "data/y02_full_reconciliation_audit.csv",
        "data/y02_physical_baseline_audit.csv", "data/SAF_Hydrocarbon_Dataset_v1.1.0.xlsx",
        "training/saf_reproduce.py", "training/config.json", "training/input_schema.csv",
        "training/README.md", "scripts/compute_descriptors.py", "scripts/evaluate_heldout.py",
        "scripts/build_validation_metadata.py", "scripts/build_validation_report.py",
        "scripts/build_release_manifest.py", "scripts/run_release_qa.py",
        "scripts/compare_cv_artifacts.py", "validation/development_descriptor_audit.json",
        "validation/heldout_descriptor_audit.json", "validation/full_nested_cv_reproduction_check.json",
        "validation/training_export_reproduction_check.json",
        "validation/v1_0_to_v1_1_six_target_regression_check.json",
        "artifacts/cv/fold_manifest.csv",
        "artifacts/cv/outer_fold_scores.csv", "artifacts/cv/oof_predictions.csv",
        "artifacts/cv/model_target_summary.csv", "artifacts/cv/provenance_stratified_metrics.csv",
        "artifacts/cv/y02_direct_vs_physical_baseline.csv", "artifacts/cv/protocol.json",
        "artifacts/cv/xgboost_reference_vs_deployment_consistency.csv",
        "artifacts/cv/qa.json", "scripts/build_model_consistency_table.py",
        "scripts/build_model_artifact_manifest.py", "scripts/build_training_export_reproduction_check.py",
        "scripts/compare_model_versions.py",
        "RELEASE_CHECKS.json",
    ]
    missing_required = [name for name in required if not (ROOT / name).is_file()]

    artifact_manifest = json.loads(ARTIFACT_MANIFEST.read_text(encoding="utf-8"))
    public = artifact_manifest["public_deidentified_artifacts"]
    public_joblib_path = ROOT / public["joblib_path"]
    public_browser_bundle = ROOT / public["browser_model_bundle_js_path"]
    public_model_json = ROOT / public["model_bundle_json_path"]
    public_wide_predictions = ROOT / public["wide_holdout_predictions_path"]
    public_long_predictions = ROOT / public["long_holdout_predictions_path"]
    sanitization_checks_path = ROOT / public["sanitization_check_path"]

    public_joblib = joblib.load(public_joblib_path)
    public_browser_json = json.loads(public_model_json.read_text(encoding="utf-8"))
    sanitization_checks = json.loads(sanitization_checks_path.read_text(encoding="utf-8"))
    evaluation_checks = json.loads(
        (ROOT / "validation" / "held_out_evaluation_checks.json").read_text(encoding="utf-8")
    )
    held_out_summary = json.loads(
        (ROOT / "validation" / "held_out_test_summary.json").read_text(encoding="utf-8")
    )
    held_out_reproduction = json.loads(
        (ROOT / "validation" / "held_out_reproduction_check.json").read_text(encoding="utf-8")
    )
    nested_cv_reproduction = json.loads(
        (ROOT / "validation" / "full_nested_cv_reproduction_check.json").read_text(encoding="utf-8")
    )
    training_export_reproduction = json.loads(
        (ROOT / "validation" / "training_export_reproduction_check.json").read_text(encoding="utf-8")
    )
    model_version_comparison = json.loads(
        (ROOT / "validation" / "v1_0_to_v1_1_six_target_regression_check.json").read_text(
            encoding="utf-8"
        )
    )
    data_release_checks = json.loads(
        (ROOT / "data" / "data_release_checks.json").read_text(encoding="utf-8")
    )
    development_descriptor_check = json.loads(
        (ROOT / "validation" / "development_descriptor_audit.json").read_text(encoding="utf-8")
    )
    heldout_descriptor_check = json.loads(
        (ROOT / "validation" / "heldout_descriptor_audit.json").read_text(encoding="utf-8")
    )
    cv_protocol = json.loads((ROOT / "artifacts" / "cv" / "protocol.json").read_text(encoding="utf-8"))
    cv_qa = json.loads((ROOT / "artifacts" / "cv" / "qa.json").read_text(encoding="utf-8"))

    wide_rows = read_csv(public_wide_predictions)
    long_rows = read_csv(public_long_predictions)
    metric_rows = read_csv(ROOT / "validation" / "held_out_test_metrics.csv")
    source_metric_rows = read_csv(ROOT / "validation" / "held_out_source_supported_metrics.csv")
    development_rows = read_csv(ROOT / "data" / "development.csv")
    held_out_rows = read_csv(ROOT / "data" / "held_out_test.csv")
    provenance_rows = read_csv(ROOT / "data" / "provenance_long.csv")
    reference_rows = read_csv(ROOT / "data" / "references.csv")
    fold_rows = read_csv(ROOT / "data" / "cv_fold_manifest.csv")
    cv_oof_rows = read_csv(ROOT / "artifacts" / "cv" / "oof_predictions.csv")
    consistency_rows = read_csv(
        ROOT / "artifacts" / "cv" / "xgboost_reference_vs_deployment_consistency.csv"
    )

    y02_development = [row for row in development_rows if finite(row.get("Y02"))]
    y02_heldout = [row for row in held_out_rows if finite(row.get("Y02"))]
    y02_provenance = [row for row in provenance_rows if row.get("property_code") == "Y02"]
    y02_long = [row for row in long_rows if row.get("target") == "Y02"]
    y02_long_available = [row for row in y02_long if truth(row.get("label_available")) is True]
    y02_long_missing = [row for row in y02_long if truth(row.get("label_available")) is False]
    metric_by_target = {row["target"]: row for row in metric_rows}
    source_metric_by_target = {row["target"]: row for row in source_metric_rows}

    y02_differences = [
        abs(float(row["Y02_pred"]) - float(row["Y01_pred"]) * float(row["Y03_pred"]))
        for row in wide_rows
        if all(finite(row.get(field)) for field in ("Y01_pred", "Y02_pred", "Y03_pred"))
    ]
    y02_max_difference = max(y02_differences, default=0.0)

    public_records = public_browser_json.get("ood", {}).get("training_records", [])
    browser_model_keys = set(public_browser_json.get("models", {}))
    joblib_model_keys = set(public_joblib.get("models", {}))
    browser_metadata_reduced = (
        len(public_records) == 300
        and all(
            set(record) == {"record_id", "x"}
            and record["record_id"] == f"DEV-{index:04d}"
            and len(record["x"]) == 7
            and all(finite(value) for value in record["x"])
            for index, record in enumerate(public_records, start=1)
        )
        and all(
            "name_zh" not in entry
            for table in (public_browser_json.get("input_meta", {}), public_browser_json.get("output_meta", {}))
            for entry in table.values()
        )
    )

    semantic = sanitization_checks.get("semantic_equivalence", {})
    semantic_check_ok = (
        sanitization_checks.get("record_type")
        == "self-generated release-integrity record; not an independent audit"
        and semantic.get("canonical_scientific_payload_equal") is True
        and semantic.get("trained_estimator_joblib_hashes_equal") is True
        and set(semantic.get("trained_estimator_hashes", {})) == set(TARGETS)
        and semantic.get("predictions_exact_for_all_checked_rows") is True
        and semantic.get("prediction_rows_checked") == 310
        and semantic.get("public_record_schema_valid") is True
        and semantic.get("name_zh_absent") is True
    )

    forbidden_nearest_fields = {
        "nearest_training_record", "nearest_training_molecule", "nearest_training_formula",
        "nearest_development_molecule", "nearest_development_formula",
    }
    wide_fields = set(wide_rows[0]) if wide_rows else set()
    long_fields = set(long_rows[0]) if long_rows else set()
    nearest_ids = {row.get("nearest_development_record", "") for row in wide_rows + long_rows}

    if args.node:
        node_result = subprocess.run(
            [str(args.node), str(ROOT / "tests" / "test_js_against_python_reference_cases.js")],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        six_case_agreement_ok = node_result.returncode == 0
        six_case_agreement_output = (node_result.stdout + node_result.stderr).strip()
    else:
        six_case_agreement_ok = False
        six_case_agreement_output = "Node.js was not found. Supply --node /path/to/node."

    html_reference_count, missing_html_references = check_html_references()
    marker_hits, cjk_hits = scan_public_text()
    citation_text = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    zenodo = json.loads((ROOT / ".zenodo.json").read_text(encoding="utf-8"))
    version_text = (ROOT / "VERSION.txt").read_text(encoding="utf-8")
    workflow_text = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted((ROOT / ".github" / "workflows").glob("*.yml"))
    )
    current_narrative_paths = [
        ROOT / "index.html",
        ROOT / "README.md",
        ROOT / "README.html",
        ROOT / "VERSION.txt",
        ROOT / "data" / "README.md",
        ROOT / "models" / "MODEL_CARD.md",
    ]
    current_narratives = {
        path.relative_to(ROOT).as_posix(): path.read_text(encoding="utf-8")
        for path in current_narrative_paths
    }
    forbidden_y02_definition = re.compile(
        r"Y02\s+(?:is\s+)?(?:calculated|derived|constructed)\s+from\s+"
        r"(?:the\s+)?(?:predicted\s+)?Y01|"
        r"predicted\s+Y02\s*=\s*predicted\s+Y01|"
        r"\|\s*Y02\s*\|[^\n]*\|\s*Derived\s+from\s+Y01",
        re.IGNORECASE,
    )
    forbidden_y02_definition_files = sorted(
        name for name, text in current_narratives.items() if forbidden_y02_definition.search(text)
    )
    stale_current_version = re.compile(
        r"SAF-Predict\s+v1\.0\.0|Version:\s*1\.0\.0|"
        r'"version"\s*:\s*"1\.0\.0"',
        re.IGNORECASE,
    )
    stale_current_version_files = sorted(
        name for name, text in current_narratives.items() if stale_current_version.search(text)
    )
    model_card_text = current_narratives["models/MODEL_CARD.md"]

    source_expected_n = {
        target: sum(
            row.get("property_code") == target
            and finite(row.get("final_value"))
            and truth(row.get("high_confidence_source_subset")) is True
            for row in provenance_rows
            if row.get("partition") == "internal_heldout_test"
        )
        for target in TARGETS
    }
    source_metric_n_matches = all(
        int(float(source_metric_by_target[target].get("n", -1))) == source_expected_n[target]
        for target in TARGETS
        if target in source_metric_by_target
    ) and set(source_metric_by_target) == set(TARGETS)

    training_export_seeds = training_export_reproduction.get("training", {}).get("deployment_seeds", {})
    training_export_model_check = (
        training_export_reproduction.get("status") == "pass"
        and training_export_reproduction.get("python_artifact", {}).get("prediction_equivalence_passed") is True
        and training_export_reproduction.get("browser_artifact", {}).get("prediction_equivalence_passed") is True
        and set(training_export_seeds) == set(TARGETS)
        and training_export_seeds.get("Y02") == 20269837
    )

    checks = {
        "required_v1_1_release_files_present": not missing_required,
        "public_joblib_hash_matches_artifact_manifest": hash_matches(public_joblib_path, public.get("joblib_sha256")),
        "public_browser_bundle_hash_matches_artifact_manifest": hash_matches(public_browser_bundle, public.get("browser_model_bundle_js_sha256")),
        "public_model_json_hash_matches_artifact_manifest": hash_matches(public_model_json, public.get("model_bundle_json_sha256")),
        "public_wide_prediction_hash_matches_artifact_manifest": hash_matches(public_wide_predictions, public.get("wide_holdout_predictions_sha256")),
        "public_long_prediction_hash_matches_artifact_manifest": hash_matches(public_long_predictions, public.get("long_holdout_predictions_sha256")),
        "public_sanitization_check_hash_matches_artifact_manifest": hash_matches(sanitization_checks_path, public.get("sanitization_check_sha256")),
        "public_sanitization_record_reports_scoped_semantic_equivalence": semantic_check_ok,
        "public_browser_bundle_metadata_reduced": browser_metadata_reduced,
        "private_source_joblib_absent_from_public_release": not (ROOT / "models" / "SAF_Predict_model_bundle_v1.joblib").exists(),
        "validation_nearest_development_metadata_reduced": (
            not (wide_fields | long_fields).intersection(forbidden_nearest_fields)
            and "nearest_development_record" in wide_fields
            and "nearest_development_record" in long_fields
            and bool(nearest_ids)
            and all(re.fullmatch(r"DEV-\d{4}", value) for value in nearest_ids)
        ),
        "held_out_metrics_hash_matches_evaluation_checks": hash_matches(
            ROOT / "validation" / "held_out_test_metrics.csv",
            evaluation_checks.get("public_deidentified_artifacts", {}).get("held_out_test_metrics_sha256"),
        ),
        "held_out_source_supported_metrics_hash_matches_evaluation_checks": hash_matches(
            ROOT / "validation" / "held_out_source_supported_metrics.csv",
            evaluation_checks.get("public_deidentified_artifacts", {}).get("source_supported_metrics_sha256"),
        ),
        "held_out_summary_is_consistent_with_v1_1_bundle": (
            held_out_summary.get("version") == VERSION
            and held_out_summary.get("n_records") == 10
            and held_out_summary.get("Y02_reference_labels") == 8
            and held_out_summary.get("public_deidentified_artifacts", {}).get("joblib_sha256")
            == sha256(public_joblib_path)
        ),
        "held_out_evaluation_metadata_confirms_direct_y02": (
            evaluation_checks.get("all_listed_checks_passed") is True
            and evaluation_checks.get("bundle_schema_version") == 2
            and set(evaluation_checks.get("separately_fitted_model_targets", [])) == set(TARGETS)
            and evaluation_checks.get("label_available_by_target", {}).get("Y02") == 8
            and evaluation_checks.get("Y02_separately_modelled_canary") is True
        ),
        "browser_and_joblib_each_contain_seven_direct_target_models": (
            browser_model_keys == set(TARGETS) and joblib_model_keys == set(TARGETS)
        ),
        "browser_and_joblib_schema_version_is_2": (
            public_browser_json.get("schema_version") == 2 and public_joblib.get("schema_version") == 2
        ),
        "y02_deployment_model_is_random_forest": (
            public_browser_json.get("deployment_models", {}).get("Y02") == "RF"
            and public_joblib.get("deployment_models", {}).get("Y02") == "RF"
            and public_browser_json.get("models", {}).get("Y02", {}).get("kind") == "random_forest"
        ),
        "y02_prediction_is_not_identically_y01_times_y03": y02_max_difference > 1e-6,
        "held_out_prediction_rows_equal_10": len(wide_rows) == 10,
        "held_out_record_property_rows_equal_70": len(long_rows) == 70,
        "held_out_metric_rows_equal_7": len(metric_rows) == 7 and set(metric_by_target) == set(TARGETS),
        "held_out_source_supported_metric_rows_equal_7": len(source_metric_rows) == 7,
        "held_out_all_point_predictions_finite": all(
            finite(row.get(f"{target}_pred")) for row in wide_rows for target in TARGETS
        ),
        "held_out_y02_metric_uses_8_available_labels": int(float(metric_by_target.get("Y02", {}).get("n", -1))) == 8,
        "held_out_y02_long_rows_mark_8_available_and_2_missing": (
            len(y02_long) == 10 and len(y02_long_available) == 8 and len(y02_long_missing) == 2
        ),
        "held_out_missing_y02_labels_have_no_numeric_error_or_coverage": all(
            not finite(row.get("actual"))
            and not finite(row.get("residual_actual_minus_predicted"))
            and truth(row.get("covered90")) is False
            and truth(row.get("covered95")) is False
            for row in y02_long_missing
        ),
        "source_supported_metric_counts_match_property_provenance": source_metric_n_matches,
        "development_records_equal_300": len(development_rows) == 300,
        "internal_held_out_records_equal_10": len(held_out_rows) == 10,
        "record_property_provenance_rows_equal_2170": len(provenance_rows) == 2170,
        "reference_registry_rows_equal_492": len(reference_rows) == 492,
        "fixed_outer_fold_manifest_rows_equal_300": len(fold_rows) == 300,
        "development_records_have_73_formula_groups": len({row["molecular_formula"] for row in development_rows}) == 73,
        "development_y02_has_162_labels_in_66_formula_groups": (
            len(y02_development) == 162
            and len({row["molecular_formula"] for row in y02_development}) == 66
            and len(development_rows) - len(y02_development) == 138
        ),
        "held_out_y02_has_8_labels_and_2_missing": len(y02_heldout) == 8 and len(held_out_rows) - len(y02_heldout) == 2,
        "y02_status_columns_match_availability": all(
            (finite(row.get("Y02")) and "author-curated independent response" in row.get("Y02_label_status", ""))
            or (not finite(row.get("Y02")) and row.get("Y02_label_status") == "not available")
            for row in development_rows + held_out_rows
        ),
        "y02_provenance_has_170_independent_and_140_unavailable_rows": (
            len(y02_provenance) == 310
            and sum(row.get("value_origin_class") == "author-curated independently collected response" for row in y02_provenance) == 170
            and sum(row.get("value_origin_class") == "not available" for row in y02_provenance) == 140
        ),
        "y02_provenance_contains_no_constructed_target_identity": all(
            "construct" not in row.get("value_origin_class", "").lower()
            and "derived" not in row.get("value_origin_class", "").lower()
            for row in y02_provenance
        ),
        "release_data_invariants_pass": (
            data_release_checks.get("pass") is True
            and data_release_checks.get("release_version") == VERSION
            and data_release_checks.get("y02_independent_labels_development") == 162
            and data_release_checks.get("y02_independent_labels_held_out") == 8
        ),
        "cv_archive_uses_direct_y02_on_162_labels": (
            cv_protocol.get("target_records", {}).get("Y02") == 162
            and cv_protocol.get("target_formula_groups", {}).get("Y02") == 66
            and "independently collected response" in cv_protocol.get("y02_policy", "")
            and len(cv_oof_rows) == 7848
            and cv_qa.get("all_checks_passed") is True
            and cv_qa.get("checks", {}).get("y02_is_directly_modelled_on_available_labels") is True
        ),
        "xgboost_reference_vs_deployment_consistency_table_is_complete": (
            len(consistency_rows) == 7
            and {row.get("target") for row in consistency_rows} == set(TARGETS)
            and all(
                int(float(row.get("n", -1))) == (162 if row.get("target") == "Y02" else 300)
                and finite(row.get("prediction_pearson_r_xgboost_vs_deployment"))
                and finite(row.get("delta_r2_deployment_minus_xgboost"))
                for row in consistency_rows
            )
        ),
        "development_descriptor_recalculation_passes": development_descriptor_check.get("status") == "pass",
        "held_out_descriptor_recalculation_passes": heldout_descriptor_check.get("status") == "pass",
        "held_out_reproduction_matches_released_v1_1_results": (
            held_out_reproduction.get("status") == "pass"
            and held_out_reproduction.get("Y02_available_reference_labels") == 8
            and held_out_reproduction.get("Y02_separately_modelled_canary") is True
        ),
        "full_nested_cv_reproduction_matches_archived_artifacts": (
            nested_cv_reproduction.get("status") == "pass"
            and nested_cv_reproduction.get("all_comparisons_passed") is True
            and nested_cv_reproduction.get("numeric_tolerance") == 1e-12
        ),
        "training_export_recovers_released_seven_model_predictions": training_export_model_check,
        "v1_0_to_v1_1_unchanged_target_regression_check_passes": (
            model_version_comparison.get("all_checks_passed") is True
            and model_version_comparison.get("rows_checked") == 310
            and set(model_version_comparison.get("unchanged_target_predictions_within_tolerance", {}))
            == {"Y01", "Y03", "Y04", "Y05", "Y06", "Y07"}
            and model_version_comparison.get("v1_1_Y02_separately_modelled_canary") is True
        ),
        "software_and_data_rights_files_present": (ROOT / "LICENSE").is_file() and (ROOT / "DATA_LICENSE.md").is_file(),
        "citation_zenodo_and_version_metadata_are_v1_1_0": (
            "version: 1.1.0" in citation_text
            and "Version 1.1.0" in zenodo.get("description", "")
            and zenodo.get("version") == VERSION
            and version_text.startswith("SAF-Predict 1.1.0")
            and artifact_manifest.get("version") == VERSION
        ),
        "current_narrative_files_do_not_define_y02_as_constructed": not forbidden_y02_definition_files,
        "current_narrative_files_have_no_stale_v1_0_identity": not stale_current_version_files,
        "human_model_card_matches_direct_y02_release": (
            "Version: 1.1.0" in model_card_text
            and "| Y02 | Volumetric net heat of combustion | RF | 162 / 66 |" in model_card_text
            and "Y01 × Y03 product is retained solely as a physical-baseline audit" in model_card_text
            and "8 records with a retained Y02 reference label" in model_card_text
        ),
        "github_full_reproduction_uses_v1_1_0": "--version 1.1.0" in workflow_text and "--version 1.0.0" not in workflow_text,
        "javascript_matches_stored_python_reference_on_six_cases": six_case_agreement_ok,
        "local_html_references_resolve": not missing_html_references,
        "release_text_has_no_placeholder_or_local_path_markers": not marker_hits,
        "release_text_has_no_cjk_metadata": not cjk_hits,
        "generated_report_has_no_malformed_empty_class_attribute": "class=>" not in (
            ROOT / "validation" / "held_out_test_report.html"
        ).read_text(encoding="utf-8"),
    }

    if args.verify_release_manifest:
        manifest_verification = verify_release_manifest_and_checksums()
        checks["final_release_manifest_and_checksums_valid"] = bool(manifest_verification["passed"])
    else:
        manifest_verification = {
            "status": "not_run",
            "note": (
                "Run scripts/build_release_manifest.py after this report is written, then "
                "run scripts/run_release_qa.py --verify-release-manifest."
            ),
        }

    report = {
        "platform": "SAF-Predict",
        "version": VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "check_scope": (
            "Released-file integrity and internal consistency only; not an independent audit, "
            "browser/operating-system compatibility certification, or proof of analysis chronology."
        ),
        "all_listed_checks_passed": all(checks.values()),
        "checks": checks,
        "details": {
            "missing_required_files": missing_required,
            "held_out_prediction_rows": len(wide_rows),
            "held_out_record_property_rows": len(long_rows),
            "held_out_metric_rows": len(metric_rows),
            "held_out_source_supported_metric_rows": len(source_metric_rows),
            "development_records": len(development_rows),
            "internal_held_out_records": len(held_out_rows),
            "development_y02_labels": len(y02_development),
            "held_out_y02_labels": len(y02_heldout),
            "provenance_rows": len(provenance_rows),
            "reference_rows": len(reference_rows),
            "fixed_fold_rows": len(fold_rows),
            "cv_oof_rows": len(cv_oof_rows),
            "xgboost_reference_vs_deployment_consistency_rows": len(consistency_rows),
            "maximum_abs_y02_prediction_minus_y01_times_y03": y02_max_difference,
            "html_references_checked": html_reference_count,
            "missing_html_references": missing_html_references,
            "placeholder_or_local_path_marker_files": marker_hits,
            "cjk_metadata_files": cjk_hits,
            "forbidden_constructed_y02_definition_files": forbidden_y02_definition_files,
            "stale_current_version_identity_files": stale_current_version_files,
            "javascript_python_six_case_agreement_output": six_case_agreement_output,
            "computed_public_artifact_sha256": {
                "joblib": sha256(public_joblib_path) if public_joblib_path.is_file() else None,
                "browser_model_bundle_js": sha256(public_browser_bundle) if public_browser_bundle.is_file() else None,
                "model_bundle_json": sha256(public_model_json) if public_model_json.is_file() else None,
                "wide_holdout_predictions": sha256(public_wide_predictions) if public_wide_predictions.is_file() else None,
                "long_holdout_predictions": sha256(public_long_predictions) if public_long_predictions.is_file() else None,
            },
            "release_manifest_verification": manifest_verification,
        },
        "author_reported_analysis_design": {
            "development_data": "data/development.csv n=300; 73 molecular-formula groups",
            "y02_training_data": "n=162 available independently collected labels; 66 molecular-formula groups; missing labels not imputed",
            "held_out_role": "data/held_out_test.csv n=10; structurally stratified internal held-out test set; Y02 labels n=8",
            "held_out_used_for_tuning": False,
            "post_test_refitting_performed": False,
            "independently_timestamped_pre_evaluation_release_present_in_repository": False,
            "note": "These are author-reported workflow statements and are not automated integrity-check results.",
        },
        "y02_rule": "separately fitted random-forest prediction; Y01 multiplied by Y03 is an audit baseline only",
    }
    if not args.verify_release_manifest:
        RELEASE_CHECKS.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    print(json.dumps({"all_listed_checks_passed": report["all_listed_checks_passed"], "checks": checks}))
    raise SystemExit(0 if report["all_listed_checks_passed"] else 1)


if __name__ == "__main__":
    main()
