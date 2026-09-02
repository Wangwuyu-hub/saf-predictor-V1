#!/usr/bin/env python3
"""Run scoped release-integrity and six-case numerical-agreement checks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_MANIFEST = ROOT / "models" / "model_artifact_manifest.json"
RELEASE_CHECKS = ROOT / "RELEASE_CHECKS.json"
RELEASE_MANIFEST = ROOT / "release_manifest.json"
CHECKSUMS = ROOT / "checksums.sha256"
EXCLUDED_PUBLIC_PATHS = {
    "models/SAF_Predict_model_bundle_v1.joblib",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class LocalReferenceParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.references: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attribute = "src" if tag in {"script", "img"} else "href" if tag in {"a", "link"} else None
        if not attribute:
            return
        for name, value in attrs:
            if name == attribute and value:
                self.references.append(value)


def check_html_references() -> tuple[int, list[str]]:
    checked = 0
    missing: list[str] = []
    for html_path in sorted(ROOT.rglob("*.html")):
        if any(part in {".git", "run_outputs", ".venv", "venv", "node_modules"} for part in html_path.parts):
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


def hash_matches(path: Path, expected: object) -> bool:
    return path.is_file() and isinstance(expected, str) and len(expected) == 64 and sha256(path) == expected


def public_release_files() -> list[Path]:
    """Return the files that build_release_manifest.py must content-address.

    The manifest and checksum list are deliberately excluded: the checksum list
    covers the manifest, but never itself. This gives a finite, non-circular
    integrity chain.
    """
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
    """Read-only verification of the final content-addressed release state."""
    details: dict[str, object] = {
        "manifest_present": RELEASE_MANIFEST.is_file(),
        "checksums_present": CHECKSUMS.is_file(),
        "manifest_json_valid": False,
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
    actual_paths = {
        path.relative_to(ROOT).as_posix(): path for path in public_release_files()
    }
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
    details["manifest_file_count_matches"] = manifest.get(
        "file_count_excluding_manifest_and_checksum_list"
    ) == len(actual_paths)

    checksum_entries: dict[str, str] = {}
    malformed_checksum_lines: list[str] = []
    duplicate_checksum_paths: list[str] = []
    for line_number, line in enumerate(CHECKSUMS.read_text(encoding="utf-8").splitlines(), start=1):
        match = re.fullmatch(r"([0-9a-f]{64}) \*(.+)", line)
        if not match:
            malformed_checksum_lines.append(str(line_number))
            continue
        digest, name = match.groups()
        if name in checksum_entries:
            duplicate_checksum_paths.append(name)
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
    details["malformed_checksum_lines"] = malformed_checksum_lines
    details["duplicate_checksum_paths"] = sorted(set(duplicate_checksum_paths))
    details["missing_checksum_paths"] = missing_checksum_paths
    details["unexpected_checksum_paths"] = unexpected_checksum_paths
    details["checksum_hash_mismatches"] = checksum_hash_mismatches
    details["checksum_paths_complete"] = (
        not malformed_checksum_lines
        and not duplicate_checksum_paths
        and not missing_checksum_paths
        and not unexpected_checksum_paths
    )
    details["checksum_hashes_match"] = not checksum_hash_mismatches
    details["passed"] = all(
        bool(details[name])
        for name in (
            "manifest_json_valid",
            "manifest_file_count_matches",
            "manifest_paths_complete",
            "manifest_hashes_match",
            "checksum_paths_complete",
            "checksum_hashes_match",
        )
    )
    return details


def scan_public_text() -> tuple[list[str], list[str]]:
    suffixes = {".html", ".md", ".json", ".js", ".css", ".py", ".txt", ".csv", ".sha256"}
    special_names = {".gitattributes", ".gitignore", ".nojekyll"}
    marker_pattern = re.compile(
        r"C:\\Users\\|pending\s+author\s+confirmation|\bTODO\b|\bFIXME\b|lorem\s+ipsum",
        re.IGNORECASE,
    )
    cjk_pattern = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
    marker_hits: list[str] = []
    cjk_hits: list[str] = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or any(
            part in {".git", "run_outputs", ".venv", "venv", "node_modules", "__pycache__"}
            for part in path.parts
        ):
            continue
        if path.resolve() == Path(__file__).resolve():
            continue  # This script contains the marker patterns used for the scan.
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
        "models/model_bundle.json", "models/SAF_Predict_public_model_bundle_v1.joblib",
        "models/model_card.json", "models/model_artifact_manifest.json", "models/MODEL_CARD.md",
        "models/public_artifact_sanitization_checks.json", "scripts/sanitize_public_bundle.py",
        "validation/held_out_predictions.csv",
        "validation/held_out_test_predictions_long.csv", "validation/held_out_test_metrics.csv",
        "validation/held_out_test_summary.json", "validation/held_out_evaluation_checks.json",
        "validation/held_out_test_report.html", "examples/batch_input_template.csv",
        "tests/test_js_against_python_reference_cases.js",
        "tests/python_reference_predictions_6cases.json",
        "LICENSE", "DATA_LICENSE.md", "CITATION.cff", ".zenodo.json",
        "requirements.txt", "requirements-lock.txt", "environment.yml",
        "REPRODUCIBILITY.md", "data/README.md",
        "data/development.csv", "data/held_out_test.csv",
        "data/provenance_long.csv", "data/references.csv",
        "data/data_dictionary.csv", "data/cv_fold_manifest.csv",
        "data/data_release_checks.json", "data/SAF_Hydrocarbon_Dataset_v1.0.0.xlsx",
        "training/saf_reproduce.py", "training/config.json",
        "training/input_schema.csv", "training/README.md",
        "scripts/compute_descriptors.py", "scripts/evaluate_heldout.py",
        "scripts/compare_cv_artifacts.py",
        "validation/development_descriptor_audit.json",
        "validation/heldout_descriptor_audit.json",
        "validation/held_out_reproduction_check.json",
        "validation/full_nested_cv_reproduction_check.json",
        "validation/training_export_reproduction_check.json",
        "artifacts/cv/fold_manifest.csv", "artifacts/cv/outer_fold_scores.csv",
        "artifacts/cv/oof_predictions.csv", "artifacts/cv/model_target_summary.csv",
        "artifacts/cv/protocol.json", "artifacts/cv/qa.json",
        "RELEASE_CHECKS.json",
    ]
    missing_required = [name for name in required if not (ROOT / name).is_file()]

    manifest = json.loads(ARTIFACT_MANIFEST.read_text(encoding="utf-8"))
    public = manifest["public_deidentified_artifacts"]
    public_joblib = ROOT / public["joblib_path"]
    public_browser_bundle = ROOT / public["browser_model_bundle_js_path"]
    public_model_json = ROOT / public["model_bundle_json_path"]
    public_wide_predictions = ROOT / public["wide_holdout_predictions_path"]
    public_long_predictions = ROOT / public["long_holdout_predictions_path"]
    sanitization_checks_path = ROOT / public["sanitization_check_path"]
    sanitization_checks = json.loads(sanitization_checks_path.read_text(encoding="utf-8"))
    evaluation_checks = json.loads(
        (ROOT / "validation" / "held_out_evaluation_checks.json").read_text(encoding="utf-8")
    )
    public_browser_json = json.loads(public_model_json.read_text(encoding="utf-8"))

    with public_wide_predictions.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    y02_max_abs_error = max(
        abs(float(row["Y02_pred"]) - float(row["Y01_pred"]) * float(row["Y03_pred"]))
        for row in rows
    )
    with public_long_predictions.open(encoding="utf-8-sig", newline="") as handle:
        long_rows = list(csv.DictReader(handle))
    with (ROOT / "validation" / "held_out_test_metrics.csv").open(encoding="utf-8-sig", newline="") as handle:
        metric_rows = list(csv.DictReader(handle))
    with (ROOT / "data" / "development.csv").open(encoding="utf-8-sig", newline="") as handle:
        development_rows = list(csv.DictReader(handle))
    with (ROOT / "data" / "held_out_test.csv").open(encoding="utf-8-sig", newline="") as handle:
        held_out_rows = list(csv.DictReader(handle))
    with (ROOT / "data" / "provenance_long.csv").open(encoding="utf-8-sig", newline="") as handle:
        provenance_rows = list(csv.DictReader(handle))
    with (ROOT / "data" / "references.csv").open(encoding="utf-8-sig", newline="") as handle:
        reference_rows = list(csv.DictReader(handle))
    with (ROOT / "data" / "cv_fold_manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        fold_rows = list(csv.DictReader(handle))
    data_release_checks = json.loads((ROOT / "data" / "data_release_checks.json").read_text(encoding="utf-8"))
    development_descriptor_check = json.loads(
        (ROOT / "validation" / "development_descriptor_audit.json").read_text(encoding="utf-8")
    )
    held_out_descriptor_check = json.loads(
        (ROOT / "validation" / "heldout_descriptor_audit.json").read_text(encoding="utf-8")
    )
    held_out_reproduction = json.loads(
        (ROOT / "validation" / "held_out_reproduction_check.json").read_text(encoding="utf-8")
    )
    nested_cv_reproduction = json.loads(
        (ROOT / "validation" / "full_nested_cv_reproduction_check.json").read_text(
            encoding="utf-8"
        )
    )
    training_export_reproduction = json.loads(
        (ROOT / "validation" / "training_export_reproduction_check.json").read_text(
            encoding="utf-8"
        )
    )

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

    reference_count, missing_references = check_html_references()
    marker_hits, cjk_hits = scan_public_text()
    forbidden_nearest_fields = {
        "nearest_training_record", "nearest_training_molecule", "nearest_training_formula",
        "nearest_development_molecule", "nearest_development_formula",
    }
    wide_fields = set(rows[0]) if rows else set()
    long_fields = set(long_rows[0]) if long_rows else set()
    nearest_ids = {
        row.get("nearest_development_record", "") for row in rows + long_rows
    }
    public_records = public_browser_json["ood"]["training_records"]
    browser_metadata_reduced = (
        len(public_records) == 300
        and all(
            set(record) == {"record_id", "x"}
            and record["record_id"] == f"DEV-{index:04d}"
            and len(record["x"]) == 7
            for index, record in enumerate(public_records, start=1)
        )
        and all("name_zh" not in entry for table in (public_browser_json["input_meta"], public_browser_json["output_meta"]) for entry in table.values())
    )
    semantic = sanitization_checks.get("semantic_equivalence", {})
    semantic_check_ok = (
        sanitization_checks.get("record_type") == "self-generated release-integrity record; not an independent audit"
        and semantic.get("canonical_scientific_payload_equal") is True
        and semantic.get("trained_estimator_joblib_hashes_equal") is True
        and semantic.get("predictions_exact_for_all_checked_rows") is True
        and semantic.get("prediction_rows_checked") == 310
        and semantic.get("public_record_schema_valid") is True
        and semantic.get("name_zh_absent") is True
    )
    checks = {
        "required_release_files_present": not missing_required,
        "public_joblib_hash_matches_artifact_manifest": hash_matches(public_joblib, public.get("joblib_sha256")),
        "public_browser_bundle_hash_matches_artifact_manifest": hash_matches(
            public_browser_bundle, public.get("browser_model_bundle_js_sha256")
        ),
        "public_model_json_hash_matches_artifact_manifest": hash_matches(
            public_model_json, public.get("model_bundle_json_sha256")
        ),
        "public_wide_prediction_hash_matches_artifact_manifest": hash_matches(
            public_wide_predictions, public.get("wide_holdout_predictions_sha256")
        ),
        "public_long_prediction_hash_matches_artifact_manifest": hash_matches(
            public_long_predictions, public.get("long_holdout_predictions_sha256")
        ),
        "public_sanitization_check_hash_matches_artifact_manifest": hash_matches(
            sanitization_checks_path, public.get("sanitization_check_sha256")
        ),
        "public_sanitization_record_reports_scoped_semantic_equivalence": semantic_check_ok,
        "public_browser_bundle_metadata_reduced": browser_metadata_reduced,
        "private_source_joblib_absent_from_public_release": not (
            ROOT / "models" / "SAF_Predict_model_bundle_v1.joblib"
        ).exists(),
        "validation_nearest_development_metadata_reduced": (
            not (wide_fields | long_fields).intersection(forbidden_nearest_fields)
            and "nearest_development_record" in wide_fields
            and "nearest_development_record" in long_fields
            and bool(nearest_ids)
            and all(re.fullmatch(r"DEV-\d{4}", value) for value in nearest_ids)
        ),
        "held_out_metrics_hash_matches_evaluation_checks": hash_matches(
            ROOT / "validation" / "held_out_test_metrics.csv",
            evaluation_checks["public_deidentified_artifacts"].get("held_out_test_metrics_sha256"),
        ),
        "held_out_summary_hash_matches_evaluation_checks": hash_matches(
            ROOT / "validation" / "held_out_test_summary.json",
            evaluation_checks["public_deidentified_artifacts"].get("held_out_test_summary_sha256"),
        ),
        "held_out_prediction_rows_equal_10": len(rows) == 10,
        "held_out_record_property_rows_equal_70": len(long_rows) == 70,
        "held_out_metric_rows_equal_7": len(metric_rows) == 7,
        "development_records_equal_300": len(development_rows) == 300,
        "internal_held_out_records_equal_10": len(held_out_rows) == 10,
        "record_property_provenance_rows_equal_2170": len(provenance_rows) == 2170,
        "reference_registry_rows_equal_491": len(reference_rows) == 491,
        "fixed_outer_fold_manifest_rows_equal_300": len(fold_rows) == 300,
        "development_records_have_73_formula_groups": len({row["molecular_formula"] for row in development_rows}) == 73,
        "release_data_invariants_pass": data_release_checks.get("pass") is True,
        "development_descriptor_recalculation_passes": development_descriptor_check.get("status") == "pass",
        "held_out_descriptor_recalculation_passes": held_out_descriptor_check.get("status") == "pass",
        "held_out_reproduction_matches_archived_results": held_out_reproduction.get("status") == "pass",
        "full_nested_cv_reproduction_matches_archived_artifacts": (
            nested_cv_reproduction.get("status") == "pass"
            and nested_cv_reproduction.get("all_comparisons_passed") is True
            and nested_cv_reproduction.get("numeric_tolerance") == 1e-12
        ),
        "training_export_recovers_archived_predictions": (
            training_export_reproduction.get("status") == "pass"
            and training_export_reproduction.get("python_artifact", {}).get(
                "prediction_equivalence_passed"
            )
            is True
            and training_export_reproduction.get("browser_artifact", {}).get(
                "prediction_equivalence_passed"
            )
            is True
        ),
        "software_and_data_rights_files_present": (ROOT / "LICENSE").is_file() and (ROOT / "DATA_LICENSE.md").is_file(),
        "citation_and_zenodo_metadata_present": (ROOT / "CITATION.cff").is_file() and (ROOT / ".zenodo.json").is_file(),
        "y02_identity_holds_within_1e-12": y02_max_abs_error <= 1e-12,
        "javascript_matches_stored_python_reference_on_six_cases": six_case_agreement_ok,
        "local_html_references_resolve": not missing_references,
        "release_text_has_no_placeholder_or_local_path_markers": not marker_hits,
        "release_text_has_no_cjk_metadata": not cjk_hits,
        "generated_report_has_no_malformed_empty_class_attribute": "class=>" not in (
            ROOT / "validation" / "held_out_test_report.html"
        ).read_text(encoding="utf-8"),
    }
    manifest_verification: dict[str, object]
    if args.verify_release_manifest:
        manifest_verification = verify_release_manifest_and_checksums()
        checks["final_release_manifest_and_checksums_valid"] = bool(
            manifest_verification["passed"]
        )
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
        "version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "check_scope": "Released-file integrity and internal consistency only; not an independent audit, browser/operating-system compatibility certification, or proof of analysis chronology.",
        "all_listed_checks_passed": all(checks.values()),
        "checks": checks,
        "details": {
            "missing_required_files": missing_required,
            "held_out_prediction_rows": len(rows),
            "held_out_record_property_rows": len(long_rows),
            "held_out_metric_rows": len(metric_rows),
            "development_records": len(development_rows),
            "internal_held_out_records": len(held_out_rows),
            "provenance_rows": len(provenance_rows),
            "reference_rows": len(reference_rows),
            "fixed_fold_rows": len(fold_rows),
            "y02_max_abs_error": y02_max_abs_error,
            "html_references_checked": reference_count,
            "missing_html_references": missing_references,
            "placeholder_or_local_path_marker_files": marker_hits,
            "cjk_metadata_files": cjk_hits,
            "javascript_python_six_case_agreement_output": six_case_agreement_output,
            "computed_public_artifact_sha256": {
                "joblib": sha256(public_joblib) if public_joblib.is_file() else None,
                "browser_model_bundle_js": sha256(public_browser_bundle) if public_browser_bundle.is_file() else None,
                "model_bundle_json": sha256(public_model_json) if public_model_json.is_file() else None,
                "wide_holdout_predictions": sha256(public_wide_predictions) if public_wide_predictions.is_file() else None,
                "long_holdout_predictions": sha256(public_long_predictions) if public_long_predictions.is_file() else None,
            },
            "release_manifest_verification": manifest_verification,
        },
        "author_reported_analysis_design": {
            "development_data": "data/development.csv n=300; 73 molecular-formula groups",
            "held_out_role": "data/held_out_test.csv n=10; structurally stratified internal held-out test set",
            "sheet2_used_for_tuning": False,
            "post_test_refitting_performed": False,
            "independently_timestamped_pre_evaluation_release_present_in_repository": False,
            "note": "These are author-reported workflow statements and are not automated integrity-check results."
        },
        "y02_rule": "predicted Y01 multiplied by predicted Y03",
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
