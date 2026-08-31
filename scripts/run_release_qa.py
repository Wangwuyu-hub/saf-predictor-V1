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
        if ".git" in html_path.parts:
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
        if not path.is_file() or ".git" in path.parts:
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
    args = parser.parse_args()

    if not RELEASE_CHECKS.is_file():
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
        "y02_identity_holds_within_1e-12": y02_max_abs_error <= 1e-12,
        "javascript_matches_stored_python_reference_on_six_cases": six_case_agreement_ok,
        "local_html_references_resolve": not missing_references,
        "release_text_has_no_placeholder_or_local_path_markers": not marker_hits,
        "release_text_has_no_cjk_metadata": not cjk_hits,
        "generated_report_has_no_malformed_empty_class_attribute": "class=>" not in (
            ROOT / "validation" / "held_out_test_report.html"
        ).read_text(encoding="utf-8"),
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
        },
        "author_reported_analysis_design": {
            "development_data": "Sheet1 n=300",
            "held_out_role": "Sheet2 n=10, internal held-out test set",
            "sheet2_used_for_tuning": False,
            "post_test_refitting_performed": False,
            "independently_timestamped_pre_evaluation_release_present_in_repository": False,
            "note": "These are author-reported workflow statements and are not automated integrity-check results."
        },
        "y02_rule": "predicted Y01 multiplied by predicted Y03",
    }
    RELEASE_CHECKS.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({"all_listed_checks_passed": report["all_listed_checks_passed"], "checks": checks}))
    raise SystemExit(0 if report["all_listed_checks_passed"] else 1)


if __name__ == "__main__":
    main()
