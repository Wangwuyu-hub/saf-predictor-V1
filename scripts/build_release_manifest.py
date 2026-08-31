#!/usr/bin/env python3
"""Create SHA256 manifests for the public SAF-Predict release files."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "release_manifest.json"
CHECKSUMS = ROOT / "checksums.sha256"
ARTIFACT_MANIFEST = ROOT / "models" / "model_artifact_manifest.json"

EXCLUDED_PUBLIC_PATHS = {
    "models/SAF_Predict_model_bundle_v1.joblib",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def public_files() -> list[Path]:
    excluded_names = {MANIFEST.name, CHECKSUMS.name}
    excluded_parts = {".git", "__pycache__", ".venv", "node_modules"}
    return [
        path
        for path in sorted(ROOT.rglob("*"))
        if path.is_file()
        and path.name not in excluded_names
        and not excluded_parts.intersection(path.parts)
        and path.relative_to(ROOT).as_posix() not in EXCLUDED_PUBLIC_PATHS
    ]


def main() -> None:
    artifact_metadata = json.loads(ARTIFACT_MANIFEST.read_text(encoding="utf-8"))
    hashes = {path.relative_to(ROOT).as_posix(): sha256(path) for path in public_files()}
    browser_program = [
        "index.html",
        "styles.css",
        "app.js",
        "predictor_core.js",
        "model_bundle.js",
        "held_out_test_data.js",
    ]
    model_and_evaluation_paths = [
        "models/SAF_Predict_public_model_bundle_v1.joblib",
        "models/model_bundle.json",
        "models/public_artifact_sanitization_checks.json",
        "model_bundle.js",
        "validation/held_out_predictions.csv",
        "validation/held_out_test_predictions_long.csv",
        "validation/held_out_test_metrics.csv",
        "validation/held_out_test_summary.json",
        "validation/held_out_evaluation_checks.json",
        "validation/held_out_test_report.html",
        "RELEASE_CHECKS.json",
    ]
    missing_expected = [
        name for name in browser_program + model_and_evaluation_paths if name not in hashes
    ]
    if missing_expected:
        raise FileNotFoundError(f"Expected public release files are missing: {missing_expected}")

    manifest = {
        "platform": "SAF-Predict",
        "version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Every public release file listed below is content-addressed; checksums.sha256 additionally covers this manifest.",
        "claim_boundary": "File-content identification and release integrity only; not independent audit, browser/operating-system compatibility certification, or proof of analysis chronology.",
        "file_count_excluding_manifest_and_checksum_list": len(hashes),
        "browser_prediction_program": {name: hashes[name] for name in browser_program},
        "model_and_evaluation_artifacts": {
            name: hashes[name] for name in model_and_evaluation_paths
        },
        "recorded_private_source_artifact_identifiers": artifact_metadata[
            "source_artifact_identifiers"
        ],
        "all_files": hashes,
    }
    MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    checksum_paths = [path for path in public_files() if path != CHECKSUMS] + [MANIFEST]
    checksum_paths = sorted(set(checksum_paths))
    CHECKSUMS.write_text(
        "\n".join(
            f"{sha256(path)} *{path.relative_to(ROOT).as_posix()}"
            for path in checksum_paths
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "files": len(checksum_paths),
                "manifest": str(MANIFEST),
                "checksums": str(CHECKSUMS),
            }
        )
    )


if __name__ == "__main__":
    main()
