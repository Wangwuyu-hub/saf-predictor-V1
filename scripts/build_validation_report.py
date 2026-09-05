#!/usr/bin/env python3
"""Regenerate the English internal-test report and browser data artifact."""

from __future__ import annotations

import csv
import html
import json
import math
import re
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[1]
EVALUATION = ROOT / "validation"


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def display(value: object) -> str:
    """Normalize machine-readable property/unit labels for the HTML report."""
    text = str(value)
    replacements = {
        "Solid-liquid": "Solid–liquid",
        "40 degC": "40 °C",
        "MJ kg^-1": "MJ kg−1",
        "MJ L^-1": "MJ L−1",
        "g cm^-3": "g cm−3",
        "degC": "°C",
        "mm^2 s^-1": "mm² s−1",
    }
    for source, replacement in replacements.items():
        text = text.replace(source, replacement)
    return text


def number(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def fmt(value: object, digits: int = 3, missing: str = "—") -> str:
    parsed = number(value)
    return missing if parsed is None else f"{parsed:.{digits}f}"


def first(row: dict[str, str], *names: str, default: str = "") -> str:
    for name in names:
        value = row.get(name)
        if value not in {None, ""}:
            return str(value)
    return default


def truth(value: object) -> bool | None:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    return None


def yes_no(value: object, available: bool) -> str:
    if not available:
        return "—"
    parsed = truth(value)
    return "yes" if parsed is True else "no" if parsed is False else "—"


def coverage(row: dict[str, str], level: str) -> str:
    count = number(row.get(f"coverage{level}_count"))
    n = number(row.get(f"coverage{level}_n", row.get("n")))
    fraction = number(row.get(f"coverage{level}_fraction"))
    if count is None or n is None or n <= 0:
        return "—"
    if fraction is None:
        fraction = count / n
    lower = number(row.get(f"coverage{level}_exact95_lower"))
    upper = number(row.get(f"coverage{level}_exact95_upper"))
    interval = "" if lower is None or upper is None else f"; 95% CI {lower:.2f}–{upper:.2f}"
    return f"{int(count)}/{int(n)} ({100 * fraction:.1f}%{interval})"


def metric_with_bootstrap_ci(row: dict[str, str], metric: str) -> str:
    point = fmt(row.get(metric))
    lower = number(row.get(f"{metric}_group_bootstrap_ci95_lower"))
    upper = number(row.get(f"{metric}_group_bootstrap_ci95_upper"))
    if lower is None or upper is None:
        return point
    return f"{point} [{lower:.3f}, {upper:.3f}]"


def display_hash(value: object) -> str:
    text = str(value or "").strip()
    return text if re.fullmatch(r"[0-9a-f]{64}", text) else "not recorded in this artifact"


def release_version() -> str:
    text = (ROOT / "VERSION.txt").read_text(encoding="utf-8").splitlines()[0]
    match = re.search(r"\d+\.\d+\.\d+", text)
    return match.group(0) if match else "1.1.0"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


metrics = read_rows(EVALUATION / "held_out_test_metrics.csv")
source_supported_metrics = read_rows(EVALUATION / "held_out_source_supported_metrics.csv")
predictions = read_rows(EVALUATION / "held_out_test_predictions_long.csv")
development = read_rows(ROOT / "data" / "development.csv")
held_out = read_rows(ROOT / "data" / "held_out_test.csv")
provenance = read_rows(ROOT / "data" / "provenance_long.csv")

artifact_manifest = json.loads(
    (ROOT / "models" / "model_artifact_manifest.json").read_text(encoding="utf-8")
)
summary = json.loads((EVALUATION / "held_out_test_summary.json").read_text(encoding="utf-8"))

version = release_version()
metric_by_target = {row["target"]: row for row in metrics}
y02_n = int(number(metric_by_target.get("Y02", {}).get("n")) or 0)
y04 = metric_by_target.get("Y04", {})

development_formulae = {row["molecular_formula"] for row in development}
heldout_formulae = {row["molecular_formula"] for row in held_out}
records_seen = sum(row["molecular_formula"] in development_formulae for row in held_out)
formulae_seen = len(heldout_formulae & development_formulae)

distance_by_record: dict[str, float] = {}
for row in predictions:
    value = number(row.get("ood_distance_percentile"))
    if value is not None:
        distance_by_record.setdefault(first(row, "record_id", "test_case_id"), value)
median_distance = median(distance_by_record.values()) if distance_by_record else None

provenance_by_key = {
    (row["record_id"], row["property_code"]): row.get("value_origin_class", "")
    for row in provenance
}

metric_rows = "".join(
    "<tr>"
    f"<td><strong>{esc(row['target'])}</strong> · {esc(display(row['property']))}</td>"
    f"<td>{esc(row.get('n', '—'))} / {esc(row.get('formula_groups', '—'))}</td>"
    f"<td class=\"{'negative' if (number(row.get('r2')) or 0) < 0 else ''}\">{metric_with_bootstrap_ci(row, 'r2')}</td>"
    f"<td>{metric_with_bootstrap_ci(row, 'rmse')}</td><td>{metric_with_bootstrap_ci(row, 'mae')}</td>"
    f"<td>{coverage(row, '90')}</td><td>{coverage(row, '95')}</td>"
    f"<td>{fmt(row.get('interval90_width'))}</td><td>{fmt(row.get('interval95_width'))}</td>"
    "</tr>"
    for row in metrics
)

source_metric_rows = "".join(
    "<tr>"
    f"<td><strong>{esc(row['target'])}</strong> · {esc(display(row['property']))}</td>"
    f"<td>{esc(row.get('n', '0'))} / {esc(row.get('formula_groups', '0'))}</td>"
    f"<td>{metric_with_bootstrap_ci(row, 'r2')}</td>"
    f"<td>{metric_with_bootstrap_ci(row, 'rmse')}</td>"
    f"<td>{metric_with_bootstrap_ci(row, 'mae')}</td>"
    f"<td>{esc(row.get('reason') or 'estimable')}</td>"
    "</tr>"
    for row in source_supported_metrics
)

prediction_rows_parts: list[str] = []
for row in predictions:
    record_id = first(row, "record_id", "test_case_id")
    actual_text = first(row, "actual", "reference_value")
    explicitly_available = truth(row.get("label_available"))
    available = number(actual_text) is not None if explicitly_available is None else explicitly_available
    source = first(row, "value_origin_class", "provenance_class", default="") or provenance_by_key.get(
        (record_id, row["target"]), "not recorded"
    )
    prediction_rows_parts.append(
        "<tr>"
        f"<td>{esc(record_id)}</td><td>{esc(row['target'])}</td>"
        f"<td>{fmt(actual_text) if available else '—'}</td><td>{fmt(row.get('predicted'))}</td>"
        f"<td>{fmt(first(row, 'residual_actual_minus_predicted', 'residual_reference_minus_predicted')) if available else '—'}</td>"
        f"<td>{yes_no(row.get('covered90'), available)}</td>"
        f"<td>{yes_no(row.get('covered95'), available)}</td>"
        f"<td>{fmt(row.get('ood_distance_percentile'), 1)}</td>"
        f"<td>{esc(source)}</td>"
        "</tr>"
    )
prediction_rows = "".join(prediction_rows_parts)

private_ids = artifact_manifest.get("source_artifact_identifiers", {})
public_ids = artifact_manifest.get("public_deidentified_artifacts", {})
public_joblib_hash = display_hash(public_ids.get("joblib_sha256"))
public_prediction_hash = display_hash(public_ids.get("wide_holdout_predictions_sha256"))

y04_notice = (
    f"Y04 internal-holdout performance was R² = {fmt(y04.get('r2'))}; "
    f"MAE = {fmt(y04.get('mae'), 1)} {esc(display(y04.get('unit', '')))} "
    f"(n = {esc(y04.get('n', '—'))}). Experimental confirmation is required; "
    "Y04 should not be used alone to rank or select compounds."
)

document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="icon" href="data:,">
<title>SAF-Predict Internal Held-Out Test Report</title>
<style>
:root{{--ink:#17242d;--muted:#5e6b73;--line:#d8e0e4;--navy:#173b57;--blue:#2c6e9f;--canvas:#f4f7f8;--red:#a3413d;--amber:#b66a2c}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--canvas);color:var(--ink);font-family:Arial,sans-serif;line-height:1.55}}
header{{padding:42px max(28px,calc((100vw - 1120px)/2));color:#fff;background:linear-gradient(145deg,#173b57,#2b6874)}}
header h1{{margin:5px 0;font-size:38px}}header p{{max-width:850px;color:#dce9ee}}.tag{{display:inline-block;padding:5px 9px;border:1px solid rgba(255,255,255,.25);border-radius:999px;font-size:11px;font-weight:700}}
main{{max-width:1120px;margin:0 auto;padding:32px 28px 60px}}.card{{margin:18px 0;padding:22px;background:#fff;border:1px solid var(--line);border-radius:16px;box-shadow:0 14px 36px rgba(21,50,68,.07)}}
h2{{margin:0 0 12px}}p,li{{color:var(--muted)}}.metrics{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}}.metric{{padding:15px;background:#fff;border:1px solid var(--line);border-radius:12px}}.metric strong{{display:block;font-size:25px;color:var(--navy)}}.metric span{{font-size:11px;color:var(--muted);font-weight:700}}
.alert{{padding:13px 15px;border-radius:10px;color:#74400e;background:#fff3e8;border:1px solid #efc79f;font-weight:700}}.table-scroll{{overflow:auto;border:1px solid var(--line);border-radius:10px}}table{{width:100%;border-collapse:collapse;background:#fff;font-size:12px}}th{{color:#fff;background:var(--navy);text-align:left;white-space:nowrap}}th,td{{padding:9px 10px;border-bottom:1px solid #e5eaed}}tbody tr:hover{{background:#f5f9fb}}.negative{{color:var(--red);font-weight:800}}
code{{padding:2px 5px;border-radius:5px;background:#eef3f5}}a{{color:var(--blue);font-weight:700}}.hash{{word-break:break-all;font-family:Consolas,monospace;font-size:11px}}.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}}
@media(max-width:760px){{.metrics,.grid{{grid-template-columns:1fr 1fr}}header h1{{font-size:30px}}}}@media(max-width:520px){{.metrics,.grid{{grid-template-columns:1fr}}main{{padding:24px 14px}}}}
</style></head><body>
<header><span class="tag">SAF-Predict v{esc(version)}</span><h1>Internal Held-Out Test Report</h1><p>The released package contains 300 development records and 10 structurally stratified internal held-out records. This is not independent external, prospective, or experimental validation.</p></header>
<main>
<section class="metrics"><div class="metric"><strong>{len(held_out)}</strong><span>Internal test records</span></div><div class="metric"><strong>{len(heldout_formulae)}</strong><span>Unique molecular formulae</span></div><div class="metric"><strong>{records_seen}/{len(held_out)}</strong><span>Formula seen in development set</span></div><div class="metric"><strong>{fmt(median_distance, 1)}</strong><span>Median AD-distance percentile</span></div></section>
<div class="alert">{y04_notice}</div>
<section class="card"><h2>Performance by property</h2><div class="table-scroll"><table><thead><tr><th>Property</th><th>n / formula groups</th><th>R² [group-bootstrap 95% CI]</th><th>RMSE [group-bootstrap 95% CI]</th><th>MAE [group-bootstrap 95% CI]</th><th>90% coverage</th><th>95% coverage</th><th>90% width</th><th>95% width</th></tr></thead><tbody>{metric_rows}</tbody></table></div><p>Confidence intervals for R², RMSE, and MAE use 10,000 deterministic molecular-formula-group bootstrap draws. Coverage confidence intervals are exact binomial intervals. Y02 is scored on {y02_n} available reference labels; the other {len(held_out) - y02_n} Y02 labels remain missing and are not imputed. With these small target-specific sample sizes, all estimates remain imprecise.</p></section>
<section class="card"><h2>Source-supported label audit</h2><p>This table repeats the error calculation only for held-out labels flagged as high-confidence source-supported records in the property-level provenance table. A zero or non-estimable entry is reported explicitly and is not replaced by the full-set result.</p><div class="table-scroll"><table><thead><tr><th>Property</th><th>n / formula groups</th><th>R² [group-bootstrap 95% CI]</th><th>RMSE [group-bootstrap 95% CI]</th><th>MAE [group-bootstrap 95% CI]</th><th>Status</th></tr></thead><tbody>{source_metric_rows}</tbody></table></div></section>
<section class="card"><h2>Record- and property-level results</h2><p>The complete value-origin classes, source links, and condition or method notes are available in <a href="../data/provenance_long.csv">the record-property provenance table</a> and <a href="../data/references.csv">the reference registry</a>. A source link does not by itself establish a direct measurement under harmonized conditions.</p><div class="table-scroll"><table><thead><tr><th>Test case</th><th>Target</th><th>Reference value</th><th>Predicted</th><th>Residual</th><th>90% covered</th><th>95% covered</th><th>AD-distance percentile</th><th>Label category</th></tr></thead><tbody>{prediction_rows}</tbody></table></div></section>
<section class="card"><h2>Y02 definition in v1.1.0</h2><p>Y02 is produced by its own random-forest model, fitted to 162 independently collected development responses in 66 molecular-formula groups. It is not calculated from the deployed Y01 and Y03 predictions. The Y01 × Y03 quantity is retained only as a physical-baseline audit. Identity linkage is recorded, but exact primary-source and measurement-condition verification remains incomplete for some legacy Y02 rows.</p></section>
<section class="card"><h2>Traceability and release checks</h2><div class="grid"><p><strong>Public Joblib SHA256</strong><br><span class="hash">{esc(public_joblib_hash)}</span></p><p><strong>Public held-out prediction-file SHA256</strong><br><span class="hash">{esc(public_prediction_hash)}</span></p><p><strong>Original private Joblib SHA256</strong><br><span class="hash">{esc(display_hash(private_ids.get('private_joblib_sha256')))}</span></p><p><strong>Original wide prediction-file SHA256</strong><br><span class="hash">{esc(display_hash(private_ids.get('original_wide_holdout_predictions_sha256')))}</span></p></div><p>The hashes identify file contents. Analysis-design statements are author-reported metadata; no independently timestamped pre-evaluation model release is included in this repository.</p></section>
<section class="card"><h2>Limitations</h2><p>The held-out set contains {len(held_out)} records and {len(heldout_formulae)} molecular formulae; {records_seen} records and {formulae_seen} formulae overlap the development-set formula space. It therefore does not demonstrate external generalization or prediction in a new chemical domain. The 95th-percentile descriptor-distance rule is a high-distance warning calibrated on the development set, not an externally validated absolute out-of-domain boundary.</p><p>The application predicts pure-compound physicochemical properties only. It does not assess fuel-specification compliance, blend performance, sustainability, synthesis, safety, emissions, cost, or engine performance.</p><p><a href="../index.html">Return to SAF-Predict</a> · <a href="held_out_test_metrics.csv">Metrics CSV</a> · <a href="held_out_source_supported_metrics.csv">Source-supported metrics CSV</a> · <a href="held_out_test_predictions_long.csv">Record-level predictions CSV</a> · <a href="held_out_evaluation_checks.json">Release checks</a></p></section>
</main></body></html>
"""

report_path = EVALUATION / "held_out_test_report.html"
report_path.write_text(document, encoding="utf-8", newline="\n")

browser_data = (
    "(function(root,factory){var value=factory();"
    "if(typeof module==='object'&&module.exports){module.exports=value;}"
    "else{root.SAF_VALIDATION=value;}})"
    "(typeof self!=='undefined'?self:this,function(){return "
    + json.dumps(summary, ensure_ascii=False, separators=(",", ":"))
    + ";});\n"
)
browser_data_path = ROOT / "held_out_test_data.js"
browser_data_path.write_text(browser_data, encoding="utf-8", newline="\n")

print(json.dumps({"report": str(report_path), "browser_data": str(browser_data_path)}))
