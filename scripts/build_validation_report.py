#!/usr/bin/env python3
"""Regenerate the English internal-test report and browser data artifact."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVALUATION = ROOT / "validation"


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def fmt(value: str | float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def first(row: dict[str, str], *names: str) -> str:
    for name in names:
        value = row.get(name)
        if value not in {None, ""}:
            return value
    raise KeyError(f"None of the expected columns are present: {', '.join(names)}")


def display_hash(value: object) -> str:
    return str(value) if value else "pending final release assembly"


with (EVALUATION / "held_out_test_metrics.csv").open(encoding="utf-8-sig", newline="") as handle:
    metrics = list(csv.DictReader(handle))
with (EVALUATION / "held_out_test_predictions_long.csv").open(encoding="utf-8-sig", newline="") as handle:
    predictions = list(csv.DictReader(handle))

artifact_manifest = json.loads((ROOT / "models" / "model_artifact_manifest.json").read_text(encoding="utf-8"))
evaluation_checks = json.loads((EVALUATION / "held_out_evaluation_checks.json").read_text(encoding="utf-8"))
summary = json.loads((EVALUATION / "held_out_test_summary.json").read_text(encoding="utf-8"))

metric_rows = "".join(
    "<tr>"
    f"<td><strong>{esc(row['target'])}</strong> · {esc(row['property'])}</td>"
    f"<td class=\"{'negative' if float(row['r2']) < 0 else ''}\">{fmt(row['r2'])}</td>"
    f"<td>{fmt(row['rmse'])}</td><td>{fmt(row['mae'])}</td>"
    f"<td>{fmt(row['median_absolute_error'])}</td>"
    f"<td>{fmt(row['mean_signed_error_actual_minus_predicted'])}</td>"
    f"<td>{esc(row['coverage90_count'])}/{esc(row['coverage90_n'])}</td>"
    f"<td>{esc(row['coverage95_count'])}/{esc(row['coverage95_n'])}</td>"
    "</tr>"
    for row in metrics
)

prediction_rows = "".join(
    "<tr>"
    f"<td>{esc(first(row, 'test_case_id', 'record_id'))}</td><td>{esc(row['target'])}</td>"
    f"<td>{fmt(first(row, 'reference_value', 'actual'))}</td><td>{fmt(row['predicted'])}</td>"
    f"<td>{fmt(first(row, 'residual_reference_minus_predicted', 'residual_actual_minus_predicted'))}</td>"
    f"<td>{'yes' if row['covered90'].lower() == 'true' else 'no'}</td>"
    f"<td>{'yes' if row['covered95'].lower() == 'true' else 'no'}</td>"
    f"<td>{float(row['ood_distance_percentile']):.1f}</td>"
    f"<td>{esc(row['provenance_class'])}</td>"
    "</tr>"
    for row in predictions
)

private_ids = artifact_manifest["source_artifact_identifiers"]
public_ids = artifact_manifest["public_deidentified_artifacts"]
public_joblib_hash = display_hash(public_ids.get("joblib_sha256"))
public_prediction_hash = display_hash(public_ids.get("wide_holdout_predictions_sha256"))

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
<header><span class="tag">SAF-Predict v1.0.0</span><h1>Internal Held-Out Test Report</h1><p>The released package records Sheet1 (n = 300) as the development set and Sheet2 (n = 10) as an internal held-out test set. This is not independent external, prospective, or experimental validation.</p></header>
<main>
<section class="metrics"><div class="metric"><strong>{summary['n_records']}</strong><span>Internal test records</span></div><div class="metric"><strong>{summary['unique_formulae']}</strong><span>Unique molecular formulae</span></div><div class="metric"><strong>{summary['records_with_formula_seen_in_development']}/{summary['n_records']}</strong><span>Formula seen in development set</span></div><div class="metric"><strong>{summary['ood']['median_distance_percentile']:.1f}</strong><span>Median applicability-domain (AD) distance percentile</span></div></section>
<div class="alert">Y04 performed poorly on the internal holdout (R² = −0.021; MAE = 25.2 °C; n = 10). Do not use Y04 alone to rank or select compounds; experimental confirmation is required.</div>
<section class="card"><h2>Performance by property</h2><div class="table-scroll"><table><thead><tr><th>Property</th><th>R²</th><th>RMSE</th><th>MAE</th><th>MedAE</th><th>Mean signed error<br>(reference − predicted)</th><th>90% coverage</th><th>95% coverage</th></tr></thead><tbody>{metric_rows}</tbody></table></div><p>With n = 10, R² and empirical coverage estimates are highly unstable: one record changes a coverage estimate by ten percentage points. Record-level residuals are therefore retained below.</p></section>
<section class="card"><h2>Record- and property-level results</h2><p><code>source-supported</code>, <code>formula-derived</code>, and <code>constructed</code> are author-assigned curation categories. The record-level source table is not included in this software-only repository, and source-supported does not by itself establish a direct measurement under harmonized conditions.</p><div class="table-scroll"><table><thead><tr><th>Test case</th><th>Target</th><th>Reference value</th><th>Predicted</th><th>Residual</th><th>90% covered</th><th>95% covered</th><th>Applicability-domain (AD) distance percentile</th><th>Label category</th></tr></thead><tbody>{prediction_rows}</tbody></table></div></section>
<section class="card"><h2>Traceability and release checks</h2><div class="grid"><p><strong>Public Joblib SHA256</strong><br><span class="hash">{esc(public_joblib_hash)}</span></p><p><strong>Public held-out prediction-file SHA256</strong><br><span class="hash">{esc(public_prediction_hash)}</span></p><p><strong>Original private Joblib SHA256</strong><br><span class="hash">{esc(private_ids['private_joblib_sha256'])}</span></p><p><strong>Original wide prediction-file SHA256</strong><br><span class="hash">{esc(private_ids['original_wide_holdout_predictions_sha256'])}</span></p></div><p>The hashes identify file contents. The analysis-design statements are author-reported metadata; no independently timestamped pre-evaluation model release is included in this repository.</p></section>
<section class="card"><h2>Limitations</h2><p>This report summarizes released predictions and reference-value comparisons within the same database-building workflow; it is not independent external, prospective, or experimental validation.</p><p>The application predicts pure-compound physicochemical properties only. It does not assess fuel-specification compliance, blend performance, sustainability, synthesis, safety, emissions, cost, or engine performance.</p><p><a href="../index.html">Return to SAF-Predict</a> · <a href="held_out_test_metrics.csv">Metrics CSV</a> · <a href="held_out_test_predictions_long.csv">Record-level predictions CSV</a> · <a href="held_out_evaluation_checks.json">Release checks</a></p></section>
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
