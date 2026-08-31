# SAF-Predict

**SAF-Predict** is an offline, browser-based web application for research-stage prediction of seven physicochemical properties of neutral, single-component, SAF-relevant hydrocarbons from seven molecular descriptors.

The application runs locally in a browser with JavaScript enabled. It does not require Python, a web server, package installation, or an internet connection, and molecular inputs are not uploaded. This release does not include a browser/operating-system compatibility matrix.

## Run the application

1. Download or clone this repository.
2. Open [`index.html`](index.html) in a current desktop browser.
3. Enter X01–X07 or use [`examples/batch_input_template.csv`](examples/batch_input_template.csv) for batch prediction.

`Start_SAF_Predictor.bat` is an optional Windows shortcut; `index.html` is the application entry point.

## Inputs

| Code | Descriptor | Unit |
|---|---|---|
| X01 | Molar mass | g mol−1 |
| X02 | H/C atomic ratio | 1 |
| X03 | Aromatic carbon fraction | 0–1 |
| X04 | Ring count | count |
| X05 | Branching-carbon count | count |
| X06 | Graph-automorphism descriptor | 1 |
| X07 | Kier κ2 shape index | 1 |

Exact definitions are provided in the [user guide](README.html) and in the application interface. Inputs must be calculated using the same definitions used for model development.

## Outputs and deployed models

| Code | Property | Deployed method |
|---|---|---|
| Y01 | Mass-based net heat of combustion | Random Forest |
| Y02 | Volumetric net heat of combustion | Predicted Y01 × predicted Y03 |
| Y03 | Density | Support vector regression |
| Y04 | Solid–liquid phase-transition temperature | XGBoost |
| Y05 | Boiling point | Support vector regression |
| Y06 | Flash point | Support vector regression |
| Y07 | Kinematic viscosity at 40 °C | Support vector regression |

The interface reports point estimates, 90% and 95% empirical prediction intervals calibrated from molecular-formula-grouped out-of-fold residuals, development-label provenance, input-range diagnostics, and the applicability-domain (AD) distance percentile.

## Internal holdout evaluation

According to the recorded workflow, model selection and hyperparameter selection used the development set (Sheet1; n = 300), followed by evaluation on the internal holdout set (Sheet2; n = 10). Current hashes identify the released artifacts but do not independently establish pre-evaluation chronology.

| Target | R² | RMSE | MAE |
|---|---:|---:|---:|
| Y01 | 0.902 | 0.577 | 0.463 |
| Y02 | 0.773 | 1.219 | 0.967 |
| Y03 | 0.873 | 0.0325 | 0.0238 |
| Y04 | −0.021 | 36.371 °C | 25.231 °C |
| Y05 | 0.979 | 8.236 °C | 6.437 °C |
| Y06 | 0.920 | 10.697 °C | 7.657 °C |
| Y07 | 0.820 | 0.295 mm² s−1 | 0.196 mm² s−1 |

Nine of the ten internal holdout records, representing eight of nine unique molecular formulae, share a formula with the development set. This is an **internal holdout evaluation**, not independent external, prospective, or experimental validation. Full record-level results are in [`validation/`](validation/).

## Limitations

- Y04 performed poorly on the internal holdout (R² = −0.021; MAE = 25.2 °C; n = 10). Do not use Y04 alone to rank or select compounds; experimental confirmation is required.
- Source-supported, formula-derived, and constructed are author-assigned curation categories for the development labels. Source-supported means that a traceable publication or database entry exists; it does not necessarily denote a direct, condition-matched experiment. The record-level provenance table is not included in this software-only repository.
- Y04 is the pure-compound solid–liquid phase-transition temperature, not the freezing point of a finished fuel. Y05 is not a fuel distillation curve, and Y07 is not a low-temperature aviation-fuel viscosity specification.
- The applicability-domain (AD) distance percentile reports position in the development-set reference-distance distribution; it is not a calibrated prediction-error probability.
- The application predicts pure-compound physicochemical properties only. It does not assess fuel-specification compliance, blend performance, sustainability, synthesis, safety, emissions, cost, or engine performance.

## Traceability and release artifacts

- [`models/MODEL_CARD.md`](models/MODEL_CARD.md)
- [`models/model_artifact_manifest.json`](models/model_artifact_manifest.json)
- [Public metadata-reduced Joblib model](models/SAF_Predict_public_model_bundle_v1.joblib)
- [Public browser-model JSON](models/model_bundle.json)
- [Public-artifact sanitization checks](models/public_artifact_sanitization_checks.json)
- [Development summary](models/model_training_summary.csv)
- [`held_out_test_data.js`](held_out_test_data.js)
- [`validation/held_out_test_report.html`](validation/held_out_test_report.html)
- [`validation/held_out_predictions.csv`](validation/held_out_predictions.csv)
- [`validation/held_out_test_metrics.csv`](validation/held_out_test_metrics.csv)
- [`validation/held_out_test_predictions_long.csv`](validation/held_out_test_predictions_long.csv)
- [`RELEASE_CHECKS.json`](RELEASE_CHECKS.json)
- [`tests/test_js_against_python_reference_cases.js`](tests/test_js_against_python_reference_cases.js)
- [`release_manifest.json`](release_manifest.json)
- [`checksums.sha256`](checksums.sha256)

## Repository scope

This repository contains the English browser application, fixed serialized model artifacts, internal holdout evaluation outputs, and release checks. It does not contain the complete development dataset, record-level provenance table, descriptor-generation workflow, or end-to-end model-development scripts and is therefore not a complete manuscript reproducibility archive.

The public model artifacts omit molecule names, molecular formulae, original development-record identifiers, and bilingual display metadata. X01–X07 development vectors remain because the application uses them for nearest-neighbour applicability-domain calculations; these vectors may be linkable to a known source dataset.

## License and citation

No open-source license is included in v1.0.0. Unless a license is added, copyright law reserves all rights. This repository does not grant redistribution rights for source records compiled from third-party publications or databases.

## Version

SAF-Predict v1.0.0 uses fixed serialized model artifacts. Interface localization does not alter their stored coefficients, support vectors, trees, or hyperparameters.
