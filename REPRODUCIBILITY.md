# Reproducibility guide

This guide separates direct use of the frozen v1.1.0 predictor from full model
development. Direct browser prediction is installation-free; scientific
reproduction uses Python 3.12 and the pinned environment.

## Level 1: direct prediction

Extract the release archive and open `index.html`. All prediction logic and
model parameters are stored locally in JavaScript. No network connection is
required.

## Level 2: frozen-model parity

Create an isolated environment and install the pinned dependencies:

```bash
python -m venv .venv
python -m pip install -r requirements-lock.txt
```

Then run:

```bash
python training/saf_reproduce.py validate --input data/development.csv
python scripts/compute_descriptors.py --input data/development.csv \
  --output run_outputs/development_descriptor_audit.csv \
  --report run_outputs/development_descriptor_audit.json \
  --verify-existing --tolerance 1e-9
python scripts/evaluate_heldout.py --data data/held_out_test.csv \
  --model models/SAF_Predict_public_model_bundle_v1.joblib \
  --reference-dir validation --output-dir run_outputs/heldout
node tests/test_js_against_python_reference_cases.js
```

The held-out evaluator compares regenerated point predictions and R²/RMSE/MAE
with the archived v1.1.0 outputs. The JavaScript test checks all seven exported
models against stored Python reference predictions and verifies that Y02 is a
separate prediction rather than the product of Y01 and Y03.

## Final release-integrity verification

Once every release file is final, use this order:

```bash
python scripts/run_release_qa.py
python scripts/build_release_manifest.py
python scripts/run_release_qa.py --verify-release-manifest
```

The first command writes the QA record. The manifest generator then
content-addresses that record and every other public release file, while the
checksum list additionally contains the manifest's digest but never its own.
The final command is read-only. It validates this finite integrity chain and
fails on omitted files, stale digests, malformed or duplicate checksum entries,
or an incorrect manifest file count.

## Level 3: full model development

Run the complete four-algorithm comparison and deployment export:

```bash
python training/saf_reproduce.py all \
  --input data/development.csv \
  --config training/config.json \
  --output-dir run_outputs/full \
  --version 1.1.0 \
  --n-jobs 1
```

This performs nested molecular-formula-grouped cross-validation for XGBoost,
random forest, support vector regression, and artificial neural network models.
Y01 and Y03–Y07 use all 300 development records. Y02 is fitted separately on
the 162 identity-resolved, independently collected Y02 labels spanning 66 molecular-formula
groups; its 138 missing development labels are neither imputed nor reconstructed.
The workflow then fits the recorded deployment families, calibrates
residual-based intervals, constructs the applicability-domain reference, and
exports Python and browser artifacts.

The exporter rejects smoke-grid results and cross-validation artifacts whose
recorded input-data hash differs from the supplied development CSV.

For maintainer release assembly, first create a private metadata-bearing export
from the archived full-CV results, evaluate that frozen export, and then build
the metadata-reduced public copy:

```bash
python scripts/build_model_consistency_table.py \
  --oof artifacts/cv/oof_predictions.csv \
  --config training/config.json \
  --output artifacts/cv/xgboost_reference_vs_deployment_consistency.csv
python training/saf_reproduce.py train-export \
  --input data/development.csv \
  --config training/config.json \
  --cv-dir artifacts/cv \
  --output-dir run_outputs/release_v1_1_private \
  --version 1.1.0 --n-jobs 1 --no-deidentify-ood
python scripts/evaluate_heldout.py \
  --data data/held_out_test.csv \
  --provenance data/provenance_long.csv \
  --model run_outputs/release_v1_1_private/models/SAF_Predict_model_bundle_v1.joblib \
  --output-dir validation
python -c "from shutil import copy2; copy2('run_outputs/release_v1_1_private/models/model_bundle.json', 'models/model_bundle.json')"
python -c "from shutil import copy2; copy2('run_outputs/release_v1_1_private/models/model_training_summary.csv', 'models/model_training_summary.csv')"
python -c "from shutil import copy2; copy2('run_outputs/release_v1_1_private/models/model_card.json', 'models/model_card.json')"
python -c "from shutil import copy2; copy2('run_outputs/release_v1_1_private/model_bundle.js', 'model_bundle.js')"
python -c "from shutil import copy2; copy2('run_outputs/release_v1_1_private/tests/python_reference_predictions_6cases.json', 'tests/python_reference_predictions_6cases.json')"
python scripts/sanitize_public_bundle.py \
  --source-joblib run_outputs/release_v1_1_private/models/SAF_Predict_model_bundle_v1.joblib
python scripts/build_model_artifact_manifest.py \
  --root . \
  --private-joblib run_outputs/release_v1_1_private/models/SAF_Predict_model_bundle_v1.joblib
python scripts/build_training_export_reproduction_check.py \
  --root . \
  --private-model run_outputs/release_v1_1_private/models/SAF_Predict_model_bundle_v1.joblib
python scripts/build_validation_metadata.py --root .
python scripts/build_validation_report.py
```

`sanitize_public_bundle.py` also accepts
`--expected-source-joblib-sha256`, `--expected-wide-csv-sha256`, and
`--expected-long-csv-sha256`. Maintainers may supply the freshly calculated
private-input hashes to pin the exact inputs before metadata reduction. These
arguments are optional so a clean reproduction is not tied to one serialized
file hash.

The v1.0.0-to-v1.1.0 regression audit for the six unchanged targets requires a
separately obtained v1.0.0 Joblib artifact; that historical artifact is not
duplicated in the v1.1.0 archive. When it is available, run:

```bash
python scripts/compare_model_versions.py \
  --old-model path/to/v1.0.0/SAF_Predict_public_model_bundle_v1.joblib \
  --new-model models/SAF_Predict_public_model_bundle_v1.joblib \
  --output validation/v1_0_to_v1_1_six_target_regression_check.json
```

The generated JSON reports pass/fail results and maximum prediction differences;
the outcome must be inspected rather than presumed.

## Fixed analysis design

- Development records: 300; molecular-formula groups: 73.
- Y02 development labels: 162; molecular-formula groups: 66.
- Outer validation: five-fold `GroupKFold` by molecular formula.
- Inner tuning: three-fold `GroupKFold` by molecular formula.
- Y02 uses the same fixed outer group assignment after filtering to records with
  an available Y02 label.
- Selection metric: negative root mean squared error.
- Base seed: 20260831.
- Nested-CV estimator seed: base seed + 100 × target index + 10 × model index
  + outer-fold number.
- Final-fit seed: base seed + 9000 + deployment index, with the index zero-based
  in the fixed order Y01, Y03, Y04, Y05, Y06, Y07, Y02. The corresponding seeds
  are 20269831–20269837; appending Y02 preserves the v1.0.0 seeds for the other
  six targets.
- Y02 deployment: a separately fitted random-forest regressor.
- Internal held-out records: 10; excluded from tuning, fitting, interval
  calibration, and applicability-domain calibration. Eight have an available
  Y02 reference label and two remain missing.

The complete grids and fixed estimator options are machine-readable in
`training/config.json`. Archived fold-level and out-of-fold results are in
`artifacts/cv/`.

Y02 was independently collected relative to the current Y01 and Y03 release
columns. Identity reconciliation is documented in
`data/y02_direct_response_reconciliation.csv` and
`data/y02_full_reconciliation_audit.csv`. Row-level direct experimental status,
primary-source identity, and measurement conditions remain unresolved for some
legacy entries, so these labels are not automatically classified as
high-confidence measurements. `data/y02_physical_baseline_audit.csv` retains
Y01 × Y03 only as a physical-baseline audit; it is not a training target and
is not substituted for missing Y02 values.

The public development table retains `reference_map` rather than redundant
per-target analysis-stratum columns. For Y01 and Y03–Y07, the training workflow
reconstructs the archived `formula-derived`/`source-supported` audit strata from
that map and the reference IDs recorded in `training/config.json`. Y02 uses the
separate independent-response/not-available status described above.

Under the lockfile versions and `--n-jobs 1`, a full rerun should reproduce the
fixed fold assignment, seed-dependent model selection, predictions, and
numerical CV metrics represented in `artifacts/cv/`. It is not a byte-for-byte
artifact rebuild: timestamps, timing measurements, input/config hashes, and
some generated metadata are expected to differ.

Compare the full generated nested-CV directory against the archive with the
portable, row-order-independent checker:

```bash
python scripts/build_model_consistency_table.py \
  --oof run_outputs/full/cv/oof_predictions.csv \
  --config training/config.json \
  --output run_outputs/full/cv/xgboost_reference_vs_deployment_consistency.csv
python scripts/compare_cv_artifacts.py \
  --generated-dir run_outputs/full/cv \
  --archive-dir artifacts/cv \
  --output run_outputs/full_nested_cv_reproduction_check.json
```

It requires exact categorical fields and uses an absolute numerical tolerance
of `1e-12` by default. Elapsed fit times, timestamps, hashes, paths, and other
run-instance metadata are intentionally excluded from comparison. The shipped
result is `validation/full_nested_cv_reproduction_check.json`.

`validation/training_export_reproduction_check.json` records a fresh
installation and final fit/export. It verifies the regenerated Joblib artifact
on the held-out records and the regenerated browser bundle on the stored Python
reference cases. Prediction equivalence, rather than serialized file identity,
is the criterion because run-specific metadata are embedded.

## Interpretation boundary

The 10-record set is a structurally stratified internal held-out test from the
same curation workflow. Nine records, representing eight of nine unique
formulae, share a molecular formula with the development set. It is not
independent external, prospective, or experimental validation. The three
author-measured compounds discussed separately in the manuscript are application
cases, not members of this internal held-out set.

Property-specific metrics and interval coverage are generated directly from
the current frozen model and are reported in
`validation/held_out_test_metrics.csv` and
`validation/held_out_test_report.html`. The small test size, the two missing Y02
labels, and the limited chemical-space separation must be considered when
interpreting those estimates. In particular, solid–liquid phase-transition
predictions require experimental confirmation and should not be used alone to
rank or select compounds.
