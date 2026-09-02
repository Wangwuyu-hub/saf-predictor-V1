# Reproducibility guide

This guide separates direct use of the frozen v1.0.0 predictor from full model
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

The held-out evaluator fails if regenerated point predictions or R²/RMSE/MAE
differ from the archived v1.0.0 values by more than 1e-9.

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
The final command is read-only, so it can validate this finite integrity chain
without invalidating it. It fails on omitted files, stale digests, malformed or
duplicate checksum entries, or an incorrect manifest file count.

## Level 3: full model development

Run the complete four-algorithm comparison and deployment export:

```bash
python training/saf_reproduce.py all \
  --input data/development.csv \
  --config training/config.json \
  --output-dir run_outputs/full \
  --version 1.0.0 \
  --n-jobs 1
```

This performs nested formula-grouped cross-validation for XGBoost, random
forest, support vector regression, and artificial neural network models on all
seven targets. It then fits the recorded deployment families, derives Y02 from
the Y01 and Y03 predictions, calibrates residual-based intervals, constructs
the applicability-domain reference, and exports Python and browser artifacts.

The exporter rejects smoke-grid results and rejects cross-validation artifacts
whose recorded input-data hash differs from the supplied development CSV.

## Fixed analysis design

- Development records: 300; molecular-formula groups: 73.
- Outer validation: five-fold GroupKFold by molecular formula.
- Inner tuning: three-fold GroupKFold by molecular formula.
- Selection metric: negative root mean squared error.
- Base seed: 20260831.
- Nested-CV estimator seed: base seed + 100 x target index + 10 x model index + outer-fold number.
- Final-fit seed: base seed + 9000 + deployment index, with the index zero-based
  in the fixed order Y01, Y03, Y04, Y05, Y06, Y07 (20269831–20269836).
- Y02 deployment: predicted Y01 x predicted Y03; no separately deployed Y02 regressor.
- For the Figure 4 direct-model benchmark only, the loader reconstructs the
  Y02 target as Y01 x Y03 after CSV parsing. This preserves the exact workbook
  formula values across decimal serialization.
- Internal held-out records: 10; excluded from tuning, fitting, interval calibration, and applicability-domain calibration.

The complete grids and fixed estimator options are machine-readable in
`training/config.json`. Archived fold-level and out-of-fold results are in
`artifacts/cv/`.

The public development table retains `reference_map` rather than redundant
per-target Figure 4 stratum columns. The training workflow reconstructs the
archived `formula-derived`/`source-supported` stratification from that map and
the reference IDs recorded in `training/config.json`; Y02 is `constructed`.
This two-class analysis stratum is distinct from the detailed value-origin
metadata in `data/provenance_long.csv`.

Under the lockfile versions and `--n-jobs 1`, a full rerun should reproduce the
fixed fold assignment, seed-dependent model selection, predictions, and
numerical CV metrics represented in `artifacts/cv/`. It is not a byte-for-byte
artifact rebuild: timestamps, timing measurements, input/config hashes, and
some generated metadata are expected to differ.

Compare the full generated nested-CV directory against the archive with the
portable, row-order-independent checker:

```bash
python scripts/compare_cv_artifacts.py \
  --generated-dir run_outputs/full/cv \
  --archive-dir artifacts/cv \
  --output run_outputs/full_nested_cv_reproduction_check.json
```

It requires exact categorical fields and uses an absolute numerical tolerance
of `1e-12` by default. Elapsed fit times, timestamps, hashes, paths, and other
run-instance metadata are intentionally excluded from comparison. The shipped
result is `validation/full_nested_cv_reproduction_check.json`.

`validation/training_export_reproduction_check.json` additionally records a
fresh installation and final fit/export. It verifies the regenerated Joblib
artifact on the held-out records and the regenerated browser bundle on the six
stored Python reference cases. Prediction equivalence, rather than serialized
file identity, is the criterion because run-specific metadata are embedded.

## Interpretation boundary

The 10-record set is an internal held-out test from the same curation workflow.
Nine records, representing eight of nine unique formulae, share a molecular
formula with the development set. It is not independent external, prospective,
or experimental validation.

Y04 performed poorly on this test (R² = -0.021; MAE = 25.2 °C; n = 10). Do not
use Y04 alone to rank or select compounds; experimental confirmation is
required.
