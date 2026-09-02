# SAF-Predict v1.0.0 training workflow

`saf_reproduce.py` is the portable command-line implementation of the complete
model-development workflow. It reads the public development CSV, compares four
regression families with nested molecular-formula-grouped cross-validation,
fits the recorded deployment models, and exports Python and browser artifacts.
No machine-specific path is embedded in the code.

## Scientific stages

1. XGBoost, random forest (RF), support vector regression (SVR), and an
   artificial neural network (ANN) are compared for Y01-Y07 with outer
   five-fold and inner three-fold `GroupKFold`. Molecular formula is the group.
2. The recorded deployment family for each property is tuned on the complete
   300-record development table and exported as Joblib, JSON, and browser-ready
   JavaScript.

Direct Y02 regressors are retained only as model-comparison benchmarks. The
deployed prediction is always:

```text
predicted Y02 = predicted Y01 * predicted Y03
```

When the development CSV is loaded, the training script also reconstructs the
direct-benchmark Y02 target from Y01 and Y03. This preserves the exact
spreadsheet identity across decimal CSV round-tripping and matches the target
values used for the archived Figure 4 calculation.

## Files

- `saf_reproduce.py`: validation, smoke testing, nested CV, final fitting, and
  artifact export.
- `config.json`: folds, model order, fixed settings, complete grids, deployment
  mapping, base seed, and seed formulas.
- `input_schema.csv`: minimum canonical training-table schema.

The repository root contains `requirements.txt`, `requirements-lock.txt`, and
`environment.yml`.

After `train-export`, the regenerated browser bundle can be checked against the
stored Python reference cases by setting `SAF_MODEL_BUNDLE_PATH` before running
`node tests/test_js_against_python_reference_cases.js`.

## Canonical input

The required columns are:

```text
record_id,molecular_formula,X01,X02,X03,X04,X05,X06,X07,Y01,Y02,Y03,Y04,Y05,Y06,Y07
```

`molecule` or the public-table spelling `molecule_name` is optional and is
used only as a display label in manifests and prediction tables. The default
validation requires exactly 300 development records, at least five
molecular-formula groups, and finite, non-constant X/Y columns.
`data/development.csv` contains 300 records in 73 formula groups.

The command accepts development data only. `data/held_out_test.csv` has no role
in tuning, fitting, residual-interval calibration, or applicability-domain
calibration.

## Provenance strata and archived outputs

For Figure 4 comparability, the nested-CV output separates each direct target
into `formula-derived` and `source-supported` strata. Public
`development.csv` retains the per-target `reference_map`; the workflow
reconstructs these two analysis strata from that map and the explicit reference
IDs in `config.json`. Y02 is always `constructed`. These are the original
Figure 4 analysis strata, not replacements for the more detailed value-origin
classes in `data/provenance_long.csv`.

With the released `data/development.csv`, pinned package versions, and
`--n-jobs 1`, the regenerated fold assignment, stochastic seeds, selected
parameters, predictions, and numerical CV metrics are expected to agree with
the archived records in `artifacts/cv/` up to ordinary floating-point/runtime
serialization differences. Do not use file hashes or elapsed `fit_seconds` as
the comparison criterion: generated timestamps, input/config hashes, metadata
wording, and timing fields make a byte-for-byte match inappropriate.

After `nested-cv`, compare the complete output directory against the archived
records without relying on row order or file hashes:

```bash
python scripts/compare_cv_artifacts.py \
  --generated-dir run_outputs/cv \
  --archive-dir artifacts/cv \
  --output run_outputs/full_nested_cv_reproduction_check.json
```

The comparator enforces exact categorical fields and a default absolute
numerical tolerance of `1e-12`; only run-instance timing, timestamps, hashes,
paths, and related metadata are excluded.

## Commands from the repository root

Validate without fitting:

```bash
python training/saf_reproduce.py validate --input data/development.csv
```

Run a short Y01/fold-1 check across all four algorithms:

```bash
python training/saf_reproduce.py smoke \
  --input data/development.csv \
  --config training/config.json \
  --output-dir run_outputs/smoke
```

Smoke output is never a manuscript result and cannot be passed to the final
exporter.

Run the complete nested CV:

```bash
python training/saf_reproduce.py nested-cv \
  --input data/development.csv \
  --config training/config.json \
  --output-dir run_outputs/cv \
  --n-jobs 1
```

Train and export deployment models after the complete CV run:

```bash
python training/saf_reproduce.py train-export \
  --input data/development.csv \
  --config training/config.json \
  --cv-dir run_outputs/cv \
  --output-dir run_outputs/release \
  --version 1.0.0 \
  --n-jobs 1
```

Run both stages:

```bash
python training/saf_reproduce.py all \
  --input data/development.csv \
  --config training/config.json \
  --output-dir run_outputs/full \
  --version 1.0.0 \
  --n-jobs 1
```

## Seed policy

- Base seed: `20260831`.
- Nested-CV seed:
  `base_seed + 100 * target_index + 10 * model_index + outer_fold_number`.
- Final-fit seed: `base_seed + 9000 + deployment_index`, where
  `deployment_index` is zero-based in the fixed order Y01, Y03, Y04, Y05, Y06,
  Y07 (thus 20269831 through 20269836).

`GroupKFold` is deterministic and does not shuffle. Stochastic estimators use
one thread. Grid-search parallelism is controlled separately by `--n-jobs`;
`--n-jobs 1` is the conservative deterministic setting.

## Deployment mapping

| Target | Deployment |
|---|---|
| Y01 | RF |
| Y02 | predicted Y01 multiplied by predicted Y03 |
| Y03 | SVR |
| Y04 | XGBoost |
| Y05 | SVR |
| Y06 | SVR |
| Y07 | SVR |

ANN remains in the complete comparison but is not a v1.0.0 browser-deployment
model.

## Generated nested-CV files

- `fold_manifest.csv`
- `outer_fold_scores.csv`
- `oof_predictions.csv`
- `model_target_summary.csv`
- `provenance_stratified_metrics.csv`
- `y02_direct_vs_derived_sensitivity.csv`
- `protocol.json`
- `qa.json`

A complete run produces 8,400 out-of-fold rows: 300 records x 4 algorithms x
7 targets. QA verifies one prediction per record/model/target and zero
molecular-formula overlap between each outer training and validation subset.

## Generated deployment files

```text
release/
  artifact_manifest.json
  model_bundle.js
  training_config.json
  models/
    SAF_Predict_public_model_bundle_v1.joblib
    final_cv_search_results.csv
    model_bundle.json
    model_card.json
    model_training_summary.csv
  tests/
    python_reference_predictions_6cases.json
```

The public applicability-domain records retain deterministic `DEV-0001` style
IDs and X01-X07 vectors. The exporter rejects CV files generated from a
different input-data hash and rejects smoke-grid results.
