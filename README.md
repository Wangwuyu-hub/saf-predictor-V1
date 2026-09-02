# SAF-Predict v1.0.0

SAF-Predict is a versioned, offline research tool for estimating seven
physicochemical properties of neutral, single-component hydrocarbon molecules
from seven structure descriptors. The repository contains the browser
application, source code, fixed model artifacts, development and internal-test
data, record-property provenance, the complete model-training workflow, fixed
cross-validation folds, and archived evaluation outputs.

The tool supports early-stage molecular prescreening. It does not establish
aviation-fuel qualification, finished-blend performance, sustainability,
synthetic feasibility, safety, emissions, cost, or engine compatibility.

## Use the predictor without installing software

1. Download the `SAF-Predict-v1.0.0-offline.zip` asset from the GitHub release,
   or download this repository as a ZIP.
2. Extract the archive.
3. Open `index.html` in a current desktop browser. On Windows,
   `Start_SAF_Predictor.bat` provides the same entry point.
4. Enter X01-X07, or import `examples/batch_input_template.csv`.

The predictor runs locally with JavaScript enabled. It does not require Python,
a web server, package installation, or an internet connection, and it does not
upload molecular inputs.

## Inputs and outputs

| Code | Input descriptor | Unit |
|---|---|---|
| X01 | Molar mass | g mol^-1 |
| X02 | H/C atomic ratio | 1 |
| X03 | Aromatic carbon fraction | 0-1 |
| X04 | Ring count | count |
| X05 | Branching-carbon count | count |
| X06 | Graph-automorphism descriptor, ln(1+n) | 1 |
| X07 | Kier kappa2 shape index | 1 |

| Code | Predicted property | Deployed method |
|---|---|---|
| Y01 | Mass-based net heat of combustion | Random forest |
| Y02 | Volumetric net heat of combustion | Predicted Y01 x predicted Y03 |
| Y03 | Density | Support vector regression |
| Y04 | Solid-liquid phase-transition temperature | XGBoost |
| Y05 | Boiling point | Support vector regression |
| Y06 | Flash point | Support vector regression |
| Y07 | Kinematic viscosity at 40 °C | Support vector regression |

The interface also reports empirical 90% and 95% prediction intervals,
input-range checks, and an applicability-domain distance percentile. The
distance percentile locates an input within the development-set reference
distribution; it is not a calibrated probability of prediction error.

## Data release

The release is a **provenance-aware hybrid dataset**, not an all-experimental
database.

- `data/development.csv`: 300 records in 73 molecular-formula groups; used for
  grouped cross-validation, tuning, final fitting, interval calibration, and
  applicability-domain calibration.
- `data/held_out_test.csv`: 10 structurally stratified internal held-out records
  in 9 formula groups; kept outside model selection and fitting.
- `data/provenance_long.csv`: one row for every record-property pair (2,170
  rows), with value origin, source link, retained conditions or method notes,
  and a conservative quality field.
- `data/references.csv`: 491 source records with DOI or URL where available.
- `data/data_dictionary.csv`: field definitions, units, allowed values, and
  missing-value policies.
- `data/cv_fold_manifest.csv`: the fixed outer GroupKFold assignment for all
  300 development records.
- `data/SAF_Hydrocarbon_Dataset_v1.0.0.xlsx`: formatted workbook containing the
  same release tables.

Every record is a pure, single-component C/H-only molecule. Mixture and
oxygenate flags are explicit and false throughout this release. Aromaticity and
structure class are separate fields; the density >=0.90 g cm^-3 flag is only a
screening label, not a chemical identity class. Missing CAS RNs or PubChem CIDs
were left missing when no unambiguous identifier was retained; InChIKey and
canonical SMILES are complete.

## Reproduce the checks

Python 3.12 is recommended. `requirements-lock.txt` pins the direct and
transitive packages used for the v1.0.0 checks; `environment.yml` provides the
same environment specification for Conda users.

```bash
python -m venv .venv
python -m pip install -r requirements-lock.txt
```

On Windows, use `.venv\Scripts\python` in the following commands. On macOS or
Linux, use `.venv/bin/python`.

Validate the public development table:

```bash
python training/saf_reproduce.py validate --input data/development.csv
```

Recompute all seven descriptors from SMILES and compare them with the release
table:

```bash
python scripts/compute_descriptors.py --input data/development.csv \
  --output run_outputs/development_with_recomputed_descriptors.csv \
  --report run_outputs/development_descriptor_audit.json \
  --verify-existing --tolerance 1e-9
```

Run a short four-model structural test. This is a software check, not a
manuscript result:

```bash
python training/saf_reproduce.py smoke --input data/development.csv \
  --config training/config.json --output-dir run_outputs/smoke
```

Reproduce the archived predictions and metrics for the internal held-out set:

```bash
python scripts/evaluate_heldout.py \
  --data data/held_out_test.csv \
  --model models/SAF_Predict_public_model_bundle_v1.joblib \
  --reference-dir validation \
  --output-dir run_outputs/heldout
```

The JavaScript inference implementation is checked against six Python reference
cases with:

```bash
node tests/test_js_against_python_reference_cases.js
```

## Finalize a release manifest

After all release files are final, write the QA report, generate the manifest
and checksum list, then run the read-only final verification in this order:

```bash
python scripts/run_release_qa.py
python scripts/build_release_manifest.py
python scripts/run_release_qa.py --verify-release-manifest
```

The first command updates `RELEASE_CHECKS.json`, which is then included in
`release_manifest.json`. The final command does not write any file; it verifies
that the manifest covers every public release file and that every listed SHA256
digest, including the digest of `release_manifest.json`, matches. Rerun the
three commands after any release-file change.

## Reproduce model development

The full comparison evaluates XGBoost, random forest, support vector regression,
and an artificial neural network for Y01-Y07 using outer five-fold and inner
three-fold `GroupKFold`, with molecular formula as the blocking variable. Inner
selection minimizes RMSE. All grids, fixed parameters, deployment mappings, and
seed derivations are recorded in `training/config.json`.

```bash
python training/saf_reproduce.py nested-cv \
  --input data/development.csv \
  --config training/config.json \
  --output-dir run_outputs/cv \
  --n-jobs 1

python training/saf_reproduce.py train-export \
  --input data/development.csv \
  --config training/config.json \
  --cv-dir run_outputs/cv \
  --output-dir run_outputs/release \
  --version 1.0.0 \
  --n-jobs 1
```

`--n-jobs 1` is the conservative deterministic setting. The first command is
computationally expensive. `artifacts/cv/` contains the archived v1.0.0 fold
manifest, outer-fold scores, 8,400 out-of-fold predictions, model-target
summary, source-stratified metrics, Y02 sensitivity analysis, protocol, and QA
records. During a rerun, the training workflow reconstructs the archived
`formula-derived`/`source-supported` Figure 4 strata from each record's
`reference_map`; Y02 is `constructed`. These analysis strata are distinct from
the detailed value-origin metadata in `data/provenance_long.csv`.

With the lockfile versions and `--n-jobs 1`, compare a new full run with the
archive by fold assignments, selected parameters, predictions, and numerical
metrics—not by file checksums. Generated timestamps, timing fields, and run
metadata mean a byte-for-byte rebuild is not expected. `training/README.md`
documents the output tree and exporter safeguards.

Use the supplied stable-key comparator for a complete nested-CV check. It
requires exact categorical values, compares numerical values with a default
absolute tolerance of `1e-12`, and ignores only timing, timestamp, hash, and
run-instance metadata fields:

```bash
python scripts/compare_cv_artifacts.py \
  --generated-dir run_outputs/cv \
  --archive-dir artifacts/cv \
  --output run_outputs/full_nested_cv_reproduction_check.json
```

The release record for this check is
`validation/full_nested_cv_reproduction_check.json`.
The separate `validation/training_export_reproduction_check.json` records a
fresh locked-environment final fit/export and its numerical agreement with the
released Python and browser predictions. Generated serialized files need not
have the same checksum because they contain run-specific metadata.

## Internal held-out evaluation

| Target | R² | RMSE | MAE |
|---|---:|---:|---:|
| Y01 | 0.902 | 0.577 MJ kg^-1 | 0.463 MJ kg^-1 |
| Y02 | 0.773 | 1.219 MJ L^-1 | 0.967 MJ L^-1 |
| Y03 | 0.873 | 0.0325 g cm^-3 | 0.0238 g cm^-3 |
| Y04 | -0.021 | 36.371 °C | 25.231 °C |
| Y05 | 0.979 | 8.236 °C | 6.437 °C |
| Y06 | 0.920 | 10.697 °C | 7.657 °C |
| Y07 | 0.820 | 0.295 mm² s^-1 | 0.196 mm² s^-1 |

Nine of ten held-out records, representing eight of nine unique formulae, share
a molecular formula with the development set. These results are therefore an
**internal held-out evaluation**, not independent external, prospective, or
experimental validation. File hashes document the released artifact contents
but do not independently prove the chronology of label access.

Y04 performed poorly on the internal holdout (R² = -0.021; MAE = 25.2 °C;
n = 10). Do not use Y04 alone to rank or select compounds; experimental
confirmation is required.

## Repository map

```text
data/          development, held-out, provenance, references, dictionary, folds
training/      complete grouped nested-CV, final-fit, and model-export workflow
models/        fixed Python and browser model artifacts plus model metadata
validation/    archived test predictions, metrics, reports, and parity checks
artifacts/cv/  archived nested-CV predictions, scores, protocol, and QA
scripts/       descriptor audit, held-out evaluation, and release QA utilities
tests/         browser-versus-Python numerical parity tests
```

## Limitations

- The seven descriptors are a coarse representation and cannot resolve all
  constitutional, positional, stereochemical, conformational, or crystal-
  packing effects.
- Solid-liquid phase-transition temperature is a pure-component equilibrium
  property, not the freezing point of a finished fuel blend.
- Boiling point is a pure-component property, not a fuel distillation curve.
- Kinematic viscosity is defined at 40 °C and is not a low-temperature
  aviation-fuel viscosity specification.
- Density and flash-point records retain heterogeneous source conditions rather
  than one certification method; consult `data/provenance_long.csv`.
- Predictions require experimental confirmation before candidate selection or
  engineering use.

## License, citation, and archival release

- Source code: MIT License (`LICENSE`).
- Project-authored data curation: CC BY 4.0 to the extent rights are held;
  third-party source material is not relicensed (`DATA_LICENSE.md`).
- Citation metadata: `CITATION.cff`.
- Zenodo deposit metadata: `.zenodo.json`.

Use the immutable `v1.0.0` GitHub Release for software downloads and the
version-specific Zenodo DOI for scholarly citation. The DOI is assigned by
Zenodo when the release is archived; the repository landing page and release
notes will link to the resulting record.
