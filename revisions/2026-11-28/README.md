# Hydrocarbon prediction platform — 28 November 2026

This directory identifies the **version 3.1.0** source and numerical results for
the current manuscript revision. The complete source package is available in
the [dated release](https://github.com/Wangwuyu-hub/saf-predictor-V1/releases/tag/v3.1.0-20261128).

## Downloads

- [Prediction_Platform_Source_20261128.zip](https://github.com/Wangwuyu-hub/saf-predictor-V1/releases/download/v3.1.0-20261128/Prediction_Platform_Source_20261128.zip):
  complete source, frozen data and folds, fitted models, training and analysis
  code, figures, and executed verification records (1,098 files).
- [Supporting_Information_2_20261128.xlsx](https://github.com/Wangwuyu-hub/saf-predictor-V1/releases/download/v3.1.0-20261128/Supporting_Information_2_20261128.xlsx):
  original source records, condition-matched references, updated predictions,
  full metrics, and ANN convergence and budget-sensitivity summaries.
- [Prediction_Platform_Outputs_20261128.xlsx](https://github.com/Wangwuyu-hub/saf-predictor-V1/releases/download/v3.1.0-20261128/Prediction_Platform_Outputs_20261128.xlsx):
  all ten platform cases, comparison eligibility and numerical consistency checks.
- [SHA256SUMS.txt](SHA256SUMS.txt) and [SOURCE_FILES_SHA256.json](https://github.com/Wangwuyu-hub/saf-predictor-V1/releases/download/v3.1.0-20261128/SOURCE_FILES_SHA256.json):
  archive/export digests and the contents of the extracted source package.

The source ZIP SHA-256 is `8a67b8d083bebcca1706f263ed0c88447fa0855eba7346180df6f460e9fa58b1`.

## Verified numerical revision

The data and molecular-formula groups are unchanged. All model searches and
fitting were rerun using a common ANN L-BFGS budget of 50,000 iterations
and 1,000,000 function evaluations, with tolerance 1e-6. The budget was
fixed using optimizer termination diagnostics before the final analysis.

All 1,473 audited fits completed without convergence warnings or iteration-limit
termination. Doubling the budgets for 72 selected ANN refits reproduced
all 2,517 checked predictions and training losses exactly. The independent clean
rebuild reproduced 16 numerical report CSVs, four selection manifests and
all ten final platform cases exactly. The source package contains the underlying
candidate-level records and the complete clean-rebuild command.

The current density and viscosity outputs refer to 20 °C and 40 °C, respectively.
Volumetric net heat of combustion is the product of predicted mass-based heat
and density; the direct estimate is returned separately. All ten comparison
records and their source-defined metric eligibility remain visible.

## Download and run

Download the ZIP directly, or save `download_source.py` and run:

```bash
python download_source.py --output .
cd Prediction_Platform_Source_20261128
python run_cli.py --input data/example_inputs.csv --output predictions.csv
```

The downloader checks the release SHA-256 before extraction. The full README
inside the source archive provides desktop GUI, SMILES, independent training,
figure-regeneration and native Windows build instructions. Earlier dated
releases retain their historical numerical results.
