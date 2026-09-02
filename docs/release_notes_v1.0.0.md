# SAF-Predict v1.0.0

This is the first archival release of SAF-Predict, an offline browser tool and
reproducibility package for descriptor-based physicochemical-property
prediction of pure hydrocarbon fuel candidates.

## Included

- Installation-free English browser interface for single and batch prediction.
- Fixed Python Joblib and browser JavaScript/JSON model artifacts.
- Seven inputs and seven outputs with explicit definitions and units.
- A 300-record development table and a 10-record structurally stratified
  internal held-out test table.
- One provenance row for each of 2,170 molecule-property pairs and a 491-record
  source registry.
- A formatted seven-sheet XLSX data release and machine-readable CSV tables.
- Fixed five-fold formula-group assignments, complete four-algorithm nested-CV
  artifacts, hyperparameter grids, random seeds, and final deployment mapping.
- Scripts for descriptor recalculation, data validation, nested CV, model
  export, held-out evaluation, browser/Python parity, and release integrity.
- Machine-readable checks showing exact recovery of the archived nested-CV
  predictions and numerical recovery of the deployed Python/browser outputs.
- Pinned Python environment, MIT software license, mixed-source data-rights
  statement, citation metadata, and Zenodo metadata.

## Internal held-out performance

| Target | R² | RMSE | MAE |
|---|---:|---:|---:|
| Mass-based net heat of combustion | 0.902 | 0.577 MJ kg^-1 | 0.463 MJ kg^-1 |
| Volumetric net heat of combustion | 0.773 | 1.219 MJ L^-1 | 0.967 MJ L^-1 |
| Density | 0.873 | 0.0325 g cm^-3 | 0.0238 g cm^-3 |
| Solid-liquid phase-transition temperature | -0.021 | 36.371 °C | 25.231 °C |
| Boiling point | 0.979 | 8.236 °C | 6.437 °C |
| Flash point | 0.920 | 10.697 °C | 7.657 °C |
| Kinematic viscosity at 40 °C | 0.820 | 0.295 mm² s^-1 | 0.196 mm² s^-1 |

The 10 records form an internal held-out test, not independent external,
prospective, or experimental validation. Nine records, representing eight of
nine unique formulae, share a molecular formula with the development set.

Y04 performed poorly on the internal holdout (R² = -0.021; MAE = 25.2 °C;
n = 10). Do not use Y04 alone to rank or select compounds; experimental
confirmation is required.

## Download choices

- `SAF-Predict-v1.0.0-offline.zip`: smallest package for direct browser use.
- `SAF-Predict-v1.0.0-reproducibility.zip`: data, provenance, models, scripts,
  environment files, tests, and archived analysis artifacts.
- `SAF_Hydrocarbon_Dataset_v1.0.0.xlsx`: formatted data and source workbook.
- `release-assets.sha256`: SHA256 digests for the attached archives and
  workbook. The repository-level `checksums.sha256` covers individual files
  inside the reproducibility snapshot.

The version-specific Zenodo record is the recommended scholarly citation. The
GitHub release is the recommended software download location.
