# SAF-Predict model card

- Version: 1.0.0
- Metadata generated: 2026-08-31T15:03:44.071220+00:00
- Purpose: Preliminary physicochemical screening of pure SAF-relevant hydrocarbons from seven molecular descriptors
- Recorded development data: Sheet1, n = 300, 73 molecular-formula groups
- Internal held-out test: Sheet2, n = 10; not external or prospective validation
- Test-set overlap: 9/10 records (8/9 unique formulae) share a molecular formula with Sheet1

The analysis-design statements in this card are author-provided metadata. The repository does not contain an independently timestamped pre-evaluation model release and therefore does not independently establish when Sheet2 reference values were first accessed.

## Deployment mapping

| Target | Property | Deployment | Recorded Sheet1 group-CV RMSE |
|---|---|---|---:|
| Y01 | Mass-based net heat of combustion | RF | 0.316261 |
| Y02 | Volumetric net heat of combustion | Derived from Y01 and Y03 predictions | derived |
| Y03 | Density | SVR | 0.0200269 |
| Y04 | Solid–liquid phase-transition temperature | XGBoost | 23.0725 |
| Y05 | Boiling point | SVR | 9.81759 |
| Y06 | Flash point | SVR | 7.64792 |
| Y07 | Kinematic viscosity at 40 °C | SVR | 0.390172 |

## Limitations

- In the authors' curation schema, labels are categorized as source-supported, formula-derived, or constructed. The record-level provenance table is not included in this software-only repository, so these categories cannot be independently traced here to condition-matched measurements.
- The model is intended for neutral, single-component hydrocarbon molecules with descriptors computed using the documented definitions.
- Solid–liquid phase-transition temperature is not the freezing point of a finished fuel blend.
- Kinematic viscosity is defined at 40 °C and is not a low-temperature aviation-fuel viscosity specification.
- Boiling point is a pure-component property, not a finished-fuel distillation curve.
- Predictions do not establish conformity with ASTM specifications, finished-fuel suitability, sustainability, synthetic feasibility, safety, cost, emissions, or engine performance.
- According to the curation metadata, density and flash-point labels retain heterogeneous source conditions rather than one standardized certification method. The full condition table is outside this software-only release.
- The ten-record Sheet2 set is an internal held-out test set, not independent external or prospective validation.
- Nine of the ten held-out records, representing eight of nine unique formulae, share a molecular formula with the development set.
- Y04 performed poorly on the internal holdout (R² = −0.021; MAE = 25.2 °C; n = 10). Do not use Y04 alone to rank or select compounds; experimental confirmation is required.

## Artifact identifiers and repository scope

- Recorded source-workbook SHA256 (workbook not included): `584ff1ba94401a00ffd9197808ade834314183c87adde2422dc33d7aa685c897`
- Original private Joblib SHA256: `e5a648cba47e0ba98e902144e45b937610f82c0d96719b4ef248cf3ab903ef73`
- Original private browser-bundle SHA256: `37146ab167a02c560b40f5296b62cf48cab8f9ea1abb2ec70db34bc59b89fa2f`
- Original wide held-out-prediction SHA256: `040a212e0ff390af7d7ec8a0e68e7c77c5c19b09d2bbd3c6b274a5b245e19018`

The public de-identified artifact hashes are recorded separately in `model_artifact_manifest.json` and the final release manifest. Molecule names, molecular formulae, original development-record identifiers, and bilingual display metadata were removed; X01–X07 vectors remain for applicability-domain calculations and may be linkable to a known source dataset. Hashes identify file contents; they do not prove analysis chronology. This software repository does not include the complete development dataset, record-level provenance table, descriptor-generation workflow, or end-to-end model-development pipeline and is therefore not a complete computational reproducibility archive.

Predictions are intended solely for research-stage physicochemical prescreening of pure hydrocarbon molecules within the represented descriptor space.
