# Dataset and provenance tables

This directory is the machine-readable v1.0.0 data release used by the training
and evaluation scripts. It contains 300 development records and 10 structurally
stratified internal held-out records. The held-out set is not independent
external validation.

| File | Rows excluding header | Purpose |
|---|---:|---|
| `development.csv` | 300 | Grouped cross-validation, tuning, final fitting, interval calibration, and applicability-domain calibration |
| `held_out_test.csv` | 10 | One-time internal software/model evaluation |
| `provenance_long.csv` | 2,170 | One source record for each molecule-property pair, Y01-Y07 |
| `references.csv` | 491 | Citation, DOI/URL, source class, host, and access note |
| `data_dictionary.csv` | 95 | Field definitions, units, rules, and missing-value policies |
| `cv_fold_manifest.csv` | 300 | Fixed five-fold outer GroupKFold assignment |
| `SAF_Hydrocarbon_Dataset_v1.0.0.xlsx` | 7 sheets | Formatted copy of the release tables plus a README sheet |
| `data_release_checks.json` | - | Counts and invariant checks generated during release construction |

## Value-origin classes

`provenance_long.csv` distinguishes experimental measurements,
source-reported experimental values, evaluated or compiled literature values,
values calculated from literature thermochemistry, formula or correlation
estimates, source-database values whose measurement status could not be
resolved, and the constructed Y02 endpoint. These labels describe provenance;
they do not imply harmonized experimental conditions.

Y02 is always constructed as Y01 multiplied by Y03. The field
`high_confidence_source_subset` is a conservative source-sensitivity flag, not
a guarantee that all measurements were made under identical conditions.

## Identity and scope

All release records are neutral, pure, single-component C/H-only molecules.
InChIKey and canonical SMILES are complete. Missing CAS RNs and PubChem CIDs
remain missing when no unambiguous identifier was retained. The source table
does not fabricate missing identifiers.

See `../DATA_LICENSE.md` before redistributing source-derived values or
reference material.
