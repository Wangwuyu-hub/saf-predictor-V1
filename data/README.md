# Dataset and provenance tables

This directory is the machine-readable v1.1.0 data release used by the training
and evaluation scripts. It contains 300 development records and 10 structurally
stratified internal held-out records. The held-out set is not independent
external validation.

| File | Rows excluding header | Purpose |
|---|---:|---|
| `development.csv` | 300 | Grouped cross-validation, tuning, final fitting, interval calibration, and applicability-domain calibration |
| `held_out_test.csv` | 10 | One-time internal software/model evaluation |
| `provenance_long.csv` | 2,170 | One provenance record for each molecule-property pair, Y01-Y07; unavailable Y02 labels are explicit |
| `references.csv` | 492 | Citation, DOI/URL, source class, host, and access note |
| `data_dictionary.csv` | 104 | Field definitions, units, rules, and missing-value policies |
| `cv_fold_manifest.csv` | 300 | Fixed five-fold outer GroupKFold assignment |
| `y02_direct_response_reconciliation.csv` | 170 | Identity linkage for the 162 development and 8 held-out Y02 labels retained in v1.1.0 |
| `y02_full_reconciliation_audit.csv` | 243 | Complete legacy-Y02 matching audit, including unmatched and conflicting rows |
| `y02_physical_baseline_audit.csv` | 310 | Y01 × Y03 comparator retained outside the modelling tables for sensitivity analysis only |
| `SAF_Hydrocarbon_Dataset_v1.1.0.xlsx` | 10 sheets | Formatted copy of the release and Y02 audit tables plus a README sheet |
| `data_release_checks.json` | - | Counts and invariant checks generated during release construction |

## Value-origin classes

`provenance_long.csv` distinguishes experimental measurements,
source-reported experimental values, evaluated or compiled literature values,
values calculated from literature thermochemistry, formula or correlation
estimates, source-database values whose measurement status could not be
resolved, independently collected Y02 responses, and unavailable Y02 labels.
These labels describe provenance;
they do not imply harmonized experimental conditions.

Y02 is a separately curated response rather than a value constructed from the
current Y01 and Y03 columns. It is available for 162 development records in 66
molecular-formula groups and 8 of the 10 internal held-out records. The other
Y02 cells remain missing and are excluded from Y02 fitting and evaluation.
The old product is reproducible in `y02_physical_baseline_audit.csv` but is not
a v1.1.0 target label.

The Y02 identity matches are documented, but the legacy table does not resolve
direct experimental reporting, primary-source lineage, or measurement
conditions for every row. Accordingly, these records are described as
independently collected responses, not uniformly as experimental or
high-confidence measurements. The field
`high_confidence_source_subset` is a conservative source-sensitivity flag, not
a guarantee that all measurements were made under identical conditions.

## Identity and scope

All release records are neutral, pure, single-component C/H-only molecules.
InChIKey and canonical SMILES are complete. Missing CAS RNs and PubChem CIDs
remain missing when no unambiguous identifier was retained. The source table
does not fabricate missing identifiers.

See `../DATA_LICENSE.md` before redistributing source-derived values or
reference material.
