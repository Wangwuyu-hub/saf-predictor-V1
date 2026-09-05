# SAF-Predict v1.1.0

Version 1.1.0 changes the scientific meaning and implementation of Y02
(volumetric net heat of combustion). Users should not mix v1.0.0 Y02 outputs or
artifacts with this release.

## Scientific and data changes

- Y02 is an independently collected response rather than the algebraic product
  of Y01 and Y03.
- The development table retains 162 identity-resolved Y02 labels spanning 66
  molecular-formula groups. The other 138 Y02 cells remain missing and are not
  imputed or reconstructed.
- Eight of the ten structurally stratified internal held-out records have a Y02
  reference label; metrics for Y02 therefore use n = 8.
- Y02 identity reconciliation and exclusions are documented in
  `data/y02_direct_response_reconciliation.csv` and
  `data/y02_full_reconciliation_audit.csv`.
- The Y01 × Y03 product is retained only in
  `data/y02_physical_baseline_audit.csv` and the corresponding nested-CV
  sensitivity artifact. It is not a deployed prediction rule.
- The release workbook is `data/SAF_Hydrocarbon_Dataset_v1.1.0.xlsx`.

The Y02 identity linkage is recorded, but direct experimental status, exact
primary source, and measurement conditions remain unresolved for some legacy
entries. The release therefore does not automatically classify all retained
Y02 values as high-confidence experimental measurements.

## Model and software changes

- The browser and Python bundles now contain seven fitted deployment models.
- Y02 is predicted by a separately fitted random-forest regressor trained on
  the 162 available development labels.
- Missing Y02 labels are excluded only from Y02 fitting and evaluation; the six
  other targets continue to use all 300 development records.
- Cross-validation, uncertainty calibration, held-out evaluation, and the
  JavaScript/Python parity tests now apply the direct-response Y02 definition.
- The browser parity test explicitly rejects a bundle whose Y02 predictions are
  identically Y01 × Y03 on the stored reference cases.
- A release-assembly check verifies numerical equivalence among the private
  export, metadata-reduced public Joblib, and browser runtime.
- The release includes an XGBoost-reference versus deployment-model consistency
  table and a numerical v1.0.0-to-v1.1.0 regression audit for the six targets
  whose data and deployment definitions were intended to remain unchanged.

## Interpretation boundary

The 10-record set remains a structurally stratified internal held-out test from
the same curation workflow. It is not independent external, prospective, or
experimental validation. Current property-specific metrics, record-level
residuals, and 90%/95% interval coverage are provided in the `validation/`
directory; no metric is carried forward from v1.0.0.

The v1.0.0 tag remains the immutable archive for the earlier constructed-Y02
release.
