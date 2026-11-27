# Hydrocarbon prediction platform — 27 November 2026

**Platform version 3.0.0.** This dated snapshot contains the complete source,
fitted models, reference data, formula-group folds, evaluation records, figures
and reproducibility code supporting the current manuscript revision. Historical
root-level files and earlier releases remain available for their original
versions; use this snapshot for the current analysis.

## Download and run

Download [the complete source ZIP](https://github.com/Wangwuyu-hub/saf-predictor-V1/releases/download/v3.0.0-20261127/Prediction_Platform_Source_20261127.zip)
from the [dated release](https://github.com/Wangwuyu-hub/saf-predictor-V1/releases/tag/v3.0.0-20261127), extract it, and open
the extracted `Prediction_Platform_Source_20261127` folder. Standard-library Python supports
descriptor-only CSV prediction:

```bash
cd Prediction_Platform_Source_20261127
python run_cli.py --input data/example_inputs.csv --output predictions.csv
python -m unittest discover -s tests
```

The desktop interface requires Tk/Tcl support. For SMILES entry and Excel import,
install the descriptor dependencies, then launch the interface:

```bash
python -m pip install -r requirements-descriptors.txt
python run_app.py
```

The source README documents the property definitions, data selection,
model-development commands, locked environments and independent rebuild checks.
Volumetric NHOC is reported as the product of predicted mass-based NHOC and
density; the separately fitted value is retained as a diagnostic.

## Complete copy retained in this directory

The four `.partNN` files are ordered binary segments of the same complete ZIP.
They preserve every source and model file within this dated Git history. They
are not four independent ZIP files. After downloading this directory or cloning
the repository, run:

```bash
python assemble_source.py
cd source/Prediction_Platform_Source_20261127
python run_cli.py --input data/example_inputs.csv --output predictions.csv
```

The assembly script verifies every part and the complete archive, safely extracts
it, and verifies all 1,079 source files against `SOURCE_FILES_SHA256.json`.
`ARCHIVE_PARTS_SHA256.json` records the ordered parts, complete archive digest
and expected source directory. Assembly requires only standard-library Python.

## Spreadsheet exports

- [Supporting Information 2](Supporting_Information_2_20261127.xlsx)
- [Platform output records](Prediction_Platform_Outputs_20261127.xlsx)

The portable inference checks run on this exact extracted snapshot. The full
training workflow and its previously executed reproduction evidence are included
in the source archive; the routine repository check does not repeat model
training. No native Windows executable is supplied or claimed to have been tested
by this publication package.
