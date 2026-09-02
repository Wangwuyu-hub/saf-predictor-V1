#!/usr/bin/env python3
"""Compute and audit the seven SAF-Predict molecular descriptors from SMILES."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors, GraphDescriptors, rdMolDescriptors


FEATURES = [f"X{i:02d}" for i in range(1, 8)]


def descriptor_vector(smiles: str) -> dict[str, float]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("SMILES could not be parsed")
    if len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError("Only single-component molecules are supported")
    if any(atom.GetAtomicNum() != 6 for atom in mol.GetAtoms()):
        raise ValueError("Only C/H-only hydrocarbons are supported")

    carbon_atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6]
    carbon_count = len(carbon_atoms)
    if carbon_count == 0:
        raise ValueError("A hydrocarbon must contain at least one carbon atom")
    hydrogen_count = sum(int(atom.GetTotalNumHs(includeNeighbors=True)) for atom in carbon_atoms)
    aromatic_carbon_count = sum(int(atom.GetIsAromatic()) for atom in carbon_atoms)
    branch_count = sum(int(atom.GetDegree() >= 3) for atom in carbon_atoms)

    # The release definition intentionally ignores stereochemistry for X06.
    graph_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    graph_mol = Chem.MolFromSmiles(graph_smiles)
    if graph_mol is None:
        raise ValueError("Canonical non-isomeric graph could not be reconstructed")
    automorphism_count = len(
        graph_mol.GetSubstructMatches(graph_mol, uniquify=False, maxMatches=1_000_000)
    )

    return {
        "X01": float(Descriptors.MolWt(mol)),
        "X02": float(hydrogen_count / carbon_count),
        "X03": float(aromatic_carbon_count / carbon_count),
        "X04": float(rdMolDescriptors.CalcNumRings(mol)),
        "X05": float(branch_count),
        "X06": float(math.log1p(automorphism_count)),
        "X07": float(GraphDescriptors.Kappa2(mol)),
    }


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("Input CSV has no header")
        rows = list(reader)
        return list(reader.fieldnames), rows


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="CSV containing canonical_smiles")
    parser.add_argument("--output", type=Path, required=True, help="CSV with computed X01-X07")
    parser.add_argument("--report", type=Path, help="JSON audit report path")
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="Compare computed descriptors with existing X01-X07 columns",
    )
    parser.add_argument("--tolerance", type=float, default=1e-9)
    args = parser.parse_args()

    headers, rows = read_rows(args.input)
    if "canonical_smiles" not in headers:
        raise ValueError("Input CSV must contain canonical_smiles")
    if args.verify_existing and any(feature not in headers for feature in FEATURES):
        raise ValueError("--verify-existing requires X01-X07 in the input")

    output_headers = list(headers)
    for feature in FEATURES:
        if feature not in output_headers:
            output_headers.append(feature)

    output_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    maxima = {feature: 0.0 for feature in FEATURES}
    mismatches = {feature: 0 for feature in FEATURES}
    for index, row in enumerate(rows, start=2):
        record_id = row.get("record_id") or f"row-{index}"
        try:
            computed = descriptor_vector(row["canonical_smiles"])
        except Exception as exc:
            failures.append({"record_id": record_id, "error": str(exc)})
            continue
        if args.verify_existing:
            for feature in FEATURES:
                difference = abs(float(row[feature]) - computed[feature])
                maxima[feature] = max(maxima[feature], difference)
                mismatches[feature] += int(difference > args.tolerance)
        output_rows.append({**row, **{key: f"{value:.15g}" for key, value in computed.items()}})

    if failures:
        raise ValueError(f"Descriptor computation failed for {len(failures)} row(s): {failures[:5]}")
    write_rows(args.output, output_headers, output_rows)

    report = {
        "status": "pass" if not any(mismatches.values()) else "fail",
        "input": str(args.input),
        "output": str(args.output),
        "records": len(rows),
        "rdkit_version": rdBase.rdkitVersion,
        "verify_existing": args.verify_existing,
        "tolerance": args.tolerance,
        "maximum_absolute_difference": maxima,
        "mismatch_count": mismatches,
        "definitions": {
            "X01": "RDKit Descriptors.MolWt average molecular mass",
            "X02": "total hydrogen atom count divided by carbon atom count",
            "X03": "aromatic carbon atom count divided by carbon atom count",
            "X04": "RDKit rdMolDescriptors.CalcNumRings",
            "X05": "carbon atoms bonded directly to at least three carbon atoms",
            "X06": "ln(1 + graph automorphism match count), ignoring stereochemistry",
            "X07": "RDKit GraphDescriptors.Kappa2",
        },
    }
    report_path = args.report or args.output.with_suffix(".audit.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["status"] != "pass":
        sys.exit(1)


if __name__ == "__main__":
    main()
