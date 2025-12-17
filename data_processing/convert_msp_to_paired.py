"""Convert an MSP file into per-spectrum .ms files and a labels TSV for Spec2Mol.

This script parses a typical MSP (NIST-style) file with entries like:
  Name: ...
  SMILES: ...
  Formula: ...
  InChIKey: ...
  Num Peaks: N
  <m/z> <intensity>

It writes:
- a `labels.tsv` with columns: spec, formula, smiles, inchikey, instrument
- one `.ms` file per spectrum in a simple SIRIUS-like format that
  `src.datasets.spectra_utils.parse_spectra` can read.

Usage:
  python convert_msp_to_paired.py \
      --msp-file ../data/antibio/antibio_new2.msp \
      --out-spectra-dir ../data/antibio/spec_files \
      --out-labels ../data/antibio/labels.tsv \
      --split 0.8 0.1 0.1 \
      --out-split ../data/antibio/split_80_10_10.tsv \
      --seed 42
"""

import argparse
import os
from pathlib import Path
import re
import csv
import random
from typing import List, Dict, Tuple, Optional
import pandas as pd



def parse_msp_records(msp_path: str) -> List[Dict]:
    """Parse MSP file into a list of records (dicts).

    Each record contains metadata keys (Name, SMILES, Formula, InChIKey, Instrument,
    PrecursorMZ, Num Peaks, etc.) and a `peaks` list of (mz, intensity) tuples.
    """
    text = open(msp_path, 'r', encoding='utf-8', errors='ignore').read()
    # Split records by blank line(s) -- robust to Windows/Unix
    raw_records = re.split(r"\n\s*\n", text.strip())
    records = []
    for raw in raw_records:
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        if not lines:
            continue
        rec = {}
        peaks: List[Tuple[float, float]] = []
        peak_section = False
        for i, line in enumerate(lines):
            if ':' in line and not re.match(r"^\s*\d", line):
                # metadata line like 'Name: ...' or 'SMILES: ...'
                key, val = line.split(':', 1)
                rec[key.strip()] = val.strip()
                # detect start of peaks following 'Num Peaks' optionally
                if key.strip().lower() == 'num peaks':
                    peak_section = True
            else:
                # If line looks like two floats, treat as peak
                m = re.match(r"^([0-9]+\.?[0-9eE+-]*)\s+([0-9]+\.?[0-9eE+-]*)", line)
                if m:
                    mz = float(m.group(1))
                    inten = float(m.group(2))
                    peaks.append((mz, inten))
                    peak_section = True
                else:
                    # fallback: sometimes metadata keys have no colon, ignore
                    pass

        rec['peaks'] = peaks
        records.append(rec)

    return records


def write_ms_file(rec: Dict, out_path: Path) -> None:
    """Write a minimal SIRIUS-like .ms file that `parse_spectra` can parse.

    The format written here contains lines starting with '#' for metadata
    and one '>' block with peak lines.
    """
    lines = []
    # Common metadata keys mapped to SIRIUS-like metadata
    if 'Instrument' in rec:
        lines.append(f"#INSTRUMENT TYPE {rec.get('Instrument')}")
    elif 'Instrument_type' in rec:
        lines.append(f"#INSTRUMENT TYPE {rec.get('Instrument_type')}")

    if 'PrecursorMZ' in rec:
        lines.append(f"#PEPMASS {rec.get('PrecursorMZ')}")
    
    if 'Precursor_type' in rec:
        lines.append(f"#PRECURSORTYPE {rec.get('Precursor_type')}")
    
    if 'Collision_energy' in rec:
        lines.append(f"#COLLISIONENERGY {rec.get('Collision_energy')}")

    if 'Formula' in rec:
        lines.append(f"#FORMULA {rec.get('Formula')}")
    if 'SMILES' in rec:
        lines.append(f"#SMILES {rec.get('SMILES')}")
    if 'InChIKey' in rec:
        lines.append(f"#INCHIKEY {rec.get('InChIKey')}")

    # Add a simple spectrum block
    spectrum_name = out_path.stem
    lines.append(f">{spectrum_name}")
    for mz, inten in rec.get('peaks', []):
        lines.append(f"{mz} {inten}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding='utf-8')


def build_labels(records, out_dir):
    """build_labels.
    Args:
        records:
        out_dir:
    """
    names, formulas, smiles, inchikeys, spec_files, precursor_types, collision_energies = [], [], [], [], [], [], []
    if not Path(out_dir).exists():
        Path(out_dir).mkdir(exist_ok=True, parents=True)
    for i, record in enumerate(records):
        spec_name = f"spec_{i:06d}"
        names.append(spec_name)
        formulas.append(record.get("Formula", "Unknown"))
        smiles.append(record.get("SMILES", "Unknown"))
        inchikeys.append(record.get("InChIKey", "Unknown"))
        precursor_types.append(record.get("Precursor_type", "Unknown"))
        collision_energies.append(record.get("Collision_energy", "0.0"))
        spec_file = Path(out_dir) / f"{spec_name}.ms"
        spec_files.append(str(spec_file))
    df = pd.DataFrame(
        {
            "name": names,
            "formula": formulas,
            "smiles": smiles,
            "inchikey": inchikeys,
            "spec_file": spec_files,
            "precursor_type": precursor_types,
            "collision_energy": collision_energies,
        }
    )
    return df


def write_split(specs: List[str], out_split: Path, ratios: Tuple[float, float, float], seed: int = 42) -> None:
    """Create a simple preset split TSV with columns name, split (train/val/test).

    ratios: (train, val, test) summing to ~1.0
    Matches the format of existing split files (canopus_hplus_*.tsv)
    """
    random.seed(seed)
    specs_shuffled = specs[:]
    random.shuffle(specs_shuffled)
    n = len(specs_shuffled)
    t = int(ratios[0] * n)
    v = int(ratios[1] * n)
    train = specs_shuffled[:t]
    val = specs_shuffled[t:t+v]
    test = specs_shuffled[t+v:]

    out_split.parent.mkdir(parents=True, exist_ok=True)
    with open(out_split, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['name', 'split'])
        for s in train:
            writer.writerow([s, 'train'])
        for s in val:
            writer.writerow([s, 'val'])
        for s in test:
            writer.writerow([s, 'test'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--msp-file', required=True, help='Path to MSP file')
    parser.add_argument('--out-spectra-dir', required=True, help='Directory to write per-spectrum .ms files')
    parser.add_argument('--out-labels', required=True, help='Path to output labels.tsv (tsv)')
    parser.add_argument('--out-split', required=False, help='Optional output split file (tsv)')
    parser.add_argument('--split', nargs=3, type=float, default=(0.8, 0.1, 0.1), help='Train/Val/Test ratios')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    records = parse_msp_records(args.msp_file)
    if not records:
        print('No records parsed from MSP file.')
        return

    specs = build_labels(records, Path(args.out_spectra_dir))

    # Write individual spectra files
    for record, spec_file in zip(records, specs['spec_file']):
        write_ms_file(record, Path(spec_file))

    # Save labels file
    specs.to_csv(args.out_labels, sep='\t', index=False)
    print(f'Wrote {len(specs)} spectra to {args.out_spectra_dir} and labels to {args.out_labels}')

    if args.out_split:
        # Pass list of names, not dataframe
        write_split(specs['name'].tolist(), Path(args.out_split), tuple(args.split), seed=args.seed)
        print(f'Wrote split file to {args.out_split}')


if __name__ == '__main__':
    main()
