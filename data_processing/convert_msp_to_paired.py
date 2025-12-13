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
  python data_processing/convert_msp_to_paired.py \
      --msp-file path/to/file.msp \
      --out-spectra-dir ../data/your_dataset/spectra \
      --out-labels ../data/your_dataset/labels.tsv \
      --split 0.8 0.1 0.1
"""

import argparse
import os
from pathlib import Path
import re
import csv
import random
from typing import List, Dict, Tuple, Optional


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
    if 'Formula' in rec:
        lines.append(f"#FORMULA {rec.get('Formula')}")
    if 'SMILES' in rec:
        lines.append(f"#SMILES {rec.get('SMILES')}")
    if 'SMILES' not in rec and 'SMILES' in rec:
        lines.append(f"#SMILES {rec.get('SMILES')}")
    if 'InChIKey' in rec:
        lines.append(f"#INCHIKEY {rec.get('InChIKey')}")

    # Add a simple spectrum block
    spectrum_name = rec.get('Name', out_path.stem)
    lines.append(f">{spectrum_name}")
    for mz, inten in rec.get('peaks', []):
        lines.append(f"{mz} {inten}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding='utf-8')


def build_labels(records: List[Dict], spectra_dir: Path, labels_path: Path) -> List[str]:
    """Write labels.tsv and return list of spec names (filenames without ext).

    Columns: spec, formula, smiles, inchikey, instrument
    """
    rows = []
    for i, rec in enumerate(records):
        # Use numeric naming: spec_000001, spec_000002, etc.
        spec_stem = f'spec_{i:06d}'

        # Write .ms file
        ms_path = spectra_dir / f"{spec_stem}.ms"
        write_ms_file(rec, ms_path)

        rows.append({
            'spec': spec_stem,
            'formula': rec.get('Formula', ''),
            'smiles': rec.get('SMILES', ''),
            'inchikey': rec.get('InChIKey', ''),
            'instrument': rec.get('Instrument', rec.get('Instrument_type', '')),
        })

    # Write TSV
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    with open(labels_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['spec', 'formula', 'smiles', 'inchikey', 'instrument'], delimiter='\t')
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    return [r['spec'] for r in rows]


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

    specs = build_labels(records, Path(args.out_spectra_dir), Path(args.out_labels))
    print(f'Wrote {len(specs)} spectra to {args.out_spectra_dir} and labels to {args.out_labels}')

    if args.out_split:
        write_split(specs, Path(args.out_split), tuple(args.split), seed=args.seed)
        print(f'Wrote split file to {args.out_split}')


if __name__ == '__main__':
    main()
