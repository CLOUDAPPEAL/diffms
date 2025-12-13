"""Generate subformulae JSON files from MSP data and spectrum files.

This script creates JSON files in the CANOPUS subformulae format,
matching the structure shown in CCMSLIB00000001563.json and CCMSLIB00000001566.json.

Usage:
  python data_processing/generate_subformulae.py \
      --labels-file data/antibio/labels.tsv \
      --spec-folder data/antibio/spec_files \
      --out-dir data/antibio/subformulae \
      --precursor-mz-field PrecursorMZ
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdMolDescriptors import CalcMolFormula


def parse_ms_file(ms_path: Path) -> Tuple[Dict, List[Tuple[float, float]]]:
    """Parse a SIRIUS-like .ms file.
    
    Returns:
        (metadata_dict, peaks_list) where peaks_list is [(mz, intensity), ...]
    """
    lines = open(ms_path, 'r', encoding='utf-8').readlines()
    metadata = {}
    peaks = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith('#'):
            # Metadata line: #KEY value
            parts = line[1:].split(None, 1)
            if len(parts) == 2:
                key, val = parts
                metadata[key.lower()] = val
        elif line.startswith('>'):
            # Spectrum header (ignore for now)
            pass
        else:
            # Try to parse as peak (mz intensity)
            try:
                parts = line.split()
                if len(parts) >= 2:
                    mz = float(parts[0])
                    intensity = float(parts[1])
                    peaks.append((mz, intensity))
            except (ValueError, IndexError):
                pass
    
    return metadata, peaks


def inchi_to_formula(inchi: str) -> Optional[str]:
    """Convert InChI to molecular formula."""
    try:
        mol = Chem.MolFromInchi(inchi)
        if mol is None:
            return None
        return CalcMolFormula(mol)
    except:
        return None


def smiles_to_formula(smiles: str) -> Optional[str]:
    """Convert SMILES to molecular formula."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return CalcMolFormula(mol)
    except:
        return None


def get_precursor_mz(row: pd.Series, precursor_field: str) -> Optional[float]:
    """Extract precursor m/z from row."""
    if precursor_field in row and pd.notna(row[precursor_field]):
        try:
            return float(row[precursor_field])
        except (ValueError, TypeError):
            pass
    return None


def normalize_intensities(intensities: List[float]) -> List[float]:
    """Normalize intensities to [0, 1] range with max = 1.0"""
    if not intensities or max(intensities) == 0:
        return intensities
    max_inten = max(intensities)
    return [i / max_inten for i in intensities]


def compute_monoisotopic_mass(formula: str) -> Optional[float]:
    """Compute monoisotopic mass from formula string."""
    try:
        mol = Chem.MolFromSmiles('')  # dummy molecule to get access to functions
        # Better: use formula directly if possible
        from rdkit.Chem import Descriptors
        # Parse formula and compute mass manually
        # For now, use simple approach via RDKit
        from rdkit.Chem import rdMolDescriptors
        # We need the actual molecule - let's try a different approach
        # Just return None for now and let user provide it
        return None
    except:
        return None


def compute_mass_diff_ppm(observed_mz: float, theoretical_mz: float) -> float:
    """Compute mass difference in ppm."""
    if theoretical_mz == 0:
        return 0
    return (observed_mz - theoretical_mz) / theoretical_mz * 1e6


def create_subformulae_entry(
    spec_name: str,
    smiles: str,
    formula: str,
    peaks: List[Tuple[float, float]],
    precursor_mz: Optional[float] = None,
    ion_type: str = "[M+H]+"
) -> Dict:
    """Create a subformulae JSON entry matching the CANOPUS format.
    
    Args:
        spec_name: spectrum name/ID
        smiles: SMILES string
        formula: molecular formula
        peaks: list of (mz, intensity) tuples (sorted by m/z)
        precursor_mz: precursor m/z value
        ion_type: ionization type (e.g., "[M+H]+", "[M+Na]+")
    
    Returns:
        dict matching the subformulae JSON structure
    """
    if not peaks:
        peaks = [(0, 0)]
    
    # Sort peaks by m/z
    peaks = sorted(peaks, key=lambda x: x[0])
    mz_list, inten_list = zip(*peaks)
    mz_list = list(mz_list)
    inten_list = list(inten_list)
    
    # Normalize intensities
    normalized_inten = normalize_intensities(inten_list)
    
    # For now, we'll create simplified entries
    # In a real scenario, you'd compute theoretical masses for fragments
    output_tbl = {
        "mz": mz_list,
        "ms2_inten": normalized_inten,
        "mono_mass": mz_list,  # Placeholder - ideally compute from fragments
        "abs_mass_diff": [0.0] * len(mz_list),  # Placeholder
        "mass_diff": [0.0] * len(mz_list),  # Placeholder
        "formula": [formula] * len(mz_list),  # Simplified - one formula per peak
        "ions": [ion_type] * len(mz_list)
    }
    
    entry = {
        "cand_form": formula,
        "cand_ion": ion_type,
        "output_tbl": output_tbl
    }
    
    return entry


def main():
    parser = argparse.ArgumentParser(
        description="Generate subformulae JSON files from MSP-derived spectrum and label data."
    )
    parser.add_argument('--labels-file', required=True, help='Path to labels.tsv')
    parser.add_argument('--spec-folder', required=True, help='Folder containing .ms files')
    parser.add_argument('--out-dir', required=True, help='Output directory for JSON files')
    parser.add_argument('--precursor-mz-field', default='PrecursorMZ', 
                        help='Column name in labels for precursor m/z')
    parser.add_argument('--ion-type', default='[M+H]+', help='Default ionization type')
    
    args = parser.parse_args()
    
    # Read labels
    labels_df = pd.read_csv(args.labels_file, sep='\t')
    spec_folder = Path(args.spec_folder)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    count = 0
    skipped = 0
    
    for idx, row in labels_df.iterrows():
        spec_name = row['spec']
        smiles = row.get('smiles', '')
        formula = row.get('formula', '')
        
        # Try to get formula from SMILES if not provided
        if not formula and smiles:
            formula = smiles_to_formula(smiles)
        
        if not formula:
            print(f"⚠ Skipping {spec_name}: no formula found")
            skipped += 1
            continue
        
        # Load spectrum peaks
        ms_path = spec_folder / f"{spec_name}.ms"
        if not ms_path.exists():
            print(f"⚠ Skipping {spec_name}: .ms file not found")
            skipped += 1
            continue
        
        metadata, peaks = parse_ms_file(ms_path)
        
        if not peaks:
            print(f"⚠ Skipping {spec_name}: no peaks found")
            skipped += 1
            continue
        
        # Get precursor m/z
        precursor_mz = get_precursor_mz(row, args.precursor_mz_field)
        
        # Create entry
        entry = create_subformulae_entry(
            spec_name=spec_name,
            smiles=smiles,
            formula=formula,
            peaks=peaks,
            precursor_mz=precursor_mz,
            ion_type=args.ion_type
        )
        
        # Write JSON
        out_file = out_dir / f"{spec_name}.json"
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(entry, f, indent=4)
        
        count += 1
        if (count % 100) == 0:
            print(f"✓ Generated {count} subformulae JSON files...")
    
    print(f"\n✓ Done! Generated {count} JSON files in {out_dir}")
    if skipped > 0:
        print(f"⚠ Skipped {skipped} entries")


if __name__ == '__main__':
    main()
