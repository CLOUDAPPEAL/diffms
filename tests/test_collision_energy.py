
import sys
import os
import torch
import numpy as np
from unittest.mock import MagicMock

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

# Mock torch_geometric if not present
try:
    import torch_geometric
except ImportError:
    print("Mocking torch_geometric...")
    mock_tg = MagicMock()
    mock_tg.data = MagicMock()
    mock_tg.data.Data = MagicMock
    mock_tg.data.Batch = MagicMock
    sys.modules["torch_geometric"] = mock_tg
    sys.modules["torch_geometric.data"] = mock_tg.data

from mist.data import featurizers, data
from mist.models import modules

def test_collision_energy_integration():
    print("Testing Collision Energy Integration...")

    # 1. Test Spectra.get_collision_energy logic (Unit Test)
    print("1. Testing Spectra.get_collision_energy...")
    spec = data.Spectra(spectrum_file="dummy.msp")
    spec.meta = {"Collision_energy": "35.0 eV", "parentmass": "100"}
    spec._is_loaded = True # Mock loaded state
    spec.parentmass = 100.0
    
    # Manually trigger the extraction logic (usually done in _load_spectra, but we can simulate or test a helper if we extracted one.
    # Since we modified _load_spectra, we can't easily test it without a file. 
    # But we added `get_collision_energy` which checks `_load_spectra`.
    # Let's mock `_load_spectra` to just set the value if not set.
    
    # Actually, we modified `_load_spectra` to set `self.collision_energy`. 
    # Let's just create a Spectra object and manually set `self.collision_energy` to verify getter works.
    spec.collision_energy = 35.0
    assert spec.get_collision_energy() == 35.0
    print("Spectra getter works.")

    # 2. Test PeakFormula.collate_fn (Integration Test)
    print("2. Testing PeakFormula.collate_fn...")
    
    # Create dummy featurized outputs (as if _featurize returned them)
    # We need to match what _featurize returns
    dummy_feat_1 = {
        "peak_type": np.array([0, 3]), # Type 3 is CLS
        "form_vec": [[0]*10, [0]*10], 
        "ion_vec": [0, 1],
        "frag_intens": np.array([1.0, 0.5], dtype=np.float32),
        "name": "spec1",
        "magma_fps": np.zeros(10),
        "magma_aux_loss": 0,
        "instrument": 0,
        "precursor_type": 0,
        "collision_energy": 35.0
    }
    
    dummy_feat_2 = {
        "peak_type": np.array([3]), # Type 3 is CLS
        "form_vec": [[0]*10],
        "ion_vec": [0],
        "frag_intens": np.array([1.0], dtype=np.float32),
        "name": "spec2",
        "magma_fps": np.zeros(10),
        "magma_aux_loss": 0,
        "instrument": 0,
        "precursor_type": 0,
        "collision_energy": 20.0
    }
    
    batch_list = [dummy_feat_1, dummy_feat_2]
    
    # collate_fn is likely a static method or instance method used by DataLoader
    # Check PeakFormula.collate_fn
    # It seems to be an instance method in common usage patterns or just a function
    # Let's assume instance method for now based on 'self' usage in some collates, but PeakFormula likely uses `collate_fn` as static or standalone?
    # Actually, usually it's `PeakFormula.collate_fn(batch_list)` (static) or `pf.collate_fn`.
    # Let's check imports. `featurizers.PeakFormula` inherits from something?
    # Let's try instantiating it.
    pf = featurizers.PeakFormula(subform_folder="dummy_path")
    
    # We need to mock utils.pad_sequence or ensure lists are tensors
    # featurizers.py implementation of collate_fn handles lists conversion to tensors
    
    # However, `form_vec` needs to be numpy array of correct shape potentially?
    # In `_featurize`, `forms_vec` is list of lists, then converted to np.array.
    dummy_feat_1["form_vec"] = np.array(dummy_feat_1["form_vec"])
    dummy_feat_2["form_vec"] = np.array(dummy_feat_2["form_vec"])
    
    # colalte_fn
    batch = pf.collate_fn(batch_list)
    
    # Check if collision_energy is in batch
    if "collision_energy" not in batch:
        print("Error: collision_energy not in collated batch!")
        exit(1)
        
    ce_batch = batch["collision_energy"]
    print("Collated Collision Energy Shape:", ce_batch.shape)
    assert ce_batch.shape == (2, 1)
    assert ce_batch[0].item() == 35.0
    assert ce_batch[1].item() == 20.0
    print("Collation works.")
    
    # 3. Test FormulaTransformer Forward (Model Test)
    print("3. Testing FormulaTransformer Forward...")
    
    # We need to ensure batch has all keys model expects
    # Model expects: num_peaks, types, instruments, ion_vec, form_vec, intens, collision_energy
    
    # The `batch` from collate_fn should have these.
    # keys map: 
    # "peak_type" -> "types"
    # "frag_intens" -> "intens"
    # "instrument" -> "instruments"
    # "ion_vec" -> "ion_vec"? No, wait.
    
    # Let's check collate_fn output keys in featurizers.py:
    # return_dict = { "num_peaks": ..., "types": ..., "form_vec": ..., "intens": ..., "ion_vec": ..., "instruments": ..., "precursor_types": ..., "collision_energy": ... }
    
    # So `batch` should be ready for model.
    # Except `ion_vec` key in featurizers seems to be `ion_vec`.
    # In modules.py:
    # adducts = batch["ion_vec"]
    
    # So keys match.
    
    model = modules.FormulaTransformer(
        hidden_size=64,
        peak_attn_layers=1,
        output_size=128,
        form_embedder="float" # simple
    )
    # We need to mock form_embedders.get_embedder because default might differ
    # But if we use "float", it might work natively if imported correctly.
    
    # FormulaTransformer init:
    # self.form_embedder_mod = form_embedders.get_embedder(self.form_embedder)
    # self.formula_dim = self.form_embedder_mod.full_dim
    # We need to make sure form_vec in batch matches formula_dim.
    
    # Check real formula dim.
    real_formula_dim = model.formula_dim
    print("Model Formula Dim:", real_formula_dim)
    
    # Resize our dummy form_vec to match
    batch["form_vec"] = torch.randn(2, batch["form_vec"].shape[1], real_formula_dim) 
    # Note: batch["form_vec"] from collate might be (B, Np, D).
    # collate_fn: padded_form_vec = pad_sequence(...) -> (B, Np, D)
    
    # Ensure other tensors are on CPU float
    batch["intens"] = batch["intens"].float()
    batch["collision_energy"] = batch["collision_energy"].float()
    
    try:
        output, aux = model(batch, return_aux=True)
        print("Forward pass successful!")
        print("Output shape:", output.shape)
        assert output.shape == (2, 64)
    except Exception as e:
        print(f"Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    test_collision_energy_integration()
