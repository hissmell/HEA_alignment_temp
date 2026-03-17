#!/usr/bin/env python3
"""
UMA baseline performance on 25Cao dataset
Conda env: fairchem
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from ase.io import read
from ase import Atoms
from ase.build import molecule
from ase.optimize import BFGS
from tqdm import tqdm
import torch
import argparse
import pickle
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path("/DATA/user_scratch/pn50212/2024/12_AtomAttention")
DATA_DIR = BASE_DIR / "datasets/25Cao"
MODEL_DIR = BASE_DIR / "MLPs"
OUTPUT_DIR = BASE_DIR / "baseline_results"
OUTPUT_DIR.mkdir(exist_ok=True)

DATASET = "25Cao"
FAMILY = "UMA"

UMA_MODELS = {
    'uma-s-1p1': MODEL_DIR / 'uma-s-1p1.pt',
    'uma-m-1p1': MODEL_DIR / 'uma-m-1p1.pt',
}


# ── Taskname parser ──────────────────────────────────────────────────────────

def parse_taskname_to_path(taskname, adsorbate="O"):
    """Parse taskname to CONTCAR file path"""
    parts = taskname.split('&')

    # Extract order number
    order_str = parts[0]
    order_num = order_str.replace('87777order', '')

    # Extract site type
    site_str = parts[1].replace('fix', '')

    # Handle different suffix conventions
    if adsorbate == 'O':
        if site_str == 'topO':
            site_type = 'top'
        elif site_str == 'hollow':
            site_type = 'hcp'
        elif site_str == 'fcchollow':
            site_type = 'fcc'
        elif site_str == 'bridge':
            site_type = 'bridge'
        else:
            raise ValueError(f"Unknown site type: {site_str} for O adsorbate")
    else:  # OH
        site_str = site_str.replace('OH', '')
        if site_str == 'top':
            site_type = 'top'
        elif site_str == 'hollow':
            site_type = 'hcp'
        elif site_str == 'fcchollow':
            site_type = 'fcc'
        elif site_str == 'bridge':
            site_type = 'bridge'
        else:
            raise ValueError(f"Unknown site type: {site_str} for OH adsorbate")

    # Extract CONTCAR number (0-indexed to 1-indexed)
    contcar_idx = int(parts[2]) + 1

    # Construct path
    base_path = str(DATA_DIR / "sourcedata")
    file_path = f'{base_path}/{adsorbate}/{site_type}/order{order_num}/CONTCAR{contcar_idx}'

    return file_path


# ── Dataset ───────────────────────────────────────────────────────────────────

def load_25cao_dataset():
    structures, dft_energies = {}, {}
    stats = {'O': 0, 'OH': 0, 'failed': 0}
    failed_paths = []

    for adsorbate in ['O', 'OH']:
        json_path = DATA_DIR / f"{adsorbate} adsorption.json"
        with open(json_path) as f:
            data = json.load(f)

        for task_id, task_data in tqdm(data.items(), desc=f"Loading {adsorbate}"):
            ads_energy = task_data.get('ads_energy')
            if ads_energy is None:
                continue

            taskname = task_data.get('taskname', '')
            try:
                file_path = parse_taskname_to_path(taskname, adsorbate)
            except (ValueError, IndexError) as e:
                stats['failed'] += 1
                failed_paths.append(f"{taskname}: {e}")
                continue

            if not Path(file_path).exists():
                stats['failed'] += 1
                failed_paths.append(f"{taskname}: {file_path} not found")
                continue

            atoms = read(file_path)
            sid = f"{adsorbate}_{task_id}"
            structures[sid] = atoms
            dft_energies[sid] = ads_energy
            stats[adsorbate] += 1

    print(f"Loaded {len(structures)} structures (O: {stats['O']}, OH: {stats['OH']}, failed: {stats['failed']})")
    if failed_paths:
        print(f"  First 5 failures: {failed_paths[:5]}")
    return structures, dft_energies


# ── Metrics ───────────────────────────────────────────────────────────────────

def calculate_metrics(detailed_df):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from scipy import stats

    df = detailed_df.dropna(subset=['mlip_adsorption_energy'])
    if len(df) == 0:
        return {'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan,
                'Pearson_r': np.nan, 'N_samples': 0}

    y_true = df['dft_adsorption_energy'].values
    y_pred = df['mlip_adsorption_energy'].values

    pearson_r, _ = stats.pearsonr(y_true, y_pred)
    spearman_r, _ = stats.spearmanr(y_true, y_pred)

    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred),
        'Pearson_r': pearson_r,
        'Spearman_r': spearman_r,
        'N_samples': len(y_true),
        'Mean_DFT': np.mean(y_true),
        'Mean_Pred': np.mean(y_pred),
    }


# ── UMA Predictor ─────────────────────────────────────────────────────────────

class UMAPredictor:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.calculator = None
        self._gas_refs = None

    def load_model(self):
        from fairchem.core import FAIRChemCalculator
        from fairchem.core.units.mlip_unit import load_predict_unit

        model_path = UMA_MODELS.get(self.model_name)
        if model_path is None or not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        predictor = load_predict_unit(path=str(model_path), device=device)
        self.calculator = FAIRChemCalculator(predictor, task_name="omat")
        print(f"Loaded UMA model: {self.model_name} on {device}")

    def predict(self, atoms: Atoms) -> float:
        a = atoms.copy()
        a.set_constraint()
        a.calc = self.calculator
        return a.get_potential_energy()

    def calculate_gas_references(self):
        cache_file = OUTPUT_DIR / f"gas_refs_{self.model_name}.pkl"
        if cache_file.exists():
            print(f"  Loading cached gas refs from {cache_file.name}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

        print("  Calculating gas-phase references...")
        refs = {}
        for mol_name in ['H2O', 'H2']:
            mol = molecule(mol_name)
            mol.center(vacuum=10.0)
            mol.calc = self.calculator
            opt = BFGS(mol, trajectory=None, logfile=None)
            opt.run(fmax=0.01, steps=100)
            refs[mol_name] = mol.get_potential_energy()
            print(f"    E({mol_name}) = {refs[mol_name]:.4f} eV")

        refs['O'] = refs['H2O'] - refs['H2']
        refs['OH'] = refs['H2O'] - 0.5 * refs['H2']
        print(f"    E(O)  = {refs['O']:.4f} eV")
        print(f"    E(OH) = {refs['OH']:.4f} eV")

        with open(cache_file, 'wb') as f:
            pickle.dump(refs, f)
        return refs

    def predict_adsorption_energy(self, atoms: Atoms, adsorbate: str) -> dict:
        """Return dict with all intermediate energies."""
        if self._gas_refs is None:
            self._gas_refs = self.calculate_gas_references()

        e_adslab = self.predict(atoms)

        slab = atoms.copy()
        slab.set_constraint()
        if adsorbate == 'O':
            del slab[-1]
        elif adsorbate == 'OH':
            del slab[-2:]
        e_slab = self.predict(slab)

        gas_ref = self._gas_refs[adsorbate]
        e_ads = e_adslab - e_slab - gas_ref

        return {
            'mlip_adsorption_energy': e_ads,
            'mlip_adslab_energy': e_adslab,
            'mlip_slab_energy': e_slab,
            'mlip_gas_ref_energy': gas_ref,
        }

    def cleanup(self):
        del self.calculator
        torch.cuda.empty_cache()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=list(UMA_MODELS.keys()))
    args = parser.parse_args()

    print("Loading 25Cao dataset...")
    structures, dft_energies = load_25cao_dataset()

    for model_name in args.models:
        print(f"\n{'='*60}\nEvaluating {model_name}\n{'='*60}")
        predictor = UMAPredictor(model_name)

        try:
            predictor.load_model()

            records = []
            for sid, atoms in tqdm(structures.items(), desc=model_name):
                adsorbate = sid.split('_')[0]
                try:
                    result = predictor.predict_adsorption_energy(atoms, adsorbate)
                    dft_e = dft_energies[sid]
                    error = result['mlip_adsorption_energy'] - dft_e
                    records.append({
                        'adslab_id': sid,
                        'adsorbate': adsorbate,
                        'dft_adsorption_energy': dft_e,
                        **result,
                        'error': error,
                        'abs_error': abs(error),
                    })
                except Exception as e:
                    print(f"  Failed {sid}: {e}")
                    records.append({
                        'adslab_id': sid,
                        'adsorbate': adsorbate,
                        'dft_adsorption_energy': dft_energies[sid],
                        'mlip_adsorption_energy': np.nan,
                        'mlip_adslab_energy': np.nan,
                        'mlip_slab_energy': np.nan,
                        'mlip_gas_ref_energy': np.nan,
                        'error': np.nan,
                        'abs_error': np.nan,
                    })

            detailed_df = pd.DataFrame(records)
            metrics = calculate_metrics(detailed_df)

            print(f"  MAE={metrics['MAE']:.4f} eV  RMSE={metrics['RMSE']:.4f} eV  "
                  f"R²={metrics['R2']:.4f}  N={metrics['N_samples']}")

            # Save to DATASET/FAMILY/model_name/ subfolder
            model_dir = OUTPUT_DIR / DATASET / FAMILY / model_name
            model_dir.mkdir(parents=True, exist_ok=True)

            # summary.xlsx
            summary_df = pd.DataFrame([{'Model': model_name, **metrics}])
            summary_path = model_dir / 'summary.xlsx'
            summary_df.to_excel(summary_path, index=False, engine='openpyxl')

            # detailed.xlsx
            detailed_path = model_dir / 'detailed.xlsx'
            detailed_df.to_excel(detailed_path, index=False, engine='openpyxl')

            print(f"  Saved: {summary_path}")
            print(f"  Saved: {detailed_path}")

        except Exception as e:
            print(f"Error: {e}")
        finally:
            predictor.cleanup()


if __name__ == "__main__":
    main()
