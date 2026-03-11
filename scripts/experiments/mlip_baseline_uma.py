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

UMA_MODELS = {
    'uma-s-1p1': MODEL_DIR / 'uma-s-1p1.pt',
    'uma-m-1p1': MODEL_DIR / 'uma-m-1p1.pt',
}


# ── Dataset ───────────────────────────────────────────────────────────────────

def load_25cao_dataset():
    structures, dft_energies = {}, {}

    for adsorbate in ['O', 'OH']:
        json_path = DATA_DIR / f"{adsorbate} adsorption.json"
        with open(json_path) as f:
            data = json.load(f)
        sourcedata_dir = DATA_DIR / "sourcedata" / adsorbate

        for task_id, task_data in tqdm(data.items(), desc=f"Loading {adsorbate}"):
            ads_energy = task_data.get('ads_energy')
            if ads_energy is None:
                continue

            taskname = task_data.get('taskname', '')
            order_match = taskname.split('order')
            if len(order_match) <= 1:
                continue

            order_num = order_match[1].split('&')[0]
            site = task_data.get('site', 'unknown')
            site_map = {'hollow': 'fcc', 'bridge': 'bridge', 'top': 'top',
                        'fcc': 'fcc', 'hcp': 'hcp'}
            site_dir = site_map.get(site, site)

            contcar_dir = sourcedata_dir / site_dir / f"order{order_num}"
            if not contcar_dir.exists():
                continue
            contcar_files = sorted(contcar_dir.glob("CONTCAR*"))
            if not contcar_files:
                continue

            atoms = read(str(contcar_files[0]))
            sid = f"{adsorbate}_{task_id}"
            structures[sid] = atoms
            dft_energies[sid] = ads_energy

    print(f"Loaded {len(structures)} structures")
    return structures, dft_energies


# ── Metrics ───────────────────────────────────────────────────────────────────

def calculate_metrics(dft_energies, predictions):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from scipy import stats

    common = [sid for sid in dft_energies if sid in predictions
              and not np.isnan(predictions[sid])]
    if not common:
        return {'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan,
                'Pearson_r': np.nan, 'N_samples': 0}

    y_true = np.array([dft_energies[sid] for sid in common])
    y_pred = np.array([predictions[sid] for sid in common])

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

    def predict_adsorption_energy(self, atoms: Atoms, adsorbate: str) -> float:
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

        return e_adslab - e_slab - self._gas_refs[adsorbate]

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

    all_metrics = {}
    all_predictions = {}

    for model_name in args.models:
        print(f"\n{'='*60}\nEvaluating {model_name}\n{'='*60}")
        predictor = UMAPredictor(model_name)

        try:
            predictor.load_model()
            predictions = {}
            for sid, atoms in tqdm(structures.items(), desc=model_name):
                adsorbate = sid.split('_')[0]
                try:
                    predictions[sid] = predictor.predict_adsorption_energy(atoms, adsorbate)
                except Exception as e:
                    print(f"  Failed {sid}: {e}")
                    predictions[sid] = np.nan

            metrics = calculate_metrics(dft_energies, predictions)
            all_metrics[model_name] = metrics
            all_predictions[model_name] = predictions

            print(f"  MAE={metrics['MAE']:.4f} eV  RMSE={metrics['RMSE']:.4f} eV  "
                  f"R²={metrics['R2']:.4f}  N={metrics['N_samples']}")
        except Exception as e:
            print(f"Error: {e}")
            all_metrics[model_name] = {'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan,
                                       'N_samples': 0, 'Error': str(e)}
        finally:
            predictor.cleanup()

    # Save results
    results_df = pd.DataFrame(all_metrics).T.reset_index().rename(columns={'index': 'Model'})
    excel_path = OUTPUT_DIR / 'baseline_uma.xlsx'
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        results_df.to_excel(writer, sheet_name='Summary', index=False)
        for model_name, preds in all_predictions.items():
            pred_df = pd.DataFrame({
                'Structure_ID': list(preds.keys()),
                'DFT_Energy': [dft_energies.get(sid, np.nan) for sid in preds],
                'Predicted_Energy': list(preds.values()),
            })
            pred_df['Error'] = pred_df['Predicted_Energy'] - pred_df['DFT_Energy']
            pred_df.to_excel(writer, sheet_name=model_name[:31], index=False)

    print(f"\nSaved: {excel_path}")
    print(results_df[['Model', 'MAE', 'RMSE', 'R2', 'N_samples']].to_string(index=False))


if __name__ == "__main__":
    main()
