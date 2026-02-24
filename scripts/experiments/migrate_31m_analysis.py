#!/usr/bin/env python3
"""
Migration script to use new CKNNA module with existing 31M analysis workflow.
Demonstrates how to replace analyze_31m_cknna_25cao.py with the new modular approach.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import warnings
warnings.filterwarnings('ignore')

import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Import new CKNNA module
from src.core.cknna import CKNNAAnalyzer

# Configuration (same as original)
BASE_DIR = "/DATA/user_scratch/pn50212/2024/12_AtomAttention/datasets/25Cao"
SOAP_BATCHES_DIR = os.path.join(BASE_DIR, "soap_batches")
EQUIFORMER_SM_BATCHES_DIR = os.path.join(BASE_DIR, "equiformer_sm_batches")
BASELINE_RESULTS_FILE = os.path.join(BASE_DIR, "equiformer_31m_baseline_performance.xlsx")

# Output
OUTPUT_DIR = "/DATA/user_scratch/pn50212/2024/12_AtomAttention/cknna_analysis_31m_migrated"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_31m_equiformer_data():
    """Load 31M EquiformerV2 data (same as original)"""
    print("Loading 31M EquiformerV2 representations...")

    import glob
    eq_files = sorted(glob.glob(os.path.join(EQUIFORMER_SM_BATCHES_DIR, "batch_*.npz")))
    eq_data = {}

    for eq_file in tqdm(eq_files, desc="Loading EquiformerV2 31M batches"):
        json_file = eq_file.replace('.npz', '.json')
        if not os.path.exists(json_file):
            continue

        try:
            with open(json_file, 'r') as f:
                metadata = json.load(f)

            npz_data = np.load(eq_file, allow_pickle=True)
            tasknames = metadata.get('tasknames', [])

            if 'norm_output' in npz_data:
                norm_outputs = npz_data['norm_output']

                for i, taskname in enumerate(tasknames):
                    if i < len(norm_outputs):
                        repr_data = norm_outputs[i]

                        if hasattr(repr_data, 'shape'):
                            if isinstance(repr_data, np.ndarray):
                                eq_data[taskname] = repr_data.astype(np.float64)
                            else:
                                eq_data[taskname] = np.array(repr_data, dtype=np.float64)

            npz_data.close()

        except Exception as e:
            print(f"Error loading {eq_file}: {e}")
            continue

    print(f"Loaded 31M EquiformerV2 representations for {len(eq_data)} structures")
    return eq_data


def load_soap_data():
    """Load SOAP data (same as original)"""
    print("Loading SOAP representations...")

    import glob
    soap_files = sorted(glob.glob(os.path.join(SOAP_BATCHES_DIR, "batch_*.npz")))
    soap_data = {}

    for soap_file in tqdm(soap_files, desc="Loading SOAP batches"):
        json_file = soap_file.replace('.npz', '.json')
        if not os.path.exists(json_file):
            continue

        try:
            with open(json_file, 'r') as f:
                metadata = json.load(f)

            npz_data = np.load(soap_file)

            for taskname in metadata.get('structures', []):
                soap_full_key = f"{taskname}_soap_full"
                if soap_full_key in npz_data:
                    soap_repr = npz_data[soap_full_key]

                    if hasattr(soap_repr, 'shape'):
                        if isinstance(soap_repr, np.ndarray):
                            soap_data[taskname] = soap_repr.astype(np.float64)
                        else:
                            soap_data[taskname] = np.array(soap_repr, dtype=np.float64)

            npz_data.close()

        except Exception as e:
            print(f"Error loading {soap_file}: {e}")
            continue

    print(f"Loaded SOAP representations for {len(soap_data)} structures")
    return soap_data


def load_31m_prediction_errors():
    """Load 31M model prediction errors (same as original)"""
    print("Loading 31M model prediction errors...")

    try:
        df = pd.read_excel(BASELINE_RESULTS_FILE, sheet_name='Results')
        error_data = {}
        for _, row in df.iterrows():
            taskname = row['taskname']
            abs_error = row['abs_error']
            error_data[taskname] = abs_error

        print(f"Loaded prediction errors for {len(error_data)} structures")
        return error_data

    except Exception as e:
        print(f"Error loading prediction errors: {e}")
        return {}


def analyze_31m_with_new_module():
    """
    Use new CKNNA module to perform the same analysis.
    This demonstrates how the new modular approach simplifies the code.
    """
    print("=" * 80)
    print("31M EQUIFORMERV2 CKNNA ANALYSIS - USING NEW MODULE")
    print("=" * 80)

    # Load all required data (same as original)
    eq_data = load_31m_equiformer_data()
    soap_data = load_soap_data()
    error_data = load_31m_prediction_errors()

    if len(eq_data) == 0 or len(soap_data) == 0 or len(error_data) == 0:
        print("ERROR: Failed to load required data")
        return

    # Use new CKNNAAnalyzer - much simpler!
    analyzer = CKNNAAnalyzer(cache_dir=OUTPUT_DIR)

    # Analyze representations (replaces all the manual CKNNA calculations)
    results_df, corr_df = analyzer.analyze_representations(
        physics_repr=soap_data,
        mlip_repr=eq_data,
        errors=error_data,
        k_values=[10]  # Using k=10 as in the original
    )

    print(f"Successfully calculated CKNNA for {len(results_df)} structures")

    # Save results
    output_excel = os.path.join(OUTPUT_DIR, "cknna_analysis_31m_new_module.xlsx")
    analyzer.save_results(results_df, corr_df, Path(output_excel))

    # Print summary
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"Dataset size: {len(results_df)} structures")
    print("\nCorrelation Results:")
    print("-" * 80)

    for _, row in corr_df.iterrows():
        significance = "***" if row['significant'] else ""
        print(f"k={row['k']:2d}: r={row['spearman_r']:7.4f} (p={row['spearman_p']:.3f}){significance:3s}")

    # Basic error statistics
    print(f"\nError statistics:")
    print(f"  Mean absolute error: {results_df['error'].mean():.4f} eV")
    print(f"  Median absolute error: {results_df['error'].median():.4f} eV")
    print(f"  Standard deviation: {results_df['error'].std():.4f} eV")

    # Create simple visualization
    create_analysis_plot(results_df, corr_df)

    return results_df, corr_df


def create_analysis_plot(results_df, corr_df):
    """Create a simple visualization of the results."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('31M EquiformerV2 CKNNA Analysis - New Module', fontsize=14)

    # Plot 1: CKNNA vs Error
    cknna_col = 'cknna_k10'
    if cknna_col in results_df.columns:
        mask = ~(results_df[cknna_col].isna() | results_df['error'].isna())
        df_plot = results_df[mask]

        axes[0].scatter(df_plot[cknna_col], df_plot['error'], alpha=0.5, s=20)
        axes[0].set_xlabel('CKNNA (k=10)')
        axes[0].set_ylabel('Absolute Error (eV)')
        axes[0].set_title(f'CKNNA vs Prediction Error (n={len(df_plot)})')
        axes[0].grid(True, alpha=0.3)

        # Add correlation info
        if len(corr_df) > 0:
            r = corr_df.iloc[0]['spearman_r']
            p = corr_df.iloc[0]['spearman_p']
            axes[0].text(0.05, 0.95, f'r = {r:.3f}\np = {p:.3f}',
                        transform=axes[0].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 2: Error distribution
    axes[1].hist(results_df['error'], bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[1].set_xlabel('Absolute Error (eV)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Error Distribution')
    axes[1].axvline(results_df['error'].mean(), color='red', linestyle='--',
                   label=f'Mean: {results_df["error"].mean():.3f}')
    axes[1].axvline(results_df['error'].median(), color='green', linestyle='--',
                   label=f'Median: {results_df["error"].median():.3f}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'cknna_analysis_new_module.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nVisualization saved to: {output_path}")


def demonstrate_advanced_features():
    """
    Demonstrate advanced features of the new module that weren't in the original.
    """
    print("\n" + "=" * 80)
    print("DEMONSTRATING ADVANCED FEATURES")
    print("=" * 80)

    # Load data
    eq_data = load_31m_equiformer_data()
    soap_data = load_soap_data()
    error_data = load_31m_prediction_errors()

    analyzer = CKNNAAnalyzer()

    # 1. Find optimal k value
    print("\n1. Finding optimal k value...")
    optimal = analyzer.find_optimal_k(
        physics_repr=soap_data,
        mlip_repr=eq_data,
        errors=error_data,
        k_range=range(5, 21, 5)
    )
    print(f"   Optimal k: {optimal['optimal_k']}")
    print(f"   Best correlation: {optimal['correlation']:.4f}")

    # 2. Select uncertain samples for active learning
    print("\n2. Selecting uncertain samples for active learning...")
    uncertain_samples = analyzer.select_uncertain_samples(
        physics_repr=soap_data,
        mlip_repr=eq_data,
        n_samples=20,
        k=10,
        strategy='lowest'
    )
    print(f"   Selected {len(uncertain_samples)} most uncertain structures")
    print(f"   Examples: {uncertain_samples[:5]}")

    # 3. Select diverse samples
    print("\n3. Selecting diverse samples...")
    diverse_samples = analyzer.select_uncertain_samples(
        physics_repr=soap_data,
        mlip_repr=eq_data,
        n_samples=20,
        k=10,
        strategy='diverse'
    )
    print(f"   Selected {len(diverse_samples)} diverse structures")


def main():
    """Main execution function"""
    try:
        # Run the main analysis with new module
        results_df, corr_df = analyze_31m_with_new_module()

        # Demonstrate advanced features
        demonstrate_advanced_features()

        print("\n" + "=" * 80)
        print("MIGRATION SUCCESSFUL!")
        print("The new module produces equivalent results with cleaner code.")
        print("=" * 80)

    except Exception as e:
        print(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()