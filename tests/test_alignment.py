"""
Test suite for alignment module.
Validates different alignment metrics and rcut consolidation.
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.alignment import (
    AlignmentAnalyzer,
    AlignmentConfig,
    SOAPAlignmentAnalyzer,
    compute_cknna_alignment,
    compute_dcor_alignment
)


def test_alignment_metrics():
    """Test different alignment metrics."""
    print("Testing alignment metrics...")

    np.random.seed(42)
    n_samples = 30
    n_features = 50

    # Create test data
    X = np.random.randn(n_samples, n_features)
    Y = np.random.randn(n_samples, n_features)

    analyzer = AlignmentAnalyzer()

    # Test CKNNA
    cknna_score = analyzer.compute_alignment(X, Y, metric='cknna')
    assert -1 <= cknna_score <= 1
    print(f"✓ CKNNA: {cknna_score:.4f}")

    # Test dCor
    dcor_score = analyzer.compute_alignment(X, Y, metric='dcor')
    assert 0 <= dcor_score <= 1
    print(f"✓ dCor: {dcor_score:.4f}")

    # Test cosine similarity
    cosine_score = analyzer.compute_alignment(X, Y, metric='cosine')
    assert -1 <= cosine_score <= 1
    print(f"✓ Cosine: {cosine_score:.4f}")

    # Test Procrustes
    procrustes_score = analyzer.compute_alignment(X, Y, metric='procrustes')
    assert 0 <= procrustes_score <= 1
    print(f"✓ Procrustes: {procrustes_score:.4f}")


def test_dimension_matching():
    """Test automatic dimension matching."""
    print("\nTesting dimension matching...")

    np.random.seed(42)

    # Different dimensions
    X = np.random.randn(20, 100)
    Y = np.random.randn(20, 50)

    config = AlignmentConfig(pca_dim=30)
    analyzer = AlignmentAnalyzer(config)

    # Should handle dimension mismatch
    score = analyzer.compute_alignment(X, Y)
    assert not np.isnan(score)
    print(f"✓ Handled dimension mismatch: {score:.4f}")


def test_alignment_with_errors():
    """Test alignment analysis with error correlation."""
    print("\nTesting alignment with error correlation...")

    np.random.seed(42)
    n_structures = 50

    # Create mock data
    repr1_dict = {}
    repr2_dict = {}
    errors = {}

    for i in range(n_structures):
        taskname = f"struct_{i}"
        n_atoms = np.random.randint(15, 40)

        # Create correlated representations for first half
        if i < 25:
            base = np.random.randn(n_atoms, 50)
            repr1_dict[taskname] = base
            repr2_dict[taskname] = base + np.random.randn(n_atoms, 50) * 0.3
            errors[taskname] = np.random.rand() * 0.2
        else:
            repr1_dict[taskname] = np.random.randn(n_atoms, 50)
            repr2_dict[taskname] = np.random.randn(n_atoms, 50)
            errors[taskname] = np.random.rand() * 0.8

    analyzer = AlignmentAnalyzer()

    # Analyze with multiple metrics
    results_df, corr_df = analyzer.analyze_with_errors(
        repr1_dict, repr2_dict, errors,
        metrics=['cknna', 'dcor', 'cosine'],
        k_values=[5, 10]
    )

    assert len(results_df) == n_structures
    assert 'cknna_k5' in results_df.columns
    assert 'cknna_k10' in results_df.columns
    assert 'dcor' in results_df.columns
    assert 'cosine' in results_df.columns

    print(f"✓ Analyzed {n_structures} structures")
    print(f"✓ Found {len(corr_df)} correlation results")

    # Check if we found significant correlations
    significant = corr_df[corr_df['significant']]
    if len(significant) > 0:
        best = significant.loc[significant['spearman_r'].abs().idxmax()]
        print(f"✓ Best metric: {best['metric']} (r={best['spearman_r']:.3f})")


def test_soap_rcut_analysis():
    """Test SOAP rcut analysis functionality."""
    print("\nTesting SOAP rcut analysis...")

    np.random.seed(42)

    # Mock SOAP loader
    # Fix: Use consistent n_atoms for each structure
    n_atoms_per_struct = {}
    for i in range(20):
        n_atoms_per_struct[f"struct_{i}"] = np.random.randint(20, 40)

    def soap_loader(rcut):
        # Simulate different SOAP representations for different rcuts
        soap_dict = {}
        for i in range(20):
            taskname = f"struct_{i}"
            n_atoms = n_atoms_per_struct[taskname]
            # Add rcut-dependent variation
            soap_dict[taskname] = np.random.randn(n_atoms, 100) * (1 + rcut/10)
        return soap_dict

    # Mock MLIP representations and errors
    mlip_repr = {}
    errors = {}
    for i in range(20):
        taskname = f"struct_{i}"
        n_atoms = n_atoms_per_struct[taskname]  # Use same n_atoms
        mlip_repr[taskname] = np.random.randn(n_atoms, 100)
        errors[taskname] = np.random.rand()

    # Test rcut range analysis
    soap_analyzer = SOAPAlignmentAnalyzer(rcut=8.0)
    rcut_df = soap_analyzer.analyze_rcut_range(
        soap_loader,
        mlip_repr,
        errors,
        rcut_range=[4.0, 6.0, 8.0, 10.0]
    )

    assert len(rcut_df) == 4
    assert 'rcut' in rcut_df.columns
    assert 'correlation' in rcut_df.columns

    print(f"✓ Analyzed {len(rcut_df)} rcut values")

    # Find optimal rcut
    optimal_idx = rcut_df['correlation'].abs().idxmax()
    optimal_rcut = rcut_df.loc[optimal_idx, 'rcut']
    print(f"✓ Optimal rcut: {optimal_rcut}")


def test_representation_comparison():
    """Test comparison of multiple representations."""
    print("\nTesting representation comparison...")

    np.random.seed(42)

    # Create multiple representations
    representations = {
        'SOAP': {},
        'MLIP': {},
        'Random': {}
    }

    errors = {}

    for i in range(30):
        taskname = f"struct_{i}"
        n_atoms = np.random.randint(15, 35)

        base = np.random.randn(n_atoms, 50)
        representations['SOAP'][taskname] = base
        representations['MLIP'][taskname] = base + np.random.randn(n_atoms, 50) * 0.5
        representations['Random'][taskname] = np.random.randn(n_atoms, 50)

        errors[taskname] = np.random.rand()

    analyzer = AlignmentAnalyzer()
    comparison_df = analyzer.compare_representations(representations, errors)

    # Should have pairwise comparisons
    n_repr = len(representations)
    expected_comparisons = n_repr * (n_repr - 1) // 2
    assert len(comparison_df) == expected_comparisons

    print(f"✓ Compared {n_repr} representations")
    print(f"✓ Generated {len(comparison_df)} pairwise comparisons")

    # Check best alignment
    if len(comparison_df) > 0:
        best_idx = comparison_df['correlation'].abs().idxmax()
        best = comparison_df.loc[best_idx]
        print(f"✓ Best alignment: {best['repr1']} vs {best['repr2']} (r={best['correlation']:.3f})")


def test_backward_compatibility():
    """Test backward compatibility functions."""
    print("\nTesting backward compatibility...")

    np.random.seed(42)
    X = np.random.randn(25, 60)
    Y = np.random.randn(25, 60)

    # Test standalone functions
    cknna = compute_cknna_alignment(X, Y, k=10)
    dcor = compute_dcor_alignment(X, Y)

    assert -1 <= cknna <= 1
    assert 0 <= dcor <= 1

    print(f"✓ compute_cknna_alignment: {cknna:.4f}")
    print(f"✓ compute_dcor_alignment: {dcor:.4f}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("ALIGNMENT MODULE TEST SUITE")
    print("=" * 60)

    test_functions = [
        test_alignment_metrics,
        test_dimension_matching,
        test_alignment_with_errors,
        test_soap_rcut_analysis,
        test_representation_comparison,
        test_backward_compatibility
    ]

    for test_func in test_functions:
        try:
            test_func()
        except Exception as e:
            print(f"✗ {test_func.__name__} failed: {e}")
            raise

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()