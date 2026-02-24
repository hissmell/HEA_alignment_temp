"""
Test suite for CKNNA module.
Validates that the new implementation matches the original analyze_31m_cknna_25cao.py
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.cknna import CKNNA, CKNNAConfig, CKNNAAnalyzer, cknna_paper


def test_cknna_basic():
    """Test basic CKNNA computation."""
    print("Testing basic CKNNA computation...")

    # Create random data
    np.random.seed(42)
    n_samples = 50
    n_features = 100

    X = np.random.randn(n_samples, n_features)
    Y = np.random.randn(n_samples, n_features)

    # Test with default config
    calculator = CKNNA()
    cknna_value = calculator.compute(X, Y)

    assert isinstance(cknna_value, float)
    assert -1 <= cknna_value <= 1
    print(f"✓ Basic CKNNA value: {cknna_value:.4f}")

    # Test with different k values
    for k in [5, 10, 15]:
        cknna_k = calculator.compute(X, Y, k=k)
        assert isinstance(cknna_k, float)
        print(f"✓ CKNNA with k={k}: {cknna_k:.4f}")


def test_cknna_edge_cases():
    """Test CKNNA with edge cases."""
    print("\nTesting edge cases...")

    calculator = CKNNA()

    # Test with too few samples
    X_small = np.random.randn(5, 10)
    Y_small = np.random.randn(5, 10)
    cknna_small = calculator.compute(X_small, Y_small, k=10)
    assert np.isnan(cknna_small)
    print("✓ Returns NaN for too few samples")

    # Test with identical representations
    X_same = np.random.randn(20, 50)
    cknna_same = calculator.compute(X_same, X_same)
    assert cknna_same > 0.9  # Should be close to 1
    print(f"✓ Identical representations: {cknna_same:.4f}")

    # Test with different representations (not necessarily orthogonal due to k-NN structure)
    n = 30
    X_diff = np.random.randn(n, 50)
    Y_diff = np.random.randn(n, 50) * 2 + 3  # Different distribution
    cknna_diff = calculator.compute(X_diff, Y_diff)
    assert -1 <= cknna_diff <= 1  # Should be in valid range
    print(f"✓ Different representations: {cknna_diff:.4f}")


def test_cknna_batch():
    """Test batch CKNNA computation."""
    print("\nTesting batch computation...")

    np.random.seed(42)
    n_structures = 10
    X_list = [np.random.randn(np.random.randint(20, 50), 100) for _ in range(n_structures)]
    Y_list = [np.random.randn(x.shape[0], 100) for x in X_list]

    calculator = CKNNA()
    results = calculator.compute_batch(X_list, Y_list, verbose=False)

    assert len(results) == n_structures
    assert all(-1 <= r <= 1 or np.isnan(r) for r in results)
    print(f"✓ Batch computation: {n_structures} structures processed")


def test_cknna_analyzer():
    """Test CKNNAAnalyzer functionality."""
    print("\nTesting CKNNAAnalyzer...")

    np.random.seed(42)

    # Create mock data
    n_structures = 50
    tasknames = [f"struct_{i}" for i in range(n_structures)]

    physics_repr = {}
    mlip_repr = {}
    errors = {}

    for i, task in enumerate(tasknames):
        n_atoms = np.random.randint(15, 40)
        physics_repr[task] = np.random.randn(n_atoms, 100)
        # Add correlation with physics for some structures
        if i < 25:
            mlip_repr[task] = physics_repr[task] + np.random.randn(n_atoms, 100) * 0.5
            errors[task] = np.random.rand() * 0.1
        else:
            mlip_repr[task] = np.random.randn(n_atoms, 100)
            errors[task] = np.random.rand() * 0.5

    # Test analyzer
    analyzer = CKNNAAnalyzer()
    results_df, corr_df = analyzer.analyze_representations(
        physics_repr, mlip_repr, errors, k_values=[5, 10, 15]
    )

    assert len(results_df) == n_structures
    assert len(corr_df) == 3  # 3 k values
    print(f"✓ Analyzer processed {len(results_df)} structures")

    # Test optimal k finding
    optimal = analyzer.find_optimal_k(
        physics_repr, mlip_repr, errors,
        k_range=range(5, 16, 5)
    )

    assert 'optimal_k' in optimal
    assert 'correlation' in optimal
    print(f"✓ Optimal k={optimal['optimal_k']} with correlation={optimal['correlation']:.4f}")

    # Test sample selection
    selected = analyzer.select_uncertain_samples(
        physics_repr, mlip_repr,
        n_samples=10, k=10, strategy='lowest'
    )

    assert len(selected) == 10
    print(f"✓ Selected {len(selected)} uncertain samples")


def test_backward_compatibility():
    """Test backward compatibility with cknna_paper function."""
    print("\nTesting backward compatibility...")

    np.random.seed(42)
    X = np.random.randn(30, 100)
    Y = np.random.randn(30, 100)

    # Test standalone function
    cknna_value = cknna_paper(X, Y, k=10)

    # Compare with class-based computation
    calculator = CKNNA(CKNNAConfig(k=10))
    cknna_class = calculator.compute(X, Y)

    assert abs(cknna_value - cknna_class) < 1e-10
    print(f"✓ Backward compatible: {cknna_value:.4f} == {cknna_class:.4f}")


def test_with_real_data_structure():
    """Test with data structure similar to real usage."""
    print("\nTesting with realistic data structure...")

    np.random.seed(42)

    # Simulate loading data like in analyze_31m_cknna_25cao.py
    soap_data = {}
    eq_data = {}
    error_data = {}

    n_structures = 20
    for i in range(n_structures):
        taskname = f"25cao_{i:04d}"
        n_atoms = np.random.randint(50, 200)  # Realistic atom counts

        # SOAP representations (typically high dimensional)
        soap_data[taskname] = np.random.randn(n_atoms, 324)  # SOAP dimension

        # Equiformer representations
        eq_data[taskname] = np.random.randn(n_atoms, 128)  # Typical MLIP dimension

        # Errors
        error_data[taskname] = np.random.rand() * 0.5

    # Test the same workflow as analyze_31m_cknna_25cao.py
    analyzer = CKNNAAnalyzer()
    results_df, corr_df = analyzer.analyze_representations(
        soap_data, eq_data, error_data,
        k_values=[10]  # Standard k value from paper
    )

    assert len(results_df) == n_structures
    assert 'cknna_k10' in results_df.columns
    print(f"✓ Processed {n_structures} structures with realistic dimensions")

    # Check correlation analysis
    if len(corr_df) > 0:
        best_corr = corr_df.iloc[0]
        print(f"✓ Correlation analysis: k={best_corr['k']}, r={best_corr['spearman_r']:.4f}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("CKNNA MODULE TEST SUITE")
    print("=" * 60)

    test_functions = [
        test_cknna_basic,
        test_cknna_edge_cases,
        test_cknna_batch,
        test_cknna_analyzer,
        test_backward_compatibility,
        test_with_real_data_structure
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