"""
Simple test script for Coulomb Matrix extractor without pytest dependency.
"""

import numpy as np
from ase.build import molecule, bulk, fcc111, add_adsorbate
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.representations.physics_inspired.coulomb_matrix import CoulombMatrixExtractor

def test_basic_functionality():
    """Test basic Coulomb Matrix functionality."""
    print("Testing CoulombMatrix Extractor...")

    # Create test molecules
    water = molecule('H2O')
    co2 = molecule('CO2')

    print(f"\n1. Testing basic extraction:")
    print(f"   Water molecule: {len(water)} atoms")
    print(f"   CO2 molecule: {len(co2)} atoms")

    # Test default configuration
    extractor = CoulombMatrixExtractor(n_atoms_max=10)

    # Extract for single molecule
    result = extractor.extract_single(water)
    print(f"\n2. Extracted CM for water:")
    print(f"   Shape: {result['cm_all'].shape}")
    print(f"   Number of atoms: {result['cm_all_natoms'][0]}")

    # Test eigenspectrum
    extractor_eigen = CoulombMatrixExtractor(
        n_atoms_max=10,
        permutation='eigenspectrum'
    )

    result_eigen = extractor_eigen.extract_single(co2)
    print(f"\n3. Eigenspectrum for CO2:")
    print(f"   Shape: {result_eigen['cm_all'].shape}")
    print(f"   First 5 eigenvalues: {result_eigen['cm_all'][:5]}")

    # Test batch extraction
    structures = [water, co2]
    tasknames = ['water', 'co2']

    batch_results = extractor.extract_batch(structures, tasknames)
    print(f"\n4. Batch extraction results:")
    for name in tasknames:
        if name in batch_results:
            print(f"   {name}: {batch_results[name]['cm_all'].shape}")

    # Test with surface + adsorbate
    slab = fcc111('Cu', size=(2, 2, 3), vacuum=10.0)
    co_molecule = molecule('CO')
    add_adsorbate(slab, co_molecule, 2.0, 'ontop')
    n_atoms = len(slab)
    adsorbate_indices = [n_atoms - 2, n_atoms - 1]

    print(f"\n5. Testing with surface + adsorbate ({n_atoms} atoms):")

    extractor_surface = CoulombMatrixExtractor(n_atoms_max=50)

    # All atoms
    result_all = extractor_surface.extract_single(slab, atom_selection="all")
    print(f"   All atoms: {result_all['cm_all_natoms'][0]} atoms")

    # Slab only
    result_slab = extractor_surface.extract_single(
        slab,
        atom_selection="slab",
        adsorbate_indices=adsorbate_indices
    )
    print(f"   Slab only: {result_slab['cm_slab_natoms'][0]} atoms")

    # Test different permutation methods
    print(f"\n6. Testing different permutation methods:")

    permutations = ['none', 'sorted_l2', 'eigenspectrum']
    for perm in permutations:
        ext = CoulombMatrixExtractor(n_atoms_max=10, permutation=perm)
        res = ext.extract_single(water)
        print(f"   {perm}: shape={res['cm_all'].shape}")

    # Test random permutation
    extractor_random = CoulombMatrixExtractor(
        n_atoms_max=10,
        permutation='random',
        sigma=0.1,
        seed=42
    )
    result_random = extractor_random.extract_single(water)
    print(f"   random: shape={result_random['cm_all'].shape}")

    print("\n✓ All tests passed successfully!")

if __name__ == "__main__":
    test_basic_functionality()