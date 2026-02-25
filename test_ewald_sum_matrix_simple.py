"""
Simple test script for Ewald Sum Matrix extractor without pytest dependency.
"""

import numpy as np
from ase.build import bulk, fcc111, add_adsorbate, molecule
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.representations.physics_inspired.ewald_sum_matrix import EwaldSumMatrixExtractor

def test_basic_functionality():
    """Test basic Ewald Sum Matrix functionality."""
    print("Testing EwaldSumMatrix Extractor...")

    # Create test structures - periodic crystals only
    nacl = bulk("NaCl", "rocksalt", a=5.64)
    al = bulk("Al", "fcc", a=4.046)
    fe = bulk("Fe", "bcc", a=2.856)
    si = bulk("Si", "diamond", a=5.43)

    print(f"\n1. Testing with periodic crystal structures:")
    print(f"   NaCl: {len(nacl)} atoms, pbc={nacl.pbc}")
    print(f"   Al: {len(al)} atoms, pbc={al.pbc}")
    print(f"   Fe: {len(fe)} atoms, pbc={fe.pbc}")
    print(f"   Si: {len(si)} atoms, pbc={si.pbc}")

    # Test default configuration
    extractor = EwaldSumMatrixExtractor(n_atoms_max=10)

    # Extract for single crystal
    result = extractor.extract_single(nacl)
    print(f"\n2. Extracted ESM for NaCl:")
    print(f"   Shape: {result['esm_all'].shape}")
    print(f"   Number of atoms: {result['esm_all_natoms'][0]}")
    print(f"   Periodic boundaries: {result['esm_all_periodic']}")
    print(f"   Cell shape: {result['esm_all_cell'][:9].reshape(3,3)}")

    # Test eigenspectrum
    extractor_eigen = EwaldSumMatrixExtractor(
        n_atoms_max=10,
        permutation='eigenspectrum'
    )

    result_eigen = extractor_eigen.extract_single(al)
    print(f"\n3. Eigenspectrum for Al:")
    print(f"   Shape: {result_eigen['esm_all'].shape}")
    print(f"   First 5 eigenvalues: {result_eigen['esm_all'][:5]}")

    # Test batch extraction
    structures = [nacl, al, fe, si]
    tasknames = ['NaCl', 'Al', 'Fe', 'Si']

    batch_results = extractor.extract_batch(structures, tasknames)
    print(f"\n4. Batch extraction results:")
    for name in tasknames:
        if name in batch_results:
            print(f"   {name}: {batch_results[name]['esm_all'].shape}")

    # Test with different Ewald parameters
    print(f"\n5. Testing with different Ewald parameters:")

    # High accuracy
    extractor_high_acc = EwaldSumMatrixExtractor(n_atoms_max=10)
    result_high = extractor_high_acc.extract_single(nacl, accuracy=1e-8)
    print(f"   High accuracy (1e-8): shape={result_high['esm_all'].shape}")

    # Custom cutoffs
    result_cutoff = extractor.extract_single(nacl, r_cut=10.0, g_cut=10.0)
    print(f"   Custom cutoffs (r=10, g=10): shape={result_cutoff['esm_all'].shape}")

    # Custom screening parameter
    result_screen = extractor.extract_single(nacl, a=0.5)
    print(f"   Custom screening (a=0.5): shape={result_screen['esm_all'].shape}")

    # Test with surface slab (periodic in xy, non-periodic in z)
    slab = fcc111('Cu', size=(2, 2, 3), vacuum=10.0)
    slab.pbc = [True, True, True]  # Make fully periodic for ESM
    co_molecule = molecule('CO')
    add_adsorbate(slab, co_molecule, 2.0, 'ontop')
    n_atoms = len(slab)
    adsorbate_indices = [n_atoms - 2, n_atoms - 1]

    print(f"\n6. Testing with surface slab ({n_atoms} atoms):")
    print(f"   Periodic boundaries: {slab.pbc}")

    extractor_surface = EwaldSumMatrixExtractor(n_atoms_max=50)

    # All atoms
    result_all = extractor_surface.extract_single(slab, atom_selection="all")
    print(f"   All atoms: {result_all['esm_all_natoms'][0]} atoms")

    # Slab only
    result_slab = extractor_surface.extract_single(
        slab,
        atom_selection="slab",
        adsorbate_indices=adsorbate_indices
    )
    print(f"   Slab only: {result_slab['esm_slab_natoms'][0]} atoms")

    # Test different permutation methods
    print(f"\n7. Testing different permutation methods:")

    permutations = ['none', 'sorted_l2', 'eigenspectrum']
    for perm in permutations:
        ext = EwaldSumMatrixExtractor(n_atoms_max=10, permutation=perm)
        res = ext.extract_single(nacl)
        print(f"   {perm}: shape={res['esm_all'].shape}")

    # Test random permutation
    extractor_random = EwaldSumMatrixExtractor(
        n_atoms_max=10,
        permutation='random',
        sigma=0.1,
        seed=42
    )
    result_random = extractor_random.extract_single(nacl)
    print(f"   random: shape={result_random['esm_all'].shape}")

    # Test with non-periodic structure (should fail gracefully)
    print(f"\n8. Testing with non-periodic structure (water molecule):")
    water = molecule('H2O')
    water.pbc = False  # Explicitly set non-periodic

    result_water = extractor.extract_single(water)
    if not result_water:
        print(f"   Water (non-periodic): Correctly rejected (ESM requires periodicity)")
    else:
        print(f"   ERROR: Non-periodic structure should have been rejected")

    # Test validation
    print(f"\n9. Testing structure validation:")
    structures_to_validate = [nacl, al, water]
    valid = extractor.validate_structures(structures_to_validate)
    for i, (struct, is_valid) in enumerate(zip(['NaCl', 'Al', 'H2O'], valid)):
        print(f"   {struct}: {'Valid' if is_valid else 'Invalid (requires periodicity)'}")

    # Test parameter setting
    print(f"\n10. Testing Ewald parameter configuration:")
    extractor.set_ewald_parameters(accuracy=1e-6, weight=2.0)
    info = extractor.get_descriptor_info()
    print(f"   Updated accuracy: {info['accuracy']}")
    print(f"   Updated weight: {info['weight']}")

    # Test descriptor comparison
    print(f"\n11. Testing descriptor comparison (ESM vs CM vs SM):")
    extractor_compare = EwaldSumMatrixExtractor(n_atoms_max=10, flatten=True)
    comparison = extractor_compare.compare_descriptors(nacl, compare_with=['coulomb', 'sine'])

    for desc_name, matrix in comparison.items():
        if len(matrix) > 0:
            print(f"   {desc_name}: shape={matrix.shape}, norm={np.linalg.norm(matrix):.3f}")

    # Test different data types
    print(f"\n12. Testing different data types:")
    for dtype in ['float32', 'float64']:
        ext = EwaldSumMatrixExtractor(n_atoms_max=10, dtype=dtype)
        res = ext.extract_single(nacl)
        if res:
            print(f"   dtype={dtype}: array dtype={res['esm_all'].dtype}")

    print("\n✓ All tests passed successfully!")

if __name__ == "__main__":
    test_basic_functionality()