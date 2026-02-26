"""
Simple test script for Sine Matrix extractor without pytest dependency.
"""

import numpy as np
from ase.build import bulk, fcc111, add_adsorbate, molecule
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.representations.physics_inspired.sine_matrix import SineMatrixExtractor

def test_basic_functionality():
    """Test basic Sine Matrix functionality."""
    print("Testing SineMatrix Extractor...")

    # Create test structures - periodic crystals
    nacl = bulk("NaCl", "rocksalt", a=5.64)
    al = bulk("Al", "fcc", a=4.046)
    fe = bulk("Fe", "bcc", a=2.856)

    print(f"\n1. Testing with periodic structures:")
    print(f"   NaCl: {len(nacl)} atoms, pbc={nacl.pbc}")
    print(f"   Al: {len(al)} atoms, pbc={al.pbc}")
    print(f"   Fe: {len(fe)} atoms, pbc={fe.pbc}")

    # Test default configuration
    extractor = SineMatrixExtractor(n_atoms_max=10)

    # Extract for single crystal
    result = extractor.extract_single(nacl)
    print(f"\n2. Extracted SM for NaCl:")
    print(f"   Shape: {result['sm_all'].shape}")
    print(f"   Number of atoms: {result['sm_all_natoms'][0]}")
    print(f"   Periodic boundaries: {result['sm_all_periodic']}")

    # Test eigenspectrum
    extractor_eigen = SineMatrixExtractor(
        n_atoms_max=10,
        permutation='eigenspectrum'
    )

    result_eigen = extractor_eigen.extract_single(al)
    print(f"\n3. Eigenspectrum for Al:")
    print(f"   Shape: {result_eigen['sm_all'].shape}")
    print(f"   First 5 eigenvalues: {result_eigen['sm_all'][:5]}")

    # Test batch extraction
    structures = [nacl, al, fe]
    tasknames = ['NaCl', 'Al', 'Fe']

    batch_results = extractor.extract_batch(structures, tasknames)
    print(f"\n4. Batch extraction results:")
    for name in tasknames:
        if name in batch_results:
            print(f"   {name}: {batch_results[name]['sm_all'].shape}")

    # Test with surface slab (periodic in xy, non-periodic in z)
    slab = fcc111('Cu', size=(2, 2, 3), vacuum=10.0)
    co_molecule = molecule('CO')
    add_adsorbate(slab, co_molecule, 2.0, 'ontop')
    n_atoms = len(slab)
    adsorbate_indices = [n_atoms - 2, n_atoms - 1]

    print(f"\n5. Testing with surface slab ({n_atoms} atoms):")
    print(f"   Periodic boundaries: {slab.pbc}")

    extractor_surface = SineMatrixExtractor(n_atoms_max=50)

    # All atoms
    result_all = extractor_surface.extract_single(slab, atom_selection="all")
    print(f"   All atoms: {result_all['sm_all_natoms'][0]} atoms")

    # Slab only
    result_slab = extractor_surface.extract_single(
        slab,
        atom_selection="slab",
        adsorbate_indices=adsorbate_indices
    )
    print(f"   Slab only: {result_slab['sm_slab_natoms'][0]} atoms")

    # Test different permutation methods
    print(f"\n6. Testing different permutation methods:")

    permutations = ['none', 'sorted_l2', 'eigenspectrum']
    for perm in permutations:
        ext = SineMatrixExtractor(n_atoms_max=10, permutation=perm)
        res = ext.extract_single(nacl)
        print(f"   {perm}: shape={res['sm_all'].shape}")

    # Test random permutation
    extractor_random = SineMatrixExtractor(
        n_atoms_max=10,
        permutation='random',
        sigma=0.1,
        seed=42
    )
    result_random = extractor_random.extract_single(nacl)
    print(f"   random: shape={result_random['sm_all'].shape}")

    # Test with non-periodic structure (should give warning)
    print(f"\n7. Testing with non-periodic structure (water molecule):")
    water = molecule('H2O')
    water.pbc = False  # Explicitly set non-periodic

    result_water = extractor.extract_single(water)
    if result_water:
        print(f"   Water (non-periodic): shape={result_water['sm_all'].shape}")
        print(f"   Note: Sine Matrix is designed for periodic structures")

    # Test validation
    print(f"\n8. Testing structure validation:")
    structures_to_validate = [nacl, al, water]
    valid = extractor.validate_structures(structures_to_validate, check_periodicity=True)
    for i, (struct, is_valid) in enumerate(zip(['NaCl', 'Al', 'H2O'], valid)):
        print(f"   {struct}: {'Valid' if is_valid else 'Invalid'}")

    # Test different data types
    print(f"\n9. Testing different data types:")
    for dtype in ['float32', 'float64']:
        ext = SineMatrixExtractor(n_atoms_max=10, dtype=dtype)
        res = ext.extract_single(nacl)
        print(f"   dtype={dtype}: array dtype={res['sm_all'].dtype}")

    print("\n✓ All tests passed successfully!")

if __name__ == "__main__":
    test_basic_functionality()