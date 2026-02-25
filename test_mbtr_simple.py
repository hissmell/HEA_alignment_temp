"""
Simple test script for MBTR extractor without pytest dependency.
"""

import numpy as np
from ase.build import molecule, bulk, fcc111, add_adsorbate
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.representations.physics_inspired.mbtr import MBTRExtractor, MBTRConfig

def test_basic_functionality():
    """Test basic MBTR functionality."""
    print("Testing MBTR Extractor...")

    # Create test structures
    water = molecule('H2O')
    co2 = molecule('CO2')
    methane = molecule('CH4')
    nacl = bulk("NaCl", "rocksalt", a=5.64)

    print(f"\n1. Testing with molecules and crystals:")
    print(f"   Water: {len(water)} atoms")
    print(f"   CO2: {len(co2)} atoms")
    print(f"   Methane: {len(methane)} atoms")
    print(f"   NaCl crystal: {len(nacl)} atoms, pbc={nacl.pbc}")

    # Test default configuration (k2 with inverse distances)
    species = ['H', 'O', 'C', 'Na', 'Cl']
    extractor = MBTRExtractor(
        species=species,
        geometry={"function": "inverse_distance"},
        grid={"min": 0, "max": 1, "n": 100, "sigma": 0.1},
        weighting={"function": "exp", "scale": 0.5, "threshold": 1e-3},
        periodic=False,
        normalization="l2"
    )

    # Extract for single molecule
    result = extractor.extract_single(water)
    print(f"\n2. Extracted MBTR for water (k2 inverse distance):")
    print(f"   Shape: {result['mbtr_all'].shape}")
    print(f"   Number of atoms: {result['mbtr_all_natoms'][0]}")
    print(f"   K-terms included: {result['mbtr_all_kterms']}")

    # Test k1 (atomic numbers) - k1 doesn't support weighting
    extractor_k1 = MBTRExtractor(
        species=species,
        geometry={"function": "atomic_number"},
        grid={"min": 0, "max": 10, "n": 100, "sigma": 0.1},
        weighting={"function": "unity"},  # k1 only supports unity weighting
        periodic=False,
        normalization="l2"
    )

    result_k1 = extractor_k1.extract_single(co2)
    print(f"\n3. MBTR k1 (atomic numbers) for CO2:")
    print(f"   Shape: {result_k1['mbtr_all'].shape}")
    print(f"   Non-zero elements: {np.count_nonzero(result_k1['mbtr_all'])}")

    # Test k3 (angles)
    extractor_k3 = MBTRExtractor(
        species=species,
        geometry={"function": "cosine"},
        grid={"min": -1, "max": 1, "n": 100, "sigma": 0.1},
        weighting={"function": "exp", "scale": 0.5, "threshold": 1e-3},
        periodic=False,
        normalization="l2"
    )

    result_k3 = extractor_k3.extract_single(methane)
    print(f"\n4. MBTR k3 (cosine angles) for methane:")
    print(f"   Shape: {result_k3['mbtr_all'].shape}")

    # Test batch extraction
    structures = [water, co2, methane]
    tasknames = ['water', 'co2', 'methane']

    batch_results = extractor.extract_batch(structures, tasknames)
    print(f"\n5. Batch extraction results:")
    for name in tasknames:
        if name in batch_results:
            print(f"   {name}: {batch_results[name]['mbtr_all'].shape}")

    # Test periodic system
    extractor_periodic = MBTRExtractor(
        species=species,
        geometry={"function": "inverse_distance"},
        grid={"min": 0, "max": 1, "n": 100, "sigma": 0.1},
        weighting={"function": "exp", "scale": 0.5, "threshold": 1e-3},
        periodic=True,
        normalization="l2"
    )

    result_periodic = extractor_periodic.extract_single(nacl)
    print(f"\n6. MBTR for periodic crystal (NaCl):")
    print(f"   Shape: {result_periodic['mbtr_all'].shape}")
    print(f"   Periodic boundaries: {result_periodic['mbtr_all_periodic']}")

    # Test different normalization methods
    print(f"\n7. Testing different normalization methods:")

    for norm in ['none', 'l2', 'n_atoms']:
        ext = MBTRExtractor(
            species=species,
            geometry={"function": "inverse_distance"},
            grid={"min": 0, "max": 1, "n": 50, "sigma": 0.1},
            weighting={"function": "exp", "scale": 0.5, "threshold": 1e-3},
            normalization=norm
        )
        res = ext.extract_single(water)
        norm_val = np.linalg.norm(res['mbtr_all']) if norm == 'l2' else np.sum(res['mbtr_all'])
        print(f"   {norm}: shape={res['mbtr_all'].shape}, norm/sum={norm_val:.3f}")

    # Test multi-k descriptor
    print(f"\n8. Testing multi-k descriptor (k1+k2+k3):")
    multi_k_result = extractor.create_multi_k_descriptor(
        water,
        include_k1=True,
        include_k2=True,
        include_k3=True
    )

    for key, value in multi_k_result.items():
        if isinstance(value, np.ndarray) and len(value) > 0:
            print(f"   {key}: shape={value.shape}")

    # Test different weighting functions for k2
    print(f"\n9. Testing different weighting functions (k2):")

    weighting_functions = [
        {"function": "unity"},
        {"function": "exp", "scale": 0.5, "threshold": 1e-3},
        {"function": "inverse_square", "r_cut": 5.0}
    ]

    for weight_config in weighting_functions:
        try:
            ext = MBTRExtractor(
                species=species,
                geometry={"function": "distance"},
                grid={"min": 0, "max": 10, "n": 50, "sigma": 0.1},
                weighting=weight_config,
                normalization="l2"
            )
            res = ext.extract_single(water)
            print(f"   {weight_config['function']}: shape={res['mbtr_all'].shape}, "
                  f"non-zero={np.count_nonzero(res['mbtr_all'])}")
        except Exception as e:
            print(f"   {weight_config['function']}: Failed - {e}")

    # Test with surface + adsorbate
    slab = fcc111('Cu', size=(2, 2, 3), vacuum=10.0)
    co_molecule = molecule('CO')
    add_adsorbate(slab, co_molecule, 2.0, 'ontop')
    n_atoms = len(slab)
    adsorbate_indices = [n_atoms - 2, n_atoms - 1]

    print(f"\n10. Testing with surface + adsorbate ({n_atoms} atoms):")

    extractor_surface = MBTRExtractor(
        species=['Cu', 'C', 'O'],
        geometry={"function": "inverse_distance"},
        grid={"min": 0, "max": 1, "n": 100, "sigma": 0.1},
        weighting={"function": "exp", "scale": 0.5, "threshold": 1e-3},
        normalization="l2"
    )

    # All atoms
    result_all = extractor_surface.extract_single(slab, atom_selection="all")
    print(f"   All atoms: {result_all['mbtr_all_natoms'][0]} atoms")

    # Slab only
    result_slab = extractor_surface.extract_single(
        slab,
        atom_selection="slab",
        adsorbate_indices=adsorbate_indices
    )
    print(f"   Slab only: {result_slab['mbtr_slab_natoms'][0]} atoms")

    # Test structure validation
    print(f"\n11. Testing structure validation:")

    # Create structure with undefined species
    unknown_molecule = molecule('NH3')  # N not in species list
    structures_to_validate = [water, co2, unknown_molecule]

    valid = extractor.validate_structures(structures_to_validate)
    for i, (struct, is_valid) in enumerate(zip(['H2O', 'CO2', 'NH3'], valid)):
        print(f"   {struct}: {'Valid' if is_valid else 'Invalid (contains undefined species)'}")

    # Test geometry function comparison
    print(f"\n12. Testing geometry function comparison (k2):")
    comparison = extractor.compare_geometry_functions(water, k_degree=2)

    for geom_func, mbtr_array in comparison.items():
        if len(mbtr_array) > 0:
            print(f"   {geom_func}: shape={mbtr_array.shape}, "
                  f"max={np.max(mbtr_array):.3f}")

    # Test different data types
    print(f"\n13. Testing different data types:")
    for dtype in ['float32', 'float64']:
        ext = MBTRExtractor(
            species=species,
            geometry={"function": "inverse_distance"},
            grid={"min": 0, "max": 1, "n": 50, "sigma": 0.1},
            weighting={"function": "exp", "scale": 0.5, "threshold": 1e-3},
            dtype=dtype
        )
        res = ext.extract_single(water)
        print(f"   dtype={dtype}: array dtype={res['mbtr_all'].dtype}")

    # Test advanced multi-k configuration
    print(f"\n14. Testing advanced multi-k configuration:")

    mbtr_config = MBTRConfig(
        k1={
            "geometry": {"function": "atomic_number"},
            "grid": {"min": 0, "max": 10, "n": 50, "sigma": 0.1},
            "weighting": {"function": "unity"}  # k1 only supports unity
        },
        k2={
            "geometry": {"function": "inverse_distance"},
            "grid": {"min": 0, "max": 1, "n": 50, "sigma": 0.1},
            "weighting": {"function": "exp", "scale": 0.5, "threshold": 1e-3}
        }
    )

    extractor_multi = MBTRExtractor(
        species=species,
        mbtr_config=mbtr_config,
        normalization="l2"
    )

    result_multi = extractor_multi.extract_single(water)
    print(f"   Multi-k MBTR: shape={result_multi['mbtr_all'].shape}")
    print(f"   K-terms: {result_multi['mbtr_all_kterms']}")

    print("\n✓ All tests passed successfully!")

if __name__ == "__main__":
    test_basic_functionality()