"""
Test Local Many-Body Tensor Representation (LMBTR) extractor
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parents[2]
sys.path.append(str(project_root))

from src.representations.physics_inspired.local_mbtr import LocalMBTRExtractor, create_25cao_lmbtr_extractor
from ase import Atoms
from ase.build import fcc100, add_adsorbate


def create_test_structure():
    """Create a test HEA surface structure with adsorbate for testing."""
    # Create simple metal surface (mimicking 25Cao)
    slab = fcc100('Pt', size=(3, 3, 4), a=4.0, vacuum=10.0)

    # Replace some atoms to simulate HEA composition
    symbols = slab.get_chemical_symbols()
    hea_elements = ['Ag', 'Ir', 'Pd', 'Pt', 'Ru']

    # Replace atoms with HEA elements
    for i in range(0, len(symbols), 3):
        if i < len(symbols):
            symbols[i] = hea_elements[i % len(hea_elements)]

    slab.set_chemical_symbols(symbols)

    # Add O adsorbate on top
    add_adsorbate(slab, 'O', 2.0, position='ontop')

    return slab


def create_simple_molecule():
    """Create simple molecule for testing."""
    atoms = Atoms(['O', 'H', 'H'],
                  positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    atoms.center(vacuum=5.0)
    atoms.pbc = False
    return atoms


class TestLocalMBTRExtractor:
    """Test LocalMBTRExtractor functionality."""

    @pytest.fixture
    def extractor(self):
        """Create test extractor with 25Cao species."""
        species = ['Ag', 'Ir', 'Pd', 'Pt', 'Ru', 'O', 'H']
        return LocalMBTRExtractor(
            species=species,
            r_cut=6.0,
            geometry={"function": "inverse_distance"},
            grid={"min": 0, "max": 1, "n": 50, "sigma": 0.02},  # Smaller grid for testing
            weighting={"function": "exp", "scale": 0.5, "threshold": 1e-3}
        )

    @pytest.fixture
    def hea_structure(self):
        """Create HEA test structure."""
        return create_test_structure()

    @pytest.fixture
    def molecule(self):
        """Create simple molecule."""
        return create_simple_molecule()

    def test_initialization(self, extractor):
        """Test LocalMBTRExtractor initialization."""
        assert extractor.species == ['Ag', 'Ir', 'Pd', 'Pt', 'Ru', 'O', 'H']
        assert extractor.r_cut == 6.0
        assert extractor.geometry["function"] == "inverse_distance"
        assert extractor.grid["n"] == 50
        assert extractor.lmbtr is None  # Not initialized until setup

    def test_setup(self, extractor):
        """Test LMBTR setup with dscribe."""
        extractor.setup()
        assert extractor.lmbtr is not None
        print(f"LMBTR setup successful")

    def test_extract_adsorbates_centers(self, extractor, hea_structure):
        """Test extraction with adsorbates as centers."""
        result = extractor.extract_single(hea_structure, centers="adsorbates")

        assert 'lmbtr_all' in result
        assert 'centers' in result
        assert 'n_centers' in result
        assert 'metadata' in result

        # Check shapes
        lmbtr_features = result['lmbtr_all']
        assert isinstance(lmbtr_features, np.ndarray)
        assert lmbtr_features.ndim == 2  # (n_centers, n_features)

        # Should find O atoms as adsorbates
        assert result['n_centers'] >= 1
        print(f"Found {result['n_centers']} adsorbate centers")
        print(f"LMBTR shape: {lmbtr_features.shape}")

    def test_extract_specific_atoms(self, extractor, hea_structure):
        """Test extraction with specific atom indices."""
        # Use first few atoms as centers
        test_centers = [0, 1, 2]
        result = extractor.extract_single(hea_structure, centers=test_centers)

        assert 'lmbtr_all' in result
        lmbtr_features = result['lmbtr_all']

        # Should have features for each center
        expected_centers = len(test_centers)  # May be less if some fail
        assert result['n_centers'] <= expected_centers
        print(f"Requested {len(test_centers)} centers, got {result['n_centers']}")
        print(f"LMBTR shape: {lmbtr_features.shape}")

    def test_extract_all_atoms(self, extractor, molecule):
        """Test extraction with all atoms as centers using molecule."""
        result = extractor.extract_single(molecule, centers="all")

        assert 'lmbtr_all' in result
        lmbtr_features = result['lmbtr_all']

        # Should have features for all atoms (or fallback to global)
        assert result['n_centers'] >= 1
        print(f"All atoms mode: {result['n_centers']} centers")
        print(f"LMBTR shape: {lmbtr_features.shape}")

    def test_metadata(self, extractor, molecule):
        """Test metadata extraction."""
        result = extractor.extract_single(molecule, centers="adsorbates")
        metadata = result['metadata']

        assert metadata['descriptor'] == 'local_mbtr'
        assert metadata['species'] == ['Ag', 'Ir', 'Pd', 'Pt', 'Ru', 'O', 'H']
        assert metadata['r_cut'] == 6.0
        assert 'geometry' in metadata
        assert 'grid' in metadata
        assert 'n_atoms' in metadata
        assert metadata['n_atoms'] == len(molecule)

    def test_feature_shape(self, extractor, molecule):
        """Test feature shape consistency."""
        result = extractor.extract_single(molecule, centers=[0])

        assert 'feature_shape' in result
        feature_shape = result['feature_shape']
        lmbtr_features = result['lmbtr_all']

        if lmbtr_features.ndim > 1:
            assert feature_shape == lmbtr_features.shape[1:]
        print(f"Feature shape: {feature_shape}")

    def test_different_geometry_functions(self):
        """Test different geometry function configurations."""
        species = ['O', 'H']

        # Test k2 distance
        extractor_k2_dist = LocalMBTRExtractor(
            species=species,
            geometry={"function": "distance"},
            grid={"min": 0, "max": 10, "n": 20, "sigma": 0.1}
        )

        # Test k3 angle
        extractor_k3_angle = LocalMBTRExtractor(
            species=species,
            geometry={"function": "angle"},
            grid={"min": 0, "max": 180, "n": 20, "sigma": 2}
        )

        molecule = create_simple_molecule()

        # Test k2 distance
        result_k2 = extractor_k2_dist.extract_single(molecule, centers=[0])
        assert 'lmbtr_all' in result_k2
        print(f"K2 distance shape: {result_k2['lmbtr_all'].shape}")

        # Test k3 angle
        result_k3 = extractor_k3_angle.extract_single(molecule, centers=[0])
        assert 'lmbtr_all' in result_k3
        print(f"K3 angle shape: {result_k3['lmbtr_all'].shape}")


class TestCreate25CaoExtractor:
    """Test the 25Cao-specific extractor creator function."""

    def test_k2_inverse_distance(self):
        """Test k2 inverse distance configuration."""
        extractor = create_25cao_lmbtr_extractor(
            geometry_function="inverse_distance",
            k_term="k2"
        )

        assert extractor.species == ['Ag', 'Ir', 'Pd', 'Pt', 'Ru', 'O', 'H']
        assert extractor.geometry["function"] == "inverse_distance"
        assert extractor.grid["max"] == 1
        print("25Cao k2_inverse_distance extractor created successfully")

    def test_k3_angle(self):
        """Test k3 angle configuration."""
        extractor = create_25cao_lmbtr_extractor(
            geometry_function="angle",
            k_term="k3"
        )

        assert extractor.species == ['Ag', 'Ir', 'Pd', 'Pt', 'Ru', 'O', 'H']
        assert extractor.geometry["function"] == "angle"
        assert extractor.grid["max"] == 180
        print("25Cao k3_angle extractor created successfully")

    def test_invalid_configurations(self):
        """Test invalid configuration handling."""
        # Invalid geometry for k2
        with pytest.raises(ValueError):
            create_25cao_lmbtr_extractor("angle", "k2")

        # Invalid k-term
        with pytest.raises(ValueError):
            create_25cao_lmbtr_extractor("distance", "k4")

    def test_extraction_with_25cao_extractor(self):
        """Test actual extraction with 25Cao extractor."""
        extractor = create_25cao_lmbtr_extractor("inverse_distance", "k2")
        hea_structure = create_test_structure()

        result = extractor.extract_single(hea_structure, centers="adsorbates")
        assert 'lmbtr_all' in result
        assert result['n_centers'] >= 1
        print(f"25Cao extractor: {result['n_centers']} centers, shape: {result['lmbtr_all'].shape}")


def test_comprehensive_workflow():
    """Test complete LMBTR workflow with different structures."""
    print("\n" + "="*60)
    print("COMPREHENSIVE LMBTR WORKFLOW TEST")
    print("="*60)

    # Create extractors for different configurations
    configs = [
        ("inverse_distance", "k2"),
        ("distance", "k2"),
        ("angle", "k3"),
        ("cosine", "k3")
    ]

    hea_structure = create_test_structure()
    molecule = create_simple_molecule()

    for geom_func, k_term in configs:
        print(f"\nTesting {k_term}_{geom_func}:")

        try:
            extractor = create_25cao_lmbtr_extractor(geom_func, k_term, r_cut=5.0)

            # Test on HEA structure with adsorbates
            result_hea = extractor.extract_single(hea_structure, centers="adsorbates")
            print(f"  HEA adsorbates: {result_hea['n_centers']} centers, shape {result_hea['lmbtr_all'].shape}")

            # Test on molecule
            result_mol = extractor.extract_single(molecule, centers=[0])
            print(f"  Molecule center: {result_mol['n_centers']} centers, shape {result_mol['lmbtr_all'].shape}")

        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    # Run comprehensive test
    test_comprehensive_workflow()

    # Run specific tests
    print("\n" + "="*60)
    print("RUNNING PYTEST TESTS")
    print("="*60)
    pytest.main([__file__, "-v"])