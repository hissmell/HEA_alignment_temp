"""
Tests for Coulomb Matrix descriptor extractor.
"""

import pytest
import numpy as np
from pathlib import Path
from ase import Atoms
from ase.build import molecule, bulk, fcc111, add_adsorbate

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.representations.physics_inspired.coulomb_matrix import CoulombMatrixExtractor
from src.representations.base import ExtractionConfig


class TestCoulombMatrixExtractor:
    """Test suite for Coulomb Matrix extractor."""

    @pytest.fixture
    def water_molecule(self):
        """Create a water molecule for testing."""
        return molecule('H2O')

    @pytest.fixture
    def co2_molecule(self):
        """Create a CO2 molecule for testing."""
        return molecule('CO2')

    @pytest.fixture
    def bulk_structure(self):
        """Create a bulk Cu structure."""
        return bulk('Cu', 'fcc', a=3.6) * (2, 2, 2)

    @pytest.fixture
    def surface_with_adsorbate(self):
        """Create a surface with CO adsorbate."""
        slab = fcc111('Cu', size=(2, 2, 3), vacuum=10.0)
        add_adsorbate(slab, 'CO', 2.0, 'ontop')
        return slab

    def test_initialization(self):
        """Test extractor initialization."""
        # Default initialization
        extractor = CoulombMatrixExtractor()
        assert extractor.n_atoms_max == 200
        assert extractor.permutation == 'sorted_l2'
        assert extractor.flatten == True
        assert not extractor.is_initialized

        # Custom initialization
        extractor = CoulombMatrixExtractor(
            n_atoms_max=50,
            permutation='eigenspectrum',
            flatten=False
        )
        assert extractor.n_atoms_max == 50
        assert extractor.permutation == 'eigenspectrum'
        assert extractor.flatten == False

        # Invalid permutation should raise error
        with pytest.raises(ValueError):
            CoulombMatrixExtractor(permutation='invalid')

        # Random permutation without sigma should raise error
        with pytest.raises(ValueError):
            CoulombMatrixExtractor(permutation='random')

    def test_setup(self):
        """Test descriptor setup."""
        extractor = CoulombMatrixExtractor(n_atoms_max=10)
        extractor.setup()
        assert extractor.is_initialized
        assert extractor.descriptor is not None

    def test_extract_single_molecule(self, water_molecule):
        """Test extraction for a single molecule."""
        extractor = CoulombMatrixExtractor(n_atoms_max=10, flatten=True)

        # Extract representation
        result = extractor.extract_single(water_molecule)

        # Check results
        assert 'cm_all' in result
        assert isinstance(result['cm_all'], np.ndarray)
        assert result['cm_all'].shape == (100,)  # 10x10 flattened
        assert 'cm_all_natoms' in result
        assert result['cm_all_natoms'][0] == 3  # H2O has 3 atoms

    def test_extract_eigenspectrum(self, co2_molecule):
        """Test eigenspectrum extraction."""
        extractor = CoulombMatrixExtractor(
            n_atoms_max=10,
            permutation='eigenspectrum'
        )

        result = extractor.extract_single(co2_molecule)

        assert 'cm_all' in result
        assert isinstance(result['cm_all'], np.ndarray)
        assert result['cm_all'].shape == (10,)  # eigenvalues
        # Check that eigenvalues are sorted
        eigenvalues = result['cm_all']
        non_zero = eigenvalues[eigenvalues != 0]
        assert np.all(np.abs(non_zero[:-1]) >= np.abs(non_zero[1:]))

    def test_extract_matrix_not_flattened(self, water_molecule):
        """Test extraction without flattening."""
        extractor = CoulombMatrixExtractor(
            n_atoms_max=10,
            flatten=False
        )

        result = extractor.extract_single(water_molecule)

        assert 'cm_all' in result
        assert isinstance(result['cm_all'], np.ndarray)
        assert result['cm_all'].shape == (10, 10)  # 2D matrix

    def test_extract_with_atom_selection(self, surface_with_adsorbate):
        """Test extraction with different atom selections."""
        # Get adsorbate indices (last 2 atoms for CO)
        n_atoms = len(surface_with_adsorbate)
        adsorbate_indices = [n_atoms - 2, n_atoms - 1]

        extractor = CoulombMatrixExtractor(n_atoms_max=50, flatten=True)

        # All atoms
        result_all = extractor.extract_single(surface_with_adsorbate, atom_selection="all")
        assert 'cm_all' in result_all

        # Slab only
        result_slab = extractor.extract_single(
            surface_with_adsorbate,
            atom_selection="slab",
            adsorbate_indices=adsorbate_indices
        )
        assert 'cm_slab' in result_slab
        assert result_slab['cm_slab_natoms'][0] == n_atoms - 2

        # Site atoms (within cutoff of adsorbate)
        result_site = extractor.extract_single(
            surface_with_adsorbate,
            atom_selection="site",
            adsorbate_indices=adsorbate_indices,
            site_cutoff=3.0
        )
        # Site may not exist if no atoms within cutoff
        if result_site:
            assert 'cm_site' in result_site or len(result_site) == 0

    def test_extract_batch(self, water_molecule, co2_molecule):
        """Test batch extraction."""
        structures = [water_molecule, co2_molecule]
        tasknames = ['water', 'co2']

        extractor = CoulombMatrixExtractor(n_atoms_max=10)
        results = extractor.extract_batch(structures, tasknames)

        assert len(results) == 2
        assert 'water' in results
        assert 'co2' in results
        assert 'cm_all' in results['water']
        assert 'cm_all' in results['co2']

    def test_extract_batch_parallel(self, water_molecule, co2_molecule):
        """Test parallel batch extraction."""
        structures = [water_molecule, co2_molecule] * 5

        extractor = CoulombMatrixExtractor(n_atoms_max=10)
        results = extractor.extract_batch(structures, n_jobs=2)

        assert len(results) == 10

    def test_structure_too_large(self, bulk_structure):
        """Test handling of structures exceeding n_atoms_max."""
        extractor = CoulombMatrixExtractor(n_atoms_max=10)  # Too small

        result = extractor.extract_single(bulk_structure)

        # Should return empty dict for too-large structure
        assert len(result) == 0

    def test_validate_structures(self, water_molecule, bulk_structure):
        """Test structure validation."""
        extractor = CoulombMatrixExtractor(n_atoms_max=10)

        valid = extractor.validate_structures([water_molecule, bulk_structure])

        assert valid[0] == True  # Water has 3 atoms
        assert valid[1] == False  # Bulk has 32 atoms > 10

    def test_random_permutation(self, water_molecule):
        """Test random permutation with seed."""
        extractor1 = CoulombMatrixExtractor(
            n_atoms_max=10,
            permutation='random',
            sigma=0.1,
            seed=42
        )

        extractor2 = CoulombMatrixExtractor(
            n_atoms_max=10,
            permutation='random',
            sigma=0.1,
            seed=42
        )

        result1 = extractor1.extract_single(water_molecule)
        result2 = extractor2.extract_single(water_molecule)

        # Same seed should give same results
        np.testing.assert_allclose(result1['cm_all'], result2['cm_all'])

        # Different seed should give different results
        extractor3 = CoulombMatrixExtractor(
            n_atoms_max=10,
            permutation='random',
            sigma=0.1,
            seed=123
        )
        result3 = extractor3.extract_single(water_molecule)

        # Results should be different (with high probability)
        assert not np.allclose(result1['cm_all'], result3['cm_all'])

    def test_get_feature_info(self):
        """Test feature information methods."""
        # Flattened matrix
        extractor = CoulombMatrixExtractor(n_atoms_max=10, flatten=True)
        names = extractor.get_feature_names()
        dims = extractor.get_feature_dimensions()

        assert len(names) == 100  # 10x10 flattened
        assert dims['flattened_matrix'] == 100

        # Eigenspectrum
        extractor = CoulombMatrixExtractor(
            n_atoms_max=10,
            permutation='eigenspectrum'
        )
        names = extractor.get_feature_names()
        dims = extractor.get_feature_dimensions()

        assert len(names) == 10  # 10 eigenvalues
        assert dims['eigenspectrum'] == 10

        # Non-flattened matrix
        extractor = CoulombMatrixExtractor(n_atoms_max=10, flatten=False)
        dims = extractor.get_feature_dimensions()
        assert dims['matrix'] == (10, 10)

    def test_descriptor_info(self):
        """Test descriptor info method."""
        extractor = CoulombMatrixExtractor(
            n_atoms_max=50,
            permutation='sorted_l2',
            flatten=True
        )

        info = extractor.get_descriptor_info()

        assert info['name'] == 'coulomb_matrix'
        assert info['n_atoms_max'] == 50
        assert info['permutation'] == 'sorted_l2'
        assert info['flatten'] == True
        assert info['sparse'] == False
        assert 'feature_size' in info


if __name__ == "__main__":
    pytest.main([__file__, "-v"])