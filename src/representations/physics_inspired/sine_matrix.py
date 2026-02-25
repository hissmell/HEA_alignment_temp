"""
Sine Matrix descriptor for periodic atomic structures.
Based on dscribe implementation, optimized for crystalline materials.
"""

import numpy as np
from typing import Dict, List, Optional, Union
from pathlib import Path
import logging

from ase import Atoms
from dscribe.descriptors import SineMatrix

from ..base import PhysicsInspiredExtractor, ExtractionConfig

logger = logging.getLogger(__name__)


class SineMatrixExtractor(PhysicsInspiredExtractor):
    """
    Sine Matrix descriptor extractor for periodic systems.

    The Sine matrix is specifically designed for periodic systems and
    encodes atomic structures based on sine-transformed distances,
    making it suitable for crystalline materials.
    """

    def __init__(
        self,
        n_atoms_max: int = 200,
        permutation: str = 'sorted_l2',
        sigma: Optional[float] = None,
        seed: Optional[int] = None,
        sparse: bool = False,
        flatten: bool = True,
        dtype: str = 'float64',
        config: Optional[ExtractionConfig] = None
    ):
        """
        Initialize Sine Matrix extractor.

        Args:
            n_atoms_max: Maximum number of atoms in any structure
            permutation: Method for handling permutational invariance
                - 'none': Return matrix in order defined by Atoms
                - 'sorted_l2': Sort rows/columns by L2 norm
                - 'eigenspectrum': Return sorted eigenvalues only
                - 'random': Sort by L2 norm with Gaussian noise
            sigma: Standard deviation for random permutation
            seed: Random seed for reproducible random permutation
            sparse: Whether to use sparse matrix representation
            flatten: Whether to flatten the output matrix
            dtype: Data type for the matrix ('float32' or 'float64')
            config: Extraction configuration
        """
        super().__init__(config)

        self.n_atoms_max = n_atoms_max
        self.permutation = permutation
        self.sigma = sigma
        self.seed = seed
        self.sparse = sparse
        self.flatten = flatten
        self.dtype = dtype

        # Validate parameters
        if permutation == 'random' and sigma is None:
            raise ValueError("sigma must be provided when using random permutation")

        if permutation not in ['none', 'sorted_l2', 'eigenspectrum', 'random']:
            raise ValueError(f"Invalid permutation method: {permutation}")

        if dtype not in ['float32', 'float64']:
            raise ValueError(f"Invalid dtype: {dtype}. Must be 'float32' or 'float64'")

    def setup(self) -> None:
        """Initialize the Sine Matrix descriptor."""
        try:
            self.descriptor = SineMatrix(
                n_atoms_max=self.n_atoms_max,
                permutation=self.permutation,
                sigma=self.sigma,
                seed=self.seed,
                sparse=self.sparse,
                dtype=self.dtype
            )
            self.is_initialized = True
            logger.info(f"Initialized SineMatrix with n_atoms_max={self.n_atoms_max}, "
                       f"permutation={self.permutation}")
        except Exception as e:
            logger.error(f"Failed to initialize SineMatrix: {e}")
            raise

    def extract_single(
        self,
        atoms: Atoms,
        atom_selection: str = "all",
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Extract Sine Matrix for a single periodic structure.

        Args:
            atoms: ASE Atoms object (should have periodic boundary conditions)
            atom_selection: Atom selection mode ("all", "slab", "site")
            **kwargs: Additional parameters (adsorbate_indices, site_cutoff)

        Returns:
            Dictionary with Sine Matrix arrays
        """
        if not self.is_initialized:
            self.setup()

        results = {}

        # Check if structure is periodic
        if not any(atoms.pbc):
            logger.warning("Sine Matrix is designed for periodic structures. "
                         "Structure has no periodic boundaries set.")

        # Select atoms based on selection criteria
        selected_atoms = self._select_atoms(atoms, atom_selection, **kwargs)

        if selected_atoms is None:
            logger.warning(f"No atoms selected for selection type: {atom_selection}")
            return results

        # Check if structure is too large
        n_atoms = len(selected_atoms)
        if n_atoms > self.n_atoms_max:
            logger.warning(f"Structure has {n_atoms} atoms, exceeding n_atoms_max={self.n_atoms_max}")
            return results

        try:
            # Create Sine Matrix
            sm = self.descriptor.create(selected_atoms)

            # Process based on output type
            if self.permutation == 'eigenspectrum':
                # Already 1D array of eigenvalues
                results[f"sm_{atom_selection}"] = sm
            else:
                # Matrix output
                if self.sparse:
                    # Convert sparse to dense for consistency
                    sm_dense = sm.toarray() if hasattr(sm, 'toarray') else sm
                else:
                    sm_dense = sm

                # Flatten if requested
                if self.flatten:
                    results[f"sm_{atom_selection}"] = sm_dense.flatten()
                else:
                    results[f"sm_{atom_selection}"] = sm_dense

            # Add metadata
            results[f"sm_{atom_selection}_natoms"] = np.array([n_atoms])
            results[f"sm_{atom_selection}_periodic"] = np.array(selected_atoms.pbc)

        except Exception as e:
            logger.error(f"Failed to create Sine Matrix: {e}")

        return results

    def extract_batch(
        self,
        structures: List[Atoms],
        tasknames: Optional[List[str]] = None,
        n_jobs: Optional[int] = None,
        check_periodicity: bool = True,
        **kwargs
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Extract Sine Matrices for multiple structures.

        Args:
            structures: List of ASE Atoms objects
            tasknames: Optional list of tasknames
            n_jobs: Number of parallel jobs (overrides config)
            check_periodicity: Whether to warn about non-periodic structures
            **kwargs: Additional parameters for extraction

        Returns:
            Dictionary mapping tasknames to SM representations
        """
        if not self.is_initialized:
            self.setup()

        if tasknames is None:
            tasknames = [f"struct_{i}" for i in range(len(structures))]

        # Check periodicity if requested
        if check_periodicity:
            non_periodic = []
            for i, atoms in enumerate(structures):
                if not any(atoms.pbc):
                    non_periodic.append(tasknames[i])

            if non_periodic:
                logger.warning(f"Following structures are non-periodic: {non_periodic[:5]}...")

        # Use parallel processing if available
        n_jobs = n_jobs or self.config.n_jobs

        if n_jobs != 1 and len(structures) > 1:
            # Create descriptors in parallel
            try:
                sm_arrays = self.descriptor.create(
                    structures,
                    n_jobs=n_jobs,
                    only_physical_cores=False,
                    verbose=False
                )

                # Package results
                results = {}
                for i, (taskname, sm) in enumerate(zip(tasknames, sm_arrays)):
                    results[taskname] = {
                        f"sm_all": sm.flatten() if self.flatten else sm,
                        f"sm_all_natoms": np.array([len(structures[i])]),
                        f"sm_all_periodic": np.array(structures[i].pbc)
                    }

                return results

            except Exception as e:
                logger.warning(f"Parallel processing failed, falling back to serial: {e}")

        # Serial processing fallback
        return super().extract_batch(structures, tasknames, **kwargs)

    @property
    def descriptor_name(self) -> str:
        """Return descriptor name."""
        return "sine_matrix"

    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        if self.permutation == 'eigenspectrum':
            return [f"eigenvalue_{i}" for i in range(self.n_atoms_max)]
        else:
            if self.flatten:
                return [f"sm_element_{i}" for i in range(self.n_atoms_max ** 2)]
            else:
                return [f"sm_matrix"]

    def get_feature_dimensions(self) -> Dict[str, Union[int, tuple]]:
        """Return feature dimensions."""
        if self.permutation == 'eigenspectrum':
            return {"eigenspectrum": self.n_atoms_max}
        else:
            if self.flatten:
                return {"flattened_matrix": self.n_atoms_max ** 2}
            else:
                return {"matrix": (self.n_atoms_max, self.n_atoms_max)}

    def validate_structures(
        self,
        structures: List[Atoms],
        check_periodicity: bool = True
    ) -> List[bool]:
        """
        Validate structures for SM extraction.

        Args:
            structures: List of ASE Atoms objects
            check_periodicity: Whether to check for periodic boundaries

        Returns:
            List of booleans indicating validity
        """
        valid = []
        for atoms in structures:
            n_atoms = len(atoms)

            # Check size
            if n_atoms > self.n_atoms_max:
                logger.warning(f"Structure has {n_atoms} atoms, exceeds n_atoms_max={self.n_atoms_max}")
                valid.append(False)
            # Check periodicity if requested
            elif check_periodicity and not any(atoms.pbc):
                logger.warning(f"Structure is non-periodic, Sine Matrix works best with periodic structures")
                valid.append(True)  # Still valid, just warning
            else:
                valid.append(True)

        return valid

    def get_descriptor_info(self) -> Dict:
        """Get information about the descriptor configuration."""
        return {
            "name": self.descriptor_name,
            "n_atoms_max": self.n_atoms_max,
            "permutation": self.permutation,
            "sparse": self.sparse,
            "flatten": self.flatten,
            "dtype": self.dtype,
            "feature_size": self.get_feature_dimensions(),
            "sigma": self.sigma if self.permutation == "random" else None,
            "seed": self.seed if self.permutation == "random" else None,
            "designed_for": "periodic structures"
        }

    def compare_with_coulomb(
        self,
        atoms: Atoms
    ) -> Dict[str, np.ndarray]:
        """
        Compare Sine Matrix with Coulomb Matrix for analysis.
        Useful for understanding the differences between periodic and non-periodic descriptors.

        Args:
            atoms: ASE Atoms object

        Returns:
            Dictionary with both Sine and Coulomb matrices
        """
        try:
            from dscribe.descriptors import CoulombMatrix

            # Create Sine Matrix
            sm_result = self.extract_single(atoms)

            # Create Coulomb Matrix with same settings
            cm = CoulombMatrix(
                n_atoms_max=self.n_atoms_max,
                permutation=self.permutation,
                sigma=self.sigma,
                seed=self.seed,
                sparse=self.sparse
            )

            cm_matrix = cm.create(atoms)

            # Flatten if needed to match Sine Matrix
            if self.flatten and self.permutation != 'eigenspectrum':
                cm_matrix = cm_matrix.flatten()

            return {
                "sine_matrix": sm_result.get(f"sm_all", np.array([])),
                "coulomb_matrix": cm_matrix,
                "difference": sm_result.get(f"sm_all", np.array([])) - cm_matrix if sm_result else np.array([])
            }

        except ImportError:
            logger.warning("CoulombMatrix not available for comparison")
            return {"sine_matrix": sm_result.get(f"sm_all", np.array([]))}
        except Exception as e:
            logger.error(f"Failed to compare with Coulomb Matrix: {e}")
            return {}