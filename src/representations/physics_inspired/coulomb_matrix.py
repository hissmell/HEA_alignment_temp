"""
Coulomb Matrix descriptor for atomic structures.
Based on dscribe implementation with optimizations for materials science applications.
"""

import numpy as np
from typing import Dict, List, Optional, Union
from pathlib import Path
import logging

from ase import Atoms
from dscribe.descriptors import CoulombMatrix

from ..base import PhysicsInspiredExtractor, ExtractionConfig

logger = logging.getLogger(__name__)


class CoulombMatrixExtractor(PhysicsInspiredExtractor):
    """
    Coulomb Matrix descriptor extractor.

    The Coulomb matrix is a simple molecular descriptor that encodes
    atomic structures based on Coulombic interactions between atoms.
    """

    def __init__(
        self,
        n_atoms_max: int = 200,
        permutation: str = 'sorted_l2',
        sigma: Optional[float] = None,
        seed: Optional[int] = None,
        sparse: bool = False,
        flatten: bool = True,
        config: Optional[ExtractionConfig] = None
    ):
        """
        Initialize Coulomb Matrix extractor.

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
            config: Extraction configuration
        """
        super().__init__(config)

        self.n_atoms_max = n_atoms_max
        self.permutation = permutation
        self.sigma = sigma
        self.seed = seed
        self.sparse = sparse
        self.flatten = flatten

        # Validate parameters
        if permutation == 'random' and sigma is None:
            raise ValueError("sigma must be provided when using random permutation")

        if permutation not in ['none', 'sorted_l2', 'eigenspectrum', 'random']:
            raise ValueError(f"Invalid permutation method: {permutation}")

    def setup(self) -> None:
        """Initialize the Coulomb Matrix descriptor."""
        try:
            self.descriptor = CoulombMatrix(
                n_atoms_max=self.n_atoms_max,
                permutation=self.permutation,
                sigma=self.sigma,
                seed=self.seed,
                sparse=self.sparse
            )
            self.is_initialized = True
            logger.info(f"Initialized CoulombMatrix with n_atoms_max={self.n_atoms_max}, "
                       f"permutation={self.permutation}")
        except Exception as e:
            logger.error(f"Failed to initialize CoulombMatrix: {e}")
            raise

    def extract_single(
        self,
        atoms: Atoms,
        atom_selection: str = "all",
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Extract Coulomb Matrix for a single structure.

        Args:
            atoms: ASE Atoms object
            atom_selection: Atom selection mode ("all", "slab", "site")
            **kwargs: Additional parameters (adsorbate_indices, site_cutoff)

        Returns:
            Dictionary with Coulomb Matrix arrays
        """
        if not self.is_initialized:
            self.setup()

        results = {}

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
            # Create Coulomb Matrix
            cm = self.descriptor.create(selected_atoms)

            # Process based on output type
            if self.permutation == 'eigenspectrum':
                # Already 1D array of eigenvalues
                results[f"cm_{atom_selection}"] = cm
            else:
                # Matrix output
                if self.sparse:
                    # Convert sparse to dense for consistency
                    cm_dense = cm.toarray() if hasattr(cm, 'toarray') else cm
                else:
                    cm_dense = cm

                # Flatten if requested
                if self.flatten:
                    results[f"cm_{atom_selection}"] = cm_dense.flatten()
                else:
                    results[f"cm_{atom_selection}"] = cm_dense

            # Add metadata
            results[f"cm_{atom_selection}_natoms"] = np.array([n_atoms])

        except Exception as e:
            logger.error(f"Failed to create Coulomb Matrix: {e}")

        return results

    def extract_batch(
        self,
        structures: List[Atoms],
        tasknames: Optional[List[str]] = None,
        n_jobs: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Extract Coulomb Matrices for multiple structures.

        Args:
            structures: List of ASE Atoms objects
            tasknames: Optional list of tasknames
            n_jobs: Number of parallel jobs (overrides config)
            **kwargs: Additional parameters for extraction

        Returns:
            Dictionary mapping tasknames to CM representations
        """
        if not self.is_initialized:
            self.setup()

        if tasknames is None:
            tasknames = [f"struct_{i}" for i in range(len(structures))]

        # Use parallel processing if available
        n_jobs = n_jobs or self.config.n_jobs

        if n_jobs != 1 and len(structures) > 1:
            # Create descriptors in parallel
            try:
                cm_arrays = self.descriptor.create(
                    structures,
                    n_jobs=n_jobs,
                    only_physical_cores=False,
                    verbose=False
                )

                # Package results
                results = {}
                for i, (taskname, cm) in enumerate(zip(tasknames, cm_arrays)):
                    results[taskname] = {
                        f"cm_all": cm.flatten() if self.flatten else cm,
                        f"cm_all_natoms": np.array([len(structures[i])])
                    }

                return results

            except Exception as e:
                logger.warning(f"Parallel processing failed, falling back to serial: {e}")

        # Serial processing fallback
        return super().extract_batch(structures, tasknames, **kwargs)

    @property
    def descriptor_name(self) -> str:
        """Return descriptor name."""
        return "coulomb_matrix"

    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        if self.permutation == 'eigenspectrum':
            return [f"eigenvalue_{i}" for i in range(self.n_atoms_max)]
        else:
            if self.flatten:
                return [f"cm_element_{i}" for i in range(self.n_atoms_max ** 2)]
            else:
                return [f"cm_matrix"]

    def get_feature_dimensions(self) -> Dict[str, int]:
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
        structures: List[Atoms]
    ) -> List[bool]:
        """
        Validate structures for CM extraction.

        Args:
            structures: List of ASE Atoms objects

        Returns:
            List of booleans indicating validity
        """
        valid = []
        for atoms in structures:
            n_atoms = len(atoms)
            if n_atoms > self.n_atoms_max:
                logger.warning(f"Structure has {n_atoms} atoms, exceeds n_atoms_max={self.n_atoms_max}")
                valid.append(False)
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
            "feature_size": self.get_feature_dimensions(),
            "sigma": self.sigma if self.permutation == "random" else None,
            "seed": self.seed if self.permutation == "random" else None
        }