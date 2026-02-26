"""
Simple Sine Matrix implementation without dscribe dependency.
Direct implementation for periodic systems.
"""

import numpy as np
from typing import Dict, List, Optional
from ase import Atoms
from ..base import PhysicsInspiredExtractor, ExtractionConfig
import logging

logger = logging.getLogger(__name__)


class SimpleSineMatrixExtractor(PhysicsInspiredExtractor):
    """
    Direct Sine Matrix implementation without dscribe.

    The Sine Matrix is designed for periodic systems and uses:
    SM_ij = Z_i * Z_j * sin(π * r_ij / r_cut) / r_ij  for i != j
    SM_ii = 0.5 * Z_i^2.4  for i == j
    """

    def __init__(
        self,
        n_atoms_max: int = 50,
        r_cut: float = 10.0,
        permutation: str = 'sorted_l2',
        flatten: bool = False,
        config: Optional[ExtractionConfig] = None
    ):
        """
        Initialize Simple Sine Matrix extractor.

        Args:
            n_atoms_max: Maximum number of atoms
            r_cut: Cutoff radius for periodic distance calculations
            permutation: Sorting method ('none', 'sorted_l2', 'eigenspectrum')
            flatten: Whether to flatten the output matrix
            config: Extraction configuration
        """
        super().__init__(config)
        self.n_atoms_max = n_atoms_max
        self.r_cut = r_cut
        self.permutation = permutation
        self.flatten = flatten
        self.is_initialized = True  # No setup needed

    def setup(self) -> None:
        """No setup needed for simple implementation."""
        self.is_initialized = True

    def _compute_sine_matrix(self, atoms: Atoms) -> np.ndarray:
        """
        Compute Sine Matrix for a periodic structure.

        Args:
            atoms: ASE Atoms object

        Returns:
            Sine Matrix as numpy array
        """
        n_atoms = len(atoms)
        positions = atoms.positions
        atomic_numbers = atoms.get_atomic_numbers()

        # Initialize matrix
        matrix = np.zeros((n_atoms, n_atoms))

        # Get cell for periodic boundary conditions
        if any(atoms.pbc):
            cell = atoms.cell
        else:
            # If no PBC, use large dummy cell
            cell = np.eye(3) * 1000

        # Fill matrix elements
        for i in range(n_atoms):
            for j in range(n_atoms):
                if i == j:
                    # Diagonal elements
                    matrix[i, j] = 0.5 * atomic_numbers[i] ** 2.4
                else:
                    # Off-diagonal elements with periodic distance
                    # Calculate distance considering PBC
                    diff = positions[j] - positions[i]

                    # Apply minimum image convention for periodic boundaries
                    if any(atoms.pbc):
                        # Convert to fractional coordinates
                        frac_diff = np.linalg.solve(cell.T, diff)
                        # Apply PBC
                        for idx, periodic in enumerate(atoms.pbc):
                            if periodic:
                                frac_diff[idx] -= np.round(frac_diff[idx])
                        # Convert back to Cartesian
                        diff = cell.T @ frac_diff

                    r_ij = np.linalg.norm(diff)

                    if r_ij < self.r_cut and r_ij > 0:
                        # Sine transformation for periodic systems
                        matrix[i, j] = (atomic_numbers[i] * atomic_numbers[j] *
                                      np.sin(np.pi * r_ij / self.r_cut) / r_ij)
                    else:
                        matrix[i, j] = 0.0

        return matrix

    def _sort_matrix(self, matrix: np.ndarray, method: str) -> np.ndarray:
        """
        Apply permutation/sorting to the matrix.

        Args:
            matrix: Input matrix
            method: Sorting method

        Returns:
            Sorted/permuted matrix or eigenvalues
        """
        if method == 'none':
            return matrix

        elif method == 'sorted_l2':
            # Sort by L2 norm of rows
            norms = np.linalg.norm(matrix, axis=1)
            sorted_indices = np.argsort(norms)[::-1]  # Descending order
            # Sort rows and columns
            sorted_matrix = matrix[sorted_indices][:, sorted_indices]
            return sorted_matrix

        elif method == 'eigenspectrum':
            # Return sorted eigenvalues
            eigenvalues = np.linalg.eigvalsh(matrix)
            return np.sort(eigenvalues)[::-1]  # Descending order

        else:
            logger.warning(f"Unknown permutation method: {method}")
            return matrix

    def extract_single(
        self,
        atoms: Atoms,
        atom_selection: str = "all",
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Extract Sine Matrix for a single structure.

        Args:
            atoms: ASE Atoms object
            atom_selection: Atom selection mode
            **kwargs: Additional parameters

        Returns:
            Dictionary with Sine Matrix
        """
        results = {}

        # Check if structure is periodic
        if not any(atoms.pbc):
            logger.warning("Sine Matrix is designed for periodic structures")

        # Select atoms
        selected_atoms = self._select_atoms(atoms, atom_selection, **kwargs)

        if selected_atoms is None:
            logger.warning(f"No atoms selected for selection: {atom_selection}")
            return results

        n_atoms = len(selected_atoms)
        if n_atoms > self.n_atoms_max:
            logger.warning(f"Structure has {n_atoms} atoms, exceeds max {self.n_atoms_max}")
            return results

        try:
            # Compute Sine Matrix
            sm = self._compute_sine_matrix(selected_atoms)

            # Pad to n_atoms_max if needed
            if n_atoms < self.n_atoms_max:
                padded_sm = np.zeros((self.n_atoms_max, self.n_atoms_max))
                padded_sm[:n_atoms, :n_atoms] = sm
                sm = padded_sm

            # Apply permutation
            if self.permutation == 'eigenspectrum':
                sm = self._sort_matrix(sm[:n_atoms, :n_atoms], self.permutation)
                # Pad eigenvalues
                if len(sm) < self.n_atoms_max:
                    sm = np.pad(sm, (0, self.n_atoms_max - len(sm)), 'constant')
                results[f"sm_{atom_selection}"] = sm
            else:
                sm = self._sort_matrix(sm, self.permutation)

                if self.flatten:
                    results[f"sm_{atom_selection}"] = sm.flatten()
                else:
                    results[f"sm_{atom_selection}"] = sm

            # Add metadata
            results[f"sm_{atom_selection}_natoms"] = np.array([n_atoms])
            results[f"sm_{atom_selection}_periodic"] = np.array(selected_atoms.pbc)

        except Exception as e:
            logger.error(f"Failed to create Sine Matrix: {e}")

        return results

    @property
    def descriptor_name(self) -> str:
        """Return descriptor name."""
        return "simple_sine_matrix"

    def get_feature_names(self) -> List[str]:
        """Return feature names."""
        if self.permutation == 'eigenspectrum':
            return [f"eigenvalue_{i}" for i in range(self.n_atoms_max)]
        else:
            if self.flatten:
                return [f"sm_element_{i}" for i in range(self.n_atoms_max ** 2)]
            else:
                return [f"sm_matrix"]

    def get_feature_dimensions(self) -> Dict:
        """Return feature dimensions."""
        if self.permutation == 'eigenspectrum':
            return {"eigenspectrum": self.n_atoms_max}
        else:
            if self.flatten:
                return {"flattened_matrix": self.n_atoms_max ** 2}
            else:
                return {"matrix": (self.n_atoms_max, self.n_atoms_max)}

    def get_descriptor_info(self) -> Dict:
        """Get descriptor configuration info."""
        return {
            "name": self.descriptor_name,
            "n_atoms_max": self.n_atoms_max,
            "r_cut": self.r_cut,
            "permutation": self.permutation,
            "flatten": self.flatten,
            "feature_size": self.get_feature_dimensions(),
            "implementation": "simple (no dscribe)"
        }