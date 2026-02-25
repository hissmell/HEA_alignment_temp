"""
Ewald Sum Matrix descriptor for periodic atomic structures.
Based on dscribe implementation, optimized for periodic crystals with electrostatic interactions.
"""

import numpy as np
from typing import Dict, List, Optional, Union
from pathlib import Path
import logging

from ase import Atoms
from dscribe.descriptors import EwaldSumMatrix

from ..base import PhysicsInspiredExtractor, ExtractionConfig

logger = logging.getLogger(__name__)


class EwaldSumMatrixExtractor(PhysicsInspiredExtractor):
    """
    Ewald Sum Matrix descriptor extractor for periodic systems.
    
    The Ewald Sum Matrix is an extension of the Coulomb Matrix for periodic
    systems, modeling interactions between atoms in periodic crystals through
    electrostatic interactions using the Ewald summation technique.
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
        Initialize Ewald Sum Matrix extractor.
        
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
        
        # Default Ewald parameters
        self.accuracy = 1e-5
        self.weight = 1.0
        self.r_cut = None
        self.g_cut = None
        self.a = None
        
        # Validate parameters
        if permutation == 'random' and sigma is None:
            raise ValueError("sigma must be provided when using random permutation")
        
        if permutation not in ['none', 'sorted_l2', 'eigenspectrum', 'random']:
            raise ValueError(f"Invalid permutation method: {permutation}")
        
        if dtype not in ['float32', 'float64']:
            raise ValueError(f"Invalid dtype: {dtype}. Must be 'float32' or 'float64'")
    
    def setup(self) -> None:
        """Initialize the Ewald Sum Matrix descriptor."""
        try:
            self.descriptor = EwaldSumMatrix(
                n_atoms_max=self.n_atoms_max,
                permutation=self.permutation,
                sigma=self.sigma,
                seed=self.seed,
                sparse=self.sparse,
                dtype=self.dtype
            )
            self.is_initialized = True
            logger.info(f"Initialized EwaldSumMatrix with n_atoms_max={self.n_atoms_max}, "
                       f"permutation={self.permutation}")
        except Exception as e:
            logger.error(f"Failed to initialize EwaldSumMatrix: {e}")
            raise
    
    def extract_single(
        self,
        atoms: Atoms,
        atom_selection: str = "all",
        accuracy: Optional[float] = None,
        w: Optional[float] = None,
        r_cut: Optional[float] = None,
        g_cut: Optional[float] = None,
        a: Optional[float] = None,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Extract Ewald Sum Matrix for a single periodic structure.
        
        Args:
            atoms: ASE Atoms object (must have periodic boundary conditions)
            atom_selection: Atom selection mode ("all", "slab", "site")
            accuracy: Convergence accuracy (default: 1e-5)
            w: Weight parameter for real/reciprocal space balance
            r_cut: Real space cutoff radius
            g_cut: Reciprocal space cutoff radius
            a: Screening parameter (Gaussian width)
            **kwargs: Additional parameters (adsorbate_indices, site_cutoff)
        
        Returns:
            Dictionary with Ewald Sum Matrix arrays
        """
        if not self.is_initialized:
            self.setup()
        
        results = {}
        
        # Check if structure is periodic
        if not any(atoms.pbc):
            logger.error("Ewald Sum Matrix requires periodic structures. "
                        "Structure has no periodic boundaries set.")
            return results
        
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
            # Use provided parameters or defaults
            accuracy = accuracy or self.accuracy
            w = w or self.weight
            
            # Create Ewald Sum Matrix
            esm = self.descriptor.create(
                selected_atoms,
                accuracy=accuracy,
                w=w,
                r_cut=r_cut,
                g_cut=g_cut,
                a=a
            )
            
            # Process based on output type
            if self.permutation == 'eigenspectrum':
                # Already 1D array of eigenvalues
                results[f"esm_{atom_selection}"] = esm
            else:
                # Matrix output
                if self.sparse:
                    # Convert sparse to dense for consistency
                    esm_dense = esm.toarray() if hasattr(esm, 'toarray') else esm
                else:
                    esm_dense = esm
                
                # Flatten if requested
                if self.flatten:
                    results[f"esm_{atom_selection}"] = esm_dense.flatten()
                else:
                    results[f"esm_{atom_selection}"] = esm_dense
            
            # Add metadata
            results[f"esm_{atom_selection}_natoms"] = np.array([n_atoms])
            results[f"esm_{atom_selection}_periodic"] = np.array(selected_atoms.pbc)
            results[f"esm_{atom_selection}_cell"] = selected_atoms.cell.array.flatten()
            
        except Exception as e:
            logger.error(f"Failed to create Ewald Sum Matrix: {e}")
        
        return results
    
    def extract_batch(
        self,
        structures: List[Atoms],
        tasknames: Optional[List[str]] = None,
        n_jobs: Optional[int] = None,
        accuracy: Optional[float] = None,
        w: Optional[float] = None,
        r_cut: Optional[float] = None,
        g_cut: Optional[float] = None,
        a: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Extract Ewald Sum Matrices for multiple structures.
        
        Args:
            structures: List of ASE Atoms objects
            tasknames: Optional list of tasknames
            n_jobs: Number of parallel jobs (overrides config)
            accuracy: Convergence accuracy for all structures
            w: Weight parameter for all structures
            r_cut: Real space cutoff for all structures
            g_cut: Reciprocal space cutoff for all structures  
            a: Screening parameter for all structures
            **kwargs: Additional parameters for extraction
        
        Returns:
            Dictionary mapping tasknames to ESM representations
        """
        if not self.is_initialized:
            self.setup()
        
        if tasknames is None:
            tasknames = [f"struct_{i}" for i in range(len(structures))]
        
        # Check periodicity
        non_periodic = []
        for i, atoms in enumerate(structures):
            if not any(atoms.pbc):
                non_periodic.append(tasknames[i])
        
        if non_periodic:
            logger.error(f"Following structures are non-periodic (required for ESM): {non_periodic[:5]}...")
            # Filter out non-periodic structures
            valid_pairs = [(s, t) for s, t in zip(structures, tasknames) if any(s.pbc)]
            if not valid_pairs:
                return {}
            structures, tasknames = zip(*valid_pairs)
            structures = list(structures)
            tasknames = list(tasknames)
        
        # Use parallel processing if available
        n_jobs = n_jobs or self.config.n_jobs
        
        # Use provided parameters or defaults
        accuracy = accuracy or self.accuracy
        w = w or self.weight
        
        if n_jobs != 1 and len(structures) > 1:
            # Create descriptors in parallel
            try:
                esm_arrays = self.descriptor.create(
                    structures,
                    accuracy=accuracy,
                    w=w,
                    r_cut=r_cut,
                    g_cut=g_cut,
                    a=a,
                    n_jobs=n_jobs,
                    only_physical_cores=False,
                    verbose=False
                )
                
                # Package results
                results = {}
                for i, (taskname, esm) in enumerate(zip(tasknames, esm_arrays)):
                    results[taskname] = {
                        f"esm_all": esm.flatten() if self.flatten else esm,
                        f"esm_all_natoms": np.array([len(structures[i])]),
                        f"esm_all_periodic": np.array(structures[i].pbc),
                        f"esm_all_cell": structures[i].cell.array.flatten()
                    }
                
                return results
                
            except Exception as e:
                logger.warning(f"Parallel processing failed, falling back to serial: {e}")
        
        # Serial processing fallback
        results = {}
        for struct, name in zip(structures, tasknames):
            result = self.extract_single(
                struct, 
                accuracy=accuracy,
                w=w,
                r_cut=r_cut,
                g_cut=g_cut,
                a=a,
                **kwargs
            )
            if result:
                results[name] = result
        
        return results
    
    @property
    def descriptor_name(self) -> str:
        """Return descriptor name."""
        return "ewald_sum_matrix"
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        if self.permutation == 'eigenspectrum':
            return [f"eigenvalue_{i}" for i in range(self.n_atoms_max)]
        else:
            if self.flatten:
                return [f"esm_element_{i}" for i in range(self.n_atoms_max ** 2)]
            else:
                return [f"esm_matrix"]
    
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
        check_cell: bool = True
    ) -> List[bool]:
        """
        Validate structures for ESM extraction.
        
        Args:
            structures: List of ASE Atoms objects
            check_cell: Whether to check for valid unit cell
        
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
            # Check periodicity (required for ESM)
            elif not any(atoms.pbc):
                logger.error(f"Structure is non-periodic, Ewald Sum Matrix requires periodic structures")
                valid.append(False)
            # Check for valid unit cell if requested
            elif check_cell and np.linalg.det(atoms.cell.array) == 0:
                logger.error(f"Structure has invalid unit cell (determinant = 0)")
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
            "dtype": self.dtype,
            "feature_size": self.get_feature_dimensions(),
            "accuracy": self.accuracy,
            "weight": self.weight,
            "sigma": self.sigma if self.permutation == "random" else None,
            "seed": self.seed if self.permutation == "random" else None,
            "designed_for": "periodic structures only",
            "method": "Ewald summation for electrostatic interactions"
        }
    
    def compare_descriptors(
        self,
        atoms: Atoms,
        compare_with: List[str] = ['coulomb', 'sine']
    ) -> Dict[str, np.ndarray]:
        """
        Compare Ewald Sum Matrix with other matrix descriptors.
        
        Args:
            atoms: ASE Atoms object (must be periodic)
            compare_with: List of descriptors to compare ('coulomb', 'sine')
        
        Returns:
            Dictionary with different matrix descriptors
        """
        results = {}
        
        # Get Ewald Sum Matrix
        esm_result = self.extract_single(atoms)
        if esm_result:
            results["ewald_sum_matrix"] = esm_result.get(f"esm_all", np.array([]))
        
        # Compare with other descriptors if requested
        if 'coulomb' in compare_with:
            try:
                from .coulomb_matrix import CoulombMatrixExtractor
                cm_ext = CoulombMatrixExtractor(
                    n_atoms_max=self.n_atoms_max,
                    permutation=self.permutation,
                    flatten=self.flatten,
                    sparse=self.sparse,
                    dtype=self.dtype
                )
                cm_result = cm_ext.extract_single(atoms)
                if cm_result:
                    results["coulomb_matrix"] = cm_result.get(f"cm_all", np.array([]))
            except Exception as e:
                logger.warning(f"Could not compare with Coulomb Matrix: {e}")
        
        if 'sine' in compare_with:
            try:
                from .sine_matrix import SineMatrixExtractor
                sm_ext = SineMatrixExtractor(
                    n_atoms_max=self.n_atoms_max,
                    permutation=self.permutation,
                    flatten=self.flatten,
                    sparse=self.sparse,
                    dtype=self.dtype
                )
                sm_result = sm_ext.extract_single(atoms)
                if sm_result:
                    results["sine_matrix"] = sm_result.get(f"sm_all", np.array([]))
            except Exception as e:
                logger.warning(f"Could not compare with Sine Matrix: {e}")
        
        return results
    
    def set_ewald_parameters(
        self,
        accuracy: Optional[float] = None,
        weight: Optional[float] = None,
        r_cut: Optional[float] = None,
        g_cut: Optional[float] = None,
        a: Optional[float] = None
    ) -> None:
        """
        Set default Ewald parameters for subsequent extractions.
        
        Args:
            accuracy: Convergence accuracy
            weight: Computational balance parameter
            r_cut: Real space cutoff
            g_cut: Reciprocal space cutoff
            a: Screening parameter
        """
        if accuracy is not None:
            self.accuracy = accuracy
        if weight is not None:
            self.weight = weight
        if r_cut is not None:
            self.r_cut = r_cut
        if g_cut is not None:
            self.g_cut = g_cut
        if a is not None:
            self.a = a
        
        logger.info(f"Updated Ewald parameters: accuracy={self.accuracy}, "
                   f"weight={self.weight}, r_cut={self.r_cut}, "
                   f"g_cut={self.g_cut}, a={self.a}")
