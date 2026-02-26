"""
Many-body Tensor Representation (MBTR) descriptor for atomic structures.
Based on dscribe implementation, suitable for both finite and periodic systems.
"""

import numpy as np
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import logging

from ase import Atoms
from dscribe.descriptors import MBTR

from ..base import PhysicsInspiredExtractor, ExtractionConfig

logger = logging.getLogger(__name__)


class MBTRConfig:
    """Configuration for MBTR descriptor."""
    
    def __init__(
        self,
        k1: Optional[Dict[str, Any]] = None,
        k2: Optional[Dict[str, Any]] = None,
        k3: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize MBTR configuration for different k-terms.
        
        Args:
            k1: Configuration for 1-body terms (atomic numbers)
                Example: {"geometry": {"function": "atomic_number"}, 
                         "grid": {"min": 0, "max": 100, "n": 100, "sigma": 0.1}}
            k2: Configuration for 2-body terms (distances)
                Example: {"geometry": {"function": "inverse_distance"}, 
                         "grid": {"min": 0, "max": 1, "n": 100, "sigma": 0.1},
                         "weighting": {"function": "exp", "scale": 0.5, "threshold": 1e-3}}
            k3: Configuration for 3-body terms (angles)
                Example: {"geometry": {"function": "cosine"},
                         "grid": {"min": -1, "max": 1, "n": 100, "sigma": 0.1},
                         "weighting": {"function": "exp", "scale": 0.5, "threshold": 1e-3}}
        """
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        
        # Validate that at least one k-term is specified
        if not any([k1, k2, k3]):
            raise ValueError("At least one k-term (k1, k2, or k3) must be specified")


class MBTRExtractor(PhysicsInspiredExtractor):
    """
    Many-body Tensor Representation descriptor extractor.
    
    MBTR encodes structures using distributions of structural motifs,
    making it suitable for interpretable machine learning applications.
    """
    
    def __init__(
        self,
        species: List[Union[str, int]],
        geometry: Optional[Dict[str, Any]] = None,
        grid: Optional[Dict[str, Any]] = None,
        weighting: Optional[Dict[str, Any]] = None,
        mbtr_config: Optional[MBTRConfig] = None,
        periodic: bool = False,
        normalization: str = 'l2',
        normalize_gaussians: bool = True,
        sparse: bool = False,
        dtype: str = 'float64',
        config: Optional[ExtractionConfig] = None
    ):
        """
        Initialize MBTR extractor.
        
        Args:
            species: List of chemical species (symbols or atomic numbers)
            geometry: Geometry function configuration (for simple single k-term setup)
            grid: Grid discretization parameters (for simple single k-term setup)
            weighting: Weighting function configuration (for simple single k-term setup)
            mbtr_config: Advanced configuration for multiple k-terms
            periodic: Whether to respect periodic boundaries
            normalization: Normalization method ('none', 'l2', 'n_atoms', 'valle_oganov')
            normalize_gaussians: Whether to normalize gaussians to area of 1
            sparse: Whether to use sparse matrix representation
            dtype: Data type for the output ('float32' or 'float64')
            config: Extraction configuration
        """
        super().__init__(config)
        
        self.species = species
        self.periodic = periodic
        self.normalization = normalization
        self.normalize_gaussians = normalize_gaussians
        self.sparse = sparse
        self.dtype = dtype
        
        # Handle configuration
        if mbtr_config is not None:
            # Use advanced multi-k configuration
            self.mbtr_config = mbtr_config
            self.geometry = None
            self.grid = None
            self.weighting = None
        else:
            # Use simple single-k configuration
            if geometry is None:
                # Default to k2 with inverse distances
                geometry = {"function": "inverse_distance"}
            if grid is None:
                grid = {"min": 0, "max": 1, "n": 100, "sigma": 0.1}
            if weighting is None:
                weighting = {"function": "exp", "scale": 0.5, "threshold": 1e-3}
            
            self.geometry = geometry
            self.grid = grid
            self.weighting = weighting
            self.mbtr_config = None
        
        # Validate parameters
        if normalization not in ['none', 'l2', 'n_atoms', 'valle_oganov']:
            raise ValueError(f"Invalid normalization method: {normalization}")
        
        if dtype not in ['float32', 'float64']:
            raise ValueError(f"Invalid dtype: {dtype}. Must be 'float32' or 'float64'")
    
    def setup(self) -> None:
        """Initialize the MBTR descriptor."""
        try:
            # Monkey patch dscribe's System.from_atoms to avoid _get_constraints issue
            from dscribe.core.system import System

            def patched_from_atoms(atoms):
                """Patched version that doesn't call _get_constraints"""
                system = System(
                    symbols=atoms.get_chemical_symbols(),
                    positions=atoms.get_positions(),
                    cell=atoms.get_cell(),
                    pbc=atoms.get_pbc()
                )
                return system

            # Apply monkey patch
            System.from_atoms = staticmethod(patched_from_atoms)
            if self.mbtr_config is not None:
                # Setup with multiple k-terms
                kwargs = {}
                if self.mbtr_config.k1:
                    kwargs.update(self.mbtr_config.k1)
                if self.mbtr_config.k2:
                    kwargs.update(self.mbtr_config.k2)
                if self.mbtr_config.k3:
                    kwargs.update(self.mbtr_config.k3)
                
                self.descriptor = MBTR(
                    species=self.species,
                    periodic=self.periodic,
                    normalization=self.normalization,
                    normalize_gaussians=self.normalize_gaussians,
                    sparse=self.sparse,
                    dtype=self.dtype,
                    **kwargs
                )
            else:
                # Simple setup with single k-term
                self.descriptor = MBTR(
                    species=self.species,
                    geometry=self.geometry,
                    grid=self.grid,
                    weighting=self.weighting,
                    periodic=self.periodic,
                    normalization=self.normalization,
                    normalize_gaussians=self.normalize_gaussians,
                    sparse=self.sparse,
                    dtype=self.dtype
                )
            
            self.is_initialized = True
            logger.info(f"Initialized MBTR with species={self.species}, "
                       f"periodic={self.periodic}, normalization={self.normalization}")
        except Exception as e:
            logger.error(f"Failed to initialize MBTR: {e}")
            raise
    
    def extract_single(
        self,
        atoms: Atoms,
        atom_selection: str = "all",
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Extract MBTR for a single structure.
        
        Args:
            atoms: ASE Atoms object
            atom_selection: Atom selection mode ("all", "slab", "site")
            **kwargs: Additional parameters (adsorbate_indices, site_cutoff)
        
        Returns:
            Dictionary with MBTR arrays
        """
        if not self.is_initialized:
            self.setup()
        
        results = {}
        
        # Select atoms based on selection criteria
        selected_atoms = self._select_atoms(atoms, atom_selection, **kwargs)
        
        if selected_atoms is None:
            logger.warning(f"No atoms selected for selection type: {atom_selection}")
            return results
        
        try:
            # Create MBTR
            mbtr_output = self.descriptor.create(selected_atoms)
            
            # Process based on output type
            if self.sparse:
                # Convert sparse to dense for consistency
                mbtr_dense = mbtr_output.toarray() if hasattr(mbtr_output, 'toarray') else mbtr_output
            else:
                mbtr_dense = mbtr_output
            
            # Store results
            results[f"mbtr_{atom_selection}"] = mbtr_dense
            
            # Add metadata
            results[f"mbtr_{atom_selection}_natoms"] = np.array([len(selected_atoms)])
            results[f"mbtr_{atom_selection}_periodic"] = np.array(selected_atoms.pbc)
            
            # Add information about which k-terms are included
            k_terms = []
            if self.mbtr_config:
                if self.mbtr_config.k1:
                    k_terms.append(1)
                if self.mbtr_config.k2:
                    k_terms.append(2)
                if self.mbtr_config.k3:
                    k_terms.append(3)
            else:
                # Infer from geometry function
                geom_func = self.geometry.get("function", "")
                if "atomic_number" in geom_func:
                    k_terms.append(1)
                elif any(x in geom_func for x in ["distance", "inverse_distance"]):
                    k_terms.append(2)
                elif any(x in geom_func for x in ["angle", "cosine"]):
                    k_terms.append(3)
            
            results[f"mbtr_{atom_selection}_kterms"] = np.array(k_terms)
            
        except Exception as e:
            logger.error(f"Failed to create MBTR: {e}")
        
        return results
    
    def extract_batch(
        self,
        structures: List[Atoms],
        tasknames: Optional[List[str]] = None,
        n_jobs: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Extract MBTR for multiple structures.
        
        Args:
            structures: List of ASE Atoms objects
            tasknames: Optional list of tasknames
            n_jobs: Number of parallel jobs (overrides config)
            **kwargs: Additional parameters for extraction
        
        Returns:
            Dictionary mapping tasknames to MBTR representations
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
                mbtr_arrays = self.descriptor.create(
                    structures,
                    n_jobs=n_jobs,
                    only_physical_cores=False,
                    verbose=False
                )
                
                # Package results
                results = {}
                for i, (taskname, mbtr) in enumerate(zip(tasknames, mbtr_arrays)):
                    results[taskname] = {
                        f"mbtr_all": mbtr,
                        f"mbtr_all_natoms": np.array([len(structures[i])]),
                        f"mbtr_all_periodic": np.array(structures[i].pbc)
                    }
                
                return results
                
            except Exception as e:
                logger.warning(f"Parallel processing failed, falling back to serial: {e}")
        
        # Serial processing fallback
        return super().extract_batch(structures, tasknames, **kwargs)
    
    @property
    def descriptor_name(self) -> str:
        """Return descriptor name."""
        return "mbtr"
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        if not self.is_initialized:
            self.setup()
        
        n_features = self.descriptor.get_number_of_features()
        return [f"mbtr_feature_{i}" for i in range(n_features)]
    
    def get_feature_dimensions(self) -> Dict[str, Union[int, tuple]]:
        """Return feature dimensions."""
        if not self.is_initialized:
            self.setup()
        
        n_features = self.descriptor.get_number_of_features()
        return {"mbtr_vector": n_features}
    
    def validate_structures(
        self,
        structures: List[Atoms]
    ) -> List[bool]:
        """
        Validate structures for MBTR extraction.
        
        Args:
            structures: List of ASE Atoms objects
        
        Returns:
            List of booleans indicating validity
        """
        valid = []
        for atoms in structures:
            # Check if all species are in the defined list
            atom_symbols = set(atoms.get_chemical_symbols())
            defined_species = set(self.species if isinstance(self.species[0], str) 
                                else [str(s) for s in self.species])
            
            if not atom_symbols.issubset(defined_species):
                missing = atom_symbols - defined_species
                logger.warning(f"Structure contains undefined species: {missing}")
                valid.append(False)
            else:
                valid.append(True)
        
        return valid
    
    def get_descriptor_info(self) -> Dict:
        """Get information about the descriptor configuration."""
        info = {
            "name": self.descriptor_name,
            "species": self.species,
            "periodic": self.periodic,
            "normalization": self.normalization,
            "normalize_gaussians": self.normalize_gaussians,
            "sparse": self.sparse,
            "dtype": self.dtype,
        }
        
        if self.mbtr_config:
            k_info = {}
            if self.mbtr_config.k1:
                k_info["k1"] = self.mbtr_config.k1
            if self.mbtr_config.k2:
                k_info["k2"] = self.mbtr_config.k2
            if self.mbtr_config.k3:
                k_info["k3"] = self.mbtr_config.k3
            info["k_terms"] = k_info
        else:
            info["geometry"] = self.geometry
            info["grid"] = self.grid
            info["weighting"] = self.weighting
        
        if self.is_initialized:
            info["n_features"] = self.descriptor.get_number_of_features()
        
        return info
    
    def create_multi_k_descriptor(
        self,
        atoms: Atoms,
        include_k1: bool = True,
        include_k2: bool = True,
        include_k3: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Create MBTR with multiple k-terms for comprehensive structural encoding.
        
        Args:
            atoms: ASE Atoms object
            include_k1: Include 1-body terms (atomic numbers)
            include_k2: Include 2-body terms (distances)
            include_k3: Include 3-body terms (angles)
        
        Returns:
            Dictionary with MBTR outputs for different k-terms
        """
        results = {}
        
        # K1: Atomic numbers
        if include_k1:
            k1_config = MBTRConfig(
                k1={
                    "geometry": {"function": "atomic_number"},
                    "grid": {"min": 0, "max": 100, "n": 100, "sigma": 0.1},
                    "weighting": {"function": "unity"}  # k1 only supports unity weighting
                }
            )
            mbtr_k1 = MBTRExtractor(
                species=self.species,
                mbtr_config=k1_config,
                periodic=self.periodic,
                normalization=self.normalization,
                dtype=self.dtype
            )
            results.update(mbtr_k1.extract_single(atoms))
            results["mbtr_k1"] = results.pop("mbtr_all", np.array([]))
        
        # K2: Distances
        if include_k2:
            k2_config = MBTRConfig(
                k2={
                    "geometry": {"function": "inverse_distance"},
                    "grid": {"min": 0, "max": 1, "n": 100, "sigma": 0.1},
                    "weighting": {"function": "exp", "scale": 0.5, "threshold": 1e-3}
                }
            )
            mbtr_k2 = MBTRExtractor(
                species=self.species,
                mbtr_config=k2_config,
                periodic=self.periodic,
                normalization=self.normalization,
                dtype=self.dtype
            )
            k2_result = mbtr_k2.extract_single(atoms)
            results["mbtr_k2"] = k2_result.get("mbtr_all", np.array([]))
        
        # K3: Angles
        if include_k3:
            k3_config = MBTRConfig(
                k3={
                    "geometry": {"function": "cosine"},
                    "grid": {"min": -1, "max": 1, "n": 100, "sigma": 0.1},
                    "weighting": {"function": "exp", "scale": 0.5, "threshold": 1e-3}
                }
            )
            mbtr_k3 = MBTRExtractor(
                species=self.species,
                mbtr_config=k3_config,
                periodic=self.periodic,
                normalization=self.normalization,
                dtype=self.dtype
            )
            k3_result = mbtr_k3.extract_single(atoms)
            results["mbtr_k3"] = k3_result.get("mbtr_all", np.array([]))
        
        # Concatenate all k-terms if more than one is included
        included_terms = []
        if include_k1 and "mbtr_k1" in results:
            included_terms.append(results["mbtr_k1"])
        if include_k2 and "mbtr_k2" in results:
            included_terms.append(results["mbtr_k2"])
        if include_k3 and "mbtr_k3" in results:
            included_terms.append(results["mbtr_k3"])
        
        if len(included_terms) > 1:
            results["mbtr_combined"] = np.concatenate(included_terms)
        
        return results
    
    def compare_geometry_functions(
        self,
        atoms: Atoms,
        k_degree: int = 2
    ) -> Dict[str, np.ndarray]:
        """
        Compare different geometry functions for a given k-degree.
        
        Args:
            atoms: ASE Atoms object
            k_degree: Degree of MBTR (1, 2, or 3)
        
        Returns:
            Dictionary with MBTR outputs for different geometry functions
        """
        results = {}
        
        if k_degree == 1:
            # Only atomic_number for k1
            geometry_functions = [("atomic_number", {"min": 0, "max": 100})]
        elif k_degree == 2:
            # Distance and inverse distance for k2
            geometry_functions = [
                ("distance", {"min": 0, "max": 10}),
                ("inverse_distance", {"min": 0, "max": 2})
            ]
        elif k_degree == 3:
            # Angle and cosine for k3
            geometry_functions = [
                ("angle", {"min": 0, "max": 180}),
                ("cosine", {"min": -1, "max": 1})
            ]
        else:
            raise ValueError(f"Invalid k_degree: {k_degree}. Must be 1, 2, or 3")
        
        for geom_func, grid_range in geometry_functions:
            try:
                geometry = {"function": geom_func}
                grid = {
                    "min": grid_range["min"],
                    "max": grid_range["max"],
                    "n": 100,
                    "sigma": 0.1
                }
                
                # Add weighting for k2 and k3
                weighting = None
                if k_degree in [2, 3]:
                    weighting = {"function": "exp", "scale": 0.5, "threshold": 1e-3}
                
                mbtr = MBTRExtractor(
                    species=self.species,
                    geometry=geometry,
                    grid=grid,
                    weighting=weighting,
                    periodic=self.periodic,
                    normalization=self.normalization,
                    dtype=self.dtype
                )
                
                result = mbtr.extract_single(atoms)
                if result:
                    results[f"mbtr_{geom_func}"] = result.get("mbtr_all", np.array([]))
                    
            except Exception as e:
                logger.warning(f"Failed to create MBTR with {geom_func}: {e}")
        
        return results
