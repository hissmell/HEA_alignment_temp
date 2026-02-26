"""
Local Many-Body Tensor Representation (LMBTR) extractor
Atom-centered local environment analysis using dscribe's LocalMBTR
"""

import numpy as np
from typing import Dict, Any, List, Union, Optional, Tuple
from ase import Atoms

from ..base import PhysicsInspiredExtractor


class LocalMBTRExtractor(PhysicsInspiredExtractor):
    """
    Local Many-Body Tensor Representation (LMBTR) extractor using dscribe.

    LMBTR creates atom-centered local environment descriptors by computing
    MBTR features within a local neighborhood of specified center atoms.
    This allows per-atom analysis of local structural environments.

    Key Features:
    - Per-atom local environment descriptors
    - Support for k1, k2, k3 terms with various geometry/weighting functions
    - Atom selection modes: specific atoms, all atoms, or spatial positions
    - Suitable for analyzing adsorption sites and local coordination

    Perfect for 25Cao dataset analysis where we want to understand
    adsorbate-centered local environments.
    """

    def __init__(
        self,
        species: List[str],
        r_cut: float = 6.0,
        geometry: Dict[str, Any] = None,
        grid: Dict[str, Any] = None,
        weighting: Dict[str, Any] = None,
        normalize_gaussians: bool = True,
        normalization: str = 'l2',
        sparse: bool = False,
        dtype: str = 'float64'
    ):
        """
        Initialize LocalMBTRExtractor.

        Args:
            species: List of chemical species in the system
            r_cut: Cutoff radius for local environment (Angstrom)
            geometry: Geometry function config (default: k2 inverse_distance)
            grid: Grid discretization config
            weighting: Weighting function config
            normalize_gaussians: Whether to normalize Gaussian functions
            normalization: Normalization method ('l2', 'none', 'n_atoms', 'valle_oganov')
            sparse: Whether to use sparse output
            dtype: Output data type ('float32' or 'float64')
        """
        super().__init__()
        self.species = species
        self.r_cut = r_cut
        self.normalize_gaussians = normalize_gaussians
        self.normalization = normalization
        self.sparse = sparse
        self.dtype = dtype

        # Default parameters optimized for 25Cao analysis
        self.geometry = geometry or {"function": "inverse_distance"}
        self.grid = grid or {"min": 0, "max": 1, "n": 200, "sigma": 0.02}
        self.weighting = weighting or {"function": "exp", "scale": 0.5, "threshold": 1e-3}

        self.lmbtr = None
        self._feature_shape = None

    def setup(self) -> None:
        """Initialize the LMBTR descriptor with dscribe compatibility fixes."""
        try:
            # Apply dscribe compatibility monkey patch
            from dscribe.core.system import System

            def patched_from_atoms(atoms):
                """Patched version that avoids _get_constraints() call"""
                system = System(
                    symbols=atoms.get_chemical_symbols(),
                    positions=atoms.get_positions(),
                    cell=atoms.get_cell(),
                    pbc=atoms.get_pbc()
                )
                return system

            # Apply the monkey patch
            System.from_atoms = staticmethod(patched_from_atoms)

            # Import and setup LMBTR after patching
            from dscribe.descriptors import MBTR

            # Create LMBTR descriptor
            self.lmbtr = MBTR(
                species=self.species,
                geometry=self.geometry,
                grid=self.grid,
                weighting=self.weighting,
                normalize_gaussians=self.normalize_gaussians,
                normalization=self.normalization,
                sparse=self.sparse,
                dtype=self.dtype
            )

            print(f"LocalMBTRExtractor initialized:")
            print(f"  Species: {self.species}")
            print(f"  r_cut: {self.r_cut} Å")
            print(f"  Geometry: {self.geometry['function']}")
            print(f"  Grid: n={self.grid['n']}, range=[{self.grid['min']}, {self.grid['max']}]")
            print(f"  Weighting: {self.weighting['function']}")
            print(f"  Normalization: {self.normalization}")

        except Exception as e:
            raise ImportError(f"Failed to initialize LMBTR: {e}")

    def extract_single(
        self,
        atoms: Atoms,
        centers: Union[List[int], List[List[float]], str] = "adsorbates",
        atom_selection: str = "all"
    ) -> Dict[str, Any]:
        """
        Extract LMBTR for specified centers.

        Args:
            atoms: ASE Atoms object
            centers: Center specification:
                - List[int]: Atom indices for local environment centers
                - List[List[float]]: Cartesian positions as centers
                - "adsorbates": Auto-detect O and H atoms as centers
                - "all": Use all atoms as centers
            atom_selection: Atom selection mode (for compatibility, not used in LMBTR)

        Returns:
            Dictionary with extraction results:
                - 'lmbtr_all': LMBTR features (n_centers, n_features)
                - 'centers': Center indices or positions used
                - 'n_centers': Number of centers
                - 'feature_shape': Shape of individual descriptor
                - 'metadata': Extraction metadata
        """
        if self.lmbtr is None:
            self.setup()

        # Determine centers based on input
        if isinstance(centers, str):
            if centers == "adsorbates":
                # Find O and H atoms (adsorbates in 25Cao)
                symbols = atoms.get_chemical_symbols()
                center_indices = [i for i, symbol in enumerate(symbols) if symbol in ['O', 'H']]
                if not center_indices:
                    # Fallback: use all atoms if no adsorbates found
                    center_indices = list(range(len(atoms)))
            elif centers == "all":
                center_indices = list(range(len(atoms)))
            else:
                raise ValueError(f"Unknown center string: {centers}")

            centers_to_use = center_indices

        elif isinstance(centers, list):
            if len(centers) > 0 and isinstance(centers[0], (list, tuple, np.ndarray)):
                # Cartesian positions provided
                centers_to_use = np.array(centers)
            else:
                # Atom indices provided
                centers_to_use = centers
        else:
            raise ValueError("Centers must be 'adsorbates', 'all', list of indices, or list of positions")

        try:
            # Extract LMBTR features
            # Note: dscribe's MBTR doesn't have built-in local mode
            # We simulate it by extracting features for each center individually
            if isinstance(centers_to_use, (list, np.ndarray)) and len(centers_to_use) > 0:
                if isinstance(centers_to_use[0], (int, np.integer)):
                    # Atom indices - extract local environment for each
                    lmbtr_features = []
                    valid_centers = []

                    for center_idx in centers_to_use:
                        try:
                            # Create local neighborhood around center
                            local_atoms = self._create_local_environment(atoms, center_idx, self.r_cut)

                            # Extract MBTR for local environment
                            mbtr_local = self.lmbtr.create(local_atoms)
                            lmbtr_features.append(mbtr_local)
                            valid_centers.append(center_idx)

                        except Exception as e:
                            print(f"Warning: Failed to extract LMBTR for center {center_idx}: {e}")
                            continue

                    if lmbtr_features:
                        lmbtr_all = np.array(lmbtr_features)
                        centers_used = valid_centers
                    else:
                        # Fallback: regular MBTR
                        lmbtr_all = self.lmbtr.create(atoms).reshape(1, -1)
                        centers_used = [0]

                else:
                    # Cartesian positions - not directly supported by MBTR
                    # Fallback to regular MBTR
                    print("Warning: Cartesian position centers not supported. Using regular MBTR.")
                    lmbtr_all = self.lmbtr.create(atoms).reshape(1, -1)
                    centers_used = centers_to_use
            else:
                # No valid centers, use regular MBTR
                lmbtr_all = self.lmbtr.create(atoms).reshape(1, -1)
                centers_used = [0]

            # Store feature shape for metadata
            if lmbtr_all.ndim == 1:
                self._feature_shape = (lmbtr_all.shape[0],)
                lmbtr_all = lmbtr_all.reshape(1, -1)
            else:
                self._feature_shape = lmbtr_all.shape[1:] if lmbtr_all.ndim > 1 else lmbtr_all.shape

            # Create metadata
            metadata = {
                'descriptor': 'local_mbtr',
                'species': self.species,
                'r_cut': self.r_cut,
                'geometry': self.geometry,
                'grid': self.grid,
                'weighting': self.weighting,
                'normalization': self.normalization,
                'n_atoms': len(atoms),
                'n_centers': len(centers_used),
                'centers_type': 'atom_indices' if isinstance(centers_used[0], (int, np.integer)) else 'positions'
            }

            return {
                'lmbtr_all': lmbtr_all,
                'centers': centers_used,
                'n_centers': len(centers_used),
                'feature_shape': self._feature_shape,
                'metadata': metadata
            }

        except Exception as e:
            raise RuntimeError(f"Failed to extract LMBTR: {e}")

    def _create_local_environment(self, atoms: Atoms, center_idx: int, r_cut: float) -> Atoms:
        """
        Create local environment around a center atom.

        Args:
            atoms: Full structure
            center_idx: Index of center atom
            r_cut: Cutoff radius for local environment

        Returns:
            Local Atoms object with atoms within cutoff
        """
        from ase.neighborlist import NeighborList

        # Create neighbor list
        cutoffs = [r_cut / 2] * len(atoms)
        nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
        nl.update(atoms)

        # Get neighbors within cutoff
        indices, offsets = nl.get_neighbors(center_idx)

        # Include center atom
        local_indices = [center_idx] + list(indices)

        # Create local structure
        local_atoms = atoms[local_indices]

        # Recenter around center atom (now at index 0)
        center_pos = local_atoms.positions[0]
        local_atoms.positions -= center_pos

        return local_atoms

    def get_feature_shape(self) -> Optional[Tuple[int, ...]]:
        """Get the shape of individual LMBTR features."""
        return self._feature_shape

    @property
    def descriptor_name(self) -> str:
        """Return the name of this descriptor type."""
        return "local_mbtr"

    def get_feature_names(self) -> List[str]:
        """Return list of feature names for this representation."""
        if self.lmbtr is None:
            self.setup()

        # LMBTR features are histogram bins, so create generic names
        n_features = self._get_n_features()
        return [f"lmbtr_bin_{i}" for i in range(n_features)]

    def get_feature_dimensions(self) -> Dict[str, int]:
        """Return dictionary mapping representation names to their dimensions."""
        if self.lmbtr is None:
            self.setup()

        n_features = self._get_n_features()
        return {"lmbtr_all": n_features}

    def _get_n_features(self) -> int:
        """Calculate number of features for current configuration."""
        # Feature count depends on species and configuration
        n_species = len(self.species)

        # Estimate based on k-term and geometry
        geometry_func = self.geometry.get("function", "inverse_distance")

        if geometry_func in ["distance", "inverse_distance"]:
            # k2 term: n_species * (n_species + 1) / 2 * n_grid_points
            return int(n_species * (n_species + 1) // 2 * self.grid["n"])
        elif geometry_func in ["angle", "cosine"]:
            # k3 term: n_species * (n_species + 1) * (n_species + 2) / 6 * n_grid_points
            return int(n_species * (n_species + 1) * (n_species + 2) // 6 * self.grid["n"])
        else:
            # Default fallback
            return self.grid["n"]


def create_25cao_lmbtr_extractor(
    geometry_function: str = "inverse_distance",
    k_term: str = "k2",
    r_cut: float = 6.0
) -> LocalMBTRExtractor:
    """
    Create LMBTR extractor optimized for 25Cao dataset analysis.

    Args:
        geometry_function: Geometry function ('distance', 'inverse_distance', 'angle', 'cosine')
        k_term: K-term type ('k1', 'k2', 'k3')
        r_cut: Local environment cutoff radius

    Returns:
        Configured LocalMBTRExtractor for 25Cao
    """
    # 25Cao species: HEA (Ag, Ir, Pd, Pt, Ru) + adsorbates (O, H)
    species = ['Ag', 'Ir', 'Pd', 'Pt', 'Ru', 'O', 'H']

    # Configure geometry and grid based on function and k-term
    if k_term == "k2":
        if geometry_function == "distance":
            geometry = {"function": "distance"}
            grid = {"min": 0, "max": 20, "n": 200, "sigma": 0.1}
        elif geometry_function == "inverse_distance":
            geometry = {"function": "inverse_distance"}
            grid = {"min": 0, "max": 1, "n": 200, "sigma": 0.02}
        else:
            raise ValueError(f"Unsupported k2 geometry function: {geometry_function}")

    elif k_term == "k3":
        if geometry_function == "angle":
            geometry = {"function": "angle"}
            grid = {"min": 0, "max": 180, "n": 180, "sigma": 2}
        elif geometry_function == "cosine":
            geometry = {"function": "cosine"}
            grid = {"min": -1, "max": 1, "n": 200, "sigma": 0.02}
        else:
            raise ValueError(f"Unsupported k3 geometry function: {geometry_function}")

    else:
        raise ValueError(f"Unsupported k-term: {k_term}")

    # Default weighting
    weighting = {"function": "exp", "scale": 0.5, "threshold": 1e-3}

    return LocalMBTRExtractor(
        species=species,
        r_cut=r_cut,
        geometry=geometry,
        grid=grid,
        weighting=weighting,
        normalization='l2',
        normalize_gaussians=True,
        sparse=False,
        dtype='float64'
    )