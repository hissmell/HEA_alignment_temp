"""
Base classes for representation extraction.
Supports both physics-inspired descriptors and MLIP embeddings.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
from dataclasses import dataclass
from ase import Atoms
import logging
import json

logger = logging.getLogger(__name__)


@dataclass
class ExtractionConfig:
    """Configuration for representation extraction."""
    batch_size: int = 1000
    output_dir: Optional[Path] = None
    checkpoint_interval: int = 1000
    save_format: str = "npz"  # "npz", "json", "both"
    resume_from_checkpoint: bool = True
    device: str = "auto"  # "cpu", "cuda", "auto"
    n_jobs: int = -1  # For parallel processing in physics-inspired methods


class RepresentationExtractor(ABC):
    """
    Abstract base class for all representation extractors.
    Defines common interface for both physics-inspired and MLIP methods.
    """

    def __init__(self, config: Optional[ExtractionConfig] = None):
        """
        Initialize representation extractor.

        Args:
            config: Configuration for extraction parameters
        """
        self.config = config or ExtractionConfig()
        self.name = self.__class__.__name__
        self.is_initialized = False

    @abstractmethod
    def setup(self) -> None:
        """Setup the extractor (load models, initialize descriptors, etc.)"""
        pass

    @abstractmethod
    def extract_single(self, atoms: Atoms, **kwargs) -> Dict[str, np.ndarray]:
        """
        Extract representation for a single structure.

        Args:
            atoms: ASE Atoms object
            **kwargs: Additional arguments specific to extractor type

        Returns:
            Dictionary mapping representation names to numpy arrays
        """
        pass

    def extract_batch(
        self,
        structures: List[Atoms],
        tasknames: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Extract representations for a batch of structures.

        Args:
            structures: List of ASE Atoms objects
            tasknames: Optional list of tasknames (will generate if None)
            **kwargs: Additional arguments for extraction

        Returns:
            Dictionary mapping taskname to representation dictionary
        """
        if tasknames is None:
            tasknames = [f"struct_{i}" for i in range(len(structures))]

        if len(tasknames) != len(structures):
            raise ValueError("Number of tasknames must match number of structures")

        results = {}
        for taskname, atoms in zip(tasknames, structures):
            try:
                repr_dict = self.extract_single(atoms, **kwargs)
                if repr_dict:
                    results[taskname] = repr_dict
            except Exception as e:
                logger.warning(f"Failed to extract {taskname}: {e}")

        return results

    def save_representations(
        self,
        representations: Dict[str, Dict[str, np.ndarray]],
        output_file: Path,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Save extracted representations to file.

        Args:
            representations: Dictionary mapping taskname to representations
            output_file: Output file path
            metadata: Optional metadata to include
        """
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if self.config.save_format in ["npz", "both"]:
            # Save numpy arrays to .npz file
            arrays_dict = {}
            for taskname, repr_dict in representations.items():
                for repr_name, array in repr_dict.items():
                    arrays_dict[f"{taskname}_{repr_name}"] = array

            npz_file = output_file.with_suffix('.npz')
            np.savez_compressed(npz_file, **arrays_dict)

        if self.config.save_format in ["json", "both"]:
            # Save metadata to .json file
            json_data = {
                "metadata": {
                    "extractor_name": self.name,
                    "num_structures": len(representations),
                    **(metadata or {})
                },
                "structures": {}
            }

            # Add structure metadata (shapes, etc.)
            for taskname, repr_dict in representations.items():
                json_data["structures"][taskname] = {
                    "representations": {
                        name: {
                            "shape": list(array.shape),
                            "dtype": str(array.dtype)
                        }
                        for name, array in repr_dict.items()
                    }
                }

            json_file = output_file.with_suffix('.json')
            with open(json_file, 'w') as f:
                json.dump(json_data, f, indent=2)

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Return list of feature names for this representation."""
        pass

    @abstractmethod
    def get_feature_dimensions(self) -> Dict[str, int]:
        """Return dictionary mapping representation names to their dimensions."""
        pass


class PhysicsInspiredExtractor(RepresentationExtractor):
    """
    Base class for physics-inspired descriptor extractors.
    Examples: SOAP, ACSF, MBTR, etc.
    """

    def __init__(self, config: Optional[ExtractionConfig] = None):
        super().__init__(config)
        self.descriptor = None

    def extract_single(
        self,
        atoms: Atoms,
        atom_selection: Optional[str] = "all",  # "all", "slab", "site"
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Extract physics-inspired descriptors for a single structure.

        Args:
            atoms: ASE Atoms object
            atom_selection: Which atoms to include ("all", "slab", "site")
            **kwargs: Additional descriptor-specific arguments

        Returns:
            Dictionary with descriptor arrays for different atom selections
        """
        if not self.is_initialized:
            self.setup()

        results = {}

        if atom_selection in ["all", "slab", "site"]:
            selected_atoms = self._select_atoms(atoms, atom_selection, **kwargs)
            if selected_atoms is not None:
                descriptors = self.descriptor.create(selected_atoms)
                results[f"{self.descriptor_name}_{atom_selection}"] = descriptors

        return results

    def _select_atoms(
        self,
        atoms: Atoms,
        selection: str,
        adsorbate_indices: Optional[List[int]] = None,
        site_cutoff: float = 3.0
    ) -> Optional[Atoms]:
        """
        Select atoms based on selection criteria.

        Args:
            atoms: ASE Atoms object
            selection: Selection type ("all", "slab", "site")
            adsorbate_indices: Indices of adsorbate atoms
            site_cutoff: Cutoff distance for site selection

        Returns:
            Selected atoms or None if selection is invalid
        """
        if selection == "all":
            return atoms

        elif selection == "slab":
            # Remove adsorbate atoms
            if adsorbate_indices:
                slab_indices = [i for i in range(len(atoms)) if i not in adsorbate_indices]
                return atoms[slab_indices]
            else:
                # Assume last 1-2 atoms are adsorbate (heuristic)
                return atoms[:-2]  # Conservative: remove last 2 atoms

        elif selection == "site":
            # Select atoms within cutoff of adsorbate
            if adsorbate_indices:
                adsorbate_positions = atoms.positions[adsorbate_indices]
                distances = np.linalg.norm(
                    atoms.positions[:, np.newaxis, :] - adsorbate_positions[np.newaxis, :, :],
                    axis=2
                )
                min_distances = distances.min(axis=1)
                site_mask = min_distances <= site_cutoff
                site_indices = np.where(site_mask)[0]

                # Filter out adsorbate atoms themselves
                site_slab_indices = [i for i in site_indices if i not in adsorbate_indices]

                if len(site_slab_indices) >= 2:
                    return atoms[site_slab_indices]

        return None

    @property
    @abstractmethod
    def descriptor_name(self) -> str:
        """Return the name of this descriptor type."""
        pass


class MLIPEmbeddingExtractor(RepresentationExtractor):
    """
    Base class for MLIP embedding extractors.
    Examples: EquiformerV2, MACE, UMA, etc.
    """

    def __init__(
        self,
        model_name: str,
        model_config: Optional[Dict] = None,
        config: Optional[ExtractionConfig] = None
    ):
        super().__init__(config)
        self.model_name = model_name
        self.model_config = model_config or {}
        self.model = None
        self.calculator = None
        self.hooks = []
        self.representations = {}

    def setup(self) -> None:
        """Setup MLIP model and hooks for representation extraction."""
        self._load_model()
        self._setup_hooks()
        self.is_initialized = True

    @abstractmethod
    def _load_model(self) -> None:
        """Load the MLIP model."""
        pass

    @abstractmethod
    def _setup_hooks(self) -> None:
        """Setup forward hooks for representation extraction."""
        pass

    def extract_single(
        self,
        atoms: Atoms,
        layers: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Extract MLIP embeddings for a single structure.

        Args:
            atoms: ASE Atoms object
            layers: Specific layers to extract (None for default)
            **kwargs: Additional model-specific arguments

        Returns:
            Dictionary mapping layer names to embedding arrays
        """
        if not self.is_initialized:
            self.setup()

        # Clear previous representations
        self.representations = {}

        try:
            # Set up atoms for model prediction
            atoms_copy = atoms.copy()
            atoms_copy.calc = self.calculator
            atoms_copy.set_tags(np.ones(len(atoms_copy)))
            atoms_copy.pbc = True

            # Forward pass to trigger hooks
            _ = atoms_copy.get_potential_energy()

            # Return copy of extracted representations
            return dict(self.representations)

        except Exception as e:
            logger.error(f"Failed to extract embeddings: {e}")
            return {}

    def _create_hook(self, name: str):
        """Create a forward hook for representation extraction."""
        def hook_fn(module, input, output):
            try:
                if isinstance(output, tuple):
                    # Handle multiple outputs
                    for i, out in enumerate(output):
                        if hasattr(out, 'detach'):
                            self.representations[f"{name}_{i}"] = out.detach().cpu().numpy()
                elif hasattr(output, 'detach'):
                    # Single tensor output
                    self.representations[name] = output.detach().cpu().numpy()
            except Exception as e:
                logger.warning(f"Failed to extract representation from {name}: {e}")
        return hook_fn

    def __del__(self):
        """Clean up hooks when extractor is deleted."""
        for hook in self.hooks:
            try:
                hook.remove()
            except:
                pass

    @abstractmethod
    def get_available_layers(self) -> List[str]:
        """Return list of available layers for extraction."""
        pass


class HybridRepresentation:
    """
    Combine multiple representation types for analysis.
    Supports both physics-inspired and MLIP representations.
    """

    def __init__(
        self,
        extractors: Dict[str, RepresentationExtractor],
        combination_strategy: str = "concatenate"  # "concatenate", "separate"
    ):
        """
        Initialize hybrid representation.

        Args:
            extractors: Dictionary mapping names to extractor instances
            combination_strategy: How to combine representations
        """
        self.extractors = extractors
        self.combination_strategy = combination_strategy

        # Initialize all extractors
        for extractor in self.extractors.values():
            if not extractor.is_initialized:
                extractor.setup()

    def extract_all(
        self,
        atoms: Atoms,
        **kwargs
    ) -> Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]:
        """
        Extract representations using all extractors.

        Args:
            atoms: ASE Atoms object
            **kwargs: Arguments passed to individual extractors

        Returns:
            Dictionary with all extracted representations
        """
        results = {}

        for name, extractor in self.extractors.items():
            try:
                repr_dict = extractor.extract_single(atoms, **kwargs)

                if self.combination_strategy == "separate":
                    results[name] = repr_dict
                elif self.combination_strategy == "concatenate":
                    # Flatten and concatenate all arrays
                    for repr_name, array in repr_dict.items():
                        results[f"{name}_{repr_name}"] = array

            except Exception as e:
                logger.warning(f"Failed to extract {name}: {e}")

        return results

    def extract_batch(
        self,
        structures: List[Atoms],
        tasknames: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        """
        Extract hybrid representations for multiple structures.

        Args:
            structures: List of ASE Atoms objects
            tasknames: Optional tasknames
            **kwargs: Additional arguments

        Returns:
            Dictionary mapping tasknames to hybrid representations
        """
        if tasknames is None:
            tasknames = [f"struct_{i}" for i in range(len(structures))]

        results = {}
        for taskname, atoms in zip(tasknames, structures):
            results[taskname] = self.extract_all(atoms, **kwargs)

        return results

    def get_feature_info(self) -> Dict[str, Dict[str, Any]]:
        """Get feature information from all extractors."""
        info = {}
        for name, extractor in self.extractors.items():
            info[name] = {
                "feature_names": extractor.get_feature_names(),
                "feature_dimensions": extractor.get_feature_dimensions()
            }
        return info


# Utility functions
def create_representation_extractor(
    extractor_type: str,
    **kwargs
) -> RepresentationExtractor:
    """
    Factory function to create representation extractors.

    Args:
        extractor_type: Type of extractor ("soap", "equiformer", etc.)
        **kwargs: Arguments for extractor initialization

    Returns:
        Initialized representation extractor
    """
    # Will be implemented as modules are added
    extractor_map = {
        # Physics-inspired
        # "soap": SOAPExtractor,
        # "acsf": ACSFExtractor,
        # "mbtr": MBTRExtractor,
        # MLIP embeddings
        # "equiformer": EquiformerExtractor,
        # "mace": MACEExtractor,
        # "uma": UMAExtractor,
    }

    if extractor_type not in extractor_map:
        available = list(extractor_map.keys())
        raise ValueError(f"Unknown extractor type: {extractor_type}. Available: {available}")

    return extractor_map[extractor_type](**kwargs)


def load_representations(
    file_path: Path,
    tasknames: Optional[List[str]] = None
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load previously saved representations.

    Args:
        file_path: Path to saved representations
        tasknames: Specific tasknames to load (None for all)

    Returns:
        Dictionary mapping tasknames to representations
    """
    file_path = Path(file_path)

    # Load from .npz file
    if file_path.suffix == '.npz':
        npz_data = np.load(file_path, allow_pickle=True)

        # Reconstruct taskname -> representation structure
        results = {}
        for key, array in npz_data.items():
            # Parse key format: "taskname_representation_name"
            parts = key.split('_', 1)
            if len(parts) == 2:
                taskname, repr_name = parts

                if tasknames is None or taskname in tasknames:
                    if taskname not in results:
                        results[taskname] = {}
                    results[taskname][repr_name] = array

        return results

    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")