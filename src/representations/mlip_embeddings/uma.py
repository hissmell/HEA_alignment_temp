"""
UMA (Universal Materials Accelerator) embedding extractor.
Based on the existing extract_uma_representations_25cao.py implementation.
"""

from typing import Dict, List, Optional, Any
import numpy as np
import torch
import logging
from pathlib import Path
from ase import Atoms
from dataclasses import dataclass

from ..base import MLIPEmbeddingExtractor, ExtractionConfig

logger = logging.getLogger(__name__)

try:
    from fairchem.core import FAIRChemCalculator
    HAS_FAIRCHEM = True
except ImportError:
    HAS_FAIRCHEM = False
    logger.error("fairchem-core not available. Please install fairchem-core")


@dataclass
class UMAConfig:
    """Configuration for UMA model."""
    model_path: str  # Path to UMA checkpoint
    task_name: str = "oc20"
    extraction_layers: List[str] = None
    device: str = "auto"

    def __post_init__(self):
        if self.extraction_layers is None:
            self.extraction_layers = ["energy_block_input"]


class UMAExtractor(MLIPEmbeddingExtractor):
    """
    UMA embedding extractor.
    Extracts representations from the energy block input (backbone output).
    """

    def __init__(
        self,
        uma_config: UMAConfig,
        config: Optional[ExtractionConfig] = None
    ):
        """
        Initialize UMA extractor.

        Args:
            uma_config: UMA-specific configuration
            config: General extraction configuration
        """
        self.uma_config = uma_config

        super().__init__(
            model_name=f"uma_{Path(uma_config.model_path).stem}",
            model_config=uma_config.__dict__,
            config=config
        )

        # Set device
        if self.uma_config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.uma_config.device

        self.captured_representations = []

    def _load_model(self) -> None:
        """Load UMA model using FAIRChemCalculator."""
        if not HAS_FAIRCHEM:
            raise ImportError("fairchem-core is required for UMA extraction")

        try:
            logger.info(f"Loading UMA model: {self.uma_config.model_path}")

            # Initialize FAIRChemCalculator
            self.calculator = FAIRChemCalculator.from_model_checkpoint(
                name_or_path=self.uma_config.model_path,
                task_name=self.uma_config.task_name,
                device=self.device
            )

            # Get direct access to model
            self.model = self.calculator.predictor.model.module

            logger.info(f"Loaded UMA model on {self.device}")
            logger.info(f"Task: {self.uma_config.task_name}")

        except Exception as e:
            logger.error(f"Failed to load UMA model: {e}")
            raise

    def _setup_hooks(self) -> None:
        """Setup forward hooks for representation extraction."""
        extraction_layers = self.uma_config.extraction_layers

        logger.info(f"Setting up hooks for layers: {extraction_layers}")

        for layer_name in extraction_layers:
            if layer_name == "energy_block_input":
                self._setup_energy_block_hook()
            else:
                logger.warning(f"Unknown layer: {layer_name}")

    def _setup_energy_block_hook(self) -> None:
        """Setup hook for energy block input (backbone output)."""
        def energy_block_hook(module, input, output):
            try:
                self.captured_representations.clear()
                # Input is a tuple, first element is the backbone representation
                if isinstance(input, tuple) and len(input) > 0:
                    backbone_repr = input[0]
                    if isinstance(backbone_repr, torch.Tensor):
                        self.representations['energy_block_input'] = backbone_repr.detach().cpu().numpy()
                        self.captured_representations.append(backbone_repr.detach().cpu().numpy())
            except Exception as e:
                logger.warning(f"Failed to extract energy block input: {e}")

        # Register hook on energy_block[0] (first Linear layer)
        try:
            energy_block = self.model.output_heads.energyandforcehead.head.energy_block
            hook = energy_block[0].register_forward_hook(energy_block_hook)
            self.hooks.append(hook)
            logger.info("Forward hook registered on energy_block[0]")
        except (AttributeError, IndexError) as e:
            logger.error(f"Could not find energy block in UMA model: {e}")
            raise RuntimeError("Could not find energy block in UMA model")

    def extract_single(
        self,
        atoms: Atoms,
        layers: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Extract UMA embeddings for a single structure.

        Args:
            atoms: ASE Atoms object
            layers: Specific layers to extract (None for all configured)
            **kwargs: Additional arguments (unused)

        Returns:
            Dictionary mapping layer names to embedding arrays
        """
        if not self.is_initialized:
            self.setup()

        # Clear previous representations
        self.representations = {}
        self.captured_representations = []

        try:
            # Set calculator and compute energy to trigger hook
            atoms_copy = atoms.copy()
            atoms_copy.calc = self.calculator
            atoms_copy.set_tags(np.ones(len(atoms_copy)))
            atoms_copy.pbc = True

            # Forward pass to trigger hooks
            _ = atoms_copy.get_potential_energy()

            # Filter results if specific layers requested
            if layers:
                filtered_results = {}
                for layer in layers:
                    if layer in self.representations:
                        filtered_results[layer] = self.representations[layer]
                return filtered_results

            return dict(self.representations)

        except Exception as e:
            logger.error(f"Failed to extract UMA embeddings: {e}")
            return {}

    def get_available_layers(self) -> List[str]:
        """Return list of available layers for extraction."""
        return ["energy_block_input"]

    def get_feature_names(self) -> List[str]:
        """Return UMA feature names."""
        return self.uma_config.extraction_layers

    def get_feature_dimensions(self) -> Dict[str, int]:
        """Return UMA feature dimensions."""
        # UMA dimensions depend on the specific model architecture
        # These are typical values, actual dimensions depend on model
        dimensions = {
            "energy_block_input": 512,  # Typical UMA embedding dimension
        }

        return {k: v for k, v in dimensions.items() if k in self.uma_config.extraction_layers}

    def __del__(self):
        """Clean up hooks when extractor is deleted."""
        super().__del__()
        self.captured_representations = []


def create_uma_extractor(
    model_path: str,
    task_name: str = "oc20",
    extraction_layers: Optional[List[str]] = None,
    **kwargs
) -> UMAExtractor:
    """
    Convenience function to create UMA extractor.

    Args:
        model_path: Path to UMA model checkpoint
        task_name: Task name for the model
        extraction_layers: Layers to extract from
        **kwargs: Additional configuration

    Returns:
        Initialized UMAExtractor
    """
    uma_config = UMAConfig(
        model_path=model_path,
        task_name=task_name,
        extraction_layers=extraction_layers or ["energy_block_input"],
        **kwargs
    )

    return UMAExtractor(uma_config)