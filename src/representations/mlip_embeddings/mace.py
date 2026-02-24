"""
MACE (Multi Atomic Cluster Expansion) embedding extractor.
Based on the existing extract_mace_representations_25cao.py implementation.
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
    from mace.calculators import mace_mp
    HAS_MACE = True
except ImportError:
    HAS_MACE = False
    logger.error("mace-torch not available. Please install mace-torch")


@dataclass
class MACEConfig:
    """Configuration for MACE model."""
    model_path: str = "medium"  # or path to checkpoint
    head: str = "omat_pbe"  # "omat_pbe" for bulk, "oc20_usemppbe" for surfaces
    extraction_layers: List[str] = None
    device: str = "auto"
    default_dtype: str = "float64"

    def __post_init__(self):
        if self.extraction_layers is None:
            self.extraction_layers = ["readout_input"]


class MACEExtractor(MLIPEmbeddingExtractor):
    """
    MACE embedding extractor with support for different model sizes.
    Extracts node features from the readout layer input.
    """

    def __init__(
        self,
        mace_config: Optional[MACEConfig] = None,
        config: Optional[ExtractionConfig] = None
    ):
        """
        Initialize MACE extractor.

        Args:
            mace_config: MACE-specific configuration
            config: General extraction configuration
        """
        self.mace_config = mace_config or MACEConfig()

        super().__init__(
            model_name=f"mace_{self.mace_config.model_path}",
            model_config=self.mace_config.__dict__,
            config=config
        )

        # Set device
        if self.mace_config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.mace_config.device

        self.captured_representations = []

    def _load_model(self) -> None:
        """Load MACE model using mace-torch."""
        if not HAS_MACE:
            raise ImportError("mace-torch is required for MACE extraction")

        try:
            logger.info(f"Loading MACE model: {self.mace_config.model_path}")

            # Initialize MACE calculator
            self.calculator = mace_mp(
                model=self.mace_config.model_path,
                default_dtype=self.mace_config.default_dtype,
                device=self.device,
                head=self.mace_config.head
            )

            # Get direct access to model
            if hasattr(self.calculator, 'models'):
                self.model = self.calculator.models[0] if isinstance(self.calculator.models, list) else self.calculator.models
            elif hasattr(self.calculator, 'model'):
                self.model = self.calculator.model
            else:
                self.model = self.calculator

            logger.info(f"Loaded MACE model on {self.device}")
            logger.info(f"Model head: {self.mace_config.head}")

        except Exception as e:
            logger.error(f"Failed to load MACE model: {e}")
            raise

    def _setup_hooks(self) -> None:
        """Setup forward hooks for representation extraction."""
        extraction_layers = self.mace_config.extraction_layers

        logger.info(f"Setting up hooks for layers: {extraction_layers}")

        for layer_name in extraction_layers:
            if layer_name == "readout_input":
                self._setup_readout_hook()
            else:
                logger.warning(f"Unknown layer: {layer_name}")

    def _setup_readout_hook(self) -> None:
        """Setup hook for readout layer input (node features)."""
        def readout_hook(module, input, output):
            try:
                self.captured_representations.clear()
                # Input is a tuple, first element is node features
                if isinstance(input, tuple) and len(input) > 0:
                    node_features = input[0]
                    if isinstance(node_features, torch.Tensor):
                        self.representations['readout_input'] = node_features.detach().cpu().numpy()
                        self.captured_representations.append(node_features.detach().cpu().numpy())
            except Exception as e:
                logger.warning(f"Failed to extract readout input: {e}")

        # Register hook on readouts[0]
        if hasattr(self.model, 'readouts') and len(self.model.readouts) > 0:
            hook = self.model.readouts[0].register_forward_hook(readout_hook)
            self.hooks.append(hook)
            logger.info("Forward hook registered on readouts[0]")
        else:
            raise RuntimeError("Could not find readouts layer in MACE model")

    def extract_single(
        self,
        atoms: Atoms,
        layers: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Extract MACE embeddings for a single structure.

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
            logger.error(f"Failed to extract MACE embeddings: {e}")
            return {}

    def get_available_layers(self) -> List[str]:
        """Return list of available layers for extraction."""
        return ["readout_input"]

    def get_feature_names(self) -> List[str]:
        """Return MACE feature names."""
        return self.mace_config.extraction_layers

    def get_feature_dimensions(self) -> Dict[str, int]:
        """Return MACE feature dimensions."""
        # MACE dimensions vary by model size
        # These are typical values, actual dimensions depend on model
        dimensions = {
            "readout_input": 256,  # Typical MACE embedding dimension
        }

        return {k: v for k, v in dimensions.items() if k in self.mace_config.extraction_layers}

    def __del__(self):
        """Clean up hooks when extractor is deleted."""
        super().__del__()
        self.captured_representations = []


def create_mace_extractor(
    model_path: str = "medium",
    head: str = "omat_pbe",
    extraction_layers: Optional[List[str]] = None,
    **kwargs
) -> MACEExtractor:
    """
    Convenience function to create MACE extractor.

    Args:
        model_path: Path to MACE model or model size ("small", "medium", "large")
        head: Model head to use
        extraction_layers: Layers to extract from
        **kwargs: Additional configuration

    Returns:
        Initialized MACEExtractor
    """
    mace_config = MACEConfig(
        model_path=model_path,
        head=head,
        extraction_layers=extraction_layers or ["readout_input"],
        **kwargs
    )

    return MACEExtractor(mace_config)