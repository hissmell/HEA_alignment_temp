"""
SevenNet embedding extractor for latent vector extraction.
Extracts per-atom features from the reduce_input_to_hidden layer.
"""

from typing import Dict, List, Optional, Union, Any
import numpy as np
import torch
import logging
from pathlib import Path
from ase import Atoms
from dataclasses import dataclass

from ..base import MLIPEmbeddingExtractor, ExtractionConfig

logger = logging.getLogger(__name__)

try:
    from sevenn.calculator import SevenNetCalculator
    HAS_SEVENNET = True
except ImportError:
    HAS_SEVENNET = False
    logger.error("SevenNet not available. Please install sevenn package")


@dataclass
class SevenNetConfig:
    """Configuration for SevenNet model."""
    model_name: str = "7net-0"
    model_path: Optional[str] = None
    device: str = "auto"
    extraction_layer: str = "reduce_input_to_hidden"
    aggregate: str = "none"  # "none", "mean", "sum" for per-atom features


class SevenNetExtractor(MLIPEmbeddingExtractor):
    """
    SevenNet embedding extractor for latent vector extraction.

    Extracts features from the reduce_input_to_hidden layer which provides
    per-atom latent representations before energy calculation.
    """

    def __init__(
        self,
        sevennet_config: Optional[SevenNetConfig] = None,
        extraction_config: Optional[ExtractionConfig] = None
    ):
        """
        Initialize SevenNet extractor.

        Args:
            sevennet_config: Configuration for SevenNet model
            extraction_config: General extraction configuration
        """
        if not HAS_SEVENNET:
            raise ImportError("SevenNet not installed. Please install sevenn package.")

        self.sevennet_config = sevennet_config or SevenNetConfig()

        # Initialize parent with model name and config
        super().__init__(
            model_name=self.sevennet_config.model_name,
            model_config={},
            config=extraction_config or ExtractionConfig()
        )

        self._setup_device()
        self.latent_dim = None  # Will be determined after first extraction
        self.is_initialized = False

    def setup(self) -> None:
        """Setup the extractor (load models, initialize descriptors, etc.)"""
        if not self.is_initialized:
            self._load_model()
            self._setup_hooks()
            self.is_initialized = True

    def _setup_device(self):
        """Setup computation device."""
        if self.sevennet_config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.sevennet_config.device
        logger.info(f"Using device: {self.device}")

    def _load_model(self):
        """Load SevenNet model."""
        try:
            if self.sevennet_config.model_path:
                # Load from specific path
                logger.info(f"Loading SevenNet from: {self.sevennet_config.model_path}")
                self.calculator = SevenNetCalculator(
                    self.sevennet_config.model_path,
                    device=self.device
                )
            else:
                # Use pre-trained model
                logger.info(f"Loading pre-trained SevenNet: {self.sevennet_config.model_name}")
                self.calculator = SevenNetCalculator(
                    self.sevennet_config.model_name,
                    device=self.device
                )

            self.model = self.calculator.model
            logger.info("SevenNet model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load SevenNet model: {e}")
            raise

    def extract_single(self, atoms: Atoms) -> Dict[str, Any]:
        """
        Extract latent vector for a single structure.

        Args:
            atoms: ASE Atoms object

        Returns:
            Dictionary containing:
                - 'embedding': Latent vector (n_atoms, hidden_dim) or aggregated
                - 'n_atoms': Number of atoms
                - 'shape': Shape of embedding
                - 'energy': Total energy (if computed)
        """
        # Ensure model is loaded
        if not self.is_initialized:
            self.setup()

        # Set calculator
        atoms_copy = atoms.copy()
        atoms_copy.calc = self.calculator

        # Storage for captured latent vector
        latent_vector = None

        def capture_hook(module, input, output):
            """Hook to capture output of reduce_input_to_hidden."""
            nonlocal latent_vector
            if isinstance(output, torch.Tensor):
                latent_vector = output.detach().cpu().numpy()

        # Register hook
        handle = None
        if hasattr(self.model, self.sevennet_config.extraction_layer):
            layer = getattr(self.model, self.sevennet_config.extraction_layer)
            handle = layer.register_forward_hook(capture_hook)
        else:
            raise ValueError(f"Layer {self.sevennet_config.extraction_layer} not found in model")

        try:
            # Forward pass
            energy = atoms_copy.get_potential_energy()

            # Remove hook
            if handle:
                handle.remove()

            # Check if latent was captured
            if latent_vector is None:
                raise RuntimeError("Failed to capture latent vector")

            # Update latent dimension if not set
            if self.latent_dim is None:
                self.latent_dim = latent_vector.shape[-1]
                logger.info(f"Detected latent dimension: {self.latent_dim}")

            # Aggregate if requested
            if self.sevennet_config.aggregate == "mean":
                embedding = np.mean(latent_vector, axis=0)
            elif self.sevennet_config.aggregate == "sum":
                embedding = np.sum(latent_vector, axis=0)
            else:  # "none"
                embedding = latent_vector

            return {
                'embedding': embedding,
                'n_atoms': len(atoms),
                'shape': embedding.shape,
                'energy': float(energy),
                'model': self.sevennet_config.model_name
            }

        except Exception as e:
            # Make sure to remove hook even if error occurs
            if handle:
                handle.remove()
            logger.error(f"Failed to extract embedding: {e}")
            raise

    def extract_batch(
        self,
        atoms_list: List[Atoms],
        batch_size: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract embeddings for a batch of structures.

        Args:
            atoms_list: List of ASE Atoms objects
            batch_size: Not used for SevenNet (processes one at a time)

        Returns:
            List of extraction results
        """
        results = []

        for i, atoms in enumerate(atoms_list):
            if i % 100 == 0:
                logger.info(f"Processing structure {i+1}/{len(atoms_list)}")

            try:
                result = self.extract_single(atoms)
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to extract structure {i}: {e}")
                results.append({
                    'embedding': None,
                    'error': str(e),
                    'n_atoms': len(atoms)
                })

        return results

    def get_embedding_dim(self) -> Optional[int]:
        """
        Get the dimension of the latent vectors.

        Returns:
            Latent dimension or None if not yet determined
        """
        return self.latent_dim

    def _setup_hooks(self) -> None:
        """Setup forward hooks for representation extraction."""
        # Hooks are registered dynamically in extract_single
        # No persistent hooks needed for SevenNet
        pass

    def get_available_layers(self) -> List[str]:
        """Return list of available layers for extraction."""
        return [self.sevennet_config.extraction_layer]

    def get_feature_names(self) -> List[str]:
        """Return list of feature names for this representation."""
        if self.sevennet_config.aggregate == "none":
            return [f"sevennet_{self.model_name}_per_atom"]
        else:
            return [f"sevennet_{self.model_name}_{self.sevennet_config.aggregate}"]

    def get_feature_dimensions(self) -> Dict[str, int]:
        """Return dictionary mapping representation names to their dimensions."""
        if self.latent_dim is None:
            # Need to run once to determine dimension
            return {self.get_feature_names()[0]: -1}

        feature_names = self.get_feature_names()
        return {name: self.latent_dim for name in feature_names}


def create_sevennet_extractor(
    model_name: str = "7net-0",
    model_path: Optional[str] = None,
    device: str = "auto",
    aggregate: str = "none"
) -> SevenNetExtractor:
    """
    Factory function to create SevenNet extractor.

    Args:
        model_name: Name of pre-trained model or identifier
        model_path: Path to model checkpoint (overrides model_name)
        device: Device to use ("auto", "cuda", "cpu")
        aggregate: How to aggregate per-atom features ("none", "mean", "sum")

    Returns:
        Configured SevenNetExtractor instance
    """
    config = SevenNetConfig(
        model_name=model_name,
        model_path=model_path,
        device=device,
        aggregate=aggregate
    )

    return SevenNetExtractor(sevennet_config=config)


# Convenience functions for specific SevenNet models
def create_sevennet_omni_extractor(**kwargs) -> SevenNetExtractor:
    """Create extractor for SevenNet-Omni model."""
    path = "/home/pn50212/anaconda3/envs/fairchem/lib/python3.10/site-packages/sevenn/pretrained_potentials/SevenNet_omni/checkpoint_sevennet_omni.pth"
    return create_sevennet_extractor(
        model_name="SevenNet-Omni",
        model_path=path,
        **kwargs
    )


def create_sevennet_omat_extractor(**kwargs) -> SevenNetExtractor:
    """Create extractor for SevenNet-OMAT model."""
    path = "/home/pn50212/anaconda3/envs/fairchem/lib/python3.10/site-packages/sevenn/pretrained_potentials/SevenNet_omat/checkpoint_sevennet_omat.pth"
    return create_sevennet_extractor(
        model_name="SevenNet-OMAT",
        model_path=path,
        **kwargs
    )