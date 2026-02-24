"""
EquiformerV2 embedding extractor.
Based on the existing extract_equiformer_31M_representations_25cao.py implementation.
"""

from typing import Dict, List, Optional, Union, Any
import numpy as np
import torch
import yaml
import logging
from pathlib import Path
from ase import Atoms
from dataclasses import dataclass

from ..base import MLIPEmbeddingExtractor, ExtractionConfig

logger = logging.getLogger(__name__)

try:
    from fairchem.core import OCPCalculator
    from fairchem.core.common.utils import setup_imports
    HAS_FAIRCHEM = True
except ImportError:
    HAS_FAIRCHEM = False
    logger.error("fairchem-core not available. Please install fairchem package")


@dataclass
class EquiformerConfig:
    """Configuration for EquiformerV2 model."""
    model_name: str = "eq2_31M_ec4_allmd"
    model_path: Optional[str] = None
    config_path: str = "/DATA/user_scratch/pn50212/2024/12_AtomAttention/MLPs/mlp_infos.yml"
    extraction_layers: List[str] = None
    device: str = "auto"

    def __post_init__(self):
        if self.extraction_layers is None:
            self.extraction_layers = ["norm_output"]


class EquiformerExtractor(MLIPEmbeddingExtractor):
    """
    EquiformerV2 embedding extractor with support for multiple model sizes.
    Supports extraction from different layers of the model.
    """

    def __init__(
        self,
        equiformer_config: Optional[EquiformerConfig] = None,
        config: Optional[ExtractionConfig] = None
    ):
        """
        Initialize EquiformerV2 extractor.

        Args:
            equiformer_config: EquiformerV2-specific configuration
            config: General extraction configuration
        """
        self.equiformer_config = equiformer_config or EquiformerConfig()

        super().__init__(
            model_name=self.equiformer_config.model_name,
            model_config=self.equiformer_config.__dict__,
            config=config
        )

        # Set device
        if self.equiformer_config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.equiformer_config.device

    def _load_model(self) -> None:
        """Load EquiformerV2 model using OCPCalculator."""
        if not HAS_FAIRCHEM:
            raise ImportError("fairchem-core is required for EquiformerV2 extraction")

        setup_imports()

        try:
            # Load model configuration from YAML
            if self.equiformer_config.model_path is None:
                config_path = Path(self.equiformer_config.config_path)
                if not config_path.exists():
                    raise FileNotFoundError(f"Model config not found: {config_path}")

                with open(config_path, 'r') as f:
                    mlp_info = yaml.safe_load(f)

                if self.model_name not in mlp_info:
                    available_models = list(mlp_info.keys())
                    raise ValueError(f"Model {self.model_name} not found. Available: {available_models}")

                model_info = mlp_info[self.model_name]
                model_path = model_info['path']
            else:
                model_path = self.equiformer_config.model_path

            # Initialize OCPCalculator
            self.calculator = OCPCalculator(
                checkpoint_path=model_path,
                cpu=self.device == "cpu"
            )

            # Get direct access to model
            self.model = self.calculator.trainer.model.to(self.device)

            logger.info(f"Loaded {self.model_name} model on {self.device}")
            logger.info(f"Model path: {model_path}")

        except Exception as e:
            logger.error(f"Failed to load EquiformerV2 model: {e}")
            raise

    def _setup_hooks(self) -> None:
        """Setup forward hooks for representation extraction."""
        extraction_layers = self.equiformer_config.extraction_layers

        logger.info(f"Setting up hooks for layers: {extraction_layers}")

        for layer_name in extraction_layers:
            if layer_name == "norm_output":
                self._setup_norm_hook()
            elif layer_name.startswith("layer_"):
                layer_idx = int(layer_name.split("_")[1])
                self._setup_layer_hook(layer_idx)
            elif layer_name == "embedding":
                self._setup_embedding_hook()
            else:
                logger.warning(f"Unknown layer: {layer_name}")

    def _setup_norm_hook(self) -> None:
        """Setup hook for norm layer (final representation)."""
        def norm_hook(module, input, output):
            try:
                if isinstance(output, torch.Tensor):
                    # EquiformerV2 norm output shape: (N_atoms, 49, 128)
                    # Average over SO3 coefficients to get atom-wise representation
                    if len(output.shape) == 3:
                        # Take mean over middle dimension (SO3 coefficients)
                        atom_repr = torch.mean(output, dim=1)  # (N_atoms, 128)
                        self.representations['norm_output'] = atom_repr.detach().cpu().numpy()
                    else:
                        self.representations['norm_output'] = output.detach().cpu().numpy()
                else:
                    logger.warning("Norm output is not a tensor")
            except Exception as e:
                logger.warning(f"Failed to extract norm output: {e}")

        hook = self.model.norm.register_forward_hook(norm_hook)
        self.hooks.append(hook)

    def _setup_layer_hook(self, layer_idx: int) -> None:
        """Setup hook for a specific layer."""
        if not hasattr(self.model, 'layers') or layer_idx >= len(self.model.layers):
            logger.warning(f"Layer {layer_idx} not found in model")
            return

        def layer_hook(module, input, output):
            try:
                if isinstance(output, torch.Tensor):
                    self.representations[f'layer_{layer_idx}'] = output.detach().cpu().numpy()
                elif isinstance(output, (list, tuple)):
                    for i, out in enumerate(output):
                        if isinstance(out, torch.Tensor):
                            self.representations[f'layer_{layer_idx}_{i}'] = out.detach().cpu().numpy()
            except Exception as e:
                logger.warning(f"Failed to extract from layer {layer_idx}: {e}")

        hook = self.model.layers[layer_idx].register_forward_hook(layer_hook)
        self.hooks.append(hook)

    def _setup_embedding_hook(self) -> None:
        """Setup hook for embedding layer."""
        if not hasattr(self.model, 'emb'):
            logger.warning("Embedding layer not found in model")
            return

        def embedding_hook(module, input, output):
            try:
                if isinstance(output, torch.Tensor):
                    self.representations['embedding'] = output.detach().cpu().numpy()
            except Exception as e:
                logger.warning(f"Failed to extract embedding: {e}")

        hook = self.model.emb.register_forward_hook(embedding_hook)
        self.hooks.append(hook)

    def extract_single(
        self,
        atoms: Atoms,
        layers: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Extract EquiformerV2 embeddings for a single structure.

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

        try:
            # Prepare atoms for model
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
            logger.error(f"Failed to extract EquiformerV2 embeddings: {e}")
            return {}

    def get_available_layers(self) -> List[str]:
        """Return list of available layers for extraction."""
        available = ["norm_output", "embedding"]

        # Add layer-specific hooks if model is loaded
        if self.model and hasattr(self.model, 'layers'):
            for i in range(len(self.model.layers)):
                available.append(f"layer_{i}")

        return available

    def get_feature_names(self) -> List[str]:
        """Return EquiformerV2 feature names."""
        return self.equiformer_config.extraction_layers

    def get_feature_dimensions(self) -> Dict[str, int]:
        """Return EquiformerV2 feature dimensions."""
        # Standard EquiformerV2 dimensions (model-dependent)
        dimensions = {
            "norm_output": 128,  # Standard for 31M model
            "embedding": 256,    # Typical embedding dimension
        }

        # Add layer dimensions if known
        if self.model:
            for layer_name in self.equiformer_config.extraction_layers:
                if layer_name.startswith("layer_") and layer_name not in dimensions:
                    dimensions[layer_name] = 128  # Default assumption

        return {k: v for k, v in dimensions.items() if k in self.equiformer_config.extraction_layers}

    def extract_batch_with_checkpointing(
        self,
        structures: Dict[str, Atoms],
        batch_size: int = 1000,
        output_dir: Optional[Path] = None,
        resume: bool = True
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Extract embeddings for multiple structures with checkpointing.
        Replicates the batch processing from the original script.

        Args:
            structures: Dictionary mapping tasknames to structures
            batch_size: Number of structures per batch
            output_dir: Directory for saving batches
            resume: Whether to resume from existing checkpoints

        Returns:
            Dictionary mapping tasknames to embeddings
        """
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            batch_dir = output_dir / "equiformer_batches"
            batch_dir.mkdir(exist_ok=True)

        tasknames = list(structures.keys())
        all_results = {}
        batch_num = 0

        # Process in batches
        for i in range(0, len(tasknames), batch_size):
            batch_tasknames = tasknames[i:i + batch_size]
            batch_results = []

            logger.info(f"Processing batch {batch_num + 1} ({len(batch_tasknames)} structures)")

            for taskname in batch_tasknames:
                atoms = structures[taskname]
                embeddings = self.extract_single(atoms)

                if embeddings:
                    batch_results.append({
                        'taskname': taskname,
                        'representations': embeddings
                    })
                    all_results[taskname] = embeddings

            # Save batch if output directory provided
            if output_dir and batch_results:
                self._save_batch_results(batch_results, batch_num, batch_dir)

            batch_num += 1

        return all_results

    def _save_batch_results(
        self,
        batch_results: List[Dict],
        batch_num: int,
        batch_dir: Path
    ) -> None:
        """Save batch results in the same format as original script."""
        import json

        json_file = batch_dir / f"batch_{batch_num:04d}.json"
        npz_file = batch_dir / f"batch_{batch_num:04d}.npz"

        # Prepare metadata
        metadata = {
            "model": self.model_name,
            "repr_dimension": self.get_feature_dimensions(),
            "extraction_layers": self.equiformer_config.extraction_layers,
            "batch_size": len(batch_results),
            "batch_number": batch_num
        }

        # Prepare data structures
        tasknames = []
        structures_info = []
        all_representations = {}

        for i, result in enumerate(batch_results):
            taskname = result['taskname']
            representations = result['representations']

            tasknames.append(taskname)
            structures_info.append({
                "taskname": taskname,
                "structure_id": i
            })

            # Collect representations
            for layer_name, repr_data in representations.items():
                if layer_name not in all_representations:
                    all_representations[layer_name] = []
                all_representations[layer_name].append(repr_data)

        # Convert to numpy arrays
        for layer_name in all_representations:
            all_representations[layer_name] = np.array(all_representations[layer_name], dtype=object)

        # Save JSON metadata
        json_data = {
            "metadata": metadata,
            "tasknames": tasknames,
            "structures": structures_info
        }

        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2)

        # Save NPZ representations
        np.savez_compressed(npz_file, **all_representations)

        logger.info(f"Saved batch {batch_num} to {json_file} and {npz_file}")


def create_equiformer_extractor(
    model_name: str = "eq2_31M_ec4_allmd",
    extraction_layers: Optional[List[str]] = None,
    **kwargs
) -> EquiformerExtractor:
    """
    Convenience function to create EquiformerV2 extractor.

    Args:
        model_name: Name of the EquiformerV2 model
        extraction_layers: Layers to extract from
        **kwargs: Additional configuration

    Returns:
        Initialized EquiformerExtractor
    """
    equiformer_config = EquiformerConfig(
        model_name=model_name,
        extraction_layers=extraction_layers or ["norm_output"],
        **kwargs
    )

    return EquiformerExtractor(equiformer_config)