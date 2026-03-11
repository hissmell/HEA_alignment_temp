"""
UMA Latent Vector Extractor

Extracts per-atom latent representations from UMA models
by hooking into the output_heads.energyandforcehead.head.energy_block.2 layer output.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path
from ase import Atoms
import warnings
warnings.filterwarnings('ignore')


class UMALatentExtractor:
    """Extract latent vectors from UMA models"""

    def __init__(
        self,
        model_name: str = "uma_s_1p1",
        model_path: str = "/DATA/user_scratch/pn50212/2024/12_AtomAttention/MLPs/uma-s-1p1.pt",
        device: str = "cuda"
    ):
        """
        Initialize UMA latent extractor

        Args:
            model_name: Model identifier (uma_s_1p1, uma_m_1p1)
            model_path: Path to model checkpoint
            device: Device for computation
        """
        self.model_name = model_name
        self.model_path = model_path
        self.device = device if torch.cuda.is_available() else "cpu"
        self.extraction_layer = "output_heads.energyandforcehead.head.energy_block.2"
        # Note: self.model points to predictor.model.module (inside AveragedModel),
        # so the layer path is relative to that module (no extra "module." prefix)

        self.model = None
        self.calculator = None
        self.latent_vectors = None
        self.hook_handle = None

        self._setup_model()

    def _setup_model(self):
        """Load UMA model and setup calculator"""
        try:
            from fairchem.core import FAIRChemCalculator
            from fairchem.core.units.mlip_unit import load_predict_unit

            print(f"Loading UMA model: {self.model_name} from {self.model_path}")

            predictor = load_predict_unit(
                path=self.model_path,
                device=self.device,
            )

            self.calculator = FAIRChemCalculator(predictor, task_name="omat")

            # predictor.model is AveragedModel; .module is the actual backbone
            self.model = predictor.model.module

            # Register hook for latent extraction
            self._register_hook()

            print(f"Loaded UMA model: {self.model_name}")
            print(f"Model type: {type(self.model).__name__}")
            print(f"Device: {self.device}")

        except Exception as e:
            raise RuntimeError(f"Failed to load UMA model: {e}")

    def _register_hook(self):
        """Register forward hook to capture latent vectors"""
        def hook_fn(module, input, output):
            # Store the output (latent vectors)
            if isinstance(output, tuple):
                output = output[0]
            self.latent_vectors = output.detach().cpu().numpy()

        # Navigate to the target layer
        target_module = self.model
        for part in self.extraction_layer.split('.'):
            target_module = getattr(target_module, part)

        self.hook_handle = target_module.register_forward_hook(hook_fn)
        print(f"Registered hook on layer: {self.extraction_layer}")

    def extract_single(self, atoms: Atoms) -> Dict[str, Any]:
        """
        Extract latent vectors for a single structure

        Args:
            atoms: ASE Atoms object

        Returns:
            Dictionary containing latent vectors and metadata
        """
        # Reset stored vectors
        self.latent_vectors = None

        # Set calculator and compute energy (triggers forward pass)
        atoms_copy = atoms.copy()
        atoms_copy.calc = self.calculator

        try:
            # This triggers the forward pass and hook
            _ = atoms_copy.get_potential_energy()

            if self.latent_vectors is None:
                raise ValueError("Failed to capture latent vectors")

            # Get the latent vectors
            latent = self.latent_vectors.copy()

            # For UMA, the output might be batched or include padding
            # Shape could be (batch_size, n_atoms, hidden_dim) or (n_atoms, hidden_dim)
            if len(latent.shape) == 3:
                latent = latent[0]  # Take first batch

            # Verify shape and trim if needed
            n_atoms = len(atoms)
            if latent.shape[0] > n_atoms:
                # UMA might pad atoms, extract only the real atoms
                latent = latent[:n_atoms]
            elif latent.shape[0] < n_atoms:
                raise ValueError(f"Shape mismatch: expected at least {n_atoms} atoms, got {latent.shape[0]}")

            return {
                'latent_vectors': latent,
                'shape': latent.shape,
                'model': self.model_name,
                'extraction_layer': self.extraction_layer,
                'n_atoms': n_atoms,
                'latent_dim': latent.shape[1]
            }

        except Exception as e:
            print(f"Extraction failed: {e}")
            return None

    def extract_batch(self, atoms_list: List[Atoms]) -> List[Dict[str, Any]]:
        """
        Extract latent vectors for multiple structures

        Args:
            atoms_list: List of ASE Atoms objects

        Returns:
            List of extraction results
        """
        results = []
        for atoms in atoms_list:
            result = self.extract_single(atoms)
            results.append(result)

            # Clear GPU cache periodically
            if len(results) % 100 == 0 and self.device == "cuda":
                torch.cuda.empty_cache()

        return results

    def cleanup(self):
        """Remove hook and cleanup"""
        if self.hook_handle:
            self.hook_handle.remove()
            print("Removed hook")

        if self.device == "cuda":
            torch.cuda.empty_cache()


def create_uma_extractor(model_name: str = "uma_s_1p1") -> UMALatentExtractor:
    """Factory function to create UMA extractor with appropriate settings"""

    model_configs = {
        "uma_s_1p1": {
            "model_path": "/DATA/user_scratch/pn50212/2024/12_AtomAttention/MLPs/uma-s-1p1.pt"
        },
        "uma_m_1p1": {
            "model_path": "/DATA/user_scratch/pn50212/2024/12_AtomAttention/MLPs/uma-m-1p1.pt"
        }
    }

    if model_name not in model_configs:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(model_configs.keys())}")

    config = model_configs[model_name]
    return UMALatentExtractor(
        model_name=model_name,
        model_path=config["model_path"]
    )