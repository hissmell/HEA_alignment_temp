"""
Equiformer Latent Vector Extractor

Extracts per-atom latent representations from EquiformerV2 models
by hooking into the appropriate layer before energy prediction.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path
from ase import Atoms
import warnings
warnings.filterwarnings('ignore')


class EquiformerLatentExtractor:
    """Extract latent vectors from EquiformerV2 models"""

    def __init__(
        self,
        model_name: str = "eqV2_31M_omat",
        model_path: str = "/DATA/user_scratch/pn50212/2024/12_AtomAttention/MLPs/eqV2_31M_omat.pt",
        device: str = "cuda"
    ):
        """
        Initialize Equiformer latent extractor

        Args:
            model_name: Model identifier
            model_path: Path to model checkpoint
            device: Device for computation
        """
        self.model_name = model_name
        self.model_path = model_path
        self.device = device if torch.cuda.is_available() else "cpu"

        self.model = None
        self.calculator = None
        self.latent_vectors = None
        self.hook_handle = None
        self.extraction_layer = None

        self._setup_model()

    def _setup_model(self):
        """Load Equiformer model and setup calculator"""
        try:
            from fairchem.core.common.relaxation.ase_utils import OCPCalculator

            # Load Equiformer calculator
            print(f"Loading Equiformer model: {self.model_name}")
            print(f"From: {self.model_path}")

            self.calculator = OCPCalculator(
                checkpoint_path=self.model_path,
                cpu=(self.device == "cpu")
            )

            # Get the model
            self.model = self.calculator.trainer.model

            # Find the appropriate layer to hook
            self._find_extraction_layer()

            # Register hook for latent extraction
            if self.extraction_layer:
                self._register_hook()

            print(f"Loaded Equiformer model: {self.model_name}")
            print(f"Model type: {type(self.model).__name__}")
            print(f"Device: {self.device}")

        except Exception as e:
            raise RuntimeError(f"Failed to load Equiformer model: {e}")

    def _find_extraction_layer(self):
        """Find the appropriate layer to extract latent vectors from"""

        # For EquiformerV2 (HydraModel), the best extraction point is:
        # output_heads.energy.energy_block.scalar_mlp
        # This gives per-atom features with shape [n_atoms, 1, 128]

        target_layer = "output_heads.energy.energy_block.scalar_mlp"

        for name, module in self.model.named_modules():
            if name == target_layer:
                self.extraction_layer = name
                self.extraction_module = module
                print(f"Selected extraction layer: {name}")
                return

        # Fallback: try alternative energy-related layers
        print(f"Warning: Could not find {target_layer}, searching alternatives...")

        possible_layers = []
        for name, module in self.model.named_modules():
            # Look for energy block components
            if 'energy_block' in name and 'scalar' in name:
                possible_layers.append((name, module))
            elif 'energy' in name.lower() and 'mlp' in name.lower():
                possible_layers.append((name, module))

        if possible_layers:
            self.extraction_layer = possible_layers[0][0]
            self.extraction_module = possible_layers[0][1]
            print(f"Using alternative extraction layer: {self.extraction_layer}")
        else:
            print("Warning: Could not find suitable extraction layer")
            self.extraction_layer = None
            self.extraction_module = None

    def _register_hook(self):
        """Register forward hook to capture latent vectors"""

        def hook_fn(module, input, output):
            # For scalar_mlp, we want the input (features before MLP)
            # Shape is typically [n_atoms, 1, hidden_dim]
            if isinstance(input, tuple):
                tensor = input[0]
            else:
                tensor = input

            # Handle different tensor formats
            if hasattr(tensor, 'x'):
                # EquiformerV2 uses a special data structure
                tensor = tensor.x

            # Convert to numpy and handle shape
            latent = tensor.detach().cpu().numpy()

            # If shape is [n_atoms, 1, hidden_dim], squeeze the middle dimension
            if len(latent.shape) == 3 and latent.shape[1] == 1:
                latent = latent.squeeze(1)  # Now [n_atoms, hidden_dim]

            self.latent_vectors = latent

        if hasattr(self, 'extraction_module'):
            self.hook_handle = self.extraction_module.register_forward_hook(hook_fn)
            print(f"Registered hook on layer: {self.extraction_layer}")
        else:
            print("Warning: No extraction module found, will try default approach")

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
                # Try alternative extraction if hook didn't work
                # Get the internal batch representation
                if hasattr(self.calculator.trainer, 'batch'):
                    batch = self.calculator.trainer.batch
                    if hasattr(batch, 'x'):
                        self.latent_vectors = batch.x.detach().cpu().numpy()

                if self.latent_vectors is None:
                    raise ValueError("Failed to capture latent vectors")

            # Get the latent vectors
            latent = self.latent_vectors.copy()

            # Handle batched data
            if len(latent.shape) == 3:
                latent = latent[0]  # Take first batch

            # Verify and adjust shape if needed
            n_atoms = len(atoms)

            # EquiformerV2 might include padding or extra nodes
            if latent.shape[0] > n_atoms:
                # Trim to actual number of atoms
                latent = latent[:n_atoms]
            elif latent.shape[0] < n_atoms:
                print(f"Warning: Expected {n_atoms} atoms, got {latent.shape[0]}")

            return {
                'latent_vectors': latent,
                'shape': latent.shape,
                'model': self.model_name,
                'extraction_layer': self.extraction_layer or 'default',
                'n_atoms': n_atoms,
                'latent_dim': latent.shape[1] if len(latent.shape) > 1 else 1
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


def create_equiformer_extractor(model_name: str = "eqV2_31M_omat") -> EquiformerLatentExtractor:
    """Factory function to create Equiformer extractor with appropriate settings"""

    model_configs = {
        "eqV2_31M_omat": {
            "model_path": "/DATA/user_scratch/pn50212/2024/12_AtomAttention/MLPs/eqV2_31M_omat.pt"
        },
        "eqV2_86M_omat": {
            "model_path": "/DATA/user_scratch/pn50212/2024/12_AtomAttention/MLPs/eqV2_86M_omat.pt"
        },
        "eqV2_153M_omat": {
            "model_path": "/DATA/user_scratch/pn50212/2024/12_AtomAttention/MLPs/eqV2_153M_omat.pt"
        },
        "eqV2_31M_omat_mp_salex": {
            "model_path": "/DATA/user_scratch/pn50212/2024/12_AtomAttention/MLPs/eqV2_31M_omat_mp_salex.pt"
        },
        "eqV2_86M_omat_mp_salex": {
            "model_path": "/DATA/user_scratch/pn50212/2024/12_AtomAttention/MLPs/eqV2_86M_omat_mp_salex.pt"
        },
        "eqV2_153M_omat_mp_salex": {
            "model_path": "/DATA/user_scratch/pn50212/2024/12_AtomAttention/MLPs/eqV2_153M_omat_mp_salex.pt"
        }
    }

    if model_name not in model_configs:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(model_configs.keys())}")

    config = model_configs[model_name]
    return EquiformerLatentExtractor(
        model_name=model_name,
        model_path=config["model_path"]
    )