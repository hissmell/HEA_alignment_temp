#!/usr/bin/env python3
"""
Script to explore SevenNet architecture and identify energy prediction head.
This will help us find the layer that outputs features before energy calculation.
"""

import torch
import numpy as np
from pathlib import Path
from ase import Atoms
from ase.build import bulk, add_adsorbate

# Import SevenNet
from sevenn.calculator import SevenNetCalculator
# from sevenn.pretrained_potentials import pretrained_name_to_path

def explore_sevennet_model(model_name="SevenNet_0"):
    """
    Load SevenNet model and explore its architecture to find energy head.
    """
    print(f"\n{'='*60}")
    print(f"Exploring {model_name} architecture")
    print('='*60)

    # Get model path
    if model_name in ["SevenNet_0", "7net-0"]:
        # Use default SevenNet-0
        calc = SevenNetCalculator("7net-0", device='cpu')
    else:
        # Load from specific path
        model_paths = {
            "SevenNet-Omni": "/home/pn50212/anaconda3/envs/fairchem/lib/python3.10/site-packages/sevenn/pretrained_potentials/SevenNet_omni/checkpoint_sevennet_omni.pth",
            "SevenNet-MF-ompa": "/home/pn50212/anaconda3/envs/fairchem/lib/python3.10/site-packages/sevenn/pretrained_potentials/SevenNet_MF_ompa/checkpoint_sevennet_mf_ompa.pth",
            "SevenNet-omat": "/home/pn50212/anaconda3/envs/fairchem/lib/python3.10/site-packages/sevenn/pretrained_potentials/SevenNet_omat/checkpoint_sevennet_omat.pth"
        }

        if model_name in model_paths:
            calc = SevenNetCalculator(model_paths[model_name], device='cpu')
        else:
            print(f"Unknown model: {model_name}")
            return

    # Access the underlying model
    model = calc.model
    print(f"\nModel type: {type(model)}")

    # List all modules
    print("\n" + "="*40)
    print("All modules in SevenNet model:")
    print("="*40)

    for name, module in model.named_modules():
        if name:  # Skip empty root
            print(f"{name}: {module.__class__.__name__}")

    # Look for energy-related layers
    print("\n" + "="*40)
    print("Energy-related layers:")
    print("="*40)

    energy_layers = []
    for name, module in model.named_modules():
        if any(keyword in name.lower() for keyword in ['energy', 'head', 'output', 'readout', 'fc', 'linear']):
            print(f"{name}: {module.__class__.__name__}")
            if hasattr(module, 'in_features'):
                print(f"  Input dim: {module.in_features}")
            if hasattr(module, 'out_features'):
                print(f"  Output dim: {module.out_features}")
            energy_layers.append((name, module))

    # Create a test structure to trace through
    print("\n" + "="*40)
    print("Testing with sample structure:")
    print("="*40)

    # Create simple test structure
    atoms = bulk('Pt', 'fcc', a=3.92, cubic=True)
    atoms = atoms * (2, 2, 2)
    add_adsorbate(atoms, 'O', 2.0, 'ontop')
    atoms.set_calculator(calc)

    # Register hooks to capture intermediate outputs
    activations = {}

    def get_activation(name):
        def hook(model, input, output):
            if isinstance(output, torch.Tensor):
                activations[name] = output.detach().cpu()
                print(f"\nCaptured {name}:")
                print(f"  Shape: {output.shape}")
                print(f"  Mean: {output.mean().item():.6f}")
                print(f"  Std: {output.std().item():.6f}")
            elif isinstance(output, (tuple, list)):
                print(f"\nCaptured {name} (multiple outputs):")
                for i, out in enumerate(output):
                    if isinstance(out, torch.Tensor):
                        activations[f"{name}_{i}"] = out.detach().cpu()
                        print(f"  Output {i} - Shape: {out.shape}")
        return hook

    # Register hooks on potential energy head layers
    hooks = []
    for name, module in energy_layers[-5:]:  # Last 5 layers
        hook = module.register_forward_hook(get_activation(name))
        hooks.append(hook)

    # Forward pass
    print("\nPerforming forward pass...")
    energy = atoms.get_potential_energy()
    print(f"\nTotal energy: {energy:.6f} eV")

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Analyze captured activations
    print("\n" + "="*40)
    print("Analysis of captured activations:")
    print("="*40)

    # Find the layer just before energy output
    pre_energy_layer = None
    for name, tensor in activations.items():
        print(f"\n{name}:")
        print(f"  Shape: {tensor.shape}")
        print(f"  Dimension: {tensor.numel()}")

        # The pre-energy layer typically has shape (1, hidden_dim) or (n_atoms, hidden_dim)
        if len(tensor.shape) == 2:
            if tensor.shape[0] == 1 or tensor.shape[0] == len(atoms):
                pre_energy_layer = name
                print(f"  -> Likely pre-energy layer!")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"Pre-energy layer candidate: {pre_energy_layer}")

    # Try to identify the exact layer name pattern
    if hasattr(model, 'readout_module'):
        print(f"Has readout_module: {type(model.readout_module)}")
        for name, module in model.readout_module.named_modules():
            if name:
                print(f"  readout.{name}: {module.__class__.__name__}")

    return pre_energy_layer, activations


def main():
    """
    Explore all SevenNet models to identify energy heads.
    """
    models_to_test = [
        "7net-0",  # Default SevenNet-0
        "SevenNet-Omni",
        # "SevenNet-MF-ompa",  # Comment out if not available
        # "SevenNet-omat"      # Comment out if not available
    ]

    results = {}

    for model_name in models_to_test:
        try:
            pre_energy_layer, activations = explore_sevennet_model(model_name)
            results[model_name] = {
                'pre_energy_layer': pre_energy_layer,
                'activation_shapes': {k: v.shape for k, v in activations.items()}
            }
        except Exception as e:
            print(f"\nError exploring {model_name}: {e}")
            results[model_name] = {'error': str(e)}

    # Print summary
    print("\n" + "="*60)
    print("FINAL SUMMARY - Energy Head Layers")
    print("="*60)

    for model_name, info in results.items():
        print(f"\n{model_name}:")
        if 'error' in info:
            print(f"  Error: {info['error']}")
        else:
            print(f"  Pre-energy layer: {info['pre_energy_layer']}")
            if info['activation_shapes']:
                print(f"  Activation shapes:")
                for name, shape in info['activation_shapes'].items():
                    print(f"    {name}: {shape}")

    print("\n" + "="*60)
    print("Recommendation for mlp_infos.yml:")
    print("="*60)
    print("Update the 'output_layer' field with the identified pre-energy layer name")
    print("This is typically the layer that outputs features before final energy calculation")


if __name__ == "__main__":
    main()