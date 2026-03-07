#!/usr/bin/env python3
"""
Debug Equiformer extraction to understand why latent vectors aren't being captured
"""

import torch
import numpy as np
from pathlib import Path
from ase.io import read
from ase import Atoms
import sys
import json

sys.path.append('/DATA/user_scratch/pn50212/2024/12_AtomAttention')


def explore_equiformer_extraction():
    """Detailed exploration of Equiformer model for latent extraction"""

    print("=" * 70)
    print("Equiformer Model Architecture and Extraction Debug")
    print("=" * 70)

    # Import after path setup
    from fairchem.core.common.relaxation.ase_utils import OCPCalculator

    # Load model
    model_path = "/DATA/user_scratch/pn50212/2024/12_AtomAttention/MLPs/eqV2_31M_omat.pt"
    print(f"\nLoading model from: {model_path}")

    calculator = OCPCalculator(
        checkpoint_path=model_path,
        cpu=False
    )

    model = calculator.trainer.model
    print(f"Model type: {type(model).__name__}")
    print(f"Model class: {model.__class__}")

    # List all modules
    print("\n" + "="*50)
    print("ALL MODEL MODULES:")
    print("="*50)

    all_modules = []
    for name, module in model.named_modules():
        module_type = type(module).__name__
        all_modules.append((name, module_type))
        if name:  # Skip empty root
            print(f"  {name}: {module_type}")

    # Find energy-related modules
    print("\n" + "="*50)
    print("ENERGY/OUTPUT RELATED MODULES:")
    print("="*50)

    for name, module_type in all_modules:
        if any(x in name.lower() for x in ['energy', 'output', 'head', 'final', 'readout']):
            print(f"  {name}: {module_type}")

    # Check if model has specific attributes
    print("\n" + "="*50)
    print("MODEL ATTRIBUTES:")
    print("="*50)

    important_attrs = ['output_heads', 'energy_head', 'heads', 'readout',
                      'output_block', 'decoder', 'energy_block']

    for attr in important_attrs:
        if hasattr(model, attr):
            obj = getattr(model, attr)
            print(f"  {attr}: {type(obj).__name__}")

            # If it's a module, show its structure
            if hasattr(obj, 'named_modules'):
                for sub_name, sub_module in obj.named_modules():
                    if sub_name:
                        print(f"    └─ {sub_name}: {type(sub_module).__name__}")

    # Test extraction with hooks
    print("\n" + "="*50)
    print("TESTING EXTRACTION WITH MULTIPLE HOOK LOCATIONS:")
    print("="*50)

    # Create a simple test structure
    atoms = Atoms('Cu3', positions=[[0, 0, 0], [2, 0, 0], [1, 1, 0]],
                  cell=[10, 10, 10], pbc=True)

    # Storage for captured tensors
    captured_tensors = {}
    hook_handles = []

    def make_hook(layer_name):
        def hook_fn(module, input, output):
            print(f"\n  Hook triggered for: {layer_name}")

            # Capture input
            if isinstance(input, tuple):
                for i, inp in enumerate(input):
                    if inp is not None and hasattr(inp, 'shape'):
                        print(f"    Input[{i}] shape: {inp.shape}")
                        captured_tensors[f"{layer_name}_input_{i}"] = inp
            elif input is not None and hasattr(input, 'shape'):
                print(f"    Input shape: {input.shape}")
                captured_tensors[f"{layer_name}_input"] = input

            # Capture output
            if isinstance(output, tuple):
                for i, out in enumerate(output):
                    if out is not None and hasattr(out, 'shape'):
                        print(f"    Output[{i}] shape: {out.shape}")
                        captured_tensors[f"{layer_name}_output_{i}"] = out
            elif output is not None and hasattr(output, 'shape'):
                print(f"    Output shape: {output.shape}")
                captured_tensors[f"{layer_name}_output"] = output

            # Special handling for graph batches
            if hasattr(output, 'x') and hasattr(output, 'batch'):
                print(f"    Graph batch detected - x shape: {output.x.shape}")
                captured_tensors[f"{layer_name}_graph_x"] = output.x

        return hook_fn

    # Register hooks on various layers
    target_layers = []

    # Find promising layers
    for name, module in model.named_modules():
        if any(x in name.lower() for x in ['output', 'energy', 'head', 'final']):
            if 'linear' in type(module).__name__.lower() or 'block' in name.lower():
                target_layers.append((name, module))

    # Register hooks
    print("\nRegistering hooks on layers:")
    for layer_name, module in target_layers[:5]:  # Limit to 5 for debugging
        print(f"  - {layer_name}")
        handle = module.register_forward_hook(make_hook(layer_name))
        hook_handles.append(handle)

    # Run forward pass
    print("\n" + "="*50)
    print("RUNNING FORWARD PASS:")
    print("="*50)

    atoms_copy = atoms.copy()
    atoms_copy.calc = calculator

    try:
        energy = atoms_copy.get_potential_energy()
        print(f"\nEnergy calculated: {energy:.6f} eV")
    except Exception as e:
        print(f"\nError during forward pass: {e}")

    # Analyze captured tensors
    print("\n" + "="*50)
    print("CAPTURED TENSORS SUMMARY:")
    print("="*50)

    for key, tensor in captured_tensors.items():
        if hasattr(tensor, 'shape'):
            print(f"  {key}: shape={tensor.shape}, dtype={tensor.dtype}")

    # Look for per-atom features
    print("\n" + "="*50)
    print("POTENTIAL PER-ATOM FEATURES:")
    print("="*50)

    n_atoms = len(atoms)
    for key, tensor in captured_tensors.items():
        if hasattr(tensor, 'shape'):
            if len(tensor.shape) >= 2 and tensor.shape[0] == n_atoms:
                print(f"  {key}: shape={tensor.shape} ✓ (matches n_atoms={n_atoms})")

    # Clean up hooks
    for handle in hook_handles:
        handle.remove()

    print("\n" + "="*70)
    print("Debug complete!")

    return captured_tensors


if __name__ == "__main__":
    captured = explore_equiformer_extraction()