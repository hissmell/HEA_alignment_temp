#!/usr/bin/env python3
"""
Script to explore MACE architecture and identify energy prediction head.
MACE (Multi-Atomic Cluster Expansion) model analysis.
"""

import torch
import numpy as np
from pathlib import Path
from ase import Atoms
from ase.build import bulk, add_adsorbate
import json

# Import MACE
try:
    from mace.calculators import MACECalculator
    HAS_MACE = True
except ImportError:
    print("Warning: mace-torch not installed, trying alternative import")
    HAS_MACE = False

# Try alternative imports
if not HAS_MACE:
    try:
        from mace import calculators
        HAS_MACE = True
    except:
        pass


def explore_mace_model():
    """
    Load MACE model and explore its architecture to find energy head.
    """
    print(f"\n{'='*80}")
    print(f"Exploring MACE architecture")
    print('='*80)

    # Check if MACE is available
    if not HAS_MACE:
        print("Error: MACE not installed. Please install mace-torch")
        print("  pip install mace-torch")
        return None, {}

    # Load MACE model
    print("\n[1] Loading MACE model...")

    # Path to MACE model
    model_path = "/DATA/user_scratch/pn50212/2024/12_AtomAttention/MLPs/mace-mh-0.model"

    if not Path(model_path).exists():
        print(f"Error: Model file not found at {model_path}")
        return None, {}

    try:
        # Load MACE calculator
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"  Device: {device}")

        # Try loading with MACECalculator
        calc = MACECalculator(
            model_path=model_path,
            device=device,
            default_dtype='float32',
            model_type='MACE'
        )

        print(f"  ✓ MACE model loaded successfully")

        # Get the underlying model
        model = calc.models[0]  # MACE calculator can have multiple models
        print(f"  Model type: {type(model)}")

    except Exception as e:
        print(f"Error loading MACE model: {e}")
        return None, {}

    # List all modules
    print("\n[2] Model architecture:")
    print("="*60)

    all_modules = []
    for name, module in model.named_modules():
        if name:  # Skip empty root
            all_modules.append((name, module))
            print(f"{name}: {module.__class__.__name__}")

    # Look for energy-related layers
    print("\n[3] Energy-related layers:")
    print("="*60)

    energy_layers = []
    readout_layers = []

    for name, module in model.named_modules():
        # Look for keywords related to energy prediction
        if any(keyword in name.lower() for keyword in ['energy', 'head', 'output', 'readout', 'final']):
            print(f"{name}: {module.__class__.__name__}")
            energy_layers.append((name, module))

            # Check for linear layers
            if hasattr(module, 'weight'):
                if hasattr(module.weight, 'shape'):
                    print(f"  └─ Weight shape: {module.weight.shape}")

            # Special attention to readout
            if 'readout' in name.lower():
                readout_layers.append((name, module))
                print(f"  💡 Potential latent vector source!")

    # Create test structure
    print("\n[4] Testing with sample structure:")
    print("="*60)

    # Create test atoms
    atoms = bulk('Pt', 'fcc', a=3.92)
    atoms = atoms * (2, 2, 2)
    # Add O atom manually to avoid the position name issue
    from ase import Atom
    atoms.append(Atom('O', position=[atoms[0].position[0], atoms[0].position[1], atoms[0].position[2] + 2.0]))
    print(f"  Test structure: {len(atoms)} atoms, {atoms.get_chemical_formula()}")

    atoms.calc = calc

    # Register hooks to capture intermediate outputs
    captured_tensors = {}

    def get_activation(name):
        def hook(model, input, output):
            if isinstance(output, torch.Tensor):
                captured_tensors[name] = {
                    'shape': tuple(output.shape),
                    'mean': float(output.mean().item()) if output.numel() > 0 else 0,
                    'std': float(output.std().item()) if output.numel() > 1 else 0,
                    'tensor': output.detach().cpu().numpy()
                }
                print(f"\n  Captured {name}:")
                print(f"    Shape: {output.shape}")
            elif isinstance(output, (tuple, list)):
                print(f"\n  Captured {name} (multiple outputs):")
                for i, out in enumerate(output):
                    if isinstance(out, torch.Tensor):
                        captured_tensors[f"{name}_out{i}"] = {
                            'shape': tuple(out.shape),
                            'tensor': out.detach().cpu().numpy()
                        }
                        print(f"    Output {i}: {out.shape}")
        return hook

    # Register hooks on important layers
    hooks = []

    # Try to hook readout layers
    for name, module in readout_layers:
        try:
            hook = module.register_forward_hook(get_activation(name))
            hooks.append(hook)
            print(f"  ✓ Hooked: {name}")
        except:
            print(f"  ✗ Could not hook: {name}")

    # Also try common MACE layer names
    common_layers = [
        'readout_mlp',
        'node_feats_down',
        'conv_tp_weights',
        'linear_1',
        'linear_2',
        'final_linear'
    ]

    for layer_name in common_layers:
        if hasattr(model, layer_name):
            try:
                module = getattr(model, layer_name)
                hook = module.register_forward_hook(get_activation(layer_name))
                hooks.append(hook)
                print(f"  ✓ Hooked: {layer_name}")
            except:
                pass

    # Forward pass
    print("\n[5] Running forward pass...")
    try:
        energy = atoms.get_potential_energy()
        print(f"  Energy: {energy:.6f} eV")
    except Exception as e:
        print(f"  Error during forward pass: {e}")

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Analyze captured tensors
    print("\n[6] Analysis of captured tensors:")
    print("="*60)

    if not captured_tensors:
        print("  ⚠ No tensors were captured!")
    else:
        # Find per-atom representations
        per_atom_reps = []
        for name, info in captured_tensors.items():
            shape = info['shape']
            if len(shape) >= 2 and (shape[0] == len(atoms) or shape[1] == len(atoms)):
                per_atom_reps.append((name, info))
                print(f"\n{name}:")
                print(f"  Shape: {shape}")
                print(f"  Likely per-atom features!")

        # Find the pre-energy layer
        print("\n[7] Identifying latent vector layer:")
        print("="*60)

        if per_atom_reps:
            # The last per-atom representation before energy
            last_per_atom = per_atom_reps[-1]
            print(f"  Best candidate: {last_per_atom[0]}")
            print(f"  Shape: {last_per_atom[1]['shape']}")

    # Check model attributes for more clues
    print("\n[8] Model attributes analysis:")
    print("="*60)

    important_attrs = []
    for attr_name in dir(model):
        if not attr_name.startswith('_'):
            attr = getattr(model, attr_name)
            if isinstance(attr, torch.nn.Module):
                important_attrs.append(attr_name)
                print(f"  {attr_name}: {type(attr).__name__}")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    print("\nMACE architecture key points:")
    print("  1. MACE uses message passing with ACE descriptors")
    print("  2. Energy prediction happens through readout layers")
    print("  3. Look for 'readout_mlp' or similar layers for latent vectors")

    if readout_layers:
        print(f"\nFound {len(readout_layers)} readout layers:")
        for name, _ in readout_layers:
            print(f"  - {name}")

    return model, captured_tensors


def test_mace_models():
    """
    Test different MACE model variants.
    """
    print("\n" + "="*80)
    print("Testing MACE Model Variants")
    print("="*80)

    # MACE has different heads for different datasets
    heads = ['omat_pbe', 'oc20_usemppbe']  # Different heads in mace-mh-0

    for head_name in heads:
        print(f"\n--- Testing head: {head_name} ---")

        try:
            # Note: MACE-MH (multi-head) models can switch between heads
            # This would require specific configuration
            print(f"  Head configuration would go here")

        except Exception as e:
            print(f"  Error: {e}")


def main():
    """
    Main exploration function.
    """
    print("="*80)
    print("MACE Architecture Exploration")
    print("="*80)

    # Explore MACE architecture
    model, captured = explore_mace_model()

    if model is not None:
        # Save results
        results = {
            'model': 'MACE',
            'captured_layers': list(captured.keys()) if captured else [],
            'recommendations': {
                'latent_layer': 'readout_mlp or final layer before energy',
                'notes': 'MACE uses readout layers for aggregation'
            }
        }

        with open('mace_architecture_analysis.json', 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Results saved to: mace_architecture_analysis.json")

    # Test variants
    test_mace_models()

    print("\n" + "="*80)
    print("Recommendation for mlp_infos.yml:")
    print("="*80)
    print("Update the 'output_layer' field with the identified readout layer")
    print("This is typically 'readout_mlp' or the last layer before energy calculation")


if __name__ == "__main__":
    main()