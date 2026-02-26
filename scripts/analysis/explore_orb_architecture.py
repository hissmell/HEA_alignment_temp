#!/usr/bin/env python3
"""
Script to explore Orb architecture and identify energy prediction head.
Orb is a modern MLIP with strong performance on materials systems.
"""

import torch
import numpy as np
from pathlib import Path
from ase import Atoms
from ase.build import bulk
from ase import Atom
import json

# Import Orb
try:
    from orb_models.forcefield import pretrained
    from orb_models.forcefield.calculator import ORBCalculator
    HAS_ORB = True
    print("✓ Orb models imported successfully")
except ImportError as e:
    print(f"Warning: Trying alternative import: {e}")
    try:
        from orb_models.forcefield import pretrained
        from orb_models.calculator import ORBCalculator
        HAS_ORB = True
        print("✓ Orb models imported successfully (alternative)")
    except:
        HAS_ORB = False
        print("✗ Could not import Orb")

# Try alternative imports
if not HAS_ORB:
    try:
        import orbax
        print("Found orbax, trying alternative import...")
    except:
        pass


def explore_orb_model(model_name="orb_v3_conservative_inf_omat"):
    """
    Load Orb model and explore its architecture to find energy head.
    """
    print(f"\n{'='*80}")
    print(f"Exploring Orb architecture: {model_name}")
    print('='*80)

    if not HAS_ORB:
        print("Error: Orb not installed. Please install orb-models")
        print("  pip install orb-models")
        return None, {}

    # Load Orb model
    print("\n[1] Loading Orb model...")

    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"  Device: {device}")

        # Load pretrained Orb model (using precision="float32-high" as per documentation)
        if model_name == "orb_v3_conservative_inf_omat":
            orbff = pretrained.orb_v3_conservative_inf_omat(device=device, precision="float32-high")
        elif model_name == "orb_v3_conservative_inf_mpa":
            orbff = pretrained.orb_v3_conservative_inf_mpa(device=device, precision="float32-high")
        elif model_name == "orb_v3_direct_inf_omat":
            orbff = pretrained.orb_v3_direct_inf_omat(device=device, precision="float32-high")
        elif model_name == "orb_v3_direct_inf_mpa":
            orbff = pretrained.orb_v3_direct_inf_mpa(device=device, precision="float32-high")
        else:
            print(f"Unknown model: {model_name}")
            return None, {}

        # Create calculator
        calc = ORBCalculator(orbff, device=device)
        print(f"  ✓ Orb model loaded: {model_name}")

        # Get the underlying model
        model = orbff  # In Orb, the forcefield itself is the model
        print(f"  Model type: {type(model)}")

    except Exception as e:
        print(f"Error loading Orb model: {e}")
        import traceback
        traceback.print_exc()
        return None, {}

    # List all modules
    print("\n[2] Model architecture:")
    print("="*60)

    all_modules = []
    for name, module in model.named_modules():
        if name:  # Skip empty root
            all_modules.append((name, module))
            print(f"{name}: {module.__class__.__name__}")

            # Check for important attributes
            if hasattr(module, 'weight'):
                if hasattr(module.weight, 'shape'):
                    print(f"  └─ Weight shape: {module.weight.shape}")

    # Look for energy-related layers
    print("\n[3] Energy-related layers:")
    print("="*60)

    energy_layers = []
    important_layers = []

    for name, module in model.named_modules():
        # Look for keywords related to energy prediction
        keywords = ['energy', 'head', 'output', 'readout', 'final', 'predictor', 'decoder']
        if any(keyword in name.lower() for keyword in keywords):
            print(f"{name}: {module.__class__.__name__}")
            energy_layers.append((name, module))

            # Check module type
            module_type = module.__class__.__name__
            if 'Linear' in module_type or 'Dense' in module_type:
                print(f"  💡 Potential latent vector source!")
                important_layers.append((name, module))

    # Also look for specific Orb patterns
    print("\n[4] Orb-specific layers:")
    print("="*60)

    # Orb might use specific naming conventions
    orb_patterns = ['node_embedding', 'message_passing', 'aggregation', 'update', 'decoder']
    for pattern in orb_patterns:
        for name, module in all_modules:
            if pattern in name.lower():
                print(f"{name}: {module.__class__.__name__}")

    # Create test structure
    print("\n[5] Testing with sample structure:")
    print("="*60)

    # Create test atoms
    atoms = bulk('Pt', 'fcc', a=3.92)
    atoms = atoms * (2, 2, 2)
    # Add O atom manually
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

                # Check if it's per-atom
                if len(output.shape) >= 2:
                    if output.shape[0] == len(atoms) or output.shape[1] == len(atoms):
                        print(f"    → Likely per-atom features!")

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

    # Try to hook energy/output layers
    for name, module in energy_layers[:10]:  # Limit to first 10 to avoid too many hooks
        try:
            hook = module.register_forward_hook(get_activation(name))
            hooks.append(hook)
            print(f"  ✓ Hooked: {name}")
        except Exception as e:
            print(f"  ✗ Could not hook {name}: {e}")

    # Also try to hook the last few layers
    if len(all_modules) > 5:
        for name, module in all_modules[-5:]:
            if name not in [n for n, _ in energy_layers]:
                try:
                    hook = module.register_forward_hook(get_activation(f"last_{name}"))
                    hooks.append(hook)
                    print(f"  ✓ Hooked last layer: {name}")
                except:
                    pass

    # Forward pass
    print("\n[6] Running forward pass...")
    try:
        energy = atoms.get_potential_energy()
        print(f"  Energy: {energy:.6f} eV")
    except Exception as e:
        print(f"  Error during forward pass: {e}")
        import traceback
        traceback.print_exc()

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Analyze captured tensors
    print("\n[7] Analysis of captured tensors:")
    print("="*60)

    if not captured_tensors:
        print("  ⚠ No tensors were captured!")
        print("  This might be due to Orb's architecture or compilation")
    else:
        # Find per-atom representations
        per_atom_reps = []
        for name, info in captured_tensors.items():
            shape = info['shape']
            # Check for per-atom shape
            if len(shape) >= 2:
                if shape[0] == len(atoms):
                    per_atom_reps.append((name, info))
                    print(f"\n{name}:")
                    print(f"  Shape: {shape}")
                    print(f"  Type: Per-atom features ({shape[0]} atoms × {shape[1]} features)")
                elif shape[1] == len(atoms):
                    per_atom_reps.append((name, info))
                    print(f"\n{name}:")
                    print(f"  Shape: {shape}")
                    print(f"  Type: Features × atoms")

        # Find the pre-energy layer
        print("\n[8] Identifying latent vector layer:")
        print("="*60)

        if per_atom_reps:
            # The last per-atom representation before energy
            last_per_atom = per_atom_reps[-1]
            print(f"  Best candidate: {last_per_atom[0]}")
            print(f"  Shape: {last_per_atom[1]['shape']}")
            print(f"  This is likely the latent vector!")
        else:
            print("  No clear per-atom representations found")
            print("  Orb might use a different architecture pattern")

    # Check for specific Orb model attributes
    print("\n[9] Orb model specific attributes:")
    print("="*60)

    # Check what attributes the model has
    model_attrs = [attr for attr in dir(model) if not attr.startswith('_')]
    relevant_attrs = [attr for attr in model_attrs if any(
        keyword in attr.lower() for keyword in ['energy', 'output', 'head', 'decoder', 'predictor']
    )]

    if relevant_attrs:
        print("  Found relevant attributes:")
        for attr in relevant_attrs:
            print(f"    - {attr}")
            try:
                attr_obj = getattr(model, attr)
                if isinstance(attr_obj, torch.nn.Module):
                    print(f"      Type: {type(attr_obj).__name__}")
            except:
                pass
    else:
        print("  No obvious energy-related attributes found")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    print("\nOrb architecture insights:")
    print("  1. Orb uses graph neural networks for molecular systems")
    print("  2. Energy prediction likely happens through decoder/output layers")
    print("  3. Look for the last linear/dense layer before energy output")

    if energy_layers:
        print(f"\nFound {len(energy_layers)} energy-related layers")
        for name, _ in energy_layers[:5]:
            print(f"  - {name}")

    if important_layers:
        print(f"\nMost likely latent vector sources:")
        for name, _ in important_layers[:3]:
            print(f"  - {name}")

    return model, captured_tensors


def test_orb_variants():
    """
    Test different Orb model variants.
    """
    print("\n" + "="*80)
    print("Testing Orb Model Variants")
    print("="*80)

    models = [
        "orb_v3_conservative_inf_omat",
        "orb_v3_conservative_inf_mpa",
        "orb_v3_direct_inf_omat",
        "orb_v3_direct_inf_mpa"
    ]

    results = {}

    for model_name in models:
        print(f"\n--- Testing: {model_name} ---")
        try:
            model, captured = explore_orb_model(model_name)
            if model is not None:
                results[model_name] = {
                    'success': True,
                    'captured_layers': list(captured.keys()) if captured else []
                }
            else:
                results[model_name] = {'success': False}
        except Exception as e:
            print(f"  Error: {e}")
            results[model_name] = {'success': False, 'error': str(e)}

    return results


def main():
    """
    Main exploration function.
    """
    print("="*80)
    print("Orb Architecture Exploration")
    print("="*80)

    # First try the main model
    model, captured = explore_orb_model("orb_v3_conservative_inf_omat")

    # Save results
    if model is not None or captured:
        results = {
            'model': 'Orb-v3',
            'captured_layers': list(captured.keys()) if captured else [],
            'recommendations': {
                'latent_layer': 'Look for decoder or final linear layer',
                'notes': 'Orb uses graph neural networks, latent vectors are node features before final energy prediction'
            }
        }

        with open('orb_architecture_analysis.json', 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Results saved to: orb_architecture_analysis.json")

    # Test all variants
    # Uncomment if you want to test all models
    # variant_results = test_orb_variants()

    print("\n" + "="*80)
    print("Recommendation for mlp_infos.yml:")
    print("="*80)
    print("Update the 'output_layer' field with the identified decoder/output layer")
    print("This is typically the last linear layer before energy aggregation")


if __name__ == "__main__":
    main()