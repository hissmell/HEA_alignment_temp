#!/usr/bin/env python3
"""
Detailed SevenNet architecture exploration to identify latent vectors.
Focus on the reduce_hidden_to_energy layer input.
"""

import torch
import numpy as np
from pathlib import Path
from ase import Atoms
from ase.build import bulk
import json

# Import SevenNet
from sevenn.calculator import SevenNetCalculator


def analyze_sevennet_layers(model_path=None):
    """
    Analyze SevenNet model to identify the latent vector layer.
    """
    print(f"\n{'='*60}")
    print("SevenNet Architecture Analysis")
    print('='*60)

    # Load model
    if model_path is None:
        calc = SevenNetCalculator("7net-0", device='cpu')
        model_name = "7net-0"
    else:
        calc = SevenNetCalculator(model_path, device='cpu')
        model_name = Path(model_path).stem

    model = calc.model

    # Find key layers
    layers_info = {}

    for name, module in model.named_modules():
        if 'reduce_hidden_to_energy' in name:
            layers_info['energy_head'] = {
                'name': name,
                'type': module.__class__.__name__
            }
            if hasattr(module, 'linear'):
                linear = module.linear
                if hasattr(linear, 'weight'):
                    layers_info['energy_head']['input_dim'] = linear.weight.shape[1]
                    layers_info['energy_head']['output_dim'] = linear.weight.shape[0]

        elif 'reduce_input_to_hidden' in name:
            layers_info['pre_energy'] = {
                'name': name,
                'type': module.__class__.__name__
            }
            if hasattr(module, 'linear'):
                linear = module.linear
                if hasattr(linear, 'weight'):
                    layers_info['pre_energy']['input_dim'] = linear.weight.shape[1]
                    layers_info['pre_energy']['output_dim'] = linear.weight.shape[0]

    # Create test structure
    atoms = bulk('Pt', 'fcc', a=3.92)
    atoms = atoms * (2, 2, 2)
    atoms.positions[0, 2] += 0.1  # Small perturbation
    atoms.set_calculator(calc)

    # Hook to capture latent vectors
    latent_vectors = {}

    def capture_input(name):
        def hook(module, input, output):
            if isinstance(input, tuple):
                for i, inp in enumerate(input):
                    if isinstance(inp, torch.Tensor):
                        latent_vectors[f"{name}_input_{i}"] = {
                            'shape': tuple(inp.shape),
                            'mean': float(inp.mean().item()),
                            'std': float(inp.std().item()),
                            'min': float(inp.min().item()),
                            'max': float(inp.max().item())
                        }
            elif isinstance(input, torch.Tensor):
                latent_vectors[f"{name}_input"] = {
                    'shape': tuple(input.shape),
                    'mean': float(input.mean().item()),
                    'std': float(input.std().item()),
                    'min': float(input.min().item()),
                    'max': float(input.max().item())
                }

            if isinstance(output, torch.Tensor):
                latent_vectors[f"{name}_output"] = {
                    'shape': tuple(output.shape),
                    'mean': float(output.mean().item()),
                    'std': float(output.std().item()),
                    'min': float(output.min().item()),
                    'max': float(output.max().item())
                }
        return hook

    # Register hooks on key layers
    hooks = []
    for name, module in model.named_modules():
        if 'reduce_hidden_to_energy' in name and 'linear' not in name:
            hook = module.register_forward_hook(capture_input(name))
            hooks.append(hook)
            print(f"Registered hook on: {name}")
        elif 'reduce_input_to_hidden' in name and 'linear' not in name:
            hook = module.register_forward_hook(capture_input(name))
            hooks.append(hook)
            print(f"Registered hook on: {name}")

    # Forward pass
    print("\nPerforming forward pass...")
    energy = atoms.get_potential_energy()
    print(f"Energy: {energy:.6f} eV")

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Analysis results
    results = {
        'model': model_name,
        'layers': layers_info,
        'latent_vectors': latent_vectors,
        'recommendation': {
            'latent_layer': 'reduce_input_to_hidden',
            'latent_layer_output': 'reduce_input_to_hidden output',
            'energy_head_input': 'reduce_hidden_to_energy input',
            'description': 'The latent vector is the OUTPUT of reduce_input_to_hidden, which is also the INPUT to reduce_hidden_to_energy'
        }
    }

    return results


def main():
    """
    Analyze all SevenNet models.
    """
    models = {
        "7net-0": None,
        "SevenNet-Omni": "/home/pn50212/anaconda3/envs/fairchem/lib/python3.10/site-packages/sevenn/pretrained_potentials/SevenNet_omni/checkpoint_sevennet_omni.pth",
        # Add more models as needed
    }

    all_results = {}

    for model_name, model_path in models.items():
        try:
            print(f"\n{'='*60}")
            print(f"Analyzing: {model_name}")
            print('='*60)

            results = analyze_sevennet_layers(model_path)
            all_results[model_name] = results

            # Print summary
            print(f"\n{model_name} Summary:")
            print(f"  Energy head: {results['layers'].get('energy_head', {}).get('name')}")
            if 'energy_head' in results['layers']:
                print(f"    Input dim: {results['layers']['energy_head'].get('input_dim')}")
                print(f"    Output dim: {results['layers']['energy_head'].get('output_dim')}")

            print(f"  Pre-energy layer: {results['layers'].get('pre_energy', {}).get('name')}")
            if 'pre_energy' in results['layers']:
                print(f"    Input dim: {results['layers']['pre_energy'].get('input_dim')}")
                print(f"    Output dim: {results['layers']['pre_energy'].get('output_dim')}")

            print(f"\n  Latent vectors captured:")
            for key, info in results['latent_vectors'].items():
                print(f"    {key}: shape={info['shape']}")

        except Exception as e:
            print(f"Error analyzing {model_name}: {e}")
            all_results[model_name] = {'error': str(e)}

    # Save results
    output_file = Path("sevennet_architecture_analysis.json")
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nResults saved to: {output_file}")

    # Print final recommendations
    print("\n" + "="*60)
    print("RECOMMENDATION FOR mlp_infos.yml")
    print("="*60)
    print("\nFor SevenNet models, update the 'output_layer' field to:")
    print("  output_layer: 'reduce_input_to_hidden'")
    print("\nExplanation:")
    print("  - reduce_input_to_hidden: Outputs the latent representation (hidden features)")
    print("  - reduce_hidden_to_energy: Takes latent features and outputs energy")
    print("  - We want to extract the OUTPUT of reduce_input_to_hidden")
    print("  - This gives us the latent vector before energy calculation")

    # Print extraction method
    print("\n" + "="*60)
    print("EXTRACTION METHOD")
    print("="*60)
    print("""
def extract_latent_vector(model, atoms_data):
    # Register hook on reduce_input_to_hidden
    latent = None
    def hook(module, input, output):
        nonlocal latent
        latent = output.detach().cpu()

    handle = model.reduce_input_to_hidden.register_forward_hook(hook)

    # Forward pass
    _ = model(atoms_data)

    # Remove hook
    handle.remove()

    return latent  # Shape: (n_atoms, hidden_dim)
""")


if __name__ == "__main__":
    main()