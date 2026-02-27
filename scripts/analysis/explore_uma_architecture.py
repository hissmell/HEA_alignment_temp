#!/usr/bin/env python3
"""
UMA Architecture Analysis for Energy Head Identification

Analyzes UMA (Universal Materials Accelerator) models to identify:
1. Energy prediction head layers
2. Latent vector extraction points
3. Model architecture structure
4. Hook registration feasibility

UMA Models:
- uma-s-1p1: Small UMA model v1.1
- uma-m-1p1: Medium UMA model v1.1
"""

import sys
import os
sys.path.append('/DATA/user_scratch/pn50212/2024/12_AtomAttention')

import json
import torch
import warnings
warnings.filterwarnings('ignore')

def analyze_uma_model(model_path, model_name):
    """Analyze UMA model architecture to find energy head"""
    print(f"\n{'='*60}")
    print(f"ANALYZING {model_name.upper()}")
    print(f"{'='*60}")
    print(f"Model path: {model_path}")

    try:
        # Load model checkpoint
        print("\n1. Loading model checkpoint...")
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

        # Analyze checkpoint structure
        print(f"Checkpoint type: {type(checkpoint)}")
        print(f"Checkpoint class: {checkpoint.__class__}")

        # Handle MLIPInferenceCheckpoint
        if hasattr(checkpoint, '__dict__'):
            attrs = [attr for attr in dir(checkpoint) if not attr.startswith('_')]
            print(f"Available attributes: {attrs}")

        # Try to get the model/state_dict
        state_dict = None
        model_config = None

        # Handle MLIPInferenceCheckpoint
        if hasattr(checkpoint, 'model_state_dict'):
            state_dict = checkpoint.model_state_dict
            print(f"Using model_state_dict from MLIPInferenceCheckpoint")

        if hasattr(checkpoint, 'model_config'):
            model_config = checkpoint.model_config
            print(f"Model config: {model_config}")

        # Fallback methods
        if state_dict is None:
            if hasattr(checkpoint, 'model'):
                model = checkpoint.model
                print(f"Model type: {type(model)}")
                if hasattr(model, 'state_dict'):
                    state_dict = model.state_dict()
                elif hasattr(model, '__dict__'):
                    print(f"Model attributes: {[attr for attr in dir(model) if not attr.startswith('_')]}")
            elif hasattr(checkpoint, 'state_dict'):
                state_dict = checkpoint.state_dict()
            elif hasattr(checkpoint, '__getitem__'):  # dict-like
                try:
                    if 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    elif 'model' in checkpoint:
                        state_dict = checkpoint['model']
                except:
                    pass

        if state_dict is None:
            print("Could not extract state_dict from checkpoint")
            return None

        print(f"State dict type: {type(state_dict)}")
        print(f"Number of parameters: {len(state_dict)}")

        # Analyze parameter structure
        print("\n2. Analyzing model architecture...")

        # Group parameters by module
        modules = {}
        for name, param in state_dict.items():
            parts = name.split('.')
            if len(parts) > 1:
                module = parts[0]
                if module not in modules:
                    modules[module] = []
                modules[module].append(name)

        print(f"Top-level modules: {list(modules.keys())}")

        # Detailed analysis of each module
        analysis = {
            'model_name': model_name,
            'model_path': model_path,
            'total_parameters': len(state_dict),
            'model_config': model_config,
            'modules': {}
        }

        for module, params in modules.items():
            print(f"\n  Module: {module}")
            print(f"    Parameters: {len(params)}")

            # Look for energy-related parameters
            energy_params = [p for p in params if 'energy' in p.lower()]
            head_params = [p for p in params if 'head' in p.lower()]
            output_params = [p for p in params if 'output' in p.lower()]
            linear_params = [p for p in params if 'linear' in p.lower()]

            if energy_params:
                print(f"    Energy parameters: {energy_params}")
            if head_params:
                print(f"    Head parameters: {head_params}")
            if output_params:
                print(f"    Output parameters: {output_params}")
            if linear_params and len(linear_params) <= 10:  # Show if not too many
                print(f"    Linear parameters: {linear_params}")

            analysis['modules'][module] = {
                'total_params': len(params),
                'energy_params': energy_params,
                'head_params': head_params,
                'output_params': output_params,
                'linear_params': linear_params if len(linear_params) <= 20 else len(linear_params)
            }

            # Show sample parameters for structure understanding
            if len(params) <= 15:
                print(f"    All parameters:")
                for p in params:
                    shape = state_dict[p].shape if hasattr(state_dict[p], 'shape') else 'scalar'
                    print(f"      {p}: {shape}")
            else:
                print(f"    Sample parameters (first 10):")
                for p in params[:10]:
                    shape = state_dict[p].shape if hasattr(state_dict[p], 'shape') else 'scalar'
                    print(f"      {p}: {shape}")

        # Look for specific patterns that indicate energy head
        print(f"\n3. Searching for energy head patterns...")

        # Search for energy-related layers
        energy_layers = []
        for name, param in state_dict.items():
            if any(keyword in name.lower() for keyword in ['energy', 'head', 'output']):
                shape = param.shape if hasattr(param, 'shape') else 'scalar'
                energy_layers.append((name, shape))

        if energy_layers:
            print("  Potential energy head layers:")
            for name, shape in energy_layers:
                print(f"    {name}: {shape}")
        else:
            print("  No obvious energy head patterns found")

        # Search for final linear layers (common energy head pattern)
        final_layers = []
        for name, param in state_dict.items():
            if 'weight' in name and param.shape[-1] == 1:  # Output dim = 1 (energy)
                final_layers.append((name, param.shape))

        if final_layers:
            print("  Layers with output dimension 1 (potential energy outputs):")
            for name, shape in final_layers:
                print(f"    {name}: {shape}")

        analysis['energy_layers'] = energy_layers
        analysis['final_layers'] = final_layers

        # Try to load the actual model for runtime analysis
        print(f"\n4. Attempting to load actual model...")
        try:
            from fairchem.core.models import model_name_to_local_file
            from fairchem.core.common.relaxation.ase_utils import OCPCalculator

            # Try to create calculator (this loads the model)
            calc = OCPCalculator(
                model_name=model_name.replace('.pt', ''),
                local_cache='/tmp/fairchem_models',
                cpu=True
            )

            model = calc.trainer.model
            print(f"  Successfully loaded model: {type(model)}")

            # Analyze model structure
            print(f"  Model modules:")
            for name, module in model.named_modules():
                if len(name.split('.')) <= 2:  # Top-level modules only
                    print(f"    {name}: {type(module).__name__}")

            # Look for energy head in the loaded model
            energy_modules = []
            for name, module in model.named_modules():
                if any(keyword in name.lower() for keyword in ['energy', 'head', 'output']):
                    energy_modules.append((name, type(module).__name__))

            if energy_modules:
                print(f"  Energy-related modules:")
                for name, module_type in energy_modules:
                    print(f"    {name}: {module_type}")

            analysis['loaded_model'] = {
                'model_type': str(type(model)),
                'energy_modules': energy_modules
            }

        except Exception as e:
            print(f"  Failed to load model: {e}")
            analysis['loaded_model'] = None

        return analysis

    except Exception as e:
        print(f"Error analyzing {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Analyze both UMA models"""

    # UMA model paths
    models = {
        'uma-s-1p1': '/DATA/user_scratch/pn50212/2024/12_AtomAttention/MLPs/uma-s-1p1.pt',
        'uma-m-1p1': '/DATA/user_scratch/pn50212/2024/12_AtomAttention/MLPs/uma-m-1p1.pt'
    }

    all_analysis = {}

    for model_name, model_path in models.items():
        if os.path.exists(model_path):
            analysis = analyze_uma_model(model_path, model_name)
            if analysis:
                all_analysis[model_name] = analysis
        else:
            print(f"Model file not found: {model_path}")

    # Save analysis results
    output_file = '/DATA/user_scratch/pn50212/2024/12_AtomAttention/uma_architecture_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(all_analysis, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"Models analyzed: {list(all_analysis.keys())}")

if __name__ == "__main__":
    main()