"""
Extract MBTR representations for 25Cao dataset with different geometry functions
Saves in chunked JSON format (1000 structures per file)
Parameters are saved in separate directories based on configuration
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from ase.io import read
from tqdm import tqdm
import sys
sys.path.append('/DATA/user_scratch/pn50212/2024/12_AtomAttention')
from src.representations.physics_inspired.mbtr import MBTRExtractor

def parse_taskname_to_path(taskname, adsorbate="O"):
    """
    Parse taskname to CONTCAR file path

    Special rules:
    - For top site: taskname contains 'fixtopO' (for O) or 'fixtopOH' (for OH)
    - For other sites: normal naming like 'fixhollow', 'fixfcchollow', 'fixbridge'
    """
    parts = taskname.split('&')

    # Extract order number
    order_str = parts[0]
    order_num = order_str.replace('87777order', '')

    # Extract site type
    site_str = parts[1].replace('fix', '')

    # Handle different suffix conventions
    if adsorbate == 'O':
        # Only top site has O suffix
        if site_str == 'topO':
            site_str = 'top'
    elif adsorbate == 'OH':
        # All sites have OH suffix - remove it
        if site_str.endswith('OH'):
            site_str = site_str[:-2]

    # Map site names to directories
    site_mapping = {
        'hollow': 'hcp',
        'fcchollow': 'fcc',
        'bridge': 'bridge',
        'top': 'top'
    }

    site_dir = site_mapping.get(site_str, site_str)

    # Extract CONTCAR number
    contcar_num = int(parts[2]) + 1

    # Construct path
    base_path = Path("/DATA/user_scratch/pn50212/2024/12_AtomAttention/datasets/25Cao/sourcedata")
    file_path = base_path / adsorbate / site_dir / f"order{order_num}" / f"CONTCAR{contcar_num}"

    return file_path


def load_tasknames(json_path):
    """Load all tasknames from the JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    tasknames = []
    for task_key in sorted(data.keys(), key=lambda x: int(x.replace('task', '')))[:]:
        tasknames.append(data[task_key]['taskname'])

    return tasknames


def get_mbtr_configurations():
    """
    Define MBTR configurations to extract.
    Focus on k2 and k3 with different geometry functions.
    """
    configs = {
        # k2 configurations with different geometry functions
        "k2_distance": {
            "k_term": "k2",
            "geometry": {"function": "distance"},
            "grid": {"min": 0, "max": 20, "n": 200, "sigma": 0.1},
            "weighting": {"function": "exp", "scale": 0.5, "threshold": 1e-3},  # Default weight
            "description": "2-body distances"
        },
        "k2_inverse_distance": {
            "k_term": "k2",
            "geometry": {"function": "inverse_distance"},
            "grid": {"min": 0, "max": 1, "n": 200, "sigma": 0.02},
            "weighting": {"function": "exp", "scale": 0.5, "threshold": 1e-3},  # Default weight
            "description": "2-body inverse distances"
        },

        # k3 configurations with different geometry functions
        "k3_angle": {
            "k_term": "k3",
            "geometry": {"function": "angle"},
            "grid": {"min": 0, "max": 180, "n": 180, "sigma": 2},
            "weighting": {"function": "exp", "scale": 0.5, "threshold": 1e-3},  # Default weight
            "description": "3-body angles in degrees"
        },
        "k3_cosine": {
            "k_term": "k3",
            "geometry": {"function": "cosine"},
            "grid": {"min": -1, "max": 1, "n": 200, "sigma": 0.02},
            "weighting": {"function": "exp", "scale": 0.5, "threshold": 1e-3},  # Default weight
            "description": "3-body cosine of angles"
        }
    }

    return configs


def extract_mbtr_for_config(tasknames, adsorbate, config_name, config_params, chunk_size=1000):
    """Extract MBTR for a specific configuration"""

    print(f"\n{'='*60}")
    print(f"Extracting {config_name}: {config_params['description']}")
    print(f"K-term: {config_params['k_term']}")
    print(f"Geometry: {config_params['geometry']['function']}")
    print(f"Grid: min={config_params['grid']['min']}, max={config_params['grid']['max']}, n={config_params['grid']['n']}")
    print(f"{'='*60}")

    # Define species for 25Cao dataset (actual HEA + adsorbates)
    # Based on actual dataset composition: Ag, Ir, Pd, Pt, Ru + O, H (adsorbates)
    species = ['Ag', 'Ir', 'Pd', 'Pt', 'Ru', 'O', 'H']

    # Setup extractor
    extractor = MBTRExtractor(
        species=species,
        geometry=config_params['geometry'],
        grid=config_params['grid'],
        weighting=config_params['weighting'],
        periodic=True,  # 25Cao structures are periodic
        normalization='l2',  # L2 normalization for consistency
        normalize_gaussians=True,
        sparse=False,
        dtype='float64'
    )

    # Output directory - parameter-specific
    output_dir = Path(f"/DATA/user_scratch/pn50212/2024/12_AtomAttention/datasets/25Cao/representations/mbtr/{config_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process in chunks
    total_chunks = (len(tasknames) + chunk_size - 1) // chunk_size

    # Track progress - separate checkpoint for each config and adsorbate
    checkpoint_file = output_dir / f"extraction_checkpoint_{adsorbate}.json"
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
            start_chunk = checkpoint.get('last_completed_chunk', -1) + 1
    else:
        start_chunk = 0

    print(f"Total structures: {len(tasknames)}")
    print(f"Total chunks: {total_chunks}")
    print(f"Starting from chunk: {start_chunk}")

    # Extract metadata from first structure if starting fresh
    if start_chunk == 0:
        first_path = parse_taskname_to_path(tasknames[0], adsorbate)
        first_atoms = read(str(first_path))

        try:
            first_result = extractor.extract_single(first_atoms)
            first_repr = first_result.get('mbtr_all', None)

            if first_repr is not None:
                feature_shape = first_repr.shape
                print(f"Feature shape: {feature_shape}")
            else:
                feature_shape = "unknown"
                print("Warning: Could not extract first representation")
        except Exception as e:
            print(f"Warning: Failed to extract first structure: {e}")
            feature_shape = "unknown"

        # Save metadata
        metadata = {
            "descriptor": "mbtr",
            "config_name": config_name,
            "adsorbate": adsorbate,
            "parameters": {
                "k_term": config_params['k_term'],
                "geometry": config_params['geometry'],
                "grid": config_params['grid'],
                "weighting": config_params['weighting'],
                "species": species,
                "periodic": True,
                "normalization": "l2",
                "normalize_gaussians": True,
                "dtype": "float64"
            },
            "description": config_params['description'],
            "total_structures": len(tasknames),
            "chunk_size": chunk_size,
            "feature_shape": str(feature_shape),
            "extraction_date": str(np.datetime64('today'))
        }

        # Update or create metadata file
        metadata_file = output_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                all_metadata = json.load(f)
        else:
            all_metadata = {}

        all_metadata[adsorbate] = metadata

        with open(metadata_file, 'w') as f:
            json.dump(all_metadata, f, indent=2)

    # Initialize tracking for Excel metadata
    all_structure_info = []

    # Process chunks
    for chunk_idx in range(start_chunk, total_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, len(tasknames))
        chunk_tasknames = tasknames[start_idx:end_idx]

        print(f"\nProcessing chunk {chunk_idx}/{total_chunks-1} (structures {start_idx}-{end_idx-1})")

        chunk_data = {}
        failed_structures = []

        for taskname in tqdm(chunk_tasknames, desc=f"Chunk {chunk_idx}"):
            try:
                # Get file path
                file_path = parse_taskname_to_path(taskname, adsorbate)

                # Read structure
                atoms = read(str(file_path))

                # Extract representation
                result = extractor.extract_single(atoms)
                representation = result.get('mbtr_all', None)

                if representation is not None:
                    # Store in chunk data
                    structure_id = f"{adsorbate}_{taskname}"
                    chunk_data[structure_id] = representation.tolist()

                    # Track metadata
                    structure_info = {
                        'structure_id': structure_id,
                        'taskname': taskname,
                        'adsorbate': adsorbate,
                        'file_path': str(file_path),
                        'chunk_file': f"{adsorbate}_chunk_{chunk_idx:04d}.json",
                        'n_atoms': len(atoms),
                        'representation_shape': str(representation.shape),
                        'config_name': config_name,
                        'k_term': config_params['k_term'],
                        'geometry_function': config_params['geometry']['function'],
                        'grid_n': config_params['grid']['n'],
                        'extraction_success': True
                    }
                    all_structure_info.append(structure_info)
                else:
                    print(f"Warning: No representation extracted for {taskname}")
                    failed_structures.append(taskname)

            except Exception as e:
                print(f"Failed to process {taskname}: {e}")
                failed_structures.append(taskname)
                # Track failed extraction
                structure_info = {
                    'structure_id': f"{adsorbate}_{taskname}",
                    'taskname': taskname,
                    'adsorbate': adsorbate,
                    'file_path': str(parse_taskname_to_path(taskname, adsorbate)),
                    'chunk_file': None,
                    'n_atoms': None,
                    'representation_shape': None,
                    'config_name': config_name,
                    'k_term': config_params['k_term'],
                    'geometry_function': config_params['geometry']['function'],
                    'grid_n': config_params['grid']['n'],
                    'extraction_success': False,
                    'error': str(e)
                }
                all_structure_info.append(structure_info)

        # Save chunk
        if chunk_data:
            chunk_file = output_dir / f"{adsorbate}_chunk_{chunk_idx:04d}.json"
            with open(chunk_file, 'w') as f:
                json.dump(chunk_data, f)

            print(f"Saved {len(chunk_data)} structures to {chunk_file}")

        if failed_structures:
            print(f"Failed structures in chunk {chunk_idx}: {len(failed_structures)} structures")
            with open(output_dir / f"{adsorbate}_failed_chunk_{chunk_idx:04d}.json", 'w') as f:
                json.dump(failed_structures, f)

        # Update checkpoint
        with open(checkpoint_file, 'w') as f:
            json.dump({"last_completed_chunk": chunk_idx}, f)

    # Save all structure info to Excel
    if all_structure_info:
        excel_file = output_dir / f"extraction_metadata_{adsorbate}.xlsx"
        df = pd.DataFrame(all_structure_info)

        # Reorder columns
        column_order = ['structure_id', 'taskname', 'adsorbate', 'file_path', 'chunk_file',
                       'n_atoms', 'representation_shape', 'config_name', 'k_term',
                       'geometry_function', 'grid_n', 'extraction_success']
        if 'error' in df.columns:
            column_order.append('error')

        df = df[column_order]

        # Save to Excel
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Extraction_Info', index=False)

            # Adjust column widths
            worksheet = writer.sheets['Extraction_Info']
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width

        print(f"\nSaved extraction metadata to: {excel_file}")

        # Print summary
        success_count = df['extraction_success'].sum()
        total_count = len(df)
        print(f"Successfully extracted: {success_count}/{total_count} structures")

    print(f"\nExtraction completed for {config_name}!")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract MBTR for 25Cao dataset")
    parser.add_argument("--adsorbate", type=str, default="O", choices=["O", "OH"],
                        help="Adsorbate type: O or OH")
    parser.add_argument("--config", type=str, default=None,
                        help="Specific configuration to extract (e.g., k2_distance, k2_inverse_distance, k3_angle, k3_cosine)")
    parser.add_argument("--chunk-size", type=int, default=1000,
                        help="Number of structures per chunk file")
    parser.add_argument("--test", action="store_true",
                        help="Test mode: only process first 10 structures")
    args = parser.parse_args()

    # Load tasknames
    json_path = f"/DATA/user_scratch/pn50212/2024/12_AtomAttention/datasets/25Cao/{args.adsorbate} adsorption.json"
    tasknames = load_tasknames(json_path)

    print(f"Adsorbate: {args.adsorbate}")
    print(f"Loaded {len(tasknames)} tasknames")

    if args.test:
        print("\n*** TEST MODE: Processing only first 10 structures ***")
        tasknames = tasknames[:10]

    # Get configurations
    configs = get_mbtr_configurations()

    # Extract specific config or all
    if args.config:
        if args.config in configs:
            extract_mbtr_for_config(
                tasknames,
                args.adsorbate,
                args.config,
                configs[args.config],
                chunk_size=args.chunk_size
            )
        else:
            print(f"Error: Unknown configuration '{args.config}'")
            print(f"Available configurations: {list(configs.keys())}")
    else:
        # Extract all configurations
        print(f"\n{'='*60}")
        print("Extracting ALL MBTR configurations")
        print(f"Configurations: {list(configs.keys())}")
        print(f"{'='*60}")

        for config_name, config_params in configs.items():
            extract_mbtr_for_config(
                tasknames,
                args.adsorbate,
                config_name,
                config_params,
                chunk_size=args.chunk_size
            )
            print(f"\n{'='*60}")
            print(f"Completed: {config_name}")
            print(f"{'='*60}\n")