"""
Extract Coulomb Matrix representations for 25Cao dataset
Saves in chunked JSON format (1000 structures per file)
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
from src.representations.physics_inspired.coulomb_matrix import CoulombMatrixExtractor

def parse_taskname_to_path(taskname, adsorbate="O"):
    """
    Parse taskname to CONTCAR file path

    Special rules:
    - For top site: taskname contains 'fixtopO' (for O) or 'fixtopOH' (for OH)
    - For other sites: normal naming like 'fixhollow', 'fixfcchollow', 'fixbridge'

    Examples:
        O adsorbate:
            87777order11&fixhollow&0 -> sourcedata/O/hcp/order11/CONTCAR1
            87777order11&fixfcchollow&0 -> sourcedata/O/fcc/order11/CONTCAR1
            87777order11&fixbridge&0 -> sourcedata/O/bridge/order11/CONTCAR1
            87777order11&fixtopO&0 -> sourcedata/O/top/order11/CONTCAR1

        OH adsorbate:
            87777order11&fixhollow&0 -> sourcedata/OH/hcp/order11/CONTCAR1
            87777order11&fixtopOH&0 -> sourcedata/OH/top/order11/CONTCAR1
    """
    parts = taskname.split('&')

    # Extract order number
    order_str = parts[0]
    order_num = order_str.replace('87777order', '')

    # Extract site type
    site_str = parts[1].replace('fix', '')

    # Handle different suffix conventions:
    # - O adsorbate: only 'top' site has 'O' suffix (fixtopO)
    # - OH adsorbate: ALL sites have 'OH' suffix (fixhollowOH, fixbridgeOH, fixfcchollowOH, fixtopOH)
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

    # Extract CONTCAR number (increment by 1 since it starts from 1)
    contcar_num = int(parts[2]) + 1

    # Construct path
    base_path = Path(f"/DATA/user_scratch/pn50212/2024/12_AtomAttention/datasets/25Cao/sourcedata")
    file_path = base_path / adsorbate / site_dir / f"order{order_num}" / f"CONTCAR{contcar_num}"

    return file_path


def load_tasknames(json_path):
    """Load all tasknames from the JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    tasknames = []
    for task_key in sorted(data.keys(), key=lambda x: int(x.replace('task', ''))):
        tasknames.append(data[task_key]['taskname'])

    return tasknames


def extract_coulomb_matrices(tasknames, adsorbate="O", chunk_size=1000):
    """Extract Coulomb Matrix representations in chunks

    Args:
        tasknames: List of tasknames to process
        adsorbate: "O" or "OH"
        chunk_size: Number of structures per chunk file
    """

    # Setup extractor
    extractor = CoulombMatrixExtractor(
        n_atoms_max=50,  # Adjust based on max atoms in dataset
        permutation='sorted_l2',
        flatten=False  # Keep as 2D matrix
    )

    # Output directory - unified for all adsorbates
    output_dir = Path("/DATA/user_scratch/pn50212/2024/12_AtomAttention/datasets/25Cao/representations/coulomb_matrix")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process in chunks
    total_chunks = (len(tasknames) + chunk_size - 1) // chunk_size

    # Track progress - separate checkpoint for each adsorbate
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

    # Extract metadata from first structure
    if start_chunk == 0:
        first_path = parse_taskname_to_path(tasknames[0], adsorbate)
        first_atoms = read(str(first_path))
        first_result = extractor.extract_single(first_atoms)
        # Extract the actual representation array from the result dict - looking for cm_all key
        first_repr = first_result.get('cm_all', None)

        if first_repr is None:
            print("Warning: Could not extract first representation for metadata")
            print(f"Available keys in result: {first_result.keys()}")
            feature_shape = "unknown"
        else:
            feature_shape = first_repr.shape

        metadata = {
            "descriptor": "coulomb_matrix",
            "adsorbate": adsorbate,
            "parameters": {
                "n_atoms_max": extractor.n_atoms_max,
                "permutation": extractor.permutation,
                "flatten": extractor.flatten
            },
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
                # Get the actual representation array from the result dict - looking for cm_all key
                representation = result.get('cm_all', None)

                if representation is not None:
                    # Store in chunk data with adsorbate prefix in structure_id
                    structure_id = f"{adsorbate}_{taskname}"
                    chunk_data[structure_id] = representation.tolist()

                    # Track metadata for Excel
                    structure_info = {
                        'structure_id': structure_id,
                        'taskname': taskname,
                        'adsorbate': adsorbate,
                        'file_path': str(file_path),
                        'chunk_file': f"{adsorbate}_chunk_{chunk_idx:04d}.json",
                        'n_atoms': len(atoms),
                        'representation_shape': str(representation.shape),  # Record actual shape as string
                        'matrix_dim': representation.shape[0],  # First dimension of matrix
                        'permutation': extractor.permutation,
                        'n_atoms_max': extractor.n_atoms_max,
                        'flatten': extractor.flatten,
                        'extraction_success': True
                    }
                    all_structure_info.append(structure_info)
                else:
                    print(f"Warning: No representation extracted for {taskname}")
                    # Track failed extraction
                    structure_info = {
                        'structure_id': f"{adsorbate}_{taskname}",
                        'taskname': taskname,
                        'adsorbate': adsorbate,
                        'file_path': str(file_path),
                        'chunk_file': None,
                        'n_atoms': len(atoms),
                        'representation_shape': None,
                        'matrix_dim': None,
                        'permutation': extractor.permutation,
                        'n_atoms_max': extractor.n_atoms_max,
                        'flatten': extractor.flatten,
                        'extraction_success': False
                    }
                    all_structure_info.append(structure_info)

            except Exception as e:
                print(f"Failed to process {taskname}: {e}")
                failed_structures.append(taskname)
                # Track failed processing
                structure_info = {
                    'structure_id': f"{adsorbate}_{taskname}",
                    'taskname': taskname,
                    'adsorbate': adsorbate,
                    'file_path': str(parse_taskname_to_path(taskname, adsorbate)),
                    'chunk_file': None,
                    'n_atoms': None,
                    'representation_shape': None,
                    'matrix_dim': None,
                    'permutation': extractor.permutation,
                    'n_atoms_max': extractor.n_atoms_max,
                    'flatten': extractor.flatten,
                    'extraction_success': False,
                    'error': str(e)
                }
                all_structure_info.append(structure_info)

        # Save chunk with adsorbate prefix
        chunk_file = output_dir / f"{adsorbate}_chunk_{chunk_idx:04d}.json"
        with open(chunk_file, 'w') as f:
            json.dump(chunk_data, f)

        print(f"Saved {len(chunk_data)} structures to {chunk_file}")

        if failed_structures:
            print(f"Failed structures in chunk {chunk_idx}: {failed_structures}")
            # Save failed list with adsorbate prefix
            with open(output_dir / f"{adsorbate}_failed_chunk_{chunk_idx:04d}.json", 'w') as f:
                json.dump(failed_structures, f)

        # Update checkpoint
        with open(checkpoint_file, 'w') as f:
            json.dump({"last_completed_chunk": chunk_idx}, f)

    # Save all structure info to Excel
    if all_structure_info:
        excel_file = output_dir / f"extraction_metadata_{adsorbate}.xlsx"
        df = pd.DataFrame(all_structure_info)

        # Reorder columns for better readability
        column_order = ['structure_id', 'taskname', 'adsorbate', 'file_path', 'chunk_file',
                       'n_atoms', 'representation_shape', 'matrix_dim', 'extraction_success',
                       'permutation', 'n_atoms_max', 'flatten']
        if 'error' in df.columns:
            column_order.append('error')

        df = df[column_order]

        # Save to Excel with formatting
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Extraction_Info', index=False)

            # Get the worksheet
            worksheet = writer.sheets['Extraction_Info']

            # Adjust column widths
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)  # Cap at 50
                worksheet.column_dimensions[column_letter].width = adjusted_width

        print(f"\nSaved extraction metadata to: {excel_file}")

        # Print summary statistics
        success_count = df['extraction_success'].sum()
        total_count = len(df)
        print(f"Successfully extracted: {success_count}/{total_count} structures")

    print("\nExtraction completed!")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract Coulomb Matrix for 25Cao dataset")
    parser.add_argument("--adsorbate", type=str, default="O", choices=["O", "OH"],
                        help="Adsorbate type: O or OH")
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

    # Test parsing with first few
    print("\nTesting path parsing:")
    for i in range(min(3, len(tasknames))):
        taskname = tasknames[i]
        path = parse_taskname_to_path(taskname, args.adsorbate)
        print(f"{taskname} -> {path}")
        print(f"  Exists: {path.exists()}")

    if args.test:
        print("\n*** TEST MODE: Processing only first 10 structures ***")
        tasknames = tasknames[:10]

    # Start extraction
    print("\n" + "="*50)
    print(f"Starting Coulomb Matrix extraction for {args.adsorbate}")
    print("="*50)

    extract_coulomb_matrices(tasknames, adsorbate=args.adsorbate, chunk_size=args.chunk_size)