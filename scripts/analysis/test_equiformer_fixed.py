#!/usr/bin/env python3
"""
Test the fixed Equiformer latent extractor
"""

import torch
import numpy as np
from pathlib import Path
from ase.io import read
from ase import Atoms
import sys

sys.path.append('/DATA/user_scratch/pn50212/2024/12_AtomAttention')


def test_equiformer_extraction():
    """Test the fixed Equiformer extraction"""

    print("=" * 70)
    print("Testing Fixed Equiformer Latent Extraction")
    print("=" * 70)

    # Import the fixed extractor
    from src.representations.mlip_embeddings.equiformer_latent import EquiformerLatentExtractor

    # Initialize extractor
    print("\nInitializing Equiformer extractor...")
    extractor = EquiformerLatentExtractor(
        model_name="eqV2_31M_omat",
        model_path="/DATA/user_scratch/pn50212/2024/12_AtomAttention/MLPs/eqV2_31M_omat.pt",
        device="cuda"
    )

    # Create test structures with different sizes
    test_cases = [
        ("Cu3", Atoms('Cu3', positions=[[0, 0, 0], [2, 0, 0], [1, 1, 0]],
                      cell=[10, 10, 10], pbc=True)),
        ("Cu5", Atoms('Cu5', positions=[[0, 0, 0], [2, 0, 0], [1, 1, 0],
                                        [0, 2, 0], [2, 2, 0]],
                      cell=[10, 10, 10], pbc=True)),
    ]

    print("\n" + "=" * 50)
    print("TESTING EXTRACTION:")
    print("=" * 50)

    for name, atoms in test_cases:
        print(f"\n--- Testing {name} (n_atoms={len(atoms)}) ---")

        try:
            # Extract latent vectors
            result = extractor.extract_single(atoms)

            if result is not None:
                print(f"✓ Extraction successful!")
                print(f"  Latent shape: {result['latent_vectors'].shape}")
                print(f"  Expected n_atoms: {result['n_atoms']}")
                print(f"  Latent dim: {result['latent_dim']}")
                print(f"  Extraction layer: {result['extraction_layer']}")

                # Verify shape
                if result['latent_vectors'].shape[0] == len(atoms):
                    print(f"  ✓ Shape matches n_atoms")
                else:
                    print(f"  ✗ Shape mismatch: got {result['latent_vectors'].shape[0]}, expected {len(atoms)}")

                # Check for reasonable values
                latent = result['latent_vectors']
                print(f"  Value range: [{latent.min():.4f}, {latent.max():.4f}]")
                print(f"  Mean: {latent.mean():.4f}, Std: {latent.std():.4f}")

            else:
                print("✗ Extraction failed: returned None")

        except Exception as e:
            print(f"✗ Extraction error: {e}")

    # Test with actual 25Cao structure if available
    print("\n" + "=" * 50)
    print("TESTING WITH 25CAO STRUCTURE:")
    print("=" * 50)

    try:
        # Try to load an actual structure from 25Cao
        contcar_path = "/DATA/user_scratch/pn50212/2024/12_AtomAttention/sourcedata/O/hcp/order11/CONTCAR1"
        if Path(contcar_path).exists():
            atoms_25cao = read(contcar_path)
            print(f"Loaded 25Cao structure: {len(atoms_25cao)} atoms")

            result = extractor.extract_single(atoms_25cao)

            if result is not None:
                print(f"✓ Extraction successful!")
                print(f"  Latent shape: {result['latent_vectors'].shape}")
                print(f"  Expected: ({len(atoms_25cao)}, {result['latent_dim']})")

                if result['latent_vectors'].shape[0] == len(atoms_25cao):
                    print(f"  ✓ Shape correctly matches n_atoms={len(atoms_25cao)}")
                else:
                    print(f"  ✗ Shape mismatch!")
            else:
                print("✗ Extraction failed for 25Cao structure")
        else:
            print("25Cao test structure not found, skipping...")

    except Exception as e:
        print(f"Error testing 25Cao structure: {e}")

    # Clean up
    extractor.cleanup()

    print("\n" + "=" * 70)
    print("Test complete!")
    print("=" * 70)


if __name__ == "__main__":
    test_equiformer_extraction()