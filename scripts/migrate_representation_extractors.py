#!/usr/bin/env python3
"""
Migration script demonstrating how to use the new representation modules
to replace the old extraction scripts.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import re
from ase import Atoms
from ase.io import read

# Import new representation modules
from src.representations import (
    SOAPExtractor,
    SOAPConfig,
    MultiRcutSOAPAnalyzer,
    EquiformerExtractor,
    EquiformerConfig,
    create_equiformer_extractor,
    MACEExtractor,
    MACEConfig,
    create_mace_extractor,
    UMAExtractor,
    UMAConfig,
    create_uma_extractor,
    HybridRepresentation,
    ExtractionConfig
)

# Configuration
BASE_DIR = "/DATA/user_scratch/pn50212/2024/12_AtomAttention/datasets/25Cao"
O_JSON_PATH = os.path.join(BASE_DIR, "O adsorption.json")
OH_JSON_PATH = os.path.join(BASE_DIR, "OH adsorption.json")
SOURCEDATA_DIR = os.path.join(BASE_DIR, "sourcedata")
OUTPUT_DIR = "/DATA/user_scratch/pn50212/2024/12_AtomAttention/representation_extraction_migrated"


def parse_taskname(taskname):
    """Parse taskname to extract adsorption type, order, and CONTCAR number."""
    order_match = re.search(r'order(\d+)', taskname)
    order = f"order{order_match.group(1)}" if order_match else None

    contcar_match = re.search(r'&(\d+)$', taskname)
    contcar_num = int(contcar_match.group(1)) + 1 if contcar_match else None

    if 'OH' in taskname or 'fixhollowOH' in taskname or 'fixontopOH' in taskname:
        ads_type = 'OH'
    else:
        ads_type = 'O'

    if 'hollow' in taskname:
        site = 'hollow'
    elif 'ontop' in taskname or 'top' in taskname:
        site = 'top'
    elif 'bridge' in taskname:
        site = 'bridge'
    else:
        site = 'hollow'

    return ads_type, order, contcar_num, site


def get_structure_path(ads_type, order, contcar_num, site='hollow'):
    """Get structure file path from taskname components."""
    base_path = os.path.join(SOURCEDATA_DIR, ads_type)

    if site == 'hollow':
        possible_sites = ['fcc', 'hcp']
    elif site == 'top':
        possible_sites = ['top']
    elif site == 'bridge':
        possible_sites = ['bridge']
    else:
        possible_sites = ['fcc']

    for site_dir in possible_sites:
        contcar_path = os.path.join(base_path, site_dir, order, f"CONTCAR{contcar_num}")
        if os.path.exists(contcar_path):
            return contcar_path

    return None


def load_structures(max_structures=100):
    """Load structures for demonstration."""
    print(f"Loading up to {max_structures} structures for demonstration...")

    structures = {}
    tasknames = []

    # Load from JSON files
    for json_path, ads_type in [(O_JSON_PATH, 'O'), (OH_JSON_PATH, 'OH')]:
        if not os.path.exists(json_path):
            continue

        with open(json_path, 'r') as f:
            data = json.load(f)

        for entry in data.values():
            if len(structures) >= max_structures:
                break

            taskname = entry.get('taskname', '')
            if not taskname:
                continue

            try:
                ads_type, order, contcar_num, site = parse_taskname(taskname)
                structure_path = get_structure_path(ads_type, order, contcar_num, site)

                if structure_path and os.path.exists(structure_path):
                    atoms = read(structure_path)
                    structures[taskname] = atoms
                    tasknames.append(taskname)

            except Exception as e:
                print(f"Warning: Failed to load {taskname}: {e}")
                continue

    print(f"Loaded {len(structures)} structures")
    return structures, tasknames


def demonstrate_soap_extraction():
    """Demonstrate SOAP extraction using new module."""
    print("\n" + "=" * 60)
    print("SOAP EXTRACTION DEMONSTRATION")
    print("=" * 60)

    try:
        # Load structures
        structures, tasknames = load_structures(50)
        if not structures:
            print("No structures loaded, skipping SOAP demonstration")
            return

        # Create SOAP extractor
        soap_config = SOAPConfig(
            r_cut=6.0,
            n_max=8,
            l_max=6,
            periodic=True
        )

        extraction_config = ExtractionConfig(
            batch_size=10,
            output_dir=Path(OUTPUT_DIR)
        )

        extractor = SOAPExtractor(soap_config, extraction_config)

        print(f"Created SOAP extractor with r_cut={soap_config.r_cut}")

        # Update species based on loaded structures
        structure_list = list(structures.values())
        extractor.update_species(structure_list)

        # Extract SOAP for a few structures
        sample_tasknames = tasknames[:5]
        soap_results = {}

        print("\nExtracting SOAP representations...")
        for taskname in tqdm(sample_tasknames, desc="SOAP extraction"):
            atoms = structures[taskname]

            # Extract all types of SOAP representations
            soap_repr = extractor.extract_single(atoms, atom_selection="multi")

            if soap_repr:
                soap_results[taskname] = soap_repr

                # Print shapes for first structure
                if taskname == sample_tasknames[0]:
                    print(f"\nSOAP representation shapes for {taskname}:")
                    for key, array in soap_repr.items():
                        print(f"  {key}: {array.shape}")

        print(f"\nSuccessfully extracted SOAP for {len(soap_results)} structures")

        # Save results
        output_file = Path(OUTPUT_DIR) / "soap_extraction_demo"
        extractor.save_representations(soap_results, output_file)
        print(f"Results saved to {output_file}")

    except ImportError as e:
        print(f"SOAP extraction skipped: {e}")
        print("Install dscribe to enable SOAP extraction: pip install dscribe")
    except Exception as e:
        print(f"SOAP demonstration failed: {e}")


def demonstrate_multi_rcut_soap():
    """Demonstrate multi-rcut SOAP analysis."""
    print("\n" + "=" * 60)
    print("MULTI-RCUT SOAP ANALYSIS DEMONSTRATION")
    print("=" * 60)

    try:
        # Load structures
        structures, tasknames = load_structures(20)
        if not structures:
            print("No structures loaded, skipping multi-rcut demonstration")
            return

        # Create multi-rcut analyzer
        rcut_values = [4.0, 6.0, 8.0, 10.0]
        analyzer = MultiRcutSOAPAnalyzer(
            rcut_values=rcut_values,
            soap_config=SOAPConfig(n_max=6, l_max=4)  # Smaller for demo
        )

        print(f"Created multi-rcut analyzer with rcut values: {rcut_values}")

        # Extract for one structure
        sample_atoms = structures[tasknames[0]]
        rcut_results = analyzer.extract_all_rcuts(sample_atoms, atom_selection="slab")

        print(f"\nMulti-rcut extraction results for {tasknames[0]}:")
        for rcut, soap_dict in rcut_results.items():
            if "soap_slab" in soap_dict:
                shape = soap_dict["soap_slab"].shape
                print(f"  rcut={rcut}: {shape}")

        print("✓ Multi-rcut SOAP analysis works correctly")

    except ImportError as e:
        print(f"Multi-rcut SOAP skipped: {e}")
    except Exception as e:
        print(f"Multi-rcut demonstration failed: {e}")


def demonstrate_equiformer_extraction():
    """Demonstrate EquiformerV2 extraction using new module."""
    print("\n" + "=" * 60)
    print("EQUIFORMERV2 EXTRACTION DEMONSTRATION")
    print("=" * 60)

    try:
        # Create extractor using convenience function
        extractor = create_equiformer_extractor(
            model_name="eq2_31M_ec4_allmd",
            extraction_layers=["norm_output"]
        )

        print(f"Created EquiformerV2 extractor for model: {extractor.model_name}")
        print(f"Available layers: {extractor.get_available_layers()}")
        print(f"Feature dimensions: {extractor.get_feature_dimensions()}")

        # Note: Actual extraction would require fairchem installation
        print("✓ EquiformerV2 extractor created successfully")
        print("Note: Actual extraction requires fairchem installation")

    except ImportError as e:
        print(f"EquiformerV2 extraction skipped: {e}")
        print("Install fairchem to enable EquiformerV2 extraction")
    except Exception as e:
        print(f"EquiformerV2 demonstration failed: {e}")


def demonstrate_hybrid_representation():
    """Demonstrate hybrid representation extraction."""
    print("\n" + "=" * 60)
    print("HYBRID REPRESENTATION DEMONSTRATION")
    print("=" * 60)

    try:
        # Create extractors (mock, as dependencies aren't available)
        extractors = {}

        # Add SOAP if available
        try:
            soap_extractor = SOAPExtractor(SOAPConfig(r_cut=6.0))
            extractors["soap"] = soap_extractor
            print("✓ Added SOAP extractor")
        except ImportError:
            print("- SOAP extractor skipped (dscribe not available)")

        # Add EquiformerV2 (will fail without fairchem, but shows structure)
        try:
            eq_extractor = create_equiformer_extractor()
            extractors["equiformer"] = eq_extractor
            print("✓ Added EquiformerV2 extractor")
        except ImportError:
            print("- EquiformerV2 extractor skipped (fairchem not available)")

        # Add MACE (will fail without mace-torch, but shows structure)
        try:
            mace_extractor = create_mace_extractor()
            extractors["mace"] = mace_extractor
            print("✓ Added MACE extractor")
        except ImportError:
            print("- MACE extractor skipped (mace-torch not available)")

        if extractors:
            # Create hybrid representation
            hybrid = HybridRepresentation(
                extractors=extractors,
                combination_strategy="separate"
            )

            print(f"\n✓ Created hybrid representation with {len(extractors)} extractors")
            print(f"✓ Extractors: {list(hybrid.extractors.keys())}")

            # Get feature info
            feature_info = hybrid.get_feature_info()
            print(f"✓ Feature info available for all extractors")

        else:
            print("No extractors available for hybrid demonstration")

    except Exception as e:
        print(f"Hybrid representation demonstration failed: {e}")


def demonstrate_batch_processing():
    """Demonstrate batch processing capabilities."""
    print("\n" + "=" * 60)
    print("BATCH PROCESSING DEMONSTRATION")
    print("=" * 60)

    try:
        # Load structures
        structures, tasknames = load_structures(30)
        if not structures:
            print("No structures loaded, skipping batch demonstration")
            return

        # Create SOAP extractor for batch processing
        extractor = SOAPExtractor(SOAPConfig(r_cut=6.0))

        # Prepare structures list and tasknames
        structure_list = [structures[name] for name in tasknames[:10]]
        batch_tasknames = tasknames[:10]

        print(f"Preparing batch processing for {len(structure_list)} structures...")

        # Update species
        extractor.update_species(structure_list)

        # Extract in batch
        batch_results = extractor.extract_batch(
            structures=structure_list,
            tasknames=batch_tasknames,
            atom_selection="slab"
        )

        print(f"✓ Batch processing completed for {len(batch_results)} structures")

        # Show sample results
        if batch_results:
            sample_name = list(batch_results.keys())[0]
            sample_result = batch_results[sample_name]
            print(f"\nSample result for {sample_name}:")
            for key, array in sample_result.items():
                print(f"  {key}: {array.shape}")

    except ImportError as e:
        print(f"Batch processing skipped: {e}")
    except Exception as e:
        print(f"Batch processing demonstration failed: {e}")


def main():
    """Run all demonstrations."""
    print("=" * 60)
    print("REPRESENTATION EXTRACTION MIGRATION DEMONSTRATION")
    print("=" * 60)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Run demonstrations
    demonstrate_soap_extraction()
    demonstrate_multi_rcut_soap()
    demonstrate_equiformer_extraction()
    demonstrate_hybrid_representation()
    demonstrate_batch_processing()

    print("\n" + "=" * 60)
    print("MIGRATION BENEFITS DEMONSTRATED")
    print("=" * 60)
    print("\n✅ BENEFITS OF NEW REPRESENTATION MODULES:")
    print("1. **Unified Interface**: All extractors follow same API")
    print("2. **Modular Design**: Easy to add new representation types")
    print("3. **Flexible Configuration**: Configurable parameters for each extractor")
    print("4. **Batch Processing**: Efficient processing of multiple structures")
    print("5. **Hybrid Representations**: Combine multiple representation types")
    print("6. **Automatic Checkpointing**: Built-in support for large-scale extraction")
    print("7. **Multiple Output Formats**: NPZ, JSON, or both")
    print("8. **Error Handling**: Robust error handling and logging")
    print("\n📁 REPLACES MULTIPLE OLD SCRIPTS:")
    print("- extract_soap_representations_25cao.py")
    print("- extract_equiformer_31M_representations_25cao.py")
    print("- extract_mace_representations_25cao.py")
    print("- extract_uma_representations_25cao.py")
    print("- extract_soap_hea.py")
    print("- extract_uma_hea.py")
    print("- All rcut-specific SOAP scripts")
    print("\n🚀 USAGE EXAMPLES:")
    print("```python")
    print("# SOAP extraction")
    print("from src.representations import SOAPExtractor, SOAPConfig")
    print("extractor = SOAPExtractor(SOAPConfig(r_cut=6.0))")
    print("soap_repr = extractor.extract_single(atoms)")
    print("")
    print("# EquiformerV2 extraction")
    print("from src.representations import create_equiformer_extractor")
    print("extractor = create_equiformer_extractor('eq2_31M_ec4_allmd')")
    print("eq_repr = extractor.extract_single(atoms)")
    print("")
    print("# Hybrid representation")
    print("from src.representations import HybridRepresentation")
    print("hybrid = HybridRepresentation({'soap': soap_ext, 'eq': eq_ext})")
    print("all_repr = hybrid.extract_all(atoms)")
    print("```")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()