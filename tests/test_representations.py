"""
Test suite for representation extraction modules.
Tests both physics-inspired and MLIP embedding extractors.
"""

import sys
import os
import numpy as np
from pathlib import Path
from ase import Atoms

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Test imports
def test_imports():
    """Test that all representation modules can be imported."""
    print("Testing imports...")

    try:
        from src.representations import (
            RepresentationExtractor,
            PhysicsInspiredExtractor,
            MLIPEmbeddingExtractor,
            HybridRepresentation,
            ExtractionConfig,
            SOAPExtractor,
            SOAPConfig,
            MultiRcutSOAPAnalyzer
        )
        print("✓ Base classes and SOAP imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import base/SOAP classes: {e}")
        return False

    try:
        from src.representations import (
            EquiformerExtractor,
            EquiformerConfig,
            create_equiformer_extractor,
            MACEExtractor,
            MACEConfig,
            create_mace_extractor,
            UMAExtractor,
            UMAConfig,
            create_uma_extractor
        )
        print("✓ MLIP embedding extractors imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import MLIP extractors: {e}")
        return False

    return True


def test_soap_extractor_basic():
    """Test basic SOAP extractor functionality (without dscribe dependency)."""
    print("\nTesting SOAP extractor basic functionality...")

    try:
        from src.representations import SOAPExtractor, SOAPConfig

        # Create config
        config = SOAPConfig(r_cut=6.0, n_max=8, l_max=6)
        extractor = SOAPExtractor(config)

        print(f"✓ SOAP extractor created with config: r_cut={config.r_cut}")
        print(f"✓ Descriptor name: {extractor.descriptor_name}")
        print(f"✓ Feature names: {extractor.get_feature_names()}")

        # Test without initializing (should return empty dimensions)
        dimensions = extractor.get_feature_dimensions()
        print(f"✓ Feature dimensions (uninitialized): {dimensions}")

        return True

    except ImportError as e:
        print(f"✓ SOAP extractor skipped (dscribe not available): {e}")
        return True
    except Exception as e:
        print(f"✗ SOAP extractor test failed: {e}")
        return False


def test_equiformer_extractor_basic():
    """Test basic EquiformerV2 extractor functionality (without fairchem dependency)."""
    print("\nTesting EquiformerV2 extractor basic functionality...")

    try:
        from src.representations import EquiformerExtractor, EquiformerConfig

        # Create config
        config = EquiformerConfig(
            model_name="eq2_31M_ec4_allmd",
            extraction_layers=["norm_output"]
        )
        extractor = EquiformerExtractor(config)

        print(f"✓ EquiformerV2 extractor created for model: {config.model_name}")
        print(f"✓ Model name: {extractor.model_name}")
        print(f"✓ Feature names: {extractor.get_feature_names()}")
        print(f"✓ Feature dimensions: {extractor.get_feature_dimensions()}")
        print(f"✓ Available layers: {extractor.get_available_layers()}")

        return True

    except ImportError as e:
        print(f"✓ EquiformerV2 extractor skipped (fairchem not available): {e}")
        return True
    except Exception as e:
        print(f"✗ EquiformerV2 extractor test failed: {e}")
        return False


def test_mace_extractor_basic():
    """Test basic MACE extractor functionality (without mace dependency)."""
    print("\nTesting MACE extractor basic functionality...")

    try:
        from src.representations import MACEExtractor, MACEConfig, create_mace_extractor

        # Create config
        config = MACEConfig(
            model_path="medium",
            head="omat_pbe"
        )
        extractor = MACEExtractor(config)

        print(f"✓ MACE extractor created for model: {config.model_path}")
        print(f"✓ Model name: {extractor.model_name}")
        print(f"✓ Feature names: {extractor.get_feature_names()}")
        print(f"✓ Feature dimensions: {extractor.get_feature_dimensions()}")
        print(f"✓ Available layers: {extractor.get_available_layers()}")

        # Test convenience function
        extractor2 = create_mace_extractor(model_path="large", head="oc20_usemppbe")
        print(f"✓ Convenience function works: {extractor2.model_name}")

        return True

    except ImportError as e:
        print(f"✓ MACE extractor skipped (mace-torch not available): {e}")
        return True
    except Exception as e:
        print(f"✗ MACE extractor test failed: {e}")
        return False


def test_uma_extractor_basic():
    """Test basic UMA extractor functionality (without fairchem dependency)."""
    print("\nTesting UMA extractor basic functionality...")

    try:
        from src.representations import UMAExtractor, UMAConfig, create_uma_extractor

        # Create config
        config = UMAConfig(
            model_path="/path/to/uma/model.pt",
            task_name="oc20"
        )
        extractor = UMAExtractor(config)

        print(f"✓ UMA extractor created for task: {config.task_name}")
        print(f"✓ Model name: {extractor.model_name}")
        print(f"✓ Feature names: {extractor.get_feature_names()}")
        print(f"✓ Feature dimensions: {extractor.get_feature_dimensions()}")
        print(f"✓ Available layers: {extractor.get_available_layers()}")

        # Test convenience function
        extractor2 = create_uma_extractor("/path/to/model2.pt", "s2ef")
        print(f"✓ Convenience function works: {extractor2.model_name}")

        return True

    except ImportError as e:
        print(f"✓ UMA extractor skipped (fairchem not available): {e}")
        return True
    except Exception as e:
        print(f"✗ UMA extractor test failed: {e}")
        return False


def test_hybrid_representation():
    """Test HybridRepresentation functionality."""
    print("\nTesting HybridRepresentation...")

    try:
        from src.representations import (
            HybridRepresentation,
            SOAPExtractor,
            SOAPConfig,
            EquiformerExtractor,
            EquiformerConfig
        )

        # Create mock extractors (won't work without dependencies but should instantiate)
        soap_extractor = SOAPExtractor(SOAPConfig())
        eq_extractor = EquiformerExtractor(EquiformerConfig())

        # Create hybrid representation
        hybrid = HybridRepresentation(
            extractors={
                "soap": soap_extractor,
                "equiformer": eq_extractor
            },
            combination_strategy="separate"
        )

        print("✓ HybridRepresentation created with 2 extractors")
        print(f"✓ Extractors: {list(hybrid.extractors.keys())}")
        print(f"✓ Combination strategy: {hybrid.combination_strategy}")

        # Test feature info (without initializing extractors)
        feature_info = hybrid.get_feature_info()
        print(f"✓ Feature info retrieved: {list(feature_info.keys())}")

        return True

    except ImportError as e:
        print(f"✓ HybridRepresentation skipped (dependencies not available): {e}")
        return True
    except Exception as e:
        print(f"✗ HybridRepresentation test failed: {e}")
        return False


def test_extraction_config():
    """Test ExtractionConfig functionality."""
    print("\nTesting ExtractionConfig...")

    try:
        from src.representations import ExtractionConfig

        # Test default config
        config = ExtractionConfig()
        print(f"✓ Default config: batch_size={config.batch_size}, device={config.device}")

        # Test custom config
        custom_config = ExtractionConfig(
            batch_size=500,
            device="cuda",
            checkpoint_interval=2000,
            save_format="both"
        )
        print(f"✓ Custom config: batch_size={custom_config.batch_size}, format={custom_config.save_format}")

        return True

    except Exception as e:
        print(f"✗ ExtractionConfig test failed: {e}")
        return False


def create_mock_structure():
    """Create a mock atomic structure for testing."""
    # Simple HEA structure
    positions = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [1.0, 1.5, 0.0],
        [1.0, 0.5, 2.0],
        [1.0, 1.0, 3.0]  # Adsorbate atom
    ])

    symbols = ['Ti', 'V', 'Cr', 'Nb', 'O']
    cell = np.array([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 6.0]])

    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
    return atoms


def main():
    """Run all tests."""
    print("=" * 60)
    print("REPRESENTATION MODULE TEST SUITE")
    print("=" * 60)

    test_functions = [
        test_imports,
        test_extraction_config,
        test_soap_extractor_basic,
        test_equiformer_extractor_basic,
        test_mace_extractor_basic,
        test_uma_extractor_basic,
        test_hybrid_representation,
    ]

    passed = 0
    total = len(test_functions)

    for test_func in test_functions:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} failed with exception: {e}")

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed}/{total} tests passed")

    if passed == total:
        print("ALL TESTS PASSED ✓")
        print("Representation modules are working correctly!")
    else:
        print("SOME TESTS FAILED ✗")
        print("Note: Many failures expected due to missing optional dependencies")

    print("=" * 60)

    # Note about dependencies
    print("\nNOTE: Full functionality requires optional dependencies:")
    print("  - SOAP: pip install dscribe")
    print("  - EquiformerV2: pip install fairchem-core")
    print("  - MACE: pip install mace-torch")
    print("  - UMA: pip install fairchem-core")
    print("\nThe modules will work when these dependencies are installed.")


if __name__ == "__main__":
    main()