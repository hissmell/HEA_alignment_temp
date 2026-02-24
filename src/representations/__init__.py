"""
Representation extraction modules for physics-inspired and MLIP embeddings.
"""

from .base import (
    RepresentationExtractor,
    PhysicsInspiredExtractor,
    MLIPEmbeddingExtractor,
    HybridRepresentation,
    ExtractionConfig,
    create_representation_extractor,
    load_representations
)

# Import physics-inspired extractors
from .physics_inspired import (
    SOAPExtractor,
    MultiRcutSOAPAnalyzer,
    SOAPConfig
)

# Import MLIP embedding extractors
from .mlip_embeddings import (
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

__all__ = [
    # Base classes
    'RepresentationExtractor',
    'PhysicsInspiredExtractor',
    'MLIPEmbeddingExtractor',
    'HybridRepresentation',
    'ExtractionConfig',
    'create_representation_extractor',
    'load_representations',
    # Physics-inspired
    'SOAPExtractor',
    'MultiRcutSOAPAnalyzer',
    'SOAPConfig',
    # MLIP embeddings
    'EquiformerExtractor',
    'EquiformerConfig',
    'create_equiformer_extractor',
    'MACEExtractor',
    'MACEConfig',
    'create_mace_extractor',
    'UMAExtractor',
    'UMAConfig',
    'create_uma_extractor',
]