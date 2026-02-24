"""
Core modules for alignment analysis and CKNNA computation.
"""

from .cknna import (
    CKNNA,
    CKNNAConfig,
    CKNNAAnalyzer,
    cknna_paper
)

from .alignment import (
    AlignmentAnalyzer,
    AlignmentConfig,
    SOAPAlignmentAnalyzer,
    compute_cknna_alignment,
    compute_dcor_alignment
)

# Will be added when created:
# from .metrics import compute_mae, compute_rmse, compute_correlation

__all__ = [
    # CKNNA module
    'CKNNA',
    'CKNNAConfig',
    'CKNNAAnalyzer',
    'cknna_paper',
    # Alignment module
    'AlignmentAnalyzer',
    'AlignmentConfig',
    'SOAPAlignmentAnalyzer',
    'compute_cknna_alignment',
    'compute_dcor_alignment'
]