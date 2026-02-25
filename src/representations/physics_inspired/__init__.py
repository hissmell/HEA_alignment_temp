"""
Physics-inspired representation extractors.
Includes descriptors like SOAP, ACSF, MBTR, Coulomb Matrix, Sine Matrix, etc.
"""

from .soap import SOAPExtractor, MultiRcutSOAPAnalyzer, SOAPConfig
from .coulomb_matrix import CoulombMatrixExtractor
from .sine_matrix import SineMatrixExtractor

# Will be added as modules are migrated:
# from .acsf import ACSFExtractor
# from .mbtr import MBTRExtractor

__all__ = [
    'SOAPExtractor',
    'MultiRcutSOAPAnalyzer',
    'SOAPConfig',
    'CoulombMatrixExtractor',
    'SineMatrixExtractor',
    # 'ACSFExtractor',
    # 'MBTRExtractor',
]