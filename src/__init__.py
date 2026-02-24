"""
AtomAttention: Representation Alignment for Uncertainty Estimation in MLIPs
"""

__version__ = "0.2.0"
__author__ = "AtomAttention Team"

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "results"

# Model directories
MODELS_DIR = PROJECT_ROOT / "MLPs"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

# Config directories
CONFIGS_DIR = PROJECT_ROOT / "configs"
MODEL_CONFIGS = CONFIGS_DIR / "models"
EXPERIMENT_CONFIGS = CONFIGS_DIR / "experiments"