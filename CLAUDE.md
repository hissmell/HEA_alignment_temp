# Data Efficiency Experiment Progress

## Overview
Implementation of Phase 2 active learning experiments comparing CKNNA-guided vs random data selection for EquiformerV2 31m model fine-tuning.

## Completed Tasks

### 1. CKNNA-Based Data Efficiency Experiment
- **File**: `data_efficiency_experiment.py`
- **Purpose**: Active learning using CKNNA (Centered Kernel Nearest-Neighbor Alignment) for data selection
- **Key Features**:
  - Uses existing pretrained predictions from Excel file (no re-evaluation)
  - 8:2 train/validation split on entire dataset
  - CKNNA-based sorting: lowest values selected first for training
  - Separate WandB projects per dataset size: `data_effi_cknna_{dataset_size}_25cao`
  - Progress tracking with tqdm
  - Detailed result saving with multiple Excel sheets
  - Early stopping with patience-based validation

### 2. Random Sampling Baseline Experiment
- **File**: `data_efficiency_experiment_random.py`
- **Purpose**: Random sampling baseline for comparison with CKNNA strategy
- **Key Features**:
  - Multiple random seeds support (5 runs: 42, 123, 456, 789, 1024)
  - Seed-specific WandB projects: `data_effi_random_{dataset_size}_seed{seed}_25cao`
  - Seed-specific file naming for all outputs
  - Statistical significance through multiple runs
  - Same evaluation framework as CKNNA experiment

## Implementation Details

### Data Preparation
- **Dataset**: 25Cao structures
- **Split Strategy**: 8:2 train/validation on entire dataset
- **CKNNA Data**: Pre-computed values from `/DATA/user_scratch/pn50212/2024/12_AtomAttention/cknna_analysis_31m/cknna_analysis_31m_extended_results.xlsx`
- **Pretrained Predictions**: Loaded from `/DATA/user_scratch/pn50212/2024/12_AtomAttention/datasets/25Cao/equiformer_31m_baseline_performance.xlsx`

### Model Configuration
- **Model**: EquiformerV2 31m (eq2_31M_ec4_allmd)
- **Training**: Only final layers trained (backbone frozen)
- **Optimization**: AdamW optimizer with early stopping
- **Default Parameters**:
  - Learning rate: 1e-4
  - Batch size: 8
  - Max epochs: 30
  - Patience: 8

### Output Structure
```
data_efficiency_results/
├── experiment_cknna_{timestamp}/
│   ├── individual_results/
│   │   ├── result_cknna_size_{N}.xlsx
│   │   ├── detailed_predictions_cknna_size_{N}.xlsx
│   │   └── training_data_details_cknna_size_{N}.xlsx
│   ├── plots/
│   └── checkpoints/
└── experiment_random_seed{seed}_{timestamp}/
    ├── individual_results/
    │   ├── result_random_size_{N}_seed{seed}.xlsx
    │   ├── detailed_predictions_random_size_{N}_seed{seed}.xlsx
    │   └── training_data_details_random_size_{N}_seed{seed}.xlsx
    ├── plots/
    └── checkpoints/
```

### WandB Logging
- **CKNNA Projects**: `data_effi_cknna_{dataset_size}_25cao`
- **Random Projects**: `data_effi_random_{dataset_size}_seed{seed}_25cao`
- **Metrics Tracked**:
  - Epoch-wise: train_loss, train_mae, val_loss, val_mae
  - Final: validation metrics on best model

### Result Files Generated

#### Individual Results
1. **Basic Metrics**: JSON and Excel files with validation loss, MAE, RMSE
2. **Detailed Predictions**:
   - Validation set: taskname, true_value, predicted_value, prediction_error, absolute_error
   - Multiple Excel sheets: main results, statistics, worst/best predictions
3. **Training Data Details**:
   - Selected training data with pretrained predictions and CKNNA values
   - Selection order and sampling strategy information

## Key Improvements Made

### Error Fixes
1. **Validation Efficiency**: Eliminated 4+ hour validation by using cached pretrained predictions
2. **Data Split Correction**: Fixed to use proper 8:2 split instead of high CKNNA for validation
3. **Progress Tracking**: Added comprehensive tqdm progress bars
4. **WandB Structure**: Separate projects per dataset size with epoch-wise logging

### Statistical Significance
- **Random Experiments**: 5 different seeds per dataset size
- **Seed Management**: All outputs are seed-specific for proper comparison
- **Reproducibility**: Fixed random states for consistent results

## Usage

### CKNNA Experiment
```bash
python data_efficiency_experiment.py --dataset_sizes 100,200,500,1000 --use_wandb
```

### Random Experiment (Multiple Seeds)
```bash
python data_efficiency_experiment_random.py --dataset_sizes 100,200,500,1000 --random_seeds 42,123,456,789,1024 --use_wandb
```

## Current Status
✅ **Complete**: Both CKNNA and random sampling experiments ready for execution
✅ **Complete**: Multiple seed support for statistical significance
✅ **Complete**: Comprehensive result logging and visualization
✅ **Complete**: WandB integration with proper project organization

## Next Steps
- Execute experiments across different dataset sizes
- Compare CKNNA vs random sampling performance
- Statistical analysis of multiple seed results
- Generate final comparison plots and reports

## File Dependencies
- `MLPs/mlp_infos.yml`: Model configuration
- CKNNA analysis results Excel file
- Pretrained model predictions Excel file
- 25Cao dataset structures

## Notes
- All experiments use the same validation set for fair comparison
- CKNNA values are included in random experiment files for reference analysis
- Early stopping prevents overfitting across all experiments
- Results are automatically organized by timestamp and seed for easy tracking