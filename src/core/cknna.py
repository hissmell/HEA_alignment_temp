"""
Centered Kernel Nearest-Neighbor Alignment (CKNNA) module.
Standard implementation based on the paper method used in analyze_31m_cknna_25cao.py
"""

from typing import Optional, Dict, List, Tuple, Union, Any
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.neighbors import NearestNeighbors
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CKNNAConfig:
    """Configuration for CKNNA computation."""
    k: int = 10
    normalize: bool = True
    center_kernel: bool = True
    mutual_nn_only: bool = True
    eps: float = 1e-10


class CKNNA:
    """
    Centered Kernel Nearest-Neighbor Alignment (CKNNA) calculator.

    This is the standard CKNNA implementation based on local alignment metrics
    that measure the similarity of local neighborhoods between two representations.
    """

    def __init__(self, config: Optional[CKNNAConfig] = None):
        """
        Initialize CKNNA calculator.

        Args:
            config: Configuration for CKNNA computation
        """
        self.config = config or CKNNAConfig()

    def compute(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        k: Optional[int] = None
    ) -> float:
        """
        Compute CKNNA between two representations.

        This is the standard paper implementation using:
        1. Normalized representations
        2. Centered kernel matrices
        3. Mutual nearest neighbors
        4. Local alignment metric

        Args:
            X: First representation (n_samples x n_features)
            Y: Second representation (n_samples x n_features)
            k: Number of nearest neighbors (default: config.k)

        Returns:
            CKNNA value between -1 and 1
        """
        k = k or self.config.k
        N = X.shape[0]

        # Validate inputs
        if N < k + 1:
            logger.warning(f"Too few samples ({N}) for k={k}")
            return np.nan

        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"Shape mismatch: X={X.shape}, Y={Y.shape}")

        try:
            # Normalize representations
            if self.config.normalize:
                X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + self.config.eps)
                Y_norm = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + self.config.eps)
            else:
                X_norm = X
                Y_norm = Y

            # Compute kernel matrices (cosine similarity)
            K = X_norm @ X_norm.T
            L = Y_norm @ Y_norm.T

            # Center kernels
            if self.config.center_kernel:
                H = np.eye(N) - np.ones((N, N)) / N
                K_centered = H @ K @ H
                L_centered = H @ L @ H
            else:
                K_centered = K
                L_centered = L

            # Find k-nearest neighbors
            nn_X = NearestNeighbors(n_neighbors=k).fit(X_norm)
            nn_Y = NearestNeighbors(n_neighbors=k).fit(Y_norm)

            _, idx_X = nn_X.kneighbors(X_norm)
            _, idx_Y = nn_Y.kneighbors(Y_norm)

            # Compute mutual nearest neighbors mask
            if self.config.mutual_nn_only:
                alpha = np.zeros((N, N))
                for i in range(N):
                    mutual = set(idx_X[i]) & set(idx_Y[i])
                    for j in mutual:
                        alpha[i, j] = 1
            else:
                # Use all k-NN (not just mutual)
                alpha = np.zeros((N, N))
                for i in range(N):
                    for j in idx_X[i]:
                        alpha[i, j] = 1
                    for j in idx_Y[i]:
                        alpha[i, j] = 1

            # Compute alignment
            align_KL = np.sum(alpha * K_centered * L_centered)
            align_KK = np.sum(alpha * K_centered * K_centered)
            align_LL = np.sum(alpha * L_centered * L_centered)

            if align_KK * align_LL == 0:
                return 0.0

            cknna_value = align_KL / np.sqrt(align_KK * align_LL)

            # Ensure value is in valid range
            cknna_value = np.clip(cknna_value, -1.0, 1.0)

            return cknna_value

        except Exception as e:
            logger.error(f"Error computing CKNNA: {e}")
            return np.nan

    def compute_batch(
        self,
        X_list: List[np.ndarray],
        Y_list: List[np.ndarray],
        k: Optional[int] = None,
        verbose: bool = True
    ) -> np.ndarray:
        """
        Compute CKNNA for multiple structure pairs.

        Args:
            X_list: List of first representations
            Y_list: List of second representations
            k: Number of nearest neighbors
            verbose: Show progress bar

        Returns:
            Array of CKNNA values
        """
        if len(X_list) != len(Y_list):
            raise ValueError(f"List length mismatch: {len(X_list)} vs {len(Y_list)}")

        results = []

        if verbose:
            from tqdm import tqdm
            iterator = tqdm(zip(X_list, Y_list), total=len(X_list), desc="Computing CKNNA")
        else:
            iterator = zip(X_list, Y_list)

        for X, Y in iterator:
            cknna_value = self.compute(X, Y, k=k)
            results.append(cknna_value)

        return np.array(results)

    def compute_with_errors(
        self,
        X_list: List[np.ndarray],
        Y_list: List[np.ndarray],
        errors: np.ndarray,
        k_values: List[int] = [5, 10, 15, 20],
        tasknames: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compute CKNNA for different k values and correlate with errors.

        Args:
            X_list: List of first representations
            Y_list: List of second representations
            errors: Prediction errors for each structure
            k_values: List of k values to test
            tasknames: Optional structure identifiers

        Returns:
            DataFrame with CKNNA values and correlations
        """
        results = []

        for i, (X, Y) in enumerate(zip(X_list, Y_list)):
            result = {
                'index': i,
                'taskname': tasknames[i] if tasknames else f"structure_{i}",
                'error': errors[i],
                'n_atoms': X.shape[0]
            }

            # Compute CKNNA for each k value
            for k in k_values:
                if X.shape[0] >= k + 1:  # Check if we have enough atoms
                    cknna_value = self.compute(X, Y, k=k)
                    result[f'cknna_k{k}'] = cknna_value
                else:
                    result[f'cknna_k{k}'] = np.nan

            results.append(result)

        df = pd.DataFrame(results)

        # Calculate correlations for each k
        correlation_results = []
        for k in k_values:
            col_name = f'cknna_k{k}'

            # Remove NaN values
            mask = ~(df[col_name].isna() | df['error'].isna())
            df_clean = df[mask]

            if len(df_clean) > 10:  # Need sufficient samples for correlation
                spearman_r, spearman_p = stats.spearmanr(df_clean[col_name], df_clean['error'])
                pearson_r, pearson_p = stats.pearsonr(df_clean[col_name], df_clean['error'])

                correlation_results.append({
                    'k': k,
                    'n_samples': len(df_clean),
                    'spearman_r': spearman_r,
                    'spearman_p': spearman_p,
                    'pearson_r': pearson_r,
                    'pearson_p': pearson_p,
                    'significant': spearman_p < 0.05
                })

        corr_df = pd.DataFrame(correlation_results)

        return df, corr_df


class CKNNAAnalyzer:
    """
    Analyzer for CKNNA-based uncertainty estimation and active learning.
    """

    def __init__(
        self,
        cknna_calculator: Optional[CKNNA] = None,
        cache_dir: Optional[Path] = None
    ):
        """
        Initialize CKNNA analyzer.

        Args:
            cknna_calculator: CKNNA calculator instance
            cache_dir: Directory for caching results
        """
        self.cknna = cknna_calculator or CKNNA()
        self.cache_dir = Path(cache_dir) if cache_dir else None

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def analyze_representations(
        self,
        physics_repr: Dict[str, np.ndarray],
        mlip_repr: Dict[str, np.ndarray],
        errors: Dict[str, float],
        k_values: List[int] = [5, 10, 15, 20]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Analyze alignment between physics-inspired and MLIP representations.

        Args:
            physics_repr: Dict mapping taskname to physics representation
            mlip_repr: Dict mapping taskname to MLIP representation
            errors: Dict mapping taskname to prediction errors
            k_values: List of k values to test

        Returns:
            Tuple of (results DataFrame, correlation DataFrame)
        """
        # Find common structures
        common_tasks = set(physics_repr.keys()) & set(mlip_repr.keys()) & set(errors.keys())
        logger.info(f"Found {len(common_tasks)} common structures")

        if len(common_tasks) == 0:
            raise ValueError("No common structures found")

        # Sort tasknames for consistency
        tasknames = sorted(common_tasks)

        # Prepare data lists
        X_list = [physics_repr[task] for task in tasknames]
        Y_list = [mlip_repr[task] for task in tasknames]
        error_array = np.array([errors[task] for task in tasknames])

        # Compute CKNNA with errors
        results_df, corr_df = self.cknna.compute_with_errors(
            X_list, Y_list, error_array, k_values, tasknames
        )

        return results_df, corr_df

    def find_optimal_k(
        self,
        physics_repr: Dict[str, np.ndarray],
        mlip_repr: Dict[str, np.ndarray],
        errors: Dict[str, float],
        k_range: range = range(3, 31, 2)
    ) -> Dict[str, Any]:
        """
        Find optimal k value that maximizes correlation with errors.

        Args:
            physics_repr: Physics-inspired representations
            mlip_repr: MLIP representations
            errors: Prediction errors
            k_range: Range of k values to test

        Returns:
            Dictionary with optimal k and analysis results
        """
        results_df, corr_df = self.analyze_representations(
            physics_repr, mlip_repr, errors, list(k_range)
        )

        # Find k with best correlation
        best_idx = corr_df['spearman_r'].abs().idxmax()
        best_k = corr_df.loc[best_idx, 'k']
        best_corr = corr_df.loc[best_idx, 'spearman_r']

        return {
            'optimal_k': best_k,
            'correlation': best_corr,
            'p_value': corr_df.loc[best_idx, 'spearman_p'],
            'results_df': results_df,
            'correlation_df': corr_df
        }

    def select_uncertain_samples(
        self,
        physics_repr: Dict[str, np.ndarray],
        mlip_repr: Dict[str, np.ndarray],
        n_samples: int,
        k: int = 10,
        strategy: str = 'lowest'
    ) -> List[str]:
        """
        Select samples for active learning based on CKNNA values.

        Args:
            physics_repr: Physics-inspired representations
            mlip_repr: MLIP representations
            n_samples: Number of samples to select
            k: Number of nearest neighbors
            strategy: Selection strategy ('lowest', 'highest', 'diverse')

        Returns:
            List of selected tasknames
        """
        # Compute CKNNA for all structures
        tasknames = sorted(set(physics_repr.keys()) & set(mlip_repr.keys()))
        cknna_values = []

        for task in tasknames:
            cknna = self.cknna.compute(physics_repr[task], mlip_repr[task], k=k)
            cknna_values.append((task, cknna))

        # Sort by CKNNA value
        cknna_values.sort(key=lambda x: x[1])

        # Select based on strategy
        if strategy == 'lowest':
            # Select samples with lowest CKNNA (highest uncertainty)
            selected = [task for task, _ in cknna_values[:n_samples]]

        elif strategy == 'highest':
            # Select samples with highest CKNNA (most certain)
            selected = [task for task, _ in cknna_values[-n_samples:]]

        elif strategy == 'diverse':
            # Select diverse samples across CKNNA range
            step = len(cknna_values) // n_samples
            selected = [cknna_values[i * step][0] for i in range(n_samples)]

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return selected

    def save_results(
        self,
        results_df: pd.DataFrame,
        corr_df: pd.DataFrame,
        output_path: Path
    ):
        """
        Save analysis results to Excel file.

        Args:
            results_df: DataFrame with CKNNA values
            corr_df: DataFrame with correlation analysis
            output_path: Path to output Excel file
        """
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            results_df.to_excel(writer, sheet_name='CKNNA_Results', index=False)
            corr_df.to_excel(writer, sheet_name='Correlations', index=False)

            # Add summary statistics
            summary_stats = pd.DataFrame([{
                'metric': 'Best k value',
                'value': corr_df.loc[corr_df['spearman_r'].abs().idxmax(), 'k']
            }, {
                'metric': 'Best correlation',
                'value': corr_df['spearman_r'].abs().max()
            }, {
                'metric': 'Total structures',
                'value': len(results_df)
            }])
            summary_stats.to_excel(writer, sheet_name='Summary', index=False)

        logger.info(f"Results saved to {output_path}")


# Convenience functions for backward compatibility
def cknna_paper(X: np.ndarray, Y: np.ndarray, k: int = 10) -> float:
    """
    Standard CKNNA computation (paper implementation).
    Kept for backward compatibility with existing code.

    Args:
        X: First representation
        Y: Second representation
        k: Number of nearest neighbors

    Returns:
        CKNNA value
    """
    calculator = CKNNA(CKNNAConfig(k=k))
    return calculator.compute(X, Y)