"""
Unified alignment analysis module for representation comparison.
Consolidates functionality from multiple alignment analysis scripts.
"""

from typing import Optional, Dict, List, Tuple, Union, Any
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import logging
from dataclasses import dataclass

# Import CKNNA from the core module
from .cknna import CKNNA, CKNNAConfig

logger = logging.getLogger(__name__)


@dataclass
class AlignmentConfig:
    """Configuration for alignment analysis."""
    metric: str = 'cknna'  # 'cknna', 'dcor', 'cosine', 'procrustes'
    k_neighbors: int = 10
    normalize: bool = True
    pca_dim: Optional[int] = None  # For dimension reduction
    cache_dir: Optional[Path] = None


class AlignmentAnalyzer:
    """
    Unified class for alignment analysis between different representations.
    Consolidates functionality from multiple analysis scripts.
    """

    def __init__(self, config: Optional[AlignmentConfig] = None):
        """
        Initialize alignment analyzer.

        Args:
            config: Configuration for alignment analysis
        """
        self.config = config or AlignmentConfig()
        self.cknna_calculator = CKNNA(CKNNAConfig(k=self.config.k_neighbors))

        if self.config.cache_dir:
            self.config.cache_dir.mkdir(parents=True, exist_ok=True)

    def compute_alignment(
        self,
        repr1: np.ndarray,
        repr2: np.ndarray,
        metric: Optional[str] = None
    ) -> float:
        """
        Compute alignment between two representations.

        Args:
            repr1: First representation matrix (n_samples x n_features)
            repr2: Second representation matrix (n_samples x n_features)
            metric: Alignment metric to use (default: config.metric)

        Returns:
            Alignment score
        """
        metric = metric or self.config.metric

        # Validate inputs
        if repr1.shape[0] != repr2.shape[0]:
            raise ValueError(f"Sample size mismatch: {repr1.shape[0]} vs {repr2.shape[0]}")

        # Apply PCA if dimensions differ or if requested
        if self.config.pca_dim or repr1.shape[1] != repr2.shape[1]:
            repr1, repr2 = self._match_dimensions(repr1, repr2)

        # Compute alignment based on metric
        if metric == 'cknna':
            return self.cknna_calculator.compute(repr1, repr2)
        elif metric == 'dcor':
            return self._compute_dcor(repr1, repr2)
        elif metric == 'cosine':
            return self._compute_cosine_mean(repr1, repr2)
        elif metric == 'procrustes':
            return self._compute_procrustes(repr1, repr2)
        else:
            raise ValueError(f"Unknown alignment metric: {metric}")

    def _match_dimensions(
        self,
        repr1: np.ndarray,
        repr2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Match dimensions of two representations using PCA.
        """
        # Determine safe target dimension
        max_dim = min(repr1.shape[0] - 1, repr2.shape[0] - 1)  # PCA constraint

        if self.config.pca_dim:
            target_dim = min(self.config.pca_dim, max_dim)
        else:
            target_dim = min(repr1.shape[1], repr2.shape[1], max_dim)

        # Apply PCA to both representations
        if repr1.shape[1] > target_dim:
            pca1 = PCA(n_components=target_dim)
            repr1 = pca1.fit_transform(repr1)

        if repr2.shape[1] > target_dim:
            pca2 = PCA(n_components=target_dim)
            repr2 = pca2.fit_transform(repr2)

        return repr1, repr2

    def _compute_dcor(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Compute Distance Correlation (dCor).
        Global alignment metric - measures structural similarity.
        """
        if X.shape[0] < 2:
            return np.nan

        try:
            # Compute distance matrices
            a = squareform(pdist(X))
            b = squareform(pdist(Y))

            def double_center(D):
                n = D.shape[0]
                row_mean = D.mean(axis=1, keepdims=True)
                col_mean = D.mean(axis=0, keepdims=True)
                grand_mean = D.mean()
                return D - row_mean - col_mean + grand_mean

            A = double_center(a)
            B = double_center(b)

            dcov_xy = np.sqrt(np.abs(np.mean(A * B)))
            dcov_xx = np.sqrt(np.abs(np.mean(A * A)))
            dcov_yy = np.sqrt(np.abs(np.mean(B * B)))

            if dcov_xx * dcov_yy == 0:
                return 0.0

            return dcov_xy / np.sqrt(dcov_xx * dcov_yy)
        except Exception as e:
            logger.error(f"Error computing dCor: {e}")
            return np.nan

    def _compute_cosine_mean(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Compute mean pairwise cosine similarity.
        """
        try:
            # Normalize representations
            X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
            Y_norm = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-10)

            # Compute pairwise cosine similarities
            cos_sims = np.sum(X_norm * Y_norm, axis=1)
            return np.mean(cos_sims)
        except Exception as e:
            logger.error(f"Error computing cosine similarity: {e}")
            return np.nan

    def _compute_procrustes(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Compute Procrustes alignment (rotation + scaling).
        Returns similarity score (1 - normalized distance).
        """
        try:
            from scipy.spatial import procrustes

            # Center the data
            X_centered = X - X.mean(axis=0)
            Y_centered = Y - Y.mean(axis=0)

            # Compute Procrustes distance
            _, _, disparity = procrustes(X_centered, Y_centered)

            # Convert to similarity (1 - normalized distance)
            max_dist = np.sqrt(2)  # Maximum possible normalized distance
            similarity = 1 - (disparity / max_dist)

            return similarity
        except Exception as e:
            logger.error(f"Error computing Procrustes: {e}")
            return np.nan

    def analyze_with_errors(
        self,
        repr1_dict: Dict[str, np.ndarray],
        repr2_dict: Dict[str, np.ndarray],
        errors: Dict[str, float],
        metrics: List[str] = ['cknna', 'dcor', 'cosine'],
        k_values: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Analyze alignment with multiple metrics and correlate with errors.

        Args:
            repr1_dict: Dict mapping taskname to first representation
            repr2_dict: Dict mapping taskname to second representation
            errors: Dict mapping taskname to prediction errors
            metrics: List of alignment metrics to compute
            k_values: List of k values for CKNNA (if applicable)

        Returns:
            DataFrame with alignment scores and correlations
        """
        # Find common structures
        common_tasks = set(repr1_dict.keys()) & set(repr2_dict.keys()) & set(errors.keys())
        tasknames = sorted(common_tasks)

        if len(tasknames) == 0:
            raise ValueError("No common structures found")

        results = []

        for taskname in tasknames:
            result = {
                'taskname': taskname,
                'error': errors[taskname],
                'n_atoms': repr1_dict[taskname].shape[0]
            }

            # Compute each metric
            for metric in metrics:
                if metric == 'cknna' and k_values:
                    # Test multiple k values
                    for k in k_values:
                        self.cknna_calculator.config.k = k
                        score = self.compute_alignment(
                            repr1_dict[taskname],
                            repr2_dict[taskname],
                            metric='cknna'
                        )
                        result[f'cknna_k{k}'] = score
                else:
                    score = self.compute_alignment(
                        repr1_dict[taskname],
                        repr2_dict[taskname],
                        metric=metric
                    )
                    result[metric] = score

            results.append(result)

        df = pd.DataFrame(results)

        # Calculate correlations
        correlation_results = []

        for col in df.columns:
            if col in ['taskname', 'error', 'n_atoms']:
                continue

            # Remove NaN values
            mask = ~(df[col].isna() | df['error'].isna())
            if mask.sum() < 10:
                continue

            df_clean = df[mask]

            # Calculate correlations
            spearman_r, spearman_p = stats.spearmanr(df_clean[col], df_clean['error'])
            pearson_r, pearson_p = stats.pearsonr(df_clean[col], df_clean['error'])

            correlation_results.append({
                'metric': col,
                'n_samples': len(df_clean),
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'significant': spearman_p < 0.05
            })

        corr_df = pd.DataFrame(correlation_results)

        return df, corr_df

    def compare_representations(
        self,
        representations: Dict[str, Dict[str, np.ndarray]],
        errors: Dict[str, float],
        output_dir: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Compare multiple representations against each other.

        Args:
            representations: Dict of representation name to taskname->array mapping
            errors: Prediction errors
            output_dir: Optional output directory for results

        Returns:
            DataFrame with pairwise comparison results
        """
        repr_names = list(representations.keys())
        comparison_results = []

        # Pairwise comparisons
        for i, repr1_name in enumerate(repr_names):
            for j, repr2_name in enumerate(repr_names):
                if i >= j:  # Skip self and duplicate comparisons
                    continue

                logger.info(f"Comparing {repr1_name} vs {repr2_name}")

                # Analyze alignment
                results_df, corr_df = self.analyze_with_errors(
                    representations[repr1_name],
                    representations[repr2_name],
                    errors,
                    metrics=['cknna', 'dcor', 'cosine']
                )

                # Get best correlation
                best_idx = corr_df['spearman_r'].abs().idxmax()
                best_metric = corr_df.loc[best_idx]

                comparison_results.append({
                    'repr1': repr1_name,
                    'repr2': repr2_name,
                    'best_metric': best_metric['metric'],
                    'correlation': best_metric['spearman_r'],
                    'p_value': best_metric['spearman_p'],
                    'n_structures': len(results_df)
                })

                # Save detailed results if output_dir provided
                if output_dir:
                    output_file = output_dir / f"alignment_{repr1_name}_vs_{repr2_name}.xlsx"
                    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                        results_df.to_excel(writer, sheet_name='Alignment_Scores', index=False)
                        corr_df.to_excel(writer, sheet_name='Correlations', index=False)

        comparison_df = pd.DataFrame(comparison_results)

        if output_dir:
            comparison_df.to_excel(output_dir / 'representation_comparison.xlsx', index=False)

        return comparison_df


class SOAPAlignmentAnalyzer(AlignmentAnalyzer):
    """
    Specialized analyzer for SOAP representations with different cutoffs.
    Replaces the multiple rcut-specific scripts.
    """

    def __init__(
        self,
        rcut: float = 8.0,
        config: Optional[AlignmentConfig] = None
    ):
        """
        Initialize SOAP alignment analyzer.

        Args:
            rcut: SOAP cutoff radius
            config: Alignment configuration
        """
        super().__init__(config)
        self.rcut = rcut

    def analyze_rcut_range(
        self,
        soap_loader,  # Function to load SOAP with given rcut
        mlip_repr: Dict[str, np.ndarray],
        errors: Dict[str, float],
        rcut_range: List[float],
        output_dir: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Analyze alignment across different SOAP cutoff values.

        Args:
            soap_loader: Function that takes rcut and returns SOAP representations
            mlip_repr: MLIP representations
            errors: Prediction errors
            rcut_range: List of rcut values to test
            output_dir: Output directory for results

        Returns:
            DataFrame with results for each rcut
        """
        rcut_results = []

        for rcut in rcut_range:
            logger.info(f"Analyzing rcut={rcut}")

            # Load SOAP with specific rcut
            soap_repr = soap_loader(rcut)

            # Analyze alignment
            results_df, corr_df = self.analyze_with_errors(
                soap_repr,
                mlip_repr,
                errors,
                metrics=['cknna', 'dcor', 'cosine']
            )

            # Get best metric
            best_idx = corr_df['spearman_r'].abs().idxmax()

            rcut_results.append({
                'rcut': rcut,
                'best_metric': corr_df.loc[best_idx, 'metric'],
                'correlation': corr_df.loc[best_idx, 'spearman_r'],
                'p_value': corr_df.loc[best_idx, 'spearman_p'],
                'cknna_corr': corr_df[corr_df['metric'].str.contains('cknna')]['spearman_r'].max() if 'cknna' in corr_df['metric'].values else np.nan,
                'dcor_corr': corr_df[corr_df['metric'] == 'dcor']['spearman_r'].values[0] if 'dcor' in corr_df['metric'].values else np.nan,
                'cosine_corr': corr_df[corr_df['metric'] == 'cosine']['spearman_r'].values[0] if 'cosine' in corr_df['metric'].values else np.nan,
                'n_structures': len(results_df)
            })

            # Save individual results
            if output_dir:
                rcut_dir = output_dir / f"rcut_{rcut}"
                rcut_dir.mkdir(exist_ok=True, parents=True)

                with pd.ExcelWriter(rcut_dir / 'alignment_results.xlsx', engine='openpyxl') as writer:
                    results_df.to_excel(writer, sheet_name='Alignment', index=False)
                    corr_df.to_excel(writer, sheet_name='Correlations', index=False)

        rcut_df = pd.DataFrame(rcut_results)

        # Find optimal rcut
        optimal_idx = rcut_df['correlation'].abs().idxmax()
        optimal_rcut = rcut_df.loc[optimal_idx, 'rcut']

        logger.info(f"Optimal rcut: {optimal_rcut} (correlation: {rcut_df.loc[optimal_idx, 'correlation']:.4f})")

        if output_dir:
            rcut_df.to_excel(output_dir / 'rcut_comparison.xlsx', index=False)

        return rcut_df


# Convenience functions for backward compatibility
def compute_cknna_alignment(X: np.ndarray, Y: np.ndarray, k: int = 10) -> float:
    """Backward compatible CKNNA computation."""
    analyzer = AlignmentAnalyzer(AlignmentConfig(metric='cknna', k_neighbors=k))
    return analyzer.compute_alignment(X, Y)


def compute_dcor_alignment(X: np.ndarray, Y: np.ndarray) -> float:
    """Backward compatible dCor computation."""
    analyzer = AlignmentAnalyzer(AlignmentConfig(metric='dcor'))
    return analyzer.compute_alignment(X, Y)