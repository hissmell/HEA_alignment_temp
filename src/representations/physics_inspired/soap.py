"""
SOAP (Smooth Overlap of Atomic Positions) representation extractor.
Based on the existing extract_soap_representations_25cao.py implementation.
"""

from typing import Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from ase import Atoms
from dataclasses import dataclass

from ..base import PhysicsInspiredExtractor, ExtractionConfig

logger = logging.getLogger(__name__)

try:
    from dscribe.descriptors import SOAP
    HAS_DSCRIBE = True
except ImportError:
    HAS_DSCRIBE = False
    logger.error("dscribe not available. Please install: pip install dscribe")


@dataclass
class SOAPConfig:
    """Configuration for SOAP descriptor."""
    r_cut: float = 6.0
    n_max: int = 8
    l_max: int = 6
    periodic: bool = True
    sparse: bool = False
    sigma: float = 1.0
    species: Optional[List[str]] = None
    rcut_values: Optional[List[float]] = None  # For multi-rcut analysis


class SOAPExtractor(PhysicsInspiredExtractor):
    """
    SOAP representation extractor with support for multiple atom selections.
    Consolidates functionality from multiple SOAP extraction scripts.
    """

    def __init__(
        self,
        soap_config: Optional[SOAPConfig] = None,
        config: Optional[ExtractionConfig] = None
    ):
        """
        Initialize SOAP extractor.

        Args:
            soap_config: SOAP-specific configuration
            config: General extraction configuration
        """
        super().__init__(config)
        self.soap_config = soap_config or SOAPConfig()
        self.soap_full = None
        self.soap_slab = None
        self._species_full = None
        self._species_slab = None

    @property
    def descriptor_name(self) -> str:
        return "soap"

    def setup(self) -> None:
        """Initialize SOAP descriptors."""
        if not HAS_DSCRIBE:
            raise ImportError("dscribe is required for SOAP extraction")

        # Determine species
        if self.soap_config.species:
            self._species_full = self.soap_config.species.copy()
        else:
            # Default HEA species (will be updated when processing structures)
            self._species_full = ['Ti', 'Zr', 'Hf', 'V', 'Nb', 'Ta', 'Cr', 'Mo', 'W', 'O', 'H']

        # Species for slab-only calculations (exclude adsorbate atoms O, H)
        self._species_slab = [s for s in self._species_full if s not in ['O', 'H']]

        # Initialize SOAP descriptors
        self.soap_full = SOAP(
            species=self._species_full,
            periodic=self.soap_config.periodic,
            sparse=self.soap_config.sparse,
            r_cut=self.soap_config.r_cut,
            n_max=self.soap_config.n_max,
            l_max=self.soap_config.l_max,
            sigma=self.soap_config.sigma
        )

        self.soap_slab = SOAP(
            species=self._species_slab,
            periodic=self.soap_config.periodic,
            sparse=self.soap_config.sparse,
            r_cut=self.soap_config.r_cut,
            n_max=self.soap_config.n_max,
            l_max=self.soap_config.l_max,
            sigma=self.soap_config.sigma
        )

        self.is_initialized = True
        logger.info(f"SOAP descriptors initialized:")
        logger.info(f"  Full species: {self._species_full}")
        logger.info(f"  Slab species: {self._species_slab}")
        logger.info(f"  Parameters: r_cut={self.soap_config.r_cut}, n_max={self.soap_config.n_max}, l_max={self.soap_config.l_max}")

    def extract_single(
        self,
        atoms: Atoms,
        atom_selection: str = "all",
        adsorbate_indices: Optional[List[int]] = None,
        site_cutoff: float = 3.0,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Extract SOAP descriptors for a single structure.

        Args:
            atoms: ASE Atoms object
            atom_selection: Which atoms to include ("all", "slab", "site", "multi")
            adsorbate_indices: Indices of adsorbate atoms (auto-detected if None)
            site_cutoff: Distance cutoff for site region selection
            **kwargs: Additional arguments

        Returns:
            Dictionary with SOAP arrays for requested selections
        """
        if not self.is_initialized:
            self.setup()

        # Auto-detect adsorbate indices if not provided
        if adsorbate_indices is None:
            adsorbate_indices = self._detect_adsorbate_indices(atoms)

        results = {}

        try:
            if atom_selection in ["all", "multi"]:
                # Full structure SOAP
                soap_full_vectors = self.soap_full.create(atoms)
                results["soap_full"] = soap_full_vectors

            if atom_selection in ["slab", "multi"]:
                # Slab-only SOAP
                slab_atoms = self._get_slab_atoms(atoms, adsorbate_indices)
                if slab_atoms is not None:
                    soap_slab_vectors = self.soap_slab.create(slab_atoms)
                    results["soap_slab"] = soap_slab_vectors

            if atom_selection in ["site", "multi"]:
                # Site region SOAP
                site_atoms = self._get_site_atoms(atoms, adsorbate_indices, site_cutoff)
                if site_atoms is not None:
                    soap_site_vectors = self.soap_slab.create(site_atoms)  # Use slab descriptor
                    results["soap_site"] = soap_site_vectors

        except Exception as e:
            logger.error(f"Failed to extract SOAP: {e}")
            return {}

        return results

    def extract_multi_rcut(
        self,
        atoms: Atoms,
        rcut_values: List[float],
        atom_selection: str = "all",
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Extract SOAP descriptors with multiple cutoff values.

        Args:
            atoms: ASE Atoms object
            rcut_values: List of cutoff values to test
            atom_selection: Which atoms to include
            **kwargs: Additional arguments

        Returns:
            Dictionary mapping rcut values to SOAP arrays
        """
        results = {}

        for rcut in rcut_values:
            # Create temporary SOAP descriptor with specific rcut
            temp_config = SOAPConfig(
                r_cut=rcut,
                n_max=self.soap_config.n_max,
                l_max=self.soap_config.l_max,
                periodic=self.soap_config.periodic,
                sparse=self.soap_config.sparse,
                sigma=self.soap_config.sigma,
                species=self._species_full
            )

            temp_extractor = SOAPExtractor(temp_config, self.config)
            temp_extractor.setup()

            soap_results = temp_extractor.extract_single(atoms, atom_selection, **kwargs)

            # Store with rcut prefix
            for key, value in soap_results.items():
                results[f"{key}_rcut{rcut}"] = value

        return results

    def _detect_adsorbate_indices(self, atoms: Atoms) -> List[int]:
        """
        Auto-detect adsorbate atom indices.
        Uses heuristic: last 1-2 atoms are typically adsorbates.
        """
        symbols = atoms.get_chemical_symbols()

        # Check for O and OH adsorbates
        if len(symbols) >= 2 and symbols[-2:] == ['O', 'H']:
            return [len(symbols) - 2, len(symbols) - 1]  # OH adsorbate
        elif len(symbols) >= 1 and symbols[-1] == 'O':
            return [len(symbols) - 1]  # O adsorbate
        else:
            # Fallback: assume last atom
            return [len(symbols) - 1] if len(symbols) > 0 else []

    def _get_slab_atoms(self, atoms: Atoms, adsorbate_indices: List[int]) -> Optional[Atoms]:
        """Get slab atoms (excluding adsorbates)."""
        if not adsorbate_indices:
            return atoms

        slab_indices = [i for i in range(len(atoms)) if i not in adsorbate_indices]
        return atoms[slab_indices] if slab_indices else None

    def _get_site_atoms(
        self,
        atoms: Atoms,
        adsorbate_indices: List[int],
        site_cutoff: float
    ) -> Optional[Atoms]:
        """Get atoms in the adsorption site region."""
        if not adsorbate_indices:
            return None

        # Calculate distances to adsorbate atoms
        adsorbate_positions = atoms.positions[adsorbate_indices]
        distances = np.linalg.norm(
            atoms.positions[:, np.newaxis, :] - adsorbate_positions[np.newaxis, :, :],
            axis=2
        )
        min_distances = distances.min(axis=1)

        # Select atoms within cutoff (excluding adsorbate atoms themselves)
        site_mask = min_distances <= site_cutoff
        site_indices = np.where(site_mask)[0]
        site_slab_indices = [i for i in site_indices if i not in adsorbate_indices]

        if len(site_slab_indices) >= 2:
            return atoms[site_slab_indices]

        return None

    def get_feature_names(self) -> List[str]:
        """Return SOAP feature names."""
        if not self.is_initialized:
            return []

        return [
            "soap_full",
            "soap_slab",
            "soap_site"
        ]

    def get_feature_dimensions(self) -> Dict[str, int]:
        """Return SOAP feature dimensions."""
        if not self.is_initialized:
            self.setup()

        return {
            "soap_full": self.soap_full.get_number_of_features(),
            "soap_slab": self.soap_slab.get_number_of_features()
        }

    def update_species(self, atoms_list: List[Atoms]) -> None:
        """
        Update species list based on all structures in the dataset.

        Args:
            atoms_list: List of all structures to analyze
        """
        all_species = set()
        for atoms in atoms_list:
            all_species.update(atoms.get_chemical_symbols())

        # Update species and reinitialize
        self._species_full = sorted(list(all_species))
        self._species_slab = [s for s in self._species_full if s not in ['O', 'H']]

        self.is_initialized = False
        self.setup()

        logger.info(f"Updated SOAP species: {self._species_full}")


class MultiRcutSOAPAnalyzer:
    """
    Analyzer for SOAP representations across different cutoff values.
    Replaces the functionality from multiple rcut-specific scripts.
    """

    def __init__(
        self,
        rcut_values: List[float],
        soap_config: Optional[SOAPConfig] = None,
        config: Optional[ExtractionConfig] = None
    ):
        """
        Initialize multi-rcut SOAP analyzer.

        Args:
            rcut_values: List of cutoff values to analyze
            soap_config: Base SOAP configuration
            config: Extraction configuration
        """
        self.rcut_values = rcut_values
        self.soap_config = soap_config or SOAPConfig()
        self.config = config or ExtractionConfig()
        self.extractors = {}

        # Create extractor for each rcut value
        for rcut in rcut_values:
            rcut_config = SOAPConfig(
                r_cut=rcut,
                n_max=self.soap_config.n_max,
                l_max=self.soap_config.l_max,
                periodic=self.soap_config.periodic,
                sparse=self.soap_config.sparse,
                sigma=self.soap_config.sigma,
                species=self.soap_config.species
            )
            self.extractors[rcut] = SOAPExtractor(rcut_config, config)

    def extract_all_rcuts(
        self,
        atoms: Atoms,
        atom_selection: str = "slab",
        **kwargs
    ) -> Dict[float, Dict[str, np.ndarray]]:
        """
        Extract SOAP for all rcut values.

        Args:
            atoms: ASE Atoms object
            atom_selection: Which atoms to include
            **kwargs: Additional arguments

        Returns:
            Dictionary mapping rcut values to SOAP results
        """
        results = {}

        for rcut, extractor in self.extractors.items():
            try:
                soap_results = extractor.extract_single(atoms, atom_selection, **kwargs)
                results[rcut] = soap_results
            except Exception as e:
                logger.warning(f"Failed to extract SOAP for rcut={rcut}: {e}")

        return results

    def analyze_rcut_range(
        self,
        structures: Dict[str, Atoms],
        mlip_representations: Dict[str, np.ndarray],
        errors: Dict[str, float],
        output_dir: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Analyze SOAP alignment across rcut values.

        Args:
            structures: Dictionary mapping tasknames to structures
            mlip_representations: MLIP representations for comparison
            errors: Prediction errors
            output_dir: Output directory for results

        Returns:
            DataFrame with rcut analysis results
        """
        from ...core.alignment import AlignmentAnalyzer

        analyzer = AlignmentAnalyzer()
        rcut_results = []

        for rcut in self.rcut_values:
            logger.info(f"Analyzing rcut={rcut}")

            # Extract SOAP for all structures with this rcut
            soap_representations = {}
            extractor = self.extractors[rcut]

            for taskname, atoms in structures.items():
                soap_result = extractor.extract_single(atoms, "slab")
                if "soap_slab" in soap_result:
                    soap_representations[taskname] = soap_result["soap_slab"]

            # Analyze alignment with MLIP representations
            results_df, corr_df = analyzer.analyze_with_errors(
                soap_representations,
                mlip_representations,
                errors,
                metrics=['cknna', 'dcor', 'cosine']
            )

            # Get best correlation
            best_idx = corr_df['spearman_r'].abs().idxmax()
            best_result = corr_df.loc[best_idx]

            rcut_results.append({
                'rcut': rcut,
                'best_metric': best_result['metric'],
                'correlation': best_result['spearman_r'],
                'p_value': best_result['spearman_p'],
                'n_structures': len(results_df),
                'soap_dimension': extractor.get_feature_dimensions()['soap_slab']
            })

            # Save detailed results if output directory provided
            if output_dir:
                rcut_dir = output_dir / f"rcut_{rcut}"
                rcut_dir.mkdir(exist_ok=True, parents=True)

                results_df.to_excel(rcut_dir / "alignment_results.xlsx", index=False)
                corr_df.to_excel(rcut_dir / "correlations.xlsx", index=False)

        # Create summary DataFrame
        rcut_df = pd.DataFrame(rcut_results)

        # Find optimal rcut
        optimal_idx = rcut_df['correlation'].abs().idxmax()
        optimal_rcut = rcut_df.loc[optimal_idx, 'rcut']

        logger.info(f"Optimal rcut: {optimal_rcut} (correlation: {rcut_df.loc[optimal_idx, 'correlation']:.4f})")

        if output_dir:
            rcut_df.to_excel(output_dir / "rcut_comparison.xlsx", index=False)

        return rcut_df