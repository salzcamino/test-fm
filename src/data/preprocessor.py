"""Preprocessing pipeline for scRNA-seq data."""

import anndata as ad
import scanpy as sc
import numpy as np
import pandas as pd
from typing import Optional, List, Union
import logging

logger = logging.getLogger(__name__)


class scRNAPreprocessor:
    """Preprocessor for single-cell RNA-seq data."""

    def __init__(
        self,
        min_genes: int = 200,
        min_cells: int = 3,
        max_genes: Optional[int] = None,
        max_pct_mito: float = 20.0,
        target_sum: float = 1e4,
        n_top_genes: int = 2000,
        normalize: bool = True,
        log_transform: bool = True,
        scale: bool = False
    ):
        """Initialize preprocessor.

        Args:
            min_genes: Minimum genes per cell
            min_cells: Minimum cells per gene
            max_genes: Maximum genes per cell (doublet detection)
            max_pct_mito: Maximum mitochondrial percentage
            target_sum: Target sum for normalization
            n_top_genes: Number of highly variable genes to select
            normalize: Whether to normalize data
            log_transform: Whether to apply log1p transformation
            scale: Whether to scale data
        """
        self.min_genes = min_genes
        self.min_cells = min_cells
        self.max_genes = max_genes
        self.max_pct_mito = max_pct_mito
        self.target_sum = target_sum
        self.n_top_genes = n_top_genes
        self.normalize = normalize
        self.log_transform = log_transform
        self.scale = scale

        self.hvg_genes = None  # Will store highly variable genes

    def calculate_qc_metrics(self, adata: ad.AnnData, inplace: bool = True) -> Optional[ad.AnnData]:
        """Calculate QC metrics.

        Args:
            adata: AnnData object
            inplace: Whether to modify in place

        Returns:
            AnnData object if not inplace
        """
        if not inplace:
            adata = adata.copy()

        logger.info("Calculating QC metrics")

        # Identify mitochondrial genes
        adata.var['mt'] = adata.var_names.str.startswith('MT-')

        # Calculate QC metrics
        sc.pp.calculate_qc_metrics(
            adata,
            qc_vars=['mt'],
            percent_top=None,
            log1p=False,
            inplace=True
        )

        if not inplace:
            return adata

    def filter_cells(self, adata: ad.AnnData, inplace: bool = True) -> ad.AnnData:
        """Filter cells based on QC metrics.

        Args:
            adata: AnnData object
            inplace: Whether to modify in place

        Returns:
            Filtered AnnData object
        """
        if not inplace:
            adata = adata.copy()

        n_cells_before = adata.n_obs
        logger.info(f"Filtering cells (starting with {n_cells_before} cells)")

        # Filter by minimum genes
        sc.pp.filter_cells(adata, min_genes=self.min_genes)

        # Filter by maximum genes (doublets)
        if self.max_genes is not None:
            adata = adata[adata.obs['n_genes_by_counts'] < self.max_genes, :]

        # Filter by mitochondrial percentage
        if 'pct_counts_mt' in adata.obs.columns:
            adata = adata[adata.obs['pct_counts_mt'] < self.max_pct_mito, :]

        n_cells_after = adata.n_obs
        logger.info(f"Filtered {n_cells_before - n_cells_after} cells, {n_cells_after} remaining")

        return adata

    def filter_genes(self, adata: ad.AnnData, inplace: bool = True) -> Optional[ad.AnnData]:
        """Filter genes based on expression.

        Args:
            adata: AnnData object
            inplace: Whether to modify in place

        Returns:
            AnnData object if not inplace
        """
        if not inplace:
            adata = adata.copy()

        n_genes_before = adata.n_vars
        logger.info(f"Filtering genes (starting with {n_genes_before} genes)")

        # Filter by minimum cells
        sc.pp.filter_genes(adata, min_cells=self.min_cells)

        n_genes_after = adata.n_vars
        logger.info(f"Filtered {n_genes_before - n_genes_after} genes, {n_genes_after} remaining")

        if not inplace:
            return adata

    def normalize_data(self, adata: ad.AnnData, inplace: bool = True) -> Optional[ad.AnnData]:
        """Normalize data.

        Args:
            adata: AnnData object
            inplace: Whether to modify in place

        Returns:
            AnnData object if not inplace
        """
        if not inplace:
            adata = adata.copy()

        if self.normalize:
            logger.info(f"Normalizing to {self.target_sum} counts per cell")
            sc.pp.normalize_total(adata, target_sum=self.target_sum, inplace=True)

        if self.log_transform:
            logger.info("Applying log1p transformation")
            sc.pp.log1p(adata)

        if not inplace:
            return adata

    def find_highly_variable_genes(
        self,
        adata: ad.AnnData,
        flavor: str = 'seurat_v3',
        inplace: bool = True
    ) -> Optional[ad.AnnData]:
        """Identify highly variable genes.

        Args:
            adata: AnnData object
            flavor: Method for HVG selection
            inplace: Whether to modify in place

        Returns:
            AnnData object if not inplace
        """
        if not inplace:
            adata = adata.copy()

        logger.info(f"Selecting {self.n_top_genes} highly variable genes using {flavor}")

        # Find HVGs
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=self.n_top_genes,
            flavor=flavor,
            subset=False,
            inplace=True
        )

        # Store HVG names
        self.hvg_genes = adata.var_names[adata.var['highly_variable']].tolist()
        logger.info(f"Found {len(self.hvg_genes)} highly variable genes")

        if not inplace:
            return adata

    def subset_to_hvg(self, adata: ad.AnnData, inplace: bool = True) -> Optional[ad.AnnData]:
        """Subset data to highly variable genes.

        Args:
            adata: AnnData object
            inplace: Whether to modify in place

        Returns:
            AnnData object (always returns for consistency)
        """
        if not inplace:
            adata = adata.copy()

        if 'highly_variable' not in adata.var.columns:
            raise ValueError("Highly variable genes not identified. Run find_highly_variable_genes first.")

        logger.info(f"Subsetting to {adata.var['highly_variable'].sum()} highly variable genes")
        result = adata[:, adata.var['highly_variable']]

        return result

    def scale_data(self, adata: ad.AnnData, max_value: float = 10.0, inplace: bool = True) -> Optional[ad.AnnData]:
        """Scale data.

        Args:
            adata: AnnData object
            max_value: Maximum value after scaling
            inplace: Whether to modify in place

        Returns:
            AnnData object if not inplace
        """
        if not inplace:
            adata = adata.copy()

        if self.scale:
            logger.info("Scaling data")
            sc.pp.scale(adata, max_value=max_value)

        if not inplace:
            return adata

    def preprocess(
        self,
        adata: ad.AnnData,
        return_hvg_subset: bool = False,
        save_raw: bool = True
    ) -> ad.AnnData:
        """Run full preprocessing pipeline.

        Args:
            adata: AnnData object
            return_hvg_subset: Whether to return only HVG subset
            save_raw: Whether to save raw counts

        Returns:
            Preprocessed AnnData object
        """
        logger.info("Starting preprocessing pipeline")

        # Save raw counts if requested
        if save_raw:
            adata.raw = adata

        # Validate input
        if adata.n_obs == 0:
            raise ValueError("Cannot preprocess empty AnnData object (0 cells)")
        if adata.n_vars == 0:
            raise ValueError("Cannot preprocess AnnData object with 0 genes")

        # Calculate QC metrics
        self.calculate_qc_metrics(adata, inplace=True)

        # Filter cells and genes
        adata = self.filter_cells(adata, inplace=True)
        self.filter_genes(adata, inplace=True)

        # Normalize
        self.normalize_data(adata, inplace=True)

        # Find highly variable genes
        self.find_highly_variable_genes(adata, inplace=True)

        # Optionally subset to HVGs
        if return_hvg_subset:
            adata = self.subset_to_hvg(adata, inplace=False)

        # Scale if requested
        self.scale_data(adata, inplace=True)

        logger.info("Preprocessing complete")
        logger.info(f"Final data shape: {adata.n_obs} cells × {adata.n_vars} genes")

        return adata

    def save_hvg_list(self, output_path: str):
        """Save list of highly variable genes.

        Args:
            output_path: Path to save gene list
        """
        if self.hvg_genes is None:
            raise ValueError("No HVG genes identified yet")

        pd.DataFrame({'gene': self.hvg_genes}).to_csv(output_path, index=False)
        logger.info(f"Saved HVG list to {output_path}")

    def load_hvg_list(self, input_path: str) -> List[str]:
        """Load list of highly variable genes.

        Args:
            input_path: Path to gene list

        Returns:
            List of gene names
        """
        df = pd.read_csv(input_path)
        self.hvg_genes = df['gene'].tolist()
        logger.info(f"Loaded {len(self.hvg_genes)} genes from {input_path}")
        return self.hvg_genes
