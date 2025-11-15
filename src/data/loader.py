"""Data loading utilities for scRNA-seq data."""

import anndata as ad
import scanpy as sc
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, Optional, List
import logging

logger = logging.getLogger(__name__)


class scRNADataLoader:
    """Data loader for single-cell RNA-seq data."""

    def __init__(self, data_path: Union[str, Path]):
        """Initialize data loader.

        Args:
            data_path: Path to the data file
        """
        self.data_path = Path(data_path)
        self.adata = None

    def load_h5ad(self, backed: bool = False) -> ad.AnnData:
        """Load data from h5ad file.

        Args:
            backed: Whether to load data in backed mode

        Returns:
            AnnData object
        """
        logger.info(f"Loading h5ad file from {self.data_path}")

        if backed:
            self.adata = ad.read_h5ad(self.data_path, backed='r')
        else:
            self.adata = ad.read_h5ad(self.data_path)

        logger.info(f"Loaded {self.adata.n_obs} cells and {self.adata.n_vars} genes")
        return self.adata

    def load_10x_mtx(self, matrix_path: str, features_path: str, barcodes_path: str) -> ad.AnnData:
        """Load 10X format data (matrix, features, barcodes).

        Args:
            matrix_path: Path to matrix.mtx file
            features_path: Path to features.tsv file
            barcodes_path: Path to barcodes.tsv file

        Returns:
            AnnData object
        """
        logger.info(f"Loading 10X data from {matrix_path}")

        self.adata = sc.read_10x_mtx(
            path=Path(matrix_path).parent,
            var_names='gene_symbols',
            cache=False
        )

        logger.info(f"Loaded {self.adata.n_obs} cells and {self.adata.n_vars} genes")
        return self.adata

    def load_csv(
        self,
        transpose: bool = True,
        first_column_names: bool = True
    ) -> ad.AnnData:
        """Load data from CSV file.

        Args:
            transpose: Whether to transpose (genes as rows)
            first_column_names: Whether first column contains cell/gene names

        Returns:
            AnnData object
        """
        logger.info(f"Loading CSV file from {self.data_path}")

        df = pd.read_csv(self.data_path, index_col=0 if first_column_names else None)

        if transpose:
            df = df.T

        self.adata = ad.AnnData(X=df.values, obs=pd.DataFrame(index=df.index))
        self.adata.var_names = df.columns

        logger.info(f"Loaded {self.adata.n_obs} cells and {self.adata.n_vars} genes")
        return self.adata

    def load_loom(self) -> ad.AnnData:
        """Load data from loom file.

        Returns:
            AnnData object
        """
        logger.info(f"Loading loom file from {self.data_path}")

        self.adata = ad.read_loom(self.data_path)

        logger.info(f"Loaded {self.adata.n_obs} cells and {self.adata.n_vars} genes")
        return self.adata

    def auto_load(self) -> ad.AnnData:
        """Automatically detect and load data based on file extension.

        Returns:
            AnnData object
        """
        suffix = self.data_path.suffix.lower()

        if suffix == '.h5ad':
            return self.load_h5ad()
        elif suffix == '.loom':
            return self.load_loom()
        elif suffix in ['.csv', '.tsv', '.txt']:
            return self.load_csv()
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    def get_data(self) -> ad.AnnData:
        """Get the loaded AnnData object.

        Returns:
            AnnData object
        """
        if self.adata is None:
            raise ValueError("No data loaded. Call a load method first.")
        return self.adata


def load_multiple_datasets(
    data_paths: List[Union[str, Path]],
    merge: bool = True,
    batch_key: str = "batch"
) -> Union[ad.AnnData, List[ad.AnnData]]:
    """Load multiple datasets.

    Args:
        data_paths: List of paths to data files
        merge: Whether to merge datasets
        batch_key: Key for batch information when merging

    Returns:
        Single merged AnnData or list of AnnData objects
    """
    adatas = []

    for i, path in enumerate(data_paths):
        loader = scRNADataLoader(path)
        adata = loader.auto_load()
        adata.obs[batch_key] = f"batch_{i}"
        adatas.append(adata)

    if merge:
        logger.info(f"Merging {len(adatas)} datasets")
        merged_adata = ad.concat(adatas, label=batch_key, keys=[f"batch_{i}" for i in range(len(adatas))])
        return merged_adata

    return adatas


def download_example_dataset(dataset_name: str = "pbmc3k", save_dir: str = "data/raw") -> ad.AnnData:
    """Download example dataset from scanpy.

    Args:
        dataset_name: Name of the dataset (pbmc3k, pbmc68k, etc.)
        save_dir: Directory to save the dataset

    Returns:
        AnnData object
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading {dataset_name} dataset")

    if dataset_name == "pbmc3k":
        adata = sc.datasets.pbmc3k()
    elif dataset_name == "pbmc68k":
        adata = sc.datasets.pbmc68k_reduced()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Save dataset
    output_path = save_path / f"{dataset_name}.h5ad"
    adata.write(output_path)
    logger.info(f"Saved dataset to {output_path}")

    return adata
