"""PyTorch dataset classes for scRNA-seq data."""

import torch
from torch.utils.data import Dataset
import numpy as np
import anndata as ad
from typing import Optional, List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class scRNADataset(Dataset):
    """PyTorch Dataset for scRNA-seq data."""

    def __init__(
        self,
        adata: ad.AnnData,
        gene_names: Optional[List[str]] = None,
        expression_bins: int = 50,
        mask_prob: float = 0.15,
        use_augmentation: bool = False,
        dropout_prob: float = 0.1,
        gaussian_noise: float = 0.01
    ):
        """Initialize dataset.

        Args:
            adata: AnnData object
            gene_names: List of gene names to use (if None, use all)
            expression_bins: Number of bins for expression discretization
            mask_prob: Probability of masking genes (for MLM)
            use_augmentation: Whether to use data augmentation
            dropout_prob: Probability of dropping out gene expression
            gaussian_noise: Standard deviation of Gaussian noise
        """
        # Validate input parameters
        if expression_bins <= 0:
            raise ValueError(f"expression_bins must be positive, got {expression_bins}")
        if not 0 <= mask_prob <= 1:
            raise ValueError(f"mask_prob must be between 0 and 1, got {mask_prob}")
        if not 0 <= dropout_prob <= 1:
            raise ValueError(f"dropout_prob must be between 0 and 1, got {dropout_prob}")
        if gaussian_noise < 0:
            raise ValueError(f"gaussian_noise must be non-negative, got {gaussian_noise}")

        self.adata = adata
        self.expression_bins = expression_bins
        self.mask_prob = mask_prob
        self.use_augmentation = use_augmentation
        self.dropout_prob = dropout_prob
        self.gaussian_noise = gaussian_noise

        # Get gene names
        if gene_names is not None:
            # Subset to specified genes
            self.gene_names = gene_names
            gene_mask = adata.var_names.isin(gene_names)
            self.data = adata[:, gene_mask].X
            self.var_names = adata.var_names[gene_mask].tolist()
        else:
            self.gene_names = adata.var_names.tolist()
            self.var_names = self.gene_names
            self.data = adata.X

        # Convert to dense if sparse
        if hasattr(self.data, 'toarray'):
            self.data = self.data.toarray()

        self.n_cells = self.data.shape[0]
        self.n_genes = self.data.shape[1]

        # Create gene to index mapping
        self.gene2idx = {gene: idx for idx, gene in enumerate(self.var_names)}

        # Calculate expression statistics for binning
        self._calculate_expression_stats()

        logger.info(f"Initialized dataset with {self.n_cells} cells and {self.n_genes} genes")

    def _calculate_expression_stats(self):
        """Calculate expression statistics for discretization."""
        # Get min and max expression values
        self.expr_min = np.min(self.data)
        self.expr_max = np.max(self.data)

        # Create bins for discretization
        self.bins = np.linspace(self.expr_min, self.expr_max, self.expression_bins + 1)

        logger.info(f"Expression range: [{self.expr_min:.3f}, {self.expr_max:.3f}]")

    def discretize_expression(self, expression: np.ndarray) -> np.ndarray:
        """Discretize expression values into bins.

        Args:
            expression: Expression values

        Returns:
            Discretized expression (bin indices)
        """
        # Use np.digitize to assign bin indices
        # Bins are 1-indexed, so we add 2 for special tokens (0: mask, 1: pad)
        discretized = np.digitize(expression, self.bins) + 2
        # Clip to valid range
        discretized = np.clip(discretized, 2, self.expression_bins + 2)
        return discretized

    def augment_cell(self, expression: np.ndarray) -> np.ndarray:
        """Apply data augmentation to cell expression.

        Args:
            expression: Cell expression values

        Returns:
            Augmented expression
        """
        if not self.use_augmentation:
            return expression

        expression = expression.copy()

        # Random dropout
        if self.dropout_prob > 0:
            dropout_mask = np.random.random(expression.shape) > self.dropout_prob
            expression = expression * dropout_mask

        # Gaussian noise
        if self.gaussian_noise > 0:
            noise = np.random.normal(0, self.gaussian_noise, expression.shape)
            expression = expression + noise
            expression = np.clip(expression, self.expr_min, self.expr_max)

        return expression

    def create_mlm_sample(
        self,
        expression: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create masked language modeling sample.

        Args:
            expression: Cell expression values

        Returns:
            Tuple of (masked_expression, mask, labels)
        """
        # Create mask
        mask = np.random.random(expression.shape) < self.mask_prob

        # Create labels (only for masked positions)
        labels = self.discretize_expression(expression).copy()

        # Mask the expression
        masked_expression = expression.copy()
        masked_expression[mask] = 0  # Set masked positions to 0

        # Discretize masked expression
        masked_expression_discrete = self.discretize_expression(masked_expression)
        masked_expression_discrete[mask] = 0  # Set mask token (0) for masked positions

        return masked_expression_discrete, mask.astype(np.float32), labels

    def __len__(self) -> int:
        """Return dataset size."""
        return self.n_cells

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single cell sample.

        Args:
            idx: Cell index

        Returns:
            Dictionary containing cell data
        """
        # Get cell expression
        expression = self.data[idx].astype(np.float32)

        # Augment if enabled
        expression = self.augment_cell(expression)

        # Create MLM sample
        input_ids, mask, labels = self.create_mlm_sample(expression)

        # Convert to tensors
        sample = {
            'input_ids': torch.LongTensor(input_ids),
            'attention_mask': torch.ones(self.n_genes, dtype=torch.long),
            'mask': torch.FloatTensor(mask),
            'labels': torch.LongTensor(labels),
            'raw_expression': torch.FloatTensor(self.data[idx])
        }

        return sample


class scRNADatasetContinuous(Dataset):
    """PyTorch Dataset for scRNA-seq data with continuous expression values."""

    def __init__(
        self,
        adata: ad.AnnData,
        gene_names: Optional[List[str]] = None,
        mask_prob: float = 0.15,
        use_augmentation: bool = False,
        dropout_prob: float = 0.1,
        gaussian_noise: float = 0.01
    ):
        """Initialize dataset with continuous expression values.

        Args:
            adata: AnnData object
            gene_names: List of gene names to use
            mask_prob: Probability of masking genes
            use_augmentation: Whether to use data augmentation
            dropout_prob: Probability of dropout
            gaussian_noise: Gaussian noise std
        """
        self.adata = adata
        self.mask_prob = mask_prob
        self.use_augmentation = use_augmentation
        self.dropout_prob = dropout_prob
        self.gaussian_noise = gaussian_noise

        # Get gene data
        if gene_names is not None:
            self.gene_names = gene_names
            gene_mask = adata.var_names.isin(gene_names)
            self.data = adata[:, gene_mask].X
            self.var_names = adata.var_names[gene_mask].tolist()
        else:
            self.gene_names = adata.var_names.tolist()
            self.var_names = self.gene_names
            self.data = adata.X

        if hasattr(self.data, 'toarray'):
            self.data = self.data.toarray()

        self.n_cells = self.data.shape[0]
        self.n_genes = self.data.shape[1]

        logger.info(f"Initialized continuous dataset with {self.n_cells} cells and {self.n_genes} genes")

    def augment_cell(self, expression: np.ndarray) -> np.ndarray:
        """Apply data augmentation."""
        if not self.use_augmentation:
            return expression

        expression = expression.copy()

        if self.dropout_prob > 0:
            dropout_mask = np.random.random(expression.shape) > self.dropout_prob
            expression = expression * dropout_mask

        if self.gaussian_noise > 0:
            noise = np.random.normal(0, self.gaussian_noise, expression.shape)
            expression = expression + noise
            expression = np.maximum(expression, 0)  # Ensure non-negative

        return expression

    def __len__(self) -> int:
        return self.n_cells

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single cell sample with continuous values."""
        expression = self.data[idx].astype(np.float32)
        expression = self.augment_cell(expression)

        # Create mask for MLM
        mask = np.random.random(expression.shape) < self.mask_prob

        # Create masked input (set masked positions to 0)
        masked_expression = expression.copy()
        masked_expression[mask] = 0.0

        sample = {
            'input_values': torch.FloatTensor(masked_expression),
            'attention_mask': torch.ones(self.n_genes, dtype=torch.long),
            'mask': torch.FloatTensor(mask.astype(np.float32)),
            'labels': torch.FloatTensor(expression),
            'raw_expression': torch.FloatTensor(self.data[idx])
        }

        return sample


def create_dataloaders(
    adata: ad.AnnData,
    batch_size: int = 32,
    train_split: float = 0.8,
    val_split: float = 0.1,
    num_workers: int = 4,
    **dataset_kwargs
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train, validation, and test dataloaders.

    Args:
        adata: AnnData object
        batch_size: Batch size
        train_split: Fraction for training
        val_split: Fraction for validation
        num_workers: Number of workers for data loading
        **dataset_kwargs: Additional arguments for dataset

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Split data
    n_cells = adata.n_obs
    indices = np.random.permutation(n_cells)

    train_end = int(train_split * n_cells)
    val_end = int((train_split + val_split) * n_cells)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    # Create datasets
    train_dataset = scRNADataset(adata[train_indices], **dataset_kwargs)
    val_dataset = scRNADataset(adata[val_indices], **dataset_kwargs)
    test_dataset = scRNADataset(adata[test_indices], **dataset_kwargs)

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    logger.info(f"Created dataloaders: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

    return train_loader, val_loader, test_loader
