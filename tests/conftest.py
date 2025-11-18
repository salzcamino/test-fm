"""Pytest configuration and fixtures for testing."""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_expression_data():
    """Create sample expression data for testing."""
    n_cells = 100
    n_genes = 50
    # Random expression data (log-normalized)
    data = np.random.rand(n_cells, n_genes) * 10
    return data


@pytest.fixture
def small_expression_data():
    """Create small expression data for quick tests."""
    n_cells = 10
    n_genes = 20
    data = np.random.rand(n_cells, n_genes) * 5
    return data


@pytest.fixture
def mock_adata(sample_expression_data):
    """Create mock AnnData object."""
    try:
        import anndata as ad
        import pandas as pd

        n_cells, n_genes = sample_expression_data.shape

        obs = pd.DataFrame(index=[f'cell_{i}' for i in range(n_cells)])
        var = pd.DataFrame(index=[f'gene_{i}' for i in range(n_genes)])

        adata = ad.AnnData(X=sample_expression_data, obs=obs, var=var)
        return adata
    except ImportError:
        pytest.skip("anndata not installed")


@pytest.fixture
def small_mock_adata(small_expression_data):
    """Create small mock AnnData object."""
    try:
        import anndata as ad
        import pandas as pd

        n_cells, n_genes = small_expression_data.shape

        obs = pd.DataFrame(index=[f'cell_{i}' for i in range(n_cells)])
        var = pd.DataFrame(index=[f'gene_{i}' for i in range(n_genes)])

        adata = ad.AnnData(X=small_expression_data, obs=obs, var=var)
        return adata
    except ImportError:
        pytest.skip("anndata not installed")


@pytest.fixture
def sample_config():
    """Create sample configuration dictionary."""
    return {
        'model': {
            'n_genes': 100,
            'gene_embedding_dim': 64,
            'expression_bins': 50,
            'hidden_dim': 128,
            'num_layers': 2,
            'num_heads': 4,
            'ff_dim': 512,
            'dropout': 0.1,
            'activation': 'gelu',
            'use_positional_encoding': True,
            'use_mlm_head': True,
            'use_contrastive_head': True,
            'projection_dim': 64,
            'expression_encoding': 'binned'
        },
        'training': {
            'batch_size': 8,
            'num_epochs': 2,
            'learning_rate': 1e-4,
            'weight_decay': 0.01,
            'mlm_weight': 1.0,
            'contrastive_weight': 0.5,
            'contrastive_temperature': 0.07
        },
        'data': {
            'train_split': 0.8,
            'val_split': 0.1,
            'preprocessing': {
                'min_genes_per_cell': 10,
                'min_cells_per_gene': 3,
                'max_genes_per_cell': 5000,
                'max_pct_mito': 20,
                'target_sum': 1e4,
                'normalize_total': True,
                'log1p': True,
                'scale': False
            }
        }
    }


@pytest.fixture
def torch_available():
    """Check if PyTorch is available."""
    try:
        import torch
        return True
    except ImportError:
        return False


@pytest.fixture
def skip_if_no_torch():
    """Skip test if PyTorch not available."""
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not installed")
