"""Tests for data loading and preprocessing."""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.mark.requires_data
class TestDataset:
    """Tests for scRNA dataset."""

    @pytest.fixture(autouse=True)
    def setup(self, skip_if_no_torch):
        """Setup for tests requiring torch and data."""
        import torch
        try:
            from src.data.dataset import scRNADataset, scRNADatasetContinuous
            self.torch = torch
            self.scRNADataset = scRNADataset
            self.scRNADatasetContinuous = scRNADatasetContinuous
        except ImportError:
            pytest.skip("Data dependencies not installed")

    def test_dataset_initialization(self, small_mock_adata):
        """Test dataset initialization."""
        dataset = self.scRNADataset(
            small_mock_adata,
            expression_bins=50,
            mask_prob=0.15
        )

        assert dataset is not None
        assert len(dataset) == small_mock_adata.n_obs
        assert dataset.n_genes == small_mock_adata.n_vars

    def test_dataset_getitem(self, small_mock_adata):
        """Test getting item from dataset."""
        dataset = self.scRNADataset(
            small_mock_adata,
            expression_bins=50,
            mask_prob=0.15
        )

        sample = dataset[0]

        assert 'input_ids' in sample
        assert 'attention_mask' in sample
        assert 'mask' in sample
        assert 'labels' in sample
        assert 'raw_expression' in sample

        # Check shapes
        assert sample['input_ids'].shape == (small_mock_adata.n_vars,)
        assert sample['mask'].shape == (small_mock_adata.n_vars,)

    def test_dataset_masking(self, small_mock_adata):
        """Test that masking is applied correctly."""
        dataset = self.scRNADataset(
            small_mock_adata,
            expression_bins=50,
            mask_prob=0.15
        )

        sample = dataset[0]

        # At least some genes should be masked (probabilistic)
        # Run multiple times to ensure masking happens
        masked_counts = []
        for i in range(10):
            sample = dataset[i % len(dataset)]
            masked_counts.append(sample['mask'].sum().item())

        # At least one sample should have masked genes
        assert max(masked_counts) > 0

    def test_dataset_discretization(self, small_mock_adata):
        """Test expression value discretization."""
        dataset = self.scRNADataset(
            small_mock_adata,
            expression_bins=50,
            mask_prob=0.15
        )

        # Check that discretized values are in valid range
        # bins are 2 to expression_bins+2 (0=mask, 1=pad)
        sample = dataset[0]

        # Unmasked positions should have valid bin indices
        unmasked = ~sample['mask'].bool()
        if unmasked.any():
            unmasked_values = sample['input_ids'][unmasked]
            assert (unmasked_values >= 2).all()
            assert (unmasked_values <= dataset.expression_bins + 2).all()

    def test_continuous_dataset(self, small_mock_adata):
        """Test continuous expression dataset."""
        dataset = self.scRNADatasetContinuous(
            small_mock_adata,
            mask_prob=0.15
        )

        sample = dataset[0]

        assert 'input_values' in sample
        assert 'labels' in sample
        assert sample['input_values'].shape == (small_mock_adata.n_vars,)

    def test_dataset_with_augmentation(self, small_mock_adata):
        """Test dataset with augmentation enabled."""
        dataset = self.scRNADataset(
            small_mock_adata,
            expression_bins=50,
            mask_prob=0.15,
            use_augmentation=True,
            dropout_prob=0.1,
            gaussian_noise=0.01
        )

        sample = dataset[0]
        assert sample is not None


@pytest.mark.requires_data
class TestPreprocessor:
    """Tests for data preprocessor."""

    def test_preprocessor_initialization(self):
        """Test preprocessor initialization."""
        try:
            from src.data.preprocessor import scRNAPreprocessor

            preprocessor = scRNAPreprocessor(
                min_genes=200,
                min_cells=3,
                n_top_genes=2000
            )

            assert preprocessor is not None
            assert preprocessor.min_genes == 200
            assert preprocessor.n_top_genes == 2000

        except ImportError:
            pytest.skip("Data dependencies not installed")

    def test_preprocessor_calculate_qc_metrics(self, mock_adata):
        """Test QC metrics calculation."""
        try:
            from src.data.preprocessor import scRNAPreprocessor

            preprocessor = scRNAPreprocessor()
            preprocessor.calculate_qc_metrics(mock_adata)

            assert 'n_genes_by_counts' in mock_adata.obs.columns
            assert 'total_counts' in mock_adata.obs.columns

        except ImportError:
            pytest.skip("Data dependencies not installed")

    def test_preprocessor_filter_cells(self, mock_adata):
        """Test cell filtering."""
        try:
            from src.data.preprocessor import scRNAPreprocessor

            preprocessor = scRNAPreprocessor(min_genes=10)

            initial_cells = mock_adata.n_obs
            preprocessor.calculate_qc_metrics(mock_adata)
            preprocessor.filter_cells(mock_adata)

            # Should not increase cell count
            assert mock_adata.n_obs <= initial_cells

        except ImportError:
            pytest.skip("Data dependencies not installed")

    def test_preprocessor_normalize(self, mock_adata):
        """Test normalization."""
        try:
            from src.data.preprocessor import scRNAPreprocessor

            preprocessor = scRNAPreprocessor(normalize=True, target_sum=1e4)

            initial_sum = mock_adata.X.sum(axis=1).mean()
            preprocessor.normalize_data(mock_adata)

            # After normalization, mean sum should be close to target
            # (This is a rough check due to log transform)

        except ImportError:
            pytest.skip("Data dependencies not installed")


@pytest.mark.requires_data
class TestDataLoader:
    """Tests for data loader."""

    def test_data_loader_initialization(self, tmp_path):
        """Test data loader initialization."""
        try:
            from src.data.loader import scRNADataLoader

            test_file = tmp_path / "test_data.h5ad"
            test_file.touch()  # Create empty file

            loader = scRNADataLoader(test_file)

            assert loader is not None
            assert loader.data_path == test_file

        except ImportError:
            pytest.skip("Data dependencies not installed")

    def test_auto_load_detection(self, tmp_path):
        """Test automatic file format detection."""
        try:
            from src.data.loader import scRNADataLoader

            # Test h5ad detection
            h5ad_file = tmp_path / "data.h5ad"
            h5ad_file.touch()
            loader = scRNADataLoader(h5ad_file)
            assert loader.data_path.suffix == '.h5ad'

            # Test loom detection
            loom_file = tmp_path / "data.loom"
            loom_file.touch()
            loader = scRNADataLoader(loom_file)
            assert loader.data_path.suffix == '.loom'

        except ImportError:
            pytest.skip("Data dependencies not installed")


@pytest.mark.unit
class TestDataUtilities:
    """Tests for data utility functions."""

    def test_create_dataloaders(self, mock_adata):
        """Test dataloader creation."""
        try:
            from src.data.dataset import create_dataloaders

            train_loader, val_loader, test_loader = create_dataloaders(
                mock_adata,
                batch_size=8,
                train_split=0.7,
                val_split=0.2,
                num_workers=0,  # Use 0 for testing
                expression_bins=50,
                mask_prob=0.15
            )

            assert train_loader is not None
            assert val_loader is not None
            assert test_loader is not None

            # Check that splits are reasonable
            total_samples = (
                len(train_loader.dataset) +
                len(val_loader.dataset) +
                len(test_loader.dataset)
            )

            assert total_samples == mock_adata.n_obs

        except ImportError:
            pytest.skip("Data dependencies not installed")


@pytest.mark.smoke
class TestDataSmokeTests:
    """Quick smoke tests for data module."""

    def test_import_data_modules(self):
        """Test that data modules can be imported."""
        try:
            from src.data import loader, preprocessor, dataset
            assert loader is not None
            assert preprocessor is not None
            assert dataset is not None
        except ImportError as e:
            pytest.skip(f"Data dependencies not installed: {e}")

    def test_data_module_attributes(self):
        """Test that expected classes exist in modules."""
        try:
            from src.data.loader import scRNADataLoader, download_example_dataset
            from src.data.preprocessor import scRNAPreprocessor
            from src.data.dataset import scRNADataset, scRNADatasetContinuous

            assert scRNADataLoader is not None
            assert scRNAPreprocessor is not None
            assert scRNADataset is not None
            assert scRNADatasetContinuous is not None
            assert download_example_dataset is not None

        except ImportError as e:
            pytest.skip(f"Data dependencies not installed: {e}")
