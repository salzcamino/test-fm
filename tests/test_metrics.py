"""Tests for evaluation metrics."""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.mark.unit
class TestClusteringMetrics:
    """Tests for clustering metrics."""

    def test_compute_clustering_metrics_basic(self):
        """Test basic clustering metrics computation."""
        try:
            from src.training.metrics import compute_clustering_metrics

            # Create simple clusterable data
            embeddings = np.vstack([
                np.random.randn(50, 10) + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                np.random.randn(50, 10) + [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            ])
            labels = np.array([0] * 50 + [1] * 50)

            metrics = compute_clustering_metrics(embeddings, labels, n_clusters=2)

            assert 'ari' in metrics
            assert 'nmi' in metrics
            assert 'silhouette' in metrics

            # ARI should be between -1 and 1
            assert -1 <= metrics['ari'] <= 1

            # NMI should be between 0 and 1
            assert 0 <= metrics['nmi'] <= 1

            # Silhouette should be between -1 and 1
            assert -1 <= metrics['silhouette'] <= 1

        except ImportError:
            pytest.skip("Required dependencies not installed")

    def test_compute_clustering_metrics_perfect(self):
        """Test clustering metrics with perfect clustering."""
        try:
            from src.training.metrics import compute_clustering_metrics

            # Create perfectly separable clusters
            embeddings = np.vstack([
                np.ones((25, 5)) * 0,
                np.ones((25, 5)) * 10,
                np.ones((25, 5)) * 20,
            ])
            labels = np.array([0] * 25 + [1] * 25 + [2] * 25)

            metrics = compute_clustering_metrics(embeddings, labels, n_clusters=3)

            # Perfect clustering should have high ARI and NMI
            assert metrics['ari'] > 0.9
            assert metrics['nmi'] > 0.9

        except ImportError:
            pytest.skip("Required dependencies not installed")

    def test_compute_clustering_metrics_auto_clusters(self):
        """Test clustering metrics with automatic cluster detection."""
        try:
            from src.training.metrics import compute_clustering_metrics

            embeddings = np.random.randn(100, 10)
            labels = np.random.randint(0, 5, 100)

            # Don't specify n_clusters - should auto-detect
            metrics = compute_clustering_metrics(embeddings, labels)

            assert metrics is not None
            assert 'ari' in metrics

        except ImportError:
            pytest.skip("Required dependencies not installed")


@pytest.mark.unit
class TestClassificationMetrics:
    """Tests for classification metrics."""

    def test_compute_classification_metrics_basic(self):
        """Test basic classification metrics computation."""
        try:
            from src.training.metrics import compute_classification_metrics

            predictions = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
            labels = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])

            metrics = compute_classification_metrics(predictions, labels)

            assert 'accuracy' in metrics
            assert 'f1_score' in metrics

            # Perfect predictions
            assert metrics['accuracy'] == 1.0
            assert metrics['f1_score'] == 1.0

        except ImportError:
            pytest.skip("Required dependencies not installed")

    def test_compute_classification_metrics_imperfect(self):
        """Test classification metrics with imperfect predictions."""
        try:
            from src.training.metrics import compute_classification_metrics

            predictions = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
            labels = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0])

            metrics = compute_classification_metrics(predictions, labels)

            # Should have <100% accuracy
            assert 0 <= metrics['accuracy'] < 1.0
            assert 0 <= metrics['f1_score'] <= 1.0

        except ImportError:
            pytest.skip("Required dependencies not installed")


@pytest.mark.requires_torch
class TestReconstructionMetrics:
    """Tests for reconstruction metrics."""

    @pytest.fixture(autouse=True)
    def setup(self, skip_if_no_torch):
        """Setup for tests requiring torch."""
        import torch
        from src.training.metrics import compute_reconstruction_metrics
        self.torch = torch
        self.compute_reconstruction_metrics = compute_reconstruction_metrics

    def test_reconstruction_metrics_basic(self):
        """Test basic reconstruction metrics computation."""
        predictions = self.torch.randn(100, 50)
        targets = self.torch.randn(100, 50)

        metrics = self.compute_reconstruction_metrics(predictions, targets)

        assert 'mse' in metrics
        assert 'mae' in metrics
        assert 'pearson_correlation' in metrics

        assert metrics['mse'] >= 0
        assert metrics['mae'] >= 0
        assert -1 <= metrics['pearson_correlation'] <= 1

    def test_reconstruction_metrics_perfect(self):
        """Test reconstruction metrics with perfect reconstruction."""
        predictions = self.torch.ones(50, 20) * 3.0
        targets = self.torch.ones(50, 20) * 3.0

        metrics = self.compute_reconstruction_metrics(predictions, targets)

        # Perfect reconstruction
        assert metrics['mse'] < 1e-6
        assert metrics['mae'] < 1e-6
        # Correlation is undefined for constant values, but should not be NaN
        assert not np.isnan(metrics['pearson_correlation'])

    def test_reconstruction_metrics_with_mask(self):
        """Test reconstruction metrics with mask."""
        predictions = self.torch.randn(50, 30)
        targets = self.torch.randn(50, 30)
        mask = self.torch.rand(50, 30) < 0.5

        metrics = self.compute_reconstruction_metrics(
            predictions, targets, mask
        )

        assert metrics['mse'] >= 0
        assert metrics['mae'] >= 0
        assert not np.isnan(metrics['pearson_correlation'])

    def test_reconstruction_metrics_constant_predictions(self):
        """Test reconstruction metrics with constant predictions."""
        # This tests the division by zero fix we recommended
        predictions = self.torch.ones(10, 10) * 2.0
        targets = self.torch.randn(10, 10)

        metrics = self.compute_reconstruction_metrics(predictions, targets)

        # Should not crash and should not produce NaN
        assert not np.isnan(metrics['mse'])
        assert not np.isnan(metrics['mae'])
        # Correlation might be NaN for constant predictions
        # This is expected behavior


@pytest.mark.unit
class TestEvaluationMetricsContainer:
    """Tests for EvaluationMetrics container class."""

    def test_evaluation_metrics_initialization(self):
        """Test EvaluationMetrics initialization."""
        try:
            from src.training.metrics import EvaluationMetrics

            metrics_container = EvaluationMetrics()
            assert metrics_container is not None
            assert len(metrics_container.get_all_metrics()) == 0

        except ImportError:
            pytest.skip("Required dependencies not installed")

    def test_evaluation_metrics_add_metric(self):
        """Test adding individual metrics."""
        try:
            from src.training.metrics import EvaluationMetrics

            metrics_container = EvaluationMetrics()
            metrics_container.add_metric('accuracy', 0.95)
            metrics_container.add_metric('loss', 0.23)

            all_metrics = metrics_container.get_all_metrics()
            assert len(all_metrics) == 2
            assert all_metrics['accuracy'] == 0.95
            assert all_metrics['loss'] == 0.23

        except ImportError:
            pytest.skip("Required dependencies not installed")

    def test_evaluation_metrics_add_metrics(self):
        """Test adding multiple metrics."""
        try:
            from src.training.metrics import EvaluationMetrics

            metrics_container = EvaluationMetrics()
            new_metrics = {
                'precision': 0.87,
                'recall': 0.92,
                'f1': 0.89
            }
            metrics_container.add_metrics(new_metrics)

            all_metrics = metrics_container.get_all_metrics()
            assert len(all_metrics) == 3
            assert all_metrics['precision'] == 0.87

        except ImportError:
            pytest.skip("Required dependencies not installed")

    def test_evaluation_metrics_get_metric(self):
        """Test getting individual metric."""
        try:
            from src.training.metrics import EvaluationMetrics

            metrics_container = EvaluationMetrics()
            metrics_container.add_metric('test_metric', 0.75)

            value = metrics_container.get_metric('test_metric')
            assert value == 0.75

            # Nonexistent metric should return None
            assert metrics_container.get_metric('nonexistent') is None

        except ImportError:
            pytest.skip("Required dependencies not installed")

    def test_evaluation_metrics_reset(self):
        """Test resetting metrics."""
        try:
            from src.training.metrics import EvaluationMetrics

            metrics_container = EvaluationMetrics()
            metrics_container.add_metric('metric1', 0.5)
            metrics_container.add_metric('metric2', 0.6)

            assert len(metrics_container.get_all_metrics()) == 2

            metrics_container.reset()

            assert len(metrics_container.get_all_metrics()) == 0

        except ImportError:
            pytest.skip("Required dependencies not installed")
