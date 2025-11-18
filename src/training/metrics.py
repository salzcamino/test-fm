"""Evaluation metrics for scRNA-seq foundation model."""

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    accuracy_score,
    f1_score
)
from sklearn.cluster import KMeans
from typing import Optional, Dict, Any
import torch


def compute_clustering_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_clusters: Optional[int] = None
) -> Dict[str, float]:
    """Compute clustering metrics.

    Args:
        embeddings: Cell embeddings [n_cells, embedding_dim]
        labels: True cell type labels
        n_clusters: Number of clusters (if None, use number of unique labels)

    Returns:
        Dictionary of clustering metrics
    """
    if n_clusters is None:
        n_clusters = len(np.unique(labels))

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    predicted_labels = kmeans.fit_predict(embeddings)

    # Calculate metrics
    metrics = {
        'ari': adjusted_rand_score(labels, predicted_labels),
        'nmi': normalized_mutual_info_score(labels, predicted_labels),
        'silhouette': silhouette_score(embeddings, labels) if len(np.unique(labels)) > 1 else 0.0
    }

    return metrics


def compute_classification_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    average: str = 'weighted'
) -> Dict[str, float]:
    """Compute classification metrics.

    Args:
        predictions: Predicted labels
        labels: True labels
        average: Averaging method for F1 score

    Returns:
        Dictionary of classification metrics
    """
    metrics = {
        'accuracy': accuracy_score(labels, predictions),
        'f1_score': f1_score(labels, predictions, average=average)
    }

    return metrics


def compute_reconstruction_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """Compute reconstruction metrics.

    Args:
        predictions: Predicted expression values
        targets: Target expression values
        mask: Mask for computing metrics on specific positions

    Returns:
        Dictionary of reconstruction metrics
    """
    if mask is not None:
        predictions = predictions[mask.bool()]
        targets = targets[mask.bool()]

    # MSE
    mse = torch.mean((predictions - targets) ** 2).item()

    # MAE
    mae = torch.mean(torch.abs(predictions - targets)).item()

    # Pearson correlation
    pred_mean = torch.mean(predictions)
    target_mean = torch.mean(targets)

    pred_centered = predictions - pred_mean
    target_centered = targets - target_mean

    # Add epsilon to avoid division by zero
    eps = 1e-8
    correlation = (
        torch.sum(pred_centered * target_centered) /
        (torch.sqrt(torch.sum(pred_centered ** 2)) * torch.sqrt(torch.sum(target_centered ** 2)) + eps)
    ).item()

    metrics = {
        'mse': mse,
        'mae': mae,
        'pearson_correlation': correlation
    }

    return metrics


class EvaluationMetrics:
    """Container for evaluation metrics."""

    def __init__(self):
        """Initialize metrics container."""
        self.metrics = {}

    def add_metric(self, name: str, value: float):
        """Add a metric.

        Args:
            name: Metric name
            value: Metric value
        """
        self.metrics[name] = value

    def add_metrics(self, metrics: Dict[str, float]):
        """Add multiple metrics.

        Args:
            metrics: Dictionary of metrics
        """
        self.metrics.update(metrics)

    def get_metric(self, name: str) -> Optional[float]:
        """Get a metric value.

        Args:
            name: Metric name

        Returns:
            Metric value or None if not found
        """
        return self.metrics.get(name)

    def get_all_metrics(self) -> Dict[str, float]:
        """Get all metrics.

        Returns:
            Dictionary of all metrics
        """
        return self.metrics.copy()

    def reset(self):
        """Reset all metrics."""
        self.metrics = {}

    def __repr__(self) -> str:
        """String representation."""
        lines = ["Evaluation Metrics:"]
        for name, value in self.metrics.items():
            lines.append(f"  {name}: {value:.4f}")
        return "\n".join(lines)
