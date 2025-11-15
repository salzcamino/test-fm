"""Visualization utilities."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import umap
from typing import Optional, Tuple
import torch


def plot_umap(
    embeddings: np.ndarray,
    labels: Optional[np.ndarray] = None,
    title: str = "UMAP Projection",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot UMAP projection of embeddings.

    Args:
        embeddings: Cell embeddings [n_cells, embedding_dim]
        labels: Cell labels for coloring
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    # Compute UMAP
    reducer = umap.UMAP(random_state=42)
    embedding_2d = reducer.fit_transform(embeddings)

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    if labels is not None:
        scatter = ax.scatter(
            embedding_2d[:, 0],
            embedding_2d[:, 1],
            c=labels,
            cmap='tab20',
            s=10,
            alpha=0.6
        )
        plt.colorbar(scatter, ax=ax)
    else:
        ax.scatter(
            embedding_2d[:, 0],
            embedding_2d[:, 1],
            s=10,
            alpha=0.6
        )

    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title(title)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_tsne(
    embeddings: np.ndarray,
    labels: Optional[np.ndarray] = None,
    title: str = "t-SNE Projection",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot t-SNE projection of embeddings.

    Args:
        embeddings: Cell embeddings
        labels: Cell labels
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    # Compute t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embedding_2d = tsne.fit_transform(embeddings)

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    if labels is not None:
        scatter = ax.scatter(
            embedding_2d[:, 0],
            embedding_2d[:, 1],
            c=labels,
            cmap='tab20',
            s=10,
            alpha=0.6
        )
        plt.colorbar(scatter, ax=ax)
    else:
        ax.scatter(
            embedding_2d[:, 0],
            embedding_2d[:, 1],
            s=10,
            alpha=0.6
        )

    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title(title)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_attention_heatmap(
    attention_weights: torch.Tensor,
    gene_names: Optional[list] = None,
    layer_idx: int = 0,
    head_idx: int = 0,
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot attention weights as heatmap.

    Args:
        attention_weights: Attention weights [num_layers, batch_size, num_heads, seq_len, seq_len]
        gene_names: List of gene names
        layer_idx: Which layer to plot
        head_idx: Which head to plot
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    # Extract attention for specific layer and head
    attn = attention_weights[layer_idx][0, head_idx].detach().cpu().numpy()

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        attn,
        cmap='viridis',
        xticklabels=gene_names if gene_names else False,
        yticklabels=gene_names if gene_names else False,
        ax=ax
    )

    ax.set_title(f'Attention Weights (Layer {layer_idx}, Head {head_idx})')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_gene_importance(
    importance_scores: np.ndarray,
    gene_names: list,
    top_k: int = 20,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot gene importance scores.

    Args:
        importance_scores: Gene importance scores [n_genes]
        gene_names: List of gene names
        top_k: Number of top genes to plot
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    # Get top k genes
    top_indices = np.argsort(importance_scores)[-top_k:][::-1]
    top_genes = [gene_names[i] for i in top_indices]
    top_scores = importance_scores[top_indices]

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    ax.barh(range(len(top_genes)), top_scores)
    ax.set_yticks(range(len(top_genes)))
    ax.set_yticklabels(top_genes)
    ax.set_xlabel('Importance Score')
    ax.set_title(f'Top {top_k} Important Genes')
    ax.invert_yaxis()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_training_curves(
    train_losses: list,
    val_losses: Optional[list] = None,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot training and validation loss curves.

    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(train_losses, label='Training Loss', linewidth=2)

    if val_losses is not None:
        ax.plot(val_losses, label='Validation Loss', linewidth=2)

    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Training Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig
