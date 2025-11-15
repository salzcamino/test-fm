"""Gene encoder and expression embedding layers."""

import torch
import torch.nn as nn
import math
from typing import Optional


class GeneEmbedding(nn.Module):
    """Gene embedding layer that learns representations for each gene."""

    def __init__(
        self,
        n_genes: int,
        embedding_dim: int,
        padding_idx: int = 1
    ):
        """Initialize gene embedding.

        Args:
            n_genes: Number of genes in vocabulary
            embedding_dim: Dimension of gene embeddings
            padding_idx: Index for padding token
        """
        super().__init__()
        self.n_genes = n_genes
        self.embedding_dim = embedding_dim

        # Gene embeddings (learnable)
        self.gene_embeddings = nn.Embedding(
            n_genes,
            embedding_dim,
            padding_idx=padding_idx
        )

        # Initialize embeddings
        nn.init.normal_(self.gene_embeddings.weight, mean=0, std=0.02)

    def forward(self, gene_indices: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            gene_indices: Tensor of gene indices [batch_size, n_genes]

        Returns:
            Gene embeddings [batch_size, n_genes, embedding_dim]
        """
        return self.gene_embeddings(gene_indices)


class ExpressionEmbedding(nn.Module):
    """Expression value embedding layer."""

    def __init__(
        self,
        expression_bins: int,
        embedding_dim: int,
        padding_idx: int = 1
    ):
        """Initialize expression embedding for discrete expression values.

        Args:
            expression_bins: Number of expression bins
            embedding_dim: Dimension of embeddings
            padding_idx: Index for padding token
        """
        super().__init__()
        # Add special tokens: 0 = mask, 1 = pad
        vocab_size = expression_bins + 3  # bins + mask + pad

        self.expression_embeddings = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=padding_idx
        )

        nn.init.normal_(self.expression_embeddings.weight, mean=0, std=0.02)

    def forward(self, expression_values: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            expression_values: Discretized expression values [batch_size, n_genes]

        Returns:
            Expression embeddings [batch_size, n_genes, embedding_dim]
        """
        return self.expression_embeddings(expression_values)


class ContinuousExpressionEmbedding(nn.Module):
    """Continuous expression value embedding using MLP."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: Optional[int] = None
    ):
        """Initialize continuous expression embedding.

        Args:
            embedding_dim: Output dimension
            hidden_dim: Hidden layer dimension (defaults to embedding_dim * 2)
        """
        super().__init__()
        hidden_dim = hidden_dim or embedding_dim * 2

        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(self, expression_values: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            expression_values: Continuous expression values [batch_size, n_genes]

        Returns:
            Expression embeddings [batch_size, n_genes, embedding_dim]
        """
        # Add channel dimension
        x = expression_values.unsqueeze(-1)  # [batch_size, n_genes, 1]
        return self.mlp(x)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(
        self,
        embedding_dim: int,
        max_position_embeddings: int = 2048,
        dropout: float = 0.1
    ):
        """Initialize positional encoding.

        Args:
            embedding_dim: Dimension of embeddings
            max_position_embeddings: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        position = torch.arange(max_position_embeddings).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim)
        )

        pe = torch.zeros(1, max_position_embeddings, embedding_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: Input tensor [batch_size, seq_len, embedding_dim]

        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """Learnable positional encoding."""

    def __init__(
        self,
        embedding_dim: int,
        max_position_embeddings: int = 2048,
        dropout: float = 0.1
    ):
        """Initialize learnable positional encoding.

        Args:
            embedding_dim: Dimension of embeddings
            max_position_embeddings: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.position_embeddings = nn.Embedding(
            max_position_embeddings,
            embedding_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: Input tensor [batch_size, seq_len, embedding_dim]

        Returns:
            Tensor with positional encoding added
        """
        batch_size, seq_len, _ = x.size()
        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        position_embeddings = self.position_embeddings(position_ids)
        x = x + position_embeddings
        return self.dropout(x)


class GeneEncoder(nn.Module):
    """Complete gene encoder combining gene and expression embeddings."""

    def __init__(
        self,
        n_genes: int,
        gene_embedding_dim: int,
        expression_bins: int,
        hidden_dim: int,
        use_positional_encoding: bool = True,
        positional_encoding_type: str = "learnable",
        dropout: float = 0.1,
        use_continuous_expression: bool = False
    ):
        """Initialize gene encoder.

        Args:
            n_genes: Number of genes
            gene_embedding_dim: Dimension of gene embeddings
            expression_bins: Number of expression bins (if discrete)
            hidden_dim: Hidden dimension for output
            use_positional_encoding: Whether to use positional encoding
            positional_encoding_type: Type of positional encoding ("sinusoidal" or "learnable")
            dropout: Dropout probability
            use_continuous_expression: Whether to use continuous expression values
        """
        super().__init__()

        self.use_continuous_expression = use_continuous_expression

        # Gene embeddings
        self.gene_embedding = GeneEmbedding(n_genes, gene_embedding_dim)

        # Expression embeddings
        if use_continuous_expression:
            self.expression_embedding = ContinuousExpressionEmbedding(gene_embedding_dim)
        else:
            self.expression_embedding = ExpressionEmbedding(expression_bins, gene_embedding_dim)

        # Combine gene and expression embeddings
        self.combine_proj = nn.Linear(gene_embedding_dim * 2, hidden_dim)

        # Positional encoding
        self.use_positional_encoding = use_positional_encoding
        if use_positional_encoding:
            if positional_encoding_type == "sinusoidal":
                self.pos_encoding = PositionalEncoding(hidden_dim, dropout=dropout)
            else:
                self.pos_encoding = LearnablePositionalEncoding(hidden_dim, dropout=dropout)

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: Discretized expression values [batch_size, n_genes]
            input_values: Continuous expression values [batch_size, n_genes]
            attention_mask: Attention mask [batch_size, n_genes]

        Returns:
            Encoded representations [batch_size, n_genes, hidden_dim]
        """
        batch_size = input_ids.size(0) if input_ids is not None else input_values.size(0)
        n_genes = input_ids.size(1) if input_ids is not None else input_values.size(1)

        # Create gene indices
        gene_indices = torch.arange(n_genes, device=input_ids.device if input_ids is not None else input_values.device)
        gene_indices = gene_indices.unsqueeze(0).expand(batch_size, -1)

        # Get gene embeddings
        gene_embeds = self.gene_embedding(gene_indices)

        # Get expression embeddings
        if self.use_continuous_expression:
            expr_embeds = self.expression_embedding(input_values)
        else:
            expr_embeds = self.expression_embedding(input_ids)

        # Combine embeddings
        combined = torch.cat([gene_embeds, expr_embeds], dim=-1)
        x = self.combine_proj(combined)

        # Add positional encoding
        if self.use_positional_encoding:
            x = self.pos_encoding(x)

        # Layer norm and dropout
        x = self.layer_norm(x)
        x = self.dropout(x)

        return x
