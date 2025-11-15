"""Transformer architecture for scRNA-seq foundation model."""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import math


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention layer."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        """Initialize multi-head attention.

        Args:
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        # Linear projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]
            attention_mask: Attention mask [batch_size, seq_len]
            return_attention: Whether to return attention weights

        Returns:
            Output tensor and optionally attention weights
        """
        batch_size, seq_len, _ = x.size()

        # Linear projections and reshape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # Apply attention mask if provided
        if attention_mask is not None:
            # Reshape mask for broadcasting
            mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Attention weights
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        context = torch.matmul(attn_weights, v)

        # Reshape and project output
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        output = self.out_proj(context)

        if return_attention:
            return output, attn_weights
        return output, None


class FeedForward(nn.Module):
    """Feed-forward network."""

    def __init__(
        self,
        hidden_dim: int,
        ff_dim: int,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        """Initialize feed-forward network.

        Args:
            hidden_dim: Hidden dimension
            ff_dim: Feed-forward dimension
            dropout: Dropout probability
            activation: Activation function name
        """
        super().__init__()

        self.fc1 = nn.Linear(hidden_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, hidden_dim)

        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]

        Returns:
            Output tensor
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerLayer(nn.Module):
    """Single transformer layer."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        """Initialize transformer layer.

        Args:
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            dropout: Dropout probability
            activation: Activation function
        """
        super().__init__()

        self.attention = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.feed_forward = FeedForward(hidden_dim, ff_dim, dropout, activation)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]
            attention_mask: Attention mask [batch_size, seq_len]
            return_attention: Whether to return attention weights

        Returns:
            Output tensor and optionally attention weights
        """
        # Self-attention with residual connection
        attn_output, attn_weights = self.attention(
            self.norm1(x),
            attention_mask=attention_mask,
            return_attention=return_attention
        )
        x = x + self.dropout(attn_output)

        # Feed-forward with residual connection
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_output)

        return x, attn_weights


class TransformerEncoder(nn.Module):
    """Transformer encoder stack."""

    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        """Initialize transformer encoder.

        Args:
            num_layers: Number of transformer layers
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            dropout: Dropout probability
            activation: Activation function
        """
        super().__init__()

        self.layers = nn.ModuleList([
            TransformerLayer(
                hidden_dim,
                num_heads,
                ff_dim,
                dropout,
                activation
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_all_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """Forward pass.

        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]
            attention_mask: Attention mask [batch_size, seq_len]
            return_all_attentions: Whether to return all attention weights

        Returns:
            Output tensor and optionally list of attention weights
        """
        all_attentions = [] if return_all_attentions else None

        for layer in self.layers:
            x, attn_weights = layer(
                x,
                attention_mask=attention_mask,
                return_attention=return_all_attentions
            )

            if return_all_attentions:
                all_attentions.append(attn_weights)

        x = self.norm(x)

        return x, all_attentions
