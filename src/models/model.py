"""Main scRNA-seq foundation model."""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from .encoder import GeneEncoder
from .transformer import TransformerEncoder


class MLMHead(nn.Module):
    """Masked Language Modeling head for predicting masked gene expressions."""

    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int
    ):
        """Initialize MLM head.

        Args:
            hidden_dim: Hidden dimension
            vocab_size: Vocabulary size (number of expression bins + special tokens)
        """
        super().__init__()

        self.dense = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.decoder = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Hidden states [batch_size, seq_len, hidden_dim]

        Returns:
            Logits for each token [batch_size, seq_len, vocab_size]
        """
        x = self.dense(x)
        x = self.activation(x)
        x = self.layer_norm(x)
        x = self.decoder(x)
        return x


class ContinuousMLMHead(nn.Module):
    """MLM head for predicting continuous expression values."""

    def __init__(
        self,
        hidden_dim: int
    ):
        """Initialize continuous MLM head.

        Args:
            hidden_dim: Hidden dimension
        """
        super().__init__()

        self.dense = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.decoder = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Hidden states [batch_size, seq_len, hidden_dim]

        Returns:
            Predicted expression values [batch_size, seq_len]
        """
        x = self.dense(x)
        x = self.activation(x)
        x = self.layer_norm(x)
        x = self.decoder(x).squeeze(-1)
        return x


class ContrastiveHead(nn.Module):
    """Contrastive learning head for cell embeddings."""

    def __init__(
        self,
        hidden_dim: int,
        projection_dim: int
    ):
        """Initialize contrastive head.

        Args:
            hidden_dim: Hidden dimension
            projection_dim: Projection dimension for contrastive learning
        """
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, projection_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Cell embeddings [batch_size, hidden_dim]

        Returns:
            Projected embeddings [batch_size, projection_dim]
        """
        return self.projection(x)


class scRNAFoundationModel(nn.Module):
    """scRNA-seq Foundation Model."""

    def __init__(
        self,
        n_genes: int,
        gene_embedding_dim: int = 128,
        expression_bins: int = 50,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_positional_encoding: bool = True,
        use_mlm_head: bool = True,
        use_contrastive_head: bool = True,
        projection_dim: int = 128,
        use_continuous_expression: bool = False
    ):
        """Initialize scRNA-seq foundation model.

        Args:
            n_genes: Number of genes in vocabulary
            gene_embedding_dim: Dimension of gene embeddings
            expression_bins: Number of expression bins
            hidden_dim: Hidden dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            dropout: Dropout probability
            activation: Activation function
            use_positional_encoding: Whether to use positional encoding
            use_mlm_head: Whether to use MLM head
            use_contrastive_head: Whether to use contrastive head
            projection_dim: Projection dimension for contrastive learning
            use_continuous_expression: Whether to use continuous expression values
        """
        super().__init__()

        # Validate parameters
        if n_genes <= 0:
            raise ValueError(f"n_genes must be positive, got {n_genes}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
            )
        if not 0 <= dropout <= 1:
            raise ValueError(f"dropout must be between 0 and 1, got {dropout}")

        self.n_genes = n_genes
        self.hidden_dim = hidden_dim
        self.use_mlm_head = use_mlm_head
        self.use_contrastive_head = use_contrastive_head
        self.use_continuous_expression = use_continuous_expression

        # Gene encoder
        self.encoder = GeneEncoder(
            n_genes=n_genes,
            gene_embedding_dim=gene_embedding_dim,
            expression_bins=expression_bins,
            hidden_dim=hidden_dim,
            use_positional_encoding=use_positional_encoding,
            dropout=dropout,
            use_continuous_expression=use_continuous_expression
        )

        # Transformer encoder
        self.transformer = TransformerEncoder(
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            activation=activation
        )

        # Output heads
        if use_mlm_head:
            if use_continuous_expression:
                self.mlm_head = ContinuousMLMHead(hidden_dim)
            else:
                vocab_size = expression_bins + 3  # bins + mask + pad
                self.mlm_head = MLMHead(hidden_dim, vocab_size)

        if use_contrastive_head:
            self.contrastive_head = ContrastiveHead(hidden_dim, projection_dim)

        # Pooling for cell embeddings
        self.pooling = nn.AdaptiveAvgPool1d(1)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        return_embeddings: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            input_ids: Discretized expression values [batch_size, n_genes]
            input_values: Continuous expression values [batch_size, n_genes]
            attention_mask: Attention mask [batch_size, n_genes]
            return_attention: Whether to return attention weights
            return_embeddings: Whether to return cell embeddings

        Returns:
            Dictionary containing model outputs
        """
        # Encode genes and expressions
        encoded = self.encoder(
            input_ids=input_ids,
            input_values=input_values,
            attention_mask=attention_mask
        )

        # Transform with transformer
        hidden_states, all_attentions = self.transformer(
            encoded,
            attention_mask=attention_mask,
            return_all_attentions=return_attention
        )

        outputs = {
            "hidden_states": hidden_states
        }

        # MLM predictions
        if self.use_mlm_head:
            mlm_logits = self.mlm_head(hidden_states)
            outputs["mlm_logits"] = mlm_logits

        # Cell embeddings (mean pooling)
        cell_embeddings = hidden_states.mean(dim=1)  # [batch_size, hidden_dim]
        outputs["cell_embeddings"] = cell_embeddings

        # Contrastive projections
        if self.use_contrastive_head:
            contrastive_embeddings = self.contrastive_head(cell_embeddings)
            outputs["contrastive_embeddings"] = contrastive_embeddings

        if return_attention:
            outputs["attentions"] = all_attentions

        return outputs

    def get_cell_embeddings(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get cell embeddings.

        Args:
            input_ids: Discretized expression values
            input_values: Continuous expression values
            attention_mask: Attention mask

        Returns:
            Cell embeddings [batch_size, hidden_dim]
        """
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                input_values=input_values,
                attention_mask=attention_mask
            )
            return outputs["cell_embeddings"]

    def get_gene_importance(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get gene importance scores from attention.

        Args:
            input_ids: Discretized expression values
            input_values: Continuous expression values
            attention_mask: Attention mask

        Returns:
            Gene importance scores [batch_size, n_genes]
        """
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                input_values=input_values,
                attention_mask=attention_mask,
                return_attention=True
            )

            # Average attention across all heads and layers
            attentions = outputs["attentions"]
            avg_attention = torch.stack(attentions).mean(dim=0)  # [batch_size, num_heads, seq_len, seq_len]
            avg_attention = avg_attention.mean(dim=1)  # [batch_size, seq_len, seq_len]

            # Get attention to each gene (column-wise mean)
            gene_importance = avg_attention.mean(dim=1)  # [batch_size, seq_len]

            return gene_importance


def create_model(config: Dict[str, Any]) -> scRNAFoundationModel:
    """Create model from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        scRNA foundation model
    """
    model_config = config.get('model', {})

    model = scRNAFoundationModel(
        n_genes=model_config.get('n_genes', 2000),
        gene_embedding_dim=model_config.get('gene_embedding_dim', 128),
        expression_bins=model_config.get('expression_bins', 50),
        hidden_dim=model_config.get('hidden_dim', 256),
        num_layers=model_config.get('num_layers', 4),
        num_heads=model_config.get('num_heads', 8),
        ff_dim=model_config.get('ff_dim', 1024),
        dropout=model_config.get('dropout', 0.1),
        activation=model_config.get('activation', 'gelu'),
        use_positional_encoding=model_config.get('use_positional_encoding', True),
        use_mlm_head=model_config.get('use_mlm_head', True),
        use_contrastive_head=model_config.get('use_contrastive_head', True),
        projection_dim=model_config.get('projection_dim', 128),
        use_continuous_expression=model_config.get('expression_encoding', 'binned') == 'continuous'
    )

    return model
