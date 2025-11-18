"""Tests for model architectures."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.mark.requires_torch
class TestGeneEncoder:
    """Tests for gene encoder module."""

    @pytest.fixture(autouse=True)
    def setup(self, skip_if_no_torch):
        """Setup for tests requiring torch."""
        import torch
        from src.models.encoder import GeneEncoder
        self.torch = torch
        self.GeneEncoder = GeneEncoder

    def test_gene_encoder_initialization(self):
        """Test gene encoder initialization."""
        encoder = self.GeneEncoder(
            n_genes=100,
            gene_embedding_dim=64,
            expression_bins=50,
            hidden_dim=128,
            use_positional_encoding=True,
            dropout=0.1
        )

        assert encoder is not None
        assert encoder.gene_embedding is not None
        assert encoder.expression_embedding is not None

    def test_gene_encoder_forward_discrete(self):
        """Test gene encoder forward pass with discrete input."""
        encoder = self.GeneEncoder(
            n_genes=50,
            gene_embedding_dim=64,
            expression_bins=50,
            hidden_dim=128,
            use_continuous_expression=False
        )

        batch_size = 4
        input_ids = self.torch.randint(0, 50, (batch_size, 50))

        output = encoder(input_ids=input_ids)

        assert output.shape == (batch_size, 50, 128)

    def test_gene_encoder_forward_continuous(self):
        """Test gene encoder forward pass with continuous input."""
        encoder = self.GeneEncoder(
            n_genes=50,
            gene_embedding_dim=64,
            expression_bins=50,
            hidden_dim=128,
            use_continuous_expression=True
        )

        batch_size = 4
        input_values = self.torch.randn(batch_size, 50)

        output = encoder(input_values=input_values)

        assert output.shape == (batch_size, 50, 128)


@pytest.mark.requires_torch
class TestTransformer:
    """Tests for transformer architecture."""

    @pytest.fixture(autouse=True)
    def setup(self, skip_if_no_torch):
        """Setup for tests requiring torch."""
        import torch
        from src.models.transformer import TransformerEncoder, MultiHeadAttention
        self.torch = torch
        self.TransformerEncoder = TransformerEncoder
        self.MultiHeadAttention = MultiHeadAttention

    def test_multihead_attention_initialization(self):
        """Test multi-head attention initialization."""
        attention = self.MultiHeadAttention(
            hidden_dim=256,
            num_heads=8,
            dropout=0.1
        )

        assert attention is not None
        assert attention.num_heads == 8
        assert attention.head_dim == 32

    def test_multihead_attention_forward(self):
        """Test multi-head attention forward pass."""
        attention = self.MultiHeadAttention(
            hidden_dim=256,
            num_heads=8
        )

        batch_size, seq_len = 4, 20
        x = self.torch.randn(batch_size, seq_len, 256)

        output, _ = attention(x)

        assert output.shape == (batch_size, seq_len, 256)

    def test_transformer_encoder_initialization(self):
        """Test transformer encoder initialization."""
        transformer = self.TransformerEncoder(
            num_layers=4,
            hidden_dim=256,
            num_heads=8,
            ff_dim=1024,
            dropout=0.1
        )

        assert transformer is not None
        assert len(transformer.layers) == 4

    def test_transformer_encoder_forward(self):
        """Test transformer encoder forward pass."""
        transformer = self.TransformerEncoder(
            num_layers=2,
            hidden_dim=128,
            num_heads=4,
            ff_dim=512
        )

        batch_size, seq_len = 4, 20
        x = self.torch.randn(batch_size, seq_len, 128)

        output, _ = transformer(x)

        assert output.shape == (batch_size, seq_len, 128)

    def test_transformer_encoder_with_attention_return(self):
        """Test transformer encoder returns attention weights."""
        transformer = self.TransformerEncoder(
            num_layers=2,
            hidden_dim=128,
            num_heads=4,
            ff_dim=512
        )

        batch_size, seq_len = 4, 20
        x = self.torch.randn(batch_size, seq_len, 128)

        output, attentions = transformer(x, return_all_attentions=True)

        assert output.shape == (batch_size, seq_len, 128)
        assert attentions is not None
        assert len(attentions) == 2  # 2 layers


@pytest.mark.requires_torch
class TestscRNAFoundationModel:
    """Tests for main foundation model."""

    @pytest.fixture(autouse=True)
    def setup(self, skip_if_no_torch):
        """Setup for tests requiring torch."""
        import torch
        from src.models.model import scRNAFoundationModel, create_model
        self.torch = torch
        self.scRNAFoundationModel = scRNAFoundationModel
        self.create_model = create_model

    def test_model_initialization(self):
        """Test model initialization with default parameters."""
        model = self.scRNAFoundationModel(
            n_genes=100,
            gene_embedding_dim=64,
            expression_bins=50,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            ff_dim=512
        )

        assert model is not None
        assert model.n_genes == 100
        assert model.hidden_dim == 128

    def test_model_forward_pass(self):
        """Test model forward pass."""
        model = self.scRNAFoundationModel(
            n_genes=50,
            hidden_dim=128,
            num_layers=2,
            num_heads=4
        )

        batch_size = 4
        input_ids = self.torch.randint(0, 50, (batch_size, 50))

        outputs = model(input_ids=input_ids)

        assert 'cell_embeddings' in outputs
        assert 'hidden_states' in outputs
        assert outputs['cell_embeddings'].shape == (batch_size, 128)
        assert outputs['hidden_states'].shape == (batch_size, 50, 128)

    def test_model_with_mlm_head(self):
        """Test model with MLM head."""
        model = self.scRNAFoundationModel(
            n_genes=50,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            use_mlm_head=True,
            expression_bins=50
        )

        batch_size = 4
        input_ids = self.torch.randint(0, 50, (batch_size, 50))

        outputs = model(input_ids=input_ids)

        assert 'mlm_logits' in outputs
        # vocab_size = expression_bins + 3 (mask, pad, etc)
        assert outputs['mlm_logits'].shape == (batch_size, 50, 53)

    def test_model_with_contrastive_head(self):
        """Test model with contrastive head."""
        model = self.scRNAFoundationModel(
            n_genes=50,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            use_contrastive_head=True,
            projection_dim=64
        )

        batch_size = 4
        input_ids = self.torch.randint(0, 50, (batch_size, 50))

        outputs = model(input_ids=input_ids)

        assert 'contrastive_embeddings' in outputs
        assert outputs['contrastive_embeddings'].shape == (batch_size, 64)

    def test_model_get_cell_embeddings(self):
        """Test getting cell embeddings."""
        model = self.scRNAFoundationModel(
            n_genes=50,
            hidden_dim=128,
            num_layers=2,
            num_heads=4
        )

        batch_size = 4
        input_ids = self.torch.randint(0, 50, (batch_size, 50))

        with self.torch.no_grad():
            embeddings = model.get_cell_embeddings(input_ids=input_ids)

        assert embeddings.shape == (batch_size, 128)

    def test_model_get_gene_importance(self):
        """Test getting gene importance scores."""
        model = self.scRNAFoundationModel(
            n_genes=50,
            hidden_dim=128,
            num_layers=2,
            num_heads=4
        )

        batch_size = 4
        input_ids = self.torch.randint(0, 50, (batch_size, 50))

        with self.torch.no_grad():
            importance = model.get_gene_importance(input_ids=input_ids)

        assert importance.shape == (batch_size, 50)

    def test_create_model_from_config(self, sample_config):
        """Test creating model from configuration."""
        model = self.create_model(sample_config)

        assert model is not None
        assert model.n_genes == sample_config['model']['n_genes']
        assert model.hidden_dim == sample_config['model']['hidden_dim']

    def test_model_parameter_count(self):
        """Test model parameter count is reasonable."""
        model = self.scRNAFoundationModel(
            n_genes=2000,
            hidden_dim=256,
            num_layers=4,
            num_heads=8
        )

        total_params = sum(p.numel() for p in model.parameters())

        # Should be in the range of 25-35M parameters for default config
        assert 20_000_000 < total_params < 40_000_000
