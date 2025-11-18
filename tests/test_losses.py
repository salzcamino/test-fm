"""Tests for loss functions."""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.mark.requires_torch
class TestMaskedLMLoss:
    """Tests for Masked Language Modeling loss."""

    @pytest.fixture(autouse=True)
    def setup(self, skip_if_no_torch):
        """Setup for tests requiring torch."""
        import torch
        from src.training.losses import MaskedLMLoss
        self.torch = torch
        self.MaskedLMLoss = MaskedLMLoss

    def test_mlm_loss_initialization(self):
        """Test MLM loss initialization."""
        loss_fn = self.MaskedLMLoss()
        assert loss_fn is not None

    def test_mlm_loss_with_mask(self):
        """Test MLM loss calculation with mask."""
        loss_fn = self.MaskedLMLoss()

        batch_size, seq_len, vocab_size = 2, 10, 50
        logits = self.torch.randn(batch_size, seq_len, vocab_size)
        labels = self.torch.randint(0, vocab_size, (batch_size, seq_len))
        mask = self.torch.rand(batch_size, seq_len) < 0.3

        loss = loss_fn(logits, labels, mask)

        assert loss.item() >= 0
        assert not self.torch.isnan(loss)
        assert not self.torch.isinf(loss)

    def test_mlm_loss_without_mask(self):
        """Test MLM loss calculation without mask."""
        loss_fn = self.MaskedLMLoss()

        batch_size, seq_len, vocab_size = 2, 10, 50
        logits = self.torch.randn(batch_size, seq_len, vocab_size)
        labels = self.torch.randint(0, vocab_size, (batch_size, seq_len))

        loss = loss_fn(logits, labels, mask=None)

        assert loss.item() >= 0
        assert not self.torch.isnan(loss)

    def test_mlm_loss_empty_mask(self):
        """Test MLM loss with empty mask (no masked positions)."""
        loss_fn = self.MaskedLMLoss()

        batch_size, seq_len, vocab_size = 2, 10, 50
        logits = self.torch.randn(batch_size, seq_len, vocab_size)
        labels = self.torch.randint(0, vocab_size, (batch_size, seq_len))
        mask = self.torch.zeros(batch_size, seq_len)  # No masked positions

        loss = loss_fn(logits, labels, mask)

        # Should return 0 for empty mask
        assert loss.item() == 0.0


@pytest.mark.requires_torch
class TestNTXentLoss:
    """Tests for contrastive learning loss."""

    @pytest.fixture(autouse=True)
    def setup(self, skip_if_no_torch):
        """Setup for tests requiring torch."""
        import torch
        from src.training.losses import NTXentLoss
        self.torch = torch
        self.NTXentLoss = NTXentLoss

    def test_ntxent_loss_initialization(self):
        """Test NT-Xent loss initialization."""
        loss_fn = self.NTXentLoss(temperature=0.07)
        assert loss_fn is not None
        assert loss_fn.temperature == 0.07

    def test_ntxent_loss_calculation(self):
        """Test NT-Xent loss calculation."""
        loss_fn = self.NTXentLoss(temperature=0.07)

        batch_size, dim = 8, 128
        z_i = self.torch.randn(batch_size, dim)
        z_j = self.torch.randn(batch_size, dim)

        loss = loss_fn(z_i, z_j)

        assert loss.item() >= 0
        assert not self.torch.isnan(loss)
        assert not self.torch.isinf(loss)

    def test_ntxent_loss_identical_embeddings(self):
        """Test NT-Xent loss with identical embeddings."""
        loss_fn = self.NTXentLoss(temperature=0.07)

        batch_size, dim = 4, 64
        z = self.torch.randn(batch_size, dim)

        loss = loss_fn(z, z)

        # Loss should be low (not zero due to temperature)
        assert loss.item() >= 0
        assert not self.torch.isnan(loss)

    def test_ntxent_loss_temperature_effect(self):
        """Test effect of temperature on loss."""
        batch_size, dim = 4, 64
        z_i = self.torch.randn(batch_size, dim)
        z_j = self.torch.randn(batch_size, dim)

        loss_low_temp = self.NTXentLoss(temperature=0.01)(z_i, z_j)
        loss_high_temp = self.NTXentLoss(temperature=1.0)(z_i, z_j)

        # Different temperatures should give different losses
        assert loss_low_temp.item() != loss_high_temp.item()


@pytest.mark.requires_torch
class TestCombinedLoss:
    """Tests for combined loss function."""

    @pytest.fixture(autouse=True)
    def setup(self, skip_if_no_torch):
        """Setup for tests requiring torch."""
        import torch
        from src.training.losses import CombinedLoss
        self.torch = torch
        self.CombinedLoss = CombinedLoss

    def test_combined_loss_initialization(self):
        """Test combined loss initialization."""
        loss_fn = self.CombinedLoss(
            mlm_weight=1.0,
            contrastive_weight=0.5
        )

        assert loss_fn is not None
        assert loss_fn.mlm_weight == 1.0
        assert loss_fn.contrastive_weight == 0.5

    def test_combined_loss_mlm_only(self):
        """Test combined loss with only MLM component."""
        loss_fn = self.CombinedLoss(mlm_weight=1.0, contrastive_weight=0.0)

        batch_size, seq_len, vocab_size = 2, 10, 50
        logits = self.torch.randn(batch_size, seq_len, vocab_size)
        labels = self.torch.randint(0, vocab_size, (batch_size, seq_len))
        mask = self.torch.rand(batch_size, seq_len) < 0.15

        losses = loss_fn(
            mlm_logits=logits,
            mlm_labels=labels,
            mlm_mask=mask
        )

        assert 'total_loss' in losses
        assert 'mlm_loss' in losses
        assert losses['total_loss'].item() >= 0

    def test_combined_loss_contrastive_only(self):
        """Test combined loss with only contrastive component."""
        loss_fn = self.CombinedLoss(mlm_weight=0.0, contrastive_weight=1.0)

        batch_size, dim = 4, 128
        embeddings_1 = self.torch.randn(batch_size, dim)
        embeddings_2 = self.torch.randn(batch_size, dim)

        losses = loss_fn(
            embeddings_1=embeddings_1,
            embeddings_2=embeddings_2
        )

        assert 'total_loss' in losses
        assert 'contrastive_loss' in losses
        assert losses['total_loss'].item() >= 0

    def test_combined_loss_both_components(self):
        """Test combined loss with both components."""
        loss_fn = self.CombinedLoss(mlm_weight=1.0, contrastive_weight=0.5)

        batch_size, seq_len, vocab_size = 2, 10, 50
        logits = self.torch.randn(batch_size, seq_len, vocab_size)
        labels = self.torch.randint(0, vocab_size, (batch_size, seq_len))
        mask = self.torch.rand(batch_size, seq_len) < 0.15

        dim = 128
        embeddings_1 = self.torch.randn(batch_size, dim)
        embeddings_2 = self.torch.randn(batch_size, dim)

        losses = loss_fn(
            mlm_logits=logits,
            mlm_labels=labels,
            mlm_mask=mask,
            embeddings_1=embeddings_1,
            embeddings_2=embeddings_2
        )

        assert 'total_loss' in losses
        assert 'mlm_loss' in losses
        assert 'contrastive_loss' in losses

        # Total should be weighted sum
        expected_total = (
            loss_fn.mlm_weight * losses['mlm_loss'] +
            loss_fn.contrastive_weight * losses['contrastive_loss']
        )

        assert self.torch.allclose(losses['total_loss'], expected_total)


@pytest.mark.requires_torch
class TestContinuousMSELoss:
    """Tests for continuous expression MSE loss."""

    @pytest.fixture(autouse=True)
    def setup(self, skip_if_no_torch):
        """Setup for tests requiring torch."""
        import torch
        from src.training.losses import ContinuousMSELoss
        self.torch = torch
        self.ContinuousMSELoss = ContinuousMSELoss

    def test_mse_loss_initialization(self):
        """Test MSE loss initialization."""
        loss_fn = self.ContinuousMSELoss()
        assert loss_fn is not None

    def test_mse_loss_calculation(self):
        """Test MSE loss calculation."""
        loss_fn = self.ContinuousMSELoss()

        predictions = self.torch.randn(10, 20)
        labels = self.torch.randn(10, 20)

        loss = loss_fn(predictions, labels)

        assert loss.item() >= 0
        assert not self.torch.isnan(loss)

    def test_mse_loss_with_mask(self):
        """Test MSE loss with mask."""
        loss_fn = self.ContinuousMSELoss()

        predictions = self.torch.randn(10, 20)
        labels = self.torch.randn(10, 20)
        mask = self.torch.rand(10, 20) < 0.5

        loss = loss_fn(predictions, labels, mask)

        assert loss.item() >= 0
        assert not self.torch.isnan(loss)

    def test_mse_loss_perfect_prediction(self):
        """Test MSE loss with perfect predictions."""
        loss_fn = self.ContinuousMSELoss()

        predictions = self.torch.ones(5, 10) * 3.0
        labels = self.torch.ones(5, 10) * 3.0

        loss = loss_fn(predictions, labels)

        # Perfect prediction should give ~0 loss
        assert loss.item() < 1e-6
