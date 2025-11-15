"""Loss functions for pre-training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MaskedLMLoss(nn.Module):
    """Masked Language Modeling loss for discrete tokens."""

    def __init__(self, ignore_index: int = -100):
        """Initialize MLM loss.

        Args:
            ignore_index: Index to ignore in loss calculation
        """
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Calculate MLM loss.

        Args:
            logits: Predicted logits [batch_size, seq_len, vocab_size]
            labels: Target labels [batch_size, seq_len]
            mask: Binary mask indicating which positions to compute loss for

        Returns:
            Loss value
        """
        if mask is not None:
            # Only compute loss for masked positions
            masked_logits = logits[mask.bool()]
            masked_labels = labels[mask.bool()]

            if masked_logits.numel() == 0:
                return torch.tensor(0.0, device=logits.device)

            loss = self.loss_fn(masked_logits, masked_labels)
        else:
            # Reshape for cross entropy
            batch_size, seq_len, vocab_size = logits.shape
            logits = logits.view(-1, vocab_size)
            labels = labels.view(-1)
            loss = self.loss_fn(logits, labels)

        return loss


class ContinuousMSELoss(nn.Module):
    """MSE loss for continuous expression prediction."""

    def __init__(self):
        """Initialize MSE loss."""
        super().__init__()
        self.loss_fn = nn.MSELoss(reduction='none')

    def forward(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Calculate MSE loss.

        Args:
            predictions: Predicted values [batch_size, seq_len]
            labels: Target values [batch_size, seq_len]
            mask: Binary mask for loss computation

        Returns:
            Loss value
        """
        loss = self.loss_fn(predictions, labels)

        if mask is not None:
            # Only compute loss for masked positions
            loss = loss * mask
            loss = loss.sum() / (mask.sum() + 1e-8)
        else:
            loss = loss.mean()

        return loss


class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy (NT-Xent) loss for contrastive learning."""

    def __init__(self, temperature: float = 0.07):
        """Initialize NT-Xent loss.

        Args:
            temperature: Temperature parameter for scaling
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor
    ) -> torch.Tensor:
        """Calculate contrastive loss.

        Args:
            z_i: First set of embeddings [batch_size, projection_dim]
            z_j: Second set of embeddings [batch_size, projection_dim]

        Returns:
            Loss value
        """
        batch_size = z_i.size(0)

        # Normalize embeddings
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Concatenate embeddings
        z = torch.cat([z_i, z_j], dim=0)  # [2 * batch_size, projection_dim]

        # Compute similarity matrix
        similarity_matrix = torch.matmul(z, z.T) / self.temperature  # [2B, 2B]

        # Create mask to exclude self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))

        # Positive pairs are at positions (i, i+B) and (i+B, i)
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=z.device),
            torch.arange(0, batch_size, device=z.device)
        ])

        # Cross entropy loss
        loss = F.cross_entropy(similarity_matrix, labels)

        return loss


class SimCLRLoss(nn.Module):
    """SimCLR-style contrastive loss with augmentation."""

    def __init__(self, temperature: float = 0.07):
        """Initialize SimCLR loss.

        Args:
            temperature: Temperature parameter
        """
        super().__init__()
        self.temperature = temperature
        self.ntxent = NTXentLoss(temperature)

    def forward(
        self,
        embeddings: torch.Tensor,
        augmented_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Calculate SimCLR loss.

        Args:
            embeddings: Original embeddings
            augmented_embeddings: Augmented embeddings

        Returns:
            Loss value
        """
        return self.ntxent(embeddings, augmented_embeddings)


class CombinedLoss(nn.Module):
    """Combined loss for multiple objectives."""

    def __init__(
        self,
        mlm_weight: float = 1.0,
        contrastive_weight: float = 0.5,
        use_continuous: bool = False,
        temperature: float = 0.07
    ):
        """Initialize combined loss.

        Args:
            mlm_weight: Weight for MLM loss
            contrastive_weight: Weight for contrastive loss
            use_continuous: Whether to use continuous MSE loss
            temperature: Temperature for contrastive loss
        """
        super().__init__()

        self.mlm_weight = mlm_weight
        self.contrastive_weight = contrastive_weight

        # MLM loss
        if use_continuous:
            self.mlm_loss = ContinuousMSELoss()
        else:
            self.mlm_loss = MaskedLMLoss()

        # Contrastive loss
        self.contrastive_loss = NTXentLoss(temperature)

    def forward(
        self,
        mlm_logits: Optional[torch.Tensor] = None,
        mlm_labels: Optional[torch.Tensor] = None,
        mlm_mask: Optional[torch.Tensor] = None,
        embeddings_1: Optional[torch.Tensor] = None,
        embeddings_2: Optional[torch.Tensor] = None
    ) -> dict:
        """Calculate combined loss.

        Args:
            mlm_logits: MLM predictions
            mlm_labels: MLM labels
            mlm_mask: Mask for MLM loss
            embeddings_1: First set of embeddings for contrastive learning
            embeddings_2: Second set of embeddings for contrastive learning

        Returns:
            Dictionary containing loss components and total loss
        """
        losses = {}
        total_loss = 0.0

        # MLM loss
        if mlm_logits is not None and mlm_labels is not None:
            mlm_loss_value = self.mlm_loss(mlm_logits, mlm_labels, mlm_mask)
            losses['mlm_loss'] = mlm_loss_value
            total_loss += self.mlm_weight * mlm_loss_value

        # Contrastive loss
        if embeddings_1 is not None and embeddings_2 is not None:
            contrastive_loss_value = self.contrastive_loss(embeddings_1, embeddings_2)
            losses['contrastive_loss'] = contrastive_loss_value
            total_loss += self.contrastive_weight * contrastive_loss_value

        losses['total_loss'] = total_loss

        return losses
