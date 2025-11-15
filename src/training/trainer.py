"""Training pipeline for scRNA-seq foundation model."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Optional, Dict, Any
import json

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from .losses import CombinedLoss

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer for scRNA-seq foundation model."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[Dict[str, Any]] = None,
        device: str = "cuda"
    ):
        """Initialize trainer.

        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Configuration dictionary
            device: Device to use for training
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or {}
        self.device = device

        # Move model to device
        self.model.to(device)

        # Training config
        training_config = self.config.get('training', {})
        self.num_epochs = training_config.get('num_epochs', 100)
        self.gradient_accumulation_steps = training_config.get('gradient_accumulation_steps', 1)
        self.max_grad_norm = training_config.get('max_grad_norm', 1.0)
        self.logging_steps = training_config.get('logging_steps', 100)
        self.eval_steps = training_config.get('eval_steps', 1000)
        self.save_steps = training_config.get('save_steps', 5000)
        self.save_total_limit = training_config.get('save_total_limit', 3)

        # Checkpoint directory
        self.checkpoint_dir = Path(training_config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Setup optimizer
        self._setup_optimizer()

        # Setup loss function
        self._setup_loss()

        # Setup logging
        self._setup_logging()

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.checkpoints = []

    def _setup_optimizer(self):
        """Setup optimizer and learning rate scheduler."""
        training_config = self.config.get('training', {})

        # Optimizer parameters
        lr = training_config.get('learning_rate', 1e-4)
        weight_decay = training_config.get('weight_decay', 0.01)
        betas = (
            training_config.get('adam_beta1', 0.9),
            training_config.get('adam_beta2', 0.999)
        )
        eps = training_config.get('adam_epsilon', 1e-8)

        # Create optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay
        )

        # Learning rate scheduler
        scheduler_type = training_config.get('lr_scheduler', 'cosine')
        warmup_steps = training_config.get('warmup_steps', 1000)

        if scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.num_epochs * len(self.train_loader)
            )
        elif scheduler_type == 'linear':
            self.scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=self.num_epochs * len(self.train_loader)
            )
        else:
            self.scheduler = None

        logger.info(f"Setup optimizer: AdamW with lr={lr}, scheduler={scheduler_type}")

    def _setup_loss(self):
        """Setup loss function."""
        training_config = self.config.get('training', {})
        model_config = self.config.get('model', {})

        mlm_weight = training_config.get('mlm_weight', 1.0)
        contrastive_weight = training_config.get('contrastive_weight', 0.5)
        temperature = training_config.get('contrastive_temperature', 0.07)
        use_continuous = model_config.get('expression_encoding') == 'continuous'

        self.loss_fn = CombinedLoss(
            mlm_weight=mlm_weight,
            contrastive_weight=contrastive_weight,
            use_continuous=use_continuous,
            temperature=temperature
        )

        logger.info(f"Setup loss: MLM weight={mlm_weight}, Contrastive weight={contrastive_weight}")

    def _setup_logging(self):
        """Setup logging (WandB if available)."""
        training_config = self.config.get('training', {})
        use_wandb = training_config.get('use_wandb', False) and WANDB_AVAILABLE

        if use_wandb:
            project = training_config.get('wandb_project', 'scrna-foundation-model')
            wandb.init(project=project, config=self.config)
            logger.info(f"Initialized WandB logging for project: {project}")
        else:
            logger.info("WandB logging not available or disabled")

        self.use_wandb = use_wandb

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute single training step.

        Args:
            batch: Batch of data

        Returns:
            Dictionary of losses
        """
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # Forward pass
        if 'input_ids' in batch:
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch.get('attention_mask')
            )
        else:
            outputs = self.model(
                input_values=batch['input_values'],
                attention_mask=batch.get('attention_mask')
            )

        # Calculate loss
        losses = self.loss_fn(
            mlm_logits=outputs.get('mlm_logits'),
            mlm_labels=batch.get('labels'),
            mlm_mask=batch.get('mask'),
            embeddings_1=outputs.get('contrastive_embeddings'),
            embeddings_2=None  # For now, single pass
        )

        # Backward pass
        loss = losses['total_loss'] / self.gradient_accumulation_steps
        loss.backward()

        return {k: v.item() for k, v in losses.items()}

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.

        Returns:
            Dictionary of average losses
        """
        self.model.train()
        epoch_losses = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")

        for step, batch in enumerate(pbar):
            # Training step
            losses = self.train_step(batch)
            epoch_losses.append(losses)

            # Gradient accumulation
            if (step + 1) % self.gradient_accumulation_steps == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )

                # Optimizer step
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()

                self.global_step += 1

                # Logging
                if self.global_step % self.logging_steps == 0:
                    avg_loss = sum(l['total_loss'] for l in epoch_losses[-self.logging_steps:]) / min(self.logging_steps, len(epoch_losses))
                    pbar.set_postfix({'loss': f'{avg_loss:.4f}'})

                    if self.use_wandb:
                        wandb.log({
                            'train/total_loss': avg_loss,
                            'train/step': self.global_step
                        })

                # Evaluation
                if self.val_loader is not None and self.global_step % self.eval_steps == 0:
                    val_metrics = self.evaluate()
                    logger.info(f"Step {self.global_step} - Val loss: {val_metrics['val_loss']:.4f}")

                    if self.use_wandb:
                        wandb.log(val_metrics)

                # Checkpointing
                if self.global_step % self.save_steps == 0:
                    self.save_checkpoint()

        # Calculate epoch average
        avg_losses = {}
        for key in epoch_losses[0].keys():
            avg_losses[key] = sum(l[key] for l in epoch_losses) / len(epoch_losses)

        return avg_losses

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set.

        Returns:
            Dictionary of validation metrics
        """
        if self.val_loader is None:
            return {}

        self.model.eval()
        val_losses = []

        for batch in tqdm(self.val_loader, desc="Evaluating"):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Forward pass
            if 'input_ids' in batch:
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch.get('attention_mask')
                )
            else:
                outputs = self.model(
                    input_values=batch['input_values'],
                    attention_mask=batch.get('attention_mask')
                )

            # Calculate loss
            losses = self.loss_fn(
                mlm_logits=outputs.get('mlm_logits'),
                mlm_labels=batch.get('labels'),
                mlm_mask=batch.get('mask')
            )

            val_losses.append({k: v.item() for k, v in losses.items()})

        # Average losses
        metrics = {}
        for key in val_losses[0].keys():
            avg_value = sum(l[key] for l in val_losses) / len(val_losses)
            metrics[f'val_{key}'] = avg_value

        self.model.train()
        return metrics

    def train(self):
        """Run full training loop."""
        logger.info(f"Starting training for {self.num_epochs} epochs")

        for epoch in range(self.num_epochs):
            self.epoch = epoch
            logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")

            # Train epoch
            epoch_losses = self.train_epoch()
            logger.info(f"Epoch {epoch + 1} - Train loss: {epoch_losses['total_loss']:.4f}")

            # Evaluate
            if self.val_loader is not None:
                val_metrics = self.evaluate()
                logger.info(f"Epoch {epoch + 1} - Val loss: {val_metrics.get('val_total_loss', 0):.4f}")

                # Save best model
                if val_metrics.get('val_total_loss', float('inf')) < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_total_loss']
                    self.save_checkpoint(is_best=True)

        logger.info("Training complete!")

    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint.

        Args:
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'epoch': self.epoch,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }

        # Save checkpoint
        if is_best:
            checkpoint_path = self.checkpoint_dir / 'best_model.pt'
        else:
            checkpoint_path = self.checkpoint_dir / f'checkpoint_step_{self.global_step}.pt'

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

        # Track checkpoints
        if not is_best:
            self.checkpoints.append(checkpoint_path)

            # Remove old checkpoints
            if len(self.checkpoints) > self.save_total_limit:
                old_checkpoint = self.checkpoints.pop(0)
                if old_checkpoint.exists():
                    old_checkpoint.unlink()

    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint.

        Args:
            checkpoint_path: Path to checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']

        logger.info(f"Loaded checkpoint from {checkpoint_path}")
