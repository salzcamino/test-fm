"""Quick training script for minimal configuration on limited hardware."""

import argparse
import torch
import logging
from pathlib import Path

from src.utils.config import load_all_configs
from src.utils.logger import setup_logger
from src.data.loader import download_example_dataset, scRNADataLoader
from src.data.preprocessor import scRNAPreprocessor
from src.data.dataset import create_dataloaders
from src.models.model import create_model
from src.training.trainer import Trainer


def main():
    """Minimal training script."""
    print("=" * 70)
    print("MINIMAL TRAINING SETUP - Optimized for Limited Hardware")
    print("=" * 70)

    # Setup logger
    logger = setup_logger(log_file='logs/training_minimal.log')
    logger.info("Starting minimal training setup")

    # Load minimal configurations
    config = {}
    config['model'] = load_all_configs('configs')

    # Override with minimal configs
    import yaml
    with open('configs/model_config_minimal.yaml', 'r') as f:
        model_cfg = yaml.safe_load(f)
        config.update(model_cfg)

    with open('configs/training_config_minimal.yaml', 'r') as f:
        train_cfg = yaml.safe_load(f)
        config.update(train_cfg)

    with open('configs/data_config.yaml', 'r') as f:
        data_cfg = yaml.safe_load(f)
        config.update(data_cfg)

    logger.info("Loaded minimal configuration")

    # Auto-detect device
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU detected: {gpu_name} ({gpu_memory:.1f} GB)")
    elif torch.backends.mps.is_available():
        device = "mps"
        logger.info("Apple Silicon (MPS) detected")
    else:
        device = "cpu"
        logger.warning("No GPU detected, using CPU (this will be slow!)")

    config['training']['device'] = device

    # Create output directory
    output_dir = Path('outputs_minimal')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download small dataset
    logger.info("Downloading PBMC3k dataset (small, good for testing)")
    adata = download_example_dataset('pbmc3k', save_dir='data/raw')
    logger.info(f"Loaded data: {adata.n_obs} cells × {adata.n_vars} genes")

    # Subsample to 5k cells for even faster training
    if adata.n_obs > 5000:
        import scanpy as sc
        logger.info("Subsampling to 5000 cells for faster training")
        sc.pp.subsample(adata, n_obs=5000)

    # Preprocess data
    logger.info("Preprocessing data")
    preprocessor = scRNAPreprocessor(
        min_genes=200,
        min_cells=3,
        max_genes=5000,
        max_pct_mito=20,
        target_sum=1e4,
        n_top_genes=config['model']['n_genes'],  # Use minimal gene count
        normalize=True,
        log_transform=True,
        scale=False
    )

    adata = preprocessor.preprocess(adata, return_hvg_subset=True)
    logger.info(f"Preprocessed data: {adata.n_obs} cells × {adata.n_vars} genes")

    # Create dataloaders with small batch size
    logger.info("Creating dataloaders")
    train_loader, val_loader, test_loader = create_dataloaders(
        adata,
        batch_size=config['training']['batch_size'],
        train_split=0.8,
        val_split=0.1,
        num_workers=config['training']['num_workers'],
        expression_bins=config['model']['expression_bins'],
        mask_prob=config['training']['mlm_probability']
    )

    logger.info(f"Created dataloaders - Train: {len(train_loader.dataset)}, "
                f"Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

    # Create model
    logger.info("Creating minimal model")
    model = create_model(config)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,} total ({n_params/1e6:.2f}M), "
                f"{n_trainable:,} trainable")

    # Estimate memory
    param_memory_mb = (n_params * 4) / (1024 ** 2)
    logger.info(f"Estimated model size: {param_memory_mb:.1f} MB")

    # Create trainer
    logger.info("Creating trainer")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )

    # Train
    print("\n" + "=" * 70)
    print("Starting training...")
    print(f"Device: {device}")
    print(f"Model size: {n_params/1e6:.2f}M parameters")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Epochs: {config['training']['num_epochs']}")
    print("=" * 70 + "\n")

    try:
        trainer.train()
        logger.info("Training complete!")

        # Save final model
        final_model_path = output_dir / 'final_model_minimal.pt'
        torch.save(model.state_dict(), final_model_path)
        logger.info(f"Saved final model to {final_model_path}")

        print("\n" + "=" * 70)
        print("✅ Training completed successfully!")
        print(f"Model saved to: {final_model_path}")
        print("=" * 70)

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        print("\n⚠️  Training interrupted. Saving checkpoint...")
        trainer.save_checkpoint()
        print("Checkpoint saved. You can resume training later.")

    except RuntimeError as e:
        if "out of memory" in str(e):
            logger.error("GPU out of memory!")
            print("\n❌ GPU out of memory!")
            print("\nTry these solutions:")
            print("1. Reduce batch_size in configs/training_config_minimal.yaml")
            print("2. Reduce n_genes in configs/model_config_minimal.yaml")
            print("3. Enable gradient checkpointing")
            print("4. Use CPU instead: set device='cpu' in config")
        else:
            raise


if __name__ == '__main__':
    main()
