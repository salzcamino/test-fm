"""CPU-optimized training script for laptops without GPU.

Designed for: Lenovo T480s and similar laptops
- Intel Core i5/i7 (quad-core)
- 16GB RAM
- No dedicated GPU
- Integrated graphics only

Expected training time: 2-6 hours for 3-5k cells
"""

import argparse
import torch
import logging
import os
from pathlib import Path

# CPU optimizations
os.environ['OMP_NUM_THREADS'] = '8'  # Use all available threads
os.environ['MKL_NUM_THREADS'] = '8'

from src.utils.logger import setup_logger
from src.data.loader import download_example_dataset
from src.data.preprocessor import scRNAPreprocessor
from src.data.dataset import create_dataloaders
from src.models.model import scRNAFoundationModel
from src.training.trainer import Trainer


def load_cpu_config():
    """Load CPU-optimized configuration."""
    import yaml

    config = {}

    with open('configs/model_config_cpu.yaml', 'r') as f:
        model_cfg = yaml.safe_load(f)
        config.update(model_cfg)

    with open('configs/training_config_cpu.yaml', 'r') as f:
        train_cfg = yaml.safe_load(f)
        config.update(train_cfg)

    with open('configs/data_config.yaml', 'r') as f:
        data_cfg = yaml.safe_load(f)
        config.update(data_cfg)

    return config


def main():
    """CPU-optimized training."""
    print("=" * 70)
    print("CPU-ONLY TRAINING - Optimized for Laptops")
    print("=" * 70)
    print("\n⚠️  Note: This will take several hours on CPU.")
    print("For faster training, consider using Google Colab (free GPU).\n")

    # Setup
    logger = setup_logger(log_file='logs/training_cpu.log')
    logger.info("Starting CPU-only training")

    # Set CPU threads
    torch.set_num_threads(8)
    logger.info(f"Using {torch.get_num_threads()} CPU threads")

    # Load config
    config = load_cpu_config()

    # Create output directory
    output_dir = Path('outputs_cpu')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download PBMC3k dataset (small)
    logger.info("Downloading PBMC3k dataset")
    adata = download_example_dataset('pbmc3k', save_dir='data/raw')
    logger.info(f"Loaded: {adata.n_obs} cells × {adata.n_vars} genes")

    # Subsample to 3k cells for faster CPU training
    if adata.n_obs > 3000:
        import scanpy as sc
        logger.info("Subsampling to 3000 cells for faster CPU training")
        sc.pp.subsample(adata, n_obs=3000)
        logger.info(f"Subsampled to {adata.n_obs} cells")

    # Preprocess
    logger.info("Preprocessing data")
    preprocessor = scRNAPreprocessor(
        min_genes=200,
        min_cells=3,
        n_top_genes=config['model']['n_genes'],  # 500 genes only
        normalize=True,
        log_transform=True,
        scale=False
    )

    adata = preprocessor.preprocess(adata, return_hvg_subset=True)
    logger.info(f"Preprocessed: {adata.n_obs} cells × {adata.n_vars} genes")

    # Create dataloaders
    logger.info("Creating dataloaders")
    train_loader, val_loader, test_loader = create_dataloaders(
        adata,
        batch_size=config['training']['batch_size'],
        train_split=0.8,
        val_split=0.1,
        num_workers=0,  # Important for CPU!
        expression_bins=config['model']['expression_bins'],
        mask_prob=config['training']['mlm_probability']
    )

    logger.info(f"Train: {len(train_loader.dataset)}, "
                f"Val: {len(val_loader.dataset)}, "
                f"Test: {len(test_loader.dataset)}")

    # Create tiny model
    logger.info("Creating CPU-optimized model")
    model = scRNAFoundationModel(
        n_genes=config['model']['n_genes'],
        gene_embedding_dim=config['model']['gene_embedding_dim'],
        expression_bins=config['model']['expression_bins'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        ff_dim=config['model']['ff_dim'],
        dropout=config['model']['dropout'],
        use_mlm_head=True,
        use_contrastive_head=False  # Disabled for CPU
    )

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {n_params:,} parameters ({n_params/1e6:.2f}M)")

    # Estimate time
    batches_per_epoch = len(train_loader)
    total_batches = batches_per_epoch * config['training']['num_epochs']
    est_seconds = total_batches * 0.5  # ~0.5 sec per batch on modern CPU
    est_hours = est_seconds / 3600

    print("\n" + "=" * 70)
    print("TRAINING CONFIGURATION:")
    print(f"  Device: CPU ({torch.get_num_threads()} threads)")
    print(f"  Model size: {n_params/1e6:.2f}M parameters")
    print(f"  Training samples: {len(train_loader.dataset)}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Epochs: {config['training']['num_epochs']}")
    print(f"  Estimated time: {est_hours:.1f}-{est_hours*2:.1f} hours")
    print("\n⚠️  Your laptop will:")
    print("  - Get warm (this is normal)")
    print("  - Fan will run continuously")
    print("  - Use significant power (keep plugged in)")
    print("  - Be slower for other tasks")
    print("\n💡 Tip: You can pause with Ctrl+C and resume later!")
    print("=" * 70 + "\n")

    input("Press Enter to start training (or Ctrl+C to cancel)...")

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device='cpu'
    )

    # Train
    try:
        logger.info("Starting training...")
        trainer.train()

        # Save model
        final_path = output_dir / 'final_model_cpu.pt'
        torch.save(model.state_dict(), final_path)

        print("\n" + "=" * 70)
        print("✅ Training complete!")
        print(f"Model saved to: {final_path}")
        print("=" * 70)

    except KeyboardInterrupt:
        logger.info("Training interrupted")
        print("\n⚠️  Training paused. Saving checkpoint...")
        trainer.save_checkpoint()
        print("You can resume with the saved checkpoint later.")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"\n❌ Error: {e}")
        raise


if __name__ == '__main__':
    main()
