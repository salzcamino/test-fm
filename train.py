"""Training script for scRNA-seq foundation model."""

import argparse
import torch
import logging
from pathlib import Path

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.data.loader import download_example_dataset, scRNADataLoader
from src.data.preprocessor import scRNAPreprocessor
from src.data.dataset import create_dataloaders
from src.models.model import create_model
from src.training.trainer import Trainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train scRNA-seq foundation model')

    parser.add_argument(
        '--config_dir',
        type=str,
        default='configs',
        help='Directory containing configuration files'
    )

    parser.add_argument(
        '--data_path',
        type=str,
        default=None,
        help='Path to training data (h5ad file). If None, downloads example dataset'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs',
        help='Output directory for checkpoints and logs'
    )

    parser.add_argument(
        '--log_file',
        type=str,
        default='logs/training.log',
        help='Path to log file'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for training'
    )

    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )

    return parser.parse_args()


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()

    # Setup logger
    logger = setup_logger(log_file=args.log_file)
    logger.info("Starting scRNA-seq foundation model training")

    # Load configuration
    config = Config(args.config_dir)
    logger.info(f"Loaded configuration from {args.config_dir}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load or download data
    if args.data_path is None:
        logger.info("No data path provided, downloading example dataset (PBMC3k)")
        adata = download_example_dataset('pbmc3k', save_dir='data/raw')
    else:
        logger.info(f"Loading data from {args.data_path}")
        loader = scRNADataLoader(args.data_path)
        adata = loader.auto_load()

    logger.info(f"Loaded data: {adata.n_obs} cells × {adata.n_vars} genes")

    # Preprocess data
    logger.info("Preprocessing data")
    data_config = config.get('data.preprocessing', {})

    preprocessor = scRNAPreprocessor(
        min_genes=data_config.get('min_genes_per_cell', 200),
        min_cells=data_config.get('min_cells_per_gene', 3),
        max_genes=data_config.get('max_genes_per_cell', 5000),
        max_pct_mito=data_config.get('max_pct_mito', 20),
        target_sum=data_config.get('target_sum', 1e4),
        n_top_genes=config.get('model.n_genes', 2000),
        normalize=data_config.get('normalize_total', True),
        log_transform=data_config.get('log1p', True),
        scale=data_config.get('scale', False)
    )

    adata = preprocessor.preprocess(adata, return_hvg_subset=True)
    logger.info(f"Preprocessed data: {adata.n_obs} cells × {adata.n_vars} genes")

    # Save HVG list
    hvg_path = output_dir / 'hvg_genes.csv'
    preprocessor.save_hvg_list(hvg_path)
    logger.info(f"Saved highly variable genes to {hvg_path}")

    # Create dataloaders
    logger.info("Creating dataloaders")
    training_config = config.get('training', {})
    data_split_config = config.get('data', {})

    train_loader, val_loader, test_loader = create_dataloaders(
        adata,
        batch_size=training_config.get('batch_size', 32),
        train_split=data_split_config.get('train_split', 0.8),
        val_split=data_split_config.get('val_split', 0.1),
        num_workers=training_config.get('num_workers', 4),
        expression_bins=config.get('model.expression_bins', 50),
        mask_prob=training_config.get('mlm_probability', 0.15)
    )

    logger.info(f"Created dataloaders - Train: {len(train_loader.dataset)}, "
                f"Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

    # Create model
    logger.info("Creating model")
    model = create_model(config.to_dict())

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,} total, {n_trainable:,} trainable")

    # Create trainer
    logger.info("Creating trainer")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config.to_dict(),
        device=args.device
    )

    # Resume from checkpoint if provided
    if args.resume is not None:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train
    logger.info("Starting training")
    trainer.train()

    logger.info("Training complete!")

    # Save final model
    final_model_path = output_dir / 'final_model.pt'
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Saved final model to {final_model_path}")


if __name__ == '__main__':
    main()
