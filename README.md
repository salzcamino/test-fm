# scRNA-seq Foundation Model

A mini foundation model for single-cell RNA sequencing (scRNAseq) analysis, built with PyTorch. This model learns representations of cells and genes through self-supervised pre-training on scRNA-seq data.

## Features

- **Transformer-based architecture** for learning cell and gene representations
- **Multiple pre-training objectives**:
  - Masked Gene Expression Modeling (MGEM)
  - Contrastive learning for cell embeddings
- **Flexible data processing** pipeline supporting multiple formats (h5ad, 10X, CSV, loom)
- **Comprehensive preprocessing** with quality control and normalization
- **Easy-to-use training pipeline** with checkpointing and logging
- **Evaluation metrics** for clustering, classification, and reconstruction
- **Visualization tools** for embeddings and attention patterns

## Architecture

The model consists of:
1. **Gene Encoder**: Combines gene embeddings with expression value embeddings
2. **Transformer Backbone**: Multi-layer transformer encoder (4 layers, 8 heads by default)
3. **Output Heads**:
   - MLM head for masked gene expression prediction
   - Contrastive head for cell representation learning

**Model Size**: ~10-50M parameters (configurable)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/scrna-foundation-model.git
cd scrna-foundation-model

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Hardware Requirements

### Can I train this on my PC? **YES!**

This model is designed to be trainable on local hardware:

| Setup | GPU | RAM | Suitable For |
|-------|-----|-----|--------------|
| **Minimal** | RTX 3050+ (4GB VRAM) or CPU | 8-16 GB | Learning, testing (~2-4 hours) |
| **Default** | RTX 3060+ (8GB VRAM) | 16-32 GB | Research, production (~4-8 hours) |
| **Large** | RTX 3080+ (12GB VRAM) | 32-64 GB | Large datasets, publications (~8-16 hours) |

**Training times** are for 10-50k cells, 50-100 epochs on mid-range GPU.

**CPU-only training** is possible but slow (2-6 hours for ultra-minimal config).

**Quick start guides**:
```bash
# GPU training (minimal config)
python train_minimal.py

# CPU-only training (laptops without GPU)
python train_cpu.py
```

📖 **Detailed guides**:
- **[Hardware Requirements](docs/HARDWARE_REQUIREMENTS.md)** - Full specs and optimization
- **[CPU Training Guide](docs/CPU_TRAINING_GUIDE.md)** - For laptops without GPU (T480s, MacBook, etc.)

## Quick Start

### 1. Basic Usage

```python
from src.data.loader import download_example_dataset
from src.data.preprocessor import scRNAPreprocessor
from src.data.dataset import scRNADataset
from src.models.model import scRNAFoundationModel

# Load and preprocess data
adata = download_example_dataset('pbmc3k')
preprocessor = scRNAPreprocessor(n_top_genes=2000)
adata = preprocessor.preprocess(adata, return_hvg_subset=True)

# Create dataset
dataset = scRNADataset(adata, expression_bins=50, mask_prob=0.15)

# Create model
model = scRNAFoundationModel(
    n_genes=2000,
    hidden_dim=256,
    num_layers=4,
    num_heads=8
)

# Get cell embeddings
cell_embeddings = model.get_cell_embeddings(input_ids=batch['input_ids'])
```

### 2. Training from Command Line

```bash
# Train with default configuration
python train.py

# Train with custom data
python train.py --data_path /path/to/data.h5ad

# Resume from checkpoint
python train.py --resume checkpoints/best_model.pt

# Specify device
python train.py --device cuda
```

### 3. Using Configuration Files

Edit configuration files in `configs/`:
- `model_config.yaml`: Model architecture settings
- `training_config.yaml`: Training hyperparameters
- `data_config.yaml`: Data processing settings

## Project Structure

```
scrna-foundation-model/
├── configs/              # Configuration files
│   ├── model_config.yaml
│   ├── training_config.yaml
│   └── data_config.yaml
├── src/
│   ├── data/            # Data loading and preprocessing
│   │   ├── loader.py
│   │   ├── preprocessor.py
│   │   └── dataset.py
│   ├── models/          # Model architectures
│   │   ├── encoder.py
│   │   ├── transformer.py
│   │   └── model.py
│   ├── training/        # Training pipeline
│   │   ├── trainer.py
│   │   ├── losses.py
│   │   └── metrics.py
│   └── utils/           # Utilities
│       ├── config.py
│       ├── logger.py
│       └── visualization.py
├── examples/            # Example scripts
├── notebooks/           # Jupyter notebooks
├── tests/              # Unit tests
├── data/               # Data directory
├── train.py            # Training script
└── requirements.txt    # Dependencies
```

## Data Preparation

### Supported Formats
- **h5ad**: AnnData format (recommended)
- **10X**: Matrix Market format
- **CSV/TSV**: Expression matrices
- **Loom**: Loom format

### Preprocessing Pipeline

```python
from src.data.preprocessor import scRNAPreprocessor

preprocessor = scRNAPreprocessor(
    min_genes=200,           # Minimum genes per cell
    min_cells=3,             # Minimum cells per gene
    max_genes=5000,          # Maximum genes (doublet filter)
    max_pct_mito=20,         # Maximum mitochondrial %
    target_sum=1e4,          # Normalization target
    n_top_genes=2000,        # Number of HVGs to select
    normalize=True,          # Normalize counts
    log_transform=True       # Log1p transformation
)

adata = preprocessor.preprocess(adata)
```

## Model Configuration

Key hyperparameters (edit in `configs/model_config.yaml`):

```yaml
model:
  n_genes: 2000              # Number of genes in vocabulary
  gene_embedding_dim: 128    # Gene embedding dimension
  expression_bins: 50        # Expression discretization bins
  hidden_dim: 256           # Hidden dimension
  num_layers: 4             # Number of transformer layers
  num_heads: 8              # Number of attention heads
  ff_dim: 1024              # Feed-forward dimension
  dropout: 0.1              # Dropout rate
```

## Training

The training pipeline includes:
- **Mixed precision training** (optional)
- **Gradient accumulation**
- **Learning rate scheduling** (cosine/linear)
- **Automatic checkpointing**
- **WandB integration** for experiment tracking

```python
from src.training.trainer import Trainer

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config,
    device='cuda'
)

trainer.train()
```

## Evaluation

### Clustering Metrics
```python
from src.training.metrics import compute_clustering_metrics

metrics = compute_clustering_metrics(
    embeddings=cell_embeddings,
    labels=true_labels
)
# Returns: ARI, NMI, Silhouette score
```

### Visualization
```python
from src.utils.visualization import plot_umap

plot_umap(
    embeddings=cell_embeddings,
    labels=cell_types,
    save_path='umap_plot.png'
)
```

## Advanced Usage

### Custom Training Loop

```python
import torch
from src.models.model import create_model
from src.training.losses import CombinedLoss

# Create model
config = load_config('configs')
model = create_model(config)

# Setup training
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
loss_fn = CombinedLoss(mlm_weight=1.0, contrastive_weight=0.5)

# Training loop
for batch in train_loader:
    outputs = model(input_ids=batch['input_ids'])

    losses = loss_fn(
        mlm_logits=outputs['mlm_logits'],
        mlm_labels=batch['labels'],
        mlm_mask=batch['mask']
    )

    loss = losses['total_loss']
    loss.backward()
    optimizer.step()
```

### Fine-tuning for Downstream Tasks

```python
# Load pre-trained model
model = scRNAFoundationModel(...)
model.load_state_dict(torch.load('checkpoints/best_model.pt'))

# Add classification head
classifier = nn.Linear(model.hidden_dim, num_cell_types)

# Fine-tune on labeled data
for batch in labeled_data_loader:
    embeddings = model.get_cell_embeddings(batch['input_ids'])
    predictions = classifier(embeddings)
    loss = criterion(predictions, batch['labels'])
    # ... backward pass
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{scrna_foundation_model,
  title={scRNA-seq Foundation Model},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/scrna-foundation-model}
}
```

## License

MIT License

## Acknowledgments

- Built with PyTorch
- Uses Scanpy for scRNA-seq preprocessing
- Inspired by recent work in foundation models for genomics

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For questions and issues, please open an issue on GitHub.
