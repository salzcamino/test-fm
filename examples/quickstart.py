"""Quick start example for scRNA-seq foundation model."""

import torch
from src.data.loader import download_example_dataset
from src.data.preprocessor import scRNAPreprocessor
from src.data.dataset import scRNADataset
from src.models.model import scRNAFoundationModel


def main():
    """Quick start example."""
    print("=" * 50)
    print("scRNA-seq Foundation Model - Quick Start")
    print("=" * 50)

    # 1. Load example data
    print("\n1. Loading example dataset (PBMC3k)...")
    adata = download_example_dataset('pbmc3k', save_dir='../data/raw')
    print(f"   Loaded: {adata.n_obs} cells × {adata.n_vars} genes")

    # 2. Preprocess data
    print("\n2. Preprocessing data...")
    preprocessor = scRNAPreprocessor(
        min_genes=200,
        min_cells=3,
        n_top_genes=2000,
        normalize=True,
        log_transform=True
    )

    adata = preprocessor.preprocess(adata, return_hvg_subset=True)
    print(f"   Preprocessed: {adata.n_obs} cells × {adata.n_vars} genes")

    # 3. Create dataset
    print("\n3. Creating PyTorch dataset...")
    dataset = scRNADataset(
        adata,
        expression_bins=50,
        mask_prob=0.15
    )
    print(f"   Dataset size: {len(dataset)} cells")

    # 4. Create model
    print("\n4. Creating model...")
    model = scRNAFoundationModel(
        n_genes=2000,
        gene_embedding_dim=128,
        expression_bins=50,
        hidden_dim=256,
        num_layers=4,
        num_heads=8,
        ff_dim=1024,
        dropout=0.1
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {n_params:,}")

    # 5. Forward pass example
    print("\n5. Running forward pass...")
    batch = dataset[0]

    # Add batch dimension
    input_ids = batch['input_ids'].unsqueeze(0)
    attention_mask = batch['attention_mask'].unsqueeze(0)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

    print(f"   Output shapes:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"     {key}: {value.shape}")

    # 6. Extract cell embeddings
    print("\n6. Extracting cell embeddings...")
    cell_embeddings = model.get_cell_embeddings(
        input_ids=input_ids,
        attention_mask=attention_mask
    )
    print(f"   Cell embeddings shape: {cell_embeddings.shape}")

    print("\n" + "=" * 50)
    print("Quick start complete!")
    print("=" * 50)


if __name__ == '__main__':
    main()
