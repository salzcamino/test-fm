"""Script to estimate model size and memory requirements."""

import torch
from src.models.model import scRNAFoundationModel


def estimate_model_requirements(config_name="default"):
    """Estimate memory and computational requirements."""

    configs = {
        "minimal": {
            "n_genes": 1000,
            "gene_embedding_dim": 64,
            "hidden_dim": 128,
            "num_layers": 2,
            "num_heads": 4,
            "ff_dim": 512,
        },
        "default": {
            "n_genes": 2000,
            "gene_embedding_dim": 128,
            "hidden_dim": 256,
            "num_layers": 4,
            "num_heads": 8,
            "ff_dim": 1024,
        },
        "large": {
            "n_genes": 5000,
            "gene_embedding_dim": 256,
            "hidden_dim": 512,
            "num_layers": 6,
            "num_heads": 8,
            "ff_dim": 2048,
        }
    }

    print("=" * 70)
    print(f"MODEL REQUIREMENTS ESTIMATION - {config_name.upper()} CONFIG")
    print("=" * 70)

    cfg = configs[config_name]

    # Create model
    model = scRNAFoundationModel(
        n_genes=cfg["n_genes"],
        gene_embedding_dim=cfg["gene_embedding_dim"],
        expression_bins=50,
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        ff_dim=cfg["ff_dim"],
        dropout=0.1
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Estimate memory
    param_memory_mb = (total_params * 4) / (1024 ** 2)  # 4 bytes per float32

    # Optimizer states (AdamW has 2 states per parameter)
    optimizer_memory_mb = (total_params * 4 * 2) / (1024 ** 2)

    # Gradients
    gradient_memory_mb = param_memory_mb

    # Activations (rough estimate for batch_size=32)
    batch_size = 32
    seq_len = cfg["n_genes"]
    hidden_dim = cfg["hidden_dim"]
    num_layers = cfg["num_layers"]

    # Approximate activation memory per sample
    activation_per_sample = seq_len * hidden_dim * num_layers * 4 / (1024 ** 2)
    activation_memory_mb = activation_per_sample * batch_size

    total_memory_mb = param_memory_mb + optimizer_memory_mb + gradient_memory_mb + activation_memory_mb
    total_memory_gb = total_memory_mb / 1024

    print(f"\n📊 MODEL STATISTICS:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {total_params / 1e6:.2f}M parameters")

    print(f"\n💾 MEMORY REQUIREMENTS (estimated):")
    print(f"  Model parameters: {param_memory_mb:.1f} MB")
    print(f"  Optimizer states (AdamW): {optimizer_memory_mb:.1f} MB")
    print(f"  Gradients: {gradient_memory_mb:.1f} MB")
    print(f"  Activations (batch_size=32): {activation_memory_mb:.1f} MB")
    print(f"  ─────────────────────────────────")
    print(f"  TOTAL (training): ~{total_memory_gb:.2f} GB")
    print(f"  Inference only: ~{param_memory_mb / 1024:.2f} GB")

    print(f"\n🖥️  HARDWARE RECOMMENDATIONS:")

    if total_memory_gb < 4:
        print(f"  ✅ GPU: Entry-level (GTX 1660, RTX 3050) - 4-6GB VRAM")
        print(f"  ✅ CPU Training: Possible but slow")
    elif total_memory_gb < 8:
        print(f"  ✅ GPU: Mid-range (RTX 3060, RTX 3070) - 8-12GB VRAM")
        print(f"  ⚠️  CPU Training: Very slow, not recommended")
    else:
        print(f"  ⚠️  GPU: High-end (RTX 3080, RTX 4080, A100) - 12-24GB VRAM")
        print(f"  ❌ CPU Training: Not practical")

    print(f"  RAM: {max(16, int(total_memory_gb * 2))}GB+ recommended")

    print(f"\n⏱️  ESTIMATED TRAINING TIME (10k cells, 100 epochs):")

    # Very rough estimates
    if config_name == "minimal":
        print(f"  GPU (RTX 3060): ~2-4 hours")
        print(f"  GPU (RTX 4090): ~1-2 hours")
        print(f"  CPU (16 cores): ~24-48 hours")
    elif config_name == "default":
        print(f"  GPU (RTX 3060): ~4-8 hours")
        print(f"  GPU (RTX 4090): ~2-4 hours")
        print(f"  CPU (16 cores): ~48-96 hours (not recommended)")
    else:
        print(f"  GPU (RTX 3080): ~8-16 hours")
        print(f"  GPU (RTX 4090): ~4-8 hours")
        print(f"  CPU: Not practical")

    print(f"\n📦 DATASET REQUIREMENTS:")
    cells_10k = seq_len * 10000 * 4 / (1024 ** 2)
    cells_100k = seq_len * 100000 * 4 / (1024 ** 2)
    print(f"  10k cells: ~{cells_10k:.1f} MB")
    print(f"  100k cells: ~{cells_100k:.1f} MB")
    print(f"  1M cells: ~{cells_100k * 10 / 1024:.1f} GB")

    print("\n" + "=" * 70)
    print()


if __name__ == "__main__":
    # Estimate for all configurations
    for config in ["minimal", "default", "large"]:
        estimate_model_requirements(config)
        print("\n")
