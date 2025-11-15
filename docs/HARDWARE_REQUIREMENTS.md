# Hardware Requirements for Local Training

This guide helps you understand the computational requirements for training the scRNA-seq foundation model locally.

## TL;DR - Can I train this on my PC?

**YES!** With the right configuration:

- ✅ **Gaming PC with GPU (RTX 3060+)**: Excellent, recommended
- ✅ **Laptop with dedicated GPU (RTX 3050+)**: Good for smaller configs
- ⚠️ **CPU-only modern PC (16+ cores)**: Possible but very slow
- ❌ **Old laptop/CPU-only (< 8 cores)**: Not practical

## Model Configurations

We provide three pre-configured model sizes:

### 1. MINIMAL (For laptops/limited hardware)

**Model Size**: ~5-10M parameters

**Configuration**:
```yaml
n_genes: 1000
gene_embedding_dim: 64
hidden_dim: 128
num_layers: 2
num_heads: 4
ff_dim: 512
```

**Requirements**:
- **GPU VRAM**: 2-4 GB (GTX 1650, RTX 3050, MX450)
- **RAM**: 8 GB minimum, 16 GB recommended
- **Storage**: 5 GB
- **Training Time** (10k cells, 50 epochs):
  - RTX 3050: ~2-3 hours
  - GTX 1660: ~3-4 hours
  - CPU (8 cores): ~24-36 hours

**Best for**: Learning, testing, proof-of-concept

---

### 2. DEFAULT (Recommended for most use cases)

**Model Size**: ~25-35M parameters

**Configuration**:
```yaml
n_genes: 2000
gene_embedding_dim: 128
hidden_dim: 256
num_layers: 4
num_heads: 8
ff_dim: 1024
```

**Requirements**:
- **GPU VRAM**: 6-8 GB (RTX 3060, RTX 3070, RTX 4060)
- **RAM**: 16 GB minimum, 32 GB recommended
- **Storage**: 10 GB
- **Training Time** (10k cells, 100 epochs):
  - RTX 3060: ~4-6 hours
  - RTX 4070: ~2-3 hours
  - RTX 4090: ~1-2 hours
  - CPU (16 cores): ~60-80 hours (not recommended)

**Best for**: Research, production models, most datasets

---

### 3. LARGE (For high-end hardware)

**Model Size**: ~80-120M parameters

**Configuration**:
```yaml
n_genes: 5000
gene_embedding_dim: 256
hidden_dim: 512
num_layers: 6
num_heads: 8
ff_dim: 2048
```

**Requirements**:
- **GPU VRAM**: 12-16 GB (RTX 3080, RTX 4080, A100)
- **RAM**: 32 GB minimum, 64 GB recommended
- **Storage**: 20 GB
- **Training Time** (100k cells, 100 epochs):
  - RTX 3080: ~12-16 hours
  - RTX 4090: ~6-8 hours
  - A100: ~4-6 hours

**Best for**: Large-scale datasets, publication-quality models

## Memory Breakdown

### GPU Memory Usage (DEFAULT config, batch_size=32)

| Component | Memory |
|-----------|--------|
| Model Parameters | ~200 MB |
| Optimizer States (AdamW) | ~400 MB |
| Gradients | ~200 MB |
| Activations | ~1.5-2 GB |
| PyTorch Overhead | ~500 MB |
| **TOTAL** | **~3-4 GB** |

### RAM Usage

| Dataset Size | Preprocessed Data | Total RAM Needed |
|--------------|-------------------|------------------|
| 10k cells | ~80 MB | 8 GB min |
| 50k cells | ~400 MB | 16 GB min |
| 100k cells | ~800 MB | 32 GB min |
| 500k cells | ~4 GB | 64 GB min |

## Reducing Memory Requirements

If you have limited hardware, here are strategies to reduce memory usage:

### 1. Reduce Batch Size
```yaml
# In configs/training_config.yaml
training:
  batch_size: 16  # or even 8 for very limited VRAM
  gradient_accumulation_steps: 2  # Maintain effective batch size
```

### 2. Use Smaller Model
```yaml
# In configs/model_config.yaml
model:
  n_genes: 1000  # Instead of 2000
  hidden_dim: 128  # Instead of 256
  num_layers: 2  # Instead of 4
```

### 3. Reduce Sequence Length
```yaml
# In configs/data_config.yaml
data:
  preprocessing:
    n_top_genes: 1000  # Use fewer genes
```

### 4. Enable Mixed Precision Training
```yaml
# In configs/training_config.yaml
training:
  fp16: false
  bf16: true  # Use bfloat16 on newer GPUs (Ampere+)
```

This can reduce memory by 30-40%!

### 5. Subsample Your Dataset
```python
# In your data loading script
import scanpy as sc

adata = sc.read_h5ad('data.h5ad')
# Use only 10k cells for initial training
adata_subset = sc.pp.subsample(adata, n_obs=10000, copy=True)
```

## CPU-Only Training

Training on CPU is possible but **significantly slower**. Here's how to optimize:

### 1. Enable CPU Optimizations
```python
# Set environment variables before importing PyTorch
import os
os.environ['OMP_NUM_THREADS'] = '16'  # Use all CPU cores
os.environ['MKL_NUM_THREADS'] = '16'

import torch
torch.set_num_threads(16)
```

### 2. Reduce Model Complexity
Use the MINIMAL configuration and reduce batch size:
```yaml
training:
  batch_size: 8
  num_epochs: 20  # Fewer epochs for CPU
```

### 3. Expect Longer Training Times
- **Minimal model, 10k cells, 20 epochs**: 12-24 hours on modern CPU
- **Default model**: Not recommended for CPU

## Recommended Local Setups

### Budget Setup (~$500-1000)
- **GPU**: RTX 3060 (12GB VRAM) or RTX 4060 (8GB VRAM)
- **RAM**: 16 GB DDR4
- **Storage**: 512 GB SSD
- **Can train**: MINIMAL and DEFAULT configs comfortably

### Mid-Range Setup (~$1500-2500)
- **GPU**: RTX 4070 (12GB VRAM) or RTX 3080 (10-12GB VRAM)
- **RAM**: 32 GB DDR4/DDR5
- **Storage**: 1 TB NVMe SSD
- **Can train**: All configs including LARGE with medium datasets

### High-End Setup (~$3000+)
- **GPU**: RTX 4090 (24GB VRAM) or A5000 (24GB VRAM)
- **RAM**: 64-128 GB DDR5
- **Storage**: 2 TB NVMe SSD
- **Can train**: Any configuration with large datasets (1M+ cells)

## Cloud Alternatives

If local training is not feasible:

### Free Options (with limitations)
- **Google Colab** (Free tier): T4 GPU (16GB), limited to 12h sessions
- **Kaggle Notebooks**: P100 GPU (16GB), 30h/week limit
- **Google Colab Pro**: ~$10/month, better GPUs, longer sessions

### Paid Options
- **AWS EC2** (p3.2xlarge): V100 GPU, ~$3/hour
- **Google Cloud** (n1-standard-8 + T4): ~$0.50-1/hour
- **Paperspace Gradient**: Starting at $0.45/hour

### Cost Estimate for Cloud Training
- **Minimal model** (10k cells): $1-3
- **Default model** (50k cells): $5-15
- **Large model** (100k cells): $20-50

## Quick Start for Limited Hardware

1. **Start with MINIMAL config**:
```bash
# Edit configs/model_config.yaml to use minimal settings
python train.py --data_path data/pbmc3k.h5ad
```

2. **Monitor GPU usage**:
```bash
# In another terminal
watch -n 1 nvidia-smi
```

3. **If you run out of memory**:
   - Reduce batch_size in `configs/training_config.yaml`
   - Reduce n_genes in `configs/model_config.yaml`
   - Enable gradient checkpointing (requires code modification)

## Estimation Script

Run the included script to estimate requirements for your specific configuration:

```bash
python examples/estimate_requirements.py
```

This will show you:
- Exact parameter counts
- Memory requirements
- Estimated training times
- Hardware recommendations

## FAQ

**Q: Can I train on Apple Silicon (M1/M2/M3)?**
A: Yes! PyTorch supports MPS backend. Use `--device mps` when training. M1 Pro/Max/Ultra chips work well for MINIMAL and DEFAULT configs.

**Q: My training is very slow on GPU. Why?**
A: Check:
1. Data loading bottleneck (increase `num_workers`)
2. Small batch size (GPU underutilized)
3. Mixed precision not enabled
4. CPU-GPU transfer overhead

**Q: Can I pause and resume training?**
A: Yes! The trainer automatically saves checkpoints. Resume with:
```bash
python train.py --resume checkpoints/checkpoint_step_5000.pt
```

**Q: How much data do I need?**
A: Minimum:
- 5-10k cells for learning/testing
- 50-100k cells for decent performance
- 500k+ cells for best results
- Multiple datasets recommended for robust pre-training

**Q: Can I use multiple GPUs?**
A: Not currently implemented, but can be added with PyTorch DDP (Distributed Data Parallel). Let me know if you need this feature!

## Monitoring Training

### GPU Monitoring
```bash
# Real-time GPU usage
nvidia-smi -l 1

# Or use nvtop (more user-friendly)
nvtop
```

### Training Metrics
- WandB dashboard (if enabled)
- TensorBoard (logs saved to `logs/`)
- Console output every N steps

## Conclusion

**For most users**: A modern gaming PC with an RTX 3060 (12GB) or better and 16GB RAM is perfectly sufficient to train the DEFAULT configuration on typical scRNA-seq datasets (10-100k cells).

**Bottom line**: You don't need a server or cloud resources to get started. Local training is very feasible!
