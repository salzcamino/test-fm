# Google Colab Training Guide

## Quick Start (5 minutes)

Train your scRNA-seq foundation model using **FREE Google Colab GPU** - it's **20-30x faster** than CPU training!

## What You Get

- ✅ **Free GPU**: Tesla T4 (16GB VRAM)
- ✅ **Fast Training**: 5-15 minutes (vs 2-4 hours on laptop)
- ✅ **No Installation**: Everything runs in browser
- ✅ **Easy Download**: Get your trained model after
- ✅ **No Laptop Wear**: Runs on Google's servers

## Step-by-Step Instructions

### 1. Open the Notebook

**Option A: Direct Link** (recommended)
- Click here: [Open in Colab](https://colab.research.google.com/github/yourusername/scrna-foundation-model/blob/main/notebooks/Google_Colab_Training.ipynb)

**Option B: Manual Upload**
- Go to [colab.research.google.com](https://colab.research.google.com)
- Click **File** → **Upload notebook**
- Select `notebooks/Google_Colab_Training.ipynb` from this repository

### 2. Enable GPU (IMPORTANT!)

- Click **Runtime** → **Change runtime type**
- Under "Hardware accelerator", select **GPU**
- Click **Save**

![Enable GPU](https://i.imgur.com/XQlrK2p.png)

**Verify GPU is working**: Run the first cell to check GPU availability.

### 3. Run All Cells

**Easy way**:
- Click **Runtime** → **Run all**
- Wait 10-15 minutes while it trains
- Come back to see results!

**Step-by-step way**:
- Click each cell and press `Shift+Enter`
- Read the explanations as you go

### 4. Download Your Trained Model

At the end, the notebook will create a ZIP file with:
- Trained model weights
- Cell embeddings
- UMAP visualization
- Gene importance plot

Just click the download button that appears!

## What the Notebook Does

### Automatic Steps:

1. **Checks GPU** (verify T4/T4 GPU is available)
2. **Installs dependencies** (scanpy, PyTorch, etc.) - ~2 min
3. **Downloads PBMC3k data** (3,000 blood cells) - ~30 sec
4. **Preprocesses data** (QC, normalization, HVG selection) - ~1 min
5. **Creates model** (~25M parameters)
6. **Trains for 30 epochs** - ~5-10 min
7. **Generates visualizations** (UMAP, gene importance)
8. **Saves everything** for download

### Total Time: 10-15 minutes

## Training Configuration

Default settings (optimized for Colab T4):

```python
Model:
  - 2000 genes (highly variable)
  - 4 transformer layers
  - 8 attention heads
  - ~25M parameters

Training:
  - 30 epochs
  - Batch size: 64
  - Learning rate: 1e-4
  - AdamW optimizer
  - Cosine LR scheduling
```

You can edit these in the notebook!

## Using Your Own Data

### Upload Your Data to Colab:

```python
# Add a new cell and run:
from google.colab import files
uploaded = files.upload()  # Click "Choose Files" and select your .h5ad file

# Load your data
import anndata as ad
adata = ad.read_h5ad('your_data.h5ad')

# Continue with preprocessing...
```

### Supported Formats:
- `.h5ad` (AnnData) - recommended
- `.loom` (Loom)
- `.csv` / `.tsv` (expression matrices)
- `.h5` (10X HDF5)

## Tips for Success

### 1. Monitor GPU Usage

```python
# Add this cell to check GPU memory:
!nvidia-smi
```

Should show:
- GPU: Tesla T4 (or T4, A100)
- Memory: Some usage (~3-4GB during training)

### 2. Session Limits

**Free Tier**:
- 12-hour session limit (more than enough!)
- May disconnect after 90 min idle
- **Tip**: Keep the browser tab active

**Colab Pro** ($10/month):
- Longer sessions (24h)
- Better GPUs (A100, V100)
- Priority access

### 3. Save Your Work

The notebook auto-downloads results, but you can also:

```python
# Save to Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Save model
torch.save(model.state_dict(), '/content/drive/MyDrive/my_model.pt')
```

### 4. If Session Disconnects

Don't worry! The notebook saves checkpoints:

```python
# Resume training from checkpoint
trainer.load_checkpoint('checkpoints_colab/checkpoint_step_500.pt')
trainer.train()
```

## Troubleshooting

### "No GPU Available"

**Problem**: First cell shows "No GPU detected"

**Solution**:
1. Runtime → Change runtime type → GPU
2. Restart runtime
3. Run first cell again

### "Out of Memory" Error

**Problem**: GPU runs out of memory during training

**Solutions**:
- Reduce `batch_size` from 64 to 32 or 16
- Reduce `n_genes` from 2000 to 1000
- Reduce model size (smaller `hidden_dim`)

```python
# In the training config cell, change:
config['training']['batch_size'] = 32  # or 16
```

### "Session Disconnected"

**Problem**: Colab disconnects after 90 minutes

**Solutions**:
- Keep browser tab active
- Use this JavaScript in console to prevent disconnect:

```javascript
function ClickConnect(){
  console.log("Clicking");
  document.querySelector("colab-connect-button").click()
}
setInterval(ClickConnect, 60000)
```

### Training is Slow

**Problem**: Taking longer than 15 minutes

**Check**:
1. GPU is enabled (Runtime → Change runtime type)
2. GPU is being used (run `!nvidia-smi`)
3. Not using too small batch size

**Expected speeds**:
- **With GPU**: 5-15 minutes
- **Without GPU**: 2-4 hours (not recommended!)

## Costs

### Free Tier

- **Cost**: $0
- **GPU**: T4 (16GB)
- **Sessions**: 12 hours
- **Enough for**: Training this model many times!

### Colab Pro ($10/month)

- **Better GPUs**: A100, V100
- **Longer sessions**: 24 hours
- **Priority access**: Skip wait times
- **Worth it if**: Training large models frequently

### Colab Pro+ ($50/month)

- **Best GPUs**: More A100 access
- **Background execution**: Keep training when browser closed
- **Worth it if**: Serious research with large datasets

## Comparison: Colab vs Local

| | Google Colab | Your T480s CPU | Desktop RTX 3060 |
|---|---|---|---|
| **Cost** | $0/month | $0 | ~$400 GPU |
| **Speed** | 5-15 min | 2-4 hours | 10-20 min |
| **Setup** | None | None | Install drivers |
| **Convenience** | Browser only | Always available | Always available |
| **Limitations** | 12h sessions | Slow | None |

## Advanced: Customize Training

### Train Longer (Better Results)

```python
# Change in training config:
config['training']['num_epochs'] = 100  # Instead of 30
```

### Use Larger Model

```python
# When creating model:
model = scRNAFoundationModel(
    n_genes=5000,        # More genes
    hidden_dim=512,      # Larger model
    num_layers=6,        # More layers
    # ... other params
)
```

### Enable WandB Logging

```python
# Install wandb
!pip install wandb

# Login (get key from wandb.ai)
import wandb
wandb.login()

# Enable in config
config['training']['use_wandb'] = True
config['training']['wandb_project'] = 'my-scrna-project'
```

## What to Do After Training

### 1. Analyze Results

The notebook shows:
- UMAP visualization of cell embeddings
- Clustering metrics (ARI, NMI, Silhouette)
- Top important genes

### 2. Fine-Tune for Your Task

```python
# Add classification head
classifier = torch.nn.Linear(256, num_cell_types)

# Fine-tune on labeled data
for batch in labeled_loader:
    embeddings = model.get_cell_embeddings(batch['input_ids'])
    predictions = classifier(embeddings)
    loss = criterion(predictions, batch['labels'])
    # ... train
```

### 3. Use Embeddings for Downstream Analysis

```python
# Load saved embeddings
embeddings = np.load('cell_embeddings.npy')

# Use for:
# - Cell type classification
# - Batch effect correction
# - Trajectory analysis
# - Integration with other data
```

## FAQ

**Q: Can I use my own data?**
A: Yes! Upload your `.h5ad` file using the upload cell in the notebook.

**Q: How long can I use Colab for free?**
A: Sessions last up to 12 hours. You can start new sessions anytime.

**Q: Will my model be saved?**
A: Yes, but only for the session. Download it before closing! Or save to Google Drive.

**Q: Can I train multiple models?**
A: Yes! Run the notebook multiple times with different configs.

**Q: Is my data private?**
A: Colab is reasonably private, but don't upload sensitive patient data. Check your institution's policies.

**Q: What if I need more than 12 hours?**
A: This model trains in 5-15 min, so 12h is plenty. For longer jobs, consider Colab Pro or local GPU.

## Next Steps

1. **Try the notebook now**: [Open in Colab](https://colab.research.google.com/github/yourusername/scrna-foundation-model/blob/main/notebooks/Google_Colab_Training.ipynb)

2. **Experiment with your data**: Upload your own `.h5ad` files

3. **Share results**: Post your UMAP plots and tell us what you found!

4. **Star the repo** ⭐ if this was helpful!

---

## Need Help?

- 📖 Check the [main README](../README.md)
- 🐛 Report issues on [GitHub](https://github.com/yourusername/scrna-foundation-model/issues)
- 💬 Ask questions in Discussions

Happy training! 🚀
