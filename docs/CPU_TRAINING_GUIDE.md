# CPU-Only Training Guide

## For Laptops Without Dedicated GPU (like Lenovo T480s)

If you have a laptop with **no NVIDIA GPU** (like the Lenovo T480s, ThinkPad, MacBook, etc.), you can still train the model on CPU. Here's what you need to know:

## Your Hardware: Lenovo T480s (2019)
- ✅ **CPU**: Intel Core i5/i7 8th gen (4 cores, 8 threads) - Good!
- ✅ **RAM**: 16GB - Perfect!
- ❌ **GPU**: Intel UHD 620 (integrated) - Won't help PyTorch
- ⚠️ **Result**: CPU-only training

## Realistic Expectations

### CPU Training Times (Ultra-Minimal Config)

| Dataset Size | Epochs | Estimated Time |
|--------------|--------|----------------|
| 3k cells | 20 | 2-4 hours |
| 5k cells | 20 | 4-6 hours |
| 10k cells | 20 | 8-12 hours |
| 3k cells | 50 | 6-12 hours |

**Note**: These are estimates for a modern quad-core CPU (i5/i7 8th gen+)

### What Will Happen
- ⚠️ Laptop will get **warm** (50-70°C is normal)
- 🔊 **Fan will run** continuously (loud)
- 🔋 **Battery will drain** quickly - keep plugged in!
- 🐌 Laptop will be **slower** for other tasks
- ⏸️ You can **pause/resume** training (Ctrl+C)

## Quick Start for Your T480s

### 1. Run the CPU-Optimized Training:

```bash
python train_cpu.py
```

This uses:
- **Ultra-small model**: 500 genes, 2 layers (~1-2M parameters)
- **Small dataset**: 3k cells automatically
- **Few epochs**: 20 (enough to see it working)
- **CPU optimizations**: All threads enabled

**Expected time**: 2-4 hours

### 2. Monitor While Training

**Check CPU usage** (should be 90-100%):
```bash
# Linux/Mac
htop

# Windows Task Manager
Ctrl+Shift+Esc → Performance tab
```

**Watch temperature** (use HWMonitor on Windows or lm-sensors on Linux):
- CPU: 60-80°C is normal under load
- Above 90°C: Consider better cooling/reduce workload

### 3. Pause and Resume

If you need to stop:
```
Ctrl+C  # Pauses training
```

The model will save a checkpoint automatically. Resume with:
```bash
python train_cpu.py --resume checkpoints_cpu/checkpoint_step_XXX.pt
```

## CPU Optimization Tips

### 1. Close Other Programs
Free up RAM and CPU:
- Close browser tabs
- Stop background apps
- Disable antivirus during training (if safe)

### 2. Use All CPU Cores
The script automatically sets this, but you can verify:
```python
import torch
print(f"Using {torch.get_num_threads()} threads")
# Should show 8 for your i7
```

### 3. Keep Laptop Cool
- Use on a hard surface (not bed/couch)
- Consider a laptop cooling pad
- Ensure vents aren't blocked
- Maybe elevate the back slightly

### 4. Power Settings
**Windows**:
- Control Panel → Power Options → High Performance
- Disable sleep/hibernate during training

**Linux**:
```bash
# Prevent sleep
systemctl mask sleep.target suspend.target hibernate.target
```

## Better Alternative: Free Cloud GPUs

Honestly, for serious work, I recommend free cloud options:

### Google Colab (FREE)
- **GPU**: Tesla T4 (16GB VRAM)
- **Training time**: 20-30 minutes (vs 2-4 hours on CPU)
- **Limit**: 12-hour sessions
- **Cost**: Free

**How to use**:
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Upload your code
3. Runtime → Change runtime type → GPU (T4)
4. Run training

### Kaggle Notebooks (FREE)
- **GPU**: P100 (16GB VRAM)
- **Limit**: 30 hours/week
- **Cost**: Free

### Paperspace Gradient (FREE tier)
- **GPU**: Various (M4000+)
- **Limit**: Limited hours
- **Cost**: Free tier available

## Comparison: Your Laptop vs Cloud

| | T480s CPU | Colab GPU | Kaggle GPU |
|---|---|---|---|
| **Speed** | 1x | 20-30x | 25-35x |
| **Cost** | $0 | $0 | $0 |
| **Time (3k cells)** | 2-4 hours | 5-10 min | 5-10 min |
| **Laptop heat** | High | None | None |
| **Convenience** | Local | Need internet | Need internet |

## My Recommendation for You

### For Learning/Testing:
1. **Try CPU training** with `train_cpu.py` (2-4 hours)
2. See it work, understand the process
3. Get a feel for the model

### For Real Work:
1. **Use Google Colab** (free)
2. Much faster (30 min vs 4 hours)
3. No wear on your laptop
4. Better for experimentation

### Your Call:
- **Time-rich, no internet**: Use CPU
- **Want results quickly**: Use Colab
- **Serious research**: Consider cloud or desktop GPU

## CPU Training Checklist

Before starting CPU training on your T480s:

- [ ] Laptop plugged into power
- [ ] Close unnecessary programs
- [ ] Ensure good ventilation
- [ ] Set power mode to "High Performance"
- [ ] Have 4-6 hours available (or plan to pause)
- [ ] Temperature monitoring app installed (optional)
- [ ] Coffee ready ☕

## Troubleshooting

**Laptop too hot (>85°C)**:
- Reduce to even smaller model
- Lower batch size further
- Take breaks between epochs
- Improve cooling

**Training too slow (>8 hours for 3k cells)**:
- Check CPU usage (should be 90%+)
- Close background apps
- Verify thread count: `torch.get_num_threads()`
- Consider cloud GPU instead

**Out of RAM**:
- Reduce batch_size to 2
- Reduce n_genes to 300
- Close all other programs

**Laptop freezing**:
- Reduce num_workers to 0 (already set)
- Lower batch_size
- Add swap space (Linux)

## The Bottom Line

**YES, your T480s can train this model!** But it will take 2-6 hours for a small experiment.

**For serious work**, I strongly recommend using **Google Colab** (free GPU) instead. Your laptop will thank you, and you'll get results 20x faster.

Want to try Colab? I can create a notebook for you!
