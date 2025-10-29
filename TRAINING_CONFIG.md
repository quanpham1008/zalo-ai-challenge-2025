# 🎯 Training Configuration

**Date**: October 29, 2025  
**Status**: ✅ CONFIGURED FOR PRODUCTION

---

## 🔧 Current Training Configuration

### **Model Settings** 🤖

```yaml
Base Model: Qwen/Qwen3-VL-8B-Instruct
Model Size: 8 Billion parameters
Architecture: Vision-Language Model
LoRA Enabled: Yes (Low-Rank Adaptation)
```

**Why 8B Model?**
- ✅ Better performance than 2B
- ✅ Higher accuracy on complex tasks
- ✅ Better understanding of Vietnamese traffic scenes
- ⚠️ Requires more GPU memory (~40GB with LoRA)

---

### **WandB Configuration** 📊

```yaml
Project Name: zalo_challenge
Tracking: Enabled
Dashboard: https://wandb.ai/your-username/zalo_challenge
```

**What's Tracked:**
- Training & validation loss
- Learning rate schedule
- Hyperparameter values
- Model checkpoints
- GPU metrics

---

### **Hardware Requirements** 💻

For **Qwen3-VL-8B-Instruct**:

```yaml
Recommended GPU: 2x A100 40GB or 1x A100 80GB
Minimum GPU: 1x A100 40GB (with 4-bit quantization)
RAM: 64GB+ system RAM
Storage: 100GB+ for model + checkpoints
```

**Memory Usage Estimates:**
- Model: ~16GB (bf16)
- LoRA adapters: ~2GB
- Training batch: ~20-25GB
- **Total**: ~40GB per GPU

---

### **Training Hyperparameters** ⚙️

**Ray Tune Search Space:**
```yaml
Learning Rate: loguniform(1e-5, 5e-4)
LoRA Rank: [64, 128, 256]
LoRA Alpha: [16, 32]
LoRA Dropout: [0.05, 0.1]
Epochs: [1, 2, 3]
Batch Size: [1, 2]
Gradient Accumulation: [4, 8]
Weight Decay: [0.0, 0.01, 0.1]
LR Scheduler: [cosine, linear, polynomial]
Early Stopping: [False, True]
```

**Default Settings (Simple Training):**
```yaml
Learning Rate: 2e-4
LoRA Rank: 256
Batch Size: 1
Gradient Accumulation: 4
Epochs: 3
```

---

### **Output Configuration** 📁

```yaml
Checkpoints: ./checkpoints/lora_sft/
Logs: ./outputs/logs/
Predictions: ./outputs/predictions/
Dataset: ./datasets/traffic_dataset/
```

**Checkpoint Strategy:**
- Save every 200 steps
- Keep last 5 checkpoints
- Save best model based on eval_loss
- Final model saved at end

---

## 🚀 Quick Start Commands

### **Production Training (Recommended)**
```bash
cd /home/quanpv/project/traffic_buddy
source .venv/bin/activate
bash scripts/train_production.sh
```

**This will:**
- Use **Qwen3-VL-8B-Instruct** model
- Run 8 Ray Tune trials
- Log to **zalo_challenge** on WandB
- Save to `checkpoints/lora_sft/`

---

### **Simple Training (For Testing)**
```bash
cd /home/quanpv/project/traffic_buddy
source .venv/bin/activate
python src/training/train_simple.py
```

**Default behavior:**
- Uses **Qwen3-VL-8B-Instruct**
- Single training run (no Ray Tune)
- 3 epochs
- LoRA rank 256

---

### **Custom Configuration**

**Use 2B model (faster, less memory):**
```bash
BASE_MODEL="Qwen/Qwen3-VL-2B-Instruct" bash scripts/train_production.sh
```

**Run more trials:**
```bash
NUM_SAMPLES=20 bash scripts/train_production.sh
```

**Custom output directory:**
```bash
OUTPUT_DIR="./checkpoints/experiment_001" bash scripts/train_production.sh
```

**Change WandB project:**
```bash
WANDB_PROJECT="my_experiment" bash scripts/train_production.sh
```

---

## 📊 Expected Performance

### **Training Time** ⏱️

**With 8B Model:**
```
Single Trial (1 epoch): 3-5 hours
Full Training (8 trials): 24-40 hours
Quick Test (1 trial, 1 epoch): 3-5 hours
```

**With 2B Model (for comparison):**
```
Single Trial (1 epoch): 1-2 hours
Full Training (8 trials): 8-16 hours
Quick Test (1 trial, 1 epoch): 1-2 hours
```

*Times based on A100 40GB GPU*

---

### **Memory Usage** 💾

**8B Model:**
- Training: ~38-42GB per GPU
- Inference: ~18-20GB per GPU
- With 4-bit: ~25-30GB per GPU

**2B Model (for comparison):**
- Training: ~20-25GB per GPU
- Inference: ~8-10GB per GPU
- With 4-bit: ~12-15GB per GPU

---

## 🎯 Dataset Configuration

```yaml
Total Samples: 1,490
Training Samples: 1,341 (90%)
Validation Samples: 149 (10%)
Video Path: data/raw/videos/
Frames per Video: 8
Frame Size: 448x448
```

---

## ✅ Pre-Training Checklist

Before starting training, verify:

- [ ] ✅ GPU available (`nvidia-smi`)
- [ ] ✅ Virtual environment activated
- [ ] ✅ Dataset exists (`datasets/traffic_dataset/`)
- [ ] ✅ Videos accessible (`data/raw/videos/`)
- [ ] ✅ WandB logged in (`wandb login`)
- [ ] ✅ Sufficient disk space (100GB+)
- [ ] ✅ Validation passed (`python scripts/validate_setup.py`)

**Run validation:**
```bash
python scripts/validate_setup.py
```

---

## 🔍 Monitoring Training

### **Real-time Monitoring**

**WandB Dashboard:**
```
https://wandb.ai/your-username/zalo_challenge
```

**Local Logs:**
```bash
tail -f outputs/logs/training_*.log
```

**GPU Monitoring:**
```bash
watch -n 5 nvidia-smi
```

---

### **Key Metrics to Watch**

1. **Training Loss**: Should decrease steadily
2. **Validation Loss**: Should be lower than training (no overfitting)
3. **GPU Memory**: Should stay under 40GB
4. **GPU Utilization**: Should be 90%+

**Warning Signs:**
- ⚠️ Loss not decreasing → Increase learning rate
- ⚠️ Training loss << Val loss → Overfitting (reduce epochs)
- ⚠️ OOM errors → Reduce batch size or enable 4-bit
- ⚠️ Low GPU util → Increase batch size

---

## 📈 Expected Results

**Good Training Run:**
```
Training Loss: 0.5 → 0.2 (decreasing)
Validation Loss: 0.6 → 0.25 (decreasing)
GPU Utilization: 95%+
Training Time: 3-5 hours per epoch
```

**Best Model:**
- Saved in `checkpoints/lora_sft/trial_*/best/`
- Selected based on lowest validation loss
- Automatically loaded at end of training

---

## 🎓 Next Steps After Training

1. **Find best model:**
   ```bash
   find checkpoints/lora_sft -name "best" -type d
   ```

2. **Evaluate model:**
   ```bash
   python src/evaluation/eval_vllm.py \
       --model_path checkpoints/lora_sft/trial_*/best
   ```

3. **Generate submission:**
   ```bash
   python src/evaluation/eval_qwen3_vl_benchmark.py
   ```

4. **Analyze results:**
   - Check WandB dashboard for metrics
   - Compare different trials
   - Select best hyperparameters

---

## 🔗 Related Documentation

- [QUICK_START.md](QUICK_START.md) - Fast training guide
- [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - Detailed training guide
- [FIXES_SUMMARY.md](FIXES_SUMMARY.md) - What was fixed
- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - Project layout

---

## 📞 Support

**If you encounter issues:**

1. Check log files: `outputs/logs/training_*.log`
2. Check WandB: `https://wandb.ai/your-username/zalo_challenge`
3. Run validation: `python scripts/validate_setup.py`
4. Check GPU: `nvidia-smi`

**Common Solutions:**
- OOM → Enable 4-bit or reduce batch size
- Slow training → Check GPU utilization
- Poor results → Increase epochs or adjust learning rate

---

**Configuration Version**: 1.0  
**Model**: Qwen3-VL-8B-Instruct  
**WandB Project**: zalo_challenge  
**Status**: ✅ Ready to Train

