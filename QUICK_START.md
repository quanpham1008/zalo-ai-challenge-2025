# 🚀 Quick Start - Production Training

## ⚡ Chạy nhanh nhất (Recommended)

```bash
# 1. Đi đến project directory
cd /home/quanpv/project/traffic_buddy

# 2. Activate virtual environment (nếu có)
source .venv/bin/activate

# 3. Chạy training production
bash scripts/train_production.sh
```

**Done!** Training sẽ tự động:
- ✅ Build dataset nếu chưa có
- ✅ Tìm hyperparameters tốt nhất với Ray Tune
- ✅ Save checkpoints vào `checkpoints/lora_sft/`
- ✅ Log metrics vào WandB
- ✅ Save best model cuối cùng

---

## 🎛️ Custom Configuration

Muốn thay đổi cấu hình? Set environment variables:

```bash
# Dùng model nhỏ hơn (nhanh hơn, ít RAM hơn)
BASE_MODEL="Qwen/Qwen3-VL-2B-Instruct" bash scripts/train_production.sh

# Chạy nhiều trials hơn
NUM_SAMPLES=20 bash scripts/train_production.sh

# Đổi output directory
OUTPUT_DIR="./checkpoints/my_training" bash scripts/train_production.sh
```

---

## 📊 Monitor Training

### Option 1: WandB Dashboard (Best)
Tự động log vào WandB, mở browser xem real-time:
```
https://wandb.ai/your-username/zalo_challenge
```

### Option 2: Check Log Files
```bash
# Xem log gần nhất
tail -f outputs/logs/training_*.log

# Hoặc list all logs
ls -lth outputs/logs/
```

### Option 3: GPU Monitoring
```bash
watch -n 5 nvidia-smi
```

---

## 📁 Kết quả Training

### Best Model Location
```bash
# Tìm best model
find checkpoints/lora_sft -name "best" -type d
```

### Model Structure
```
checkpoints/lora_sft/
└── trial_lr0.0002_r128/
    ├── best/              # ⭐ Model tốt nhất (eval_loss thấp nhất)
    ├── final/             # Model cuối cùng
    └── checkpoint-*/      # Intermediate checkpoints
```

---

## 🔍 Check Training Status

### Quick Check
```bash
# Xem GPU usage
nvidia-smi

# Xem log file
tail -20 outputs/logs/training_*.log

# Check checkpoints đã tạo
ls checkpoints/lora_sft/
```

---

## ⏱️ Thời gian Training

- **Single Trial**: 2-4 hours
- **8 Trials (default)**: 16-32 hours
- **20 Trials**: 40-80 hours

*Thời gian thực tế phụ thuộc vào GPU và dataset size*

---

## 🎯 Next Steps

Sau khi training xong:

1. **Evaluate model**:
```bash
python src/evaluation/eval_vllm.py
```

2. **Submit result**:
```bash
# Generate submission file
python src/evaluation/eval_vllm.py
```

3. **Compare models**:
```bash
# Xem metrics trong WandB dashboard
```

---

## ❓ Gặp vấn đề?

1. **Out of Memory?** 
   - Giảm batch size trong code
   - Giảm num_frames

2. **Training chậm?**
   - Giảm NUM_SAMPLES
   - Tắt early stopping cho thử nghiệm

3. **Không có GPU?**
   - Dùng train_simple.py với CPU
   - Hoặc dùng Google Colab

---

## 📚 Chi tiết hơn?

Xem [TRAINING_GUIDE.md](TRAINING_GUIDE.md) để biết thêm chi tiết về:
- Hyperparameter tuning
- Custom configurations  
- Advanced options
- Troubleshooting

