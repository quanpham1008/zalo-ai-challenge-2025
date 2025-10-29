# 🚀 Traffic Buddy - Training Guide

Hướng dẫn chi tiết để chạy training cho production.

## 📋 Yêu cầu

- **GPU**: 2x A100 40GB (recommended) hoặc 1x A100 80GB
- **Python**: 3.10+
- **Dependencies**: Đã cài đặt qua `requirements.txt`

## 🎯 Các cách training

### 1. **Training với Ray Tune (Recommended cho Production)**

Tự động tìm kiếm hyperparameters tốt nhất:

```bash
# Chạy với cấu hình mặc định
bash scripts/train_production.sh

# Hoặc custom configuration
BASE_MODEL="Qwen/Qwen3-VL-8B-Instruct" \
NUM_SAMPLES=10 \
NUM_GPUS=2 \
bash scripts/train_production.sh
```

**Các tham số có thể override:**
- `BASE_MODEL`: Base model ID (default: Qwen/Qwen3-VL-8B-Instruct)
- `OUTPUT_DIR`: Nơi lưu checkpoints (default: ./checkpoints/lora_sft)
- `WANDB_PROJECT`: WandB project name (default: zalo_challenge)
- `NUM_SAMPLES`: Số lượng trials để tune (default: 8)
- `NUM_GPUS`: Số GPU sử dụng (default: 2)

### 2. **Training Manual (Single Trial)**

Chạy trực tiếp với Python:

```bash
cd /home/quanpv/project/traffic_buddy

python src/training/train.py \
    --base_model_id "Qwen/Qwen3-VL-8B-Instruct" \
    --output_dir "./checkpoints/lora_sft_manual" \
    --wandb_project "zalo_challenge" \
    --num_samples 1
```

### 3. **Training Simple (Không dùng Ray Tune)**

Sử dụng script đơn giản hơn, phù hợp cho testing:

```bash
python src/training/train_simple.py \
    --base_model_id "Qwen/Qwen3-VL-8B-Instruct" \
    --output_dir "./checkpoints/lora_simple" \
    --learning_rate 2e-4 \
    --num_epochs 3 \
    --lora_rank 256 \
    --batch_size 1 \
    --grad_accum_steps 4
```

## 📊 Monitoring Training

### WandB Dashboard
Tự động log vào WandB, xem tại:
```
https://wandb.ai/your-username/zalo_challenge
```

### Local Logs
Xem log files trong:
```bash
tail -f outputs/logs/training_*.log
```

### Ray Tune Results
Xem progress Ray Tune:
```bash
cd ray_results/training
tensorboard --logdir .
```

## 📁 Cấu trúc Output

```
checkpoints/
└── lora_sft/
    └── trial_lr0.0002_r128/
        ├── checkpoint-200/     # Intermediate checkpoint
        ├── checkpoint-400/     # ...
        ├── checkpoint-600/
        ├── checkpoint-800/
        ├── checkpoint-1000/
        ├── best/               # Best model (lowest eval_loss)
        └── final/              # Final model
```

## ⚙️ Hyperparameters Die

Ray Tune sẽ tự động tìm kiếm:

| Parameter | Search Space |
|-----------|--------------|
| Learning Rate | loguniform(1e-5, 5e-4) |
| LoRA Rank | [64, 128, 256] |
| LoRA Alpha | [16, 32] |
| LoRA Dropout | [0.05, 0.1] |
| Num Epochs | [1, 2, 3] |
| Batch Size | [1, 2] |
| Grad Accum | [4, 8] |
| Weight Decay | [0.0, 0.01, 0.1] |
| LR Scheduler | [cosine, linear, polynomial] |
| Early Stopping | [False, True] |
| Early Stopping Patience | [3, 5, 7] |

## 🔥 Optimizations Applied

### Training Stability
- ✅ Gradient clipping (max_grad_norm=1.0)
- ✅ Weight decay regularization
- ✅ Mixed precision training (bf16)

### Overfitting Prevention
- ✅ Early stopping callback
- ✅ Validation-based best model selection
- ✅ L2 regularization

### Training Efficiency
- ✅ Gradient checkpointing
- ✅ 4-bit quantization (optional)
- ✅ Ray Tune for hyperparameter search

### Model Quality
- ✅ Load best model at end
- ✅ Save based on eval_loss
- ✅ Multiple checkpoint saves

## 📝 Step-by-step Production Training

### Step 1: Prepare Environment
```bash
# Activate virtual environment
source .venv/bin/activate  # or your venv path

# Check GPU
nvidia-smi
```

### Step 2: Verify Dataset
```bash
# Check dataset exists
ls datasets/traffic_dataset/

# If not, build it
python src/data/dataset_builder.py
```

### Step 3: Start Training
```bash
# Production training với Ray Tune
bash scripts/train_production.sh
```

### Step 4: Monitor Progress
```bash
# Terminal 1: Watch logs
tail -f outputs/logs/training_*.log

# Terminal 2: Watch GPU
watch -n 5 nvidia-smi

# Browser: WandB dashboard
# https://wandb.ai/your-username/zalo_challenge
```

### Step 5: Check Results
```bash
# List all checkpoints
find checkpoints/lora_sft -name "best" -type d

# Evaluate best model
python src/evaluation/eval_vllm.py \
    --model_path checkpoints/lora_sft/trial_*/best
```

## 🐛 Troubleshooting

### Out of Memory (OOM)
- Giảm `batch_size` trong code
- Giảm `max_length` (hiện tại 1024)
- Enable `use_4bit=True` trong search space

### Training Too Slow
- Giảm `num_samples` (số trials)
- Tăng rá batch size nếu memory cho phép
- Disable early stopping cho vài trials đầu

### No Improvement
- Tăng learning rate range
- Tăng number of epochs
- Check data quality

## 📞 Support

Nếu gặp vấn đề:
1. Check log files trong `outputs/logs/`
2. Check Ray Tune errors trong `ray_results/`
3. Check WandB logs
4. Verify GPU memory: `nvidia-smi`

## 🎯 Expected Training Time

- **Single Trial**: ~2-4 hours (tùy vào số epochs và GPU)
- **Ray Tune (8 trials)**: ~16-32 hours với 2 GPU

## 📈 Expected Results

Với training tốt, bạn nên thấy:
- Training loss giảm dần và ổn định
- Eval loss thấp hơn training loss (no overfitting)
- Best model được lưu trong `best/` folder
- WandB dashboard hiển thị metrics tốt

