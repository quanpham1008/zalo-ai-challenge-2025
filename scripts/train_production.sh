#!/bin/bash
# ============================================================
# Traffic Buddy - Production Training Script
# ============================================================

set -e  # Exit on error
set -o pipefail  # Fail on pipeline errors (so tee doesn't hide failures)

echo "🚀 Starting Traffic Buddy Production Training..."
echo "=================================================="

# ============================================================
# Configuration
# ============================================================
PROJECT_ROOT="/home/quanpv/project/traffic_buddy"
cd "$PROJECT_ROOT"

# Model configuration
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"  # Can override with env var
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/lora_sft}"
WANDB_PROJECT="${WANDB_PROJECT:-zalo_challenge}"

# Ray Tune configuration
NUM_SAMPLES="${NUM_SAMPLES:-8}"  # Number of hyperparameter trials
NUM_GPUS="${NUM_GPUS:-1}"        # Number of GPUs to use

# Logging
LOG_DIR="./outputs/logs"
RAY_DIR="./outputs/ray_results"
mkdir -p "$LOG_DIR"
mkdir -p "$RAY_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/training_${TIMESTAMP}.log"

echo "Configuration:"
echo "  Model: $BASE_MODEL"
echo "  Output: $OUTPUT_DIR"
echo "  WandB Project: $WANDB_PROJECT"
echo "  Ray Trials: $NUM_SAMPLES"
echo "  GPUs: $NUM_GPUS"
echo "  Log: $LOG_FILE"
echo ""

# ============================================================
# Pre-training checks
# ============================================================
echo "📋 Pre-training checks..."
echo ""

# Check if dataset exists (check both locations), and migrate if needed
if [ -d "datasets/traffic_dataset" ] && [ ! -d "data/processed" ]; then
    echo "➡️  Found legacy dataset in datasets/traffic_dataset. Migrating to data/processed..."
    "$PYTHON_BIN" scripts/migrate_dataset_path.py || true
fi

if [ ! -d "data/processed" ] && [ ! -d "datasets/traffic_dataset" ]; then
    echo "⚠️  Dataset not found. It will be built from train.json during training..."
elif [ -d "data/processed" ]; then
    echo "✅ Found dataset in data/processed/"
elif [ -d "datasets/traffic_dataset" ]; then
    echo "✅ Found dataset in datasets/traffic_dataset/"
fi

# Check if training data exists
if [ ! -f "data/raw/train.json" ]; then
    echo "❌ Error: data/raw/train.json not found!"
    echo "Please ensure your training data is in the correct location."
    exit 1
fi

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Status:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
else
    echo "⚠️  nvidia-smi not found. Assuming no GPU available."
fi
echo ""

# Migrate legacy Ray results directory if present
if [ -d "./ray_results" ] && [ ! -d "$RAY_DIR" ]; then
    echo "➡️  Found legacy Ray results at ./ray_results. Moving to $RAY_DIR ..."
    mkdir -p "$(dirname "$RAY_DIR")"
    mv ./ray_results "$RAY_DIR"
    echo "✅ Ray results moved to $RAY_DIR"
fi

# ============================================================
# Resolve Python interpreter
# ============================================================
if [ -x ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
else
    echo "❌ Error: No Python interpreter found (python/python3)."
    exit 1
fi

# ============================================================
# Training
# ============================================================
echo "🚂 Starting training..."
echo ""

# Ensure we're in the project root directory
cd "$PROJECT_ROOT"

# Run training with proper Python path setup
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
"$PYTHON_BIN" src/training/train.py \
    --base_model_id "$BASE_MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --wandb_project "$WANDB_PROJECT" \
    --num_samples "$NUM_SAMPLES" \
    2>&1 | tee "$LOG_FILE"

# Capture python's exit code from the pipeline
TRAIN_EXIT_CODE=${PIPESTATUS[0]}

# ============================================================
# Post-training summary
# ============================================================
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ Training completed successfully!"
    echo ""
    echo "📁 Checkpoints saved to: $OUTPUT_DIR"
    echo "📊 WandB dashboard: https://wandb.ai/your-org/$WANDB_PROJECT"
    echo "📝 Full log: $LOG_FILE"
    echo ""
    echo "Best model locations:"
    find "$OUTPUT_DIR" -name "best" -type d 2>/dev/null | head -5
else
    echo ""
    echo "❌ Training failed! Check log: $LOG_FILE"
    exit 1
fi

