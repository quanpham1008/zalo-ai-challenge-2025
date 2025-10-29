# Traffic Buddy - Project Structure Guide

## Current Structure

```
traffic_buddy/
├── src/                              # Source code (Python packages)
│   ├── __init__.py
│   │
│   ├── data/                         # Xử lý dữ liệu (Data pipeline)
│   │   ├── __init__.py
│   │   ├── dataset_builder.py        # Load & format dataset
│   │   ├── video_sampler.py          # Trích khung hình video
│   │   ├── scene_change_detector.py  # Phát hiện thay đổi cảnh
│   │   └── processor_collate.py      # Processor + Collator cho training
│   │
│   ├── models/                       # Model & Adapter (LoRA, etc.)
│   │   ├── __init__.py
│   │   ├── model_utils.py            # Helper load model/processor
│   │   └── lora_adapter.py           # Fine-tuning adapter (LoRA)
│   │
│   ├── training/                     # Script training
│   │   ├── __init__.py
│   │   ├── train_simple.py           # Fine-tune cơ bản
│   │   ├── train_ray.py              # Hyperparameter tuning (Ray)
│   │   ├── trainer_utils.py          # Helper training loop, loss, metrics
│   │   └── callbacks.py              # Custom logging/checkpoint callback
│   │
│   ├── evaluation/                   # Đánh giá mô hình
│   │   ├── __init__.py
│   │   ├── eval_qwen3vl.py           # Evaluate model Qwen3-VL
│   │   ├── eval_vllm.py              # Evaluate vLLM serving
│   │   ├── eval_benchmark.py         # Benchmark scripts
│   │   ├── metrics.py                # Tính điểm chính xác / latency
│   │   └── prompt_templates.py       # Template prompt thống nhất
│   │
│   └── utils/                        # Các công cụ chung
│       ├── __init__.py
│       ├── logging_utils.py
│       ├── file_utils.py
│       ├── visualization.py          # Có thể vẽ plot, frame debug
│       └── constants.py
│
│
├── data/                             # Dữ liệu (đầu vào & đầu ra)
│   ├── raw/                          # Dữ liệu thô (video, JSON gốc)
│   │   ├── train.json
│   │   ├── public_test.json
│   │   └── videos/
│   │
│   ├── processed/                    # Dữ liệu sau khi xử lý (HuggingFace)
│   │   ├── train/
│   │   └── validation/
│   │
│   ├── public_test/                  # Test data cho inference
│   │   ├── public_test.json
│   │   └── videos/
│   │
│   └── metadata/                     # Ví dụ fps, scene index, cached frames
│
│
├── checkpoints/                      # Model checkpoints
│   ├── base_models/                  # Model gốc (VD: Qwen2-VL)
│   └── finetuned/                    # Model fine-tune (VD: lora_sft/)
│
├── scripts/                          # Shell / Python script tiện ích
│   ├── download_videos.py
│   ├── prepare_dataset.py
│   ├── train_production.sh
│   └── evaluate_submission.sh
│
├── notebooks/                        # Notebook demo hoặc phân tích
│   ├── data_preview.ipynb
│   ├── training_analysis.ipynb
│   └── model_inference.ipynb
│
├── outputs/                          # Output inference / submission
│   ├── logs/
│   ├── predictions/
│   └── benchmark_results/
│
├── tests/                            # Unit test
│   ├── test_data_pipeline.py
│   ├── test_video_sampling.py
│   └── test_trainer.py
│
├── requirements.txt
├── README.md
├── PROJECT_STRUCTURE.md
├── QUICK_START.md
└── TRAINING_GUIDE.md
```

## Structure Status & Improvements

### ✅ Completed
1. **Data Organization**: ✓ Separated `raw/`, `processed/`, `public_test/`
2. **Source Code Structure**: ✓ Organized into `data/`, `models/`, `training/`, `evaluation/`, `utils/`
3. **Scripts Directory**: ✓ Created for utility scripts
4. **Outputs Directory**: ✓ Organized `logs/`, `predictions/`, `benchmark_results/`
5. **Checkpoints**: ✓ Separated `base_models/` and `finetuned/`

## Key Files Reference

### Data Processing
- `src/data/dataset_builder.py` - Build HuggingFace dataset from JSON
- `src/data/video_sampler.py` - Extract frames from videos
- `src/data/processor_collate.py` - Process and collate data for training
- `src/data/scene_change_detector.py` - Detect scene changes in videos

### Model & Training
- `src/models/model_utils.py` - Load model and processor (to implement)
- `src/models/lora_adapter.py` - LoRA adapter configuration (to implement)
- `src/training/train_simple.py` - Simple training script
- `src/training/train_ray.py` - Ray Tune hyperparameter search (currently train.py)
- `src/training/trainer_utils.py` - Training utilities (to implement)
- `src/training/callbacks.py` - Custom callbacks (to implement)

### Evaluation
- `src/evaluation/eval_qwen3vl.py` - Standard evaluation
- `src/evaluation/eval_vllm.py` - vLLM-based evaluation
- `src/evaluation/eval_benchmark.py` - Benchmark evaluation (currently eval_qwen3_vl_benchmark.py)
- `src/evaluation/prompt_templates.py` - Prompt templates (currently in training/)
- `src/evaluation/metrics.py` - Metrics calculation (to implement)

### Utilities
- `src/utils/logging_utils.py` - Logging helpers
- `src/utils/file_utils.py` - File I/O utilities
- `src/utils/constants.py` - Project constants ✅ (centralized path management)
- `src/utils/visualization.py` - Visualization tools (to implement)

### Scripts
- `scripts/download_videos.py` - Download videos from Zalo AI Challenge
- `scripts/train_production.sh` - Production training script ✅ (fixed paths)
- `scripts/validate_setup.py` - Setup validation script ✅ (new)
- `scripts/prepare_dataset.py` - Prepare dataset (to implement)
- `scripts/evaluate_submission.sh` - Evaluation script (to implement)

### Data Files
- `data/raw/train.json` - Training annotations
- `data/raw/videos/` - Training videos
- `data/processed/train/` - Processed training dataset
- `data/processed/validation/` - Processed validation dataset
- `data/public_test/public_test.json` - Test annotations
- `data/public_test/videos/` - Test videos

### Output Files
- `outputs/predictions/` - Model predictions
- `outputs/benchmark_results/` - Benchmark results and submissions
- `outputs/logs/` - Evaluation logs
- `checkpoints/finetuned/` - Fine-tuned model checkpoints
- `checkpoints/base_models/` - Base model cache

---

**Last Updated**: 2025-10-29
**Project**: Traffic Buddy - Zalo AI Challenge 2025  
**Structure Version**: 1.1 (Fixed all training pipeline paths)
