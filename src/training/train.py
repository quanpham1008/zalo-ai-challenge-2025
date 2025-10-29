#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fine-tune Qwen3-VL on traffic video QA using LoRA (with regret) + TRL + Ray Tune + WandB.
"""

import os
import sys
from pathlib import Path

import torch
import wandb
from ray import tune
from datasets import load_from_disk, DatasetDict
from transformers import AutoTokenizer, AutoProcessor, Qwen3VLForConditionalGeneration, EarlyStoppingCallback, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from src.training.trainer_utils import clear_memory
from src.data.dataset_builder import load_traffic_dataset
from src.data.processor_collate import TrafficDataCollator, TrafficDataProcessor
from src.utils.constants import (
    TRAIN_JSON, TRAFFIC_DATASET_DIR, PROCESSED_DATA_DIR, DEFAULT_BASE_MODEL, RAY_RESULTS_DIR,
    ensure_dirs
)
from ray.air.integrations.wandb import WandbLoggerCallback


# ============================================================
# ✅ Training function (Ray Tune calls this per trial)
# ============================================================
def train_fn(config, data_path, output_dir, base_model_id):
    """
    config: hyperparams sampled by Ray Tune
    data_path: path to prepared HuggingFace DatasetDict
    """

    # Initialize wandb for this trial
    run = wandb.init(
        project=config["wandb"]["project"],
        name=f"trial_lr{config['lr']}_r{config['lora_r']}",
        config=config,
        reinit=True,  # Allow reinit for Ray Tune
        settings=wandb.Settings(start_method="thread")  # Fix for Ray Tune
    )

    # Load dataset
    dataset = load_from_disk(data_path)
    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]

    # Load processor & tokenizer
    processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load model with BitsAndBytes 4-bit quantization if requested
    quantization_config = None
    if config.get("use_4bit", True):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        quantization_config=quantization_config
    )

    # LoRA without regret config - Updated for better performance
    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        target_modules="all-linear",  # Target all linear layers for better adaptation
        lora_dropout=config["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Build processed datasets (extract frames, tokenize via processor) and collator
    data_processor = TrafficDataProcessor(
        model_path=base_model_id,
        num_frames=config.get("num_frames", 4),
        frame_size=tuple(config.get("frame_size", (336, 336))),
    )
    train_dataset = train_dataset.map(data_processor, batched=False)
    val_dataset = val_dataset.map(data_processor, batched=False)
    collator = TrafficDataCollator(processor)

    # TRL SFT config
    sft_config = SFTConfig(
        output_dir=os.path.join(output_dir, run.name),
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config.get("batch_size", 1),
        per_device_eval_batch_size=config.get("batch_size", 1),
        gradient_accumulation_steps=config.get("grad_accum_steps", 4),
        learning_rate=config["lr"],
        lr_scheduler_type=config.get("lr_scheduler", "cosine"),  # Allow tuning scheduler
        warmup_ratio=0.1,
        weight_decay=config.get("weight_decay", 0.01),  # L2 regularization
        max_grad_norm=1.0,  # Gradient clipping to prevent exploding gradients
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,  # Evaluate every N steps
        save_strategy="steps",
        save_steps=200,  # Save checkpoint every N steps
        save_total_limit=5,  # Keep last 5 checkpoints
        load_best_model_at_end=True,  # Load best model based on validation loss
        metric_for_best_model="eval_loss",  # Metric to determine best model
        greater_is_better=False,  # Lower eval_loss is better
        report_to=["wandb"],
        bf16=False,
        fp16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit" if config.get("use_4bit", True) else "adamw_torch",
        max_length=1024,
    )

    # Prepare callbacks
    callbacks = []
    if config.get("early_stopping", False):
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=config.get("early_stopping_patience", 3),
            early_stopping_threshold=0.001,  # Stop if improvement < 0.001
        )
        callbacks.append(early_stopping)

    # Trainer
    trainer = SFTTrainer(
        model=model,
        processing_class=processor,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        callbacks=callbacks,
    )

    trainer.train()

    # Evaluate
    metrics = trainer.evaluate()
    tune.report(val_loss=metrics["eval_loss"])

    # Save final model checkpoint
    trainer.save_model(os.path.join(output_dir, run.name, "final"))
    run.finish()
    clear_memory()


# ============================================================
# ✅ Main Ray Tune orchestration
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=str(TRAFFIC_DATASET_DIR))
    parser.add_argument("--base_model_id", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--output_dir", type=str, default="./checkpoints/lora_sft")
    parser.add_argument("--wandb_project", type=str, default="zalo_challenge")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_samples", type=int, default=4)
    args = parser.parse_args()

    # Ensure necessary directories exist
    ensure_dirs()

    # Resolve output_dir to an absolute path so Ray workers can save checkpoints correctly
    output_dir_abs = os.path.abspath(args.output_dir)
    os.makedirs(output_dir_abs, exist_ok=True)
    print(f"📁 Model checkpoints will be saved to: {output_dir_abs}")

    # Resolve dataset path to an absolute path so Ray workers can find it
    data_path_abs = os.path.abspath(args.data_path)

    # Prepare dataset (or load existing)
    dataset = None
    processed_dataset_path = str(PROCESSED_DATA_DIR)
    
    # Try to load from data/processed first (current location)
    if os.path.exists(processed_dataset_path):
        try:
            print(f"📂 Loading existing dataset from: {processed_dataset_path}")
            dataset = load_from_disk(processed_dataset_path)
            print(f"✅ Dataset loaded: {len(dataset.get('train', []))} train, {len(dataset.get('validation', []))} validation")
        except Exception as e:
            print(f"⚠️  Could not load from {processed_dataset_path}: {e}")
            dataset = None
    
    # If not found in processed, try the specified path
    if dataset is None:
        if os.path.exists(data_path_abs):
            try:
                print(f"📂 Loading existing dataset from: {data_path_abs}")
                dataset = load_from_disk(data_path_abs)
                print(f"✅ Dataset loaded: {len(dataset.get('train', []))} train, {len(dataset.get('validation', []))} validation")
            except Exception as e:
                print(f"⚠️  Could not load from {data_path_abs}: {e}")
                dataset = None
        
        # If still not found, build from JSON
        if dataset is None:
            # Check if training JSON exists
            if not TRAIN_JSON.exists():
                raise FileNotFoundError(
                    f"Training data not found at {TRAIN_JSON}. "
                    f"Please ensure your data is in the correct location."
                )
            
            print(f"📦 Building dataset from: {TRAIN_JSON}")
            dataset = load_traffic_dataset(str(TRAIN_JSON))
            # Save to processed data directory
            os.makedirs(processed_dataset_path, exist_ok=True)
            dataset.save_to_disk(processed_dataset_path)
            print(f"✅ Dataset saved to: {processed_dataset_path}")
    
    # Use the dataset path that worked (or was created)
    dataset_path_for_ray = processed_dataset_path if os.path.exists(processed_dataset_path) else data_path_abs
    dataset_path_for_ray = os.path.abspath(dataset_path_for_ray)
    print(f"📦 Using dataset path for Ray Tune: {dataset_path_for_ray}")

    # Ray Tune config - Updated for LoRA without regret
    search_space = {
        "lr": tune.loguniform(1e-5, 5e-4),
        "lora_r": tune.choice([64, 128]),  # Lower ranks reduce memory
        "lora_alpha": tune.choice([16, 32]),
        "lora_dropout": tune.choice([0.05, 0.1]),
        "num_epochs": tune.choice([1, 2]),  # Shorter runs first
        "batch_size": tune.choice([1]),     # Force batch size 1 on 24GB GPUs
        "grad_accum_steps": tune.choice([4]),
        "use_4bit": tune.choice([True]),    # Force 4-bit to fit 8B on 24GB
        "num_frames": tune.choice([4, 6]),  # Fewer frames to lower memory
        "frame_size": tune.choice([(336, 336)]),
        "weight_decay": tune.choice([0.0, 0.01]),
        "lr_scheduler": tune.choice(["cosine", "linear", "polynomial"]),
        "early_stopping": tune.choice([True]),
        "early_stopping_patience": tune.choice([3, 5]),
        "wandb": {
            "project": args.wandb_project,
        },
    }

    tuner = tune.Tuner(
        tune.with_resources(
            # Pass absolute paths so Ray workers can load dataset and save checkpoints correctly
            tune.with_parameters(train_fn, data_path=dataset_path_for_ray, output_dir=output_dir_abs, base_model_id=args.base_model_id),
            resources={"cpu": 8, "gpu": 1}
        ),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric="val_loss",
            mode="min",
            num_samples=args.num_samples,
            max_concurrent_trials=1,  # Avoid OOM by running 1 trial at a time per node
        ),
        run_config=tune.RunConfig(
            name="training",
            storage_path=os.path.abspath(str(RAY_RESULTS_DIR)),
            callbacks=[
                WandbLoggerCallback(
                    project=args.wandb_project,
                )
            ],
        ),
    )

    tuner.fit()


if __name__ == "__main__":
    main()
