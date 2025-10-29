#!/usr/bin/env python
"""
Simple LoRA fine-tuning without Ray Tune - LoRA without regret approach
"""

import os
import sys
from pathlib import Path

# Ensure project root is on sys.path for `src` imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from datasets import load_from_disk
import re
import random
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
)
from peft import LoraConfig, get_peft_model
from peft.utils.other import prepare_model_for_kbit_training
from trl import SFTConfig
from transformers import Trainer, TrainingArguments
from src.training.trainer_utils import clear_memory
from src.data.dataset_builder import load_traffic_dataset
from src.data.processor_collate import TrafficDataProcessor, TrafficDataCollator
from src.data.video_sampler import sample_frames
from transformers import EarlyStoppingCallback, TrainerCallback
import wandb
from src.utils.constants import (
    TRAIN_JSON, TRAFFIC_DATASET_DIR, PROCESSED_DATA_DIR, DEFAULT_BASE_MODEL,
    ensure_dirs
)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_id", type=str, default=DEFAULT_BASE_MODEL)  # Qwen3-VL-8B-Instruct
    parser.add_argument("--output_dir", type=str, default="./checkpoints/lora_simple")
    parser.add_argument("--data_path", type=str, default=str(TRAFFIC_DATASET_DIR))
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--lora_rank", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    args = parser.parse_args()

    # Ensure necessary directories exist
    ensure_dirs()

    # Prepare dataset - try data/processed first (current location)
    dataset = None
    processed_dataset_path = str(PROCESSED_DATA_DIR)
    
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
        if os.path.exists(args.data_path):
            try:
                print(f"📂 Loading existing dataset from: {args.data_path}")
                dataset = load_from_disk(args.data_path)
                print(f"✅ Dataset loaded: {len(dataset.get('train', []))} train, {len(dataset.get('validation', []))} validation")
            except Exception as e:
                print(f"⚠️  Could not load from {args.data_path}: {e}")
                dataset = None
        
        # If still not found, build from JSON
        if dataset is None:
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

    # Load model components
    processor = AutoProcessor.from_pretrained(args.base_model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load model (prefer 4-bit to fit 8B on 24GB GPUs)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base_model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=bnb_config,
    )
    try:
        model.config.use_cache = False
    except Exception:
        pass

    # Prepare model for 4-bit LoRA + gradient checkpointing
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # LoRA without regret config (your suggested approach)
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=16,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Build processed datasets and collator (align with train.py)
    data_processor = TrafficDataProcessor(model_path=args.base_model_id, num_frames=1, frame_size=(336, 336))
    print("Attaching on-the-fly processor with with_transform (no materialization)...")
    train_dataset = dataset["train"].with_transform(data_processor)
    val_dataset = dataset["validation"].with_transform(data_processor)
    collator = TrafficDataCollator(processor)

    # Training configuration
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        num_train_epochs=args.num_epochs,
        warmup_ratio=0.1,
        logging_steps=10,
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        bf16=False,
        fp16=True,
        gradient_checkpointing=True,
        optim="adamw_torch",
        report_to=["wandb"],
        run_name=f"traffic_lora_r{args.lora_rank}_lr{args.learning_rate}",
        remove_unused_columns=False,
    )

    # Callbacks (custom eval accuracy logger)
    callbacks = []

    def eval_accuracy_small(model, processor, raw_eval_dataset, num_samples=32):
        idxs = list(range(len(raw_eval_dataset)))
        random.shuffle(idxs)
        idxs = idxs[:num_samples]
        correct = 0
        total = 0
        for i in idxs:
            ex = raw_eval_dataset[i]
            video_path = ex["video_path"]
            frames, _ = sample_frames(video_path, support_frames=ex.get("support_frames", []), num_frames=4, size=(336, 336), return_format="pil")
            if not frames:
                continue
            prompt = ex["prompt"]
            messages = [
                {"role": "user", "content": [{"type": "text", "text": prompt}] + [{"type": "image", "image": img} for img in frames]},
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=frames, padding=True, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=4)
            out_text = processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            m = re.search(r"\b([A-D])\b", out_text)
            pred = m.group(1) if m else None
            ans_m = re.match(r"\s*([A-D])", ex.get("response", ""))
            gold = ans_m.group(1) if ans_m else None
            if pred and gold:
                total += 1
                if pred == gold:
                    correct += 1
        return (correct / total) if total > 0 else 0.0

    class WandbEvalAccuracyCallback(TrainerCallback):
        def on_evaluate(self, args, state, control, **kwargs):  # type: ignore
            try:
                acc = eval_accuracy_small(model, processor, dataset["validation"], num_samples=32)
                wandb.log({"eval/accuracy_small": acc, "global_step": state.global_step})
            except Exception:
                pass

    callbacks.append(WandbEvalAccuracyCallback())

    # Ensure processor exposes eos_token expected by TRL when passing processing_class
    try:
        processor.eos_token = tokenizer.eos_token  # type: ignore[attr-defined]
    except Exception:
        pass

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        callbacks=callbacks,
    )

    # Train the model
    print(f"🚀 Starting training with LoRA rank {args.lora_rank}...")
    trainer.train()

    # Save final model
    trainer.save_model(os.path.join(args.output_dir, "final"))
    print(f"✅ Training completed! Model saved to {args.output_dir}")
    clear_memory()

if __name__ == "__main__":
    main()
