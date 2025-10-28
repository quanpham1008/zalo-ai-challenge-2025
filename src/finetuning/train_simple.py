#!/usr/bin/env python
"""
Simple LoRA fine-tuning without Ray Tune - LoRA without regret approach
"""

import os
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from src.data.dataset_builder import load_traffic_dataset
from src.data.processor_collate import TrafficDataProcessor

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_id", type=str, default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/lora_simple")
    parser.add_argument("--data_path", type=str, default="datasets/traffic_dataset")
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--lora_rank", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    args = parser.parse_args()

    # Prepare dataset
    if not os.path.exists(args.data_path):
        train_json_path = "/home/quanpv/project/traffic_buddy/train/train.json"
        print(f"Loading training data from: {train_json_path}")
        dataset = load_traffic_dataset(train_json_path)
        dataset.save_to_disk(args.data_path)
    else:
        dataset = load_from_disk(args.data_path)

    # Load model components
    processor = AutoProcessor.from_pretrained(args.base_model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

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

    # Data processor for multimodal inputs
    data_processor = TrafficDataProcessor(
        model_path=args.base_model_id,
        num_frames=8,
        frame_size=(448, 448)
    )

    # Transform datasets
    print("Processing training dataset...")
    train_dataset = dataset["train"].map(data_processor, batched=False)
    val_dataset = dataset["validation"].map(data_processor, batched=False)

    # Training configuration
    training_args = SFTConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        num_train_epochs=args.num_epochs,
        warmup_ratio=0.1,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps", 
        save_steps=100,
        save_total_limit=3,
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_torch",
        max_seq_length=1024,
        report_to=["wandb"],  # Change to ["trackio"] if you prefer
        run_name=f"traffic_lora_r{args.lora_rank}_lr{args.learning_rate}",
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_config,
        dataset_text_field=None,
        packing=False,
    )

    # Train the model
    print(f"🚀 Starting training with LoRA rank {args.lora_rank}...")
    trainer.train()

    # Save final model
    trainer.save_model(os.path.join(args.output_dir, "final"))
    print(f"✅ Training completed! Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
