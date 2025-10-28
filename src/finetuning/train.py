#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fine-tune Qwen3-VL on traffic video QA using LoRA (with regret) + TRL + Ray Tune + WandB.
"""

import os
import torch
import wandb
from ray import tune
from datasets import load_from_disk, DatasetDict
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from src.data.dataset_builder import load_traffic_dataset
from src.data.processor_collate import TrafficDataCollator


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

    # Load model with quantization if needed
    model_kwargs = {}
    if config["use_4bit"]:
        model_kwargs.update({
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": torch.bfloat16,
        })

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        **model_kwargs,
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

    # Collator for multimodal (video frame embedding)
    collator = TrafficDataCollator(processor)

    # TRL SFT config
    sft_config = SFTConfig(
        output_dir=os.path.join(output_dir, run.name),
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["grad_accum_steps"],
        learning_rate=config["lr"],
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        report_to=["wandb"],
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit" if config["use_4bit"] else "adamw_torch",
        max_seq_length=1024,
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        peft_config=lora_config,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        dataset_text_field=None,
        packing=False,
    )

    trainer.train()

    # Evaluate
    metrics = trainer.evaluate()
    tune.report(val_loss=metrics["eval_loss"])

    # Save final model checkpoint
    trainer.save_model(os.path.join(output_dir, run.name, "final"))
    run.finish()


# ============================================================
# ✅ Main Ray Tune orchestration
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="datasets/traffic_dataset")
    parser.add_argument("--base_model_id", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/lora_sft")
    parser.add_argument("--wandb_project", type=str, default="traffic_vlm_finetune")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_samples", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Prepare dataset (or load existing)
    if not os.path.exists(args.data_path):
        # Use absolute path for training data
        train_json_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "train", "train.json")
        if not os.path.exists(train_json_path):
            train_json_path = "/home/quanpv/project/traffic_buddy/train/train.json"
        
        print(f"Loading training data from: {train_json_path}")
        dataset = load_traffic_dataset(train_json_path)
        dataset.save_to_disk(args.data_path)
    else:
        dataset = load_from_disk(args.data_path)

    # Ray Tune config - Updated for LoRA without regret
    search_space = {
        "lr": tune.loguniform(1e-5, 5e-4),
        "lora_r": tune.choice([64, 128, 256]),  # Higher ranks for better performance
        "lora_alpha": tune.choice([16, 32]),    # Simplified alpha search
        "lora_dropout": tune.choice([0.05, 0.1]),
        "num_epochs": tune.choice([1, 2, 3]),   # Fewer epochs with higher rank
        "batch_size": tune.choice([1, 2]),
        "grad_accum_steps": tune.choice([4, 8]), # Higher accumulation for stability
        "use_4bit": tune.choice([True, False]),
        "wandb": {
            "project": args.wandb_project,
        },
    }

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_fn, data_path=args.data_path, output_dir=args.output_dir, base_model_id=args.base_model_id),
            resources={"cpu": 8, "gpu": 1}
        ),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric="val_loss",
            mode="min",
            num_samples=args.num_samples,
        ),
        run_config=tune.RunConfig(
            name="training",
            storage_path=os.path.abspath("./ray_results"),
        ),
    )

    tuner.fit()


if __name__ == "__main__":
    main()
