#!/usr/bin/env python
"""
Unsloth-based simple VLM LoRA fine-tuning for video understanding (multi-frame).

This trainer:
- Loads a VLM via Unsloth FastVisionModel in 4-bit for memory efficiency
- Adds LoRA adapters (configurable) for parameter-efficient fine-tuning
- Samples N frames per video and formats data into Unsloth vision messages
- Uses TRL SFTTrainer with UnslothVisionDataCollator
- Reports metrics and logs to Weights & Biases
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import random
import math
import torch
from datasets import load_from_disk
from torch.utils.data import Dataset

from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

from src.data.video_sampler import sample_frames
from src.utils.constants import (
    TRAIN_JSON, TRAFFIC_DATASET_DIR, PROCESSED_DATA_DIR, DEFAULT_BASE_MODEL,
    ensure_dirs, resolve_video_path,
)
from src.training.trainer_utils import clear_memory


def _load_or_build_dataset(data_path: str) -> Dict[str, Any]:
    dataset = None
    processed_dataset_path = str(PROCESSED_DATA_DIR)

    if os.path.exists(processed_dataset_path):
        try:
            print(f"Loading dataset from: {processed_dataset_path}")
            dataset = load_from_disk(processed_dataset_path)
        except Exception as e:
            print(f"Could not load from processed path: {e}")
            dataset = None

    if dataset is None and os.path.exists(data_path):
        try:
            print(f"Loading dataset from: {data_path}")
            dataset = load_from_disk(data_path)
        except Exception as e:
            print(f"Could not load from data_path: {e}")
            dataset = None

    if dataset is None:
        if not TRAIN_JSON.exists():
            raise FileNotFoundError(
                f"Training data not found at {TRAIN_JSON}. Prepare dataset first."
            )
        from src.data.dataset_builder import load_traffic_dataset
        print(f"Building dataset from: {TRAIN_JSON}")
        dataset = load_traffic_dataset(str(TRAIN_JSON))
        os.makedirs(processed_dataset_path, exist_ok=True)
        dataset.save_to_disk(processed_dataset_path)
        print(f"Saved dataset to: {processed_dataset_path}")

    return dataset


def _convert_example_to_messages(
    ex: Dict[str, Any],
    instruction: str,
    num_frames: int,
    frame_size: Tuple[int, int],
) -> Dict[str, Any]:
    video_path = ex.get("video_path")
    if isinstance(video_path, list):
        video_path = video_path[0] if len(video_path) > 0 else None
    if isinstance(video_path, (str, os.PathLike)) and video_path:
        try:
            video_path = str(resolve_video_path(str(video_path)))
        except Exception:
            pass
    support_frames = ex.get("support_frames", [])
    frames: List[Any] = []
    if isinstance(video_path, (str, os.PathLike)) and video_path:
        frames, _ = sample_frames(
            video_path,
            support_frames=support_frames,
            num_frames=num_frames,
            size=frame_size,
            return_format="pil",
        )

    prompt = ex.get("prompt")
    response_text = ex.get("response", "")

    content = []

    for img in frames:
        content.append(
            {
                "type": "image", 
                "image": img
            }
        )
    if prompt:
        content.append(
            {
                "type": "text",
                "text": prompt
            }
        )

    messages = []
    if instruction:
        messages.append(
            {
                "role": "system",
                "content": [
                    {
                        "type": "text", 
                        "text": instruction
                    }
                ]
            }
        )
    messages.append(
        {
            "role": "user", 
            "content": content
        }
    )
    messages.append(
        {
            "role": "assistant", 
            "content": [
                {
                    "type": "text", 
                    "text": response_text
                }
            ]
        }
    )
    return {"messages": messages}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    # Model & LoRA
    parser.add_argument("--base_model_id", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=15)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--finetune_vision_layers", action="store_true", default=True)
    parser.add_argument("--finetune_language_layers", action="store_true", default=True)
    parser.add_argument("--finetune_attention_modules", action="store_true", default=True)
    parser.add_argument("--finetune_mlp_modules", action="store_true", default=True)

    # Data
    parser.add_argument("--data_path", type=str, default=str(TRAFFIC_DATASET_DIR))
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--frame_size", type=int, nargs=2, default=[336, 336])
    parser.add_argument("--instruction", type=str, default="You are a vision-language model specialized in traffic video understanding. Analyze the provided frames and answer concisely (one word, number, or short phrase) about road users, actions, traffic signals/signs, and potential violations. Avoid extra explanation unless necessary.")

    # Training
    parser.add_argument("--output_dir", type=str, default="./checkpoints/lora_sft_unsloth")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=3407)

    # Logging
    parser.add_argument("--wandb_project", type=str, default="traffic_buddy")
    parser.add_argument("--run_name", type=str, default="unsloth_vlm_simple")
    parser.add_argument("--report_to_wandb", action="store_true", default=True)

    args = parser.parse_args()

    ensure_dirs()

    # Dataset
    dataset = _load_or_build_dataset(args.data_path)

    # Load Unsloth FastVisionModel in 4-bit and set up LoRA
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=args.base_model_id,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=args.finetune_vision_layers,
        finetune_language_layers=args.finetune_language_layers,
        finetune_attention_modules=args.finetune_attention_modules,
        finetune_mlp_modules=args.finetune_mlp_modules,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
        target_modules = "all-linear"
    )

    # Build lazy transforms that produce Unsloth "messages"
    def _transform(ex: Dict[str, Any]) -> Dict[str, Any]:
        return _convert_example_to_messages(
            ex,
            instruction=args.instruction,
            num_frames=args.num_frames,
            frame_size=(args.frame_size[0], args.frame_size[1]),
        )

    # Pre-filter to only keep rows with resolvable existing videos
    def _has_video(ex: Dict[str, Any]) -> bool:
        vp = ex.get("video_path")
        if isinstance(vp, list):
            vp = vp[0] if len(vp) > 0 else None
        if not vp:
            return False
        try:
            resolved = resolve_video_path(str(vp))
            return resolved.exists()
        except Exception:
            return False

    train_ds = dataset["train"].filter(_has_video)
    val_ds = dataset["validation"].filter(_has_video)

    # Compute and show effective training schedule
    num_train_examples = len(train_ds)
    effective_batch = max(1, args.batch_size * args.grad_accum_steps)
    steps_per_epoch = math.ceil(num_train_examples / effective_batch) if num_train_examples > 0 else 0
    total_steps = steps_per_epoch * args.num_epochs
    print(
        f"Scheduling → num_train_examples={num_train_examples}, per_device_batch={args.batch_size}, "
        f"grad_accum={args.grad_accum_steps}, effective_batch={effective_batch}, "
        f"epochs={args.num_epochs}, steps_per_epoch={steps_per_epoch}, total_steps={total_steps}"
    )

    class MessagesDataset(Dataset):
        def __init__(self, hf_ds):
            self.hf_ds = hf_ds
        def __len__(self) -> int:
            return len(self.hf_ds)
        def __getitem__(self, idx: int):
            ex = self.hf_ds[idx]
            out = _convert_example_to_messages(
                ex,
                instruction=args.instruction,
                num_frames=args.num_frames,
                frame_size=(args.frame_size[0], args.frame_size[1]),
            )
            return out["messages"]

    train_dataset = MessagesDataset(train_ds)
    val_dataset = MessagesDataset(val_ds)

    # Collator
    data_collator = UnslothVisionDataCollator(model, tokenizer)

    # Trainer config
    report_to = ["wandb"] if args.report_to_wandb else []
    from transformers import EarlyStoppingCallback, TrainerCallback

    callbacks = [EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)]

    class TrainingInfoCallback(TrainerCallback):
        def on_train_begin(self, args_, state, control, **kwargs):  # type: ignore
            try:
                import wandb  # local import to avoid hard dep when disabled
                wandb.log({
                    "sched/num_train_examples": num_train_examples,
                    "sched/per_device_batch": args.batch_size,
                    "sched/grad_accum": args.grad_accum_steps,
                    "sched/effective_batch": effective_batch,
                    "sched/epochs": args.num_epochs,
                    "sched/steps_per_epoch": steps_per_epoch,
                    "sched/total_steps": total_steps,
                })
            except Exception:
                pass
            print(
                f"[TrainBegin] examples={num_train_examples}, effective_batch={effective_batch}, "
                f"epochs={args.num_epochs}, steps/epoch={steps_per_epoch}, total_steps={total_steps}"
            )

    callbacks.append(TrainingInfoCallback())

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=SFTConfig(
            output_dir=args.output_dir,
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum_steps,
            warmup_ratio=args.warmup_ratio,
            num_train_epochs=args.num_epochs,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=10,
            save_strategy="steps",
            save_steps=20,
            save_total_limit=3,
            optim="adamw_8bit",
            lr_scheduler_type=args.lr_scheduler_type,
            seed=args.seed,
            report_to=report_to,
            run_name=args.run_name,
            # Vision finetuning requirements
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=2048,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        ),
        callbacks=callbacks,
    )

    # Train
    FastVisionModel.for_training(model)
    print(
        f"Starting Unsloth VLM fine-tuning: frames={args.num_frames}, r={args.lora_r}, lr={args.learning_rate}"
    )
    trainer_stats = trainer.train()
    print(f"Training runtime: {trainer_stats.metrics.get('train_runtime', 'n/a')}s")

    # Save LoRA adapters
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(os.path.join(args.output_dir, "final_lora"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final_lora"))

    print(f"Saved LoRA adapters to {os.path.join(args.output_dir, 'final_lora')}")
    clear_memory()


if __name__ == "__main__":
    main()



