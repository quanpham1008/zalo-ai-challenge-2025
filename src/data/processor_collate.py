import torch
from transformers import AutoProcessor
from typing import Dict, List
from PIL import Image
import os
from src.data.video_sampler import sample_frames
from src.utils.constants import resolve_video_path


class TrafficDataProcessor:
    """Processor cho video traffic QA dataset (train & infer)."""

    def __init__(self, model_path: str, num_frames: int = 8, frame_size=(448, 448)):
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.num_frames = num_frames
        self.frame_size = frame_size

    def __call__(self, example: Dict):
        """Biến 1 sample thành encoded input cho model."""
        video_path_str = example["video_path"]
        # Normalize when coming from batched contexts that wrap fields in single-element lists
        if isinstance(video_path_str, list) and video_path_str:
            video_path_str = video_path_str[0]

        # Resolve path using centralized function
        try:
            video_path = resolve_video_path(video_path_str)
            if not video_path.exists():
                print(f"[ERROR] Video not found after resolution: {video_path_str}")
                print(f"  Tried path: {video_path}")
                return self._create_dummy_inputs(example)
            video_path = str(video_path)
        except Exception as e:
            print(f"[ERROR] Failed to resolve video path {video_path_str}: {e}")
            return self._create_dummy_inputs(example)

        support_frames = example.get("support_frames", [])
        frames, _ = sample_frames(
            video_path,
            support_frames=support_frames,
            num_frames=self.num_frames,
            size=self.frame_size,
            return_format="pil"
        )

        if not frames:
            print(f"[ERROR] No frames extracted: {video_path}")
            return self._create_dummy_inputs(example)

        prompt = example["prompt"]
        answer = example["response"]
        messages = [
            {"role": "user", "content": [{"type": "text", "text": prompt}] +
             [{"type": "image", "image": img} for img in frames]},
            {"role": "assistant", "content": answer}
        ]

        try:
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            model_inputs = self.processor(
                text=[text],
                images=frames,
                padding=True,
                return_tensors="pt"
            )
            # Tạo labels cho training
            model_inputs["labels"] = model_inputs["input_ids"].clone()
            return model_inputs
        except Exception as e:
            print(f"[ERROR] Processing failed for {example.get('id', 'unknown')}: {e}")
            return self._create_dummy_inputs(example)

    def _create_dummy_inputs(self, example: Dict):
        """Fallback khi video bị lỗi."""
        prompt = example["prompt"]
        answer = example["response"]

        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer}
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        model_inputs = self.processor(text=[text], padding=True, return_tensors="pt")
        model_inputs["labels"] = model_inputs["input_ids"].clone()
        return model_inputs


class TrafficDataCollator:
    """Collator để batch text + video input."""

    def __init__(self, processor: AutoProcessor, pad_to_multiple_of: int = 8):
        self.processor = processor
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        if not batch:
            return {}

        input_ids, attention_mask, pixel_values, image_grid_thw, labels = [], [], [], [], []

        for item in batch:
            if "input_ids" in item:
                input_ids.append(item["input_ids"].squeeze(0))
            if "attention_mask" in item:
                attention_mask.append(item["attention_mask"].squeeze(0))
            if "labels" in item:
                labels.append(item["labels"].squeeze(0))
            if "pixel_values" in item and item["pixel_values"] is not None:
                pixel_values.append(item["pixel_values"])
            if "image_grid_thw" in item and item["image_grid_thw"] is not None:
                image_grid_thw.append(item["image_grid_thw"])

        pad_id = self.processor.tokenizer.pad_token_id if self.processor.tokenizer.pad_token_id is not None else 0
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

        batch_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

        if pixel_values:
            batch_dict["pixel_values"] = torch.cat(pixel_values, dim=0)
        if image_grid_thw:
            image_grid_thw = [igt.squeeze(0) if igt.dim() == 2 and igt.size(0) == 1 else igt for igt in image_grid_thw]
            batch_dict["image_grid_thw"] = torch.stack(image_grid_thw, dim=0)

        return batch_dict


VideoCollator = TrafficDataCollator
