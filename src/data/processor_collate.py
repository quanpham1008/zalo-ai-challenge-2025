import torch
from transformers import AutoProcessor
from typing import Dict, List, Union
from PIL import Image
import base64
import io
import numpy as np
import os
from decord import VideoReader, cpu
from src.data.video_sampler import sample_frames


class TrafficDataProcessor:
    """
    Processor wrapper cho video traffic QA dataset.
    Dùng chung cho cả train và inference.
    """

    def __init__(self, model_path: str, num_frames: int = 8, frame_size=(448, 448)):
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.num_frames = num_frames
        self.frame_size = frame_size

    def __call__(self, example: Dict):
        """
        Transform 1 example → encoded input cho model
        """
        # Trích khung hình từ video với proper path handling
        video_path = example["video_path"]
        
        # Handle both relative and absolute paths
        if not os.path.isabs(video_path):
            # Try different possible locations for relative paths
            possible_paths = [
                video_path,  # as is
                os.path.join(".", video_path),  # current directory
                os.path.join("/home/quanpv/project/traffic_buddy", video_path)  # absolute project path
            ]
            video_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    video_path = path
                    break
            
            if video_path is None:
                print(f"[ERROR] Video not found at any location: {example['video_path']}")
                # Return empty/dummy data for missing videos
                return self._create_dummy_inputs(example)
        
        support_frames = example.get("support_frames", [])
        frames, indices = sample_frames(
            video_path, 
            support_frames=support_frames,
            num_frames=self.num_frames, 
            size=self.frame_size,
            return_format="pil"  # Request PIL images for processor
        )
        
        if not frames:
            print(f"[ERROR] No frames extracted from video: {video_path}")
            return self._create_dummy_inputs(example)

        # Tạo messages chuẩn chat template
        prompt = example["prompt"]
        answer = example["response"]
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": prompt}
            ] + [{"type": "image", "image": img} for img in frames]},
            {"role": "assistant", "content": answer}
        ]

        # Encode theo processor
        try:
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            model_inputs = self.processor(
                text=[text],
                images=frames,
                padding=True,
                return_tensors="pt"
            )
            return model_inputs
        except Exception as e:
            print(f"[ERROR] Failed to process example {example.get('id', 'unknown')}: {e}")
            return self._create_dummy_inputs(example)
    
    def _create_dummy_inputs(self, example: Dict):
        """Create dummy inputs for failed video processing"""
        prompt = example["prompt"]
        answer = example["response"]
        
        # Create a simple text-only input
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer}
        ]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        model_inputs = self.processor(
            text=[text],
            padding=True,
            return_tensors="pt"
        )
        return model_inputs


class TrafficDataCollator:
    """
    Data collator để batch nhiều sample (text + video).
    """

    def __init__(self, processor: AutoProcessor, pad_to_multiple_of: int = 8):
        self.processor = processor
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Batch các sample, padding dynamic.
        """
        # Handle empty batch
        if not batch:
            return {}
            
        # Extract components from batch
        input_ids = []
        attention_mask = []
        pixel_values = []
        image_grid_thw = []
        
        for item in batch:
            if "input_ids" in item:
                input_ids.append(item["input_ids"].squeeze(0) if item["input_ids"].dim() > 1 else item["input_ids"])
            if "attention_mask" in item:
                attention_mask.append(item["attention_mask"].squeeze(0) if item["attention_mask"].dim() > 1 else item["attention_mask"])
            if "pixel_values" in item and item["pixel_values"] is not None:
                pixel_values.append(item["pixel_values"])
            if "image_grid_thw" in item and item["image_grid_thw"] is not None:
                image_grid_thw.append(item["image_grid_thw"])

        # Padding text input
        if input_ids:
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=self.processor.tokenizer.pad_token_id
            )
        
        if attention_mask:
            attention_mask = torch.nn.utils.rnn.pad_sequence(
                attention_mask,
                batch_first=True,
                padding_value=0
            )

        # Build result dictionary
        batch_dict = {}
        if input_ids is not None and len(input_ids) > 0:
            batch_dict["input_ids"] = input_ids
        if attention_mask is not None and len(attention_mask) > 0:
            batch_dict["attention_mask"] = attention_mask
        if pixel_values:
            batch_dict["pixel_values"] = torch.cat(pixel_values, dim=0) if len(pixel_values) > 1 else pixel_values[0]
        if image_grid_thw:
            batch_dict["image_grid_thw"] = torch.cat(image_grid_thw, dim=0) if len(image_grid_thw) > 1 else image_grid_thw[0]
        
        return batch_dict


# Alias for backward compatibility with training script
VideoCollator = TrafficDataCollator