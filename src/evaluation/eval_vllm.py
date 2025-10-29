#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluation script for Qwen3-VL using vLLM
Supports multi-frame sampling from video + latency benchmarking
"""

import os
import json
import csv
import re
import time
import logging
from datetime import datetime
import numpy as np
from typing import List, Tuple
import torch
from typing import List
from PIL import Image
from decord import VideoReader, cpu
from tqdm import tqdm

from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
from src.training.prompt import SYSTEM_PROMPT

# Mitigate CUDA memory fragmentation during vLLM init
os.environ["VLLM_TORCH_COMPILE"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["VLLM_MAX_NUM_SEQS"] = "1"


def setup_logger(log_name: str, log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = os.path.join(log_dir, f"{log_name}_{timestamp}.log")

    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    logger.info(f"Logging to {logfile}")
    return logger


# ======================
# Config
# ======================
MODEL_PATH = "Qwen/Qwen3-VL-8B-Instruct"
DATA_PATH = "/home/quanpv/project/traffic_buddy/public_test/public_test.json"
OUT_CSV = "/home/quanpv/project/traffic_buddy/public_test/submission_qwen3vl_vllm.csv"
BENCH_CSV = "/home/quanpv/project/traffic_buddy/public_test/latency_log_vllm.csv"
NUM_FRAMES = 8
FRAME_SIZE = (448, 448)
MAX_NEW_TOKENS = 8
DEVICE_MEMORY_UTIL = 0.85  # lower to reduce KV-cache reservation
MAX_NUM_SEQS = 2
MAX_MODEL_LEN = 4096
CHANGE_THRESHOLD = 30.0  # for scene-change detection


# ======================
# Helper functions
# ======================
def sample_key_frames(video_path: str,
                      num_frames: int = 6,
                      size=(448, 448),
                      change_threshold: float = 30.0) -> Tuple[List[Image.Image], List[int]]:
    """
    Trích khung hình chính từ video:
      - khung đầu, giữa, cuối
      - khung có sự “thay đổi cảnh” lớn nếu có
      - tổng số khung ≤ num_frames
    Returns list of PIL.Image.
    """
    if not os.path.exists(video_path):
        return [], []
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
    except Exception:
        return [], []
    total = len(vr)
    if total == 0:
        return [], []

    # indices đầu-giữa-cuối
    idx_head = 0
    idx_mid = total // 2
    idx_tail = total - 1
    base_indices = [idx_head, idx_mid, idx_tail]

    # sampling thêm bằng uniform để có đủ num_frames
    remaining = num_frames - len(base_indices)
    if remaining > 0:
        extra = np.linspace(0, total - 1, num=remaining + 2, dtype=int)[1:-1]
        extra = [i for i in extra if i not in base_indices]
        indices = base_indices + extra
    else:
        indices = base_indices[:num_frames]
    indices = sorted(set(indices))

    # load batch
    frames_np = vr.get_batch(indices).asnumpy()

    # detect large visual changes (scene changes)
    diffs = []
    for i in range(len(frames_np) - 1):
        f0 = frames_np[i].astype(np.float32)
        f1 = frames_np[i + 1].astype(np.float32)
        diff = np.mean(np.abs(f1 - f0))
        diffs.append(diff)
    for i, d in enumerate(diffs):
        if d > change_threshold:
            idx_change = indices[i + 1]
            if idx_change not in indices and len(indices) < num_frames:
                indices.append(idx_change)
    indices = sorted(indices)[:num_frames]

    # convert to PIL Images
    images = []
    for idx in indices:
        try:
            frame = vr[idx].asnumpy()
            img = Image.fromarray(frame).convert("RGB").resize(size, Image.Resampling.LANCZOS)
            images.append(img)
        except Exception:
            continue
    return images, indices


def format_choices(choices: List[str]) -> str:
    """Format choices into 'A. xxx\nB. yyy' etc."""
    letters = ["A", "B", "C", "D", "E"]
    return "\n".join([f"{letters[i]}. {c}" for i, c in enumerate(choices)])


def extract_letter(text: str, allowed: List[str]) -> str:
    """Extract predicted option letter from model output"""
    txt = text.strip()
    for a in allowed:
        if re.fullmatch(rf"\s*{re.escape(a)}\s*\.?\s*", txt, flags=re.I):
            return a.upper()
    for a in allowed:
        if re.search(rf"\b{re.escape(a)}\b", txt, flags=re.I):
            return a.upper()
    m = re.search(r"\b([A-Z])\b\.?:?", txt)
    if m and m.group(1).upper() in {x.upper() for x in allowed}:
        return m.group(1).upper()
    return allowed[0].upper() if allowed else "A"


# ======================
# Main evaluation logic
# ======================
def main():
    logger = setup_logger("eval_vllm", os.path.join(os.path.dirname(__file__), "logs"))

    logger.info(f"Model: {MODEL_PATH}")
    logger.info(f"NUM_FRAMES={NUM_FRAMES}, FRAME_SIZE={FRAME_SIZE}, MAX_NEW_TOKENS={MAX_NEW_TOKENS}, DEVICE_MEMORY_UTIL={DEVICE_MEMORY_UTIL}")
    logger.info(f"CUDA available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        try:
            logger.info(f"CUDA device count={torch.cuda.device_count()}, name={torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA capability={torch.cuda.get_device_capability(0)}")
        except Exception as e:
            logger.warning(f"Could not query CUDA device info: {e}")

    print(f"🚀 Loading model from {MODEL_PATH} ...")

    llm = LLM(
        model=MODEL_PATH,
        gpu_memory_utilization=DEVICE_MEMORY_UTIL,
        dtype="auto",
        tensor_parallel_size=2,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=MAX_NUM_SEQS
    )

    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    sampling_params = SamplingParams(max_tokens=MAX_NEW_TOKENS, temperature=0.0)

    # Load dataset
    with open(DATA_PATH, "r") as f:
        data = json.load(f)["data"]

    total = len(data)
    total_latency = 0.0
    failed = 0

    print(f"Loaded {total} samples. Starting evaluation...\n")
    logger.info(f"Loaded {total} samples")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    
    with open(OUT_CSV, "w", newline="") as fcsv, open(BENCH_CSV, "w", newline="") as fbench:
        writer = csv.writer(fcsv)
        writer.writerow(["id", "answer"])
        bench_writer = csv.writer(fbench)
        bench_writer.writerow(["id", "video_time", "inference_time", "total_time"])

        for item in tqdm(data, total=total):
            video_path = item["video_path"]
            question = item["question"]
            choices = item["choices"]

            t0 = time.time()
            frames, frame_indices = sample_key_frames(video_path, num_frames=NUM_FRAMES, size=FRAME_SIZE)
            t_video = time.time() - t0
            if not frames:
                failed += 1
                logger.warning(f"No frames for {video_path}; id={item.get('id', '?')}")
                writer.writerow([item["id"], "A"])
                bench_writer.writerow([item["id"], f"{t_video:.3f}", "0.0", f"{t_video:.3f}"])
                continue

            # Build messages - each frame as separate image entry
            allowed_letters = [c.split(".")[0].strip() for c in choices if "." in c[:3]]
            allowed_letters = [x for x in allowed_letters if len(x) == 1 and x.isalpha()]
            allowed_letters = sorted(set(allowed_letters))
            
            choices_text = format_choices(choices)
            content = [
                {
                    "type": "text",
                    "text": (
                        f"<question>\n{question}\n</question>\n\n"
                        f"<choices>\n{choices_text}\n</choices>\n\n"
                        f"Answer with exactly one letter among {{{', '.join(allowed_letters)}}}."
                    ),
                }
            ]
            # Add each frame as a separate image entry
            for frame in frames:
                content.append({"type": "image", "image": frame})
            
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content}
            ]

            try:
                # Prepare multimodal input
                prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, _ = process_vision_info(messages)
                mm_data = {"image": image_inputs}

                # Inference
                t1 = time.time()
                outputs = llm.generate(
                    [{"prompt": prompt, "multi_modal_data": mm_data}],
                    sampling_params=sampling_params,
                )
                t_infer = time.time() - t1
                t_total = time.time() - t0
                total_latency += t_total

                output_text = outputs[0].outputs[0].text.strip()
                answer_letter = extract_letter(output_text, allowed_letters)

                writer.writerow([item["id"], answer_letter])
                bench_writer.writerow([item["id"], f"{t_video:.3f}", f"{t_infer:.3f}", f"{t_total:.3f}"])

                logger.info(
                    f"id={item.get('id','?')} video={video_path} frames={len(frames)} indices={list(frame_indices)} "
                    f"answer={answer_letter} raw='{output_text.replace(chr(10),' ')[:500]}' times(s): video={t_video:.3f} infer={t_infer:.3f} total={t_total:.3f}"
                )

            except Exception as e:
                print(f"⚠️ Error processing {video_path}: {e}")
                logger.exception(f"Error processing {video_path} (id={item.get('id','?')})")
                writer.writerow([item["id"], "A"])
                bench_writer.writerow([item["id"], f"{t_video:.3f}", "0.0", f"{t_video:.3f}"])
                failed += 1
                continue

    # Summary
    processed = total - failed
    avg_latency = total_latency / processed if processed > 0 else 0.0

    print(f"\n✅ Saved submission: {OUT_CSV}")
    print(f"🧭 Saved latency log: {BENCH_CSV}")
    print("\n===== Evaluation Summary =====")
    print(f"Model: {MODEL_PATH}")
    print(f"Samples processed: {processed}/{total}")
    print(f"Avg latency: {avg_latency:.2f}s per sample")
    print(f"Failed samples: {failed}")
    
    logger.info("Evaluation Summary")
    logger.info(f"Model: {MODEL_PATH}")
    logger.info(f"Samples processed: {processed}/{total}")
    logger.info(f"Avg latency: {avg_latency:.2f}s | Failed: {failed}")
    logger.info(f"Saved submission: {OUT_CSV}")
    logger.info(f"Saved latency log: {BENCH_CSV}")


if __name__ == "__main__":
    main()
