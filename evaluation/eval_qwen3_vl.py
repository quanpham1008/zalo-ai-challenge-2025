import os
import json
import csv
import re
import time
import logging
from datetime import datetime
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from decord import VideoReader, cpu
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from evaluation.prompt import SYSTEM_PROMPT
# ========================
# Config
# ========================
TEST_JSON = "/home/quanpv/project/traffic_buddy/public_test/public_test.json"
OUT_CSV = "/home/quanpv/project/traffic_buddy/public_test/submission_qwen3vl.csv"
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"

NUM_FRAMES = 6
FRAME_SIZE = (448, 448)
MAX_NEW_TOKENS = 4
TEMPERATURE = 0.0

# ========================
# Helper functions
# ========================

def sample_frames(video_path: str, num_frames: int = 6, size=(448, 448)) -> Tuple[List[Image.Image], List[int]]:
    """Trích đều num_frames từ video bằng decord, trả về (images, frame_indices)."""
    if not os.path.exists(video_path):
        return [], []
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
    except Exception:
        return [], []
    total = len(vr)
    if total == 0:
        return [], []
    indices = np.linspace(0, total - 1, num=min(num_frames, total), dtype=int)
    frames = vr.get_batch(indices).asnumpy()  # (N, H, W, 3)
    images = []
    for f in frames:
        img = Image.fromarray(f).resize(size)
        images.append(img)
    return images, indices.tolist()


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


def build_messages(question: str, choices: List[str], frames: List[Image.Image]):
    """Tạo prompt và nội dung cho Qwen3-VL"""
    allowed_letters = [c.split(".")[0].strip() for c in choices if "." in c[:3]]
    allowed_letters = [x for x in allowed_letters if len(x) == 1 and x.isalpha()]
    allowed_letters = sorted(set(allowed_letters))
    choice_text = "\n".join(choices)
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"<question>\n{question}\n</question>\n\n"
        f"<choices>\n{choice_text}\n</choices>\n\n"
        f"Answer with exactly one letter among {{{', '.join(allowed_letters)}}}."
    )
    content = [{"type": "text", "text": prompt}] + [{"type": "image", "image": img} for img in frames]
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "Bạn là trợ lý lái xe thông minh, chuyên trả lời câu hỏi trắc nghiệm về giao thông Việt Nam."}]},
        {"role": "user", "content": content}
    ]
    return messages, allowed_letters


def extract_letter(text: str, allowed: List[str]) -> str:
    """Lấy ra ký tự đáp án hợp lệ từ output"""
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


# ========================
# Main
# ========================

def main():
    # Device setup
    device_map = "auto"
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    logger = setup_logger("eval_qwen3_vl", os.path.join(os.path.dirname(__file__), "logs"))
    logger.info(f"Model: {MODEL_ID}")
    logger.info(f"NUM_FRAMES={NUM_FRAMES}, FRAME_SIZE={FRAME_SIZE}, MAX_NEW_TOKENS={MAX_NEW_TOKENS}, TEMP={TEMPERATURE}")
    logger.info(f"CUDA available={torch.cuda.is_available()}, device_map={device_map}, dtype={torch_dtype}")
    if torch.cuda.is_available():
        try:
            logger.info(f"CUDA device count={torch.cuda.device_count()}, name={torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA capability={torch.cuda.get_device_capability(0)}")
        except Exception as e:
            logger.warning(f"Could not query CUDA device info: {e}")

    print(f"Loading model {MODEL_ID} ...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()
    model.generation_config.do_sample = False
    model.config.use_cache = True

    # Load test set
    with open(TEST_JSON, "r") as f:
        items = json.load(f)["data"]

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    total = len(items)
    print(f"Total test samples: {total}")
    logger.info(f"Total test samples: {total}")

    with open(OUT_CSV, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["id", "answer"])

        for item in tqdm(items, total=total, desc="Evaluating"):
            video_path = item["video_path"]
            question = item["question"]
            choices = item["choices"]

            t0 = time.time()
            if not os.path.exists(video_path):
                logger.warning(f"Missing video: {video_path}. Defaulting to 'A' for id={item['id']}")
                writer.writerow([item["id"], "A"])
                continue

            frames, frame_indices = sample_frames(video_path, NUM_FRAMES, FRAME_SIZE)
            t_vid = time.time() - t0
            if not frames:
                logger.warning(f"No frames extracted from {video_path}. Defaulting to 'A' for id={item['id']}")
                writer.writerow([item["id"], "A"])
                continue

            logger.info(f"id={item['id']} video={video_path} frames={len(frames)} indices={frame_indices}")
            messages, allowed_letters = build_messages(question, choices, frames)
            chat_text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(text=[chat_text], images=frames, return_tensors="pt").to(model.device)

            t1 = time.time()
            with torch.no_grad():
                gen_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=max(TEMPERATURE, 1e-5),
                    top_p=1.0,
                    repetition_penalty=1.0,
                    eos_token_id=processor.tokenizer.eos_token_id,
                )
            t_infer = time.time() - t1
            t_total = time.time() - t0

            gen_text = processor.batch_decode(
                gen_ids[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True
            )[0]
            answer_letter = extract_letter(gen_text, allowed_letters)
            writer.writerow([item["id"], answer_letter])
            logger.info(
                f"id={item['id']} answer={answer_letter} raw='{gen_text.strip()}' times(s): video={t_vid:.3f} infer={t_infer:.3f} total={t_total:.3f}"
            )

    print(f"\n✅ Saved submission to: {OUT_CSV}")
    logger.info(f"Saved submission to: {OUT_CSV}")


if __name__ == "__main__":
    main()
