from datasets import Dataset, DatasetDict
import json
import random
from pathlib import Path
from src.utils.constants import resolve_video_path, PROCESSED_DATA_DIR


def format_prompt(question: str, choices: list[str]) -> str:
    """
    Tạo prompt nhất quán cho training/inference.
    Nếu các lựa chọn đã có tiền tố "A.", "B." trong dữ liệu thì giữ nguyên.
    Nếu chưa có, tự động thêm.
    """
    formatted_choices = []
    for i, c in enumerate(choices):
        c = c.strip()
        if not c[:2].startswith(f"{chr(65 + i)}."):
            c = f"{chr(65 + i)}. {c}"
        formatted_choices.append(c)

    choices_text = "\n".join(formatted_choices)
    return (
        f"<question>\n{question.strip()}\n</question>\n"
        f"<choices>\n{choices_text}\n</choices>\n"
        f"Trả lời (A/B/C/D):"
    )


def load_traffic_dataset(json_path: str, shuffle=True, seed=42, validate_videos=False):
    """
    Load dataset từ JSON, chia train/val.
    
    Args:
        json_path: Path to training JSON file
        shuffle: Whether to shuffle the data
        seed: Random seed for reproducibility
        validate_videos: If True, check if video files exist (slower but safer)
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)["data"]

    samples = []
    skipped = 0
    
    for item in data:
        try:
            q = item["question"]
            ch = item["choices"]
            ans = item["answer"]
            vid = item["video_path"]
            support_frames = item.get("support_frames", [])

            # Validate video exists if requested
            if validate_videos:
                try:
                    video_path = resolve_video_path(vid)
                    if not video_path.exists():
                        print(f"[WARNING] Video not found for {item['id']}: {vid}")
                        skipped += 1
                        continue
                except Exception as e:
                    print(f"[WARNING] Could not resolve video path for {item['id']}: {vid}")
                    skipped += 1
                    continue

            prompt = format_prompt(q, ch)

            samples.append({
                "id": item["id"],
                "prompt": prompt,
                "response": ans.strip(),
                "video_path": str(Path(vid)),  # Keep original path, will resolve later
                "support_frames": support_frames
            })
        except Exception as e:
            print(f"[ERROR] Skipping item {item.get('id', 'unknown')}: {e}")
            skipped += 1
            continue
    
    print(f"✅ Loaded {len(samples)} samples (skipped {skipped})")

    if shuffle:
        random.seed(seed)
        random.shuffle(samples)

    ds = Dataset.from_list(samples)
    dataset = ds.train_test_split(test_size=0.1, seed=seed)

    return DatasetDict({
        "train": dataset["train"],
        "validation": dataset["test"]
    })


if __name__ == "__main__":
    import argparse, os
    parser = argparse.ArgumentParser(description="Build and optionally save the traffic dataset")
    parser.add_argument("--json_path", type=str, default=str(Path("/home/quanpv/project/traffic_buddy/data/raw/train.json")), help="Path to training JSON")
    parser.add_argument("--out_dir", type=str, default=str(PROCESSED_DATA_DIR), help="Output directory to save dataset (defaults to data/processed)")
    parser.add_argument("--validate_videos", action="store_true", help="Validate video paths exist while building")
    args = parser.parse_args()

    ds = load_traffic_dataset(args.json_path, validate_videos=args.validate_videos)
    print(ds)
    print({k: len(v) for k, v in ds.items()})

    # Save to disk
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out_dir))
    print(f"✅ Dataset saved to: {out_dir}")
