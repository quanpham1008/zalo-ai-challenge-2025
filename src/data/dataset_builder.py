from datasets import Dataset, DatasetDict
import json
import random
from pathlib import Path

def format_prompt(question: str, choices: list[str]) -> str:
    """Tạo prompt chuẩn nhất quán cho train/infer."""
    choices_text = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
    return f"<question>\n{question}\n</question>\n<choices>\n{choices_text}\n</choices>\nTrả lời (A/B/C/D):"

def load_traffic_dataset(json_path: str, shuffle=True, seed=42):
    with open(json_path, "r") as f:
        data = json.load(f)["data"]

    samples = []
    for item in data:
        q = item["question"]
        ch = item["choices"]
        ans = item["answer"]
        vid = item["video_path"]
        support_frames = item.get("support_frames", [])

        prompt = format_prompt(q, ch)

        samples.append({
            "id": item["id"],
            "prompt": prompt,
            "response": ans.strip(),
            "video_path": str(Path(vid)),
            "support_frames": support_frames
        })

    if shuffle:
        random.seed(seed)
        random.shuffle(samples)

    ds = Dataset.from_list(samples)
    # Tách 90% train / 10% val
    dataset = ds.train_test_split(test_size=0.1, seed=seed)

    dataset_dict = DatasetDict({
        "train": dataset["train"],
        "validation": dataset["test"]
    })
    return dataset_dict


if __name__ == "__main__":
    dataset = load_traffic_dataset("/home/quanpv/project/traffic_buddy/train/train.json")
    print(dataset)
    print(dataset["train"][0])