#!/usr/bin/env python
"""
Prepare the traffic dataset and save it to data/processed.

Usage:
  python scripts/prepare_dataset.py [--json_path PATH] [--out_dir PATH] [--validate_videos]

Defaults:
  json_path: data/raw/train.json
  out_dir:   data/processed
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.constants import TRAIN_JSON, PROCESSED_DATA_DIR
from src.data.dataset_builder import load_traffic_dataset


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, default=str(TRAIN_JSON))
    parser.add_argument("--out_dir", type=str, default=str(PROCESSED_DATA_DIR))
    parser.add_argument("--validate_videos", action="store_true")
    args = parser.parse_args()

    print(f"📦 Building dataset from: {args.json_path}")
    dataset = load_traffic_dataset(args.json_path, validate_videos=args.validate_videos)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(out_dir))
    print(f"✅ Dataset saved to: {out_dir}")


if __name__ == "__main__":
    main()


