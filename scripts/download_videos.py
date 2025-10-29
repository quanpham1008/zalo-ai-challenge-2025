import json
import os
import sys
from pathlib import Path
import requests
from tqdm import tqdm

# Ensure project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.constants import (
    PROJECT_ROOT as PRJ_ROOT,
    TRAIN_JSON,
    PUBLIC_TEST_JSON,
    VIDEOS_DIR,
    TEST_VIDEOS_DIR,
)

# Configuration
url_root = "https://dl-challenge.zalo.ai/2025/TrafficBuddyHZUTAXF/"


def download_video(url, save_path):
    """Download a video file with progress bar and verification"""
    temp_path = save_path + ".tmp"

    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Stream download with progress bar to temporary file
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0

        with open(temp_path, 'wb') as f:
            if total_size == 0:
                content = response.content
                f.write(content)
                downloaded_size = len(content)
            else:
                with tqdm(total=total_size, unit='B', unit_scale=True,
                          desc=os.path.basename(save_path)) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        pbar.update(len(chunk))

        # Verify download completed fully
        if total_size > 0 and downloaded_size != total_size:
            print(f"Warning: Incomplete download. Expected {total_size}, got {downloaded_size}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return False

        # Only rename to final path if download was successful
        os.rename(temp_path, save_path)
        return True

    except Exception as e:
        print(f"Error downloading {url}: {e}")
        # Clean up temporary file if it exists
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False


def _resolve_local_path(video_path: str) -> str:
    """Map dataset-relative video path to project storage path.

    - "train/videos/<name>" -> data/raw/videos/<name>
    - "public_test/videos/<name>" -> data/public_test/videos/<name>
    - otherwise -> data/raw/videos/<name>
    """
    video_path = str(video_path)
    basename = os.path.basename(video_path)
    if video_path.startswith("public_test/videos/"):
        return str(TEST_VIDEOS_DIR / basename)
    # default to train/raw bucket
    return str(VIDEOS_DIR / basename)


def download_video_in_file(json_file):
    # Load JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)["data"]

    # Get unique video paths
    video_paths = list(set(x["video_path"] for x in data))
    print(f"Found {len(video_paths)} unique videos to download from {len(data)} records")

    # Download each video
    success_count = 0
    fail_count = 0

    for video_path in video_paths:
        # Construct full URL
        full_url = url_root + video_path

        # Construct local save path (mapped to project layout)
        local_path = _resolve_local_path(video_path)

        # Check if file already exists
        if os.path.exists(local_path):
            print(f"Skipping (already exists): {video_path}")
            success_count += 1
            continue

        print(f"Downloading: {video_path}")

        if download_video(full_url, local_path):
            success_count += 1
        else:
            fail_count += 1

    print(f"\nDownload complete for {json_file}!")
    print(f"Success: {success_count}")
    print(f"Failed: {fail_count}")
    print("-" * 50)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download videos for Traffic Buddy dataset")
    parser.add_argument("--train_json", type=str, default=str(TRAIN_JSON))
    parser.add_argument("--public_test_json", type=str, default=str(PUBLIC_TEST_JSON))
    args = parser.parse_args()

    print(f"Project root: {PRJ_ROOT}")
    print(f"Train JSON: {args.train_json}")
    print(f"Public Test JSON: {args.public_test_json}")
    print(f"Train videos dir: {VIDEOS_DIR}")
    print(f"Public test videos dir: {TEST_VIDEOS_DIR}")

    # Ensure dirs exist
    os.makedirs(str(VIDEOS_DIR), exist_ok=True)
    os.makedirs(str(TEST_VIDEOS_DIR), exist_ok=True)

    # Download
    if os.path.exists(args.public_test_json):
        download_video_in_file(json_file=args.public_test_json)
    else:
        print(f"Skip: public test JSON not found: {args.public_test_json}")
    if os.path.exists(args.train_json):
        download_video_in_file(json_file=args.train_json)
    else:
        print(f"Skip: train JSON not found: {args.train_json}")