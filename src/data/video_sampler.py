import os
import sys
from pathlib import Path
import decord
import numpy as np
from PIL import Image
from typing import List, Tuple, Union
from src.data.scene_change_detector import select_scene_frames


def sample_frames(
    video_path: str,
    support_frames=None,
    num_frames: int = 8,
    size: Tuple[int, int] = (448, 448),
    method: str = "hybrid",
    return_format: str = "numpy"
) -> Union[Tuple[np.ndarray, List[int]], Tuple[List[Image.Image], List[int]]]:
    """
    Lấy mẫu frame thông minh từ video bằng scene change detection.
    Có fallback khi video lỗi hoặc không đọc được.
    """

    if not os.path.exists(video_path):
        print(f"[WARNING] Video file not found: {video_path}")
        return ([], []) if return_format == "pil" else (np.array([]), [])

    try:
        vr = decord.VideoReader(video_path)
        total_frames = len(vr)
        fps = vr.get_avg_fps() if hasattr(vr, "get_avg_fps") else 25.0
        if not np.isfinite(fps) or fps <= 0:
            fps = 25.0
    except Exception as e:
        print(f"[WARNING] Cannot open video {video_path}: {e}")
        return ([], []) if return_format == "pil" else (np.array([]), [])

    if total_frames <= 1:
        print(f"[WARNING] Empty or too short video: {video_path}")
        return ([], []) if return_format == "pil" else (np.array([]), [])

    step = max(total_frames // 30, 1)
    sample_idx = list(range(0, total_frames, step))

    try:
        frames_np = vr.get_batch(sample_idx).asnumpy()
    except Exception as e:
        print(f"[WARNING] Failed to extract frames from {video_path}: {e}")
        return ([], []) if return_format == "pil" else (np.array([]), [])

    # Detect scene changes
    scene_idx = select_scene_frames(frames_np, method=method, max_frames=num_frames - 1)

    # Thêm support frame (timestamp → frame index)
    if support_frames:
        try:
            # Flatten in case of nested lists and keep only numeric values
            flat = []
            for sf in support_frames:
                if isinstance(sf, (list, tuple)):
                    for s in sf:
                        flat.append(s)
                else:
                    flat.append(sf)
            support_indices = [int(float(s) * fps) for s in flat if isinstance(s, (int, float, str))]
            scene_idx.extend(support_indices)
        except Exception as e:
            print(f"[WARNING] Support frame error: {e}")

    scene_idx = sorted(set(scene_idx))[:num_frames]
    original_indices = [sample_idx[min(i, len(sample_idx)-1)] for i in scene_idx]

    try:
        selected_frames = vr.get_batch(original_indices).asnumpy()
    except Exception as e:
        print(f"[WARNING] Failed to get selected frames: {e}")
        return ([], []) if return_format == "pil" else (np.array([]), [])

    if return_format == "pil":
        pil_images = []
        for frame in selected_frames:
            try:
                img = Image.fromarray(frame).convert("RGB").resize(size, Image.Resampling.LANCZOS)
                pil_images.append(img)
            except Exception as e:
                print(f"[WARNING] Frame convert failed: {e}")
        return pil_images, original_indices

    return selected_frames, original_indices
