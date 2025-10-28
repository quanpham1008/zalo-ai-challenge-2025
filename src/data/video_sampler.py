import os
import decord
import numpy as np
from PIL import Image
from typing import List, Tuple, Union
from src.data.scene_change_detector import select_scene_frames

def sample_frames(video_path: str, support_frames=None, num_frames: int = 8, 
                 size: Tuple[int, int] = (448, 448), method: str = "hybrid", 
                 return_format: str = "numpy") -> Union[Tuple[np.ndarray, List[int]], Tuple[List[Image.Image], List[int]]]:
    """
    Sampling frame thông minh bằng scene change detection.
    
    Args:
        video_path: Path to video file
        support_frames: List of support frame timestamps
        num_frames: Number of frames to sample
        size: Target frame size (width, height)
        method: Scene detection method ("hybrid", "hist", "ssim")
        return_format: "numpy" for numpy arrays, "pil" for PIL Images
        
    Returns:
        Tuple of (frames, frame_indices) where format depends on return_format
    """
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"[WARNING] Video file not found: {video_path}")
        if return_format == "pil":
            return [], []
        else:
            return np.array([]), []
    
    try:
        vr = decord.VideoReader(video_path)
        total_frames = len(vr)
    except Exception as e:
        print(f"[WARNING] Cannot open video {video_path}: {e}")
        if return_format == "pil":
            return [], []
        else:
            return np.array([]), []

    if total_frames <= 0:
        print(f"[WARNING] Empty video: {video_path}")
        if return_format == "pil":
            return [], []
        else:
            return np.array([]), []

    # Lấy subset frame để giảm compute
    step = max(total_frames // 30, 1)
    sample_idx = list(range(0, total_frames, step))
    
    try:
        frames_np = vr.get_batch(sample_idx).asnumpy()
    except Exception as e:
        print(f"[WARNING] Failed to extract frames from {video_path}: {e}")
        if return_format == "pil":
            return [], []
        else:
            return np.array([]), []

    # Detect scene changes
    scene_idx = select_scene_frames(frames_np, method=method, max_frames=num_frames - 1)

    # Thêm support frame nếu có
    if support_frames is not None and len(support_frames) > 0:
        try:
            fps = vr.get_avg_fps()
            support_indices = [int(sf * fps) for sf in support_frames]
            scene_idx.extend(support_indices)
        except Exception as e:
            print(f"[WARNING] Failed to process support frames: {e}")

    # Đảm bảo không vượt quá giới hạn và map về original indices
    scene_idx = sorted(set(scene_idx))[:num_frames]
    original_indices = [sample_idx[min(i, len(sample_idx)-1)] for i in scene_idx]
    
    try:
        selected_frames = vr.get_batch(original_indices).asnumpy()
    except Exception as e:
        print(f"[WARNING] Failed to get selected frames: {e}")
        if return_format == "pil":
            return [], []
        else:
            return np.array([]), []

    if return_format == "pil":
        # Convert to PIL Images
        pil_images = []
        for frame in selected_frames:
            if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
                continue
            try:
                img = Image.fromarray(frame).convert("RGB").resize(size, Image.Resampling.LANCZOS)
                pil_images.append(img)
            except Exception as e:
                print(f"[WARNING] Failed to convert frame to PIL: {e}")
                continue
        return pil_images, original_indices
    else:
        # Return numpy arrays
        return selected_frames, original_indices
