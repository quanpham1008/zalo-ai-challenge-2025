import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def detect_scene_changes_hist(frames_np, threshold=0.5):
    """Nhanh: so sánh histogram giữa các frame."""
    if len(frames_np) <= 1:
        return [0]
    change_indices = [0]
    prev_hist = cv2.calcHist([frames_np[0]], [0, 1, 2], None, [8, 8, 8], [0, 256]*3)
    prev_hist = cv2.normalize(prev_hist, prev_hist).flatten()

    for i in range(1, len(frames_np)):
        hist = cv2.calcHist([frames_np[i]], [0, 1, 2], None, [8, 8, 8], [0, 256]*3)
        hist = cv2.normalize(hist, hist).flatten()
        sim = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
        if sim < threshold:
            change_indices.append(i)
        prev_hist = hist
    return change_indices


def detect_scene_changes_ssim(frames_np, threshold=0.85):
    """Chính xác hơn: dùng SSIM để phát hiện thay đổi nhỏ."""
    if len(frames_np) <= 1:
        return [0]
    change_indices = [0]
    prev_gray = cv2.cvtColor(frames_np[0], cv2.COLOR_RGB2GRAY)
    for i in range(1, len(frames_np)):
        gray = cv2.cvtColor(frames_np[i], cv2.COLOR_RGB2GRAY)
        score, _ = ssim(prev_gray, gray, full=True)
        if score < threshold:
            change_indices.append(i)
        prev_gray = gray
    return change_indices


def detect_scene_changes_hybrid(frames_np, hist_th=0.6, ssim_th=0.9):
    """Kết hợp histogram + SSIM để cân bằng tốc độ & độ chính xác."""
    if len(frames_np) <= 1:
        return [0]
    idx_hist = detect_scene_changes_hist(frames_np, threshold=hist_th)
    selected = [0]
    for i in idx_hist[1:]:
        try:
            score, _ = ssim(
                cv2.cvtColor(frames_np[i-1], cv2.COLOR_RGB2GRAY),
                cv2.cvtColor(frames_np[i], cv2.COLOR_RGB2GRAY),
                full=True
            )
            if score < ssim_th:
                selected.append(i)
        except Exception:
            continue
    return sorted(set(selected))


def select_scene_frames(frames_np, method="hybrid", max_frames=8):
    """Chọn ra frame đại diện theo scene change."""
    if len(frames_np) == 0:
        return [0]

    if method == "hist":
        indices = detect_scene_changes_hist(frames_np)
    elif method == "ssim":
        indices = detect_scene_changes_ssim(frames_np)
    else:
        indices = detect_scene_changes_hybrid(frames_np)

    # Fallback nếu quá ít frame
    if len(indices) < max_frames:
        lin = np.linspace(0, len(frames_np) - 1, max_frames, dtype=int)
        indices.extend(lin.tolist())

    # Giới hạn số lượng
    indices = sorted(set(indices))[:max_frames]
    return indices
