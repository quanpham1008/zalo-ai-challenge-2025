"""
Centralized constants and paths for Traffic Buddy project.
"""
from pathlib import Path

# ============================================================
# Project Root & Directory Structure
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"

# ============================================================
# Data Directories
# ============================================================
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PUBLIC_TEST_DIR = DATA_DIR / "public_test"
METADATA_DIR = DATA_DIR / "metadata"

# ============================================================
# Data Files
# ============================================================
TRAIN_JSON = RAW_DATA_DIR / "train.json"
PUBLIC_TEST_JSON = PUBLIC_TEST_DIR / "public_test.json"
VIDEOS_DIR = RAW_DATA_DIR / "videos"
TEST_VIDEOS_DIR = PUBLIC_TEST_DIR / "videos"

# ============================================================
# Dataset Paths
# ============================================================
DATASETS_DIR = PROJECT_ROOT / "datasets"
TRAFFIC_DATASET_DIR = PROCESSED_DATA_DIR  # Dataset is stored in data/processed
# Legacy path (if dataset was saved to datasets/traffic_dataset, will fallback)
LEGACY_DATASET_DIR = DATASETS_DIR / "traffic_dataset"

# ============================================================
# Model & Checkpoint Paths
# ============================================================
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
BASE_MODELS_DIR = CHECKPOINTS_DIR / "base_models"
FINETUNED_DIR = CHECKPOINTS_DIR / "finetuned"

# ============================================================
# Output Paths
# ============================================================
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = OUTPUTS_DIR / "logs"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
BENCHMARK_DIR = OUTPUTS_DIR / "benchmark_results"
RAY_RESULTS_DIR = OUTPUTS_DIR / "ray_results"

# ============================================================
# Default Model IDs
# ============================================================
DEFAULT_BASE_MODEL_2B = "Qwen/Qwen3-VL-2B-Instruct"
DEFAULT_BASE_MODEL_8B = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_BASE_MODEL = DEFAULT_BASE_MODEL_8B  # Default to 8B for better performance

# ============================================================
# Training Constants
# ============================================================
DEFAULT_NUM_FRAMES = 8
DEFAULT_FRAME_SIZE = (448, 448)
DEFAULT_MAX_LENGTH = 1024
DEFAULT_SEED = 42

# ============================================================
# Helper Functions
# ============================================================
def resolve_video_path(video_path: str) -> Path:
    """
    Resolve video path from JSON to actual file location.
    Handles both absolute and relative paths.
    
    Args:
        video_path: Path from JSON (e.g., "train/videos/xxx.mp4")
    
    Returns:
        Resolved absolute path to video file
    """
    video_path = Path(video_path)
    
    # If already absolute and exists, return it
    if video_path.is_absolute() and video_path.exists():
        return video_path
    
    # Try different possible locations
    possible_paths = [
        video_path,  # Try as-is first
        PROJECT_ROOT / video_path,  # Relative to project root
        RAW_DATA_DIR / video_path.name,  # Just filename in raw data dir
        VIDEOS_DIR / video_path.name,  # Just filename in videos dir
    ]
    
    # Handle "train/videos/xxx.mp4" -> "data/raw/videos/xxx.mp4"
    if str(video_path).startswith("train/videos/"):
        video_name = video_path.name
        possible_paths.insert(0, VIDEOS_DIR / video_name)
    
    for path in possible_paths:
        if path.exists():
            return path
    
    # If not found, return the most likely path (will error later with clear message)
    return VIDEOS_DIR / video_path.name


def ensure_dirs():
    """Create necessary directories if they don't exist."""
    dirs_to_create = [
        DATASETS_DIR,
        TRAFFIC_DATASET_DIR,
        CHECKPOINTS_DIR,
        FINETUNED_DIR,
        OUTPUTS_DIR,
        LOGS_DIR,
        PREDICTIONS_DIR,
        BENCHMARK_DIR,
    ]
    
    for directory in dirs_to_create:
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # Test path resolution
    print("Project paths:")
    print(f"  PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"  TRAIN_JSON: {TRAIN_JSON}")
    print(f"  VIDEOS_DIR: {VIDEOS_DIR}")
    print(f"  TRAFFIC_DATASET_DIR: {TRAFFIC_DATASET_DIR}")
    print(f"\nChecking paths:")
    print(f"  TRAIN_JSON exists: {TRAIN_JSON.exists()}")
    print(f"  VIDEOS_DIR exists: {VIDEOS_DIR.exists()}")
    
    # Create directories
    ensure_dirs()
    print(f"\n✅ All necessary directories created")

