#!/usr/bin/env python
"""
Validation script to test all paths and configurations before training.
Run this script to verify everything is set up correctly.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.constants import (
    PROJECT_ROOT, TRAIN_JSON, VIDEOS_DIR, TRAFFIC_DATASET_DIR,
    PUBLIC_TEST_JSON, TEST_VIDEOS_DIR, DEFAULT_BASE_MODEL,
    ensure_dirs, resolve_video_path
)
import json


def print_header(text):
    """Print formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def check_paths():
    """Check if all required paths exist."""
    print_header("Checking File Paths")
    
    checks = {
        "Project Root": PROJECT_ROOT,
        "Training JSON": TRAIN_JSON,
        "Videos Directory": VIDEOS_DIR,
        "Public Test JSON": PUBLIC_TEST_JSON,
        "Test Videos Directory": TEST_VIDEOS_DIR,
    }
    
    all_good = True
    for name, path in checks.items():
        exists = path.exists()
        status = "✅" if exists else "❌"
        print(f"{status} {name}: {path}")
        if not exists:
            all_good = False
    
    return all_good


def check_training_data():
    """Validate training data JSON structure."""
    print_header("Validating Training Data")
    
    if not TRAIN_JSON.exists():
        print(f"❌ Training JSON not found: {TRAIN_JSON}")
        return False
    
    try:
        with open(TRAIN_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if "data" not in data:
            print("❌ JSON missing 'data' key")
            return False
        
        samples = data["data"]
        print(f"✅ Training JSON valid: {len(samples)} samples")
        
        # Check first sample structure
        if samples:
            sample = samples[0]
            required_keys = ["id", "question", "choices", "answer", "video_path"]
            missing = [k for k in required_keys if k not in sample]
            
            if missing:
                print(f"❌ Sample missing keys: {missing}")
                return False
            
            print(f"✅ Sample structure valid")
            print(f"   Sample ID: {sample['id']}")
            print(f"   Video path: {sample['video_path']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading JSON: {e}")
        return False


def check_video_paths():
    """Check if video files can be resolved."""
    print_header("Checking Video Path Resolution")
    
    if not TRAIN_JSON.exists():
        print("⏭️  Skipping (no training JSON)")
        return False
    
    try:
        with open(TRAIN_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        samples = data["data"][:5]  # Check first 5 samples
        found = 0
        not_found = 0
        
        for sample in samples:
            video_path_str = sample["video_path"]
            try:
                resolved = resolve_video_path(video_path_str)
                if resolved.exists():
                    print(f"✅ {sample['id']}: {video_path_str} -> {resolved.name}")
                    found += 1
                else:
                    print(f"❌ {sample['id']}: {video_path_str} -> NOT FOUND")
                    not_found += 1
            except Exception as e:
                print(f"❌ {sample['id']}: Failed to resolve - {e}")
                not_found += 1
        
        print(f"\nSummary: {found} found, {not_found} not found (out of {len(samples)} checked)")
        return not_found == 0
        
    except Exception as e:
        print(f"❌ Error checking videos: {e}")
        return False


def check_directories():
    """Check and create necessary directories."""
    print_header("Checking/Creating Directories")
    
    try:
        ensure_dirs()
        print("✅ All necessary directories created/verified")
        print(f"   Dataset dir: {TRAFFIC_DATASET_DIR}")
        print(f"   Dataset exists: {TRAFFIC_DATASET_DIR.exists()}")
        return True
    except Exception as e:
        print(f"❌ Error creating directories: {e}")
        return False


def check_imports():
    """Check if all required modules can be imported."""
    print_header("Checking Python Dependencies")
    
    modules = {
        "torch": "PyTorch",
        "transformers": "HuggingFace Transformers",
        "datasets": "HuggingFace Datasets",
        "peft": "PEFT (LoRA)",
        "trl": "TRL (SFT Trainer)",
        "wandb": "Weights & Biases",
        "ray": "Ray Tune",
        "PIL": "Pillow",
        "cv2": "OpenCV",
    }
    
    all_good = True
    for module, name in modules.items():
        try:
            __import__(module)
            print(f"✅ {name} ({module})")
        except ImportError:
            print(f"❌ {name} ({module}) - NOT INSTALLED")
            all_good = False
    
    return all_good


def test_dataset_builder():
    """Test dataset builder with sample data."""
    print_header("Testing Dataset Builder")
    
    try:
        from src.data.dataset_builder import load_traffic_dataset
        
        if not TRAIN_JSON.exists():
            print("⏭️  Skipping (no training JSON)")
            return False
        
        # Try loading just to test (don't save)
        print("Loading dataset (this may take a moment)...")
        dataset = load_traffic_dataset(str(TRAIN_JSON), validate_videos=False)
        
        print(f"✅ Dataset loaded successfully")
        print(f"   Train samples: {len(dataset['train'])}")
        print(f"   Validation samples: {len(dataset['validation'])}")
        
        # Check sample structure
        sample = dataset['train'][0]
        print(f"   Sample keys: {list(sample.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_summary(results):
    """Print summary of all checks."""
    print_header("Validation Summary")
    
    all_passed = all(results.values())
    
    for check, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {check}")
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All checks passed! You're ready to train.")
    else:
        print("⚠️  Some checks failed. Please fix the issues above.")
    print("=" * 60 + "\n")
    
    return all_passed


def main():
    """Run all validation checks."""
    print("""
╔══════════════════════════════════════════════════════════╗
║        Traffic Buddy - Setup Validation Script          ║
║                                                          ║
║  This script validates your setup before training       ║
╚══════════════════════════════════════════════════════════╝
""")
    
    results = {}
    
    # Run all checks
    results["Path Checks"] = check_paths()
    results["Training Data"] = check_training_data()
    results["Video Resolution"] = check_video_paths()
    results["Directory Setup"] = check_directories()
    results["Python Dependencies"] = check_imports()
    results["Dataset Builder"] = test_dataset_builder()
    
    # Print summary
    all_passed = print_summary(results)
    
    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()

