import gc
import time
import torch


def clear_memory():
    """Best-effort cleanup of CPU/GPU memory between trials or after training."""
    try:
        gc.collect()
        time.sleep(0.5)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
    except Exception:
        pass

