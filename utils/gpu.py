"""GPU helpers: free VRAM between stages."""
from __future__ import annotations

import gc


def free() -> None:
    """Release cached VRAM. Safe to call even if torch/CUDA absent."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def has_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False
