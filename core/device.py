"""Device detection, dtype selection, autocast and seeding."""
from __future__ import annotations

import contextlib
import os
import random
from typing import Iterator

import numpy as np
import torch


def resolve_device(preference: str = "auto") -> str:
    """Return 'cuda', 'mps' or 'cpu' honoring a preference with graceful fallback."""
    pref = (preference or "auto").lower()
    has_cuda = torch.cuda.is_available()
    has_mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    if pref == "cuda" and has_cuda:
        return "cuda"
    if pref == "mps" and has_mps:
        return "mps"
    if pref == "cpu":
        return "cpu"
    if pref == "auto":
        if has_cuda:
            return "cuda"
        if has_mps:
            return "mps"
    return "cpu"


def pipeline_device(device: str) -> int | str:
    """transformers.pipeline accepts an int for CUDA or a device string."""
    return 0 if device == "cuda" else device


def diffusion_dtype(device: str, amp: bool) -> torch.dtype:
    return torch.float16 if amp and device in ("cuda", "mps") else torch.float32


@contextlib.contextmanager
def autocast(device: str, enabled: bool = True) -> Iterator[None]:
    """Mixed precision on CUDA only (MPS/CPU autocast is unreliable for these models)."""
    if enabled and device == "cuda":
        with torch.autocast("cuda", dtype=torch.float16):
            yield
    else:
        yield


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def device_summary(device: str) -> str:
    if device == "cuda":
        return f"CUDA · {torch.cuda.get_device_name(0)}"
    if device == "mps":
        return "Apple Silicon GPU (MPS)"
    return f"CPU · {os.cpu_count()} threads"
