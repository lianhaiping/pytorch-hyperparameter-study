"""Device selection, determinism settings, and environment logging."""

import os

# Required by torch.use_deterministic_algorithms for cuBLAS on CUDA. It must be set
# before the CUDA context is created; setting it at import time guarantees that.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import platform
import random
import sys

import numpy as np
import sklearn
import torch


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False")
    if name not in ("cpu", "cuda"):
        raise ValueError(f"unknown device '{name}'; use auto, cpu, or cuda")
    return torch.device(name)


def set_determinism(seed: int, num_threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # seeds the CPU and all CUDA generators
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(num_threads)


def sync(device: torch.device) -> None:
    """CUDA runs asynchronously; synchronize before reading a timer."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def environment_info(device: torch.device) -> dict:
    """Software/hardware environment plus the current thread and determinism settings.
    Call after set_determinism() so the settings reflect what each training run uses."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_num_threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "default_dtype": str(torch.get_default_dtype()),
    }
