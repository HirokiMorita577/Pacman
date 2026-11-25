"""Explicit accelerator selection and reproducible execution metadata."""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch


def device_name(algorithm=None):
    requested = os.environ.get("PACMAN_DEVICE", "cpu").lower()
    if requested not in ("cpu", "cuda", "auto"):
        raise ValueError("PACMAN_DEVICE must be cpu, cuda or auto")
    if algorithm in ("random", "greedy_food", "safe_food"):
        return "cpu"
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA GPU is available")
    return requested


def execution_info(algorithm=None):
    device = device_name(algorithm)
    threads = int(os.environ.get("PACMAN_TORCH_THREADS", "1"))
    if threads < 1:
        raise ValueError("PACMAN_TORCH_THREADS must be positive")
    info = dict(device=device, torch_threads=threads, deterministic=True)
    if device == "cuda":
        info.update(gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda,
                    capability=list(torch.cuda.get_device_capability(0)))
    return info


def configure(algorithm=None):
    info = execution_info(algorithm)
    torch.set_num_threads(info["torch_threads"])
    if info["device"] == "cuda":
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    return info
