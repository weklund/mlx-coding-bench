import os
from typing import Optional

import mlx.core as mx
import psutil


def bytes_to_gib(bytes: int) -> float:
    """Convert bytes to gibibytes."""
    return bytes / (1024**3)


def get_process_memory_gib() -> float:
    """Return the current process' allocated memory in GiB."""
    pid = os.getpid()
    process = psutil.Process(pid)
    memory_info = process.memory_info()
    return bytes_to_gib(memory_info.rss)


def get_available_ram_gib() -> float:
    """Return the available RAM in GiB."""
    ram = psutil.virtual_memory().available
    return bytes_to_gib(ram)


def get_used_ram_gib() -> float:
    """Return the used RAM in GiB."""
    ram = psutil.virtual_memory().used
    return bytes_to_gib(ram)


def get_torch_memory_gib(backend: Optional[str] = None) -> float:
    """Return the memory allocated by torch tensors in GiB.

    Requires torch to be installed. Raises ImportError otherwise.
    """
    import torch

    if backend is None:
        if torch.mps.is_available():
            backend = "mps"
        elif torch.cuda.is_available():
            backend = "cuda"
        else:
            backend = "cpu"

    if backend == "mps":
        mem = torch.mps.current_allocated_memory()
    elif backend == "cuda":
        mem = torch.cuda.memory_allocated()
    else:
        mem = get_process_memory_gib()

    return bytes_to_gib(mem)


def get_mlx_memory_gib() -> float:
    """Return the memory allocated by mlx tensors in GiB."""
    mem = mx.get_active_memory()
    return bytes_to_gib(mem)


def estimate_model_size(
    num_params: int,
    dtype: str,
) -> float:
    """Estimate the model size in GiB."""
    dtype_to_bits = {
        "float32": 32,
        "bfloat16": 16,
        "float16": 16,
        "int8": 8,
        "int6": 6,
        "int4": 4,
        "int3": 3,
    }
    return bytes_to_gib(num_params * dtype_to_bits[dtype] / 8)


def get_lmstudio_memory():
    """Return the memory allocated to LmStudio processes in GiB."""
    memory = dict()
    for proc in psutil.process_iter():
        try:
            if proc.name().lower().replace(" ", "").startswith("lmstudio"):
                memory[proc.name] = bytes_to_gib(proc.memory_info().rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    memory["total"] = sum(memory.values())
    return memory
