import platform
from subprocess import check_output
from typing import Dict

import mlx.core as mx
import mlx_lm


def get_software_info() -> Dict:
    """Get an overview of SW info."""
    software_info = dict(
        platform=str(platform.platform()),
        python_version=str(platform.python_version()),
        **get_mlx_version(),
        **get_ollama_version(),
    )

    torch_ver = get_torch_version()
    if torch_ver:
        software_info.update(torch_ver)

    lmstudio_ver = get_lmstudio_version()
    if lmstudio_ver:
        software_info.update(lmstudio_ver)

    return software_info


def get_torch_version() -> Dict:
    """Get the current torch version. Returns empty dict if not installed."""
    try:
        import torch
        return dict(torch_version=torch.__version__)
    except ImportError:
        return {}


def get_mlx_version() -> Dict:
    """Get the current mlx version and version of important dependencies."""
    return dict(
        mlx_version=mx.__version__,
        mlx_lm_version=mlx_lm.__version__,
    )


def get_lmstudio_version() -> Dict:
    """Get the current lmstudio version. Returns empty dict if not installed."""
    try:
        import lmstudio
        return dict(lmstudio_version=lmstudio.__version__)
    except ImportError:
        return {}


def get_ollama_version() -> Dict:
    try:
        output = check_output(["ollama", "--version"], text=True).strip()
        if output.startswith("ollama version is "):
            version = output.split("ollama version is ", 1)[1].strip()
            return {"ollama_version": version}
        else:
            return {"ollama_version": "not running"}
    except Exception:
        return {"ollama_version": "not installed / unavailable"}
