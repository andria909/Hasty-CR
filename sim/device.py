"""Select a PyTorch accelerator consistently across training entry points."""

from __future__ import annotations


def torch_device(requested: str = "auto") -> str:
    """Return ``cuda``, ``mps`` or ``cpu`` and validate explicit requests.

    Apple Silicon was previously forced onto CPU because the project only
    checked CUDA.  Keeping the check in one place also prevents evaluation
    from accidentally using a different backend than training.
    """
    import torch

    requested = requested.lower()
    if requested not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError(f"unknown device {requested!r}; use auto/cpu/cuda/mps")
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return requested
