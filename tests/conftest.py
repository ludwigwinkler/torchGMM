"""Pytest configuration.

Single-thread PyTorch + BLAS. pytest-xdist spawns one worker per core; leaving
PyTorch/BLAS multithreaded on top would oversubscribe the CPU and dominate runtime.
"""

import os

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; suppresses plot windows in tests

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import torch  # noqa: E402

torch.set_num_threads(1)


def get_local_device() -> torch.device:
    """Best available device: CUDA → MPS → CPU. Plain function so it can be called
    at module-parse time in @pytest.mark.parametrize."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
