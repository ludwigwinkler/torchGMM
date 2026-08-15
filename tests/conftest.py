"""Pytest configuration.

Two threads per PyTorch/BLAS process and one pytest-xdist worker per two CPU cores
use the available compute without oversubscribing it.
"""

import os

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; suppresses plot windows in tests

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

import torch  # noqa: E402

torch.set_num_threads(2)


def pytest_xdist_auto_num_workers(config):
    """Use all but four available CPU cores for `pytest -n auto`."""
    return max(1, (os.cpu_count() or 1) - 4)


def get_local_device() -> torch.device:
    """Best available device: CUDA → MPS → CPU. Plain function so it can be called
    at module-parse time in @pytest.mark.parametrize."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
