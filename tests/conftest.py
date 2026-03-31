"""Pytest configuration — limit PyTorch thread parallelism to avoid CPU saturation."""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import torch

torch.set_num_threads(1)
