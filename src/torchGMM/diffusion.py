"""
Analytical Diffusion with Time-Dependent Gaussian Mixture Model

This module implements a time-dependent Gaussian Mixture Model (GMM) that:
1. Starts with two differently weighted modes at t=0
2. Evolves through the forward SDE: dX_t = -1/2 * β(t) * X_t dt + √β(t) dW_t
3. Ends in a unimodal normal distribution at t=1

The score is computed using autograd of the log probability.
"""

import torch
from tqdm import tqdm

from typing import Any


def forward_diffusion(
    schedule: torch.nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Simulate the forward diffusion process using Euler-Maruyama:
    dX_t = -1/2 * β(t) * X_t dt + √β(t) dW_t

    Args:
        potential_sampler: Time-dependent GMM model
        x: Initial samples [*BS, D] at time t (optional if n_samples provided)
        t: [T] for the time steps to simulate the diffusion
    """

    assert t.dim() == 1
    trajectory = [x.clone()]
    dt_ = t[1:] - t[:-1]  # t_{k+1} - t_k
    for t_, dt_ in zip(t[:-1], dt_):
        # Get beta value for current time step
        beta_t = schedule.beta(t_)  # [n_samples, 1]

        # Compute drift and diffusion terms for forward SDE
        drift = -0.5 * beta_t * x
        diffusion = torch.sqrt(beta_t)
        noise = torch.randn_like(x, device=x.device)
        assert x.shape == drift.shape, f"x.shape: {x.shape}, drift.shape: {drift.shape}"
        # Apply Euler-Maruyama step: x_{t+dt} = x_t + drift*dt + diffusion*sqrt(dt)*noise
        x = x + drift * dt_ + diffusion * torch.sqrt(torch.tensor(dt_, device=x.device)) * noise
        trajectory.append(x.clone())

    # Stack all trajectory steps and create time indices
    trajectory_tensor = torch.stack(trajectory)  # [n_steps+1, n_samples, 1]
    assert trajectory_tensor.shape == (
        len(t),
        *x.shape,
    ), f"trajectory_tensor.shape: {trajectory_tensor.shape}, x.shape: {x.shape}"
    return trajectory_tensor


def reverse_diffusion(
    schedule: torch.nn.Module,
    score_fn: callable,
    x: torch.Tensor,
    t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, list, int]:
    """
    Simulate the reverse diffusion process using the score function:
    dX_t = [1/2 * β(t) * X_t + β(t) * ∇_x log p_t(x)] dt + √β(t) dW_t
    """

    assert t.dim() == 1
    trajectory = [x.clone()]
    dt_ = t[1:] - t[:-1]  # t_{k+1} - t_k
    for t_, dt_ in zip(t[:-1], dt_):
        # Get beta value for current time step
        assert torch.all(dt_ < 0), f"dt_ must be negative, got {dt_}"
        beta_t = schedule.beta(t_)  # [n_samples, 1]
        diffusion = torch.sqrt(beta_t)
        # Compute drift and diffusion terms for forward SDE
        score = score_fn(x, t_)
        assert x.shape == score.shape, f"x.shape: {x.shape}, score.shape: {score.shape}"
        drift = -0.5 * beta_t * x - diffusion**2 * score

        noise = torch.randn_like(x, device=x.device)
        # Apply Euler-Maruyama step: x_{t+dt} = x_t + drift*dt + diffusion*sqrt(dt)*noise
        x = x + drift * dt_ + diffusion * torch.sqrt(dt_.abs()) * noise

        trajectory.append(x.clone())

    # Stack all trajectory steps and create time indices
    trajectory_tensor = torch.stack(trajectory)  # [n_steps+1, n_samples, 1]
    # print(trajectory_tensor.shape)
    assert trajectory_tensor.shape == (
        len(t),
        *x.shape,
    ), f"trajectory_tensor.shape: {trajectory_tensor.shape}, x.shape: {x.shape}"
    return trajectory_tensor


def compute_ess_from_log_weights(log_weight: torch.Tensor, n_particles: int) -> tuple[torch.Tensor, torch.Tensor]:
    # Compute ESS from log_weights for particles in a group
    n_samples = log_weight.shape[0]
    assert n_samples % n_particles == 0, "n_samples must be multiple of n_particles"
    n_groups = n_samples // n_particles
    unnormalized_weight = torch.exp(torch.nn.functional.log_softmax(log_weight.view(n_groups, n_particles), dim=-1))
    normalized_weight = unnormalized_weight / (unnormalized_weight.sum(dim=-1, keepdim=True) + 1e-12)
    ess = 1.0 / (normalized_weight**2).sum(dim=-1)
    ess = (ess / n_particles).mean()  # average over groups
    return ess, normalized_weight
