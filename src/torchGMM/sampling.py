from __future__ import annotations

"""
SDE/ODE sampling via Euler-Maruyama.

Two functions: forward_sampling (t increasing) and reverse_sampling (t decreasing).
The user constructs drift and diffusion callables ad-hoc before calling.
"""

import torch


def _validate_time_grid(t: torch.Tensor) -> None:
    """Validate a 1D time grid for integration."""
    if t.dim() != 1:
        raise ValueError(f"t must be 1D, got {t.shape}")
    if not torch.all(torch.isfinite(t)):
        raise ValueError("t must be finite")
    if not torch.all((t >= 0) & (t <= 1)):
        raise ValueError("t must be within [0, 1]")
    if t.numel() < 2:
        raise ValueError("t must contain at least two time points")


def _euler_maruyama(
    drift: callable,
    diffusion: callable | None,
    x: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Shared Euler-Maruyama loop. No direction validation — callers handle that."""
    trajectory = [x.clone()]
    for t_curr, dt in zip(t[:-1], t[1:] - t[:-1]):
        x = x + drift(x, t_curr) * dt
        if diffusion is not None:
            x = x + diffusion(t_curr) * torch.sqrt(dt.abs()) * torch.randn_like(x)
        trajectory.append(x.clone())
    return torch.stack(trajectory)


def forward_sampling(
    drift: callable,
    diffusion: callable | None,
    x: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Forward SDE/ODE sampling (t increasing): dx = drift(x,t) dt + diffusion(t) dW.

    Args:
        drift: (x, t) -> same shape as x
        diffusion: (t) -> scalar diffusion coefficient; None = ODE (no noise)
        x: Initial state [*shape, D]
        t: 1D strictly increasing time grid in [0, 1], >= 2 points

    Returns:
        Trajectory [len(t), *shape, D]
    """
    _validate_time_grid(t)
    if not torch.all(t[1:] > t[:-1]):
        raise ValueError("t must be strictly increasing for forward_sampling")
    return _euler_maruyama(drift, diffusion, x, t)


def reverse_sampling(
    drift: callable,
    diffusion: callable | None,
    x: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Reverse SDE/ODE sampling (t decreasing): dx = drift(x,t) dt + diffusion(t) dW.

    Args:
        drift: (x, t) -> same shape as x
        diffusion: (t) -> scalar diffusion coefficient; None = ODE (no noise)
        x: Initial state [*shape, D]
        t: 1D strictly decreasing time grid in [0, 1], >= 2 points

    Returns:
        Trajectory [len(t), *shape, D]
    """
    _validate_time_grid(t)
    if not torch.all(t[1:] < t[:-1]):
        raise ValueError("t must be strictly decreasing for reverse_sampling")
    return _euler_maruyama(drift, diffusion, x, t)


def compute_ess_from_log_weights(log_weight: torch.Tensor, n_particles: int) -> tuple[torch.Tensor, torch.Tensor]:
    n_samples = log_weight.shape[0]
    assert n_samples % n_particles == 0, "n_samples must be multiple of n_particles"
    n_groups = n_samples // n_particles
    unnormalized_weight = torch.exp(torch.nn.functional.log_softmax(log_weight.view(n_groups, n_particles), dim=-1))
    normalized_weight = unnormalized_weight / (unnormalized_weight.sum(dim=-1, keepdim=True) + 1e-12)
    ess = 1.0 / (normalized_weight**2).sum(dim=-1)
    ess = (ess / n_particles).mean()
    return ess, normalized_weight
