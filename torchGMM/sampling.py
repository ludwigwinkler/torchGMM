"""SDE/ODE sampling via Euler-Maruyama and unsteered operator splitting."""

import torch
from beartype import beartype
from beartype.typing import Callable
from jaxtyping import Float, jaxtyped
from torch import Tensor


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


@jaxtyped(typechecker=beartype)
def euler_maruyama(
    drift: Callable,
    diffusion: Callable | None,
    x: Float[Tensor, "*batch D"],
    t: Float[Tensor, " T"],
) -> Float[Tensor, "T *batch D"]:
    """Shared Euler-Maruyama loop. No direction validation — callers handle that."""
    trajectory = [x.clone()]
    for t_curr, dt in zip(t[:-1], t[1:] - t[:-1]):
        x = x + drift(x, t_curr) * dt
        if diffusion is not None:
            x = x + diffusion(t_curr) * torch.sqrt(dt.abs()) * torch.randn_like(x)
        trajectory.append(x.clone())
    return torch.stack(trajectory)


@jaxtyped(typechecker=beartype)
def forward_sampling(
    drift: Callable,
    diffusion: Callable | None,
    x: Float[Tensor, "*batch D"],
    t: Float[Tensor, " T"],
) -> Float[Tensor, "T *batch D"]:
    """Forward SDE/ODE sampling (t increasing): dx = drift(x,t) dt + diffusion(t) dW.

    Args:
        drift: (x, t) -> same shape as x
        diffusion: (t) -> scalar diffusion coefficient; None = ODE (no noise)
        x: Initial state [*batch, D]
        t: 1D strictly increasing time grid in [0, 1], >= 2 points

    Returns:
        Trajectory [T, *batch, D]
    """
    _validate_time_grid(t)
    if not torch.all(t[1:] > t[:-1]):
        raise ValueError("t must be strictly increasing for forward_sampling")
    return euler_maruyama(drift, diffusion, x, t)


@jaxtyped(typechecker=beartype)
def reverse_sampling(
    drift: Callable,
    diffusion: Callable | None,
    x: Float[Tensor, "*batch D"],
    t: Float[Tensor, " T"],
) -> Float[Tensor, "T *batch D"]:
    """Reverse SDE/ODE sampling (t decreasing): dx = drift(x,t) dt + diffusion(t) dW.

    Args:
        drift: (x, t) -> same shape as x
        diffusion: (t) -> scalar diffusion coefficient; None = ODE
        x: Initial state [*batch, D]
        t: 1D strictly decreasing time grid in [0, 1], >= 2 points

    Returns:
        Trajectory [T, *batch, D]
    """
    _validate_time_grid(t)
    if not torch.all(t[1:] < t[:-1]):
        raise ValueError("t must be strictly decreasing for reverse_sampling")
    return euler_maruyama(drift, diffusion, x, t)


@jaxtyped(typechecker=beartype)
def reverse_churn_sampling(
    velocity: Callable,
    transition: Callable,
    x: Float[Tensor, "*batch D"],
    t: Float[Tensor, " T"],
    churn: Callable | float = 1.0,
) -> Float[Tensor, "T *batch D"]:
    """Churn sampling: exact re-noising followed by probability-flow transport.

    Args:
        velocity: `(x, t) -> [*batch, D]` probability-flow velocity.
        transition: `(x, t, s) -> [*batch, D]` exact forward kernel.
        x: Initial state `[*batch, D]`.
        t: Strictly decreasing time grid `[T]`.
        churn: Non-negative scalar or `(t) -> float`; reheat size relative to
            the base reverse step.

    Returns:
        Trajectory `[T, *batch, D]`.
    """
    _validate_time_grid(t)
    if not torch.all(t[1:] < t[:-1]):
        raise ValueError("t must be strictly decreasing for reverse_churn_sampling")
    trajectory = [x.clone()]
    for t_curr, dt in zip(t[:-1], t[1:] - t[:-1]):
        churn_value = churn(t_curr) if callable(churn) else churn
        if churn_value < 0:
            raise ValueError(f"churn must be non-negative, got {churn_value}")
        t_hat = (t_curr + churn_value * dt.abs()).clamp(max=1.0)
        if t_hat > t_curr:
            x = transition(x, t_curr, t_hat)
        x = x + velocity(x, t_hat) * (t_curr + dt - t_hat)
        trajectory.append(x.clone())
    return torch.stack(trajectory)
