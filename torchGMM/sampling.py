"""
SDE/ODE sampling via Euler-Maruyama.

Two functions: forward_sampling (t increasing) and reverse_sampling (t decreasing).
The user constructs drift and diffusion Callables ad-hoc before calling.
"""

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
        diffusion: (t) -> scalar diffusion coefficient; None = ODE (no noise)
        x: Initial state [*batch, D]
        t: 1D strictly decreasing time grid in [0, 1], >= 2 points

    Returns:
        Trajectory [T, *batch, D]
    """
    _validate_time_grid(t)
    if not torch.all(t[1:] < t[:-1]):
        raise ValueError("t must be strictly decreasing for reverse_sampling")
    return euler_maruyama(drift, diffusion, x, t)


def _ess_ratio(log_w: torch.Tensor) -> float:
    lw = log_w - torch.logsumexp(log_w, 0)
    return (torch.exp(-torch.logsumexp(2 * lw, 0)) / log_w.shape[0]).item()


def _systematic_resample(log_w: torch.Tensor) -> torch.Tensor:
    w = torch.softmax(log_w, dim=0)
    N = w.shape[0]
    cdf = torch.cumsum(w, 0)
    u = (torch.arange(N, dtype=w.dtype, device=w.device) + torch.rand(1, device=w.device)) / N
    return torch.searchsorted(cdf, u).clamp(max=N - 1)


@jaxtyped(typechecker=beartype)
def steered_reverse_sampling(
    drift: Callable,
    diffusion: Callable | None,
    weight_update: Callable,
    x: Float[Tensor, "N *rest D"],
    t: Float[Tensor, " T"],
    ess_threshold: float = 0.5,
) -> tuple[Float[Tensor, "T N *rest D"], list[float]]:
    """Reverse SDE/ODE sampling with SMC particle correction via importance resampling.

    Args:
        drift:          (x, t) -> [*shape, D]
        diffusion:      (t) -> scalar; None = ODE
        weight_update:  (x, t, dt) -> [N] incremental log weight per particle
        x:              [N, *rest, D] initial state; N = number of particles
        t:              1D strictly decreasing time grid in [0, 1], >= 2 points
        ess_threshold:  resample when ESS/N drops below this value

    Returns:
        trajectory:     [T, N, *rest, D]
        ess_history:    ESS/N at each step, len = len(t) - 1
    """
    _validate_time_grid(t)
    if not torch.all(t[1:] < t[:-1]):
        raise ValueError("t must be strictly decreasing for steered_reverse_sampling")

    log_w = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
    trajectory = [x.clone()]
    ess_history: list[float] = []
    for step, (t_curr, t_next) in enumerate(zip(t[:-1], t[1:])):
        dt = t_next - t_curr
        x_prev = x
        x = x + drift(x, t_curr) * dt
        if diffusion is not None:
            x = x + diffusion(t_curr) * torch.sqrt(dt.abs()) * torch.randn_like(x)
        log_w = log_w + weight_update(x_prev, t_curr, dt)

        ess = _ess_ratio(log_w)
        ess_history.append(ess)

        if ess < ess_threshold:
            idx = _systematic_resample(log_w)
            x = x[idx]
            log_w = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)

        trajectory.append(x.clone())

    # Final resample according to accumulated weights
    idx = _systematic_resample(log_w)
    x = x[idx]
    trajectory[-1] = x.clone()

    return torch.stack(trajectory), ess_history


@jaxtyped(typechecker=beartype)
def compute_ess_from_log_weights(
    log_weight: Float[Tensor, " n_samples"],
    n_particles: int,
) -> tuple[Float[Tensor, ""], Float[Tensor, "n_groups n_particles"]]:
    """Compute the effective sample size (ESS) from unnormalized log importance weights.

    Splits log_weight into groups of n_particles, normalizes each group via softmax,
    and computes ESS = 1 / sum(w²) per group (Kish 1965). Returns the mean ESS across
    groups, normalized by n_particles so ESS=1.0 means all particles are equally weighted.

    Args:
        log_weight: [n_samples] unnormalized log importance weights. n_samples must be
                    divisible by n_particles.
        n_particles: number of particles per group.

    Returns:
        ess:              scalar in (0, 1], mean normalized ESS across groups.
        normalized_weight: [n_groups, n_particles] normalized weights per group.
    """
    n_samples = log_weight.shape[0]
    assert n_samples % n_particles == 0, "n_samples must be multiple of n_particles"
    n_groups = n_samples // n_particles
    normalized_weight = torch.softmax(log_weight.view(n_groups, n_particles), dim=-1)
    ess = 1.0 / (normalized_weight**2).sum(dim=-1)
    ess = (ess / n_particles).mean()
    return ess, normalized_weight
