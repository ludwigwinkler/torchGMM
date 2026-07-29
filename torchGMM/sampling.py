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


@jaxtyped(typechecker=beartype)
def reverse_churn_sampling(
    velocity: Callable,
    transition: Callable,
    x: Float[Tensor, "*batch D"],
    t: Float[Tensor, " T"],
    churn: float = 1.0,
) -> Float[Tensor, "T *batch D"]:
    """Churn sampling (EDM Alg. 2): re-noise to t+h, then transport back past it to t+dt.

    Not an SDE discretisation but an operator splitting, which is why it is a separate
    function rather than an integrator swapped into `reverse_sampling`. Each step composes
    two operators, each of which maps a distribution to a distribution on its own:

      1. `transition(x, t, t+h)` — the exact forward kernel, h = churn·|dt|. Unlike an
         Euler step of the forward SDE it maps p_t onto p_{t+h} exactly, for any h. That
         exactness is the point: with a first-order churn the whole scheme collapses to
         reverse-SDE Euler-Maruyama up to O(h^{3/2}) and the splitting buys nothing.
      2. `velocity(x, t+h) * (t + dt - (t+h))` — probability-flow transport, spanning
         -(|dt| + h) since it has to undo the churn as well as advance one grid step.

    The velocity is evaluated at the *reheated* time t+h, not at t: after the churn the
    state is distributed according to p_{t+h}, so the score at t is the wrong one for it
    (docs/fkc_churn_steering.md §5). Both are O(h) globally, but the mismatch is a full
    step wide at churn=1 and costs measurable accuracy at low step counts.

    Both preserve the marginal family, so the composition lands on p_{t+dt}. Contrast
    `reverse_sampling`, where drift and diffusion are the two *terms* of one SDE, added at
    the same t over the same dt.

    Reverse-only: on an increasing grid the churn would run with the direction of travel
    instead of against it, and at churn=1 the transport span collapses to zero. Forward
    noising needs no sampler at all — `schedule.transition` jumps t -> s in a single draw.

    Args:
        velocity:   (x, t) -> [*batch, D]; the probability-flow velocity, e.g. `gmm.velocity`.
                    NOT the reverse SDE drift f - g²s — the churn already supplies the
                    stochasticity, so a score-corrected drift double-counts it.
        transition: (x, t, s) -> [*batch, D]; the exact forward kernel, i.e. `schedule.transition`
        x:          Initial state [*batch, D]
        t:          1D strictly decreasing time grid in [0, 1], >= 2 points
        churn:      Churn strength as a multiple of the step size, h = churn·|dt|. This is
                    EDM's S_churn/N knob: 0 is the deterministic probability-flow ODE, 1
                    re-noises a full step back before transporting two steps down, and >1
                    over-churns (the transport then spans more than two steps). Values
                    above 1 can push t+h past 1 for the first steps, where it is clamped
                    and the transport span shrinks to match.

    Returns:
        Trajectory [T, *batch, D]
    """
    _validate_time_grid(t)
    if not torch.all(t[1:] < t[:-1]):
        raise ValueError("t must be strictly decreasing for reverse_churn_sampling")
    if churn < 0:
        raise ValueError(f"churn must be non-negative, got {churn}")
    trajectory = [x.clone()]
    for t_curr, dt in zip(t[:-1], t[1:] - t[:-1]):
        t_hat = (t_curr + churn * dt.abs()).clamp(max=1.0)
        # t̂ == t whenever churn == 0, or when the clamp bites at t == 1; both mean no churn.
        if t_hat > t_curr:
            x = transition(x, t_curr, t_hat)  # exact forward kernel, t -> t̂
        x = x + velocity(x, t_hat) * (t_curr + dt - t_hat)  # PF-ODE transport t̂ -> t+dt
        trajectory.append(x.clone())
    return torch.stack(trajectory)


def _ess_ratio(log_w: torch.Tensor) -> float:
    lw = log_w - torch.logsumexp(log_w, 0)
    return (torch.exp(-torch.logsumexp(2 * lw, 0)) / log_w.shape[0]).item()


def _systematic_resample(log_w: torch.Tensor) -> torch.Tensor:
    w = torch.softmax(log_w, dim=0)
    N = w.shape[0]
    cdf = torch.cumsum(w, 0)
    u = (torch.arange(N, dtype=w.dtype, device=w.device) + torch.rand(1, device=w.device)) / N
    return torch.searchsorted(cdf, u).clamp(max=N - 1)


def _normalized_weights(log_w: torch.Tensor) -> torch.Tensor:
    return torch.softmax(log_w, dim=0)


@jaxtyped(typechecker=beartype)
def steered_reverse_sampling(
    drift: Callable,
    diffusion: Callable | None,
    weight_update: Callable,
    x: Float[Tensor, "N *rest D"],
    t: Float[Tensor, " T"],
    ess_threshold: float | int = 0.5,
) -> tuple[Float[Tensor, "T N *rest D"], list[float], Float[Tensor, "T N"]]:
    """Reverse SDE/ODE sampling with SMC particle correction via importance resampling.

    Args:
        drift:          (x, t) -> [*shape, D]
        diffusion:      (t) -> scalar; None = ODE
        weight_update:  (x, t, dt) -> [N] incremental log weight per particle
        x:              [N, *rest, D] initial state; N = number of particles
        t:              1D strictly decreasing time grid in [0, 1], >= 2 points
        ess_threshold:  Controls the resampling strategy:
                          - 0 < ess_threshold < 1: adaptive — resample whenever
                            ESS/N drops below this value.
                          - ess_threshold >= 1 (must be a whole number): fixed-interval
                            — resample unconditionally every int(ess_threshold) steps,
                            regardless of ESS (ess_threshold == 1 resamples every step).
                          - If the interval is >= len(t) - 1 (the total number of
                            integration steps), no intermittent resample ever fires;
                            weights accumulate over the whole trajectory and only the
                            mandatory resample after the final step (always applied)
                            corrects the particles.

    Returns:
        trajectory:     [T, N, *rest, D]
        ess_history:    ESS/N at each step, len = len(t) - 1
        weight_history: [T, N] normalized particle weights aligned with trajectory;
                        rows sum to 1 and reset to uniform after resampling.
    """
    _validate_time_grid(t)
    if not torch.all(t[1:] < t[:-1]):
        raise ValueError("t must be strictly decreasing for steered_reverse_sampling")
    if ess_threshold <= 0:
        raise ValueError(f"ess_threshold must be positive, got {ess_threshold}")
    if ess_threshold >= 1 and ess_threshold != int(ess_threshold):
        raise ValueError(
            f"ess_threshold >= 1 selects fixed-interval resampling and must be a "
            f"whole number of steps, got {ess_threshold}"
        )
    interval_mode = ess_threshold >= 1
    resample_every = int(ess_threshold) if interval_mode else 1  # unused when not interval_mode

    log_w = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
    trajectory = [x.clone()]
    weight_history = [_normalized_weights(log_w)]
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

        if interval_mode:
            should_resample = (step + 1) % resample_every == 0
        else:
            should_resample = ess < ess_threshold

        if should_resample:
            idx = _systematic_resample(log_w)
            x = x[idx]
            log_w = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)

        trajectory.append(x.clone())
        weight_history.append(_normalized_weights(log_w))

    # Final resample according to accumulated weights
    idx = _systematic_resample(log_w)
    x = x[idx]
    log_w = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
    trajectory[-1] = x.clone()
    weight_history[-1] = _normalized_weights(log_w)

    return torch.stack(trajectory), ess_history, torch.stack(weight_history)


@jaxtyped(typechecker=beartype)
def steered_reverse_churn_sampling(
    velocity: Callable,
    transition: Callable,
    potential: Callable,
    x: Float[Tensor, "N *rest D"],
    t: Float[Tensor, " T"],
    churn: float = 1.0,
    ess_threshold: float | int = 0.5,
) -> tuple[Float[Tensor, "T N *rest D"], list[float], Float[Tensor, "T N"]]:
    """FKC-steered churn sampling: `reverse_churn_sampling` run as an SMC particle filter.

    Same splitting as `reverse_churn_sampling` — exact forward re-noise to t̂, then
    probability-flow transport from t̂ to t+dt — with importance weights and systematic
    resampling layered on top, exactly as `steered_reverse_sampling` layers them onto
    Euler-Maruyama. Targets the tilted marginals p_t(x) ∝ q_t(x)·exp(ρ_t(x)).

    Where `steered_reverse_sampling` takes an *incremental* `weight_update(x, t, dt)`,
    this takes the tilt exponent ρ_t(x) itself, because the churn step is a discrete
    proposal kernel rather than an SDE discretisation (docs/fkc_churn_steering.md §2-3).
    Since the base churn kernel already maps q_t onto q_{t+dt} exactly, the whole
    importance correction is the endpoint difference

        Δlog w = ρ_{t+dt}(x_{t+dt}) − ρ_t(x_t)

    which retargets the pushforward of the old tilted cloud onto the new one. Nothing
    else is needed: no β̇_t, no ∂_t r, no reward Laplacian, no score-alignment inner
    product — all of which the continuous-time weight of Proposition D.6 requires. The
    drift is *not* guided either; the base probability flow is left untouched, which is
    what keeps the correction a plain endpoint difference (§6).

    The increments telescope, so between resamples log w is bounded by the range of ρ
    rather than accumulating like a stochastic integral.

    Args:
        velocity:      (x, t) -> [N, *rest, D]; probability-flow velocity, e.g. `gmm.velocity`.
                       NOT a reward-guided drift — the weights carry the whole tilt.
        transition:    (x, t, s) -> [N, *rest, D]; exact forward kernel, i.e. `schedule.transition`
        potential:     (x, t) -> [N]; the log tilt ρ_t(x) = β(t)·r(x, t). Returns one scalar
                       per particle, already reduced over `*rest` and `D`.
        x:             [N, *rest, D] initial state; N = number of particles
        t:             1D strictly decreasing time grid in [0, 1], >= 2 points
        churn:         Churn strength as a multiple of the step size, h = churn·|dt|; see
                       `reverse_churn_sampling`. 0 makes every step a deterministic PF-ODE
                       step, and the weights then reduce to a pure change-of-target term.
        ess_threshold: Resampling strategy, identical to `steered_reverse_sampling`:
                       adaptive when 0 < ess_threshold < 1 (resample once ESS/N drops
                       below it), fixed-interval when ess_threshold >= 1 (a whole number
                       of steps). A final resample always fires after the last step.

    Returns:
        trajectory:     [T, N, *rest, D]
        ess_history:    ESS/N at each step, len = len(t) - 1
        weight_history: [T, N] normalized particle weights aligned with trajectory;
                        rows sum to 1 and reset to uniform after resampling.
    """
    _validate_time_grid(t)
    if not torch.all(t[1:] < t[:-1]):
        raise ValueError("t must be strictly decreasing for steered_reverse_churn_sampling")
    if churn < 0:
        raise ValueError(f"churn must be non-negative, got {churn}")
    if ess_threshold <= 0:
        raise ValueError(f"ess_threshold must be positive, got {ess_threshold}")
    if ess_threshold >= 1 and ess_threshold != int(ess_threshold):
        raise ValueError(
            f"ess_threshold >= 1 selects fixed-interval resampling and must be a "
            f"whole number of steps, got {ess_threshold}"
        )
    interval_mode = ess_threshold >= 1
    resample_every = int(ess_threshold) if interval_mode else 1  # unused when not interval_mode

    log_w = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
    # ρ at the current state, carried across steps: the endpoint difference needs the
    # ancestor's tilt, and recomputing it would double the cost of a `potential` that
    # backprops through an unrolled denoiser. On a resample it is gathered, not re-evaluated.
    rho = potential(x, t[0])
    trajectory = [x.clone()]
    weight_history = [_normalized_weights(log_w)]
    ess_history: list[float] = []
    for step, (t_curr, t_next) in enumerate(zip(t[:-1], t[1:])):
        t_hat = (t_curr + churn * (t_next - t_curr).abs()).clamp(max=1.0)
        if t_hat > t_curr:
            x = transition(x, t_curr, t_hat)  # exact forward kernel, t -> t̂
        x = x + velocity(x, t_hat) * (t_next - t_hat)  # PF-ODE transport t̂ -> t+dt

        rho_next = potential(x, t_next)
        log_w = log_w + rho_next - rho  # retarget q_t·e^{ρ_t} onto q_{t+dt}·e^{ρ_{t+dt}}
        rho = rho_next

        ess = _ess_ratio(log_w)
        ess_history.append(ess)

        if interval_mode:
            should_resample = (step + 1) % resample_every == 0
        else:
            should_resample = ess < ess_threshold

        if should_resample:
            idx = _systematic_resample(log_w)
            x, rho = x[idx], rho[idx]
            log_w = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)

        trajectory.append(x.clone())
        weight_history.append(_normalized_weights(log_w))

    # Final resample according to accumulated weights
    idx = _systematic_resample(log_w)
    x = x[idx]
    log_w = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
    trajectory[-1] = x.clone()
    weight_history[-1] = _normalized_weights(log_w)

    return torch.stack(trajectory), ess_history, torch.stack(weight_history)


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
