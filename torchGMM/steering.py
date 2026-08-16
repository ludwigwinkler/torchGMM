"""FKC-steered sampling with SMC importance resampling."""

import torch
from beartype import beartype
from beartype.typing import Callable
from jaxtyping import Float, jaxtyped
from torch import Tensor

from .sampling import _validate_time_grid


def _validate_sigma_grid(sigma: torch.Tensor) -> None:
    """Validate a 1D EDM noise grid."""
    if sigma.dim() != 1:
        raise ValueError(f"sigma must be 1D, got {sigma.shape}")
    if not torch.all(torch.isfinite(sigma)):
        raise ValueError("sigma must be finite")
    if not torch.all(sigma >= 0):
        raise ValueError("sigma must be non-negative")
    if sigma.numel() < 2:
        raise ValueError("sigma must contain at least two noise levels")


def _ess_ratio(log_w: torch.Tensor) -> float:
    lw = log_w - torch.logsumexp(log_w, 0)
    ess = torch.exp(-torch.logsumexp(2 * lw, 0)) / log_w.shape[0]
    return ess.clamp(0.0, 1.0).item()


def _systematic_resample(log_w: torch.Tensor) -> torch.Tensor:
    w = torch.softmax(log_w, dim=0)
    N = w.shape[0]
    cdf = torch.cumsum(w, 0)
    u = (torch.arange(N, dtype=w.dtype, device=w.device) + torch.rand(1, device=w.device)) / N
    return torch.searchsorted(cdf, u).clamp(max=N - 1)


def _normalized_weights(log_w: torch.Tensor) -> torch.Tensor:
    return torch.softmax(log_w, dim=0)


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


@jaxtyped(typechecker=beartype)
def steered_reverse_sampling(
    drift: Callable,
    diffusion: Callable | None,
    weight_update: Callable | None,
    x: Float[Tensor, "N *rest D"],
    t: Float[Tensor, " T"],
    ess_threshold: float | int = 0.5,
) -> tuple[Float[Tensor, "T N *rest D"], list[float], Float[Tensor, "T N"]]:
    """Reverse SDE/ODE sampling with SMC particle correction via importance resampling.

    Args:
        drift:          (x, t) -> [*shape, D]
        diffusion:      (t) -> scalar; None = ODE
        weight_update:  (x, t) -> [N] instantaneous log-weight rate per particle;
                        the solver multiplies it by |dt|
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
        log_w = log_w + weight_update(x_prev, t_curr) * dt.abs()

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
def steered_reverse_edm_sampling(
    denoise: Callable,
    weight_update: Callable | None,
    x: Float[Tensor, "N *rest D"],
    sigma: Float[Tensor, " S"],
    noise_scale: float = 1.0,
    step_scale: float = 1.0,
    ess_threshold: float | int = 0.5,
) -> tuple[Float[Tensor, "S N *rest D"], list[float], Float[Tensor, "S N"]]:
    """FKC-steered EDM churn sampling on a strictly decreasing noise grid.

    This follows the EDM operator split using the next larger grid level. At
    each step after the first,

    `sigma_hat = sigma[step - 1]`,
    `x_noisy = x + noise_scale
                   * sqrt(sigma_hat**2 - sigma_curr**2) * noise`,
    `direction = (x_noisy - denoise(x_noisy, sigma_hat, ...)) / sigma_hat`,
    `x_next = x_noisy
              + step_scale * (sigma_next - sigma_hat) * direction`.

    The first step cannot reheat because no larger grid level exists.
    Reheating and deterministic denoising are separate operators; this is not
    an Euler-Maruyama step. Every denoiser evaluation remains on the supplied
    Karras grid.

    The weight callback is evaluated at the reheated state and receives all
    three noise levels. It returns the complete FKC log-weight increment over
    the base grid step, so the sampler applies no additional scaling.

    Args:
        denoise: `(x_noisy, sigma_hat, sigma_curr, sigma_next) ->
            [N, *rest, D]`; denoised estimate, including any caller-provided
            guidance.
        weight_update: `(x_noisy, sigma_hat, sigma_curr, sigma_next) -> [N]`;
            direct log-weight increment over the base step.
        x: Initial particles `[N, *rest, D]`.
        sigma: Strictly decreasing 1D noise grid `[S]`, with non-negative values.
        noise_scale: Multiplier for the reheat noise.
        step_scale: Multiplier for the deterministic Euler denoising step.
        ess_threshold: Resampling strategy, identical to
            `steered_reverse_sampling`: adaptive for `0 < threshold < 1`, or a
            whole-number fixed interval for `threshold >= 1`.

    Returns:
        trajectory: Particle trajectory `[S, N, *rest, D]`.
        ess_history: ESS/N after each weight update, length `S - 1`.
        weight_history: Normalized weights `[S, N]`, reset after resampling.
    """
    _validate_sigma_grid(sigma)
    if not torch.all(sigma[1:] < sigma[:-1]):
        raise ValueError("sigma must be strictly decreasing for steered_reverse_edm_sampling")
    if weight_update is None:
        raise ValueError("weight_update must be provided for steered_reverse_edm_sampling")
    if noise_scale < 0:
        raise ValueError(f"noise_scale must be non-negative, got {noise_scale}")
    if step_scale <= 0:
        raise ValueError(f"step_scale must be positive, got {step_scale}")
    if ess_threshold <= 0:
        raise ValueError(f"ess_threshold must be positive, got {ess_threshold}")
    if ess_threshold >= 1 and ess_threshold != int(ess_threshold):
        raise ValueError(
            f"ess_threshold >= 1 selects fixed-interval resampling and must be a "
            f"whole number of steps, got {ess_threshold}"
        )
    interval_mode = ess_threshold >= 1
    resample_every = int(ess_threshold) if interval_mode else 1

    log_w = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
    trajectory = [x.clone()]
    weight_history = [_normalized_weights(log_w)]
    ess_history: list[float] = []
    for step, (sigma_curr, sigma_next) in enumerate(zip(sigma[:-1], sigma[1:])):
        sigma_hat = sigma[step - 1] if step > 0 else sigma_curr
        reheat_variance = sigma_hat.square() - sigma_curr.square()
        if reheat_variance > 0:
            x = x + noise_scale * torch.sqrt(reheat_variance) * torch.randn_like(x)

        log_w = log_w + weight_update(x, sigma_hat, sigma_curr, sigma_next)
        denoised = denoise(x, sigma_hat, sigma_curr, sigma_next)
        direction = (x - denoised) / sigma_hat
        x = x + step_scale * (sigma_next - sigma_hat) * direction

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

    idx = _systematic_resample(log_w)
    x = x[idx]
    log_w = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
    trajectory[-1] = x.clone()
    weight_history[-1] = _normalized_weights(log_w)

    return torch.stack(trajectory), ess_history, torch.stack(weight_history)


@jaxtyped(typechecker=beartype)
def steered_reverse_af3_sampling(
    denoise: Callable,
    weight_update: Callable | None,
    x: Float[Tensor, "N *rest D"],
    sigma: Float[Tensor, " S"],
    gamma_0: float = 0.8,
    gamma_min: float = 1.0,
    noise_scale: float = 1.0,
    step_scale: float = 1.0,
    ess_threshold: float | int = 0.5,
    potential: Callable | None = None,
    reheat_proposal: Callable | None = None,
) -> tuple[Float[Tensor, "S N *rest D"], list[float], Float[Tensor, "S N"]]:
    """FKC-steered AlphaFold 3 sampling on a strictly decreasing noise grid.

    At each step from `sigma_curr` to `sigma_next`, the sampler applies the
    AlphaFold 3 churn rule

    `gamma = gamma_0 if sigma_next > gamma_min else 0`,
    `sigma_hat = sigma_curr * (1 + gamma)`,
    `x_noisy = x + noise_scale
                   * sqrt(sigma_hat**2 - sigma_curr**2) * noise`,
    `direction = (x_noisy - denoise(x_noisy, sigma_hat, ...)) / sigma_hat`,
    `x_next = x_noisy
              + step_scale * (sigma_next - sigma_hat) * direction`.

    This uses the corrected noisy-state direction from the released AlphaFold 3
    implementation. The paper's rigid coordinate augmentation is intentionally
    omitted because it does not preserve a generic target distribution.

    The sampler supports two weighting modes:

    - `weight_update` is evaluated at the reheated state and returns a local
      FKC log-weight increment over the base grid step. This is the continuous
      small-step approximation used by the EDM sampler.
    - `potential(x, sigma)` returns the full log tilt at a grid marginal. The
      sampler applies the exact discrete increment
      `potential(x_next, sigma_next) - potential(x_prev, sigma_curr)` after the
      full AF3 step. This mode remains appropriate for AF3's finite reheat.

    Exactly one of `weight_update` and `potential` must be provided. Exact
    potential-ratio weighting assumes `denoise` defines the unsteered base
    transition. A guided denoiser needs an additional deterministic-transport
    proposal correction and should use `weight_update`.

    `reheat_proposal` may replace the base Gaussian reheat with a guided
    proposal. It must return both the proposed state and
    `log K(x_noisy | x) - log Q(x_noisy | x)`, where `K` is the exact AF3
    Gaussian reheat and `Q` is the guided proposal. This correction is added in
    either weighting mode. The callback is only invoked when the reheat
    variance is positive.

    Args:
        denoise: `(x_noisy, sigma_hat, sigma_curr, sigma_next) ->
            [N, *rest, D]`; denoised estimate, including any caller-provided
            guidance.
        weight_update: `(x_noisy, sigma_hat, sigma_curr, sigma_next) -> [N]`;
            local continuous-FKC log-weight increment over the base step.
        x: Initial particles `[N, *rest, D]`.
        sigma: Strictly decreasing 1D noise grid `[S]`, with non-negative values.
        gamma_0: Fractional AF3 reheat while the destination noise exceeds
            `gamma_min`.
        gamma_min: Destination-noise threshold below which churn is disabled.
        noise_scale: Multiplier for the reheat noise.
        step_scale: Multiplier for the deterministic Euler denoising step.
        ess_threshold: Resampling strategy, identical to
            `steered_reverse_sampling`: adaptive for `0 < threshold < 1`, or a
            whole-number fixed interval for `threshold >= 1`.
        potential: `(x, sigma) -> [N]`; full grid-marginal log tilt used for
            exact discrete Feynman--Kac increments.
        reheat_proposal: `(x, sigma_hat, sigma_curr, sigma_next) ->
            (x_noisy, log_base_over_proposal)`; optional guided reheat proposal
            and its per-particle log-density correction. When supplied,
            `noise_scale` is not used for reheating.

    Returns:
        trajectory: Particle trajectory `[S, N, *rest, D]`.
        ess_history: ESS/N after each weight update, length `S - 1`.
        weight_history: Normalized weights `[S, N]`, reset after resampling.
    """
    _validate_sigma_grid(sigma)
    if not torch.all(sigma[1:] < sigma[:-1]):
        raise ValueError("sigma must be strictly decreasing for steered_reverse_af3_sampling")
    if (weight_update is None) == (potential is None):
        raise ValueError("pass exactly one of weight_update or potential to steered_reverse_af3_sampling")
    if gamma_0 < 0:
        raise ValueError(f"gamma_0 must be non-negative, got {gamma_0}")
    if gamma_min < 0:
        raise ValueError(f"gamma_min must be non-negative, got {gamma_min}")
    if noise_scale < 0:
        raise ValueError(f"noise_scale must be non-negative, got {noise_scale}")
    if step_scale <= 0:
        raise ValueError(f"step_scale must be positive, got {step_scale}")
    if ess_threshold <= 0:
        raise ValueError(f"ess_threshold must be positive, got {ess_threshold}")
    if ess_threshold >= 1 and ess_threshold != int(ess_threshold):
        raise ValueError(
            f"ess_threshold >= 1 selects fixed-interval resampling and must be a "
            f"whole number of steps, got {ess_threshold}"
        )
    interval_mode = ess_threshold >= 1
    resample_every = int(ess_threshold) if interval_mode else 1

    log_w = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
    trajectory = [x.clone()]
    weight_history = [_normalized_weights(log_w)]
    ess_history: list[float] = []
    for step, (sigma_curr, sigma_next) in enumerate(zip(sigma[:-1], sigma[1:])):
        x_prev = x
        gamma = gamma_0 if sigma_next > gamma_min else 0.0
        sigma_hat = sigma_curr * (1.0 + gamma)
        reheat_variance = sigma_hat.square() - sigma_curr.square()
        if reheat_variance > 0:
            if reheat_proposal is None:
                x = x + noise_scale * torch.sqrt(reheat_variance) * torch.randn_like(x)
            else:
                x, log_base_over_proposal = reheat_proposal(x, sigma_hat, sigma_curr, sigma_next)
                log_w = log_w + log_base_over_proposal

        if weight_update is not None:
            log_w = log_w + weight_update(x, sigma_hat, sigma_curr, sigma_next)
        denoised = denoise(x, sigma_hat, sigma_curr, sigma_next)
        direction = (x - denoised) / sigma_hat
        x = x + step_scale * (sigma_next - sigma_hat) * direction
        if potential is not None:
            log_w = log_w + potential(x, sigma_next) - potential(x_prev, sigma_curr)

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

    idx = _systematic_resample(log_w)
    x = x[idx]
    log_w = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
    trajectory[-1] = x.clone()
    weight_history[-1] = _normalized_weights(log_w)

    return torch.stack(trajectory), ess_history, torch.stack(weight_history)


@jaxtyped(typechecker=beartype)
def steered_reverse_churn_sampling(
    drift: Callable,
    transition: Callable | None = None,
    weight_update: Callable | None = None,
    x: Float[Tensor, "N *rest D"] | None = None,
    t: Float[Tensor, " T"] | None = None,
    churn: Callable | float = 1.0,
    ess_threshold: float | int = 0.5,
    diffusion: Callable | None = None,
) -> tuple[Float[Tensor, "T N *rest D"], list[float], Float[Tensor, "T N"]]:
    """FKC-steered churn sampling: `reverse_churn_sampling` run as an SMC particle filter.

    Churn re-noises either through the exact forward `transition` kernel or, when no
    transition is supplied, through an additive VE Euler step from `diffusion`. The exact
    kernel takes precedence when both are supplied. Everything else — the `ess_threshold`
    contract and `(trajectory, ess_history, weight_history)` return — mirrors
    `steered_reverse_sampling`.

    Each step re-noises to t̂ with `transition`, then transports `drift` from t̂ to t+dt.
    Both operators preserve the marginal family, so the composition lands on p_{t+dt};
    the SMC layer then retargets it onto p_t(x) ∝ q_t(x)·exp(ρ_t(x)).

    **Guidance is the caller's, not the sampler's.** `drift` is the probability-flow
    velocity, and a reward-guided flow is simply that velocity plus a guidance field —
    built by the caller exactly as `guided_drift` is built for `steered_reverse_sampling`.
    The sampler never needs to know which part is guidance, because whatever compensation
    the guidance requires belongs in `weight_update`, which the caller also owns.

    `weight_update(x, t) -> [N]` is evaluated at the *reheated* pair (x̂, t̂), since that
    is where the transport starts and where the score is taken. It is identical in form
    and meaning to `steered_reverse_sampling`'s continuous-time FKC callback: the solver
    multiplies its instantaneous rate by the base reverse-grid magnitude
    `base_step = t_curr - t_next`, not the longer transport duration. The churn splitting's
    continuous limit is the standard λ-family reverse SDE at λ=√churn, and the Prop. D.6
    weight for that generator is **independent of churn** (see
    `docs/edm_fkc_steering.md` §3) — so the very same `fkc_weight_update` closure works
    here and in `steered_reverse_sampling`, at any churn strength.

    Args:
        drift:         (x, t) -> [N, *rest, D]; the probability-flow velocity, e.g.
                       `gmm.velocity`, plus any guidance field the caller wants. NOT the
                       reverse-SDE drift f − g²s — the churn supplies the stochasticity,
                       so a score-corrected drift double-counts it.
        transition: Exact forward kernel `(x, t, s) -> [N, *rest, D]`, e.g.
                    `schedule.transition`. Preferred when supplied; required for exact
                    churn on a general schedule.
            weight_update: (x, t, -dt) -> [N] instantaneous log-weight rate, evaluated at the
                       reheated pair (x̂, t̂). The solver multiplies it by the base
                       reverse-grid magnitude `base_step = t_curr - t_next`, not by the
                       longer deterministic transport duration.
        x:             [N, *rest, D] initial state; N = number of particles
        t:             1D strictly decreasing time grid in [0, 1], >= 2 points
        churn:         Non-negative scalar or `(t) -> non-negative float` churn strength
                       as a multiple of the step size, h = churn(t)·|dt|; see
                       `reverse_churn_sampling`. A value of 0 makes that step
                       deterministic and permanently disables subsequent particle
                       resampling, including the final resample; FKC weights still
                       accumulate.
        ess_threshold: Resampling strategy, identical to `steered_reverse_sampling`:
                       adaptive when 0 < ess_threshold < 1 (resample once ESS/N drops
                       below it), fixed-interval when ess_threshold >= 1 (a whole number
                       of steps). A final resample fires after the last step unless churn
                       has reached zero.
        diffusion:     Optional VE forward diffusion coefficient `g(t)`. Used only when
                       `transition` is absent, via an additive Euler reheat; this is not
                       exact for finite steps.
    Returns:
        trajectory:     [T, N, *rest, D]
        ess_history:    ESS/N at each step, len = len(t) - 1
        weight_history: [T, N] normalized particle weights aligned with trajectory;
                        rows sum to 1 and reset to uniform after resampling.
    """
    if x is None or t is None:
        raise ValueError("x and t must be provided")
    _validate_time_grid(t)
    if not torch.all(t[1:] < t[:-1]):
        raise ValueError("t must be strictly decreasing for steered_reverse_churn_sampling")
    if weight_update is None:
        raise ValueError("weight_update must be provided for steered_reverse_churn_sampling")
    if transition is None and diffusion is None:
        raise ValueError("pass transition or diffusion for steered_reverse_churn_sampling")
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
    resampling_enabled = True

    for step, (t_curr, t_next) in enumerate(zip(t[:-1], t[1:])):
        base_step = t_curr - t_next  # Positive: reverse grids satisfy t_next < t_curr.
        churn_value = churn(t_curr) if callable(churn) else churn
        if churn_value < 0:
            raise ValueError(f"churn must be non-negative, got {churn_value}")
        if churn_value == 0:
            resampling_enabled = False
        t_hat = (t_curr + churn_value * base_step).clamp(max=1.0)  # Forward reheat: t_hat >= t_curr.
        transport_dt = t_next - t_hat  # Negative deterministic denoising increment.
        assert transport_dt <= 0, "transport_dt must be non-positive"
        assert 0 <= t_next < t_curr <= t_hat <= 1, "t must be strictly decreasing"

        # ---- churn half: exact forward kernel t -> t̂ ----
        if t_hat > t_curr:
            if transition is not None:
                x = transition(x, t_curr, t_hat)
            else:
                reheat_dt = t_hat - t_curr
                x = x + diffusion(t_curr) * torch.sqrt(reheat_dt) * torch.randn_like(x)

        # ---- deterministic half: probability-flow transport t̂ -> t+dt ----
        # In an ideal world we denoise from t_hat to t_curr unguided, then
        # use the guidance-corrected drift to transport from t_curr to t_next.
        # But for simplicity and efficiency, we just rescale with the base step.
        log_w = log_w + weight_update(x, t_hat) * base_step.abs()
        x = x + drift(x, t_hat) * transport_dt

        ess = _ess_ratio(log_w)
        ess_history.append(ess)

        if interval_mode:
            should_resample = (step + 1) % resample_every == 0
        else:
            should_resample = ess < ess_threshold

        if resampling_enabled and should_resample:
            idx = _systematic_resample(log_w)
            x = x[idx]
            log_w = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)

        trajectory.append(x.clone())
        weight_history.append(_normalized_weights(log_w))

    if resampling_enabled:
        # Final resample according to accumulated weights.
        idx = _systematic_resample(log_w)
        x = x[idx]
        log_w = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        trajectory[-1] = x.clone()
        weight_history[-1] = _normalized_weights(log_w)

    return torch.stack(trajectory), ess_history, torch.stack(weight_history)
