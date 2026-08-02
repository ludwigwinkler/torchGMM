# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**torchGMM** provides analytical diffusion on Gaussian Mixture Models (GMMs) in PyTorch. The key insight is that the GMM family is closed under Gaussian convolution, so every quantity (log-prob, score, samples) remains exact at every noise level t ∈ [0, 1] under the VP-SDE forward process.

## Commands

**Install:**
```bash
uv pip install -e .
```

**Run tests:**
```bash
pytest                                                           # All tests (parallel by default)
pytest tests/test_gmm.py                                        # Single file
pytest tests/test_gmm.py::TestShapes::test_gmm_initialization   # Single test
pytest -m "not slow"                                            # Exclude slow tests
pytest -n 4                                                      # Run all tests with 4 xdist workers
```

**Lint and format:**
```bash
ruff check torchGMM tests && ruff format torchGMM tests
```

**Type check:**
```bash
ty check torchGMM
```

## Architecture

**Public API** (`torchGMM/__init__.py`):
- `GMM` — batched GMM with diffusion schedule; core class
- `Conditional` — wraps a single point x0 as a single-component GMM
- `Schedule` — base class for interpolation schedules
- `BetaSchedule` — VP-SDE schedule with linear β(t); satisfies α_t² + σ_t² = 1
- `LinearSchedule` — flow matching / rectified flow schedule; α_t = 1−t, σ_t = t
- `VESchedule` — variance-exploding (SMLD/NCSN) schedule; α_t ≡ 1, geometric σ_t
- `KarrasSchedule` — Karras et al. / AlphaFold3-style VE schedule with ρ-controlled step concentration; α_t ≡ 1
- `forward_sampling`, `reverse_sampling` — Euler-Maruyama SDE/ODE simulation
- `reverse_churn_sampling` — EDM Algorithm 2 churn sampler (exact re-noise + probability-flow transport)
- `steered_reverse_sampling` — Feynman-Kac-Corrector (FKC) steered reverse sampling with SMC importance resampling
- `steered_reverse_churn_sampling` — FKC steering on the churn sampler, weighted by the endpoint potential

**Key abstractions:**

`GMM(mu, sigma, weight, schedule)` — `torch.nn.Module` with params shaped `[*B, K, D]` (batch × components × dimensions). All methods accept inputs shaped `[*N, *B, D]` and return:
- Scalars like `log_prob` / `energy`: `[*N, *B]`
- Vectors like `score` / `sample`: `[*N, *B, D]`

`Schedule` base class provides `get_alpha_t`, `get_sigma_t`, `get_dalpha_dt`, `get_dsigma_dt`, `forward_drift`, `diffusion_coeff`, and `transition` — the exact forward kernel `q(x_s | x_t)`, which maps `p_t` onto `p_s` in a single draw for any gap (derived in `docs/schedule.md`). `BetaSchedule`, `LinearSchedule`, `VESchedule`, and `KarrasSchedule` are concrete implementations; the latter two are variance-exploding (α_t ≡ 1) and typically paired with `diffusion=None` (probability-flow ODE) or the FKC steering sampler below.

`sampling.py` implements Euler-Maruyama for forward and reverse SDEs. Both `forward_sampling` and `reverse_sampling` take `drift: callable`, `diffusion: callable | None`, initial state `x`, and a time grid `t`. The caller constructs drift/diffusion callables from the schedule and GMM score before calling.

`reverse_churn_sampling` is the EDM Algorithm 2 sampler (see `docs/churn_sampler.md`). It is an *operator splitting*, not an SDE discretisation, so it is a separate function rather than an integrator swapped into `reverse_sampling`, and its two callables are complete operators rather than the drift/diffusion terms of one SDE:

```python
reverse_churn_sampling(gmm.velocity, schedule.transition, x, t)
```

Each step re-noises to `t + |dt|` with the exact forward kernel `Schedule.transition`, then transports the probability flow over `dt - |dt| = -2|dt|`, undoing the churn as well as advancing one grid step. The velocity is evaluated at the *reheated* time, since that is where the churned state actually lives. Both operators preserve the marginal family, so the composition lands on `p_{t+dt}`. The kernel's exactness is the point: with a first-order churn the whole scheme collapses to reverse-SDE Euler-Maruyama up to `O(h^{3/2})`. It is reverse-only — on an increasing grid `dt - |dt|` is zero — and forward noising needs no sampler at all, since `schedule.transition` jumps `t -> s` exactly in one draw.

`steered_reverse_churn_sampling` is the FKC-steered churn sampler (see `docs/fkc_churn_steering.md`). It layers the same SMC machinery as `steered_reverse_sampling` — identical `ess_threshold` contract, identical `(trajectory, ess_history, weight_history)` return — onto that splitting, but accepts either an incremental `weight_update(x, t, dt)` (identical in meaning to the Euler-Maruyama sampler's — the FKC weight is churn-independent, so the same closure steers both) or a `potential(x, t) -> [N]`:

```python
steered_reverse_churn_sampling(drift, schedule.transition, weight_update, x, t, churn=1.0)
```

The signature mirrors `steered_reverse_sampling` one-for-one with `transition` in place of `diffusion` — a churn step re-noises rather than adding a Brownian increment. Guidance is the caller's: `drift` is the probability-flow velocity plus any guidance field, built exactly as `guided_drift` is for the Euler-Maruyama sampler.

The `potential` route is the one native to the splitting: because the base churn kernel already maps `q_t` onto `q_{t+dt}` exactly, the entire importance correction is the endpoint difference `ρ_{t+dt}(x_{t+dt}) − ρ_t(x_t)`. With an unguided drift that is markedly cheaper to wire up than the Euler-Maruyama route — no `β̇_t`, no `∂_t r`, no reward Laplacian, no score-alignment term, and no reward *gradient*, so a denoiser inside `potential` is never backpropagated through — and it is exact at finite step size rather than only in the limit. `tests/test_churn_steering.py` asserts both routes reach the same tilted marginals as the Euler-Maruyama sampler across ~50 intermediate time slots on `BetaSchedule` and `KarrasSchedule`, and checks the accumulated log weights against their closed form exactly when resampling only at the end.

`steered_reverse_sampling` runs the same Euler-Maruyama loop as an SMC particle filter: alongside `drift`/`diffusion` it takes a `weight_update(x, t, dt) -> [N]` callable returning incremental log importance weights, and systematically resamples particles according to `ess_threshold`, which selects one of two resampling strategies — adaptive (`0 < ess_threshold < 1`: resample whenever ESS/N drops below the threshold) or fixed-interval (`ess_threshold >= 1`, a whole number: resample unconditionally every `int(ess_threshold)` steps); setting the interval past the total step count degenerates to a single resample after the final step. This implements Feynman-Kac-Corrector steering (see `docs/fkc_steering.md`) to sample from a reward-tilted target `p(x) ∝ q(x) exp(β·r(x))` with exact importance weights, without retraining the underlying model. `notebooks/ve_steering.py` and `notebooks/karras_terminal_variance_steering.py` demonstrate the pattern: a reward `r(x_0)` on denoised samples, a tilt schedule `β(t)`, and gradients of `r` backpropagated through an unrolled ODE denoiser to build `weight_update`.

## Conventions

- **No try/except** — inputs should unambiguously define the workflow path
- **Shape comments** — annotate tensor shapes with `[*B, K, D]` style notation
- **Type hints** — use Python 3.10+ syntax (`t: torch.Tensor | None`), required for all public API functions
- **register_buffer** for non-trainable tensors in `nn.Module` subclasses
- Ruff line-length 120, ruff format replaces black/isort, ruff check replaces flake8
- jaxtyping `Float[Tensor, "*batch D"]` shape annotations on all public methods; `@jaxtyped(typechecker=beartype)` decorator enforces them at runtime
- Custom pytest markers: `@pytest.mark.slow`
- Use `@pytest.fixture` for shared setup and `@pytest.mark.parametrize` for variants
