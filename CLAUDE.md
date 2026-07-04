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
pytest -n 8                                                      # Run all tests with 8 xdist workers
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
- `steered_reverse_sampling` — Feynman-Kac-Corrector (FKC) steered reverse sampling with SMC importance resampling

**Key abstractions:**

`GMM(mu, sigma, weight, schedule)` — `torch.nn.Module` with params shaped `[*B, K, D]` (batch × components × dimensions). All methods accept inputs shaped `[*N, *B, D]` and return:
- Scalars like `log_prob` / `energy`: `[*N, *B]`
- Vectors like `score` / `sample`: `[*N, *B, D]`

`Schedule` base class provides `get_alpha_t`, `get_sigma_t`, `get_dalpha_dt`, `get_dsigma_dt`, `forward_drift`, and `diffusion_coeff`. `BetaSchedule`, `LinearSchedule`, `VESchedule`, and `KarrasSchedule` are concrete implementations; the latter two are variance-exploding (α_t ≡ 1) and typically paired with `diffusion=None` (probability-flow ODE) or the FKC steering sampler below.

`sampling.py` implements Euler-Maruyama for forward and reverse SDEs. Both `forward_sampling` and `reverse_sampling` take `drift: callable`, `diffusion: callable | None`, initial state `x`, and a time grid `t`. The caller constructs drift/diffusion callables from the schedule and GMM score before calling.

`steered_reverse_sampling` runs the same Euler-Maruyama loop as an SMC particle filter: alongside `drift`/`diffusion` it takes a `weight_update(x, t, dt) -> [N]` callable returning incremental log importance weights, and systematically resamples particles whenever ESS/N drops below `ess_threshold`. This implements Feynman-Kac-Corrector steering (see `docs/fkc_steering.md`) to sample from a reward-tilted target `p(x) ∝ q(x) exp(β·r(x))` with exact importance weights, without retraining the underlying model. `notebooks/ve_steering.py` and `notebooks/karras_terminal_variance_steering.py` demonstrate the pattern: a reward `r(x_0)` on denoised samples, a tilt schedule `β(t)`, and gradients of `r` backpropagated through an unrolled ODE denoiser to build `weight_update`.

## Conventions

- **No try/except** — inputs should unambiguously define the workflow path
- **Shape comments** — annotate tensor shapes with `[*B, K, D]` style notation
- **Type hints** — use Python 3.10+ syntax (`t: torch.Tensor | None`), required for all public API functions
- **register_buffer** for non-trainable tensors in `nn.Module` subclasses
- Ruff line-length 120, ruff format replaces black/isort, ruff check replaces flake8
- jaxtyping `Float[Tensor, "*batch D"]` shape annotations on all public methods; `@jaxtyped(typechecker=beartype)` decorator enforces them at runtime
- Custom pytest markers: `@pytest.mark.slow`
- Use `@pytest.fixture` for shared setup and `@pytest.mark.parametrize` for variants
