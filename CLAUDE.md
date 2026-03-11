# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**torchGMM** provides analytical diffusion on Gaussian Mixture Models (GMMs) in PyTorch. The key insight is that the GMM family is closed under Gaussian convolution, so every quantity (log-prob, score, samples) remains exact at every noise level t ∈ [0, 1] under the VP-SDE forward process.

## Commands

**Install:**
```bash
uv pip install -e ".[dev,test]"
```

**Run tests:**
```bash
pytest                                                           # All tests (parallel by default)
pytest tests/test_gmm.py                                        # Single file
pytest tests/test_gmm.py::TestShapes::test_gmm_initialization   # Single test
pytest -m "not slow"                                            # Exclude slow tests
```

**Lint and format:**
```bash
black src tests && isort src tests && flake8 src tests
```

## Architecture

**Public API** (`src/torchGMM/__init__.py`):
- `TimeDependentGMM` — batched GMM with diffusion schedule; core class
- `Conditional` — wraps a single point x0 as a single-component GMM
- `BetaSchedule` — linear β(t) schedule for VP-SDE
- `forward_diffusion`, `reverse_diffusion`, `reverse_step` — SDE simulation

**Key abstractions:**

`TimeDependentGMM(mu, sigma, weight, schedule)` — `torch.nn.Module` with params shaped `[*B, K, D]` (batch × components × dimensions). All methods accept inputs shaped `[*N, *B, D]` and return:
- Scalars like `log_prob` / `energy`: `[*N, *B]`
- Vectors like `score` / `sample`: `[*N, *B, D]`

`BetaSchedule` computes α_t and σ_t for the VP-SDE marginal: `p(x_t | x_0) = N(α_t · x_0, σ_t² I)`.

`diffusion.py` implements Euler-Maruyama for forward and reverse SDEs. `reverse_diffusion` takes a `score_fn: callable` and runs the reverse SDE trajectory.

## Conventions

- **No try/except** — inputs should unambiguously define the workflow path
- **Shape comments** — annotate tensor shapes with `[*B, K, D]` style notation
- **Type hints** — use Python 3.10+ syntax (`t: torch.Tensor | None`), required for all public API functions
- **register_buffer** for non-trainable tensors in `nn.Module` subclasses
- Black line-length 120, isort black profile, flake8 max-line-length 120, max-complexity 18
- Custom pytest markers: `@pytest.mark.slow`, `@pytest.mark.integration`
- Use `@pytest.fixture` for shared setup and `@pytest.mark.parametrize` for variants
