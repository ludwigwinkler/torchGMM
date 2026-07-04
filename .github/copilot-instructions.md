# torchGMM Copilot Instructions

## Build & Test Commands

```bash
# Install package with dev dependencies
uv pip install -e .

# Run all tests (parallel by default)
pytest

# Run a single test file
pytest tests/test_gmm.py

# Run a single test function
pytest tests/test_gmm.py::TestShapes::test_gmm_initialization

# Run tests excluding slow ones
pytest -m "not slow"

# Lint and format
black torchGMM tests && isort torchGMM tests && flake8 torchGMM tests
```

## Architecture

**Core abstractions** (all in `torchGMM/`):
- `GMM` — Batched GMM with diffusion schedule. Params are `[*B, K, D]` (batch, components, dims). Supports `log_prob`, `score`, `sample`, `energy` with time `t ∈ [0,1]`.
- `Conditional` — Subclass of `GMM` for conditional processes from a single point `x0`.
- `BetaSchedule` — Linear β schedule for the forward SDE. Provides `alpha_t`, `sigma_t` coefficients.
- `forward_diffusion` / `reverse_diffusion` — Euler-Maruyama simulation of the SDE.

**Shape convention**: Inputs are `[*N, *B, D]` where `N` is sample dims and `B` is batch shape from GMM init. Outputs drop `D` for scalars (`log_prob`) or keep it (`score`, `sample`).

## Key Conventions

- **No try/except blocks** — inputs should unambiguously define workflow
- **Tensor shape comments** — use `[*BS, K, D]`, `[..., D]` style annotations
- **Type hints** — use Python 3.10+ syntax (`torch.Tensor | None`)
- **Docstrings** — include `Args:` with shapes and `Returns:` for public APIs
- **Formatting** — Black (line-length 120), isort (profile black), flake8
- **Tests** — use `@pytest.fixture` for shared setup, `@pytest.mark.parametrize` for variants
- **Use `register_buffer`** for non-trainable model tensors (mu, sigma, weight)
