# CLAUDE.md — torchGMM

Package-level guidance for working inside this directory.

## Module map

| File | Responsibility |
|---|---|
| `__init__.py` | Public re-exports only — no logic |
| `gmm.py` | `GMM` and `Conditional` classes |
| `schedule.py` | `Schedule`, `BetaSchedule`, `LinearSchedule`, `VESchedule`, `KarrasSchedule` |
| `sampling.py` | `euler_maruyama`, `forward_sampling`, `reverse_sampling`, `steered_reverse_sampling` |

## Shape convention

All methods follow a strict two-group shape contract:

```
params:  [*B, K, D]   — batch × components × dim (set at GMM init)
x input: [*N, *B, D]  — optional sample dims prepended to batch
t input: [*N, *B]     — one scalar time per batch element; scalar/0-dim broadcasts
outputs: [*N, *B]     — scalars (log_prob, energy)
         [*N, *B, D]  — vectors (score, velocity, sample)
```

`*B` is `gmm.batch_shape`. `*N` is any number of leading sample dims.

## Runtime type checking

Every public method carries `@jaxtyped(typechecker=beartype)`. Shape strings use:

- `"*batch D"` — variadic batch dims + one event dim
- `"*batch"` — variadic batch dims, no event dim
- `"*t"` — independent variadic for `t` in `forward_drift`/`diffusion_coeff` (t can be 0-dim scalar from sampling loop)
- `" T"` — a leading space forces a 1D named dim (used for time grids)

When adding a new public method, decorate it and annotate `x` and the return. For `t`, use `int | float | torch.Tensor | None` (the `_expand_t` helper handles all cases internally).

## Key invariants

- **`_expand_t`** is the single normalisation point for `t`. Never expand `t` manually in a method body — always call `self._expand_t(t, sample_shape)`.
- **`_gmm_t(t_exp)`** builds the `MixtureSameFamily` at time `t`. All probability computations go through it.
- **`schedule`** is a `torch.nn.Module` stored on the GMM. Schedules must implement `get_alpha_t`, `get_sigma_t`, `get_dalpha_dt`, `get_dsigma_dt`. Derived quantities (`forward_drift`, `diffusion_coeff`) have default implementations in `Schedule` but `BetaSchedule` overrides them for efficiency.
- **`register_buffer`** for `mu`, `sigma`, `weight` — they move with `.to(device)` but are not trainable parameters.
- Weights are normalised to sum to 1 in `__init__`. Never assume raw weights sum to 1 inside methods.

## Adding a new schedule

Subclass `Schedule`, implement the four abstract methods, add `@jaxtyped(typechecker=beartype)` to each, and optionally override `forward_drift`/`diffusion_coeff` for closed-form speed-ups.

## Adding a new sampling function

Follow the pattern in `sampling.py`: validate the time grid with `_validate_time_grid`, delegate to `euler_maruyama`, annotate `x: Float[Tensor, "*batch D"]` and `t: Float[Tensor, " T"]`.

## What NOT to do

- Do not add `try/except` — let bad inputs raise naturally.
- Do not import `from typing import Callable` — use `from beartype.typing import Callable` to avoid PEP 585 deprecation warnings from beartype.
- Do not add logic to `__init__.py` — it is a re-export surface only.
- Do not call `plt.show()` in test helpers that are unconditionally invoked; keep plot calls commented out or guarded.
