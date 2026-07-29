# CLAUDE.md — torchGMM

Package-level guidance for working inside this directory.

## Module map

| File | Responsibility |
|---|---|
| `__init__.py` | Public re-exports only — no logic |
| `gmm.py` | `GMM` and `Conditional` classes |
| `schedule.py` | `Schedule`, `BetaSchedule`, `LinearSchedule`, `VESchedule`, `KarrasSchedule` |
| `sampling.py` | `euler_maruyama`, `forward_sampling`, `reverse_sampling`, `reverse_churn_sampling`, `steered_reverse_sampling`, `steered_reverse_churn_sampling` |

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

Follow the pattern in `sampling.py`: validate the time grid with `_validate_time_grid`, check the
direction, delegate to `euler_maruyama`, annotate `x: Float[Tensor, "*batch D"]` and
`t: Float[Tensor, " T"]`.

A sampler that is *not* an Euler-Maruyama discretisation gets its own function rather than an
integrator hook on the existing ones — `reverse_churn_sampling` is the example. Its two callables
are complete operators (`schedule.transition`, `gmm.velocity`), not the drift/diffusion terms of a
single SDE, so it names them for what they are and owns its own loop. Take the callables off the
`Schedule`/`GMM` directly; don't capture the schedule object wholesale or wrap it in a factory.

The velocity in both churn samplers is evaluated at the **reheated** time `t̂`, not at `t`: after
the churn the state is distributed according to `p_t̂`, so the score at `t` is the wrong one for it
(`docs/fkc_churn_steering.md` §5).

## Steering: two samplers, two weight contracts

`steered_reverse_sampling` takes an *incremental* `weight_update(x, t, dt) -> [N]` because it
discretises a continuous-time SDE, so the weight is the Prop. D.6 integrand times `dt`.

`steered_reverse_churn_sampling` instead takes the tilt exponent itself, `potential(x, t) -> [N]`,
because a churn step is a discrete proposal kernel that already maps `q_t` onto `q_{t+dt}` exactly.
The whole correction is then the endpoint difference `ρ_{t+dt}(x_{t+dt}) − ρ_t(x_t)` — no `β̇_t`,
no `∂_t r`, no reward Laplacian, no score-alignment inner product, and no reward gradient at all,
so a denoiser inside `potential` never needs to be backpropagated through. Do not hand it a
reward-guided drift: the base probability flow must stay untouched for the endpoint difference to
be the exact correction (`docs/fkc_churn_steering.md` §3, §6).

`ρ` is carried across steps and *gathered* on a resample rather than re-evaluated — `potential`
may be expensive, and recomputing the ancestor's tilt would double its cost per step.

## What NOT to do

- Do not add `try/except` — let bad inputs raise naturally.
- Do not import `from typing import Callable` — use `from beartype.typing import Callable` to avoid PEP 585 deprecation warnings from beartype.
- Do not add logic to `__init__.py` — it is a re-export surface only.
- Do not call `plt.show()` in test helpers that are unconditionally invoked; keep plot calls commented out or guarded.
