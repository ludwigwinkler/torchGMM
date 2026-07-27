# Testing philosophy

The thing worth testing here is the **stochastic process**, not the plumbing. A shape
assertion passes on a sampler that integrates the wrong drift. A Wasserstein check at
intermediate times does not.

## The core pattern

Every sampler test is the same three lines of intent:

1. Run the sampler and keep the **whole trajectory** `[T, N, *B, D]`, not just the endpoint.
2. At a set of intermediate `t_idx`, build the **analytic** marginal — `gmm.log_prob(x_grid, t=t_).exp()`
   (or `_tilted_density(...)` for steering) — on a fixed grid.
3. Assert **W1(empirical, analytic) < tol** at *each* of those times.

```python
for t_idx in range(n_steps)[::5]:
    t_ = t[t_idx]
    w1 = _wasserstein1(trajectory[t_idx, :, 0, 0], gmm, t_, x_grid, bin_edges)
    assert w1 < 0.08, f"{schedule_cls.__name__} @ t={t_:.3f}: W1 {w1:.3f}"
```

Endpoint-only checks are not enough: a sampler can drift off the marginal path in the
middle and be pulled back by the score near `t=0`. The intermediate times are where the
bugs live — wrong `dt` sign, missing `1/2` on `g²`, a schedule derivative off by a factor.

## W1 helpers

- `test_sampling.py::_wasserstein1` — grid/CDF version: histogram the samples, cumsum both
  CDFs, `|F_emp − F_target|.sum() * dx`. Cheap; used for unweighted trajectories.
- `test_steering.py::_wasserstein1` — quantile-matching version: sort samples, match rank
  levels against the target CDF, mean `|s − q|`. No binning artefacts.
- `test_steering.py::_weighted_wasserstein1` — same, with **importance weights**. This is the
  one that matters for FKC: SMC particles are only correct *as a weighted cloud*, so the
  unweighted empirical distribution is expected to be wrong between resamples.

Pair W1 with a max-histogram-deviation check when you also care about local shape; W1 alone
tolerates a mode that is right in mass but shifted, `hist_dev` alone tolerates a global shift.

## Steering tests

The target is `p_t(x) ∝ q_t(x) exp(β(t)·r(x))` — `_tilted_density`. Assert
`_weighted_wasserstein1(x_t, weight_hist[t_idx], ...)` against it at intermediate `t`. That is
the *only* check that says FKC is actually correct rather than merely producing high-reward
samples; a biased sampler with a strong reward also produces high-reward samples.

Sweep `ess_threshold` across the three regimes (adaptive `0<τ<1`, interval `τ≥1`, and
`τ > n_steps` = single final resample) — all must hit the same tilted marginals. Divergence
between them is a resampling bug.

## What not to spend tests on

Shape/dtype/device assertions, `t`-validation errors, "is the ODE deterministic", getter
round-trips — one cheap instance each, then stop. They are already implied by the marginal
tests: nothing passes W1 at 40 time points with a broken shape contract. Don't parametrize
these across schedules; do parametrize the marginal tests across schedules, `gamma`, and
`reward_center`.

The one legitimately-functional test class is `TestSteeredSamplingResampleModes`: it verifies
*resample timing* (control flow), which is discrete and cannot be observed through a W1
statistic. It uses zero drift so particle values change only via resampling, and predicts the
ESS trace analytically — no `manual_seed` needed, so it can't flake under xdist.

## Practical notes

- Keep `t` away from schedule singularities: `BetaSchedule` velocity blows up at `t=0`,
  `LinearSchedule` at `t=1` → integrate over `[eps, 1−eps]`.
- Sample counts: ~10k for unweighted W1 at tol 0.08, ~5k particles for steering. Below that the
  Monte-Carlo error in W1 is comparable to the tolerance and the test flakes.
- Mark anything running a full SDE sweep `@pytest.mark.slow`.
- `PLOT = True` in `test_steering.py` dumps density-comparison PNGs to `tests/plots/`. Use it
  when a W1 assertion fails — the plot tells you *how* the marginal is wrong (shifted, wrong
  mode weights, over-concentrated) far faster than the scalar does.
