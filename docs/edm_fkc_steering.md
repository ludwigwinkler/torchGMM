# Continuous Feynman--Kac Steering for Churn Samplers

`steered_reverse_churn_sampling` combines EDM-style churn with the
continuous Feynman--Kac-Corrector (FKC) weight used by
`steered_reverse_sampling`.  The sampler has one steering contract:

```python
trajectory, ess_history, weight_history = steered_reverse_churn_sampling(
    drift, schedule.transition, weight_update, x, t, churn=gamma
)
```

`weight_update` is required. It returns an instantaneous log-importance-weight
rate and owns all target-specific compensation for the caller's guided drift.

## Churn step semantics

Schedule time increases from data (`t=0`) to noise (`t=1`).  The sampler
requires a strictly decreasing grid.  For one step, write

$$
h_i=t_i-t_{i+1}>0,\qquad
\hat t_i=\min(t_i+\gamma h_i,1),\qquad
\Delta t_{\mathrm{transport}}=t_{i+1}-\hat t_i<0.
$$

It first draws the exact forward transition

$$
\hat x_i\sim q(\,\cdot\mid x_i;t_i,\hat t_i),
$$

then uses the caller's probability-flow drift for the deterministic move

$$
x_{i+1}=\hat x_i+
\operatorname{drift}(\hat x_i,\hat t_i)
\Delta t_{\mathrm{transport}}.
$$

The score follows the state: evaluate the velocity, guided drift, denoiser,
and FKC callback at the reheated pair $(\hat x_i,\hat t_i)$, never at
$(\hat x_i,t_i)`.  `Schedule.transition` supplies the stochasticity; the
drift must therefore be a probability-flow velocity, not a reverse-SDE
drift with an additional score correction.

In the implementation this is:

```python
base_step = t_curr - t_next
t_hat = (t_curr + churn * base_step).clamp(max=1.0)
transport_dt = t_next - t_hat

if t_hat > t_curr:
    x = transition(x, t_curr, t_hat)
log_w += weight_update(x, t_hat) * base_step
x = x + drift(x, t_hat) * transport_dt
```

The callback receives no step argument. The solver multiplies its rate by
`base_step`, the base reverse-grid magnitude; it does not use the longer
transport span.

## Continuous FKC weight and guidance

For

$$
dX_t=f(X_t,t)\,dt+g(t)\,dW_t,\qquad
v_t=f_t-\frac12g_t^2s_t,
$$

the fine-grid limit of churn strength $\gamma$ has reverse-time diffusion
$g_\gamma=\sqrt{\gamma}\,g$ and base drift

$$
b_\gamma=f-\frac{1+\gamma}{2}g^2s.
$$

For a reward tilt $\rho_t(x)=\beta(t)r_t(x)$, continuous FKC uses the
effective guidance coefficient

$$
a(t)=\frac{\beta(t)\gamma g(t)^2}{2}.
$$

The deterministic leg lasts $(1+\gamma)h_i$, so the caller divides the
guidance field by $1+\gamma$:

```python
def guided_drift(x, t):
    g2 = schedule.diffusion_coeff(t).square()
    guidance = beta(t) * gamma * g2 / (2 * (1 + gamma)) * grad_reward(x, t)
    return gmm.velocity(x, t) + guidance
```

The matching FKC log-weight is a bounded-variation increment.  Its
continuous integrand is independent of `gamma`, so the same
`weight_update(x, t)` closure is valid for both the churn and Euler--Maruyama
steered samplers. For a time-dependent reward it includes the tilt-time
derivative and the score-alignment term. Each solver applies its own
appropriate base-grid magnitude.

```python
def weight_update(x, t):
    g2 = schedule.diffusion_coeff(t).square()
    integrand = (
        tilt_time_derivative(x, t)
        + (beta(t) * grad_reward(x, t) * (g2 / 2) * gmm.score(x, t)).sum(dim=-1)
    )
    return integrand
```

The exact signs depend on whether the target is written as
`exp(beta * reward)` or `exp(-beta * energy)`; define the guided drift and
the callback from the same convention.

## SMC behavior

The sampler records normalized particle weights at every grid point and
resamples only after a complete churn-plus-transport step:

* `0 < ess_threshold < 1`: resample when `ESS / N` falls below the threshold.
* integer `ess_threshold >= 1`: resample every that many steps.
* the terminal particle cloud is always resampled and its reported weights
  are uniform.

Between resamples, evaluate correctness with the weighted cloud at
intermediate times.  The tests sweep churn, adaptive/fixed resampling, and
Beta/VE schedules against the corresponding analytic tilted marginals.

## VE and Karras schedules

For VE schedules, $\alpha_t=1$ and

$$
g(t)^2=\frac{d}{dt}\sigma(t)^2=2\sigma(t)\dot\sigma(t).
$$

The marginal variance $\sigma(t)^2$ and instantaneous diffusion rate
$g(t)^2$ are distinct.  Use the latter in the FKC guidance coefficient and
weight integrand.  Karras schedules can have large $g(t)^2$ near the noisy
end, so start just below `t=1` and use a sufficiently fine reverse grid.
`docs/schedule.md` gives the schedule-specific formulas.
