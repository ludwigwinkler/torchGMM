# Continuous Feynman--Kac Steering for Churn Samplers

`steered_reverse_churn_sampling` combines EDM-style churn with the
continuous Feynman--Kac-Corrector (FKC) weight used by
`steered_reverse_sampling`.  The sampler has one steering contract:

```python
trajectory, ess_history, weight_history = steered_reverse_churn_sampling(
    drift, schedule.transition, weight_update, x, t, churn=1.0
)
```

`weight_update` is required. It returns an instantaneous log-importance-weight
rate and owns all target-specific compensation for the caller's guided drift. `churn`
may be a non-negative scalar or a callback `churn(t) -> non-negative float`.

## Churn step semantics

Schedule time increases from data (`t=0`) to noise (`t=1`).  The sampler
requires a strictly decreasing grid.  For one step, write

$$
h_i=t_i-t_{i+1}>0,\qquad
\kappa_i=\operatorname{churn}(t_i),\qquad
\hat t_i=\min(t_i+\kappa_i h_i,1),\qquad
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
$(\hat x_i,t_i)$. `Schedule.transition` supplies the stochasticity; the
drift must therefore be a probability-flow velocity, not a reverse-SDE
drift with an additional score correction.

In the implementation this is:

```python
base_step = t_curr - t_next
churn_value = churn(t_curr) if callable(churn) else churn
t_hat = (t_curr + churn_value * base_step).clamp(max=1.0)
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

the fine-grid limit of time-dependent churn $\kappa_t$ has reverse-time
diffusion $g_{\kappa_t}=\sqrt{\kappa_t}\,g$ and base drift

$$
b_{\kappa_t}=f-\frac{1+\kappa_t}{2}g^2s.
$$

### Backward-time differential

All reverse equations in this document use the original schedule clock: time
still increases from data to noise, so sampling takes **negative** increments
$dt<0$. Let

$$
s_t(x)=\nabla_x\log q_t(x),\qquad
\kappa_t=\operatorname{churn}(t),\qquad
d\bar W_t\,d\bar W_t^\mathsf{T}=-dt\,I.
$$

The last identity defines the reverse Brownian increment: for a finite
backward step, $d\bar W_t=\sqrt{-dt}\,\epsilon$ with
$\epsilon\sim\mathcal N(0,I)$. The unsteered fine-grid process represented by
the churn split is therefore

$$
\boxed{\;
dX_t=
\left[
f(X_t,t)
-\frac{1+\kappa_t}{2}g(t)^2s_t(X_t)
\right]dt
+\sqrt{\kappa_t}\,g(t)\,d\bar W_t,
\qquad dt<0.
\;}
$$

For a reward tilt $\rho_t(x)=\beta(t)r_t(x)$, define

$$
a_t=\frac{\beta(t)\kappa_t g(t)^2}{2}.
$$

The guided process is

$$
\boxed{\;
dX_t=
\left[
f(X_t,t)
-\frac{1+\kappa_t}{2}g(t)^2s_t(X_t)
+a_t\nabla_x r_t(X_t)
\right]dt
+\sqrt{\kappa_t}\,g(t)\,d\bar W_t,
\qquad dt<0.
\;}
$$

Thus an Euler--Maruyama step on a decreasing grid
$dt_i=t_{i+1}-t_i<0$ is

$$
\begin{aligned}
X_{i+1}={}&X_i+
\left[
f(X_i,t_i)
-\frac{1+\kappa_i}{2}g(t_i)^2s_{t_i}(X_i)
+a_i\nabla_x r_{t_i}(X_i)
\right]dt_i\\
&+\sqrt{\kappa_i}\,g(t_i)\sqrt{-dt_i}\,\epsilon_i,
\qquad
\epsilon_i\sim\mathcal N(0,I),\quad
\kappa_i=\operatorname{churn}(t_i).
\end{aligned}
$$

`steered_reverse_churn_sampling` does **not** take this Euler step directly.
Instead, it realizes the same infinitesimal backward generator by an exact
forward reheat followed by a probability-flow transport. With
$h_i=-dt_i>0$, its two signed time increments are

$$
d t_{\rm reheat}=\kappa_i h_i>0,\qquad
d t_{\rm transport}=-(1+\kappa_i)h_i<0.
$$

The reheat is an exact draw from $q_{t_i}$ to $q_{t_i+d t_{\rm reheat}}$; the
transport then integrates the caller-supplied probability-flow drift over
$d t_{\rm transport}$. The factor $1/(1+\kappa_i)$ in the supplied guidance
field makes that longer transport produce exactly the $a_i\nabla r\,dt_i$
term in the boxed backward differential.

The deterministic leg lasts $(1+\kappa_i)h_i$, so the caller divides the
guidance field by $1+\kappa_t$:

```python
def guided_drift(x, t):
    kappa = churn(t) if callable(churn) else churn
    g2 = schedule.diffusion_coeff(t).square()
    guidance = beta(t) * kappa * g2 / (2 * (1 + kappa)) * grad_reward(x, t)
    return gmm.velocity(x, t) + guidance
```

Writing the backward base-grid step as $dt>0$, so that
$t_{\rm next}=t-dt$, the entire deterministic drift of the effective
guided reverse process is

$$
\boxed{\;
x_{t-dt}
=x_t-dt\left[
f(x_t,t)
-\frac{1+\kappa_t}{2}g(t)^2s_t(x_t)
+\underbrace{\frac{\beta(t)\kappa_t g(t)^2}{2}}_{a_t}
\nabla_x r_t(x_t)
\right].
\;}
$$

Equivalently,

$$
a_t=\frac{\beta(t)\kappa_t g(t)^2}{2}.
$$

The sampler realizes this through its reheat-then-transport split. Away from
the $t=1$ clamp, it first draws $\hat x_{\hat t}$ at
$\hat t=t+\kappa_t dt$, then takes the deterministic transport step

$$
\boxed{\;
x_{t-dt}
=\hat x_{\hat t}
-(1+\kappa_t)dt\left[
v(\hat x_{\hat t},\hat t)
+\frac{a_{\hat t}}{1+\kappa_t}\nabla_x r_{\hat t}(\hat x_{\hat t})
\right].
\;}
$$

The exact forward reheat supplies the stochasticity and its first-order
forward drift; together with this transport it yields the effective reverse
drift above.

### Matching $\beta(t)$ to $g(t)^2$

The field that appears in the backward guided SDE is not $\beta(t)$ alone,
but

$$
a_t=\frac12\beta(t)\kappa_t g(t)^2.
$$

Thus, choose $\beta(t)$ from the desired guidance amplitude, rather than
treating it as independent of the schedule. A regularized choice for a
nominal coefficient $a_{\rm target}(t)$ is

$$
\boxed{\;
\beta(t)=\frac{2a_{\rm target}(t)}
{\kappa_t\left(1+g(t)^2\right)}.
\;}
$$

It gives the bounded effective coefficient

$$
\boxed{\;
a_t=a_{\rm target}(t)\frac{g(t)^2}{1+g(t)^2}.
\;}
$$

In particular, choosing

$$
\beta(t)=\frac{c(t)}{1+g(t)^2}
\quad\Longrightarrow\quad
a_t=\frac{\kappa_t c(t)}{2}\frac{g(t)^2}{1+g(t)^2}
$$

caps the schedule factor at one. This is useful for VE and Karras schedules,
where $g(t)^2$ can vary by many orders of magnitude: a constant $c$ prevents
the guidance from becoming excessively large at times with large $g(t)^2$.

This is a choice of the **intermediate target path**:
$p_t^{\rm tilt}(x)\propto q_t(x)\exp(\beta(t)r_t(x))$. It is not merely a
numerical rescaling. The FKC callback must include the resulting
$\dot\beta(t)$ term, and the chosen endpoint $\beta(0)$ determines the final
reward tilt. Do not divide by $\kappa_t$ at a zero: when
$\kappa_t=0$, churn contributes no stochastic guidance, and the sampler
switches off resampling by design.

The matching FKC log-weight is a bounded-variation increment.  Its
continuous integrand is independent of `churn`, so the same
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
* the terminal particle cloud is resampled and its reported weights are uniform while
  churn remains positive. Once `churn(t)` returns zero, resampling is permanently
  disabled and the reported weights continue accumulating through the endpoint.

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
