# Karras Noise Schedule and Its Implied Forward Process

This note describes the `KarrasSchedule` used in `torchGMM`: the noise schedule,
the forward and reverse SDEs it implies, and how it composes with the
Feynman-Kac-Corrector steering of `docs/fkc_steering.md`. It deliberately ignores
the EDM/Karras denoiser parameterization, preconditioning, and sampler
corrections.

## Schedule definition

`KarrasSchedule` is a variance-exploding (VE) schedule:

$$\alpha(t) \equiv 1,\qquad t\in[0,1].$$

The marginal noise standard deviation is

$$\bar\sigma(t)
= \sigma_{\mathrm{data}}\left(
\sigma_{\min}^{1/\rho}
+ t\left(\sigma_{\max}^{1/\rho}-\sigma_{\min}^{1/\rho}\right)
\right)^\rho.$$

Equivalently, with

$$u(t)=\sigma_{\min}^{1/\rho}
+ t\left(\sigma_{\max}^{1/\rho}-\sigma_{\min}^{1/\rho}\right),
\qquad
\bar\sigma(t)=\sigma_{\mathrm{data}}\,u(t)^\rho.$$

So time is linear in $\bar\sigma^{1/\rho}$, not in $\bar\sigma$. The parameter
$\rho$ controls how the grid concentrates: $\rho=1$ is linear in $\bar\sigma$,
while the common EDM/AF3 value $\rho=7$ spends more resolution near low noise.
The derivative used by the SDE is

$$\dot{\bar\sigma}(t)
= \sigma_{\mathrm{data}}\,\rho\,u(t)^{\rho-1}
\left(\sigma_{\max}^{1/\rho}-\sigma_{\min}^{1/\rho}\right).$$

## Implied marginal path

For a clean data distribution $q_{\mathrm{data}}$, the schedule defines the
Gaussian-convolved marginal

$$q_t = q_{\mathrm{data}} * \mathcal{N}\!\left(0,\bar\sigma(t)^2I\right),$$

which for a GMM stays exact and in-family:

$$q_t(x)=\sum_k \pi_k\,
\mathcal{N}\!\left(x;\mu_k,\Sigma_k+\bar\sigma(t)^2I\right).$$

Note that $\bar\sigma(0)=\sigma_{\mathrm{data}}\sigma_{\min}>0$, so $t=0$ is a
small-noise endpoint rather than exactly clean data. With the default
$\sigma_{\min}=4\times10^{-4}$ this is a variance of order $10^{-7}\sigma_{\mathrm{data}}^2$
and is treated as the data endpoint throughout; the exact-endpoint variant is
noted once in the conditional section below.

## Implied forward SDE

Start from the noising path itself. Couple all times with the same clean sample
$X_{\mathrm{data}}\sim q_{\mathrm{data}}$ and the same Gaussian noise $\epsilon$:

$$X_t = X_{\mathrm{data}} + \bar\sigma(t)\epsilon,
\qquad \epsilon\sim\mathcal{N}(0,I).$$

At each fixed $t$ this reproduces the marginal $q_t$ above. Differentiating the
marginal variance in time, the clean-data part is constant, so

$$\frac{d}{dt}\mathrm{Var}[X_t]
= \frac{d}{dt}\mathrm{Var}\!\left[\bar\sigma(t)\epsilon\right]
=\frac{d}{dt}\bar\sigma(t)^2I
=2\bar\sigma(t)\dot{\bar\sigma}(t)I.$$

A driftless VE SDE $dX_t = g(t)\,dW_t$ accumulates covariance $g(t)^2I\,dt$ per
infinitesimal interval, so $\frac{d}{dt}\mathrm{Var}(X_t)=g(t)^2I$. Matching the
two variance growth rates gives

$$g(t)^2=\frac{d}{dt}\bar\sigma(t)^2
=2\bar\sigma(t)\dot{\bar\sigma}(t),$$

the last equality being the chain rule. This is what
`KarrasSchedule.diffusion_coeff(t)` implements:

$$g(t)=\sqrt{2\bar\sigma(t)\dot{\bar\sigma}(t)},
\qquad
dX_t = \sqrt{2\bar\sigma(t)\dot{\bar\sigma}(t)}\,dW_t.$$

### $\bar\sigma(t)^2$ is not $g(t)^2$

This is the one notation trap worth stating explicitly.

$$\bar\sigma(t)^2 \neq g(t)^2.$$

$\bar\sigma(t)^2$ is the **marginal noise variance already accumulated** by time
$t$; $g(t)^2$ is the **instantaneous variance injection rate** at time $t$. The
correct chain is

$$\bar\sigma(t)\;\longrightarrow\;\bar\sigma(t)^2
\;\longrightarrow\;\frac{d}{dt}\bar\sigma(t)^2
=2\bar\sigma(t)\dot{\bar\sigma}(t)
\;\longrightarrow\;g(t)^2.$$

Going the other way, the accumulated variance is recovered by integration, which
is just a change of variables:

$$\bar\sigma(t)^2-\bar\sigma(t_0)^2
=\int_{t_0}^{t}g(s)^2\,ds
=\int_{t_0}^{t}2\bar\sigma(s)\dot{\bar\sigma}(s)\,ds
=\int_{\bar\sigma(t_0)}^{\bar\sigma(t)}2\sigma\,d\sigma.$$

The Fokker-Planck equation of the forward SDE,

$$\partial_t q_t = \frac{1}{2}g(t)^2\Delta q_t
= \frac{1}{2}\frac{d}{dt}\bar\sigma(t)^2\Delta q_t,$$

is exactly the heat-kernel identity for Gaussian convolution, confirming the
match.

## Forward Euler-Maruyama

In infinitesimal form, with `KarrasSchedule.forward_drift` returning zero,

$$dX_t = g(t)\,dW_t,\qquad dW_t \sim \mathcal{N}(0,dt\,I),\qquad dt>0,$$

so $\mathrm{Var}[dX_t\mid X_t]=g(t)^2dt\,I$ and the local update is

$$X_{t+dt}=X_t+g(t)\sqrt{dt}\,\epsilon,\qquad
\epsilon\sim\mathcal{N}(0,I).$$

This is the local approximation
$\int_t^{t+dt}g(s)^2\,ds \approx g(t)^2dt$ of the exact transition variance
$\bar\sigma(t+dt)^2-\bar\sigma(t)^2$. When only forward noising is needed and the
interval is too coarse for that approximation, the exact transition

$$X_{t+dt}=X_t
+\sqrt{\bar\sigma(t+dt)^2-\bar\sigma(t)^2}\,\epsilon$$

is available in closed form.

## Reverse SDE

Let $s_t(x)=\nabla_x\log q_t(x)$. Since the forward drift is zero, the Anderson
reverse-time SDE is

$$dX_t = -g(t)^2s_t(X_t)\,dt + g(t)\,d\bar W_t,$$

integrated along a decreasing time grid, so $dt<0$. Because $dt<0$, the
deterministic part moves in the denoising direction $+g(t)^2s_t(X_t)|dt|$. In the
repository's `reverse_sampling` convention the drift callable is the signed
forward-time version:

```python
def reverse_drift(x, t):
    g = schedule.diffusion_coeff(t)
    return -(g**2) * gmm.score(x, t)
```

and the Euler-Maruyama update is

$$X_{t+dt}
= X_t - g(t)^2s_t(X_t)\,dt
+ g(t)\sqrt{|dt|}\,\epsilon,\qquad dt<0.$$

## Conditional reverse SDE for a single clean point

Take a single clean point $x_0$ and the idealized clean endpoint
$\bar\sigma(0)=0$. The VE marginal and its score are

$$q_t(x\mid x_0)=\mathcal{N}\!\left(x;x_0,\bar\sigma(t)^2I\right),
\qquad
s_t(x\mid x_0)=-\frac{x-x_0}{\bar\sigma(t)^2}.$$

Substituting into the reverse SDE and using
$g(t)^2=2\bar\sigma(t)\dot{\bar\sigma}(t)$:

$$
\begin{align}
dX_t
&= -g(t)^2 s_t(X_t\mid x_0)\,dt + g(t)\,d\bar W_t \\
&= g(t)^2\frac{X_t-x_0}{\bar\sigma(t)^2}\,dt + g(t)\,d\bar W_t \\
&= 2\dot{\bar\sigma}(t)\frac{X_t-x_0}{\bar\sigma(t)}\,dt
+ \sqrt{2\bar\sigma(t)\dot{\bar\sigma}(t)}\,d\bar W_t,
\qquad dt<0.
\end{align}$$

The chain is initialized at a terminal noisy sample $X_T=x_T$. Since $dt<0$, the
deterministic part contracts toward $x_0$:

$$2\dot{\bar\sigma}(t)\frac{X_t-x_0}{\bar\sigma(t)}\,dt
= -2\dot{\bar\sigma}(t)\frac{X_t-x_0}{\bar\sigma(t)}\,|dt|.$$

**Exact clean endpoint.** For the literal schedule, $\bar\sigma(0)>0$, and the
conditional formulas are made exact by replacing $\bar\sigma(t)^2$ with the
variance clock $V(t)=\bar\sigma(t)^2-\bar\sigma(0)^2$, giving
$s_t(x\mid x_0)=-(x-x_0)/V(t)$ and drift $\frac{g(t)^2}{V(t)}(X_t-x_0)$. At the
default $\sigma_{\min}$ the difference is negligible, and the rest of this note
uses $\bar\sigma(t)^2$.

**Not a bridge.** Pinning both endpoints, $X_0=x_0$ and $X_T=x_T$, is a different
conditioning problem — a diffusion bridge with marginal

$$X_t\mid x_0,x_T
\sim \mathcal{N}\!\left(
x_0+\frac{\bar\sigma(t)^2}{\bar\sigma(T)^2}(x_T-x_0),
\;\bar\sigma(t)^2\left(1-\frac{\bar\sigma(t)^2}{\bar\sigma(T)^2}\right)I
\right),$$

whereas the reverse SDE above conditions only on the current noisy state and is
initialized at $x_T$.

## Practical consequence

For $\sigma_{\max}=160$ and $\rho=7$, $g(t)^2=2\bar\sigma(t)\dot{\bar\sigma}(t)$
becomes very large near $t=1$. Euler-Maruyama is therefore sensitive to the
high-noise endpoint: stable runs want a sufficiently fine grid, a start time
slightly below $1$, or — when only forward noising is needed — the
exact-in-variance transition above.

## FKC steering under the Karras schedule

Combining the VE schedule with the steering equations of `docs/fkc_steering.md`,
the reward-tilted marginal is

$$p_t(x)\propto q_t(x)\exp\bigl(\beta(t)\,r(x,t)\bigr),$$

and for Karras

$$f_t(x)\equiv 0,\qquad
g(t)^2=2\bar\sigma(t)\dot{\bar\sigma}(t),\qquad
s_t(x)=\nabla_x\log q_t(x).$$

Writing the reverse-time increment as $d\tau=-dt>0$, the unsteered dynamics are
$dX_t = g(t)^2s_t(X_t)\,d\tau + g(t)\,dW_\tau$, and FKC adds the drift
$\beta(t)\frac{g(t)^2}{2}\nabla r$. The steered reverse SDE is therefore

$$
\begin{align}
dX_t
&= \left[
g(t)^2s_t(X_t)
+ \beta(t)\frac{g(t)^2}{2}\nabla r(X_t,t)
\right]d\tau
+ g(t)\,dW_\tau \\
&= 2\bar\sigma(t)\dot{\bar\sigma}(t)
\left[
s_t(X_t)+\frac{\beta(t)}{2}\nabla r(X_t,t)
\right]d\tau
+ \sqrt{2\bar\sigma(t)\dot{\bar\sigma}(t)}\,dW_\tau,
\end{align}
$$

with log-weight increment

$$
\begin{align}
dw_t
&= \left[
-\dot\beta(t)r(X_t,t)
-\beta(t)\partial_t r(X_t,t)
+ \left\langle\beta(t)\nabla r(X_t,t),
\frac{g(t)^2}{2}s_t(X_t)
\right\rangle
\right]d\tau \\
&= \left[
-\dot\beta(t)r(X_t,t)
-\beta(t)\partial_t r(X_t,t)
+ \beta(t)\bar\sigma(t)\dot{\bar\sigma}(t)
\left\langle\nabla r(X_t,t),s_t(X_t)\right\rangle
\right]d\tau.
\end{align}
$$

The samplers use a decreasing forward-time grid, so $dt<0$ and $d\tau=|dt|$. The
callables passed to `steered_reverse_sampling` are

```python
def guided_drift(x, t):
    g = schedule.diffusion_coeff(t)
    score = gmm.score(x, t)
    return -(g**2) * score - beta(t) * (g**2 / 2) * grad_r(x, t)


def weight_update(x, t, dt):
    g = schedule.diffusion_coeff(t)
    score = gmm.score(x, t)
    integrand = (
        -dbeta_dt(t) * r(x, t)
        - beta(t) * partial_t_r(x, t)
        + beta(t) * ((g**2) / 2) * (grad_r(x, t) * score).sum(dim=-1)
    )
    return integrand * dt.abs()
```

If the reward has no explicit time dependence, `partial_t_r` is zero.

## Choosing the tilt schedule $\beta(t)$

Write $\beta$ as a quotient of an **annealing schedule** and a **variance-matching
schedule**:

$$\beta(t)=\frac{\beta^{\mathrm{nom}}(t)}{\beta^{\mathrm{denom}}(t)},\qquad
\beta^{\mathrm{nom}}(t)=(1-t)^p,\qquad
\beta^{\mathrm{denom}}(t)=1+\bar\sigma(t)^2.$$

**Numerator — annealing.** $\beta^{\mathrm{nom}}(1)=0$, $\beta^{\mathrm{nom}}(0)=1$: the tilt is off at the noise end
and full at the data end. This is what carries correctness. The terminal target
is $p_0\propto q_0\exp(\beta(0)r)$, so sampling $q_0e^{r}$ requires $\beta(0)=1$,
and $\beta(1)=0$ makes $p_1=q_1$ so particles can be initialized from the
unsteered prior. Nothing in between matters for the target: the $-\dot\beta r$
term in $dw_t$ pays for the path, telescoping to $r(\beta(0)-\beta(1))$ when $r$
is path-constant. The shape of $\beta^{\mathrm{nom}}$ on $(0,1)$ buys ESS, not
correctness. The exponent $p>1$ is one such choice: for $p\le1$,
$\dot\beta^{\mathrm{nom}}\sim(1-t)^{p-1}$ diverges
at $t\to1$ and the $-\dot\beta r$ term spikes exactly where $r$ on a denoised
estimate is least informative.

**Denominator — variance matching.** The score scale is $1/\bar\sigma(t)^2$, so
balancing the reward gradient against it wants $\beta\propto1/\bar\sigma(t)^2$;
the $1+$ is the floor that keeps $\beta^{\mathrm{denom}}$ from collapsing as
$\bar\sigma(t)\to\bar\sigma(0)\approx0$. With an attracting quadratic reward
$r(x)=-(x-b)^2$ and $s_t(x)=-(x-x_0)/\bar\sigma(t)^2$, the drift bracket is

$$2\bar\sigma(t)\dot{\bar\sigma}(t)
\left[
-\frac{X_t-x_0}{\bar\sigma(t)^2}
-\beta(t)\left(X_t-b\right)
\right]d\tau,$$

both terms restoring, with dimensionless ratio

$$\frac{\text{reward drift}}{\text{score drift}}
=\beta(t)\,\bar\sigma(t)^2\,
\frac{\lVert X_t-b\rVert}{\lVert X_t-x_0\rVert}
= \underbrace{\beta^{\mathrm{nom}}(t)}_{\text{annealing}} \ \ \underbrace{\frac{\bar\sigma(t)^2}{1+\bar\sigma(t)^2}\,
\frac{\lVert X_t-b\rVert}{\lVert X_t-x_0\rVert}}_{\text{balanced}}.$$

The factor $\bar\sigma^2/(1+\bar\sigma^2)$ saturates to $1$ at high noise and
decays like $\bar\sigma^2$ at low noise, bounding the ratio by
$\beta^{\mathrm{nom}}(t)\le1$ everywhere. Drop the denominator and the ratio is
$\propto \beta^{\mathrm{nom}}(t)\bar\sigma(t)^2$,
which at $\sigma_{\max}=160$ is $O(10^4)$ over most of the grid — the reward
gradient overruns the score. The floor also settles the endpoint for free:
$\beta^{\mathrm{denom}}(0)=1+\bar\sigma(0)^2\to1$, so
$\beta(0)=\beta^{\mathrm{nom}}(0)=1$ (exact to $O(10^{-5})$ at the
default $\sigma_{\min}$; divide by $\beta^{\mathrm{denom}}(0)$ if
$\bar\sigma(0)$ is not small).

**Derivative.** By logarithmic differentiation with
$\frac{d}{dt}\bar\sigma(t)^2=g(t)^2$, the `dbeta_dt` callable is

$$\dot\beta(t)
=-\beta(t)\left[
\frac{p}{1-t}+\frac{g(t)^2}{1+\bar\sigma(t)^2}
\right],$$

finite at both ends: $p/(1-t)\to p$ at $t=0$, while at $t=1$ the prefactor
$\beta\sim(1-t)^p$ kills the pole for $p>1$ and $g^2/(1+\bar\sigma^2)$ stays
bounded because the denominator grows with the same variance that drives $g^2$.

The notebooks currently implement the numerator only — $(1-t)^2$ (active),
$\cos^2(\tfrac{\pi}{2}t)$, $1-t$ — without the variance denominator.
