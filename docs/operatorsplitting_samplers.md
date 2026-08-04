# Operator-Splitting Samplers for Reverse Diffusion

This document describes reverse-diffusion samplers that alternate an exact
forward noising operation with deterministic probability-flow denoising.  Time
always follows the repository convention: data are at `t = 0`, noise is at
`t = 1`, and reverse sampling uses a **strictly decreasing** grid.

Write the forward marginal path as

$$
X_t=\alpha(t)X_0+\sigma(t)\epsilon,\qquad
\epsilon\sim\mathcal N(0,I),
$$

and let $q_t$ denote its marginal density.  The square $\sigma(t)^2$ is the
noise variance accumulated in this interpolant conditional on $X_0$; it is
not, in general, the total variance of $q_t$.

## 1. Native EDM on the VE noise-scale clock

EDM is native to a variance-exploding (VE) path,

$$
X_\sigma=X_0+\sigma\epsilon,
$$

where the noise scale $\sigma$ itself is the independent integration
coordinate.  A native EDM grid is decreasing:

$$
\sigma_i>\sigma_{i+1}\geq0.
$$

### Churn / reheating

For a churn strength $\gamma_i\geq0$, EDM reheats

$$
\hat\sigma_i=(1+\gamma_i)\sigma_i
$$

and draws

$$
\hat X_i=X_i+
\sqrt{\hat\sigma_i^2-\sigma_i^2}\,\epsilon_i,
\qquad \epsilon_i\sim\mathcal N(0,I).
$$

Thus, with the usual unit noise multiplier, the injected variance per
coordinate is

$$
\hat\sigma_i^2-\sigma_i^2
=\gamma_i(2+\gamma_i)\sigma_i^2.
$$

EDM may multiply the injected standard deviation by an empirical
`S_noise`; then the injected variance is `S_noise`$^2$ times this value and
the reheating is no longer the exact VE transition unless `S_noise = 1`.
The commonly used bound $\gamma_i\leq\sqrt{2}-1$ makes
$\hat\sigma_i^2\leq2\sigma_i^2$, equivalently bounds the injected variance by
the current noise variance.  It is a churn policy, not a change of direction:
reheating always moves toward larger $\sigma$.

### Probability-flow denoising

For a VE forward SDE with $g(t)^2=d\sigma(t)^2/dt$, its probability-flow ODE
is

$$
\frac{dX_t}{dt}=-\frac12g(t)^2\nabla_x\log q_t(X_t).
$$

Changing coordinates from $t$ to $\sigma$ gives

$$
\frac{dX}{d\sigma}
=-\sigma\nabla_x\log q_\sigma(X)
=\frac{X-D(X,\sigma)}{\sigma},
$$

where $D(X,\sigma)=X+\sigma^2\nabla_x\log q_\sigma(X)$ is the posterior-mean
denoiser.  This change of variable is where EDM's sigma-based deterministic
step size comes from: it is a signed increment in the **noise-scale
coordinate**, not a duration in an arbitrary schedule time.

After reheating, the Euler probability-flow step is

$$
h_i=\sigma_{i+1}-\hat\sigma_i<0,\qquad
X_{i+1}=\hat X_i+h_i\frac{\hat X_i-D(\hat X_i,\hat\sigma_i)}{\hat\sigma_i}.
$$

The negative sign of $h_i$ is essential: the vector
$(x-D)/\sigma$ points in the increasing-noise direction, so multiplying it by
$h_i<0$ denoises.  EDM Algorithm 2 commonly improves this Euler proposal with
a Heun correction: evaluate the same derivative at the proposal and
$\sigma_{i+1}$, then replace the derivative by their average.  The terminal
step to $\sigma=0$ uses the Euler result because the displayed derivative is
singular there.

The additive Gaussian reheating is exact for VE:
if $X_i\sim q_{\sigma_i}$, then $\hat X_i\sim q_{\hat\sigma_i}$.  The
deterministic leg is an ODE integrator and preserves the intended marginal
only to its numerical integration accuracy.

## 2. EDM-style churn on the repository time clock

`reverse_churn_sampling` uses the same two-stage idea, but its independent
coordinate is exclusively schedule time $t\in[0,1]$.  It does not create a
grid in $\sigma$ or take a step size in sigma units.

For one decreasing-grid step, define

$$
t_{\rm curr}>t_{\rm next},\qquad
h=t_{\rm curr}-t_{\rm next}>0,
$$

then compute

$$
\hat t=\min\{t_{\rm curr}+\kappa h,\,1\},\qquad
\Delta t_{\rm transport}=t_{\rm next}-\hat t<0,
$$

where $\kappa\geq0$ is `churn`.  In particular, $\hat t$ is **additive in
time**.  It is not $(1+\kappa)t_{\rm curr}$.  Clamping at one shortens the
actual reheating near the noisy boundary.

The step is

$$
\hat X\sim\operatorname{Schedule.transition}(X,t_{\rm curr},\hat t),
\qquad
X_{\rm next}=\hat X+v(\hat X,\hat t)\,
\Delta t_{\rm transport},
$$

where the first operation is skipped if $\hat t=t_{\rm curr}$ and $v$ is the
probability-flow velocity.  Since

$$
\Delta t_{\rm transport}
=-\left[h+(\hat t-t_{\rm curr})\right],
$$

the deterministic leg both undoes the reheating and advances the original
reverse-grid step.  It is negative even when churn is zero.

### Schedule quantities and the exact churn kernel

At any time $u$, the schedule supplies $\alpha(u)$, $\sigma(u)$, and their
derivatives.  Its implied forward SDE is

$$
dX_t=f(X_t,t)\,dt+g(t)\,dW_t,\qquad
f(x,t)=\frac{\dot\alpha(t)}{\alpha(t)}x,
$$

with

$$
g(t)^2=2\left(
\dot\sigma(t)\sigma(t)
-\frac{\dot\alpha(t)}{\alpha(t)}\sigma(t)^2
\right).
$$

The probability-flow velocity evaluated by the deterministic leg is

$$
v(x,t)=f(x,t)-\frac12g(t)^2\nabla_x\log q_t(x).
$$

Thus the time-indexed reverse differential equation integrated by the
deterministic half is

$$
\boxed{\;
dX_t=
\left[f(X_t,t)-\frac12g(t)^2\nabla_x\log q_t(X_t)\right]dt,
\qquad dt<0.
\;}
$$

The negative increment is what makes this an integration from noise towards
data.  A churn cycle does not Euler-discretize the Anderson reverse SDE

$$
dX_t=
\left[f(X_t,t)-g(t)^2\nabla_x\log q_t(X_t)\right]dt
+g(t)\,d\bar W_t,
\qquad dt<0.
$$

Instead, its exact forward transition supplies stochasticity over the
separate interval $t_{\rm curr}\to\hat t$, then the boxed probability-flow
ODE is integrated over $\hat t\to t_{\rm next}$.

Thus the score, velocity, and any quantities used to construct them are
evaluated at the **reheated pair** $(\hat X,\hat t)$, not at
$(\hat X,t_{\rm curr})$.  The state after the churn has marginal
$q_{\hat t}$, so using the old time's score would be inconsistent.

For $s>t$, `Schedule.transition` implements the exact forward kernel

$$
X_s=
\frac{\alpha(s)}{\alpha(t)}X_t+
\sqrt{
\sigma(s)^2-
\left(\frac{\alpha(s)}{\alpha(t)}\right)^2\sigma(t)^2
}\,\epsilon.
$$

Accordingly, the time-clock churn uses $\alpha$ and $\sigma$ at
$t_{\rm curr}$ and $\hat t$ to make one finite, exact forward draw.  It does
not use the instantaneous rate $g(t)^2$ as a finite-step variance.

In the VE case, $\alpha\equiv1$, this reduces to the additive reheating rule

$$
\boxed{\;
X_{\hat t}=X_{t_{\rm curr}}+
\sqrt{\sigma(\hat t)^2-\sigma(t_{\rm curr})^2}\,\epsilon.
\;}
$$

The square root covers the whole variance difference. This transition starts
from an already-noised $X_{t_{\rm curr}}$; it is not a fresh training draw
$X_{\hat t}=\alpha(\hat t)X_0+\sigma(\hat t)\epsilon$, which would require
the unknown clean sample $X_0$.

Do not replace the exact transition with one Euler forward-SDE step unless an
SDE discretisation is intentional:

$$
X_{t+h}\approx X_t+f(X_t,t)h+g(t)\sqrt{h}\,\epsilon.
$$

That update matches the exact kernel only to first order as $h\to0$. In
contrast, `Schedule.transition(X_t,t,\hat t)` maps $q_t$ to $q_{\hat t}$
exactly for any valid finite gap, which is the marginal-preservation property
the operator split relies on.

`steered_reverse_churn_sampling` accepts either form: it uses `transition`
when supplied and otherwise applies the additive Euler update from
`diffusion(t)`. The latter is consequently appropriate only for VE schedules
with zero forward drift; the exact transition remains the general,
marginal-preserving choice.

This distinction is especially important for VE schedules.  When
$\alpha(t)=1$,

$$
g(t)^2=\frac{d}{dt}\sigma(t)^2=2\sigma(t)\dot\sigma(t),
$$

whereas $\sigma(t)^2$ is accumulated noise variance.  They have different
units and different roles.  The exact finite-gap injected variance is
$\sigma(\hat t)^2-\sigma(t_{\rm curr})^2$; $g(t)^2h$ is only its
first-order small-$h$ approximation.  For non-VE schedules the exact kernel
also rescales $X_t$, so it is not “add Gaussian noise” at all.

### Score variance versus reverse-diffusion variance

Training samples are drawn from the forward marginal kernel

$$
X_t=\alpha(t)X_0+\sigma(t)\epsilon,\qquad
\epsilon\sim\mathcal N(0,I),
$$

or, equivalently,

$$
q_{0\to t}(x_t\mid x_0)
=\mathcal N\!\left(\alpha(t)x_0,\sigma(t)^2I\right).
$$

Given a clean-data predictor
$\hat x_0=D_\theta(x_t,t)\approx\mathbb E[X_0\mid X_t=x_t]$, Tweedie's
identity converts it into a score by using the **accumulated training-noise
variance**:

$$
\boxed{\;
\nabla_{x_t}\log q_t(x_t)
\approx\frac{\alpha(t)\hat x_0-x_t}{\sigma(t)^2}.
\;}
$$

The probability-flow ODE and reverse SDE instead use the **instantaneous
variance-injection rate** $g(t)^2$.  These are distinct quantities:

| Quantity | Role |
| --- | --- |
| $\sigma(t)^2$ | Conditional variance in $q_{0\to t}(x_t\mid x_0)$; denominator in the $\hat x_0$-to-score conversion |
| $g(t)^2$ | Instantaneous SDE variance rate; coefficient in the reverse drift and stochastic noise |
| $\sigma(\hat t)^2-\sigma(t_{\rm curr})^2$ for VE | Exact finite churn-transition variance |

For a VE schedule, substituting the predicted-clean score into the
probability-flow ODE gives

$$
dX_t=
-\frac12g(t)^2
\frac{\hat x_0-X_t}{\sigma(t)^2}\,dt,
\qquad dt<0.
$$

Thus $\sigma(t)^2$ is not interchangeable with $g(t)^2$: the former is
already accumulated noise, while the latter is its rate of accumulation.

### Relation to native EDM

For a VE schedule, the exact kernel reduces to the native additive VE
transition.  The constructions are therefore the same operator pattern:
exactly re-noise, then deterministically denoise.  They are nevertheless not
the same finite-step method in general.

Native EDM chooses $\hat\sigma=(1+\gamma)\sigma$ and takes its deterministic
step in the $\sigma$ coordinate.  The time-clock sampler instead chooses
$\hat t=t+\kappa h$ and takes its deterministic step in $t$.  These select the
same reheated level only for a specially matched schedule and parameters; a
nonlinear $\sigma(t)$ generally makes them differ.  Consequently their
finite-step transports and truncation errors differ as well.  They become
related through the same underlying VE probability-flow path as steps are
refined, but one should not treat `churn` as a native EDM fractional
sigma-scale increase.

The repository implementation uses an explicit probability-flow step for the
deterministic leg.  It is intentionally a time-domain operator splitting,
not an implementation of native EDM's sigma-clock Heun integrator.

## 3. General reverse-diffusion operator splitting

The preceding samplers are instances of a more general construction.  Let

$$
K_{t\to s}(x,dy),\qquad s>t,
$$

be an exact forward transition kernel: pushing $q_t$ through it gives $q_s$.
Let $\Phi_{s\to r}$ be a deterministic reverse transport for $r<s$ whose
pushforward maps $q_s$ to $q_r$.  One split reverse step from $t$ to $r$ is

$$
X^\star\sim K_{t\to s}(X,\cdot),\qquad
X_{\rm new}=\Phi_{s\to r}(X^\star),
\qquad r<t<s.
$$

The ordering matters: the kernel always moves forward to a noisier time,
then the map moves backward to a less noisy time.  Setting $s=t$ removes the
stochastic half and leaves ordinary deterministic probability-flow transport.

Under the stated assumptions, marginal preservation is immediate:

$$
X\sim q_t
\;\Longrightarrow\;
X^\star\sim q_s
\;\Longrightarrow\;
X_{\rm new}\sim q_r.
$$

This argument requires an actual forward Markov kernel with the claimed
marginal-mapping property and a deterministic map that is the exact
probability-flow transport (or another map with the same pushforward
property).  In practice, $K$ can be exact while $\Phi$ is numerically
approximated; then only the first implication is exact, and the second has
the ODE solver's discretization error.  The usual regularity requirements for
the probability-flow ODE also apply, including a well-defined score and a
well-posed flow over the selected interval.

Native VE/EDM is recovered by taking $t=\sigma_i$,
$s=\hat\sigma_i$, $r=\sigma_{i+1}$, letting $K$ be additive Gaussian
noising, and choosing $\Phi$ as an Euler or Heun probability-flow update.
For the repository schedules, take

$$
t=t_{\rm curr},\qquad s=\hat t,\qquad r=t_{\rm next},
$$

let $K$ be `Schedule.transition`, and use a probability-flow update as
$\Phi$.  The latter kernel covers VP, linear, and VE schedules through the
same $\alpha$/$\sigma$ formula.

This is an **operator splitting**, not Euler--Maruyama.  Euler--Maruyama adds
a drift increment and a Brownian increment from one SDE at the same time and
over one signed time interval.  Here the stochastic operation is a complete,
finite-gap forward transition and the deterministic operation is a separate
reverse transport over a different interval.  Replacing the exact kernel by
an Euler--Maruyama forward noise increment discards the finite-gap
marginal-preservation property that motivates the split.

## 4. Continuous FKC guidance on the time-clock split

This section extends the unsteered split with the continuous
Feynman--Kac-Corrector (FKC) guidance used by
`steered_reverse_churn_sampling`. Let the tilted target be

$$
\pi_t(x)\propto q_t(x)\exp[-\beta(t)U(x)].
$$

For a base reverse-grid step of magnitude

$$
h=t_{\rm curr}-t_{\rm next}>0,
$$

churn $\kappa$ has effective reverse-SDE diffusion
$g_\kappa(t)^2=\kappa g(t)^2$. The FKC guidance coefficient over that
**base** interval is

$$
a(t)=\frac{\beta(t)\kappa g(t)^2}{2},
\qquad
\Delta X_{\rm guide}=-a(\hat t)\,h\,\nabla U.
$$

This is a spatial displacement for the original interval
$[t_{\rm next},t_{\rm curr}]$, not yet the velocity passed to the deterministic
split leg. That leg runs for the longer interval

$$
\Delta t_{\rm transport}=t_{\rm next}-\hat t=-(1+\kappa)h.
$$

Therefore, the guidance field added to the probability-flow velocity is

$$
\boxed{\;
u_{\rm split}(x,\hat t)
=\frac{\beta(\hat t)\kappa g(\hat t)^2}
       {2(1+\kappa)}\nabla U(x).
\;}
$$

### The guidance factor does not shorten transport

The division by $1+\kappa$ scales only the additional guidance **velocity**.
It does not replace the deterministic transport interval with $-h$. The
sampler still evaluates its complete update over the full negative interval:

$$
\begin{aligned}
X_{\rm next}
&=\hat X+
\left[
v(\hat X,\hat t)
+\frac{a(\hat t)}{1+\kappa}\nabla U(\hat X)
\right]\bigl[-(1+\kappa)h\bigr] \\
&=
\underbrace{\hat X+v(\hat X,\hat t)\bigl[-(1+\kappa)h\bigr]}
_{\text{probability-flow transport from }\hat t\text{ to }t_{\rm next}}
\;-\;
\underbrace{a(\hat t)h\nabla U(\hat X)}
_{\text{one-base-step FKC guidance displacement}}.
\end{aligned}
$$

The probability-flow part is not scaled down: it runs over the whole interval
needed to undo the reheat and reach the marginal at $t_{\rm next}$. Only the
additional guidance field is converted from a base-step displacement into a
velocity on the longer split leg. Omitting the division leaves
probability-flow transport unchanged but applies $(1+\kappa)$ times the
intended guidance displacement; at $\kappa=1$ it doubles it.

The corresponding continuous FKC log-weight **rate** is evaluated at the
reheated pair $(\hat X,\hat t)$. The solver multiplies it by the **base**
interval magnitude $h$, not $-\Delta t_{\rm transport}$; see
[`edm_fkc_steering.md`](edm_fkc_steering.md) for the full weight derivation.

### Why the base velocity and guidance have different durations

It can look inconsistent that probability-flow velocity $v$ is integrated
over $(1+\kappa)h$, while the FKC guidance displacement is associated with
only $h$. The distinction is that the churn cycle has two different jobs:

1. The artificial reheat adds noise over $\kappa h$. The base
   probability-flow velocity must run over that extra interval to undo this
   artificial operation, as well as over the original reverse step. Thus it
   transports all the way from $\hat t$ through $t_{\rm curr}$ to
   $t_{\rm next}$.
2. The guided effective reverse SDE advances its intended physical step only
   from $t_{\rm curr}$ to $t_{\rm next}$, whose duration is $h$. The guide is
   tied to that effective SDE; it is not needed merely to reverse the
   artificial reheat.

An idealized Euler implementation would express this as two deterministic
substeps. Starting from the reheated state $\hat X$, it would first undo the
artificial reheat without guidance:

$$
X_{\rm mid}
=\hat X+v(\hat X,\hat t)(t_{\rm curr}-\hat t).
$$

Because $t_{\rm curr}-\hat t=-\kappa h$, this is unguided
probability-flow transport from $\hat t$ to $t_{\rm curr}$. It would then
take the actual guided base reverse step:

$$
\begin{aligned}
X_{\rm next}
=X_{\rm mid}
&+\left[
v(X_{\rm mid},t_{\rm curr})
+\frac{\beta(t_{\rm curr})\kappa g(t_{\rm curr})^2}{2}
\nabla U(X_{\rm mid})
\right](t_{\rm next}-t_{\rm curr}).
\end{aligned}
$$

Since $t_{\rm next}-t_{\rm curr}=-h$, this second move has the intended
guidance displacement

$$
-\frac{\beta(t_{\rm curr})\kappa g(t_{\rm curr})^2}{2}
h\nabla U(X_{\rm mid}).
$$

Higher-order versions would replace each displayed Euler update with a
probability-flow ODE solve or Heun step over its respective interval.

The single-Euler-leg implementation combines them. It runs base velocity
$v$ for the complete $(1+\kappa)h$ interval, while distributing the guide's
one-base-step displacement uniformly over the same longer leg. This
distribution is exactly the factor $1/(1+\kappa)$ in $u_{\rm split}$.

One may instead choose to keep the full guidance velocity active throughout
the entire $\hat t\to t_{\rm next}$ transport. That is a valid *different*
guided proposal, but it applies $(1+\kappa)$ times the FKC guidance
displacement used here. The continuous-FKC weight described above would then
no longer be its matching correction; a different compensating weight is
required.

Other steering constructions and particle-reweighting details are outside this
document's scope; the first three sections describe the unsteered reverse
samplers.
