# Feynman-Kac Steering on the Churn Sampler

**Status:** Design note + reference note (merged)

**Related documents:** [`churn_sampler.md`](churn_sampler.md), [`fkc_steering.md`](fkc_steering.md), [`schedule.md`](schedule.md)

This document states the complete algorithm obtained by running Feynman-Kac-Corrector (FKC)
steering on top of a churn sampler whose step is (a) an *exact* noising transition to a
higher noise level followed by (b) a reverse probability-flow ODE step. It is written
entirely in terms of the tilt **potential** $\rho_t(x)$, because that — not the reward, not
its gradient, not its Laplacian — is the only object the correction needs. The implementation
of record is `steered_reverse_churn_sampling` in `torchGMM/sampling.py`.

The headline is that the finite-step churn sampler should be treated as a **discrete proposal
kernel**, not as an SDE discretisation. Proposition D.6 in
[`fkc_steering.md`](fkc_steering.md) is a continuous-time generator identity and cannot be
inserted unchanged into the reordered churn step. If one complete churn step maps the base
marginal $q_{t_i}$ to $q_{t_{i+1}}$, its exact incremental FK weight is just

$$
\boxed{\;\Delta\log w_i = \rho_{t_{i+1}}(x_{i+1})-\rho_{t_i}(x_i),\;}
$$

with no score in the weight at all. The score is still needed by the probability-flow ODE,
and it must be evaluated at the *reheated* state and *reheated* time (§3).

Two results here are not in the original design note and are derived below: §6 (the exact
weight when the deterministic half is *guided*) and §7 (the $\gamma$-invariance of the
continuous-time Proposition D.6 weight under the churn family of effective reverse SDEs).

---

## 1. Notation

**Time.** The convention throughout the repository is $t=0$ for data and $t=1$ for noise:
**increasing $t$ means noisier.** Reverse sampling therefore runs from $1$ towards $0$.

**Grid and step size.** A strictly decreasing grid in $[0,1]$,

$$t_0 > t_1 > \dots > t_N, \qquad dt_i = t_i - t_{i+1} > 0, \qquad t_{i+1} = t_i - dt_i .$$

$dt_i$ is a **positive magnitude** — a step size, never a direction. Direction is carried by
the sign of the increment attached to it, and in this document one may always read a move off
that sign: $t\to t-dt$ is a step towards data, $t\to t+\gamma\,dt$ is a step towards noise.

> **Note on the code's opposite sign.** `torchGMM/sampling.py` uses a *signed*
> `dt = t_next - t_curr`, which is **negative** on a reverse grid, and computes
> `t_hat = t_curr + churn * dt.abs()` and `span = t_next - t_hat`. That is the same sampler
> written with the opposite sign convention; the translation is $dt_{\text{doc}} =
> |dt_{\text{code}}|$, and every increment quoted below (`span`, the reheat, the transport)
> is numerically identical in the two. The `dt` that reaches a `weight_update` callable is
> the code's signed one.

**Schedule.** A schedule supplies the interpolation coefficients of
$X_t = \alpha_t X_0 + \sigma_t\varepsilon$ and hence the forward SDE
$dX_t = f(X_t,t)\,dt + g(t)\,dW_t$ with

$$
f(x,t)=\frac{\dot\alpha_t}{\alpha_t}\,x,
\qquad
g(t)^2 = 2\Bigl(\dot\sigma_t\sigma_t - \frac{\dot\alpha_t}{\alpha_t}\sigma_t^2\Bigr).
$$

Here $\sigma_t$ is the **marginal noise standard deviation** and $g(t)$ is the **diffusion
coefficient**; they are different objects. See the warning in §1.1.

**Score and velocity.** Write

$$s_t(x) = \nabla_x \log q_t(x)$$

for the score of the base marginal $q_t$. **The score is the only object in this algorithm
that costs a model evaluation.** Everything else — $\alpha_t$, $\sigma_t$, $f$, $g$, the
transition kernel, the resampling arithmetic — is closed form and free. The probability-flow
velocity is

$$\boxed{\;v_t(x) = f(x,t) - \tfrac12 g(t)^2\, s_t(x).\;}$$

It is exactly one score evaluation, and integrating $\dot x = v_t(x)$ over a negative
increment transports $q_t$ to $q_{t'}$ for $t' < t$.

**Exact transition kernel.** For $s > t$ — a **positive** increment $s-t>0$, i.e. $s$ noisier;
the radicand below goes negative in the other direction, so the kernel runs towards noise
only — [`schedule.md`](schedule.md) derives

$$
\boxed{\;
K_{t\to s}:\quad
X_s = \frac{\alpha_s}{\alpha_t}X_t + \sqrt{\,\sigma_s^2 - \frac{\alpha_s^2}{\alpha_t^2}\sigma_t^2\,}\;\eta,
\qquad \eta\sim\mathcal N(0,I).
\;}
$$

This maps $q_t$ onto $q_s$ **exactly**, for any gap $s-t$, with no discretisation error, and
it contains **no score**: it is a single closed-form Gaussian draw. In the repository it is
`Schedule.transition`. (For the VP special case $\alpha_t^2+\sigma_t^2=1$ the radicand
simplifies to $1-\alpha_{t+\gamma dt}^2/\alpha_t^2$, which is the form quoted in
[`churn_sampler.md`](churn_sampler.md); it gives **zero** injected noise on a VE or Karras
schedule, so the general form above is the one to implement.)

**Churn.** Given a churn strength $\gamma \ge 0$, the churn half re-noises from $t_i$ to

$$t_i + \gamma\,dt_i \qquad (\text{clamped at } 1),$$

a **positive** increment and hence a move towards noise. **There is no separate symbol for the
reheated time**: it is written out as $t_i+\gamma\,dt_i$ everywhere below, so that the
direction of every move can be read off its sign without a lookup. The churned *state* does
get its own symbol, $\hat x_i$ — it is a new random draw, not a relabelled time.

The transport span of the deterministic half is the increment that lands on the next grid
point,

$$\mathrm{span}_i = t_{i+1} - (t_i+\gamma\,dt_i) = -(1+\gamma)\,dt_i,$$

a **negative** increment and hence a move towards data. The arithmetic that fixes the factor
$(1+\gamma)$ is one line, and it is the whole point of the convention:

$$(t_i + \gamma\,dt_i) \;-\; (1+\gamma)\,dt_i \;=\; t_i - dt_i \;=\; t_{i+1}.$$

The state must travel from $t_i+\gamma\,dt_i$ all the way down to $t_{i+1}$, undoing the churn
as well as advancing one grid step. $\gamma = 0$ is the pure probability-flow ODE (no churn;
the implementation then skips $K$ entirely), $\gamma = 1$ re-noises one full grid step back and
transports two steps down, and $\gamma > 1$ over-churns. The clamp at $1$ keeps the reheated
time inside the schedule near $t=1$; when it bites, the sampler lands below $t_i+\gamma\,dt_i$
and the span shrinks to match, since $\mathrm{span}_i = t_{i+1} - (\text{reheated time})$ is
the definition and $-(1+\gamma)\,dt_i$ only its unclamped value.

**Potential.** The tilt exponent is

$$\boxed{\;\rho_t(x) = \beta(t)\, r(x,t).\;}$$

$\beta$ is the annealing/temperature schedule and $r$ is the reward, which may be evaluated
on the noisy state directly, $r(x)$, or on a denoised estimate, $r(D(x,t))$. The algorithm
below never needs $\rho$ decomposed into $\beta$ and $r$; it consumes $\rho_t(x)$ as a single
callable returning one scalar per particle. This is the `potential` argument of
`steered_reverse_churn_sampling`.

### 1.1 A notation clash to be aware of

[`fkc_steering.md`](fkc_steering.md), which transcribes the Proposition D.6 proof of Skreta
et al., writes $\sigma_t$ for the **diffusion coefficient** of the SDE — the object called
$g(t)$ everywhere else in this repository. In that file $\tfrac{\sigma_t^2}{2}\Delta p$ is a
diffusion term, and the "magic" constant is $a = \beta_t\sigma_t^2/2$ meaning
$\beta_t g(t)^2/2$.

In *this* document, in [`schedule.md`](schedule.md), in [`churn_sampler.md`](churn_sampler.md)
and in the code, $\sigma_t$ is the **marginal noise standard deviation** of
$X_t=\alpha_tX_0+\sigma_t\varepsilon$, and the diffusion coefficient is $g(t)$. When
transcribing formulas out of `fkc_steering.md` — as §7 does — every $\sigma_t^2$ there must be
read as $g(t)^2$ here. The two coincide only for a VE schedule with $\alpha\equiv1$ and
$2\ell = 1$, which is not a case anyone runs.

---

## 2. The target

The steered sampler targets the reward-tilted family

$$\boxed{\;\pi_t(x) = \frac{q_t(x)\,e^{\rho_t(x)}}{Z_t},\qquad Z_t=\int q_t(x)e^{\rho_t(x)}\,dx,\;}$$

**at every $t$ on the grid**, not merely at the endpoint $t_N$. This is the defining choice of
the FKC construction and the reason a weight correction exists at all.

### 2.1 Proposal propagation versus retargeting

Applying a transition does not mechanically change a particle's weight. If particles
$(x^n,w^n)$ are propagated through a Markov kernel, they initially retain the same $w^n$.
Every weight update below is a **retargeting correction**: it converts the pushforward of the
old tilted distribution into the desired tilted distribution at the new time.

If the desired target were instead defined as "the old tilted distribution propagated through
the transition", no weight update would be necessary at all. The correction exists because the
bridge $\{\pi_t\}_t$ is *prescribed* at every time, and that is in general different from
propagating $\pi_{t_0}$ under the base dynamics.

Consequently the correctness criterion is a statement about the *weighted* particle cloud at
every recorded time, and the tests assert exactly that: `_weighted_wasserstein1` against
`_tilted_density` at 49–56 intermediate slots, not just at $t_N$. An unweighted empirical
distribution between resamples is *expected* to be wrong.

---

## 3. One step as two operators

A churn step is the composition of two complete operators,

$$
x_i
\;\xrightarrow{\;K_{t_i\to t_i+\gamma dt_i}\;}\;
\hat x_i
\;\xrightarrow{\;\Phi_{t_i+\gamma dt_i\to t_{i+1}}\;}\;
x_{i+1},
\qquad
M_i = K_{t_i\to t_i+\gamma dt_i}\,\Phi_{t_i+\gamma dt_i\to t_{i+1}} .
$$

- $K$ is the exact forward noising kernel of §1. It preserves the base marginal family
  **exactly**: $q_{t_i}K_{t_i\to t_i+\gamma dt_i} = q_{t_i+\gamma dt_i}$, for any gap.
- $\Phi$ is probability-flow ODE transport. It preserves the base marginal family **up to
  solver truncation**: $q_{t_i+\gamma dt_i}\Phi_{t_i+\gamma dt_i\to t_{i+1}} = q_{t_{i+1}}$ in the exact-flow
  limit.

Hence

$$\boxed{\;q_{t_i}\,M_i = q_{t_{i+1}}\;}$$

and $M_i$ is a valid *base* proposal for a discrete Feynman-Kac model. This is the whole
correctness argument, and it is a splitting argument, not an SDE discretisation: the two
halves are operators, not the two terms of one SDE added at the same $t$ over the same $dt$.

### 3.1 The score follows the state

**The velocity is evaluated at the reheated pair.** After $K$, the state $\hat x_i$ is
distributed according to $q_{t_i+\gamma dt_i}$, not $q_{t_i}$, so the deterministic half must call
$v_{t_i+\gamma dt_i}(\hat x_i)$ — i.e. `velocity(x_hat, t_hat)`, not `velocity(x_hat, t_curr)`.
Using the score at $t_i$ is the wrong score for that state; at $\gamma=1$ the mismatch is a
full grid step wide.

The rule in full, through the split:

1. The exact forward transition $K_{t_i\to t_i+\gamma dt_i}$ requires **no score**.
2. The first PF-ODE evaluation uses $s_{t_i+\gamma dt_i}(\hat x_i)$.
3. A Heun corrector, if used, uses $s_{t_{i+1}}(x^{\mathrm{pred}}_{i+1})$.

The same rule applies to **denoised rewards**: if $r_t(x) = r(D(x,t))$, recompute
$D(\hat x_i,\,t_i+\gamma dt_i)$ after the churn rather than reusing a denoiser or score evaluated
before reheating.

---

## 4. The potential-only weight (unguided)

Because $M_i$ already maps $q_{t_i}$ onto $q_{t_{i+1}}$, the *entire* importance correction is
the endpoint difference of the potential. Carrying the weights unchanged through $M_i$
produces the unnormalised measure $\int q_{t_i}(x)e^{\rho_{t_i}(x)}M_i(x,dy)$; applying the
edge potential $G_i(x,y)=\exp(\rho_{t_{i+1}}(y)-\rho_{t_i}(x))$ gives

$$
\int q_{t_i}(x)e^{\rho_{t_i}(x)}M_i(x,dy)\,G_i(x,y)
= e^{\rho_{t_{i+1}}(y)}\int q_{t_i}(x)M_i(x,dy)
= q_{t_{i+1}}(y)\,e^{\rho_{t_{i+1}}(y)},
$$

which is $\pi_{t_{i+1}}$ up to the normaliser, and normalisers cancel when particle weights
are normalised. So

$$
\boxed{\;\Delta\log w_i \;=\; \rho_{t_{i+1}}(x_{i+1}) - \rho_{t_i}(x_i).\;}
$$

No Jacobian determinant appears for the deterministic half, because the Jacobian is already
exactly the thing that transforms $q_{t_i+\gamma dt_i}$ into $q_{t_{i+1}}$ under $\Phi$; it cancels
between base proposal and base target. (If $\Phi$ does *not* exactly map $q_{t_i+\gamma dt_i}$ to
$q_{t_{i+1}}$, the full deterministic correction carries
$\log q_{t_{i+1}}(y) - \log q_{t_i+\gamma dt_i}(\hat x_i) + \log|\det D\Phi|$ as well — see §11.2. For a numerical solver those terms are unavailable, so
the practical sampler inherits the base ODE error rather than correcting it.)

### 4.1 What is *not* needed

This is the practical payoff and is worth stating explicitly. The increment above requires

- no $\dot\beta_t$,
- no $\partial_t r$,
- no reward Laplacian $\Delta r$,
- no score-alignment inner product $\langle\beta\nabla r, \tfrac{g^2}{2}s_t-f_t\rangle$,
- **no reward gradient $\nabla_x r$ at all.**

Every one of these is required by the continuous-time weight of Proposition D.6, Eq. (276).
The consequence for a reward defined on a denoised state, $\rho_t(x)=\beta(t)r(D(x,t))$, is
structural: $D$ is evaluated but **never backpropagated through**. The steering plumbing
collapses to a single scalar-valued callable.

### 4.2 Telescoping, and why $\log w$ stays bounded

Accumulated between two resamples at steps $j$ and $k$, the increments telescope:

$$
\log w = \sum_{i=j}^{k-1}\bigl[\rho_{t_{i+1}}(x_{i+1})-\rho_{t_i}(x_i)\bigr]
= \rho_{t_k}(x_k) - \rho_{t_j}(x_j).
$$

The intermediate potentials cancel identically. Therefore

$$\bigl|\log w\bigr| \le \operatorname{range}(\rho)$$

between resamples, *no matter how many steps elapse*. Contrast the Euler-Maruyama FKC route,
where $\log w$ is a discretised stochastic integral $\int g_t\,dt$ whose variance grows with
the number of steps. This is why the churn-FKC sampler tolerates very long resampling
intervals: the measured mean $\mathrm{ESS}/N$ on the Karras denoised-reward setup is $0.997$
even with the resampling interval set past the total step count, i.e. with a single final
resample. There is essentially no depletion to repair.

---

## 5. The split at the reheated state

Because a step is two operators, the increment splits at the reheated state. Through the
**stochastic** half,

$$
\boxed{\;\Delta\log w_i^{\mathrm{churn}} = \rho_{t_i+\gamma dt_i}(\hat x_i) - \rho_{t_i}(x_i)\;}
$$

which is **exact** because $q_{t_i}K_{t_i\to t_i+\gamma dt_i} = q_{t_i+\gamma dt_i}$ holds with no
approximation whatsoever — the stochastic forward kernel introduces no additional likelihood
ratio precisely because it *is* the base kernel between those two marginals. Through the
**deterministic** half, by the same argument applied to $\Phi_{\#}q_{t_i+\gamma dt_i}=q_{t_{i+1}}$,

$$
\boxed{\;\Delta\log w_i^{\mathrm{ODE}} = \rho_{t_{i+1}}(x_{i+1}) - \rho_{t_i+\gamma dt_i}(\hat x_i).\;}
$$

Adding them,

$$
\Delta\log w_i^{\mathrm{churn}} + \Delta\log w_i^{\mathrm{ODE}}
= \rho_{t_{i+1}}(x_{i+1}) - \rho_{t_i}(x_i),
$$

which is §4 verbatim: $\rho_{t_i+\gamma dt_i}(\hat x_i)$ cancels. The split is therefore never
required for correctness — the two halves are the *same* endpoint correction, applied once
per operator rather than once per step.

**Why it exists.** The split creates an intermediate state at which the ESS can be tested and
a resample can fire — at the *reheated* noise level, where the freshly injected noise has just
maximised particle diversity, so a gather there duplicates ancestors with less loss of
diversity than one at $t_{i+1}$. The implementation does **not** take this path: it weights and
resamples once per integration step, after both halves.

**What it costs.** $\rho_{t_i+\gamma dt_i}(\hat x_i)$ is a state that is otherwise never scored, so
the split evaluates the potential exactly twice per step instead of once: measured
$2.00$ evaluations/step against $1.00$. Measured benefit on the Karras denoised-reward setup:
mean $\mathrm{ESS}/N$ moved from $0.997$ to $0.998$, and terminal $W_1$ differences were
inside the seed-to-seed spread. Given §4.2 that is the expected outcome — the telescoping
already bounds $\log w$, so there is no depletion for the extra selection opportunity to
repair. The split is worth revisiting only in a regime where ESS genuinely collapses.

---

## 6. The guided deterministic half

*This is the result that is not in the original design note; it is derived here.*

Suppose the deterministic half integrates $v_t + u$ instead of $v_t$, for an **arbitrary**
guidance field $u(x,t)$. The base flow is no longer marginal-preserving, so §4's cancellation
no longer applies. The exact repair is a single additional scalar.

### 6.1 Derivation

Work in continuous time along the transport, from $t_i+\gamma\,dt_i$ down to $t_{i+1}$. Every
$\partial_t$, every dot and every velocity is referred to the schedule time $t$ of §1, so the
trajectory obeys $\dot x = v_t + u$ while $t$ *decreases*, and every integral below runs from
the lower limit $t_i+\gamma\,dt_i$ to the upper limit $t_{i+1} < t_i+\gamma\,dt_i$ — that is, over the negative
increment $\mathrm{span}_i$. For a density evolving under a deterministic velocity field $w$
with no diffusion, the log form of the continuity equation is

$$\partial_t \log p = -\nabla\!\cdot\! w - \langle \nabla\log p,\, w\rangle .$$

Apply it to the base marginal with $w = v_t$:

$$\partial_t\log q_t = -\nabla\!\cdot\! v_t - \langle s_t, v_t\rangle . \tag{6.1}$$

The target is $\log\pi_t = \log q_t + \rho_t - \log Z_t$, and we want to write its evolution as
*transport under the guided field* $v_t+u$ plus a Feynman-Kac reweighting term
$g_t - \mathbb E_{\pi_t}[g_t]$:

$$\partial_t \pi_t = -\nabla\!\cdot\!\bigl(\pi_t (v_t+u)\bigr) + \pi_t\bigl(g_t - \mathbb E_{\pi_t}[g_t]\bigr).$$

In log form the transport block is $-\nabla\!\cdot\!(v_t+u) - \langle\nabla\log\pi_t, v_t+u\rangle$,
so $g_t$ is whatever is left over:

$$
g_t = \underbrace{\partial_t\log q_t + \partial_t\rho_t}_{\partial_t\log\pi_t\,+\,\partial_t\log Z_t}
\;+\;\nabla\!\cdot\!(v_t+u) + \langle\nabla\log\pi_t,\, v_t+u\rangle ,
$$

the $\partial_t\log Z_t$ being exactly the $-\mathbb E_{\pi_t}[g_t]$ that normalisation
supplies for free. Substituting (6.1) kills $\nabla\!\cdot\! v_t$ and leaves

$$g_t = \partial_t\rho_t + \nabla\!\cdot\! u + \langle\nabla\log\pi_t,\,v_t+u\rangle - \langle s_t, v_t\rangle .$$

Now use $\nabla\log\pi_t = s_t + \nabla\rho_t$ (the normaliser is constant in $x$):

$$
\langle s_t+\nabla\rho_t,\,v_t+u\rangle - \langle s_t,v_t\rangle
= \langle\nabla\rho_t, v_t\rangle + \langle\nabla\rho_t, u\rangle + \langle s_t, u\rangle,
$$

so

$$
g_t = \underbrace{\partial_t\rho_t + \langle\nabla\rho_t,\, v_t + u\rangle}_{\textstyle \frac{d}{dt}\rho_t(x_t)\ \text{along the guided trajectory}}
\;+\;\nabla\!\cdot\! u + \langle s_t, u\rangle . \tag{6.2}$$

The first bracket is the **material derivative of $\rho$ along the trajectory the sampler
actually follows**, because that trajectory obeys $\dot x_t = v_t + u$. Integrating (6.2) over
the span therefore collapses it to an endpoint difference again, and we obtain

$$
\boxed{\;
\Delta\log w_i^{\mathrm{ODE}}
= \rho_{t_{i+1}}(x_{i+1}) - \rho_{t_i+\gamma dt_i}(\hat x_i)
+ \int_{t_i+\gamma dt_i}^{t_{i+1}}\Bigl[\nabla\!\cdot\! u + \langle s_t, u\rangle\Bigr]dt .
\;}
$$

(An equivalent route to (6.2), matching the presentation of Prop. D.6 Step 4: write the FK
weight of the *unguided* flow with the extra divergence term added back,
$g_t = \partial_t\rho + \langle\nabla\rho, v_t\rangle + \nabla\!\cdot\! u + \langle\nabla\log p_t, u\rangle$;
rewrite the first two terms as the guided material derivative minus $\langle\nabla\rho,u\rangle$;
then $\langle\nabla\log p_t,u\rangle = \langle s_t,u\rangle + \langle\nabla\rho,u\rangle$
absorbs the deficit. Same answer.)

### 6.2 Reading the compensation term

The integrand is the **compressibility of the guidance field against the base marginal**:

$$
\nabla\!\cdot\! u + \langle s_t, u\rangle
= \frac{1}{q_t}\Bigl[q_t\,\nabla\!\cdot\! u + \langle\nabla q_t, u\rangle\Bigr]
= \frac{1}{q_t}\,\nabla\!\cdot\!\bigl(q_t\,u\bigr).
$$

It measures how much probability mass the extra field $u$ pumps into or out of a neighbourhood
*relative to $q_t$*. It vanishes identically for $u = 0$, recovering §5 exactly.

Three properties matter:

1. **Exact for any $u$ and any span.** Nothing in the derivation constrains $u$ to be a
   gradient, to be small, or to be related to $\rho$. $u$ is a free knob trading weight
   variance against drift, in precisely the sense that $a$ is free in Proposition D.6
   Step 4 — and it does *not* have to equal the D.6 "magic" constant.
2. **The Laplacian does not cancel.** This is the sharp contrast with Prop. D.6. There, the
   choice $a=\beta g^2/2$ is singled out because it makes the drift's $a\Delta r$ cancel the
   weight's $-\beta\tfrac{g^2}{2}\Delta r$, and that second term is *supplied by the diffusion
   term of the SDE*. A deterministic half has no diffusion term, hence no $-\beta\tfrac{g^2}{2}\Delta r$
   to cancel against, and $\nabla\!\cdot\! u$ survives in full. For the canonical choice
   $u = c\,\tfrac{g^2}{2}\nabla\rho$ this is $c\,\tfrac{g^2}{2}\Delta\rho$, which must be
   computed and applied. There is no value of $c$ that deletes it.
3. **$\nabla\!\cdot\! u$ must be analytic.** The caller must supply the scalar
   $\nabla\!\cdot\! u + \langle s_t,u\rangle$ alongside the field itself, because
   $\nabla\!\cdot\! u$ is closed-form for the guidance fields anyone actually uses (for
   $u = c\tfrac{g^2}{2}\nabla\rho$ with a Gaussian reward, $\Delta\rho = -\beta(t)D/\varsigma^2$
   is a constant), and a numerical divergence estimator would cost $D$ extra passes and defeat
   the purpose. A Hutchinson trace estimator does not rescue it either: the estimate enters an
   exponent, so $\mathbb E[e^X]\ne e^{\mathbb E[X]}$ and the weights come out **biased**, not
   merely noisy. **This route is for analytic rewards.** For a learned reward, prefer §4 (no
   gradients at all) or §7 (gradient, but no Laplacian).

In the implementation the integral is evaluated by a single left-endpoint quadrature at
$t_i+\gamma\,dt_i$, i.e. $\Delta\log w \mathrel{+}= \mathrm{corr}\cdot\mathrm{span}_i$ with
$\mathrm{corr} = \nabla\!\cdot\! u(\hat x_i,\,t_i+\gamma dt_i) + \langle s_{t_i+\gamma dt_i}(\hat x_i),
u(\hat x_i,\,t_i+\gamma dt_i)\rangle$ and $\mathrm{span}_i = t_{i+1}-(t_i+\gamma\,dt_i) < 0$ — the same negative
increment the transport step uses, so a compensation that inflates mass ($\mathrm{corr}>0$)
lowers the weight. This is first order in the span, matching the order of the Euler transport
step it accompanies; it is not an extra source of error relative to the flow itself.

### 6.3 The term is load-bearing

Dropping the compensation while keeping the guidance still produces high-reward samples — it
just targets the wrong distribution, which is exactly the failure a reward-only diagnostic
misses. Measured over 49 intermediate time slots with $u = c\,\tfrac{g^2}{2}\nabla\rho$, mean
$W_1$ against the analytic tilted marginal degrades by

| $c$ | with compensation | without | factor |
|---|---|---|---|
| $0.5$ | $0.023$ | $0.172$ | $7.5\times$ |
| $1.0$ | $0.035$ | $0.319$ | $9\times$ |

### 6.4 The alternative: twist the churn kernel instead

Guiding the deterministic half is not the only way to move reward information into the
dynamics. The *stochastic* half can be twisted instead, and that variant has a
straightforward likelihood ratio because both proposals are Gaussians with known densities.

Let the base churn transition be $K_i(\hat x\mid x_i)=\mathcal N(\hat x; m_i(x_i), V_i)$ and
sample instead from a reward-biased $\widetilde K_i(\hat x\mid x_i)$. The corrected full-step
weight is

$$
\boxed{\;
\Delta\log w_i
= \rho_{t_{i+1}}(x_{i+1})-\rho_{t_i}(x_i)
+ \log K_i(\hat x_i\mid x_i) - \log\widetilde K_i(\hat x_i\mid x_i).
\;}
$$

For a local linearisation of the reheated potential, define
$a_i=\nabla_{\hat x}\rho_{t_i+\gamma dt_i}(\hat x)\bigl|_{\hat x=m_i(x_i)}$ and use the mean-shifted
Gaussian $\widetilde K_i = \mathcal N(m_i+V_ia_i,\;V_i)$, whose log-density correction is
analytic:

$$
\log K_i-\log\widetilde K_i = -a_i^\top(\hat x_i-m_i)+\tfrac12 a_i^\top V_i a_i .
$$

This is the discrete analogue of moving part of the FK correction into a guidance drift, and
it was the route the original design note recommended, on the grounds that two different
deterministic flows define mutually singular transition kernels unless their inverse maps and
Jacobian determinants are tracked. §6.1 dissolves that objection — the Jacobian nearly cancels
against the density change (Liouville), and what survives is the analytic scalar
$\nabla\!\cdot\! u+\langle s_t,u\rangle$ — so the deterministic route is safe and is what the
repository implements. The twisted-kernel variant is **not implemented**; it remains the
better option if one ever wants to steer without any gradient of $\rho$ entering the flow.

### 6.5 The two guidance routes, written out

Both routes have the same skeleton — churn, transport, weight — and both carry the same
endpoint difference $\rho_{t_{i+1}}(x_{i+1})-\rho_{t_i}(x_i)$. Only the *extra* term differs,
because only one of the two operators has been modified. Written out step by step, with
$\mathrm{span}_i = t_{i+1}-(t_i+\gamma\,dt_i) = -(1+\gamma)\,dt_i$ throughout:

**Route 1 — guidance in the probability flow (§6.1).** The churn kernel is the base one; the
deterministic half integrates $v+u$.

$$
\textbf{(a) churn}\quad
\hat x_i = \frac{\alpha_{t_i+\gamma dt_i}}{\alpha_{t_i}}\,x_i
+ \sqrt{\,\sigma_{t_i+\gamma dt_i}^2 - \frac{\alpha_{t_i+\gamma dt_i}^2}{\alpha_{t_i}^2}\sigma_{t_i}^2\,}\;\eta_i,
\qquad \eta_i\sim\mathcal N(0,I)
$$

$$
\textbf{(b) guided transport}\quad
u_i = u(\hat x_i,\,t_i+\gamma dt_i),
\qquad
x_{i+1} = \hat x_i + \bigl[\,v_{t_i+\gamma dt_i}(\hat x_i) + u_i\,\bigr]\,\mathrm{span}_i
$$

$$
\textbf{(c) weight}\quad
\boxed{\;
\Delta\log w_i = \rho_{t_{i+1}}(x_{i+1}) - \rho_{t_i}(x_i)
\;+\;\Bigl[\nabla\!\cdot\! u(\hat x_i,\,t_i+\gamma dt_i)
+ \bigl\langle s_{t_i+\gamma dt_i}(\hat x_i),\,u_i\bigr\rangle\Bigr]\,\mathrm{span}_i
\;}
$$

with the canonical choice $u = c\,\tfrac{g^2}{2}\nabla\rho$ for any $c$. Needs $\nabla\rho$
(to build $u$) **and** $\Delta\rho$ (to build $\nabla\!\cdot\! u$).

**Route 2 — guidance in the reheating kernel (§6.4).** The flow is the base one; the Gaussian
churn proposal is mean-shifted.

$$
\textbf{(a) base moments}\quad
m_i = \frac{\alpha_{t_i+\gamma dt_i}}{\alpha_{t_i}}\,x_i,
\qquad
V_i = \Bigl(\sigma_{t_i+\gamma dt_i}^2 - \frac{\alpha_{t_i+\gamma dt_i}^2}{\alpha_{t_i}^2}\sigma_{t_i}^2\Bigr)I,
\qquad
K_i = \mathcal N(m_i,V_i)
$$

$$
\textbf{(b) twisted churn}\quad
a_i = \nabla_{\hat x}\,\rho_{t_i+\gamma dt_i}(\hat x)\big|_{\hat x=m_i},
\qquad
\widetilde K_i = \mathcal N(m_i+V_ia_i,\,V_i),
\qquad
\hat x_i = m_i + V_ia_i + V_i^{1/2}\eta_i
$$

$$
\textbf{(c) unguided transport}\quad
x_{i+1} = \hat x_i + v_{t_i+\gamma dt_i}(\hat x_i)\,\mathrm{span}_i
$$

$$
\textbf{(d) weight}\quad
\boxed{\;
\Delta\log w_i = \rho_{t_{i+1}}(x_{i+1}) - \rho_{t_i}(x_i)
\;\underbrace{-\;a_i^\top(\hat x_i-m_i) + \tfrac12 a_i^\top V_ia_i}_{\log K_i-\log\widetilde K_i}
\;}
$$

Substituting the draw from (b) collapses the correction to a closed form in the injected noise,

$$
\log K_i - \log\widetilde K_i = -\tfrac12\,a_i^\top V_ia_i - a_i^\top V_i^{1/2}\eta_i,
\qquad
\mathbb E_{\eta_i}\bigl[\log K_i-\log\widetilde K_i\bigr] = -\tfrac12\,a_i^\top V_ia_i ,
$$

which is worth reading twice: the correction is **stochastic**, so unlike Route 1 it adds
weight variance of its own. Needs $\nabla\rho$ only, evaluated once at the kernel *mean* $m_i$
— no Laplacian, no divergence.

| | Route 1: guided flow | Route 2: twisted kernel |
|---|---|---|
| modified operator | $\Phi$ (deterministic) | $K$ (stochastic) |
| extra weight term | $[\nabla\!\cdot\! u+\langle s,u\rangle]\cdot\mathrm{span}_i$ | $-a_i^\top(\hat x_i-m_i)+\tfrac12a_i^\top V_ia_i$ |
| needs $\nabla\rho$ | yes, at $\hat x_i$ | yes, at $m_i$ |
| needs $\Delta\rho$ | **yes** | no |
| exact for | any field $u$, any span | any $\widetilde K_i$ with known density |
| correction is | deterministic given $\hat x_i$ | stochastic — carries $\eta_i$ |
| implemented | yes (`TestSteeredChurnGuidedFlow`) | no |

The asymmetry is structural rather than incidental. A deterministic map has no density of its
own, so its correction can only be expressed as a divergence — hence the Laplacian. A Gaussian
proposal *does* have a density, so its correction is a ratio one can simply write down. That is
the whole reason Route 2 escapes $\Delta\rho$ and Route 1 cannot, and it is why Route 2 stays
the better choice for a reward whose Laplacian is unavailable (§6.2, property 3) despite the
extra weight variance it introduces.

---

## 7. The effective-SDE view and its $\gamma$-invariance

Everything above is a *discrete* Feynman-Kac factorisation and never approximates the churn
split by an SDE. It is nevertheless worth knowing what the split converges to, and what
Proposition D.6 says about that limit, because the answer is cleaner than one would guess.

### 7.1 The effective generator

For a grid step $dt>0$ and churn interval $\gamma\,dt$, the infinitesimal churn split converges
to a reverse diffusion, written here in the schedule time $t$ — so that its increments are
negative on a reverse pass, as everywhere else in this document — with drift callable and
diffusion coefficient

$$
\boxed{\;
b_\gamma(x,t) = f(x,t) - \frac{1+\gamma}{2}\,g(t)^2\,s_t(x),
\qquad
g_\gamma(t) = \sqrt{\gamma}\;g(t).
\;}
$$

The family interpolates: $\gamma=0$ gives $b_0 = f - \tfrac12 g^2 s = v_t$ and $g_0=0$, the
probability-flow ODE; $\gamma=1$ gives $b_1 = f - g^2 s$ and $g_1 = g$, the ordinary Anderson
reverse SDE that Proposition D.6 assumes; $\gamma>1$ over-churns. This is the standard
$\lambda$-family of reverse SDEs at $\lambda = \sqrt\gamma$.

### 7.2 Redoing Prop. D.6 Step 5 with $(b_\gamma, g_\gamma)$

Transcribe from [`fkc_steering.md`](fkc_steering.md), remembering the §1.1 clash: its
$\sigma_t^2$ is a diffusion coefficient squared, so here it becomes $g_\gamma^2 = \gamma g^2$.
There is a second thing to keep straight. D.6 runs the reverse SDE in a **reverse time**
$\bar t$ that *increases* as the sampler denoises — one step towards data moves $t$ by $-dt$
and $\bar t$ by $+dt$, for the same positive step size $dt$ — and its drift is therefore
$-b_\gamma$, not $b_\gamma$; write it $\tilde v_{\bar t}$ to keep it apart from the probability-flow
velocity $v_t$ of §1, with which it does not coincide:

$$\tilde v_{\bar t} = -b_\gamma = -f + \tfrac{1+\gamma}{2}g^2 s .$$

Everything in §7.2 is in that $\bar t$ parameterisation, so its increments are positive along
the reverse pass; §7.3 converts back. The pre-guidance weight Eq. (270) reads

$$
g^{\mathrm{FK}}_t = \Bigl\langle \beta\nabla r,\; \tilde v_{\bar t} - g_\gamma^2\,s_t - \tfrac{g_\gamma^2}{2}\beta\nabla r\Bigr\rangle
- \beta\,\tfrac{g_\gamma^2}{2}\Delta r + \dot\beta\, r,
$$

and Step 4's free drift $a\nabla r$ contributes $a\Delta r + a\langle\nabla\log p_t,\nabla r\rangle$.

**(i) D.6's first cancellation fails for $\gamma\ne1$.** In D.6 the packaged inner product
collapses because $\tilde v_{\bar t} - g^2 s_t = -f_t$ exactly. Here

$$
\tilde v_{\bar t} - g_\gamma^2 s_t
= -f + \frac{1+\gamma}{2}g^2 s - \gamma g^2 s
= -f + \frac{1-\gamma}{2}\,g^2 s ,
$$

so a residual score term $\tfrac{1-\gamma}{2}g^2 s$ survives, vanishing only at $\gamma=1$.

**(ii) The Laplacian cancellation moves.** The weight carries $-\beta\tfrac{g_\gamma^2}{2}\Delta r$
and the drift contributes $+a\Delta r$, so deleting $\Delta r$ now requires

$$\boxed{\;a = \beta\,\frac{g_\gamma^2}{2} = \beta\,\frac{\gamma\,g^2}{2}\;}$$

rather than $\beta g^2/2$. The guided drift therefore *does* depend on $\gamma$:

$$
dx = \Bigl(-f_t + \tfrac{1+\gamma}{2}g^2 s_t + \beta\,\tfrac{\gamma g^2}{2}\nabla r\Bigr)d\bar t
+ \sqrt{\gamma}\,g\,dW_{\bar t} .
$$

**(iii) The residual score terms recombine to exactly $\tfrac12$.** With that $a$, expand
$a\langle\nabla\log p_t,\nabla r\rangle$ using $\nabla\log p_t = s_t + \beta\nabla r$:

$$
a\langle\nabla\log p_t,\nabla r\rangle
= \frac{\beta\gamma g^2}{2}\langle s_t,\nabla r\rangle + \frac{\gamma g^2}{2}\beta^2\lVert\nabla r\rVert^2 .
$$

The $\beta^2\lVert\nabla r\rVert^2$ piece cancels against the
$-\tfrac{g_\gamma^2}{2}\beta\nabla r$ slot inside the packaged inner product, exactly as in
D.6. What remains are the two score terms, from (i) and from the drift compensation:

$$
\beta\,\frac{1-\gamma}{2}\,g^2\langle\nabla r, s_t\rangle
\;+\;
\beta\,\frac{\gamma}{2}\,g^2\langle\nabla r, s_t\rangle
\;=\;
\beta\,g^2\langle\nabla r, s_t\rangle\underbrace{\Bigl[\frac{1-\gamma}{2}+\frac{\gamma}{2}\Bigr]}_{=\;1/2}
\;=\;\beta\,\frac{g^2}{2}\langle\nabla r, s_t\rangle .
$$

The $\gamma$'s cancel identically. Therefore the final weight is

$$
\boxed{\;
d\log w = \Bigl[\;\dot\beta_t\, r(x_t) \;+\; \Bigl\langle \beta_t\nabla r(x_t),\; \tfrac{g(t)^2}{2}s_t(x_t) - f_t(x_t)\Bigr\rangle\;\Bigr]d\bar t
\;}
$$

— Eq. (276) of Proposition D.6, with the **base** $g^2$, **independent of $\gamma$**. (With a
time-dependent reward, $\dot\beta_t r \mapsto \partial\rho$ as in the generalisation at the
end of `fkc_steering.md`.)

All dots and $\partial$'s in this subsection are with respect to $\bar t$, so translating the box
back into the schedule time of §1 flips the annealing term but not the increment as a whole:
$\dot\beta_{\bar t} = -\dot\beta_t$ while the increment $d\bar t$ is the positive step size $dt$
itself, which is exactly what an implementation
computes — the $-\dot\beta_t\,r$ and $-\langle\beta\nabla r, f\rangle + \langle\beta\nabla r,
\tfrac{g^2}{2}s\rangle$ terms, multiplied by the positive step size.

**Interpretation.** The invariance is not a coincidence. The vector field appearing in the
weight is

$$\frac{g^2}{2}s_t - f_t = -\Bigl(f_t - \frac{g^2}{2}s_t\Bigr) = -\,v_t,$$

the negated probability-flow velocity. The PF velocity is the one member of the $\lambda$-family
that carries no stochasticity, and the entire $\gamma$-dependence of the family lives in the
score-plus-noise part that the FK weight is blind to. Churn changes *how* the sampler moves; it
does not change *which* reweighting makes the tilted bridge exact.

### 7.3 Two $\gamma$-factors in the discrete drift

§7.2 is a continuous-time statement. Implementing it on the *discrete* splitting means the
guidance field carries $\gamma$ in **two** places, and both are load-bearing:

1. The magic constant uses the **effective** diffusion, $a = \beta\,\gamma g^2/2$ (§7.2 (ii)),
   not the base $\beta g^2/2$.
2. The field is divided by $(1+\gamma)$, because the deterministic half transports over the
   increment $-(1+\gamma)\,dt_i$ rather than $-dt_i$ (§1), so the same field would otherwise
   produce $(1+\gamma)\times$ the displacement per grid step.

Both collapse to the familiar form at $\gamma=1$. An implementation that drops either one
therefore **passes at $\gamma=1$ and fails on both sides of it** — which is precisely why
`TestChurnSteeringWithEulerMaruyamaWeight` sweeps churn rather than testing a single value.
The base transport is the PF velocity, not the reverse-SDE drift $f-g^2s$: the churn already
supplies the stochasticity, and a score-corrected drift double-counts it.

### 7.4 The two routes are not term-by-term equal

The untwisted endpoint potential of §4 is a *different*, discrete FK factorisation. It does
not converge term by term to the finite-variation weight of Eq. (276): the endpoint difference
contains the **stochastic increment** of $\rho_t(X_t)$, which the continuous-time weight has
already moved into the dynamics via the guided drift. Recovering D.6's guided drift and weight
from the endpoint form requires the corresponding proposal change (or a continuous-time
Girsanov argument).

They target the same bridge by different bookkeeping, and empirically they agree: terminal
$W_1$ over 8 seeds on a Karras setup was $0.0357 \pm 0.0099$ for Euler-Maruyama FKC and
$0.0346 \pm 0.0080$ for churn FKC at $\gamma=1$, statistically indistinguishable. Choosing
$\gamma$ is therefore a variance/accuracy decision about the dynamics alone.

For finite churn steps the discrete endpoint-potential construction is simpler and avoids
approximating the churn split by a continuous SDE — it is exact at finite step size, where the
continuous-time route is exact only in the fine-grid limit.

---

## 8. Resampling

The weights are maintained in log space per particle, $\log w \in \mathbb R^N$, and the
diagnostic is the normalised effective sample size (Kish),

$$
\bar w = \operatorname{softmax}(\log w),
\qquad
\frac{\mathrm{ESS}}{N} = \frac{1}{N\sum_{n}\bar w_n^{2}} \in (0,1].
$$

**Systematic resampling.** When a resample fires, ancestor indices are drawn systematically —
one uniform $U\sim\mathcal U[0,1)$, strata $u_n=(n+U)/N$, and $\mathcal A_n =
\operatorname{searchsorted}(\operatorname{cumsum}\bar w,\,u_n)$. Systematic resampling has
lower variance than multinomial at the same cost and is unbiased in the usual SMC sense.
After the gather, $\log w \leftarrow 0$.

**Two threshold regimes.** The `ess_threshold` parameter $\tau$ selects the strategy:

- **Adaptive**, $0 < \tau < 1$: resample whenever $\mathrm{ESS}/N < \tau$.
- **Fixed interval**, $\tau \ge 1$ and a whole number: resample unconditionally every
  $\lfloor\tau\rfloor$ steps, regardless of ESS. $\tau=1$ resamples every step. A non-integer
  $\tau \ge 1$ is rejected, because it would silently mean "an interval of 2.5 steps".
- **Degenerate case**: if $\tau \ge N$ (the total number of integration steps), no intermittent
  resample ever fires; weights accumulate over the whole trajectory and only the mandatory
  final resample corrects the cloud. By §4.2 this is a perfectly usable regime for this
  sampler, and the tests exercise it.

**The final resample is mandatory** and always fires after the last step, regardless of
regime, so that the returned terminal particle set is an unweighted sample from $\pi_{t_N}$.

### 8.1 $\rho$ is carried and gathered, never recomputed

The one implementation detail that keeps the algorithm at a single potential evaluation per
step:

- $\rho_{t_i}(x_i)$ is **carried across steps**. The value written by step $i-1$ as
  $\rho_{t_i}(x_i)$ is the ancestor term that step $i$'s endpoint difference needs. Recomputing
  it would double the cost of a `potential` that unrolls a denoiser.
- On a resample, $\rho$ is **gathered by the ancestor index**, $\varrho \leftarrow
  \varrho[\mathcal A]$, in lockstep with $x \leftarrow x[\mathcal A]$. This is exact, not an
  approximation: a resampled particle *is* its ancestor, at the same state and the same time,
  so its potential is its ancestor's potential by definition.

Initialisation costs one extra evaluation, $\varrho_0 = \rho_{t_0}(x_0)$, so the unguided
sampler makes $N+1$ potential evaluations over $N$ steps.

---

## 9. Full algorithm

Cost annotations: **[score]** = one model/score evaluation, **[$\rho$]** = one potential
evaluation, **[bwd]** = one backward pass, **[free]** = closed-form arithmetic.

The `drift` callable is the caller's: it is the PF velocity plus whatever guidance field the
caller wants, and `weight_update` carries whatever compensation that field needs. The sampler
never learns which part is guidance.

```text
INPUT
  drift      b(x,t)                 -> PF velocity v, plus guidance u if any   [score(+bwd)]
  transition K(x,t,s)               -> exact forward kernel, s > t             [free]
  potential  rho(x,t)               -> beta(t) r(x,t), one scalar/particle     [rho]
  weight_up  W(x,t,dt_signed)       -> optional incremental log weight         [varies]
                                       dt_signed = -dt < 0, the code's sign (Sec. 1)
  x          [N,*rest,D]            -> N particles drawn from q_{t_0}
  t          t_0 > t_1 > ... > t_N  -> strictly decreasing grid in [0,1]
  gamma >= 0                        -> churn strength (`churn` in the code)
  tau                               -> resampling threshold (see Sec. 8)

INIT
  log_w <- 0                        [N]                                        [free]
  varrho <- rho(x, t_0)                                                        [rho]  <-- the +1

FOR i = 0 .. N-1:
    dt <- t[i] - t[i+1]                       # step size, > 0                 [free]

    # ---- (a) churn half: EXACT forward transition t[i] -> t[i] + gamma*dt ----
    if gamma > 0 and t[i] + gamma*dt <= 1:    # + : reheat, towards noise      [free]
        x <- K(x, t[i], t[i] + gamma*dt)      # no score, closed-form draw     [free]

    # ---- (b) deterministic half: transport t[i] + gamma*dt -> t[i+1] ----
    span <- t[i+1] - (t[i] + gamma*dt)        # = -(1+gamma)*dt, towards data  [free]
    if W is not None:                         # Sec. 6 compensation, or the
        log_w <- log_w + W(x, t[i] + gamma*dt, -dt)   # Sec. 7 D.6 weight      [varies]
    x <- x + b(x, t[i] + gamma*dt) * span                                      [score]

    # ---- (c) retarget: the endpoint potential difference (Sec. 4) ----
    if rho is not None:
        varrho_next <- rho(x, t[i+1])                                          [rho]
        log_w       <- log_w + varrho_next - varrho                            [free]
        varrho      <- varrho_next

    # ---- (d) selection ----
    (x, varrho, log_w) <- MAYBE_RESAMPLE(i, x, varrho, log_w)                  [free]
    record x, softmax(log_w)

# ---- mandatory final resample: terminal cloud is unweighted ----
A      <- SYSTEMATIC(softmax(log_w))                                           [free]
x      <- x[A];  log_w <- 0
record x, uniform

PROCEDURE MAYBE_RESAMPLE(i, x, varrho, log_w):                                 [free]
    ess <- 1 / (N * sum_n softmax(log_w)_n^2)
    if tau >= 1:  trigger <- ((i+1) mod floor(tau) == 0)     # fixed interval
    else:         trigger <- (ess < tau)                     # adaptive
    if trigger:
        A <- SYSTEMATIC(softmax(log_w))
        x <- x[A];  varrho <- varrho[A]      # GATHER rho, do not re-evaluate
        log_w <- 0
    return (x, varrho, log_w)
```

The three configurations, all supported:

- **`potential` only, unguided `drift`** — §4, the reference implementation. Exact at finite
  step size, no gradients anywhere.
- **`potential` + guided `drift` + compensation in `weight_update`** — §6. Exact for any $u$,
  but needs $\Delta\rho$.
- **`weight_update` only, carrying the Prop. D.6 weight, with the §7.3 guided drift** — §7.
  Needs $\nabla r$, no Laplacian. The *same closure* steers the Euler-Maruyama sampler.

Setting $\gamma = 0$ (`churn=0` in the code) skips the transition entirely and reduces every step to a deterministic
PF-ODE step whose weight is a pure change-of-target term; a flat potential $\rho\equiv0$ then
leaves the base churn sampler bit-for-bit unchanged, which is the invariant the tests assert.

---

## 10. Cost summary

Per integration step, in the unguided configuration:

| item | count | note |
|---|---|---|
| score evaluations | $1$ | the PF velocity at $(\hat x_i,\,t_i+\gamma dt_i)$; independent of $\gamma$ |
| potential evaluations | $1$ | at the step endpoint only |
| backward passes | $0$ | the endpoint difference needs $\rho$ *values* only |
| forward-kernel draws | $1$ | closed-form Gaussian, no score |

Guided (§6) adds one backward pass per step, because $u = c\tfrac{g^2}{2}\nabla\rho$ needs
$\nabla_x\rho$; the divergence half of the compensation stays analytic. Totals over $N$ steps:
$N$ score calls and $N+1$ potential calls unguided.

**Churn is not paid for in evaluations.** It is paid for in transport length: the deterministic
half covers $|\mathrm{span}_i| = (1+\gamma)\,dt_i$ instead of $dt_i$, so the same score budget
buys a longer, coarser Euler step.

**Contrast with the Euler-Maruyama FKC route.** `steered_reverse_sampling` on the same reward
must build, per step, the guided drift $f - g^2 s - \beta\tfrac{g^2}{2}\nabla r$ and the
Prop. D.6 weight $[\dot\beta r + \beta\partial_t r + \langle\beta\nabla r,\tfrac{g^2}{2}s-f\rangle]$.
Both need $\nabla_x r$, so a backward pass through the reward — and, for a denoised reward,
through every substep of the unrolled denoiser — is structurally required and cannot be cached
away. Instrumented over $99$ steps with a $10$-substep denoiser:

| sampler | score calls | per step | denoise | backward |
|---|---|---|---|---|
| churn, unsteered | $99$ | $1.00$ | $0$ | $0$ |
| churn-FKC, closed-form $r(x_t)$ | $99$ | $1.00$ | $0$ | $0$ |
| churn-FKC, denoised $r(D)$ | $649$ | $6.56$ | $100$ | $0$ |
| EM-FKC, denoised $r(D)$ | $1296$ | $13.09$ | $198$ | $198$ |

For a closed-form reward, **FKC on the churn sampler is free**: the steered sampler makes
exactly the same $99$ score calls as the unsteered one.

---

## 11. Exactness conditions and failure modes

The construction of §3–§6 is exact under one hypothesis, $q_{t_i}M_i = q_{t_{i+1}}$, plus
correct evaluation of $\rho$ (and of the compensation, if guiding). What breaks it:

1. **$S_{\mathrm{noise}} \ne 1$.** EDM inflates the injected churn noise by a factor
   $S_{\mathrm{noise}}$ to compensate a learned denoiser that under-restores variance. This
   deliberately breaks $q_{t_i}K = q_{t_i+\gamma dt_i}$: the post-churn marginal is a slightly wider
   mixture, not $q_{t_i+\gamma dt_i}$. The churn half's weight increment
   $\rho_{t+\gamma dt}(\hat x) - \rho_t(x)$ is then no longer the exact correction, and there is no
   cheap repair — the omitted factor is a density ratio between two different Gaussian
   convolutions of $q_t$. Use $S_{\mathrm{noise}}=1$ if exactness matters.

2. **PF-ODE solver truncation.** $\Phi$ preserves the marginals only up to its local
   truncation error ($O(h^2)$ for Euler, $O(h^3)$ for Heun). The steered sampler inherits
   exactly the base sampler's discretisation error and neither amplifies nor corrects it. The
   fully general deterministic importance correction,
   $\Delta\log w_\Phi = \rho_{t_{i+1}}(y)-\rho_{t_i+\gamma dt_i}(\hat x_i)
   + \log q_{t_{i+1}}(y) - \log q_{t_i+\gamma dt_i}(\hat x_i) + \log|\det D\Phi|$ for $y=\Phi(\hat x_i)$,
   would repair it, but the Jacobian determinant and exact inverse-density terms are
   unavailable for a numerical solver.

3. **A learned (inexact) score.** If $s_\theta \ne \nabla\log q_t$, then $\Phi$ transports the
   wrong marginal family and the base proposal no longer satisfies $q_{t_i}M_i = q_{t_{i+1}}$
   even in the exact-flow limit. This is a property of every diffusion sampler, not of FKC; in this
   repository the score is analytic, which is why the tests can assert exactness against a
   closed-form tilted density at all.

4. **A guidance field whose compensation is omitted or wrong.** By §6.2 this is a silent
   failure: the sampler still concentrates on high reward, so any reward-based diagnostic
   passes. Only a comparison against the analytic tilted marginal catches it — measured
   $7.5\times$–$9\times$ degradation in mean $W_1$ (§6.3). A field $u$ for which
   $\nabla\!\cdot\! u$ is only known approximately is not safe to use.

5. **A twisted stochastic churn proposal without its density ratio.** If the Gaussian churn
   kernel $K_i$ is replaced by a reward-biased $\widetilde K_i$ (§6.4), exactness requires
   adding $\log K_i(\hat x_i\mid x_i) - \log\widetilde K_i(\hat x_i\mid x_i)$ to the weight.

Under $S_{\mathrm{noise}}=1$, an exact score, and either $u=0$ or an exact compensation, the FK
weighting adds **no bias whatsoever** on top of the unsteered churn sampler's own ODE
discretisation error.

---

## 12. Validation

Validation compares the **weighted** particle cloud against $\pi_t \propto q_t e^{\rho_t}$ at
every recorded time, not only at the terminal endpoint — see `tests/CLAUDE.md`. The sweep
covers:

- churn strength, including $0$, $1$, and over-churn ($\gamma=2$);
- all three `ess_threshold` regimes (adaptive, fixed interval, single final resample);
- unguided endpoint potential, guided deterministic half at several $c$, and the Prop. D.6
  weight reused verbatim from the Euler-Maruyama sampler;
- direct rewards $r(x)$ and denoised rewards $r(D(x,t))$;
- both `BetaSchedule` (VP) and `KarrasSchedule` (VE).

The untwisted endpoint-potential sampler is the reference implementation; every guided variant
must match its weighted intermediate marginals. Two checks are stronger than a $W_1$ statistic
and worth keeping:

- **Reduction.** A flat potential $\rho\equiv0$ must reproduce `reverse_churn_sampling`
  bit-for-bit at every $\gamma$.
- **The weight identity.** With resampling only at the very end, the increments telescope
  (§4.2) to a closed form — $\log w_i = \rho_{t_i}(x_i)-\rho_{t_0}(x_0)$ unguided, plus
  $\sum_{j<i}\mathrm{corr}_j\cdot\mathrm{span}_j$ guided — checkable to $10^{-12}$ in float64.
  A $W_1$ check cannot see this: it only observes the weights through a resampled cloud.
