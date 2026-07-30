# The Churn-FKC Algorithm, in Terms of Potentials

**Status:** Reference note

**Related documents:** [`fkc_churn_steering.md`](fkc_churn_steering.md), [`churn_sampler.md`](churn_sampler.md), [`fkc_steering.md`](fkc_steering.md), [`schedule.md`](schedule.md), [`churn_fkc_cost.md`](churn_fkc_cost.md)

This document states the complete algorithm obtained by running Feynman-Kac-Corrector (FKC)
steering on top of a churn sampler whose step is (a) an *exact* noising transition to a
higher noise level followed by (b) a reverse probability-flow ODE step. It is written
entirely in terms of the tilt **potential** $\rho_t(x)$, because that — not the reward, not
its gradient, not its Laplacian — is the only object the correction needs. The implementation
of record is `steered_reverse_churn_sampling` in `torchGMM/sampling.py`.

The two genuinely new results relative to the design notes are §6 (the exact weight when the
deterministic half is *guided*) and §7 (the $\kappa$-invariance of the continuous-time
Proposition D.6 weight under the churn family of effective reverse SDEs).

---

## 1. Notation

**Grid.** A strictly decreasing grid in $[0,1]$,

$$t_0 > t_1 > \dots > t_N, \qquad \delta_i = t_i - t_{i+1} > 0, \qquad dt_i = t_{i+1}-t_i = -\delta_i .$$

The convention throughout the repository is $t=0$ for data and $t=1$ for noise, so reverse
sampling runs on a decreasing grid and every step size $dt_i$ is negative.

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

It is exactly one score evaluation, and integrating $\dot x = v_t(x)$ on the decreasing grid
transports $q_t$ to $q_{t'}$ for $t' < t$.

**Exact transition kernel.** For $s > t$ (i.e. $s$ noisier), [`schedule.md`](schedule.md)
derives

$$
\boxed{\;
K_{t\to s}:\quad
X_s = \frac{\alpha_s}{\alpha_t}X_t + \sqrt{\,\sigma_s^2 - \frac{\alpha_s^2}{\alpha_t^2}\sigma_t^2\,}\;\eta,
\qquad \eta\sim\mathcal N(0,I).
\;}
$$

This maps $q_t$ onto $q_s$ **exactly**, for any gap $s-t$, with no discretisation error, and
it contains **no score**: it is a single closed-form Gaussian draw. In the repository it is
`Schedule.transition`.

**Churn.** Given a churn strength $\kappa \ge 0$, the reheated time at step $i$ is

$$\hat t_i = \min\bigl(t_i + \kappa\,\delta_i,\;1\bigr),$$

and the transport span of the deterministic half is

$$t_{i+1} - \hat t_i = -(1+\kappa)\,\delta_i \quad\text{(before the clamp bites)}.$$

$\kappa = 0$ is the pure probability-flow ODE (no churn; the implementation then skips
$K$ entirely), $\kappa = 1$ re-noises one full grid step back and transports two steps down,
and $\kappa > 1$ over-churns. The clamp at $1$ is what keeps $\hat t_i$ inside the schedule
near $t=1$; when it bites, the transport span shrinks to match.

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
the FKC construction and the reason a weight correction exists at all: the bridge
$\{\pi_t\}_{t}$ is *prescribed*, and it is in general different from what one gets by pushing
$\pi_{t_0}$ forward through the base dynamics
([`fkc_churn_steering.md`](fkc_churn_steering.md) §3.1). Every increment below is a
**retargeting correction** that converts the pushforward of $\pi_{t_i}$ into $\pi_{t_{i+1}}$.

Consequently the correctness criterion is a statement about the *weighted* particle cloud at
every recorded time, and the tests assert exactly that: `_weighted_wasserstein1` against
`_tilted_density` at 49–56 intermediate slots, not just at $t_N$. An unweighted empirical
distribution between resamples is *expected* to be wrong.

---

## 3. One step as two operators

A churn step is the composition of two complete operators,

$$
x_i
\;\xrightarrow{\;K_{t_i\to\hat t_i}\;}\;
\hat x_i
\;\xrightarrow{\;\Phi_{\hat t_i\to t_{i+1}}\;}\;
x_{i+1},
\qquad
M_i = K_{t_i\to\hat t_i}\,\Phi_{\hat t_i\to t_{i+1}} .
$$

- $K$ is the exact forward noising kernel of §1. It preserves the base marginal family
  **exactly**: $q_{t_i}K_{t_i\to\hat t_i} = q_{\hat t_i}$, for any gap.
- $\Phi$ is probability-flow ODE transport. It preserves the base marginal family **up to
  solver truncation**: $q_{\hat t_i}\Phi_{\hat t_i\to t_{i+1}} = q_{t_{i+1}}$ in the exact-flow
  limit.

Hence

$$\boxed{\;q_{t_i}\,M_i = q_{t_{i+1}}\;}$$

and $M_i$ is a valid *base* proposal for a discrete Feynman-Kac model. This is the whole
correctness argument, and it is a splitting argument, not an SDE discretisation: the two
halves are operators, not the two terms of one SDE added at the same $t$ over the same $dt$.

**The velocity is evaluated at the reheated pair.** After $K$, the state $\hat x_i$ is
distributed according to $q_{\hat t_i}$, not $q_{t_i}$, so the deterministic half must call
$v_{\hat t_i}(\hat x_i)$ — i.e. `velocity(x_hat, t_hat)`, not `velocity(x_hat, t_curr)`.
Using the score at $t_i$ is the wrong score for that state; at $\kappa=1$ the mismatch is a
full grid step wide.

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
exactly the thing that transforms $q_{\hat t_i}$ into $q_{t_{i+1}}$ under $\Phi$; it cancels
between base proposal and base target.

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

Because a step is two operators, the increment splits at the reheated state:

$$
\boxed{\;\Delta\log w_i^{\mathrm{churn}} = \rho_{\hat t_i}(\hat x_i) - \rho_{t_i}(x_i)\;}
$$

which is **exact** because $q_{t_i}K_{t_i\to\hat t_i} = q_{\hat t_i}$ holds with no
approximation whatsoever — the stochastic forward kernel introduces no additional likelihood
ratio precisely because it *is* the base kernel between those two marginals — and

$$
\boxed{\;\Delta\log w_i^{\mathrm{ODE}} = \rho_{t_{i+1}}(x_{i+1}) - \rho_{\hat t_i}(\hat x_i).\;}
$$

Adding them,

$$
\Delta\log w_i^{\mathrm{churn}} + \Delta\log w_i^{\mathrm{ODE}}
= \rho_{t_{i+1}}(x_{i+1}) - \rho_{t_i}(x_i),
$$

which is §4 verbatim: $\rho_{\hat t_i}(\hat x_i)$ cancels. The split is therefore never
required for correctness.

**Why it exists.** The split creates an intermediate state at which the ESS can be tested and
a resample can fire — at the *reheated* noise level, where the freshly injected noise has just
maximised particle diversity, so a gather there duplicates ancestors with less loss of
diversity than one at $t_{i+1}$. This is the `resample_at_churn=True` path.

**What it costs.** $\rho_{\hat t_i}(\hat x_i)$ is a state that is otherwise never scored, so
the split evaluates the potential exactly twice per step instead of once: measured
$2.00$ evaluations/step against $1.00$. Measured benefit on the Karras denoised-reward setup:
mean $\mathrm{ESS}/N$ moved from $0.997$ to $0.998$, and terminal $W_1$ differences were
inside the seed-to-seed spread. Given §4.2 that is the expected outcome — the telescoping
already bounds $\log w$, so there is no depletion for the extra selection opportunity to
repair. The split is worth revisiting only in a regime where ESS genuinely collapses.

---

## 6. The guided deterministic half

*This is the result that is not in any of the design notes; it is derived here.*

Suppose the deterministic half integrates $v_t + u$ instead of $v_t$, for an **arbitrary**
guidance field $u(x,t)$. The base flow is no longer marginal-preserving, so §4's cancellation
no longer applies. The exact repair is a single additional scalar.

### 6.1 Derivation

Work in continuous time over the transport span $[\hat t_i, t_{i+1}]$, with $\partial_t$ and
all velocities referred to the same (decreasing) time parameterisation. For a density evolving
under a deterministic velocity field $w$ with no diffusion, the log form of the continuity
equation is

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
= \rho_{t_{i+1}}(x_{i+1}) - \rho_{\hat t_i}(\hat x_i)
+ \int_{\hat t_i}^{t_{i+1}}\Bigl[\nabla\!\cdot\! u + \langle s_t, u\rangle\Bigr]dt .
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
3. **$\nabla\!\cdot\! u$ must be analytic.** The implementation requires the caller to return
   $u$ *and* the scalar $\nabla\!\cdot\! u + \langle s_t,u\rangle$ from the same `guidance`
   callable, because $\nabla\!\cdot\! u$ is closed-form for the guidance fields anyone actually
   uses (for $u = c\tfrac{g^2}{2}\nabla\rho$ with a Gaussian reward,
   $\Delta\rho = -\beta(t)D/\varsigma^2$ is a constant), and a numerical divergence estimator
   would cost $D$ extra passes and defeat the purpose.

In the implementation the integral is evaluated by a single left-endpoint quadrature at
$\hat t_i$, i.e. `log_w += corr * span` with `span = t_next - t_hat`. This is first order in
the span, matching the order of the Euler transport step it accompanies; it is not an extra
source of error relative to the flow itself.

### 6.3 The term is load-bearing

Dropping the compensation while keeping the guidance still produces high-reward samples — it
just targets the wrong distribution, which is exactly the failure a reward-only diagnostic
misses. Measured over 49 intermediate time slots with $u = c\,\tfrac{g^2}{2}\nabla\rho$, mean
$W_1$ against the analytic tilted marginal degrades by

| $c$ | with compensation | without | factor |
|---|---|---|---|
| $0.5$ | $0.023$ | $0.172$ | $7.5\times$ |
| $1.0$ | $0.035$ | $0.319$ | $9\times$ |

---

## 7. The effective-SDE view and its $\kappa$-invariance

Everything above is a *discrete* Feynman-Kac factorisation and never approximates the churn
split by an SDE. It is nevertheless worth knowing what the split converges to, and what
Proposition D.6 says about that limit, because the answer is cleaner than one would guess.

### 7.1 The effective generator

For a grid step $\delta = |dt|$ and churn interval $h = \kappa\delta$, the infinitesimal churn
split converges to a reverse diffusion whose decreasing-time drift callable and diffusion
coefficient are

$$
\boxed{\;
b_\kappa(x,t) = f(x,t) - \frac{1+\kappa}{2}\,g(t)^2\,s_t(x),
\qquad
g_\kappa(t) = \sqrt{\kappa}\;g(t).
\;}
$$

The family interpolates: $\kappa=0$ gives $b_0 = f - \tfrac12 g^2 s = v_t$ and $g_0=0$, the
probability-flow ODE; $\kappa=1$ gives $b_1 = f - g^2 s$ and $g_1 = g$, the ordinary Anderson
reverse SDE that Proposition D.6 assumes; $\kappa>1$ over-churns. This is the standard
$\lambda$-family of reverse SDEs at $\lambda = \sqrt\kappa$.

### 7.2 Redoing Prop. D.6 Step 5 with $(b_\kappa, g_\kappa)$

Transcribe from [`fkc_steering.md`](fkc_steering.md), remembering the §1.1 clash: its
$\sigma_t^2$ is a diffusion coefficient squared, so here it becomes $g_\kappa^2 = \kappa g^2$.
Its increasing-reverse-time drift is $v_t = -b_\kappa = -f + \tfrac{1+\kappa}{2}g^2 s$. The
pre-guidance weight Eq. (270) reads

$$
g^{\mathrm{FK}}_t = \Bigl\langle \beta\nabla r,\; v_t - g_\kappa^2\,s_t - \tfrac{g_\kappa^2}{2}\beta\nabla r\Bigr\rangle
- \beta\,\tfrac{g_\kappa^2}{2}\Delta r + \dot\beta\, r,
$$

and Step 4's free drift $a\nabla r$ contributes $a\Delta r + a\langle\nabla\log p_t,\nabla r\rangle$.

**(i) D.6's first cancellation fails for $\kappa\ne1$.** In D.6 the packaged inner product
collapses because $v_t - g^2 s_t = -f_t$ exactly. Here

$$
v_t - g_\kappa^2 s_t
= -f + \frac{1+\kappa}{2}g^2 s - \kappa g^2 s
= -f + \frac{1-\kappa}{2}\,g^2 s ,
$$

so a residual score term $\tfrac{1-\kappa}{2}g^2 s$ survives, vanishing only at $\kappa=1$.

**(ii) The Laplacian cancellation moves.** The weight carries $-\beta\tfrac{g_\kappa^2}{2}\Delta r$
and the drift contributes $+a\Delta r$, so deleting $\Delta r$ now requires

$$\boxed{\;a = \beta\,\frac{g_\kappa^2}{2} = \beta\,\frac{\kappa\,g^2}{2}\;}$$

rather than $\beta g^2/2$. The guided drift therefore *does* depend on $\kappa$:

$$
dx_t = \Bigl(-f_t + \tfrac{1+\kappa}{2}g^2 s_t + \beta\,\tfrac{\kappa g^2}{2}\nabla r\Bigr)dt
+ \sqrt{\kappa}\,g\,dW_t .
$$

**(iii) The residual score terms recombine to exactly $\tfrac12$.** With that $a$, expand
$a\langle\nabla\log p_t,\nabla r\rangle$ using $\nabla\log p_t = s_t + \beta\nabla r$:

$$
a\langle\nabla\log p_t,\nabla r\rangle
= \frac{\beta\kappa g^2}{2}\langle s_t,\nabla r\rangle + \frac{\kappa g^2}{2}\beta^2\lVert\nabla r\rVert^2 .
$$

The $\beta^2\lVert\nabla r\rVert^2$ piece cancels against the
$-\tfrac{g_\kappa^2}{2}\beta\nabla r$ slot inside the packaged inner product, exactly as in
D.6. What remains are the two score terms, from (i) and from the drift compensation:

$$
\beta\,\frac{1-\kappa}{2}\,g^2\langle\nabla r, s_t\rangle
\;+\;
\beta\,\frac{\kappa}{2}\,g^2\langle\nabla r, s_t\rangle
\;=\;
\beta\,g^2\langle\nabla r, s_t\rangle\underbrace{\Bigl[\frac{1-\kappa}{2}+\frac{\kappa}{2}\Bigr]}_{=\;1/2}
\;=\;\beta\,\frac{g^2}{2}\langle\nabla r, s_t\rangle .
$$

The $\kappa$'s cancel identically. Therefore the final weight is

$$
\boxed{\;
dw_t = \Bigl[\;\dot\beta_t\, r(x_t) \;+\; \Bigl\langle \beta_t\nabla r(x_t),\; \tfrac{g(t)^2}{2}s_t(x_t) - f_t(x_t)\Bigr\rangle\;\Bigr]dt
\;}
$$

— Eq. (276) of Proposition D.6, with the **base** $g^2$, **independent of $\kappa$**. (With a
time-dependent reward, $\dot\beta_t r \mapsto \partial_t\rho_t$ as in the generalisation at the
end of `fkc_steering.md`.)

**Interpretation.** The invariance is not a coincidence. The vector field appearing in the
weight is

$$\frac{g^2}{2}s_t - f_t = -\Bigl(f_t - \frac{g^2}{2}s_t\Bigr) = -\,v_t,$$

the negated probability-flow velocity. The PF velocity is the one member of the $\lambda$-family
that carries no stochasticity, and the entire $\kappa$-dependence of the family lives in the
score-plus-noise part that the FK weight is blind to. Churn changes *how* the sampler moves; it
does not change *which* reweighting makes the tilted bridge exact.

The practical reading: choosing $\kappa$ is a variance/accuracy decision about the dynamics
alone. Empirically the two routes agree — terminal $W_1$ over 8 seeds on a Karras setup was
$0.0357 \pm 0.0099$ for Euler-Maruyama FKC and $0.0346 \pm 0.0080$ for churn FKC at $\kappa=1$,
statistically indistinguishable.

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
unsplit sampler makes $N+1$ potential evaluations over $N$ steps.

---

## 9. Full algorithm

Cost annotations: **[score]** = one model/score evaluation, **[$\rho$]** = one potential
evaluation, **[bwd]** = one backward pass, **[free]** = closed-form arithmetic.

```text
INPUT
  velocity  v(x,t)                 -> PF velocity f - (g^2/2) s_t          [score]
  transition K(x,t,s)              -> exact forward kernel, s > t          [free]
  potential rho(x,t)               -> beta(t) r(x,t), one scalar/particle  [rho]
  guidance  G(x,t) -> (u, corr)    -> optional; corr = div u + <s_t,u>     [rho + bwd]
  x         [N,*rest,D]            -> N particles drawn from q_{t_0}
  t         t_0 > t_1 > ... > t_N  -> strictly decreasing grid in [0,1]
  kappa >= 0                       -> churn strength
  tau                              -> resampling threshold (see Sec. 8)
  resample_at_churn: bool          -> split the increment at t_hat (Sec. 5)

INIT
  log_w <- 0                        [N]                                    [free]
  varrho <- rho(x, t_0)                                                    [rho]   <-- the +1

FOR i = 0 .. N-1:
    dt    <- t[i+1] - t[i]                    # negative                   [free]
    t_hat <- min(t[i] + kappa*|dt|, 1)                                     [free]

    # ---- (a) churn half: EXACT forward transition t_i -> t_hat ----
    if t_hat > t[i]:
        x <- K(x, t[i], t_hat)                # no score, closed-form draw [free]

    if resample_at_churn:                     # Sec. 5; optional
        varrho_hat <- rho(x, t_hat)                                        [rho]   <-- 2nd/step
        log_w      <- log_w + varrho_hat - varrho    # exact: q_t K = q_t_hat  [free]
        varrho     <- varrho_hat
        (x, varrho, log_w) <- MAYBE_RESAMPLE(i, x, varrho, log_w)          [free]

    # ---- (b) deterministic half: PF-ODE transport t_hat -> t_{i+1} ----
    span <- t[i+1] - t_hat                    # = -(1+kappa)*|dt|          [free]
    if guidance is None:
        x <- x + v(x, t_hat) * span                                        [score]
    else:
        (u, corr) <- G(x, t_hat)              # corr = div u + <s_t_hat,u> [rho+bwd]
        x         <- x + (v(x, t_hat) + u) * span                          [score]
        log_w     <- log_w + corr * span      # Sec. 6 compensation        [free]

    # ---- (c) retarget: the endpoint potential difference ----
    varrho_next <- rho(x, t[i+1])                                          [rho]
    log_w       <- log_w + varrho_next - varrho                            [free]
    varrho      <- varrho_next

    # ---- (d) selection ----
    (x, varrho, log_w) <- MAYBE_RESAMPLE(i, x, varrho, log_w)              [free]
    record x, softmax(log_w)

# ---- mandatory final resample: terminal cloud is unweighted ----
A      <- SYSTEMATIC(softmax(log_w))                                       [free]
x      <- x[A];  log_w <- 0
record x, uniform

PROCEDURE MAYBE_RESAMPLE(i, x, varrho, log_w):                             [free]
    ess <- 1 / (N * sum_n softmax(log_w)_n^2)
    if tau >= 1:  trigger <- ((i+1) mod floor(tau) == 0)     # fixed interval
    else:         trigger <- (ess < tau)                     # adaptive
    if trigger:
        A <- SYSTEMATIC(softmax(log_w))
        x <- x[A];  varrho <- varrho[A]      # GATHER rho, do not re-evaluate
        log_w <- 0
    return (x, varrho, log_w)
```

Setting `guidance = None`, `resample_at_churn = False` gives the unguided telescoped form of
§4, which is the reference implementation. Setting `kappa = 0` additionally skips the
transition entirely and reduces every step to a deterministic PF-ODE step whose weight is a
pure change-of-target term; a flat potential $\rho\equiv0$ then leaves the base churn sampler
bit-for-bit unchanged, which is the invariant the tests assert.

---

## 10. Cost summary

Per integration step, in the unguided unsplit configuration:

| item | count | note |
|---|---|---|
| score evaluations | $1$ | the PF velocity at $(\hat x_i,\hat t_i)$; independent of $\kappa$ |
| potential evaluations | $1$ | $2$ if `resample_at_churn` |
| backward passes | $0$ | the endpoint difference needs $\rho$ *values* only |
| forward-kernel draws | $1$ | closed-form Gaussian, no score |

Guided (§6) adds one backward pass per step, because $u = c\tfrac{g^2}{2}\nabla\rho$ needs
$\nabla_x\rho$; the divergence half of `corr` stays analytic. Totals over $N$ steps:
$N$ score calls and $N+1$ potential calls unguided-unsplit, $2N+1$ potential calls split.

**Churn is not paid for in evaluations.** It is paid for in transport length: the deterministic
half spans $(1+\kappa)\delta_i$ instead of $\delta_i$, so the same score budget buys a longer,
coarser Euler step.

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
correct evaluation of $\rho$ (and of `corr`, if guiding). What breaks it:

1. **$S_{\mathrm{noise}} \ne 1$.** EDM inflates the injected churn noise by a factor
   $S_{\mathrm{noise}}$ to compensate a learned denoiser that under-restores variance. This
   deliberately breaks $q_{t_i}K = q_{\hat t_i}$: the post-churn marginal is a slightly wider
   mixture, not $q_{\hat t_i}$. The churn half's weight increment
   $\rho_{\hat t}(\hat x) - \rho_t(x)$ is then no longer the exact correction, and there is no
   cheap repair — the omitted factor is a density ratio between two different Gaussian
   convolutions of $q_t$. Use $S_{\mathrm{noise}}=1$ if exactness matters.

2. **PF-ODE solver truncation.** $\Phi$ preserves the marginals only up to its local
   truncation error ($O(h^2)$ for Euler, $O(h^3)$ for Heun). The steered sampler inherits
   exactly the base sampler's discretisation error and neither amplifies nor corrects it. The
   fully general deterministic importance correction,
   $\Delta\log w_\Phi = \rho_s(y)-\rho_t(x) + \log q_s(y) - \log q_t(x) + \log|\det D\Phi|$,
   would repair it, but the Jacobian determinant and exact inverse-density terms are
   unavailable for a numerical solver.

3. **A learned (inexact) score.** If $s_\theta \ne \nabla\log q_t$, then $\Phi$ transports the
   wrong marginal family and the base proposal no longer satisfies $q_tM = q_{t+dt}$ even in
   the exact-flow limit. This is a property of every diffusion sampler, not of FKC; in this
   repository the score is analytic, which is why the tests can assert exactness against a
   closed-form tilted density at all.

4. **A guidance field whose compensation is omitted or wrong.** By §6.2 this is a silent
   failure: the sampler still concentrates on high reward, so any reward-based diagnostic
   passes. Only a comparison against the analytic tilted marginal catches it — measured
   $7.5\times$–$9\times$ degradation in mean $W_1$ (§6.3). A field $u$ for which
   $\nabla\!\cdot\! u$ is only known approximately is not safe to use.

5. **A twisted stochastic churn proposal without its density ratio.** If the Gaussian churn
   kernel $K_i$ is replaced by a reward-biased $\widetilde K_i$, exactness requires adding
   $\log K_i(\hat x_i\mid x_i) - \log\widetilde K_i(\hat x_i\mid x_i)$ to the weight
   ([`fkc_churn_steering.md`](fkc_churn_steering.md) §6). This variant is not implemented; the
   guided deterministic half of §6 is the route this repository takes instead.

Under $S_{\mathrm{noise}}=1$, an exact score, and either $u=0$ or an exact `corr`, the FK
weighting adds **no bias whatsoever** on top of the unsteered churn sampler's own ODE
discretisation error.
