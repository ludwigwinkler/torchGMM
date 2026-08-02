# The α-family of reverse SDEs

*The α-family SDE is from [A Simple Derivation of the Reverse SDE](https://ludwigwinkler.github.io/blog/SimpleReverseSDE/).
The second half applies the Feynman–Kac steering results of [`alpha_fkc_steering.md`](alpha_fkc_steering.md)
to churn-based (EDM / AF3-style) samplers; the weight updates there are original.*

**Notation warning — two unrelated α's.** $\alpha$ (bare) is the **family parameter**: how much noise
the reverse process injects. $\alpha_t$ (subscripted) is the repo's schedule interpolant,
$x_t = \alpha_t x_0 + \sigma_t\varepsilon$ (`Schedule.get_alpha_t`). $f_t$ and $g_t$ are the forward drift
and diffusion coefficients (`Schedule.forward_drift` / `Schedule.diffusion_coeff`), with
$g_t^2 = 2(\dot\sigma_t\sigma_t - \dot\alpha_t\sigma_t^2/\alpha_t)$.

## The family

Forward SDE $dX_t = f_t(X_t)\,dt + g_t\,dW_t$. In reverse time $\tau = T-t$, for **every** $\alpha\ge 0$,

$$\boxed{\;dX_\tau = \Big[-f_\tau(X_\tau) + \tfrac{1+\alpha^2}{2}g_\tau^2\,\nabla_x\log p_\tau(X_\tau)\Big]d\tau + \alpha g_\tau\,dW_\tau\;}$$

reproduces the forward marginals exactly for every $\alpha$. The score coefficient and the noise level
are locked together — you cannot change one without the other.

| $\alpha$ | score coeff. | noise | name |
| --- | --- | --- | --- |
| $0$ | $\tfrac12 g_\tau^2$ | none | probability-flow ODE |
| $1$ | $g_\tau^2$ | $g_\tau$ | standard reverse SDE (Anderson / Song et al.) |
| $>1$ | $\tfrac{1+\alpha^2}{2}g_\tau^2$ | $\alpha g_\tau$ | churn-style sampler |

Marginals are identical for every $\alpha$ — only the path measure differs, and only $\alpha=1$ is the
true time-reversal of the forward process. $\alpha=0$ is deterministic and invertible; $\alpha>0$ is
self-correcting (injected noise + strengthened score forgets earlier discretization/score error, at
the cost of more per-step noise) — the tradeoff Karras et al. expose as stochastic churn.

## Mapping onto this repo

Write $s$ for the grid variable the code indexes (decreasing, $ds<0$). Substituting $d\tau=-ds$ flips
signs relative to the boxed form: $+f$ and $-\nabla\log p$.

```python
alpha = 1.0  # 0.0 = probability-flow ODE, 1.0 = standard reverse SDE

def drift(x, t):  # [*N, *B, D]
    g_sq = schedule.diffusion_coeff(t) ** 2  # [*B]
    return schedule.forward_drift(x, t) - 0.5 * (1 + alpha**2) * g_sq.unsqueeze(-1) * gmm.score(x, t)

def diffusion(t):  # [*B]
    return alpha * schedule.diffusion_coeff(t)
# alpha == 0.0 -> pass diffusion=None for the ODE
```

For VE schedules ($\alpha_s\equiv 1$, so $f_s=0$, $g_s^2=2\sigma_s\dot\sigma_s$):

$$dx_s = -(1+\alpha^2)\sigma_s\dot\sigma_s\,\nabla\log p_s(x_s)\,ds + \alpha\sqrt{2\sigma_s\dot\sigma_s}\,dW_s.$$

**$\sigma_s^2$ and $g_s^2$ are not interchangeable.** $g_s^2 = 2\sigma_s\dot\sigma_s = \tfrac{d}{ds}\sigma_s^2$
is the variance *accumulation rate* (belongs in every drift/diffusion coefficient); $\sigma_s^2$ is
accumulated variance (belongs in Tweedie, $D(x,\sigma)=x+\sigma^2\nabla\log p_\sigma$). Substituting one
for the other misscales the drift by $2\dot\sigma_s/\sigma_s$. Reading the flow per unit $\sigma$ instead
of per unit time gives EDM's Eq. 4, $dx/d\sigma = -\sigma\nabla\log p_\sigma(x)$ — hence a churn cycle is
indexed by noise level, with no time grid needed.

---

# Churn-based SDE samplers

Plain Euler–Maruyama on any $\alpha>0$ member pits noise injection against score correction inside one
step, both frozen at the step's start. Churn samplers (EDM Algorithm 2, AF3 Algorithm 18) split the two
operations apart and implement each exactly.

## The churn-denoising cycle

Grid $s_0=T>\dots>s_M=0$, $\sigma_i := \sigma_{s_i}$.

**Churn — forward transition, exact.** Pick churn factor $\gamma_i\ge0$, inflate
$\hat\sigma_i=(1+\gamma_i)\sigma_i$, sample the forward kernel exactly. For VE this is plain noise addition:

$$\hat x = x + \sqrt{\hat\sigma_i^2-\sigma_i^2}\,\varepsilon =: x+\varsigma_i\varepsilon,\qquad \varsigma_i^2:=\hat\sigma_i^2-\sigma_i^2.$$

$\varsigma_i^2$ is the **injected variance**. This move has zero discretization error at any $\gamma_i$ — it
*is* the forward process, mapping $p_{s_i}\to p_{\hat s_i}$ exactly.

**Denoising — deterministic reverse step.** Integrate the $\alpha=0$ member (probability-flow ODE) from
$\hat s_i$ to $s_{i+1}$. VE, σ-clock:

$$\frac{dx}{d\sigma} = -\sigma\nabla\log p_\sigma(x) = \frac{x-D(x,\sigma)}{\sigma},\qquad D(x,\sigma)=x+\sigma^2\nabla\log p_\sigma(x)\ \text{(Tweedie)}.$$

Deterministic, so a high-order integrator (Heun, as in EDM/AF3) is available. **The only error a churn
sampler commits is ODE-transport error** — the stochastic half has left the error budget entirely, and
the score is evaluated at the noised point $(\hat x,\hat\sigma)$ rather than the pre-noise point.

## Churn is the α-family, operator-split

Matching the churn and denoising steps' injected variance to one α-family Euler–Maruyama step over the same net interval
gives

$$\boxed{\;\alpha_{\mathrm{eff}}^2
= \frac{\hat\sigma_i^2-\sigma_i^2}{2\sigma_i(\sigma_i-\sigma_{i+1})}
= \left(\gamma_i+\frac{\gamma_i^2}{2}\right)\frac{\sigma_i}{\sigma_i-\sigma_{i+1}}
\approx \frac{\gamma_i\sigma_i}{\sigma_i-\sigma_{i+1}}\;}$$

Churn is a first-order-consistent operator splitting of the α-family, with EDM's
$\gamma_i=\min(S_{\mathrm{churn}}/N,\sqrt2-1)$ giving a roughly time-constant $\alpha_{\mathrm{eff}}$
on a geometric grid, and AF3's $\gamma_0=0.8$ at a few hundred steps giving an $\alpha_{\mathrm{eff}}$
far larger than Euler–Maruyama could integrate stably — viable only because churn is exact.

**In this repo:** `notebooks/af3_steering.py:379-413` (`sample_diffusion`) implements AF3's Algorithm 18
verbatim — churn at `:399-403`, denoising at `:405-411`. `af3_denoiser` (`:360-376`) computes the score at
an arbitrary inflated $\hat\sigma$ in closed form by re-noising the GMM, which any steered variant can
reuse. AF3's `noise_scale` ($\lambda=1.003$) and `step_scale` ($\eta=1.5$) both break the exact-transport
premise the weights below rest on — see "AF3 knobs" below.

---

# Weight updates for the churn cycle

Target family, as in the sibling document: base marginals $q_s$, tilt exponent $\rho(x,s):=\beta_s r(x,s)$,
$\tilde p_s(x)\propto q_s(x)e^{\rho(x,s)}$.

**Lemma (exact base transport ⇒ tilt-ratio weights).** If $K(dx'|x)$ transports the base marginals
exactly ($q_aK=q_b$) and $\{(x^n,w^n)\}$ is properly weighted for $\tilde p_a$, then drawing
$x'^n\sim K(\cdot|x^n)$ and setting

$$w'^n = w^n\cdot\exp\big(\rho(x'^n,b)-\rho(x^n,a)\big)$$

is properly weighted for $\tilde p_b$. Both churn moves qualify, since both transport $q$ exactly.

**Churn.** The forward kernel gives $q_{s_i}K=q_{\hat s_i}$ at *any* $\gamma_i$:

$$\boxed{\;\Delta\log w_{\mathrm{up}} = \rho(\hat x,\hat s)-\rho(x,s)\;}\tag{U}$$

Exact for any churn size, costs only reward evaluations (no score, no $\nabla r$, no Laplacian — the
exact kernel collapses all of it), and the sign is intuitive: noising off a reward peak loses $\rho$.

**Denoising.** The exact flow map of the base ODE transports $q_{\hat s}\to q_{s'}$:

$$\boxed{\;\Delta\log w_{\mathrm{dn}} = \rho(x',s')-\rho(\hat x,\hat s)\;}\tag{D}$$

(This is the sibling document's $\alpha=0$ FKC integrand, $\tfrac{d}{dt}\rho(x_t,t)$, pre-integrated over
the leg — exact up to the integrator's own transport error, same as the unsteered sampler.)

**The cycle telescopes:**

$$\boxed{\;\Delta\log w_{\mathrm{cycle}} = \rho(x_{i+1},s_{i+1})-\rho(x_i,s_i)\;}\tag{C}$$

Consequences: $\beta$ is never evaluated at $\hat\sigma$; without resampling the whole run telescopes to
plain endpoint importance sampling ($\log w=\rho(x_M,0)-\rho(x_0,T)$) — **all of the steering power lives
in resampling**; and churn (not denoising) is what makes resampling useful, since duplicated particles
re-diversify with variance $\varsigma_i^2$ per cycle.

**Two proper weightings, two variance profiles.** Expanding $(U)$ for small $\varsigma^2$, the ratio
weights carry a martingale ($O(\varsigma)$, mean-zero) term that the sibling document's $(276')$ FKC
integrand weights do not — they're bounded-variation by construction. Neither is a discretization of the
other. Ratio weights: exact, cheap (reward evals only), but fluctuate with injected noise (ESS cost
$\sim\sum\|\nabla\rho\|^2\varsigma_i^2$). FKC weights: smooth, but carry $O(\Delta s)$ bias and need score
and reward derivatives. The $O(\varsigma^2)$ term of the expansion is the reward Laplacian — a curiosity
when unguided, but the term the FKC scheme spends to cancel the guidance drift's own Laplacian (next
section).

## Adding the guidance drift

Guidance wants a drift term, not just reweighting. There are two places to put it — **on the denoising
step**, or **as a mean shift of the churn kernel** — and these are not two options within one scheme:
each commits the *whole cycle* to a different weighting scheme. Mixing them (twisting *and* guiding)
reintroduces the reward Laplacian into the weight — see "Do not mix schemes" below.

**The α-parameterized FKC pair** (sibling document's $(275')$–$(276')$ with $\alpha$ explicit). VE,
σ-clock, time increment $|d\sigma|$, $g_\sigma^2=2\sigma$:

$$\boxed{\;\frac{dx}{d\sigma} = -(1+\alpha^2)\sigma\nabla\log q_\sigma(x) - \alpha^2\sigma\nabla\rho(x,\sigma),\quad\text{noise }\alpha\sqrt{2\sigma}\,dW\;}\tag{$275_\alpha$}$$

$$\boxed{\;\Delta\log w = \Big[-\partial_\sigma\rho(x,\sigma) + \langle\nabla\rho(x,\sigma),\,\sigma\nabla\log q_\sigma(x)\rangle\Big]|d\sigma|\;}\tag{$276_\alpha$}$$

$\alpha$ appears three times (score coefficient $1+\alpha^2$, guidance $\alpha^2$, and noise
$\alpha$), forming a single dial — you cannot strengthen guidance without injecting more noise, since the guidance
coefficient $a=\beta_\sigma\alpha^2g^2/2$ is pinned to cancel the reward Laplacian, not a free gain. $\alpha$
does **not** appear in the weight: the alignment field is always the probability-flow ($\alpha=0$)
velocity; more guidance just means particles sit closer to the tilt target, so the same formula runs quieter.

**From $\gamma$ to $\alpha_{\mathrm{eff}}$.** Matching injected variance to $\varsigma_i^2=2\alpha^2\sigma_i\Delta\sigma_i$:

$$\alpha_{\mathrm{eff}}^2 = \Big(\gamma_i+\tfrac{\gamma_i^2}{2}\Big)\frac{\sigma_i}{\Delta\sigma_i},\qquad
\alpha_{\mathrm{eff}}^2\sigma_i\Delta\sigma_i\nabla\rho = \frac{\varsigma_i^2}{2}\nabla\rho =: c_{\mathrm{FKC}}.\tag{G}$$

$(G)$ is the pivot: the FKC-pinned guidance displacement per cycle depends on churn only through
$\varsigma_i^2$, so it's the same vector whether delivered as a descent drift or a kernel mean shift.
The two schemes below differ only in *how they weight* that displacement.

### FKC scheme — guide the deterministic step

Realize $(275_\alpha)$'s noise term via churn (exact), and its score-transport + guidance + weight terms
via denoising — descend from $\hat\sigma_i$, add guidance explicitly, accumulate $(276_\alpha)$:

$$\frac{dx}{d\sigma} = -\sigma\nabla\log q_\sigma(x) - \frac{\varsigma_i^2}{2(\hat\sigma_i-\sigma_{i+1})}\nabla\rho(x,\sigma),\qquad \hat\sigma_i\to\sigma_{i+1},$$

$$\boxed{\;\Delta\log w_{\mathrm{cycle}} = \Big[-\partial_\sigma\rho + \langle\nabla\rho,\,\sigma\nabla\log q_\sigma\rangle\Big]\Delta\sigma_i\;}\tag{F}$$

No reward Laplacian: the cycle's diffusion coefficient $\tilde g=\alpha_{\mathrm{eff}}g$ (supplied by Move
A) is what the Step-5 cancellation uses, even though the guided leg itself is deterministic — the
Fokker–Planck bookkeeping is a property of the composite step. Churn carries no weight (bounded
variation weight). $\gamma$ sets guidance strength directly through $\alpha_{\mathrm{eff}}$ — turning up
churn turns up guidance.

### Ratio scheme — twist the forward kernel

Replace the exact kernel $\mathcal N(x,\varsigma^2I)$ with a mean-shifted $\mathcal N(x+c,\varsigma^2I)$
and append the exact Gaussian ratio — any $c$ stays exact, no Laplacian, no integrand:

$$\boxed{\;\Delta\log w_{\mathrm{up}} = \rho(\hat x,\hat s)-\rho(x,s) + \frac{-2\langle\hat x-x,c\rangle+\|c\|^2}{2\varsigma^2}\;}\tag{U'}$$

Denoising stays unguided; $(D)$ is unchanged. At $c=c_{\mathrm{FKC}}$ this places the particle identically to
the FKC scheme — only the weight differs (martingale vs. bounded-variation). Because the shift is priced
by an explicit density ratio rather than a Laplacian cancellation, it is **not** confined to
$c_{\mathrm{FKC}}$: the variance-optimal twist is the *full* Girsanov shift $c=\varsigma^2\nabla\rho$
(twice $c_{\mathrm{FKC}}$), which cancels the martingale fluctuation exactly. For large per-step
$\varsigma^2$, damp it: $c=(I+\varsigma^2(-\nabla^2\rho))^{-1}\varsigma^2\nabla\rho$. Measured mean ESS
($\gamma=0.3$): $0.367$ (no twist) → $0.425$ ($c_{\mathrm{FKC}}$) → $0.464$ (full shift), all unbiased.

Composed cycle:

$$\boxed{\;\Delta\log w_{\mathrm{cycle}} = \rho(x_{i+1},s_{i+1})-\rho(x_i,s_i) + \frac{-2\langle\hat x-x,c\rangle+\|c\|^2}{2\varsigma_i^2}\;}\tag{C'}$$

| | FKC scheme (guide descent) | Ratio scheme (twist kernel) |
| --- | --- | --- |
| particle displacement at $c=c_{\mathrm{FKC}}$ | identical | identical |
| weight | $(F)$ | $(U')+(D)\to(C')$ |
| weight character | bounded variation (smooth) | martingale (noisy) |
| exactness | $O(\Delta\sigma)$ bias | exact at any $\gamma$, any $c$ |
| needs $\nabla\log q,\partial_\sigma\rho$ in weight | yes | no — two reward evals |
| admissible guidance strength | pinned to $c_{\mathrm{FKC}}$ | any $c$ |

---

# Practical algorithm for VE schedules

Two ways to steer a churn sampler with FKC. **Option A — steer the churn cycle itself:** keep the exact
two-move cycle as proposal, weight by $(U)/(D)/(C)$ (or $(F)$/$(C')$ under guidance), resample at nodes.
**Option B — effective-EM reading:** regroup (denoising$_i$, churn$_{i+1}$) as one EM step of the
$\alpha_{\mathrm{eff}}$-family and apply the sibling document's machinery directly. The two coincide only
as $\gamma,\Delta\sigma\to0$. **Option A is the implementation** (exact noise placement, native Heun,
cheaper or exact weights); **Option B is the tuning model** — use it to pick $\gamma$ for a target
$\alpha_{\mathrm{eff}}$ and to predict ESS decay, borrowing the sibling document's "choosing α" discussion.

$\sigma$-grid native, like EDM/AF3 — no `Schedule` object needed beyond building the grid (e.g.
`KarrasSchedule.get_sigma_t`). Both schemes share this preamble/tail:

```python
# sigma_grid: descending [sigma_max, ..., sigma_min]; rho(x, sigma) -> [N] tilt exponent
x = sigma_grid[0] * torch.randn(N, *B, D)
log_w = torch.zeros(N)
for i in range(len(sigma_grid) - 1):
    sig, sig_next = sigma_grid[i], sigma_grid[i + 1]
    sig_hat = (1.0 + gamma(sig)) * sig
    vs2 = sig_hat**2 - sig**2                        # injected variance ς²
    ...                                              # <- body, one of the two below
    if ess_ratio(log_w) < ess_threshold:
        x = x[systematic_resample(log_w)]
        log_w = torch.zeros(N)
# final resample as in steered_reverse_sampling
```

**Ratio scheme** — eq. $(C')$; `shift = 0` gives the unguided $(C)$:

```python
    rho_prev = rho(x, sig)                           # at loop top -> resample-safe by construction
    shift = vs2 * grad_rho(x, sig)                   # full Girsanov; damp it for large ς²
    noise = torch.sqrt(vs2) * torch.randn_like(x)
    x = x + shift + noise
    twist = -(shift * (shift + 2 * noise)).sum(-1) / (2 * vs2)    # log k/k', exact
    # Heun on the BASE probability-flow ODE, UNGUIDED. Score at sig_hat via the closed-form
    # noised-GMM trick (af3_steering.py:360-376)
    d1 = -sig_hat * score(x, sig_hat)
    d2 = -sig_next * score(x + (sig_next - sig_hat) * d1, sig_next)
    x = x + (sig_next - sig_hat) * 0.5 * (d1 + d2)
    log_w = log_w + rho(x, sig_next) - rho_prev + twist
```

**FKC scheme** — eq. $(F)$; $\alpha$ never appears explicitly, $(276_\alpha)$ is α-independent and $(G)$
absorbs $\alpha_{\mathrm{eff}}$ into `vs2`:

```python
    # Weight FIRST, at the pre-move state (matches steered_reverse_sampling's weight_update convention)
    align = (grad_rho(x, sig) * sig * score(x, sig)).sum(-1)
    log_w = log_w + (-drho_dsigma(x, sig) + align) * (sig - sig_next)
    x = x + torch.sqrt(vs2) * torch.randn_like(x)    # Churn: noise only, carries NO weight
    rate = vs2 / (2 * (sig_hat - sig_next))
    d1 = -sig_hat * score(x, sig_hat) - rate * grad_rho(x, sig_hat)
    xm = x + (sig_next - sig_hat) * d1
    d2 = -sig_next * score(xm, sig_next) - rate * grad_rho(xm, sig_next)
    x = x + (sig_next - sig_hat) * 0.5 * (d1 + d2)
```

Ratio loop: `rho_prev` is recomputed each cycle from the *current* particle array, so a resample's
reshuffle is automatically picked up — the stale-cached-`log φ_prev` trap flagged in
`notebooks/af3_steering.py` doesn't arise. FKC loop: the weight is a function of current state alone, so
no such hazard — its cost is `score` inside the weight and `grad_rho` at three points per step.

`steered_reverse_sampling` (`torchGMM/sampling.py:110`) can't express either complete loop: it uses a
single-phase Euler–Maruyama update, with `weight_update(x_prev, t_curr, dt)` seeing only the pre-step
state, and resampling reshuffles `x` invisibly to any closure. The FKC scheme's weight *does* fit that
signature, but its split noise/ODE update does not; the ratio scheme additionally needs the post-step
state. A new entry point (`churn_steered_reverse_sampling`) is therefore needed.

## AF3 knobs under steering

- **`step_scale` $\eta=1.5$** stretches denoising beyond the base flow, biasing the ratio-scheme weights;
  $(D')$ prices the mismatch exactly since it's a non-reward guidance field
  ($u=(\eta-1)\sigma\nabla\log q_\sigma$). Verified on one leg: uncorrected $\eta=1.5$ contracts the
  marginal (std $2.59$ vs. base $2.86$); corrected, $2.87$. $\eta=1$ remains the cheap default.
- **`noise_scale` $\lambda=1.003$** makes churn's variance $\lambda^2\varsigma^2\ne\varsigma^2$; set
  $\lambda=1$, or append the analogous one-line Gaussian correction.
- **The churn band** ($[S_{\mathrm{tmin}},S_{\mathrm{tmax}}]$) doubles as an ESS-cost control (weight
  variance peaks where $\beta_s$ peaks, near the data end, exactly where the band already turns churn
  off) — but under the FKC scheme, turning churn off also sets $\alpha_{\mathrm{eff}}=0$, which by $(G)$
  **turns guidance off too**. If guidance near the data end matters, the band needs reconsidering.

## Choosing γ

$\gamma$ carries all of $\alpha$'s roles plus two more: it buys marginal self-correction (via
$\alpha_{\mathrm{eff}}$) and lets duplicated particles re-diversify after resampling (via $\varsigma^2$).
In the FKC scheme it also *is* the guidance strength — $c_{\mathrm{FKC}}=\varsigma^2\nabla\rho/2$ is not a
free dial, choosing churn chooses guidance. It also sets weight variance: the ratio scheme pays
$\sum\beta_s^2\|\nabla r\|^2\varsigma_i^2$ between resamples (largely cancelled by the twist); the FKC
scheme pays no martingale cost but accumulates $O(\Delta\sigma)$ bias instead. Tune $\gamma$ jointly with
`ess_threshold` — more churn tolerates and needs more frequent resampling. $\gamma\to0$ is the collapse
regime (and, under FKC, the no-steering regime), not a safe default; large $\gamma$ is safe for marginals
under exact transport (churn is exact at any $\gamma$), but it costs ESS and can amplify denoising's
numerical transport error — if guidance is too weak, add churn, not gain.
