# The α-family of reverse SDEs

*The α-family SDE below is from [A Simple Derivation of the Reverse SDE](https://ludwigwinkler.github.io/blog/SimpleReverseSDE/);
the derivation, the special-cases table and the attributions are reconstructed here rather than
transcribed, so the route may differ from the post's. The FKC extension in the second half is an
original derivation on top of it — see the flag there.*

## Notation warning

This document uses **two different α's**. They are unrelated:

| symbol | meaning |
| --- | --- |
| $\alpha$ (bare, no subscript) | the **family parameter** — how much noise the reverse process injects |
| $\alpha_t$ (subscripted) | the repo's **schedule interpolant** from `Schedule.get_alpha_t`, i.e. $x_t = \alpha_t x_0 + \sigma_t \varepsilon$ |

Throughout, $f_t(x)$ and $g_t$ are the forward SDE's drift and diffusion coefficient
(`Schedule.forward_drift` / `Schedule.diffusion_coeff`), which the schedule derives from
$(\alpha_t, \sigma_t)$ as $f_t(x) = (\dot\alpha_t/\alpha_t)x$ and $g_t^2 = 2(\dot\sigma_t\sigma_t - \dot\alpha_t\sigma_t^2/\alpha_t)$.

## Setup

Forward process, run in increasing time:

$$dX_t = f_t(X_t)\,dt + g_t\,dW_t.$$

Its marginals $p_t$ obey the Fokker–Planck equation

$$\partial_t p_t = -\langle\nabla,\, p_t f_t\rangle + \tfrac{g_t^2}{2}\Delta p_t.$$

## The family

Reverse time $\tau = T - t$, so $d\tau = -dt$ and $\tau$ runs forward as the sample denoises.
For **every** $\alpha \ge 0$, the process

$$\boxed{\;dX_\tau = \Big[-f_\tau(X_\tau) + \tfrac{1 + \alpha^2}{2}\,g_\tau^2\,\nabla_x \log p_\tau(X_\tau)\Big]d\tau \;+\; \alpha\, g_\tau\, dW_\tau\;}$$

reproduces the forward marginals exactly: $\mathrm{Law}(X_\tau) = p_{T-\tau}$ for all $\tau$, given
$X_0 \sim p_T$. (All quantities are evaluated at the *forward* time $t = T-\tau$; the subscript
$\tau$ is shorthand for that.)

So there is not *one* reverse SDE but a one-parameter family of them, all with identical
time-marginals and differing only in path measure. The score term and the noise term are locked
together: **you cannot change how much noise you inject without changing the score coefficient.**

## Why $(1+\alpha^2)/2$ — the derivation

Write $q_\tau(x) := p_{T-\tau}(x)$ for the reversed marginals. Reversing time in the forward
Fokker–Planck equation flips both signs:

$$\partial_\tau q_\tau = -\partial_t p_t\big|_{t=T-\tau} = \langle\nabla,\, q_\tau f_\tau\rangle - \tfrac{g_\tau^2}{2}\Delta q_\tau .$$

Now *demand* that $q_\tau$ is the marginal law of some Itô process with unknown drift $b_\tau$ and
diffusion coefficient $\alpha g_\tau$. That process's own Fokker–Planck equation is

$$\partial_\tau q_\tau = -\langle\nabla,\, q_\tau b_\tau\rangle + \tfrac{\alpha^2 g_\tau^2}{2}\Delta q_\tau .$$

Equate the two and move the Laplacians to one side:

$$-\langle\nabla,\, q_\tau b_\tau\rangle = \langle\nabla,\, q_\tau f_\tau\rangle - \tfrac{(1+\alpha^2) g_\tau^2}{2}\Delta q_\tau .$$

The only step that needs an identity is turning the Laplacian into a divergence of a *flux*, using
$\nabla q = q\nabla\log q$:

$$\Delta q_\tau = \langle\nabla,\, \nabla q_\tau\rangle = \langle\nabla,\, q_\tau \nabla\log q_\tau\rangle .$$

Substituting and pulling $\langle\nabla,\, q_\tau\,\cdot\,\rangle$ out of everything:

$$-\langle\nabla,\, q_\tau b_\tau\rangle = \Big\langle\nabla,\; q_\tau\Big[f_\tau - \tfrac{(1+\alpha^2)g_\tau^2}{2}\nabla\log q_\tau\Big]\Big\rangle ,$$

which is satisfied by

$$b_\tau = -f_\tau + \tfrac{1+\alpha^2}{2}\,g_\tau^2\,\nabla\log q_\tau . \qquad\blacksquare$$

The bookkeeping is the whole story. Time reversal flips the sign of the diffusion term to
$-\tfrac{g^2}{2}\Delta q$ — an *anti*-diffusion, which is not a legal SDE on its own. To simulate it
you must first pay back that $-\tfrac{g^2}{2}\Delta q$ with score transport, and then, if you *also*
want to inject fresh noise of your own at level $\alpha g$, pay back an additional
$-\tfrac{\alpha^2 g^2}{2}\Delta q$ on top. Hence $\tfrac{1}{2} + \tfrac{\alpha^2}{2}$: one half to
undo the forward diffusion, one $\alpha^2/2$ to undo your own. **Extra diffusion smears the density
outward, so the score term must be strengthened by exactly the same amount to push mass back into
the high-probability region.**

## Special cases

| $\alpha$ | drift score coefficient | noise | name |
| --- | --- | --- | --- |
| $0$ | $\tfrac{1}{2}g_\tau^2$ | none | probability-flow ODE |
| $1$ | $g_\tau^2$ | $g_\tau$ | the standard reverse SDE (Anderson / Song et al.) |
| $\alpha > 1$ | $\tfrac{1+\alpha^2}{2}g_\tau^2$ | $\alpha g_\tau$ | over-noised / "churn"-style sampler |

The $\alpha = 1$ case is the one usually derived in isolation; it is not special in any structural
sense, it is just the choice whose noise level happens to match the forward process.

What actually differs across the family:

- **Marginals** — identical, by construction. Any $\alpha$ is a correct sampler.
- **Path measure / joint law** — different. Only $\alpha = 1$ gives the true time reversal of the
  forward *process* (matching the forward joint law, not just its marginals).
- **Determinism** — $\alpha = 0$ maps noise to data bijectively (invertible, exact likelihoods,
  smooth latents); $\alpha > 0$ does not.
- **Error behaviour** — larger $\alpha$ is self-correcting: injected noise plus the strengthened
  score term contracts back toward $p_t$, so score-approximation and discretization error accumulated
  early gets partially forgotten. $\alpha = 0$ has no such mechanism and integrates its errors.
  Larger $\alpha$ pays for that with more discretization noise per step, so there is a practical
  optimum in between — this is the knob Karras et al. expose as stochastic churn.

## Mapping onto this repo

From here on, write $s$ for the **grid variable the code indexes** — the forward time $t$ of the
derivation above, walked backwards. `reverse_sampling` integrates a *decreasing* $s$ grid, i.e. it
uses $ds < 0$ rather than $d\tau > 0$. (Part 2 reuses this name for the same object; its table maps
all three variables against each other.) Substituting $d\tau = -ds$ and absorbing the sign of the
Wiener increment, which is symmetric:

$$dx_s = \Big[f_s(x_s) - \tfrac{1+\alpha^2}{2}\,g_s^2\,\nabla\log p_s(x_s)\Big]ds \;+\; \alpha\, g_s\, dW_s,
\qquad s: T \to 0 .$$

Note the signs: $+f$ and $-\nabla\log p$, the mirror image of the boxed $d\tau$ form above. That is
what the `drift` / `diffusion` callables must implement:

```python
alpha = 1.0  # family parameter: 0.0 = probability-flow ODE, 1.0 = standard reverse SDE

# `t` here is the decreasing grid variable — `s` in the text, not the sampling-direction time.
def drift(x, t):  # [*N, *B, D]
    g_sq = schedule.diffusion_coeff(t) ** 2  # [*B]
    return schedule.forward_drift(x, t) - 0.5 * (1 + alpha**2) * g_sq.unsqueeze(-1) * gmm.score(x, t)

def diffusion(t):  # [*B]
    return alpha * schedule.diffusion_coeff(t)

# alpha == 0.0 → pass diffusion=None to get the ODE with no noise term
```

This is already in the repo: `notebooks/compare_sampling.py:232-238` implements exactly the boxed
family (its `gam` is $\alpha$), and its ODE branch at `:222-225` is the $\alpha = 0$ member. The
$\alpha = 1$ member is the `f - g**2 * score` / `diffusion=schedule.diffusion_coeff` pairing used in
`notebooks/create_forwardbackward_gif.py:72-78` and, with $f_s = 0$ for VE, in
`notebooks/af3_steering.py:170-194`. The family is the interpolation (and extrapolation) between them.

For the VE schedules ($\alpha_s \equiv 1$, so $f_s = 0$ and $g_s^2 = 2\sigma_s\dot\sigma_s$) this
collapses to

$$dx_s = -\,(1+\alpha^2)\,\sigma_s\dot\sigma_s\,\nabla\log p_s(x_s)\,ds \;+\; \alpha\sqrt{2\sigma_s\dot\sigma_s}\,dW_s .$$

---

# Extension: FKC steering under the α-family, with a time-dependent reward

**Not from the blog post, and not from Skreta et al.** This section reruns the proof of Proposition
D.6 of *Feynman-Kac Correctors in Diffusion* (Skreta et al., 2025; `FeynmanKacCorrectorsFKC_Skreta.pdf`,
Eqs. 258–276) under two simultaneous generalizations:

1. the base sampler is an arbitrary member of the α-family above, not just $\alpha = 1$;
2. the reward is **time-dependent**, $r = r(x,t)$, so the tilt is $\exp(\beta_t\,r(x,t))$.

Equations are numbered $(258')$ and $(264')$–$(276')$ to line up with the paper's $(258)$ and
$(264)$–$(276)$. Setting $\alpha = 1$ and $\partial_t r \equiv 0$ recovers the original at every step.
Derived and checked here; treat it as an extension rather than a citation.

**Symbol map.** Three renamings, all to avoid collisions:

| the paper / Part 1 writes | this section writes | why |
| --- | --- | --- |
| base diffusion coefficient $\sigma_t$ | $g_t$ (and $\tilde g_t$ for the simulator's) | $\sigma_t$ is already the schedule interpolant in the notation table above |
| weight integrand $g_t(x)$ | $\mathrm{w}_t(x)$ | $g_t$ is now the diffusion coefficient |
| Part 1's forward marginals $p_t$ | $q_t$ | $p_t$ here is the **tilted target** $(265')$ |

The last row is the one most likely to trip you up: Part 1's $\nabla\log p_t$ and this section's
$\nabla\log q_t$ are the *same object* — the score of the base model. In Part 2, $p_t \ne q_t$.

On the first row, note that $\sigma_t$ and $g_t$ are related by $g_t^2 = 2\sigma_t\dot\sigma_t$ for a
VE schedule; substituting one for the other misscales the guidance drift by $2\dot\sigma_t/\sigma_t$.

Throughout, $\nabla$ and $\Delta$ are **spatial** operators applied to $x \mapsto r(x,t)$ at fixed
$t$, and $\partial_t r$ is the **partial** time derivative at fixed $x$ — see the last subsection.

**Time convention — this is the opposite of Part 1's implementation form.** Two time variables are in
play and they run in opposite directions:

| variable | direction | Part 1 calls it | used by |
| --- | --- | --- | --- |
| $t$ | **sampling direction**, noise → data | $\tau$ | the paper, and all of $(264')$–$(276')$ below |
| $s = T - t$ | noise at $s = T$, data at $s = 0$ | $s$ (= its forward time $t$) | `reverse_sampling`, which walks a *decreasing* $s$ grid with signed $ds < 0$ |

Note the third column: Part 2 inherits the paper's naming, in which $t$ denotes the *sampling*
direction — the opposite of Part 1's derivation, where $t$ is forward-process time and $\tau = T-t$
is the sampling direction. The grid variable $s$ is the same object in both parts, and is the one
the code indexes.

So $\partial/\partial t = -\,\partial/\partial s$, and the drift written in Part 1 ($+f$, $-\nabla\log p$,
integrated against a decreasing grid) is the negation of the drift written below ($-f$,
$+\nabla\log q$, integrated against $dt$) — recall from the symbol map that Part 1's $p_t$ is this
section's $q_t$, so those two scores are the same object. Consequently $(264')$–$(276')$ are **not** directly
implementable as written: every
$\partial\beta_t/\partial t$ and $\partial r/\partial t$ below is a sampling-direction derivative, and
the corresponding code quantity (`dbeta_dt`, `grad_t`) is a derivative in $s$ with the opposite sign.
The section "Implementing (275′)–(276′)" gives the conversion. The drift and the weight do **not**
flip the same way, and getting that wrong runs the sampler backwards.

## Setup

The base process is the α-family reverse SDE, which by construction has marginals $q_t$:

$$dx_t = v_t^{(\alpha)}(x_t)\,dt + \tilde g_t\,dW_t, \qquad
v_t^{(\alpha)} = -f_t + \tfrac{1+\alpha^2}{2}g_t^2\,\nabla\log q_t, \qquad \tilde g_t = \alpha\,g_t. \tag{258'}$$

Its Fokker–Planck equation — the analogue of (264) — is

$$\frac{\partial q_t(x)}{\partial t} = -\big\langle\nabla,\, q_t(x)\,v_t^{(\alpha)}(x)\big\rangle + \frac{\tilde g_t^2}{2}\Delta q_t(x). \tag{264'}$$

The target is the reward-tilted density with a time-dependent reward,

$$p_t(x) = \frac{q_t(x)\exp\!\big(\beta_t\,r(x,t)\big)}{\displaystyle\int dx\; q_t(x)\exp\!\big(\beta_t\,r(x,t)\big)}. \tag{265'}$$

Note that Steps 1–4 use only two things about the base process: that it satisfies a Fokker–Planck
equation, and with what drift and diffusion coefficient. They therefore go through unchanged for a
*general* pair $(v_t, \tilde g_t)$ — which is why the general statement at the end of Step 4 is
stated that way. The concrete form of $v_t^{(\alpha)}$ is only used in Step 5, and that is exactly
where the α's cancel.

## Step 1 — differentiate $\log p_t$ (Eq. 266′)

$\log p_t(x) = \log q_t(x) + \beta_t\,r(x,t) - \log Z_t$. The tilt exponent now needs the **product
rule**, which is the one and only place time-dependence of $r$ enters:

$$\frac{\partial}{\partial t}\big[\beta_t\,r(x,t)\big] = \frac{\partial\beta_t}{\partial t}\,r(x,t) + \beta_t\,\frac{\partial r(x,t)}{\partial t}.$$

The normalizer contributes an expectation, since
$\partial_t\big(q_t e^{\beta_t r}\big) = q_t e^{\beta_t r}\big[\partial_t\log q_t + \dot\beta_t r + \beta_t\partial_t r\big]$
and dividing by $Z_t$ turns $q_t e^{\beta_t r}/Z_t$ into $p_t$. Hence

$$\begin{aligned}
\frac{\partial}{\partial t}\log p_t(x)
&= \left(\frac{\partial \log q_t(x)}{\partial t} + \frac{\partial\beta_t}{\partial t}\, r(x,t) + \beta_t\frac{\partial r(x,t)}{\partial t}\right) \\[4pt]
&\quad - \int dx\; p_t(x)\left(\frac{\partial \log q_t(x)}{\partial t} + \frac{\partial\beta_t}{\partial t}\, r(x,t) + \beta_t\frac{\partial r(x,t)}{\partial t}\right).
\end{aligned} \tag{266'}$$

Compared with (266), the new $\beta_t\partial_t r$ term appears inside *both* brackets. The
$-\mathbb{E}_{p_t}[\cdot]$ structure is what keeps the Feynman–Kac PDE normalized, so this term will
ride along into the weight with its mean subtracted, exactly like $\dot\beta_t r$.

## Step 2 — expand $\partial_t \log q_t$ and re-express in $p_t$ (Eqs. 267′–268′)

Dividing (264′) by $q_t$ and using $\Delta q/q = \Delta\log q + \|\nabla\log q\|^2$:

$$\frac{\partial \log q_t}{\partial t} = -\big\langle\nabla, v_t^{(\alpha)}\big\rangle - \big\langle\nabla\log q_t, v_t^{(\alpha)}\big\rangle + \frac{\tilde g_t^2}{2}\Delta\log q_t + \frac{\tilde g_t^2}{2}\big\|\nabla\log q_t\big\|^2. \tag{267'}$$

Now the substitution. **This is the step where nothing changes**: $Z_t$ is constant in $x$, and
spatial derivatives are taken at fixed $t$, so the relations are the same as for a static reward,

$$\nabla\log q_t = \nabla\log p_t - \beta_t\nabla r(x,t), \qquad \Delta\log q_t = \Delta\log p_t - \beta_t\Delta r(x,t).$$

Substitute these into the three score-carrying terms of $(267')$, one at a time:

$$-\big\langle\nabla\log q_t,\, v_t^{(\alpha)}\big\rangle
= -\big\langle\nabla\log p_t,\, v_t^{(\alpha)}\big\rangle
\;+\; \beta_t\big\langle\nabla r,\, v_t^{(\alpha)}\big\rangle,$$

$$\frac{\tilde g_t^2}{2}\Delta\log q_t
= \frac{\tilde g_t^2}{2}\Delta\log p_t
\;-\; \beta_t\frac{\tilde g_t^2}{2}\Delta r,$$

$$\frac{\tilde g_t^2}{2}\big\|\nabla\log q_t\big\|^2
= \frac{\tilde g_t^2}{2}\big\|\nabla\log p_t\big\|^2
\;-\; \tilde g_t^2\,\beta_t\big\langle\nabla\log p_t,\, \nabla r\big\rangle
\;+\; \frac{\tilde g_t^2}{2}\beta_t^2\big\|\nabla r\big\|^2 .$$

The first piece of each line is the transport block written in $p_t$; the rest is leftover. In the
leftover, undo the substitution once more in the cross term — $\nabla\log p_t = \nabla\log q_t + \beta_t\nabla r$ —
so that the weight is expressed in the score you actually have a model for:

$$-\tilde g_t^2\,\beta_t\big\langle\nabla\log p_t, \nabla r\big\rangle + \frac{\tilde g_t^2}{2}\beta_t^2\big\|\nabla r\big\|^2
= -\tilde g_t^2\,\beta_t\big\langle\nabla\log q_t, \nabla r\big\rangle - \frac{\tilde g_t^2}{2}\beta_t^2\big\|\nabla r\big\|^2 .$$

The three surviving leftover terms
$\beta_t\langle\nabla r, v_t^{(\alpha)}\rangle$,
$-\tilde g_t^2\beta_t\langle\nabla r, \nabla\log q_t\rangle$ and
$-\tfrac{\tilde g_t^2}{2}\beta_t^2\|\nabla r\|^2$
share the factor $\beta_t\nabla r$ and package into a single inner product:

$$\begin{aligned}
\frac{\partial \log q_t}{\partial t}
&= \underbrace{-\big\langle\nabla, v_t^{(\alpha)}\big\rangle - \big\langle\nabla\log p_t, v_t^{(\alpha)}\big\rangle + \frac{\tilde g_t^2}{2}\Delta\log p_t + \frac{\tilde g_t^2}{2}\big\|\nabla\log p_t\big\|^2}_{\text{transport on } p_t} \\[4pt]
&\quad + \Big\langle \beta_t\nabla r,\;\; v_t^{(\alpha)} - \tilde g_t^2\,\nabla\log q_t - \frac{\tilde g_t^2}{2}\beta_t\nabla r\Big\rangle
\;-\; \beta_t\frac{\tilde g_t^2}{2}\Delta r .
\end{aligned} \tag{268'}$$

Identical to (268) under $\sigma_t \mapsto \tilde g_t$, $v_t \mapsto v_t^{(\alpha)}$ — no term here
knows about the time-dependence of $r$.

## Step 3 — read off the Feynman–Kac PDE (Eqs. 269′–270′)

Substituting (268′) into (266′), the underbraced block runs the log-identity backwards into the
transport operator on $p_t$ with the *same* drift and *same* diffusion coefficient as the base
process. Everything else is weight:

$$\frac{\partial p_t(x)}{\partial t} = -\big\langle\nabla,\, p_t(x) v_t^{(\alpha)}(x)\big\rangle + \frac{\tilde g_t^2}{2}\Delta p_t(x) + p_t(x)\Big(\mathrm{w}_t(x) - \mathbb{E}_{p_t(x)}\big[\mathrm{w}_t(x)\big]\Big), \tag{269'}$$

$$\begin{aligned}
\mathrm{w}_t(x)
&= \Big\langle \beta_t\nabla r,\;\; v_t^{(\alpha)} - \tilde g_t^2\,\nabla\log q_t - \frac{\tilde g_t^2}{2}\beta_t\nabla r\Big\rangle \\[4pt]
&\quad - \beta_t\frac{\tilde g_t^2}{2}\Delta r
\;+\; \frac{\partial\beta_t}{\partial t}\, r
\;+\; \boxed{\;\beta_t\frac{\partial r}{\partial t}\;}.
\end{aligned} \tag{270'}$$

The boxed term is the only addition to (270).

## Step 4 — add the free guidance drift $a\nabla r$ (Eqs. 271′–272′)

Split the transport term and convert the added-back divergence into weight, using
$\langle\nabla, p_t\,a\nabla r\rangle = p_t\big[a\Delta r + a\langle\nabla\log p_t, \nabla r\rangle\big]$:

$$\begin{aligned}
\frac{\partial p_t(x)}{\partial t}
&= -\Big\langle\nabla,\; p_t(x)\big(v_t^{(\alpha)}(x) + a\nabla r(x,t)\big)\Big\rangle
\;+\; \frac{\tilde g_t^2}{2}\Delta p_t(x) \\[4pt]
&\quad + p_t(x)\Big(\mathrm{w}_t(x) - \mathbb{E}_{p_t(x)}\big[\mathrm{w}_t(x)\big]\Big),
\end{aligned} \tag{271'}$$

$$\begin{aligned}
\mathrm{w}_t(x)
&= a\Delta r + a\big\langle\nabla\log p_t,\, \nabla r\big\rangle \\[4pt]
&\quad + \Big\langle \beta_t\nabla r,\;\; v_t^{(\alpha)} - \tilde g_t^2\,\nabla\log q_t - \frac{\tilde g_t^2}{2}\beta_t\nabla r\Big\rangle \\[4pt]
&\quad - \beta_t\frac{\tilde g_t^2}{2}\Delta r
\;+\; \frac{\partial\beta_t}{\partial t}\, r
\;+\; \beta_t\frac{\partial r}{\partial t}.
\end{aligned} \tag{272'}$$

$a$ is free. Because $\partial_t r$ carries no spatial derivative, it takes no part in this
continuity ↔ reweighting exchange; it simply sits in the weight.

**General statement** (the analogue of the paper's Eqs. 259–261): for any $a$ and any base pair
$(v_t, \tilde g_t)$ whose marginals are $q_t$, the tilted marginals $(265')$ are simulated by

$$dx_t = \big(v_t(x_t) + a\nabla r(x_t,t)\big)\,dt + \tilde g_t\,dW_t,$$

$$\begin{aligned}
dw_t = \Bigg[\;
&\Big\langle\nabla r,\;\; \beta_t\Big(v_t - \tilde g_t^2\,\nabla\log q_t - \frac{\tilde g_t^2}{2}\beta_t\nabla r\Big) + a\big(\nabla\log q_t + \beta_t\nabla r\big)\Big\rangle \\[4pt]
&+ \Big(a - \beta_t\frac{\tilde g_t^2}{2}\Big)\Delta r
\;+\; \frac{\partial\beta_t}{\partial t}\, r
\;+\; \beta_t\frac{\partial r}{\partial t}
\;\Bigg]dt.
\end{aligned}$$

## Step 5 — specialize: $v_t^{(\alpha)}$, $\tilde g_t = \alpha g_t$, and $a = \beta_t\tilde g_t^2/2$ (Eqs. 273′–274′)

Choose $a$ to kill the reward Laplacian, exactly as the paper does:

$$a = \beta_t\frac{\tilde g_t^2}{2} = \beta_t\frac{\alpha^2 g_t^2}{2} \quad\Longrightarrow\quad \underbrace{a\Delta r}_{\text{from the drift}} - \underbrace{\beta_t\frac{\tilde g_t^2}{2}\Delta r}_{\text{from }(270')} = 0 .$$

So the **guidance drift scales with $\alpha^2$**: a probability-flow ODE sampler ($\alpha = 0$) gets
no guidance drift at all, and everything must live in the weight. What remains is

$$\begin{aligned}
\mathrm{w}_t(x)
&= \beta_t\frac{\alpha^2 g_t^2}{2}\big\langle\nabla r,\, \nabla\log p_t\big\rangle \\[4pt]
&\quad + \Big\langle\beta_t\nabla r,\;\; v_t^{(\alpha)} - \alpha^2 g_t^2\,\nabla\log q_t - \frac{\alpha^2 g_t^2}{2}\beta_t\nabla r\Big\rangle \\[4pt]
&\quad + \frac{\partial\beta_t}{\partial t}\, r
\;+\; \beta_t\frac{\partial r}{\partial t}.
\end{aligned} \tag{273'}$$

Expand $\nabla\log p_t = \nabla\log q_t + \beta_t\nabla r$ and collect. Two cancellations:

- **The $\beta_t^2\|\nabla r\|^2$ terms cancel**, as in the paper:
  $+\beta_t\tfrac{\alpha^2 g_t^2}{2}\beta_t\|\nabla r\|^2$ from the drift term against
  $-\tfrac{\alpha^2 g_t^2}{2}\beta_t^2\|\nabla r\|^2$ from the inner product.
- **The α's cancel in the score terms.** This is what is new relative to the paper, where
  $v_t - \sigma_t^2\nabla\log q_t = -f_t$ collapses immediately. Here it does not — instead
  $v_t^{(\alpha)} - \alpha^2 g_t^2\nabla\log q_t = -f_t + \tfrac{1-\alpha^2}{2}g_t^2\nabla\log q_t$,
  and that residual score piece combines with the $\tfrac{\alpha^2 g_t^2}{2}\langle\nabla r,\nabla\log q_t\rangle$
  coming from the guidance drift:

$$\underbrace{\frac{\alpha^2}{2}}_{\text{from } a\nabla\log q_t} \;+\; \underbrace{\frac{1-\alpha^2}{2}}_{\text{from } v_t^{(\alpha)} - \tilde g_t^2\nabla\log q_t} \;=\; \frac{1}{2}\,,$$

or equivalently, in one line,

$$v_t^{(\alpha)} - \frac{\alpha^2 g_t^2}{2}\nabla\log q_t \;=\; -f_t + \frac{g_t^2}{2}\nabla\log q_t .$$

Hence

$$\begin{aligned}
\mathrm{w}_t(x)
&= \frac{\partial\beta_t}{\partial t}\,r(x,t)
\;+\; \beta_t\frac{\partial r(x,t)}{\partial t} \\[4pt]
&\quad + \Big\langle\beta_t\nabla r(x,t),\;\; \frac{g_t^2}{2}\nabla\log q_t(x) - f_t(x)\Big\rangle .
\end{aligned} \tag{274'}$$

## Result (Eqs. 275′–276′)

$$\boxed{\;
\begin{aligned}
dx_t = \Big(&-f_t(x_t) + \frac{1+\alpha^2}{2}g_t^2\,\nabla\log q_t(x_t) \\
&+ \beta_t\frac{\alpha^2 g_t^2}{2}\nabla r(x_t,t)\Big)dt \;+\; \alpha\,g_t\,dW_t
\end{aligned}
\;} \tag{275'}$$

$$\boxed{\;
\begin{aligned}
dw_t = \Bigg[\;&\frac{\partial\beta_t}{\partial t}\,r(x_t,t) \;+\; \beta_t\frac{\partial r(x_t,t)}{\partial t} \\
&+ \Big\langle\beta_t\nabla r(x_t,t),\;\; \frac{g_t^2}{2}\nabla\log q_t(x_t) - f_t(x_t)\Big\rangle\;\Bigg]dt
\end{aligned}
\;} \tag{276'}$$

## Reading the result

**The weight is α-independent in form.** The $\tfrac{g_t^2}{2}\nabla\log q_t - f_t$ vector field in
$(276')$ is the *probability-flow* velocity — the $\alpha = 0$ member — no matter which $\alpha$ you
actually simulate. All of the α-dependence sits in the drift $(275')$: as $\alpha$ grows, more of the
reward correction is carried by the guidance term $\beta_t\tfrac{\alpha^2 g_t^2}{2}\nabla r$ and by
the strengthened score, and none of it changes the weight *expression*. This is not the same as the
weights being numerically identical — the trajectories $x_t$ differ across $\alpha$, so the same
formula is evaluated at different points and its variance along a run does depend on $\alpha$.

**Both schedule terms are one object.** Define the full tilt exponent $\rho(x,t) := \beta_t\,r(x,t)$.
Then $\nabla\rho = \beta_t\nabla r$ and $\partial_t\rho = \dot\beta_t r + \beta_t\partial_t r$, so
$(276')$ collapses to

$$\begin{aligned}
dw_t = \Bigg[\;&\partial_t \rho(x_t,t) \\
&+ \Big\langle\nabla\rho(x_t,t),\;\; \frac{g_t^2}{2}\nabla\log q_t(x_t) - f_t(x_t)\Big\rangle\;\Bigg]dt,
\end{aligned}$$

and $(275')$ to

$$\begin{aligned}
dx_t = \Big(&-f_t + \frac{1+\alpha^2}{2}g_t^2\,\nabla\log q_t \\
&+ \frac{\alpha^2 g_t^2}{2}\nabla\rho\Big)dt \;+\; \alpha\,g_t\,dW_t .
\end{aligned}$$

D.6 is the same formula with $\rho(x,t) = \beta_t r(x)$: the paper already handled an arbitrary
time-dependent *scalar* $\beta_t$, and letting the *spatial profile* drift in time adds exactly one
more contribution to the same $\partial_t(\text{exponent})$ slot.

Consistency checks:

- $\alpha = 1$, $\partial_t r \equiv 0$ recovers Eqs. (275)–(276) of D.6 exactly.
- $\alpha = 1$, $r = r(x,t)$ recovers the time-dependent-reward result at the end of
  [`fkc_steering.md`](fkc_steering.md). Careful: that document *also* tags its equations
  $(270')$, $(275')$, $(276')$, for its own $\alpha = 1$ generalization — so those labels denote
  different equations in the two files.
- $\alpha = 0$ gives the probability-flow ODE with **no** guidance drift and the entire correction in
  the weight: $\mathrm{w}_t = \partial_t\rho + \langle\nabla\rho, v_t^{(0)}\rangle$, which is $(270')$
  read at $\tilde g_t = 0$ — the "steer without touching the drift at all" regime, at its highest
  weight variance.

**Choosing $\alpha$ in practice.** $\alpha$ trades weight variance against drift strength. Large
$\alpha$ moves the steering into the drift, where it costs nothing in ESS, at the price of a noisier,
more heavily discretized trajectory. Small $\alpha$ keeps the sampler close to deterministic but pushes
all steering into the importance weights, so ESS collapses faster — and there is a second, sharper
failure at the $\alpha \to 0$ end that is specific to SMC: with no injected noise the dynamics are
deterministic, so the duplicated particles produced by `_systematic_resample` follow *identical*
trajectories from that point on and never separate again. Resampling does not merely work harder
there, it stops diversifying at all — a resample can only shrink the number of distinct particles,
never restore it. Some $\alpha > 0$ is what makes `steered_reverse_sampling` a working particle filter rather
than a weighted ODE ensemble that collapses at the first resample.

## Implementing (275′)–(276′)

`reverse_sampling` and `steered_reverse_sampling` walk a decreasing $s$ grid and pass a **signed
$ds < 0$**, where $s = T - t$ as set out in the time-convention note. Converting $(275')$–$(276')$ is
**not** a uniform sign flip — the drift and the weight behave differently.

**Drift — negate the whole bracket.** $dx = B\,dt = -B\,ds$, so

$$\begin{aligned}
\texttt{drift}(x,s) = \;&f_s(x) - \frac{1+\alpha^2}{2}g_s^2\,\nabla\log q_s(x) \\
&- \beta_s\frac{\alpha^2 g_s^2}{2}\nabla r(x,s).
\end{aligned}$$

**Weight — only the explicit time-derivative terms flip.** The two schedule terms in $(276')$ are
derivatives in $t$, so re-expressing them in the code's variable gives
$\partial\beta_t/\partial t = -\,d\beta_s/ds$ and $\partial_t r = -\,\partial_s r$. The alignment inner
product contains **no** time derivative, so it is invariant. Accumulating against $|ds| = dt$:

$$\begin{aligned}
\texttt{weight\_update}(x,s,ds) = \Bigg[\;&-\frac{d\beta_s}{ds}\,r(x,s) \;-\; \beta_s\frac{\partial r(x,s)}{\partial s} \\
&+ \Big\langle\beta_s\nabla r(x,s),\;\; \frac{g_s^2}{2}\nabla\log q_s(x) - f_s(x)\Big\rangle\;\Bigg]\,|ds| .
\end{aligned}$$

Only two of the three terms pick up a minus sign. A blanket negation of the integrand — the natural
mistake, since the drift *is* negated wholesale — would put the alignment term the wrong way round.

`notebooks/karras_terminal_variance_steering.py` implements exactly this:

```python
def guided_drift(x_, t_):                              # :261-263
    ...
    return -(g**2) * sc - beta * (g**2 / 2) * grad_x

def weight_update(x_, t_):                             # :266-277
    ...
    integrand = -dbeta * rv - beta * grad_t + beta * grad_x * (g**2 / 2) * sc
    return integrand.squeeze(-1).squeeze(-1)
```

This is the $\alpha = 1$ member, with $f_s = 0$ because the Karras/VE schedules have $\alpha_s \equiv 1$.
The active `beta_fn(t) = (1 - t)**2` and `dbeta_dt(t) = -2 * (1 - t)` are functions of the *grid*
variable $s$ (tilt off at the noise end $s = 1$, full at the data end $s = 0$), so `dbeta` is
$d\beta_s/ds$ and the leading minus sign is the flip above, not a sign error. (The notebook defines
`beta_fn`/`dbeta_dt` twice — a cosine ramp first, then this quadratic one shadowing it. The second
definition is the one that runs.)

One correction to that notebook: its comment claims "the integrand below is the negation of the
eq-(276′) one." That is not right, and the code is better than the comment — the integrand is
$(276')$ re-expressed in $s$-derivatives, in which only the two schedule terms flip sign while the
alignment term is unchanged. A literal negation would flip the alignment term too.

## Partial, not material, time derivative

$\partial_t r(x_t, t)$ differentiates only the explicit $t$-argument, holding $x_t$ fixed. It is
*not* the total derivative along the trajectory,

$$\frac{d}{dt}r(x_t,t) = \frac{\partial r}{\partial t} + \big\langle\nabla r, \dot x_t\big\rangle,$$

because the convective part is already accounted for twice over — it lives in the guidance drift
$\beta_t\tfrac{\alpha^2 g_t^2}{2}\nabla r$ of $(275')$ and in the inner-product term of $(276')$.
Including it would double-count. In practice: if $r$ is closed-form in $t$, differentiate
analytically or with autograd **w.r.t. the time argument only**; if $r$ is only available on
snapshots $r(\cdot, t_k)$, use $\big(r(x_t,t_{k+1}) - r(x_t,t_k)\big)/\Delta t$ at the *fixed* sample
location $x_t$.

A worked instance is `notebooks/karras_terminal_variance_steering.py`. Its reward
`r(x0) = -0.5 * (x0 - target_c)**2 / target_s**2` (where `target_s` is a reward width, unrelated to
the time variable $s$) is a fixed function of the *denoised estimate* $\hat x_0(x_t,t)$. But
$\hat x_0$ itself depends on $t$, so as a function of $(x_t, t)$ the reward is time-dependent
**through the denoiser**, and $\partial_t r$ is not zero:

$$\partial_t r = \big\langle\nabla_{\hat x_0} r,\; \partial_t \hat x_0(x_t, t)\big\rangle \quad\text{at fixed } x_t .$$

The notebook obtains it as `grad_t` by backpropagating through the unrolled ODE denoiser with respect
to its time argument, alongside `grad_x` for the spatial gradient — one `torch.autograd.grad` call
against both leaves. That is the cleanest available estimator of $\partial_t r$: no finite
differencing, and no risk of picking up the convective term, because $x_t$ is held fixed as a
separate leaf.
