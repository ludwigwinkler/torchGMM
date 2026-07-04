# Proof Trace: Proposition D.6 (Reward-tilted SDE)

*From "Feynman-Kac Correctors in Diffusion: Annealing, Guidance, and Product of Experts" (Skreta et al., 2025), Appendix D.*

## The big picture first

Proposition D.6 answers: I have a pretrained diffusion model whose marginals are $q_t(x)$, and at inference time I want to bias generation toward high values of some reward $r(x)$, i.e. sample from

$$p_t(x) \;=\; \frac{q_t(x)\,\exp(\beta_t r(x))}{\int q_t(x)\exp(\beta_t r(x))\,dx}.$$

The thing that makes this case *different* from Product-of-Experts is stated explicitly in the paper: the reward factor $\exp(\beta_t r(x))$ is **not** assumed to evolve under the diffusion. It is a static-in-$x$ multiplicative tilt (only its temperature $\beta_t$ is time-dependent). So unlike $q^1_t q^2_t$, where both factors carry their own Fokker–Planck dynamics, here only $q_t$ diffuses. That asymmetry is exactly what produces the specific reward-gradient and reward-Laplacian terms in the weights.

The whole proof is one recipe applied mechanically: *write the time-derivative of $\log p_t$, recognize which pieces reassemble into a transport operator on $p_t$, and dump everything left over into the reweighting term $g_t$.*

## The one identity everything rests on

For any density obeying a Fokker–Planck PDE $\partial_t p = -\langle\nabla, p\,v\rangle + \tfrac{\sigma^2}{2}\Delta p$, dividing by $p$ gives the "log form":

$$\partial_t \log p \;=\; \underbrace{-\langle\nabla, v\rangle - \langle\nabla\log p,\, v\rangle}_{\text{continuity}} \;+\; \underbrace{\tfrac{\sigma^2}{2}\Delta\log p + \tfrac{\sigma^2}{2}\|\nabla\log p\|^2}_{\text{diffusion}}. \tag{$\star$}$$

The continuity part is just the product rule on $\langle\nabla, p v\rangle$. The diffusion part uses

$$\frac{\Delta p}{p} = \Delta\log p + \|\nabla\log p\|^2,$$

which follows from $\nabla\log p = \nabla p/p$ and $\Delta\log p = \tfrac{\Delta p}{p} - \|\nabla\log p\|^2$. Read **forward**, $(\star)$ turns a PDE into log-derivative terms; read **backward**, a collection of log-derivative terms in exactly this shape collapses back into a transport operator. The proof uses it both ways.

## Step 1 — Differentiate the log of the tilted density (Eq. 266)

Take $\log p_t = \log q_t + \beta_t r - \log Z_t$ with $Z_t = \int q_t e^{\beta_t r}\,dx$. Then

$$\partial_t\log p_t = \partial_t\log q_t + \dot\beta_t\, r - \partial_t\log Z_t.$$

The normalizer derivative is itself an expectation, because $\partial_t(q_t e^{\beta_t r}) = q_t e^{\beta_t r}\big[\partial_t\log q_t + \dot\beta_t r\big]$, so

$$\partial_t\log Z_t = \int p_t\big[\partial_t\log q_t + \dot\beta_t r\big]\,dx = \mathbb{E}_{p_t}\!\big[\partial_t\log q_t + \dot\beta_t r\big].$$

Hence Eq. (266):

$$\partial_t\log p_t = \big(\partial_t\log q_t + \dot\beta_t r\big) \;-\; \mathbb{E}_{p_t}\!\big[\partial_t\log q_t + \dot\beta_t r\big].$$

That subtracted expectation is precisely the $-\mathbb{E}_{p_t}[g_t]$ term that keeps the FK-PDE normalized. So from here on, "everything that is not transport becomes $g_t$, with its mean subtracted automatically."

## Step 2 — Expand $\partial_t\log q_t$ and re-express in terms of $p_t$ (Eqs. 267–268)

Apply $(\star)$ to the base model $q_t$ (drift $v_t$):

$$\partial_t\log q_t = -\langle\nabla, v_t\rangle - \langle\nabla\log q_t, v_t\rangle + \tfrac{\sigma_t^2}{2}\Delta\log q_t + \tfrac{\sigma_t^2}{2}\|\nabla\log q_t\|^2. \tag{267}$$

Now the key substitution. Since $Z_t$ is constant in $x$,

$$\nabla\log q_t = \nabla\log p_t - \beta_t\nabla r, \qquad \Delta\log q_t = \Delta\log p_t - \beta_t\Delta r.$$

Plug these in. The four terms split into "transport written in $p_t$" plus a remainder. After collecting (each cross term checks out), you get Eq. (268):

$$\partial_t\log q_t = \underbrace{-\langle\nabla, v_t\rangle - \langle\nabla\log p_t, v_t\rangle + \tfrac{\sigma_t^2}{2}\Delta\log p_t + \tfrac{\sigma_t^2}{2}\|\nabla\log p_t\|^2}_{\text{reassembles into transport on } p_t \text{ via }(\star)} \;+\;\underbrace{\Big\langle \beta_t\nabla r,\; v_t - \sigma_t^2\nabla\log q_t - \tfrac{\sigma_t^2}{2}\beta_t\nabla r\Big\rangle - \beta_t\tfrac{\sigma_t^2}{2}\Delta r}_{\text{the leftover}}.$$

One presentational choice worth flagging: the leftover keeps $\nabla\log q_t$ (not $\nabla\log p_t$) inside the inner product. That is deliberate — $\nabla\log q_t$ is the thing you actually have a trained score model for, so the weight should be expressed in computable quantities. If you expand the packaged inner product it reads

$$\beta_t\langle\nabla r, v_t\rangle - \sigma_t^2\beta_t\langle\nabla r,\nabla\log q_t\rangle - \tfrac{\sigma_t^2}{2}\beta_t^2\|\nabla r\|^2.$$

## Step 3 — Read off the FK-PDE: drift unchanged, reward goes into weights (Eqs. 269–270)

Substituting (268) into (266): the underbraced block, by $(\star)$ run backward, is exactly $\tfrac{1}{p_t}\big[-\langle\nabla, p_t v_t\rangle + \tfrac{\sigma_t^2}{2}\Delta p_t\big]$ — the original transport operator with the *same* drift $v_t$ and *same* noise $\sigma_t$. Everything else is the weight:

$$\partial_t p_t = -\langle\nabla, p_t v_t\rangle + \tfrac{\sigma_t^2}{2}\Delta p_t + p_t\big(g_t - \mathbb{E}_{p_t}[g_t]\big),$$

$$g_t(x) = \Big\langle \beta_t\nabla r,\, v_t - \sigma_t^2\nabla\log q_t - \tfrac{\sigma_t^2}{2}\beta_t\nabla r\Big\rangle - \beta_t\tfrac{\sigma_t^2}{2}\Delta r + \dot\beta_t\, r. \tag{270}$$

Notice what this already says: you can sample from the reward-tilted target **without touching the drift at all** — just run the base SDE and accumulate these weights. The catch is that pushing the entire reward signal into the weights makes them high-variance.

## Step 4 — Trade some weight for drift (Eqs. 271–272)

So they add a free drift term $a\nabla r$ and compensate in the weights. This is the continuity ↔ reweighting conversion rule. Algebraically, split the transport:

$$-\langle\nabla, p_t v_t\rangle = -\langle\nabla, p_t(v_t + a\nabla r)\rangle + \langle\nabla, p_t\, a\nabla r\rangle,$$

and the added-back divergence becomes a weight via $\langle\nabla, p_t\,a\nabla r\rangle = p_t\big[a\Delta r + a\langle\nabla\log p_t,\nabla r\rangle\big]$. So the drift gains $a\nabla r$ and the weight gains exactly that:

$$dx_t = (v_t + a\nabla r)\,dt + \sigma_t dW_t,\qquad g_t \mathrel{+}= a\Delta r + a\langle\nabla\log p_t,\nabla r\rangle. \tag{271–272}$$

$a$ is still a free knob at this point.

## Step 5 — Specialize and pick the magic $a$ (Eqs. 273–276)

Now insert the actual reverse-time diffusion drift $v_t = -f_t + \sigma_t^2\nabla\log q_t$ and choose

$$a = \beta_t\tfrac{\sigma_t^2}{2}.$$

Two cancellations make this the right choice.

**First**, inside the packaged inner product, $v_t - \sigma_t^2\nabla\log q_t = -f_t$, so that whole term collapses to $\langle\beta_t\nabla r,\, -f_t - \tfrac{\sigma_t^2}{2}\beta_t\nabla r\rangle$.

**Second** — and this is the payoff — the reward **Laplacian cancels**:

$$\underbrace{a\,\Delta r}_{\text{from drift, }=\,\frac{\beta_t\sigma_t^2}{2}\Delta r} \;-\; \underbrace{\beta_t\tfrac{\sigma_t^2}{2}\Delta r}_{\text{from }(270)} \;=\; 0.$$

$\Delta r$ is the expensive term (a divergence of a gradient you would otherwise have to estimate), and the choice of $a$ deletes it. Cleaning up the remaining gradient terms using $\nabla\log p_t = \nabla\log q_t + \beta_t\nabla r$ (the $\beta_t^2\|\nabla r\|^2$ pieces cancel against each other) leaves the final clean result:

$$dx_t = \Big(-f_t(x_t) + \sigma_t^2\nabla\log q_t(x_t) + \beta_t\tfrac{\sigma_t^2}{2}\nabla r(x_t)\Big)dt + \sigma_t dW_t, \tag{275}$$

$$dw_t = \left[\; \dot\beta_t\, r(x_t) \;+\; \Big\langle \beta_t\nabla r(x_t),\; \tfrac{\sigma_t^2}{2}\nabla\log q_t(x_t) - f_t(x_t)\Big\rangle \;\right]dt. \tag{276}$$

## How to read the result

The drift is exactly what you would write down heuristically — the base denoising drift plus a classifier-guidance-style push $\beta_t\tfrac{\sigma_t^2}{2}\nabla r$ up the reward gradient. The contribution of D.6 is that this heuristic drift, *on its own*, does not sample the prescribed tilted marginals; the weight $dw_t$ is the exact correction that makes it consistent.

The weight has a transparent interpretation:

- $\dot\beta_t\, r$ — pure annealing-schedule bookkeeping: when you ramp the tilt strength, particles in high-reward regions get up-weighted in proportion to how fast you are turning up $\beta_t$.
- $\big\langle \beta_t\nabla r,\; \tfrac{\sigma_t^2}{2}\nabla\log q_t - f_t\big\rangle$ — a weight that grows when the reward gradient **aligns with the base model's own vector field** $\tfrac{\sigma_t^2}{2}\nabla\log q_t - f_t$. Intuitively: a particle moving "downhill in $q_t$" in a direction that also increases reward is doing exactly what the tilted target wants, so it earns weight; a particle whose reward-improving direction fights the model's natural flow is penalized.

The broader lesson, consistent with the rest of the appendix: there is a continuum of valid simulators for the same target (parameterized by $a$), all differing only in how much of the correction lives in the drift versus the weights. $a = \beta_t\sigma_t^2/2$ is singled out purely for computational convenience — it gives the familiar guidance drift and removes the Laplacian.

## Contrast with Product of Experts (D.2)

The "reward does not diffuse" distinction is concrete here. In PoE, both factors $q^1_t, q^2_t$ carry diffusion, which is why their diffusion equations interact to produce a cross term in the weight of Prop. D.2: $g_t(x) = \sigma_t^2\langle\nabla\log q^1_t,\nabla\log q^2_t\rangle - \langle f_t, \nabla\log q^1_t + \nabla\log q^2_t\rangle$ — note the cross term carries a **positive** $\sigma_t^2$ coefficient in the final result (the diffusion-only lemma, Prop. C.8, contributes it with a *negative* sign in isolation, but combining it with the continuity-equation cross term from Prop. C.7 flips the net sign to positive). In the reward-tilted case, the reward factor contributes a $\Delta r$ (and reward-gradient inner products) instead of a score–score cross term — precisely because $\exp(\beta_t r)$ is not propagated through the Fokker–Planck dynamics.

---

# Generalization: time-dependent reward $r(x, t)$

**Note: everything below this point is *not* in the paper.** Skreta et al. only ever
treat time-dependence of the temperature schedule $\beta_t$ (their Prop. C.6); Prop.
D.6 itself uses a reward $r(x)$ static in $x$'s time-dependence, and their Conclusion
says so explicitly ("...allows for the use of reward models (Prop. D.6) and for a
time-dependent annealing schedule $\beta_t$ (Prop. C.6)" — no mention of a spatially-
evaluated, explicitly time-dependent $r(x,t)$). What follows is an original extension
of the same proof technique to that case, not a transcription of a paper result. It
has been independently re-derived and checked (the extra $\beta_t\partial_t r$ term is
purely additive/temporal and doesn't interact with any spatial cancellation, so the
Laplacian and $\beta_t^2\|\nabla r\|^2$ cancellations below still go through), but treat
it accordingly.

Now suppose the reward carries explicit time dependence, $r = r(x, t)$, so the tilt is $\exp(\beta_t\, r(x,t))$ and the target is

$$p_t(x) \;=\; \frac{q_t(x)\,\exp(\beta_t\, r(x,t))}{Z_t}, \qquad Z_t = \int q_t(x)\exp(\beta_t\, r(x,t))\,dx.$$

**The headline:** every spatial manipulation in the original proof is evaluated at a *fixed* time slice, so none of it changes. The only thing that changes is the time-derivative of the tilt exponent, which now picks up an extra partial-time term by the product rule. That single extra term propagates, unchanged, all the way into the final weight.

## What is and is not affected

Spatial derivatives are taken at fixed $t$, so the gradient relation is identical to before:

$$\nabla\log p_t = \nabla\log q_t + \beta_t\nabla r, \qquad \Delta\log q_t = \Delta\log p_t - \beta_t\Delta r,$$

where $\nabla r \equiv \nabla_x r(x,t)$ and $\Delta r \equiv \Delta_x r(x,t)$ are spatial operators applied to the map $x \mapsto r(x,t)$. Consequently **Step 2 (the expansion of $\partial_t\log q_t$ and the score substitution) and Step 4 (adding the drift $a\nabla r$ with its compensating weight) are word-for-word unchanged.** Only Steps 1, 3, and 5 acquire one new term.

## Step 1' — Differentiate the log of the tilted density

$$\log p_t(x) = \log q_t(x) + \beta_t\, r(x,t) - \log Z_t.$$

Differentiating in $t$ at fixed $x$, the tilt exponent now needs the product rule:

$$\partial_t\big[\beta_t\, r(x,t)\big] = \dot\beta_t\, r(x,t) + \beta_t\,\partial_t r(x,t),$$

where $\partial_t r(x,t)$ is the **partial (explicit) time derivative at fixed $x$** — *not* the material derivative along a trajectory. The normalizer derivative carries the same extra term inside the expectation:

$$\partial_t\log Z_t = \mathbb{E}_{p_t}\!\Big[\partial_t\log q_t + \dot\beta_t\, r + \beta_t\,\partial_t r\Big],$$

which follows from $\partial_t\big(q_t e^{\beta_t r}\big) = q_t e^{\beta_t r}\big[\partial_t\log q_t + \dot\beta_t r + \beta_t\partial_t r\big]$. Hence the analogue of Eq. (266):

$$\partial_t\log p_t = \Big(\partial_t\log q_t + \dot\beta_t\, r + \beta_t\,\partial_t r\Big) - \mathbb{E}_{p_t}\!\Big[\partial_t\log q_t + \dot\beta_t\, r + \beta_t\,\partial_t r\Big].$$

The $-\mathbb{E}_{p_t}[\cdot]$ structure is intact, so the new $\beta_t\,\partial_t r$ term will simply ride along into the weight with its mean subtracted.

## Step 3' — FK-PDE with the extra schedule term (analogue of Eq. 270)

Substituting the (unchanged) Eq. (268) expansion of $\partial_t\log q_t$, the transport block reassembles into the same operator on $p_t$ with the same drift $v_t$ and noise $\sigma_t$. The weight is the old one plus $\beta_t\,\partial_t r$:

$$g_t(x) = \Big\langle \beta_t\nabla r,\, v_t - \sigma_t^2\nabla\log q_t - \tfrac{\sigma_t^2}{2}\beta_t\nabla r\Big\rangle - \beta_t\tfrac{\sigma_t^2}{2}\Delta r + \dot\beta_t\, r + \boxed{\beta_t\,\partial_t r}. \tag{270'}$$

## Step 5' — Specialize $v_t = -f_t + \sigma_t^2\nabla\log q_t$ and $a = \beta_t\sigma_t^2/2$

All spatial cancellations are exactly as in the original (the $\Delta r$ term is killed by the choice of $a$; the $\beta_t^2\|\nabla r\|^2$ terms cancel pairwise). The extra $\beta_t\,\partial_t r$ term is spatially inert, so it passes through untouched. The final scheme:

$$dx_t = \Big(-f_t(x_t) + \sigma_t^2\nabla\log q_t(x_t) + \beta_t\tfrac{\sigma_t^2}{2}\nabla r(x_t,t)\Big)dt + \sigma_t dW_t, \tag{275'}$$

$$dw_t = \left[\; \dot\beta_t\, r(x_t,t) \;+\; \beta_t\,\partial_t r(x_t,t) \;+\; \Big\langle \beta_t\nabla r(x_t,t),\; \tfrac{\sigma_t^2}{2}\nabla\log q_t(x_t) - f_t(x_t)\Big\rangle \;\right]dt. \tag{276'}$$

Setting $\partial_t r \equiv 0$ recovers Proposition D.6 exactly.

## A unifying observation

Fold the temperature into the reward by defining the full tilt exponent $\rho(x,t) := \beta_t\, r(x,t)$. Then

- spatial: $\nabla\rho = \beta_t\nabla r$ and $\Delta\rho = \beta_t\Delta r$;
- temporal: $\partial_t\rho = \dot\beta_t\, r + \beta_t\,\partial_t r$.

So the two schedule terms in $(276')$ are nothing but the single partial time derivative of the exponent, $\partial_t\rho(x_t,t)$. The entire result is therefore the *same* formula as D.6 with the replacement $\beta_t r(x) \mapsto \rho(x,t)$ and the constant-tilt schedule term $\dot\beta_t r$ promoted to $\partial_t\rho$:

$$dw_t = \left[\; \partial_t\rho(x_t,t) \;+\; \Big\langle \nabla\rho(x_t,t),\; \tfrac{\sigma_t^2}{2}\nabla\log q_t(x_t) - f_t(x_t)\Big\rangle \;\right]dt.$$

This makes precise why the original derivation needed no structural change: D.6 already handled an arbitrary time-dependent *scalar* schedule $\beta_t$; allowing the *spatial profile* to drift in time adds exactly one more contribution to the same $\partial_t(\text{exponent})$ slot.

## Practical / implementation note

The new term is the **partial** time derivative $\partial_t r(x_t, t)$ evaluated along the trajectory, holding the position $x_t$ fixed and differentiating only the explicit $t$-argument. It is *not* the total/material derivative

$$\frac{d}{dt}\, r(x_t,t) = \partial_t r + \langle\nabla r,\, \dot x_t\rangle,$$

because the convective contribution $\langle\nabla r, \cdot\rangle$ is already accounted for separately — it lives in the guidance drift $\beta_t\tfrac{\sigma_t^2}{2}\nabla r$ and in the inner-product term of the weight. Double-counting it would over-correct. Concretely: if $r$ is supplied as a closed-form time-dependent reward, evaluate $\partial_t r$ analytically (or by automatic differentiation w.r.t. the time argument only); if it is only available on a schedule of snapshots $r(\cdot, t_k)$, a finite difference $\big(r(x_t, t_{k+1}) - r(x_t, t_k)\big)/\Delta t$ at the *fixed* sample location $x_t$ is the correct estimator.