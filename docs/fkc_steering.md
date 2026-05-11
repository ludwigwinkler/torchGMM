# Reward-Tilted Steering via Feynman-Kac Correctors

Equations extracted from Skreta et al., *Feynman-Kac Correctors in Diffusion:
Annealing, Guidance, and Product of Experts*, ICML 2025
(`docs/FeynmanKacCorrectorsFKC_Skreta.pdf`). All numbering matches the paper.

## Setup

A pretrained diffusion model defines marginals $q_t(x)$ that satisfy the
Fokker–Planck equation for the **forward (noising)** SDE

$$
\frac{\partial q_t}{\partial t}
= -\langle \nabla, q_t f_t \rangle + \frac{\sigma_t^2}{2}\,\Delta q_t,
\qquad
dx_t = f_t(x_t)\,dt + \sigma_t\,dW_t,
$$

with linear forward drift $f_t(x)$ and diffusion $\sigma_t$. The
corresponding **reverse / denoising SDE** (eq. 14b, generating the same
marginals when integrated from $t=1 \to 0$) is

$$
dx_t = \big(-f_t(x_t) + \sigma_t^2\,\nabla\log q_t(x_t)\big)\,dt + \sigma_t\,dW_t.
$$

The sign convention here matches eq. (22), (26), (29) in the paper where the
forward drift always enters with a minus sign.

## Feynman-Kac PDE and weighted SDE

A Feynman-Kac PDE augments transport + diffusion with a per-sample reweighting
term $g_t(x)$ (eq. 7):

$$
\frac{\partial p^{FK}_t}{\partial t}
= -\langle \nabla, p^{FK}_t v_t \rangle
+ \frac{\sigma_t^2}{2}\,\Delta p^{FK}_t
+ g_t\,p^{FK}_t.
$$

Sampling is done by simulating a weighted SDE (eq. 8):

$$
dx_t = v_t(x_t)\,dt + \sigma_t\,dW_t,
\qquad
dw_t = g_t(x_t)\,dt,
$$

then computing expectations via self-normalized importance sampling
(eq. 10):

$$
\mathbb{E}_{p_T}[\phi(x)] \;\approx\;
\sum_{k=1}^K \frac{\exp(w_T^k)}{\sum_j \exp(w_T^j)}\,\phi(x_T^k).
$$

For `T_R` discretization steps the increments are accumulated as
$w_T^k = \sum_t g_t(x_t^k)\,dt$ along each particle trajectory.
Resampling at intermediate steps (Sec. 4 of the paper) reduces variance.

## Proposition 3.4 — Reward-Tilted Target + FKC

Define a reward $r(x)$ on the data space and the time-tilted target

$$
p^{\text{reward}}_t(x) \;\propto\; q_t(x)\,\exp\!\big(\beta_t\, r(x)\big),
$$

where $\beta_t$ is a (possibly time-dependent) tilt schedule. Crucially,
$\exp(\beta_t r(x))$ does **not** evolve under the diffusion process — the
only diffusion-driven object is $q_t$. This distinguishes reward tilting
from a product of experts.

The corresponding weighted SDE is (eq. 29 / 30):

$$
dx_t = \Big(-f_t(x_t) + \sigma_t^2\big(\nabla \log q_t(x_t) + \tfrac{\beta_t}{2}\nabla r(x_t)\big)\Big)\,dt
       + \sigma_t\,dW_t,
$$

$$
dw_t = \frac{\partial \beta_t}{\partial t}\,r(x_t)\,dt
     \;-\; \big\langle \beta_t \nabla r(x_t),\; f_t(x_t) \big\rangle\,dt
     \;+\; \Big\langle \beta_t \nabla r(x_t),\; \tfrac{\sigma_t^2}{2}\nabla\log q_t(x_t) \Big\rangle\,dt.
$$

**Interpretation.** The drift mixes the unconditional score with the reward
gradient, twisted by $\beta_t/2$. Weights grow when the reward gradient
aligns with the diffusion model's vector field $\sigma_t^2 \nabla\log q_t - f_t$,
and shrink otherwise. The first term captures explicit time-dependence of
the tilt schedule.

For variance-exploding (VE) schedules used in `ve_steering.py`,
$f_t \equiv 0$ and the divergence terms vanish, leaving:

$$
dx_t = \sigma_t^2\,\big(\nabla\log q_t + \tfrac{\beta_t}{2}\nabla r\big)\,dt + \sigma_t\,dW_t,
$$

$$
dw_t = \frac{\partial \beta_t}{\partial t}\,r(x_t)\,dt
     + \Big\langle \beta_t \nabla r(x_t),\; \tfrac{\sigma_t^2}{2}\nabla\log q_t(x_t) \Big\rangle\,dt.
$$

## Pure-Bootstrap (Twist-Free) Variant

`ve_steering.py` uses a *bootstrap* simplification: keep the unguided drift
($\beta_t/2 \nabla r$ removed from the SDE), and steer entirely through the
weights. The weight increment becomes a stateless Tweedie-based potential:

$$
\Delta \log w \;=\; r\!\big(\hat x_0(x_t, t)\big)\,|dt|,
$$

with the VE Tweedie denoiser $\hat x_0(x_t,t) = x_t + \sigma_t^2 \nabla\log q_t(x_t)$.
This integrates to a cumulative log-weight $\int_0^T r(\hat x_0)\,dt$, a
valid Feynman-Kac potential targeting a tilted distribution that
concentrates on the reward mode. Trading a tighter target for numerical
robustness: twisting the drift at high $\sigma_t$ requires Jacobian
corrections and a finer integrator than Euler comfortably handles, while
the weights still re-target the tilted distribution exactly under SMC
resampling.

## Resampling (Sec. 4)

For each step $t \to t+dt$, particles accumulate increments
$\Delta w^{(k)}_t = g_t(x^{(k)}_t)\,dt$. Systematic resampling proportional
to $\exp\{\Delta w^{(k)}_t\}$ is triggered when ESS/N falls below a
threshold; weights are reset to zero after a resample. See
`steered_reverse_sampling` in `src/torchGMM/sampling.py`.

## References to the paper

- Eq. (7), (8): Feynman-Kac PDE and weighted SDE.
- Eq. (10): SNIS reweighting.
- Eq. (14): pretrained diffusion FP/SDE.
- Sec. 3.5, Prop. 3.4 (eq. 29–30): reward-tilted target.
- Sec. 4: SMC and systematic resampling.
- App. D.6: proof of Prop. 3.4.
