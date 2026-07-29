# Design Document — VP-Native Churn Integrator for Diffusion Sampling

**Status:** Draft
**Date:** 2026-07-29
**Related work:** Karras et al., *Elucidating the Design Space of Diffusion-Based Generative Models* (EDM), NeurIPS 2022; Song et al., *Score-Based Generative Modeling Through SDEs*, ICLR 2021; Ho et al., *DDPM*, NeurIPS 2020.

---

## 1. Summary

We propose a stochastic sampler that ports EDM Algorithm 2 to a **variance-preserving (VP)** noise schedule while retaining its splitting-based correctness argument. The sampler alternates an *exact* forward-noise transition ("churn") with a Heun step on the VP probability-flow ODE. In the VP case, the churn step must both scale the current sample and inject fresh Gaussian noise — this is exactly the DDPM forward transition kernel. The resulting integrator is a marginal-preserving operator-splitting scheme, not a general-purpose SDE discretizer; it inherits the same correctness properties as EDM Algorithm 2, but expressed natively in $(\alpha, \sigma)$ coordinates.

## 2. Motivation

Most public diffusion checkpoints (DDPM, iDDPM, Stable Diffusion, class-conditional ImageNet ADM) are trained on VP or subVP schedules. EDM works around this by *reparameterizing* a VP checkpoint into the $\sigma(t) = t,\ s(t) = 1$ VE form at inference time (Appendix C.1 of EDM). That reparameterization works, but adds indirection: input and output rescaling on every denoiser call, and an $\varepsilon$-parameterization wrapper for DDPM-family models. There are three reasons to prefer a VP-native integrator instead. First, it composes cleanly with existing DDPM inference plumbing — the schedule tables $\{\bar\alpha_i\}$ are used directly. Second, the stochastic sub-step *is* the DDPM forward kernel $q(\mathbf x_j\mid\mathbf x_i)$, which makes analysis and diagnostics easier. Third, at very low NFE the VE reparameterization introduces small numerical mismatches at the schedule boundaries; a VP-native path avoids them.

## 3. Background

### 3.1 EDM churn on a VE schedule

For a VE process $\mathbf x(t) = \mathbf x_0 + \sigma(t)\varepsilon$, EDM Algorithm 2 reheats $\mathbf x_i$ from $\sigma_i$ to $\hat\sigma_i = (1+\gamma_i)\sigma_i$ by

$$\hat{\mathbf x}_i \;=\; \mathbf x_i + \sqrt{\hat\sigma_i^{\,2} - \sigma_i^{\,2}}\,\varepsilon,\qquad \varepsilon \sim \mathcal N(0, I).$$

Because the VE forward kernel *is* additive Gaussian noise, this transition is exact: if $\mathbf x_i \sim p(\cdot;\sigma_i)$, then $\hat{\mathbf x}_i \sim p(\cdot;\hat\sigma_i)$ with no discretization error. The deterministic sub-step is Heun's method on the PF-ODE, which preserves the marginals to $O(h^3)$ local error. Alternation of two marginal-preserving operators is the correctness argument.

### 3.2 Why VP is not "just add noise"

For a VP schedule with $\mathbf x(t) = \alpha(t)\mathbf x_0 + \sigma(t)\varepsilon$ and $\alpha^2 + \sigma^2 = 1$, moving from $t_i$ to a hotter $\hat t_i > t_i$ requires both shrinking the signal component (by $\hat\alpha_i / \alpha_i < 1$) and injecting fresh noise. Only adding noise would leave the signal component at the wrong scale, break the $\alpha^2 + \sigma^2 = 1$ constraint, and yield a marginal that is not $p(\cdot;\hat t_i)$.

## 4. Design

### 4.1 The VP churn kernel

Given $\mathbf x_i \sim p(\cdot;t_i)$ with schedule values $(\alpha_i,\sigma_i)$, we produce $\hat{\mathbf x}_i \sim p(\cdot;\hat t_i)$ with $\hat\alpha_i < \alpha_i$ via

$$
\boxed{\;\hat{\mathbf x}_i \;=\; \frac{\hat\alpha_i}{\alpha_i}\,\mathbf x_i \;+\; \sqrt{\,1 - \frac{\hat\alpha_i^{\,2}}{\alpha_i^{\,2}}\,}\;\eta_i,\qquad \eta_i \sim \mathcal N(0, I).\;}
$$

Under VP normalization ($\alpha^2 + \sigma^2 = 1$) this is exactly the DDPM forward transition kernel from step $i$ to step $\hat i$, and equivalently the closed-form solution of the VP forward SDE $d\mathbf x = -\tfrac12\beta(t)\mathbf x\,dt + \sqrt{\beta(t)}\,d\omega_t$ over the sub-interval $[t_i,\hat t_i]$. It preserves the marginal exactly, not merely to leading order.

**Sanity check.** Expanding $\mathbf x_i = \alpha_i\mathbf x_0 + \sigma_i\varepsilon_i$ into the boxed formula:

$$
\hat{\mathbf x}_i \;=\; \hat\alpha_i\mathbf x_0 \;+\; \tfrac{\hat\alpha_i\sigma_i}{\alpha_i}\varepsilon_i \;+\; \sqrt{1 - \tfrac{\hat\alpha_i^{\,2}}{\alpha_i^{\,2}}}\,\eta_i.
$$

Because $\varepsilon_i \perp \eta_i$ are standard normal, the total noise term has variance $\tfrac{\hat\alpha_i^{\,2}\sigma_i^{\,2}}{\alpha_i^{\,2}} + 1 - \tfrac{\hat\alpha_i^{\,2}}{\alpha_i^{\,2}} = 1 - \hat\alpha_i^{\,2} = \hat\sigma_i^{\,2}$, so $\hat{\mathbf x}_i = \hat\alpha_i\mathbf x_0 + \hat\sigma_i\tilde\varepsilon$ with $\tilde\varepsilon\sim\mathcal N(0,I)$. The injected noise term is real because $\alpha_i^2\hat\sigma_i^2 - \hat\alpha_i^2\sigma_i^2 = \alpha_i^2 - \hat\alpha_i^2 > 0$ whenever $\hat\alpha_i < \alpha_i$.

### 4.2 The deterministic sub-step

After churning to $\hat t_i$, we run one Heun step of the VP probability-flow ODE from $\hat t_i$ down to $t_{i+1}$. The VP PF-ODE, expressed through the denoiser $D_\theta$ via Tweedie's formula $\nabla_{\mathbf x}\log p(\mathbf x;t) = (\alpha(t)D_\theta(\mathbf x;t) - \mathbf x)/\sigma(t)^2$, is

$$
\frac{d\mathbf x}{dt} \;=\; -\tfrac{1}{2}\beta(t)\,\mathbf x \;-\; \tfrac{1}{2}\beta(t)\,\frac{\alpha(t)D_\theta(\mathbf x;t) - \mathbf x}{\sigma(t)^2}.
$$

Denote the RHS by $f(\mathbf x, t)$. Heun's step from $\hat t_i$ to $t_{i+1}$ is: predictor $\mathbf x^{\text{pred}} = \hat{\mathbf x}_i + (t_{i+1} - \hat t_i)f(\hat{\mathbf x}_i, \hat t_i)$; corrector $\mathbf x_{i+1} = \hat{\mathbf x}_i + \tfrac12(t_{i+1} - \hat t_i)\bigl[f(\hat{\mathbf x}_i, \hat t_i) + f(\mathbf x^{\text{pred}}, t_{i+1})\bigr]$. The corrector is skipped when $t_{i+1} = 0$ to avoid a division by $\sigma = 0$; the Euler predictor is returned instead. This matches EDM's terminal handling.

### 4.3 Churn parameterization

We parameterize churn by a per-step fraction $\gamma_i \in [0,1)$ acting on $\bar\alpha \equiv \alpha^2$:

$$\hat{\bar\alpha}_i \;=\; (1 - \gamma_i)\,\bar\alpha_i,\qquad \hat\alpha_i \;=\; \sqrt{1 - \gamma_i}\,\alpha_i.$$

With this choice the injected noise variance simplifies to *exactly* $\gamma_i$ per dimension, mirroring EDM's convention that $\gamma_i$ has a clean geometric meaning. The magnitude schedule follows EDM: $\gamma_i = \min(S_{\text{churn}}/N,\ \gamma_{\max})$ inside a gating window $t_i \in [S_{\text{tmin}},S_{\text{tmax}}]$, and $\gamma_i = 0$ outside. The ceiling $\gamma_{\max}$ enforces that after churn we still have a *downhill* Heun step to $t_{i+1}$ (see §4.4).

### 4.4 Ceiling on $\gamma_i$

EDM's clamp $\gamma_i \leq \sqrt 2 - 1$ comes from bounding the injected noise variance by the current noise variance, so that the reheated $\hat\sigma$ never exceeds a doubling per step. The VP analog derived from the same principle is

$$\gamma_i \;\leq\; \frac{\bar\alpha_i - \bar\alpha_{i+1}}{\bar\alpha_i} \;\equiv\; 1 - \frac{\bar\alpha_{i+1}}{\bar\alpha_i},$$

which guarantees $\hat{\bar\alpha}_i \geq \bar\alpha_{i+1}$, i.e., the Heun step still runs in the denoising direction. This ceiling is schedule-adaptive and tighter than a constant. We recommend it as the default, with a hard fallback of $\gamma_i \leq 0.5$ to keep the churn from dominating.

### 4.5 The full algorithm

```
procedure VP_STOCHASTIC_SAMPLER(D_theta, {t_i}_{i=0..N}, {gamma_i}_{i=0..N-1},
                                 S_noise, alpha, sigma, beta):
    x <- sample from N(0, I)                              # VP prior is unit Gaussian
    for i in 0..N-1:
        a_i, s_i = alpha(t_i), sigma(t_i)
        eta ~ N(0, S_noise^2 * I)

        # ---- churn: exact VP forward transition to a hotter level t_hat ----
        hat_a  = sqrt(1 - gamma_i) * a_i
        hat_s  = sqrt(1 - hat_a**2)
        t_hat  = schedule_inverse(hat_s)                  # or precomputed
        x_hat  = (hat_a / a_i) * x + sqrt(gamma_i) * eta

        # ---- Heun on the VP PF-ODE from t_hat to t_{i+1} ----
        d      = drift(x_hat, t_hat)                      # eq. §4.2
        x_next = x_hat + (t_{i+1} - t_hat) * d
        if t_{i+1} != 0:
            d_p    = drift(x_next, t_{i+1})
            x_next = x_hat + (t_{i+1} - t_hat) * 0.5 * (d + d_p)

        x = x_next
    return x
```

## 5. Correctness

The correctness argument is a direct port of EDM's, with each half made explicit.

**Churn is exact.** For any $t_i < \hat t_i$, the boxed transition of §4.1 is the closed-form solution of the linear VP forward SDE over $[t_i,\hat t_i]$; equivalently it is the transition kernel $q(\mathbf x_{\hat t_i}\mid \mathbf x_{t_i})$. If $\mathbf x_i \sim p(\cdot;t_i)$, then $\hat{\mathbf x}_i \sim p(\cdot;\hat t_i)$ *exactly* (no discretization error, no leading-order approximation).

**ODE half is marginal-preserving up to truncation.** By construction the VP PF-ODE has the same marginals $\{p(\cdot;t)\}_t$ as the VP forward SDE (Song et al., Prop. in App. D.1). Heun's method is a second-order Runge–Kutta scheme, so its per-step deviation from the exact flow is $O(h^3)$ locally, $O(h^2)$ globally.

**Composition.** Alternation of two marginal-preserving operators yields a sampler whose marginals converge to $p_{\text{data}}$ as $N \to \infty$, provided $D_\theta$ approximates the Bayes-optimal denoiser. This is *not* a valid SDE discretizer in the Euler–Maruyama sense; it is a splitting method, of the same species as EDM Algorithm 2 and the predictor-corrector sampler of Song et al.

**Where the argument breaks.** As in EDM, two things break the exactness. First, $S_{\text{noise}} > 1$ over-injects noise on line 4.5 — the marginal after churn is no longer $p(\cdot;\hat t_i)$ but a slightly wider Gaussian mixture. This is a deliberate bias, calibrated to compensate for a learned denoiser that tends to under-restore variance. Second, whenever $D_\theta$ deviates from the Bayes-optimal denoiser, the PF-ODE step no longer preserves the marginals even in the exact-flow limit. These are properties of any diffusion sampler, not specific to the VP port.

## 6. Practical considerations

**Relationship to EDM's VE reparameterization.** Karras et al. show that a VP-trained checkpoint can be run in the EDM VE integrator by a schedule change of variables. Our VP-native integrator is an alternative, not a replacement. The two should agree in the $N \to \infty$ limit and in the exact-denoiser limit; they may differ at low NFE by a small numerical margin. This design leaves the choice as a user-facing switch, with the VE reparameterization retained as the default for consistency with published EDM numbers.

**Noise-prediction wrapper.** Many DDPM-family checkpoints predict $\varepsilon$ rather than $\mathbf x_0$. We convert via $D_\theta(\mathbf x;t) = (\mathbf x - \sigma(t)\varepsilon_\theta(\mathbf x;t))/\alpha(t)$ before evaluating the drift.

**Time grid.** We adopt EDM's $\rho$-parameterization but interpret the time axis in $\bar\alpha$ space rather than $\sigma$ space. Explicitly, choose $\{\bar\alpha_i\}$ so that $\sigma(t_i)^2 = 1 - \bar\alpha_i$ follows the EDM recipe $\sigma_i = (\sigma_{\max}^{1/\rho} + \tfrac{i}{N-1}(\sigma_{\min}^{1/\rho} - \sigma_{\max}^{1/\rho}))^\rho$, with $\rho = 7$ as default. Because VP has a finite maximum $\sigma$, cap $\sigma_{\max} \leq \sigma(t_{\max})$.

**Numerical guards.** For $\sigma \to 0$, the score expression $(\alpha D_\theta - \mathbf x)/\sigma^2$ is ill-conditioned; the terminal-step Euler fallback (line-11 branch) mirrors EDM's guard and handles the singularity.

## 7. Validation plan

The plan is a straight port of EDM Fig. 4's evaluation, retargeted at VP-native inference.

The first check is an analytic-denoiser sanity test. Replace $D_\theta$ with the Bayes-optimal denoiser for a synthetic 2D Gaussian mixture (as in EDM Fig. 1b), sweep $N \in \{16,32,64,128\}$ and $S_{\text{churn}} \in \{0, 10, 40, 80\}$, and confirm that the empirical distribution matches $p_{\text{data}}$ within a Wasserstein tolerance that scales as $N^{-1/2}$. This isolates the integrator from denoiser error.

The second check is comparative FID on standard benchmarks. On unconditional CIFAR-10 with the VP-trained DDPM++ checkpoint from EDM, and on class-conditional ImageNet-64 with the ADM checkpoint, compare FID as a function of NFE for (a) the EDM VE-reparameterized integrator (published baseline), (b) our VP-native integrator with matched $\{S_{\text{churn}}, S_{\text{tmin}}, S_{\text{tmax}}, S_{\text{noise}}\}$. Report both at the optimal-per-integrator setting and at the shared-setting operating point. Deviations $>3\%$ relative FID at any NFE are worth investigating.

The third check is an $S_{\text{noise}}$ sweep. Fix $S_{\text{churn}}$ at the EDM-optimal value for each dataset and sweep $S_{\text{noise}} \in [0.9, 1.05]$. If the optimum tracks EDM's within $\pm 0.01$, the fudge-factor interpretation transfers over cleanly; otherwise it suggests the VP-native path exposes a different bias structure and warrants a separate calibration.

## 8. Open questions and risks

**Empirical bias of $S_{\text{noise}} > 1$ in VP.** EDM's diagnostic is that $S_{\text{noise}}$ compensates a denoiser tendency to regress too aggressively toward the mean. This is a denoiser property, not a schedule property, so the same bias should exist in VP; the *magnitude* of the compensating inflation may differ.

**Interaction with terminal steps.** DDPM checkpoints are typically trained with a discrete schedule that ends at a small but nonzero $\sigma_{\min}$; extending to $t_N = 0$ requires either a schedule-consistent extension or a hard truncation. The design defaults to truncation at the trained $\sigma_{\min}$, followed by a single Euler step to $t = 0$ — mirroring EDM. This should be revisited if empirical FID at low NFE degrades.

**Adaptive step sizing.** EDM found that the black-box RK45 adaptive integrator underperforms fixed-schedule Heun even at high NFE (Fig. 2 dashed curves), because the adaptive step controller wastes evaluations near the schedule endpoints. Whether the same phenomenon appears in the VP-native path is untested and orthogonal to the correctness argument here.

**Coupling with training preconditioning.** EDM's config-D preconditioning is derived under $\sigma(t) = t$; porting it to VP for training is out of scope for this document. This design targets inference only.

---

*Appendix — derivation of the ceiling in §4.4.* Requiring $\hat{\bar\alpha}_i \geq \bar\alpha_{i+1}$ and using $\hat{\bar\alpha}_i = (1 - \gamma_i)\bar\alpha_i$ gives $\gamma_i \leq 1 - \bar\alpha_{i+1}/\bar\alpha_i$. For a schedule with $\bar\alpha_{i+1}/\bar\alpha_i \to 1$ (fine grids), the ceiling relaxes; for coarse grids it tightens, which is the desired behavior. When the ceiling binds, effective churn drops smoothly with step size instead of overshooting.