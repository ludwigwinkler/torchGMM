# Evaluation Count: Churn Sampler and Churn-FKC Sampler

**Status:** Reference note

**Related documents:** [`churn_sampler.md`](churn_sampler.md), [`fkc_churn_steering.md`](fkc_churn_steering.md)

Every model evaluation each sampler performs, written out. All counts are verified against
instrumented runs; the measured tallies are in §7.

## 1. Notation

A strictly decreasing grid $t_0 > t_1 > \dots > t_N$ in $[0,1]$, with step sizes

$$\delta_i = t_i - t_{i+1} > 0 .$$

The schedule supplies $\alpha_t,\sigma_t$ and hence

$$
f(x,t)=\frac{\dot\alpha_t}{\alpha_t}x,
\qquad
g(t)^2=2\Bigl(\dot\sigma_t\sigma_t-\frac{\dot\alpha_t}{\alpha_t}\sigma_t^2\Bigr).
$$

Write $s_t(x)=\nabla_x\log q_t(x)$ for the score. **This is the only object that costs a
model evaluation.** Everything else — $\alpha_t,\sigma_t,f,g$ — is closed form and free.

The probability-flow velocity and the exact forward kernel are

$$
v_t(x)=f(x,t)-\tfrac12 g(t)^2 s_t(x),
\qquad
K_{t\rightarrow s}(\cdot\mid x):\;
x_s=\frac{\alpha_s}{\alpha_t}x+\sqrt{\sigma_s^2-\frac{\alpha_s^2}{\alpha_t^2}\sigma_t^2}\;\eta,
\quad \eta\sim\mathcal N(0,I).
$$

Note $K$ contains **no score**: it is a closed-form Gaussian draw. Given a churn strength
$\kappa\ge 0$ the reheated time is

$$\hat t_i=\min\bigl(t_i+\kappa\,\delta_i,\;1\bigr).$$

## 2. Unsteered churn sampler

One step $x_i\mapsto x_{i+1}$, as implemented in `reverse_churn_sampling`:

$$
\begin{aligned}
\textbf{(C1)}\quad \hat x_i &= \frac{\alpha_{\hat t_i}}{\alpha_{t_i}}x_i
+\sqrt{\sigma_{\hat t_i}^2-\frac{\alpha_{\hat t_i}^2}{\alpha_{t_i}^2}\sigma_{t_i}^2}\;\eta_i,
&&\eta_i\sim\mathcal N(0,I),
&&\text{if }\hat t_i>t_i \\[4pt]
\textbf{(C2)}\quad x_{i+1} &= \hat x_i + v_{\hat t_i}(\hat x_i)\,\bigl(t_{i+1}-\hat t_i\bigr),
&&t_{i+1}-\hat t_i=-(1+\kappa)\,\delta_i .
\end{aligned}
$$

(C1) is free. (C2) costs exactly one score evaluation, at the **reheated** pair
$(\hat x_i,\hat t_i)$ — not at $(\hat x_i, t_i)$, see [`fkc_churn_steering.md`](fkc_churn_steering.md) §5.

$$
\boxed{\;\#\text{score}_{\text{churn}}
=\sum_{i=0}^{N-1} 1
= N\;}
$$

independent of $\kappa$. Churn is not paid for in evaluations — it is paid for in transport
length, since (C2) spans $(1+\kappa)\delta_i$ rather than $\delta_i$.

## 3. Churn-FKC sampler

`steered_reverse_churn_sampling` keeps (C1)–(C2) **verbatim** — the probability flow is not
guided — and adds the endpoint potential $\rho_t(x)=\beta(t)\,r(x,t)$.

Initialisation, once:

$$\textbf{(W0)}\quad \varrho_0 = \rho_{t_0}(x_0).$$

Then per step, after (C1)–(C2):

$$
\begin{aligned}
\textbf{(W1)}\quad \varrho_{i+1} &= \rho_{t_{i+1}}(x_{i+1}) \\[2pt]
\textbf{(W2)}\quad \log w^{(n)} &\mathrel{+}= \varrho_{i+1}^{(n)}-\varrho_i^{(n)} \\[2pt]
\textbf{(W3)}\quad \mathrm{ESS}/N &= \Bigl(N\textstyle\sum_n \bar w_n^2\Bigr)^{-1},
\qquad \bar w=\mathrm{softmax}(\log w) \\[2pt]
\textbf{(W4)}\quad \text{if resampling: }&\;
x\leftarrow x[\mathcal A],\quad
\varrho\leftarrow \varrho[\mathcal A],\quad
\log w\leftarrow 0 .
\end{aligned}
$$

Two things keep this at one potential evaluation per step:

- **(W2) reuses $\varrho_i$**, which is the value (W1) produced last step. The ancestor's
  tilt is never recomputed.
- **(W4) gathers $\varrho$ by the resampling index $\mathcal A$ rather than re-evaluating.**
  A resampled particle's potential is its ancestor's potential, so a gather is exact.

(W2)–(W4) are pure arithmetic on $[N]$-shaped tensors: free. Hence

$$
\boxed{\;
\#\text{score}_{\text{churn-FKC}} = N,
\qquad
\#\rho_{\text{churn-FKC}} = N+1\;}
$$

The $+1$ is (W0). **FKC costs exactly one extra potential evaluation per step**, and zero
extra score evaluations.

## 4. What one potential evaluation costs

$\rho$ is not intrinsically a model call — its cost depends entirely on the reward.

**Closed-form reward on the noisy state**, $\rho_t(x)=\beta(t)r(x)$, e.g. a Gaussian well
$r(x)=-\tfrac{1}{2\varsigma^2}\lVert x-c\rVert^2$:

$$\#\text{score}(\rho)=0 .$$

FKC is then **free**: the steered sampler makes exactly the same $N$ score calls as the
unsteered one.

**Reward on the denoised state**, $\rho_t(x)=\beta(t)\,r\bigl(D(x,t)\bigr)$, where $D$ is the
unrolled probability-flow denoiser. With $m$ substeps from $t$ down to $\epsilon$, writing
$\tau_k=t+(\epsilon-t)\tfrac{k}{m}$ and $u_0=x$:

$$
\begin{aligned}
\hat D_k &= u_{k-1}+\sigma_{\tau_{k-1}}^2\,s_{\tau_{k-1}}(u_{k-1}),
\qquad k=1,\dots,m,\\
u_k &= \frac{\sigma_{\tau_k}}{\sigma_{\tau_{k-1}}}\,u_{k-1}
+\Bigl(1-\frac{\sigma_{\tau_k}}{\sigma_{\tau_{k-1}}}\Bigr)\hat D_k,
\qquad D(x,t)=u_m .
\end{aligned}
$$

Each substep is one score call, and the substep count shrinks as $t\to 0$:

$$
m(t)=\max\bigl(1,\;\lceil (t-\epsilon)\,M\rceil\bigr)
\quad\Longrightarrow\quad
\#\text{score}(\rho_t)=m(t).
$$

So for a denoised reward the sampler totals

$$
\#\text{score}_{\text{churn-FKC}}
= \underbrace{N}_{\text{(C2)}}
+\underbrace{\sum_{i=0}^{N} m(t_i)}_{\text{(W0),(W1)}}
\;\approx\; N\bigl(1+\bar m\bigr),
\qquad
\bar m=\frac{1}{N+1}\sum_i m(t_i).
$$

**Crucially, no backward pass appears anywhere.** The endpoint difference (W2) needs
$\rho$ *values* only — never $\nabla_x r$, $\partial_t r$, $\Delta r$, or $\dot\beta$ — so
$D$ is never differentiated through.

## 5. Contrast: Euler-Maruyama FKC

For reference, `steered_reverse_sampling` on the same reward must evaluate, per step,

$$
\begin{aligned}
x_{i+1}&=x_i+\Bigl(f-g^2 s_{t_i}(x_i)-\beta\tfrac{g^2}{2}\nabla_x r\Bigr)\Delta t
+g(t_i)\sqrt{|\Delta t|}\,\xi_i,\\[4pt]
\Delta\log w&=\Bigl[\dot\beta_t\,r+\beta_t\,\partial_t r
+\bigl\langle \beta_t\nabla_x r,\;\tfrac{g^2}{2}s_{t_i}(x_i)-f\bigr\rangle\Bigr]\,|\Delta t| ,
\end{aligned}
$$

which requires $r$, $\nabla_x r$, $\partial_t r$ and $s$ — so per step one denoiser forward
**and one backward** through all $m$ substeps, at minimum. In the form written in
`tests/test_steering.py` the drift and the weight are separate callables that each rebuild
$(r,\nabla_x r,\partial_t r)$, giving $2$ forwards and $2$ backwards per step; caching
across the two halves would bring that to $1$ and $1$.

The backward pass is the structural difference and cannot be cached away: it is required by
$\nabla_x r$ in both the guided drift and the weight.

## 6. The split variant

Resampling at the reheated state ([`fkc_churn_steering.md`](fkc_churn_steering.md) §4,
option 2) splits (W1)–(W2) at $\hat x_i$:

$$
\Delta\log w_i^{\mathrm{churn}}=\rho_{\hat t_i}(\hat x_i)-\rho_{t_i}(x_i),
\qquad
\Delta\log w_i^{\mathrm{ODE}}=\rho_{t_{i+1}}(x_{i+1})-\rho_{\hat t_i}(\hat x_i).
$$

The two telescope to the single-step increment of §3, so the accumulated weight is
identical. But $\rho_{\hat t_i}(\hat x_i)$ is a *new* state never otherwise scored, so

$$\#\rho_{\text{split}}=2N+1,$$

exactly double. The extra evaluation buys only a second resampling opportunity, not
accuracy. Measured on the Karras denoised-reward setup it changed mean $\mathrm{ESS}/N$ from
$0.997$ to $0.998$ and terminal $W_1$ by less than the seed-to-seed spread, at $2\times$ the
wall clock — because the telescoping already bounds $\log w$ between resamples by the range
of $\rho$, so there is no depletion to repair. It is worth revisiting only in a regime where
ESS genuinely collapses.

## 7. Summary

Per integration step, $M=10$ denoiser substeps, $\bar m\approx 5.5$:

| sampler | score / step | denoiser fwd / step | denoiser bwd / step |
|---|---|---|---|
| churn, unsteered | $1$ | $0$ | $0$ |
| churn-FKC, closed-form $r(x_t)$ | $1$ | $0$ | $0$ |
| churn-FKC, denoised $r(D)$ | $1+\bar m$ | $1$ | $0$ |
| churn-FKC, denoised, split (§6) | $1+2\bar m$ | $2$ | $0$ |
| EM-FKC, denoised $r(D)$ (cached) | $1+\bar m$ | $1$ | $1$ |
| EM-FKC, denoised $r(D)$ (as tested) | $2+2\bar m$ | $2$ | $2$ |

Instrumented run, $99$ steps, $M=10$:

| sampler | score calls | per step | denoise | backward | wall clock |
|---|---|---|---|---|---|
| churn, unsteered | $99$ | $1.00$ | $0$ | $0$ | $0.62$ s |
| churn-FKC, $r(x_t)$ | $99$ | $1.00$ | $0$ | $0$ | $0.25$ s |
| churn-FKC, $r(D)$ | $649$ | $6.56$ | $100$ | $0$ | $1.25$ s |
| EM-FKC, $r(D)$ | $1296$ | $13.09$ | $198$ | $198$ | $3.25$ s |

The headline: **FKC on the churn sampler costs one extra potential evaluation per step —
zero extra score calls for a closed-form reward, one denoiser forward for a denoised one,
and never a backward pass.**
