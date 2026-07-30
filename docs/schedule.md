# Schedules and Their Implied Forward Processes

`torchGMM.schedule` defines noise/interpolation schedules of the form

$$X_t = \alpha(t)X_0 + \sigma(t)\epsilon,\qquad \epsilon\sim\mathcal{N}(0,I),\qquad t\in[0,1].$$

The convention is:

- `t=0`: data or low-noise endpoint.
- `t=1`: noise or high-noise endpoint.
- `alpha(t)`: signal coefficient.
- `sigma(t)`: marginal noise standard deviation.

For a GMM, this path stays analytic: every component mean is scaled by `alpha(t)` and every component covariance receives an added `sigma(t)^2 I` term.

## Generic schedule-to-SDE conversion

The base `Schedule` class represents a marginal path through `alpha(t)`, `sigma(t)`, and their derivatives. The corresponding forward SDE is written

$$dX_t = f(X_t,t)\,dt + g(t)\,dW_t.$$

For schedules with nonzero signal coefficient, the forward drift is

$$f(x,t)=\frac{\dot{\alpha}(t)}{\alpha(t)}x.$$

The diffusion coefficient is chosen so that the SDE has the same marginal variance growth as the interpolation path:

$$g(t)^2
=2\left(\dot{\sigma}(t)\sigma(t)-\frac{\dot{\alpha}(t)}{\alpha(t)}\sigma(t)^2\right).$$

This is what `Schedule.forward_drift` and `Schedule.diffusion_coeff` implement. Concrete schedules may override these methods with simpler closed forms.

## BetaSchedule

`BetaSchedule` is the variance-preserving VP-SDE schedule:

$$dX_t=-\frac12\beta(t)X_t\,dt+\sqrt{\beta(t)}\,dW_t,$$

with linear beta rate

$$\beta(t)=\beta_{\min}+t(\beta_{\max}-\beta_{\min}).$$

Define the integrated beta

$$B(t)=\int_0^t\beta(s)\,ds
=\beta_{\min}t+\frac12(\beta_{\max}-\beta_{\min})t^2.$$

Then

$$\alpha(t)=\exp\left(-\frac12 B(t)\right),$$

and the variance-preserving constraint gives

$$\sigma(t)=\sqrt{1-\alpha(t)^2}
=\sqrt{1-\exp(-B(t))}.$$

The derivative of `alpha(t)` is

$$\dot{\alpha}(t)
=\frac{d}{dt}\exp\left(-\frac12 B(t)\right)
=-\frac12\dot{B}(t)\exp\left(-\frac12 B(t)\right)
=-\frac12\beta(t)\alpha(t).$$

That is why `BetaSchedule.get_dalpha_dt()` returns

```python
-0.5 * self.beta(t) * self.get_alpha_t(t)
```

The derivative of `sigma(t)` follows from $\sigma(t)^2=1-\alpha(t)^2$:

$$2\sigma(t)\dot{\sigma}(t)
=-2\alpha(t)\dot{\alpha}(t)
=\beta(t)\alpha(t)^2,$$

so

$$\dot{\sigma}(t)=\frac12\beta(t)\frac{\alpha(t)^2}{\sigma(t)}.$$

Plugging these into the generic schedule-to-SDE formulas recovers the VP-SDE:

$$f(x,t)=-\frac12\beta(t)x,\qquad g(t)=\sqrt{\beta(t)}.$$

## LinearSchedule

`LinearSchedule` uses the straight interpolation path

$$\alpha(t)=1-t,\qquad \sigma(t)=t.$$

Thus

$$X_t=(1-t)X_0+t\epsilon.$$

The derivatives are

$$\dot{\alpha}(t)=-1,\qquad \dot{\sigma}(t)=1.$$

The implied forward drift is

$$f(x,t)=\frac{\dot{\alpha}(t)}{\alpha(t)}x=-\frac{x}{1-t},$$

and the diffusion coefficient is

$$g(t)^2
=2\left(t-\frac{(-1)t^2}{1-t}\right)
=\frac{2t}{1-t},$$

so

$$g(t)=\sqrt{\frac{2t}{1-t}}.$$

This schedule is not variance-preserving: it satisfies $\alpha(t)+\sigma(t)=1$, not $\alpha(t)^2+\sigma(t)^2=1$.

## VESchedule

`VESchedule` is a variance-exploding schedule with constant signal:

$$\alpha(t)\equiv 1.$$

The marginal noise scale grows geometrically:

$$\sigma(t)=\sigma_{\min}\left(\frac{\sigma_{\max}}{\sigma_{\min}}\right)^t.$$

Let

$$\ell=\log\frac{\sigma_{\max}}{\sigma_{\min}}.$$

Then

$$\dot{\sigma}(t)=\sigma(t)\ell.$$

Since $\dot{\alpha}(t)=0$, the forward drift is zero and the diffusion coefficient is

$$g(t)^2=2\sigma(t)\dot{\sigma}(t)=2\sigma(t)^2\ell,$$

so

$$g(t)=\sigma(t)\sqrt{2\ell}.$$

This is a pure noising process:

$$dX_t=g(t)\,dW_t.$$

## KarrasSchedule

`KarrasSchedule` is also variance-exploding with

$$\alpha(t)\equiv 1.$$

Its marginal noise scale is

$$\sigma(t)=\sigma_{\mathrm{data}}
\left(\sigma_{\min}^{1/\rho}
+t(\sigma_{\max}^{1/\rho}-\sigma_{\min}^{1/\rho})\right)^\rho.$$

If

$$u(t)=\sigma_{\min}^{1/\rho}
+t(\sigma_{\max}^{1/\rho}-\sigma_{\min}^{1/\rho}),$$

then

$$\sigma(t)=\sigma_{\mathrm{data}}u(t)^\rho,$$

and

$$\dot{\sigma}(t)
=\sigma_{\mathrm{data}}\rho u(t)^{\rho-1}
\left(\sigma_{\max}^{1/\rho}-\sigma_{\min}^{1/\rho}\right).$$

Because $\alpha(t)\equiv1$, the drift is zero and

$$g(t)=\sqrt{2\sigma(t)\dot{\sigma}(t)}.$$

The Karras schedule is covered in more detail in [`karras_schedule.md`](karras_schedule.md), including its implied forward process and reverse-time equations.

## Reverse-time usage

The reverse-time SDE uses the score

$$s_t(x)=\nabla_x\log q_t(x).$$

For a forward SDE

$$dX_t=f(X_t,t)\,dt+g(t)\,dW_t,$$

the reverse sampler in this repository is integrated on a decreasing time grid and uses the drift callable

$$f(x,t)-g(t)^2s_t(x).$$

For VP schedules this is

$$-\frac12\beta(t)x-\beta(t)s_t(x),$$

and for VE schedules this is

$$-g(t)^2s_t(x).$$

The Euler-Maruyama implementation then applies the signed time step `dt = t_next - t_curr`, which is negative during reverse sampling.

## Transition kernels

Everything above describes *marginals*: the law of $X_t$ given $X_0$. Samplers need *transitions*: the law of $X_s$ given $X_t$ for two times on the grid. This section derives the transition kernel for a general schedule, states in what sense it can be reversed, specialises it to the four concrete schedules, and shows that the churn kernel of [`churn_sampler.md`](churn_sampler.md) §4.1 is one instance of it.

Throughout, write $\alpha_t=\alpha(t)$, $\sigma_t=\sigma(t)$ and take $s>t$, i.e. $s$ is the *noisier* of the two times (recall `t=0` is data, `t=1` is noise).

### Forward transition kernel

The derivation needs nothing beyond the stochastic interpolant itself: a pair of coefficient functions $(\alpha_t,\sigma_t)$ with $\alpha_t>0$, a data sample $X_0$, and a single noise draw $\epsilon\sim\mathcal{N}(0,I)$ held fixed along the path,

$$X_t=\alpha_tX_0+\sigma_t\epsilon .$$

The difficulty is only that both coefficients move with $t$. Divide it away: normalise by the signal coefficient and define the inverse signal-to-noise ratio,

$$Y_t\equiv\frac{X_t}{\alpha_t}=X_0+\lambda_t\,\epsilon,\qquad \lambda_t\equiv\frac{\sigma_t}{\alpha_t}.$$

The signal term is now $X_0$ itself, independent of $t$, and all time dependence sits in a single noise scale. That is, *every* interpolant is variance-exploding in the normalised variable — and for a VE process a transition is just "add more noise". This uses only $\alpha_t$ and $\sigma_t$; no schedule-specific structure, no variance-preserving constraint, no assumption on $\beta$.

Concretely, $Y_t$ and $Y_s$ share the same $X_0$ and differ only in the variance of an additive Gaussian, so Gaussian additivity gives the transition immediately — no ansatz, no coefficient matching:

$$Y_s=Y_t+\sqrt{\lambda_s^2-\lambda_t^2}\;\eta,\qquad \eta\sim\mathcal{N}(0,I),$$

because $\lambda_t\epsilon+\sqrt{\lambda_s^2-\lambda_t^2}\,\eta$ is Gaussian with variance $\lambda_s^2$, i.e. $Y_s=X_0+\lambda_s\epsilon'$ is the correct marginal by construction. Undoing the change of variable with $X_s=\alpha_sY_s$ and $Y_t=X_t/\alpha_t$,

$$X_s=\frac{\alpha_s}{\alpha_t}X_t+\alpha_s\sqrt{\lambda_s^2-\lambda_t^2}\;\eta,
\qquad
\alpha_s^2\left(\lambda_s^2-\lambda_t^2\right)=\sigma_s^2-\frac{\alpha_s^2}{\alpha_t^2}\sigma_t^2,$$

so the forward kernel is

$$\boxed{\;q(x_s\mid x_t)=\mathcal{N}\!\left(\frac{\alpha_s}{\alpha_t}x_t,\;
\left(\sigma_s^2-\frac{\alpha_s^2}{\alpha_t^2}\sigma_t^2\right)I\right),\qquad
X_s=\frac{\alpha_s}{\alpha_t}X_t+\sqrt{\sigma_s^2-\frac{\alpha_s^2}{\alpha_t^2}\sigma_t^2}\;\eta.\;}$$

Write $v$ for the boxed variance. That the kernel maps the marginal at $t$ exactly onto the marginal at $s$ — no discretisation error, for any gap $s-t$ — is manifest in the $Y$ picture, where it is the statement $Y_s=X_0+\lambda_s\epsilon'$ above.

**Consistency with the SDE.** In the normalised variable the schedule-to-SDE conversion also becomes a one-liner:

$$\frac{g(u)^2}{\alpha_u^2}
=\frac{2}{\alpha_u^2}\left(\dot\sigma_u\sigma_u-\frac{\dot\alpha_u}{\alpha_u}\sigma_u^2\right)
=\frac{d}{du}\!\left(\lambda_u^2\right),$$

i.e. $g^2/\alpha^2$ is exactly the growth rate of the injected variance. Integrating it over $[t,s]$ and multiplying by $\alpha_s^2$ reproduces $v$, which is why `diffusion_coeff` and the transition kernel are two views of the same object: the former is the infinitesimal version of the latter.

**Positivity condition.** $v=\alpha_s^2(\lambda_s^2-\lambda_t^2)\ge0$ iff $\lambda_s\ge\lambda_t$: the inverse SNR must be non-decreasing in $t$, equivalently the SNR $\alpha_t^2/\sigma_t^2$ must be monotonically non-increasing. Every schedule in `schedule.py` satisfies it:

- `BetaSchedule`: $\lambda_t=\sqrt{\alpha_t^{-2}-1}=\sqrt{e^{B(t)}-1}$, increasing because $\beta>0$ makes $B$ increasing.
- `LinearSchedule`: $\lambda_t=t/(1-t)$, increasing on $[0,1)$.
- `VESchedule`, `KarrasSchedule`: $\alpha_t\equiv1$, so $\lambda_t=\sigma_t$, increasing by construction.

### The reverse direction

The reverse transition $p(x_t\mid x_s)$ for $t<s$ is **not** Gaussian in general. It is obtained from Bayes' rule,

$$p(x_t\mid x_s)=\int q(x_t\mid x_s,x_0)\,p(x_0\mid x_s)\,dx_0,$$

and therefore depends on the data distribution through the posterior $p(x_0\mid x_s)$ — equivalently through the score. Only the $x_0$-conditioned factor is Gaussian.

**(a) The exact $x_0$-conditioned posterior.** Work in $Y$ again. Conditioning on $X_0$, the chain is $x_0\to y_t\to y_s$ with $q(y_s\mid y_t)=\mathcal{N}(y_t,\,wI)$, $w\equiv\lambda_s^2-\lambda_t^2$, and $q(y_t\mid x_0)=\mathcal{N}(x_0,\lambda_t^2I)$. Their product is Gaussian in $y_t$ with precisions adding,

$$\frac{1}{w}+\frac{1}{\lambda_t^2}=\frac{\lambda_t^2+w}{w\lambda_t^2}=\frac{\lambda_s^2}{w\lambda_t^2},
\qquad\text{so}\qquad
\tilde\sigma_Y^2=\frac{w\lambda_t^2}{\lambda_s^2}=\lambda_t^2\left(1-\frac{\lambda_t^2}{\lambda_s^2}\right),$$

and mean the precision-weighted combination $\tilde\sigma_Y^2(y_s/w+x_0/\lambda_t^2)$, i.e.

$$q(y_t\mid y_s,x_0)=\mathcal{N}\!\left(\frac{\lambda_t^2}{\lambda_s^2}\,y_s+\left(1-\frac{\lambda_t^2}{\lambda_s^2}\right)x_0,\;\;\lambda_t^2\left(1-\frac{\lambda_t^2}{\lambda_s^2}\right)I\right).$$

This is the DDPM posterior for a general schedule, and in these coordinates it is transparent: the mean is a plain convex interpolation between the noisy observation $y_s$ and the clean sample $x_0$, weighted by the SNR ratio $\lambda_t^2/\lambda_s^2$, and the variance is the prior variance $\lambda_t^2$ shrunk by exactly the complementary factor. Multiplying through by $\alpha_t$ transforms it back to $X$:

$$q(x_t\mid x_s,x_0)=\mathcal{N}\!\left(\frac{\alpha_s\sigma_t^2}{\alpha_t\sigma_s^2}x_s+\frac{\alpha_tv}{\sigma_s^2}x_0,\;\frac{v\sigma_t^2}{\sigma_s^2}I\right),$$

using $\lambda_t^2/\lambda_s^2=\alpha_s^2\sigma_t^2/(\alpha_t^2\sigma_s^2)$ and $1-\lambda_t^2/\lambda_s^2=v/\sigma_s^2$.

**(b) The sampler-level approximation.** The sampler does not know $x_0$. Tweedie's formula supplies its posterior mean: differentiating $p_s(x_s)=\int\mathcal{N}(x_s;\alpha_sx_0,\sigma_s^2I)p(x_0)\,dx_0$ under the integral,

$$\nabla_{x_s}\log p_s(x_s)
=\frac{1}{p_s(x_s)}\int\frac{\alpha_sx_0-x_s}{\sigma_s^2}\,\mathcal{N}(x_s;\alpha_sx_0,\sigma_s^2I)\,p(x_0)\,dx_0
=\frac{\alpha_s\,\mathbb{E}[X_0\mid x_s]-x_s}{\sigma_s^2},$$

hence

$$\hat{x}_0(x_s)\equiv\mathbb{E}[X_0\mid x_s]=\frac{x_s+\sigma_s^2\,s_s(x_s)}{\alpha_s},\qquad s_s(x)=\nabla_x\log p_s(x).$$

In this repository $s_s$ is not a learned approximation: `GMM.score` returns it analytically at any $t$, because the GMM family is closed under Gaussian convolution.

In $Y$ coordinates Tweedie reads $\hat{x}_0=y_s+\alpha_s\lambda_s^2\,s_s(x_s)$, since $\sigma_s^2/\alpha_s=\alpha_s\lambda_s^2$. Substituting it into the interpolation collapses the two terms:

$$\tilde\mu_Y=\frac{\lambda_t^2}{\lambda_s^2}y_s+\left(1-\frac{\lambda_t^2}{\lambda_s^2}\right)\left(y_s+\alpha_s\lambda_s^2s_s\right)
=y_s+\alpha_s\left(\lambda_s^2-\lambda_t^2\right)s_s .$$

Multiplying by $\alpha_t$ and using $\alpha_t\alpha_s(\lambda_s^2-\lambda_t^2)=(\alpha_t/\alpha_s)\,v$ gives the ancestral reverse step

$$\boxed{\;X_t=\frac{\alpha_t}{\alpha_s}\Bigl(X_s+v\,s_s(X_s)\Bigr)+\sqrt{\frac{v\,\sigma_t^2}{\sigma_s^2}}\;\eta,\qquad v=\sigma_s^2-\frac{\alpha_s^2}{\alpha_t^2}\sigma_t^2.\;}$$

This is the exact mirror of the forward kernel: the same scale factor inverted, the same $v$, plus one score correction. Its noise scale is $\sqrt{v\sigma_t^2/\sigma_s^2}=\sigma_t\sqrt{1-\lambda_t^2/\lambda_s^2}$.

**What the approximation costs.** By the law of total variance applied to $p(x_t\mid x_s)=\int q(x_t\mid x_s,x_0)p(x_0\mid x_s)dx_0$, the exact reverse transition has the mean above — so the boxed step gets the *mean exactly right* — but covariance, writing $\kappa\equiv1-\lambda_t^2/\lambda_s^2$,

$$\operatorname{Cov}[X_t\mid x_s]=\sigma_t^2\kappa\,I+\alpha_t^2\kappa^2\operatorname{Cov}[X_0\mid x_s].$$

The sampler drops the second term and, more importantly, replaces a mixture by a single Gaussian. The approximation is therefore exact whenever $p(x_0\mid x_s)$ is a point mass — the small-step limit $v\to0$, or a single-component (`Conditional`) target — and biased otherwise, most visibly at large $v$ where the posterior over $x_0$ is genuinely multimodal.

### Specialisation to the concrete schedules

All four rows use the same two quantities: the scale $\alpha_s/\alpha_t$ and the injected variance $v=\alpha_s^2(\lambda_s^2-\lambda_t^2)=\sigma_s^2-(\alpha_s/\alpha_t)^2\sigma_t^2$. The two VE schedules have $\lambda_t=\sigma_t$ and the change of variable is the identity.

| Schedule | $\alpha_s/\alpha_t$ | $v$ |
|---|---|---|
| `BetaSchedule` | $\exp\!\left(-\tfrac12\int_t^s\beta(u)\,du\right)$ | $1-\dfrac{\alpha_s^2}{\alpha_t^2}=1-\exp\!\left(-\int_t^s\beta(u)\,du\right)$ |
| `LinearSchedule` | $\dfrac{1-s}{1-t}$ | $\dfrac{(s-t)(s+t-2st)}{(1-t)^2}$ |
| `VESchedule` | $1$ | $\sigma_s^2-\sigma_t^2=\sigma_{\min}^2\!\left(r^{2s}-r^{2t}\right),\;r=\sigma_{\max}/\sigma_{\min}$ |
| `KarrasSchedule` | $1$ | $\sigma_s^2-\sigma_t^2=\sigma_{\mathrm{data}}^2\!\left(u(s)^{2\rho}-u(t)^{2\rho}\right)$ |

**`BetaSchedule`.** The scale follows from $\alpha_t=\exp(-\tfrac12B(t))$, so $\alpha_s/\alpha_t=\exp(-\tfrac12(B(s)-B(t)))$. For the variance, use $\sigma^2=1-\alpha^2$:

$$v=\left(1-\alpha_s^2\right)-\frac{\alpha_s^2}{\alpha_t^2}\left(1-\alpha_t^2\right)
=1-\alpha_s^2-\frac{\alpha_s^2}{\alpha_t^2}+\alpha_s^2
=1-\frac{\alpha_s^2}{\alpha_t^2}.$$

In DDPM notation with $\bar\alpha\equiv\alpha^2$ this is $v=1-\bar\alpha_s/\bar\alpha_t$, and the kernel $X_s=\sqrt{\bar\alpha_s/\bar\alpha_t}\,X_t+\sqrt{1-\bar\alpha_s/\bar\alpha_t}\,\eta$ is exactly the DDPM forward transition. Note the collapse: under the VP constraint $v$ no longer references $\sigma$ at all.

**`LinearSchedule`.** With $\alpha_t=1-t$ and $\sigma_t=t$,

$$v=s^2-\frac{(1-s)^2}{(1-t)^2}t^2
=\frac{\bigl(s(1-t)-t(1-s)\bigr)\bigl(s(1-t)+t(1-s)\bigr)}{(1-t)^2}
=\frac{(s-t)(s+t-2st)}{(1-t)^2},$$

using $s(1-t)-t(1-s)=s-t$. Both factors are positive for $0\le t<s<1$, confirming the monotone-SNR condition directly.

**`VESchedule` and `KarrasSchedule`.** Because $\alpha_t\equiv1$, the scale factor is $1$ and the kernel collapses to plain additive noise,

$$X_s=X_t+\sqrt{\sigma_s^2-\sigma_t^2}\;\eta,$$

with no rescaling of the signal — which is why `forward_drift` returns zero for both. The reverse step likewise simplifies to $X_t=X_s+(\sigma_s^2-\sigma_t^2)s_s(X_s)+\sigma_t\sqrt{1-\sigma_t^2/\sigma_s^2}\,\eta$.

### The link to churn

[`churn_sampler.md`](churn_sampler.md) §4.1 boxes the VP churn kernel

$$\hat{X}_i=\frac{\hat\alpha_i}{\alpha_i}X_i+\sqrt{1-\frac{\hat\alpha_i^2}{\alpha_i^2}}\;\eta_i .$$

This is the general kernel above with $t\mapsto t_i$, $s\mapsto\hat t_i$, specialised by the VP constraint: the scale $\alpha_s/\alpha_t$ is verbatim, and the `BetaSchedule` row of the table shows that $v=\sigma_{\hat t}^2-(\hat\alpha/\alpha)^2\sigma_t^2$ reduces to $1-\hat\alpha_i^2/\alpha_i^2$ precisely when $\alpha^2+\sigma^2=1$. The document's §3.2 observation that VP churn "is not just add noise" is then the statement that $\alpha_s/\alpha_t\ne1$ off the VE branch; its §5 claim that churn is exact is the marginal-mapping property noted above.

The $\gamma$ parameterisation of §4.3 sets $\hat{\bar\alpha}_i=(1-\gamma_i)\bar\alpha_i$, i.e. $\hat\alpha_i^2=(1-\gamma_i)\alpha_i^2$, so

$$v=1-\frac{\hat\alpha_i^2}{\alpha_i^2}=1-(1-\gamma_i)=\gamma_i,$$

which is why the injected noise has variance exactly $\gamma_i$ per dimension — the clean geometric meaning is a consequence of the kernel, not an extra assumption.

The same formula gives EDM's VE branch (§3.1). With $\alpha\equiv1$ and $\hat\sigma_i=(1+\gamma_i)\sigma_i$,

$$v=\hat\sigma_i^2-\sigma_i^2=\bigl((1+\gamma_i)^2-1\bigr)\sigma_i^2=\gamma_i(2+\gamma_i)\,\sigma_i^2 ,$$

so EDM's clamp $\gamma_i\le\sqrt2-1$ is exactly the requirement $\hat\sigma_i^2\le2\sigma_i^2$, i.e. injected variance at most the current variance. Both EDM branches are therefore the one kernel evaluated on two different $(\alpha,\sigma)$ paths; only the scale factor distinguishes them.

### First-order (Euler-Maruyama) approximation — the road not taken

The churn samplers in `sampling.py` use the closed-form kernel above: they take
`Schedule.transition` as a callable and jump $t\rightarrow\hat t$ in one exact draw. This
subsection derives what the *alternative* would cost, because that difference is the
entire reason the exact kernel is worth threading through the API — and it is where the
$O(h^{3/2})$ figure quoted in `reverse_churn_sampling`'s docstring comes from.

The alternative is one Euler-Maruyama step of the forward SDE over $h=\gamma|dt|$, using
only `schedule.forward_drift` and `schedule.diffusion_coeff`:

$$X_{t+h}\;\approx\;X_t+f(X_t,t)\,h+g(t)\sqrt{h}\,\eta
=\left(1+\frac{\dot\alpha_t}{\alpha_t}h\right)X_t+g(t)\sqrt{h}\,\eta .$$

Compare term by term with the exact kernel at $s=t+h$.

**Scale.** Taylor-expanding the exact factor,

$$\frac{\alpha_{t+h}}{\alpha_t}=\frac{\alpha_t+\dot\alpha_th+\tfrac12\ddot\alpha_th^2+O(h^3)}{\alpha_t}
=1+\frac{\dot\alpha_t}{\alpha_t}h+\frac{\ddot\alpha_t}{2\alpha_t}h^2+O(h^3),$$

so the Euler factor $1+(\dot\alpha_t/\alpha_t)h$ agrees to $O(h)$ and errs by $\tfrac12(\ddot\alpha_t/\alpha_t)h^2$. For `VESchedule` and `KarrasSchedule` the error is zero: $\alpha\equiv1$ makes both factors exactly $1$.

**Variance.** Using $v=\alpha_{t+h}^2(\lambda_{t+h}^2-\lambda_t^2)$ and the identity $\tfrac{d}{du}\lambda_u^2=g(u)^2/\alpha_u^2$ from the forward derivation,

$$v(h)=\alpha_{t+h}^2\int_t^{t+h}\frac{d}{du}\left(\lambda_u^2\right)du
=\alpha_t^2\left(1+O(h)\right)\left(\frac{g(t)^2}{\alpha_t^2}h+O(h^2)\right)
=g(t)^2h+O(h^2).$$

So the Euler injection $g(t)\sqrt{h}$ matches the exact standard deviation $\sqrt{v(h)}=g(t)\sqrt{h}\,(1+O(h))$ up to a relative $O(h)$, i.e. an absolute $O(h^{3/2})$ discrepancy per step. The VE case makes this concrete: $v(h)=\sigma_{t+h}^2-\sigma_t^2=2\sigma_t\dot\sigma_th+(\dot\sigma_t^2+\sigma_t\ddot\sigma_t)h^2+O(h^3)$, whose leading term is $g(t)^2h$ exactly.

**What the Euler form would cost.** Both halves of the churn step would be first order: the marginal after an Euler churn is $p(\cdot;\hat t)$ only to $O(h)$, whereas the boxed kernel maps it exactly for any $h$. That would weaken the "churn is exact" leg of the splitting argument in `churn_sampler.md` §5 to "churn is exact in the small-step limit", and with it the whole reason to prefer a splitting over a plain reverse-SDE discretisation — with a first-order churn the scheme collapses back to Euler-Maruyama up to $O(h^{3/2})$ and the splitting buys nothing.

Since the exact kernel needs only $\alpha$ and $\sigma$ at the churned time $\hat t$ — closed form for every `Schedule`, and no model evaluation — the extra cost is negligible and the samplers take it unconditionally. The Euler form is retained here only as the comparison that quantifies the gap.
