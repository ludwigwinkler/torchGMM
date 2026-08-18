## Guided trajectory and weight differential

Let $q_t$ be the unsteered marginal, $f_t$ and $g_t$ the forward SDE drift
and diffusion coefficient, and $\alpha\geq0$ the reverse-SDE family
parameter. For the tilted target
$p_t(x)\propto q_t(x)\exp\!\left(\beta_t r(x,t)\right)$, the guided
trajectory in denoising-direction time is

$$
\boxed{\;
\begin{aligned}
dx_t=\Bigg[
&-f_t(x_t)
+\frac{1+\alpha^2}{2}g_t^2\nabla\log q_t(x_t)\\
&+\beta_t\frac{\alpha^2g_t^2}{2}\nabla r(x_t,t)
\Bigg]dt
+\alpha g_t\,dW_t.
\end{aligned}
\;}
\tag{1}
$$

The corresponding Feynman--Kac log-weight differential is

$$
\boxed{\;
\begin{aligned}
dw_t=\Bigg[
&\frac{\partial\beta_t}{\partial t}r(x_t,t)
+\beta_t\frac{\partial r(x_t,t)}{\partial t}\\
&+\left\langle
\beta_t\nabla r(x_t,t),
\frac{g_t^2}{2}\nabla\log q_t(x_t)-f_t(x_t)
\right\rangle
\Bigg]dt.
\end{aligned}
\;}
\tag{2}
$$

The weight has the same form for every $\alpha$; $\alpha$ changes the
trajectory on which it is evaluated. The extra diffusion $\alpha$ changes the trajectory but does not change the marginal distribution of the process, as both extra diffusion is counteracted with extra score correction.
Thus the marginal distribution of $x_t$ remains $q_t(x)$ and except for numerical considerations, the log weights are calculated independently from $\alpha$.
This is relevant to the AlphaFold 3 solver, which gates churn off when the
destination noise level reaches $1$ Å. With its published 200-step schedule,
this makes approximately the final 20% of reverse diffusion deterministic; see
[`af3_sampler.md`](af3_sampler.md).

## FKC correctors in $\sigma$-space

For VE/EDM marginals $x_\sigma=x_0+\sigma\epsilon$, let

$$
\Delta_i=\sigma_i^2-\sigma_{i+1}^2>0,\qquad
s_\sigma(x)=\nabla_x\log q(x;\sigma)
=-\frac{x - D_\theta(x;\sigma)}{\sigma^2}.
\tag{3}
$$

For an exact VE `GMM`, use Tweedie's identity directly:
`x + sigma**2 * gmm.score(x, schedule.time(sigma))`.

`KarrasSchedule` uses repository time $s$, increasing from
$\sigma_{\min}$ to $\sigma_{\max}$. Here $t$ is denoising progress:
$t=1-s$, $s=\texttt{schedule.time}(\sigma)$, and
$\sigma(t)=\texttt{schedule.get\_sigma\_t}(1-t)$, so denoising runs from
$\sigma_{\max}$ to $\sigma_{\min}$. In closed form,

$$
\sigma(t)=
\left[
\sigma_{\max}^{1/\rho}
+t\left(\sigma_{\min}^{1/\rho}-\sigma_{\max}^{1/\rho}\right)
\right]^\rho.
\tag{4}
$$

Its inverse is

$$
t(\sigma)=
\frac{\sigma_{\max}^{1/\rho}-\sigma^{1/\rho}}
{\sigma_{\max}^{1/\rho}-\sigma_{\min}^{1/\rho}}.
\tag{5}
$$

Therefore,

$$
\begin{aligned}
\frac{dt}{d\sigma}
&=\frac{1}
{\sigma_{\max}^{1/\rho}-\sigma_{\min}^{1/\rho}}
\frac{d}{d\sigma}\left(-\sigma^{1/\rho}\right)\\
&=-\frac{1}
{\sigma_{\max}^{1/\rho}-\sigma_{\min}^{1/\rho}}
\frac{1}{\rho}\sigma^{1/\rho-1}\\
&=\frac{\sigma^{1/\rho-1}}
{\rho\left(\sigma_{\min}^{1/\rho}-\sigma_{\max}^{1/\rho}\right)},\\
dt
&=\frac{\sigma^{1/\rho-1}}
{\rho\left(\sigma_{\min}^{1/\rho}-\sigma_{\max}^{1/\rho}\right)}
\,d\sigma.
\end{aligned}
\tag{6}
$$

A time-indexed tilt schedule is converted to $\sigma$-space by composition
with the inverse noise schedule:

$$
\begin{aligned}
\beta_\sigma(\sigma)
&=\beta_t(t(\sigma)),\\
\frac{d\beta_\sigma}{d\sigma}
&=\left.\frac{d\beta_t}{dt}\right|_{t=t(\sigma)}
\frac{dt}{d\sigma},\\
\beta_t(t)=1-t
&\quad\Longrightarrow\quad
\beta_\sigma(\sigma)=\beta_t(t(\sigma))=1-t(\sigma),\\
\frac{d\beta_\sigma}{d\sigma}
&=-\frac{dt}{d\sigma}.
\end{aligned}
\tag{7}
$$

Here $t(\sigma)=\sigma_{\rm sched}^{-1}(\sigma)$ denotes the inverse
schedule, not the reciprocal $1/\sigma$.

For VE, $f_t=0$, and the diffusion coefficient is fixed by the marginal
variance. On a generation clock for which $\sigma$ decreases,

$$
\boxed{\;
g_t^2\,dt=-d(\sigma^2)=-2\sigma\,d\sigma.
\;}
\tag{8}
$$

This identity holds for any monotone noise schedule. In particular, all
Karras parameters cancel: $\rho$ controls grid placement, while the
trajectory and weight depend on $-2\sigma\,d\sigma$.

### Guided equations in $\sigma$-space

Substituting $f_t=0$ and (8) into (1) gives

$$
\boxed{\;
dx=
-\sigma\left[
(1+\alpha^2)s_\sigma(x)
+\alpha^2\beta_\sigma\nabla r(x,\sigma)
\right]d\sigma
+\alpha\,d\bar W_\sigma,
\qquad
d\bar W_\sigma d\bar W_\sigma^\mathsf{T}
=-2\sigma\,d\sigma\,I.
\;}
\tag{9}
$$

Likewise, (2) becomes

$$
\boxed{\;
dw=
\left[
\frac{d\beta_\sigma}{d\sigma}r(x,\sigma)
+\beta_\sigma\frac{\partial r(x,\sigma)}{\partial\sigma}
-\sigma
\left\langle\beta_\sigma\nabla r(x,\sigma),s_\sigma(x)\right\rangle
\right]d\sigma.
\;}
\tag{10}
$$

Equation (9) is the guided proposal trajectory; equation (10) is its
Feynman--Kac correction. Their roles are distinct: the proposal moves
particles toward the tilted target, while the weight corrects the remaining
change in measure.

### Discrete Karras grid

Let $\Delta_i=\sigma_i^2-\sigma_{i+1}^2$. Reheat to the immediately
preceding, larger Karras grid level, then apply a denoiser Euler step:

$$
\boxed{\;
\begin{aligned}
\hat\sigma_i
&=\begin{cases}
\sigma_{i-1},&i>0,\\
\sigma_i,&i=0,
\end{cases}\\
\hat x_i
&=x_i+\lambda\sqrt{\hat\sigma_i^2-\sigma_i^2}\,\epsilon_i,
\qquad \epsilon_i\sim\mathcal N(0,I),\\
d_i
&=\frac{\hat x_i-D_\theta(\hat x_i;\hat\sigma_i)}
{\hat\sigma_i},\\
x_{i+1}
&=\hat x_i+\eta(\sigma_{i+1}-\hat\sigma_i)d_i.
\end{aligned}
\;}
\tag{11}
$$

The first step is deterministic because no larger grid level exists. The
reheat is the exact VE transition when $\lambda=1$; $\lambda>1$
deliberately inflates its noise.
Evaluating the churn-independent FKC correction at the reheated state gives

$$
\boxed{\;
\Delta w_i=
\beta_{i+1}r(\hat x_i,\sigma_{i+1})
-\beta_i r(\hat x_i,\sigma_i)
+\frac{1}{2}
\left\langle
\beta_{\hat\sigma_i}\nabla r(\hat x_i,\hat\sigma_i),
s_{\hat\sigma_i}(\hat x_i)
\right\rangle
\Delta_i.
\;}
\tag{12}
$$

For a time-independent reward, the first two terms in (12) reduce to
$r(\hat x_i)(\beta_{i+1}-\beta_i)$. The weight uses the base gap
$\Delta_i$, not the longer denoising interval. Guidance may be incorporated
into $D_\theta$; the weight callback still receives the reheated state.

`steered_reverse_edm_sampling` implements this operator split:

```python
trajectory, ess_history, weight_history = steered_reverse_edm_sampling(
    denoise,
    weight_update,
    x,
    sigma,
    noise_scale=1.0,
    step_scale=1.0,
    ess_threshold=0.5,
)
```

`sigma` is strictly decreasing.
`denoise(x_hat, sigma_hat, sigma_i, sigma_next)` returns the denoised
estimate used in (11), and
`weight_update(x_hat, sigma_hat, sigma_i, sigma_next)` returns the complete
increment (12). Both callbacks are evaluated after reheating.

Sources: Skreta et al., *Feynman-Kac Correctors in Diffusion* (ICML 2025,
arXiv:2503.02819); Karras et al., *Elucidating the Design Space of
Diffusion-Based Generative Models* (NeurIPS 2022).