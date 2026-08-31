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
    drift,
    weight_update,
    x,
    sigma,
    noise_scale=1.0,
    step_scale=1.0,
    ess_threshold=0.5,
)
```

`sigma` is strictly decreasing.
`drift(x_hat, sigma_hat, sigma_i, sigma_next)` returns the complete
deterministic drift with respect to $\sigma^2/2$, including guidance, and
`weight_update(x_hat, sigma_hat, sigma_i, sigma_next)` returns the complete
increment (12). Both callbacks are evaluated after reheating.


# Score, denoiser, and sigma-space motion from first principles

## 1. Forward Gaussian noising

For a variance-exploding process, a clean sample is corrupted as

$$
x_\sigma=x_0+\sigma\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I).
$$

The noisy density is therefore

$$
q_\sigma
=
q_0 * \mathcal N(0,\sigma^2 I).
$$

Increasing $\sigma^2$ applies Gaussian smoothing.

## 2. Evolution of the density

Gaussian smoothing satisfies the heat equation:

$$
\frac{\partial q_\sigma}{\partial(\sigma^2)}
=
\frac12\Delta q_\sigma.
$$

A deterministic particle flow with velocity
$u_{\sigma^2}(x)$ with respect to $\sigma^2$ satisfies the continuity
equation:

$$
\frac{\partial q_\sigma}{\partial(\sigma^2)}
=
-\nabla\cdot(q_\sigma u_{\sigma^2}).
$$

Define the score

$$
s_\sigma(x)=\nabla_x\log q_\sigma(x).
$$

Because

$$
q_\sigma(x)s_\sigma(x)=\nabla q_\sigma(x),
$$

choosing

$$
u_{\sigma^2}(x)=-\frac12s_\sigma(x)
$$

gives

$$
-\nabla\cdot(q_\sigma u_{\sigma^2})
=
\frac12\nabla\cdot(q_\sigma s_\sigma)
=
\frac12\Delta q_\sigma.
$$

Thus the probability-flow ODE directly in $\sigma^2$ is

$$
\boxed{
\frac{dx}{d(\sigma^2)}
=
-\frac12s_\sigma(x).
}
$$

Equivalently,

$$
\boxed{
dx
=
-\frac12s_\sigma(x)\,d(\sigma^2).
}
$$

## 3. Convert from variance to sigma

Since

$$
d(\sigma^2)=2\sigma\,d\sigma,
$$

the same ODE written in the sigma coordinate is

$$
\begin{aligned}
dx
&=
-\frac12s_\sigma(x)\,d(\sigma^2)\\
&=
-\frac12s_\sigma(x)\,2\sigma\,d\sigma\\
&=
-\sigma s_\sigma(x)\,d\sigma.
\end{aligned}
$$

Therefore,

$$
\boxed{
\frac{dx}{d\sigma}
=
-\sigma s_\sigma(x).
}
$$

The score by itself is not the sigma-space velocity. The factor
$-\sigma$ converts the score into the velocity with respect to sigma.

## 4. Relation between the score and denoiser

Tweedie's identity gives the denoiser

$$
D(x,\sigma)
=
x+\sigma^2s_\sigma(x).
$$

Rearranging,

$$
s_\sigma(x)
=
\frac{D(x,\sigma)-x}{\sigma^2}
=
-\frac{x-D(x,\sigma)}{\sigma^2}.
$$

This confirms that

$$
-\frac{x-D(x,\sigma)}{\sigma^2}
$$

is the score.

However, multiplying this score directly by $d\sigma$ does not produce the
probability-flow motion. The coordinate conversion from variance to sigma
still contributes the factor $-\sigma$.

## 5. Sigma-space denoiser form

Substitute Tweedie's identity into the sigma-space ODE:

$$
\begin{aligned}
dx
&=
-\sigma s_\sigma(x)\,d\sigma\\
&=
-\sigma
\frac{D(x,\sigma)-x}{\sigma^2}
d\sigma\\
&=
\frac{x-D(x,\sigma)}{\sigma}
d\sigma.
\end{aligned}
$$

Thus the usual EDM equation is

$$
\boxed{
\frac{dx}{d\sigma}
=
\frac{x-D(x,\sigma)}{\sigma}.
}
$$

An Euler step from $\sigma_i$ to $\sigma_{i+1}$ is

$$
x_{i+1}
=
x_i
+
(\sigma_{i+1}-\sigma_i)
\frac{x_i-D(x_i,\sigma_i)}{\sigma_i}.
$$

This is the standard sigma-space EDM update.

## 6. Variance-space denoiser form

Start instead from the variance-space ODE:

$$
dx
=
-\frac12s_\sigma(x)\,d(\sigma^2).
$$

Using

$$
-s_\sigma(x)
=
\frac{x-D(x,\sigma)}{\sigma^2},
$$

gives

$$
\boxed{
dx
=
\frac12
\frac{x-D(x,\sigma)}{\sigma^2}
d(\sigma^2).
}
$$

An Euler step is therefore

$$
x_{i+1}
=
x_i
+
\frac{\sigma_{i+1}^2-\sigma_i^2}{2}
\frac{x_i-D(x_i,\sigma_i)}{\sigma_i^2}.
$$

This is exactly the form currently used by `steered_reverse_edm_sampling`:

```python
direction = (x - denoised) / sigma_hat**2
x = x + (sigma_next**2 - sigma_hat**2) / 2 * direction
```

So the current base sampler is moving according to the score correctly, but
it is integrating directly with respect to $\sigma^2$ rather than with
respect to $\sigma$.

## 7. Why the sign moves particles toward the denoiser

During denoising,

$$
\sigma_{i+1}<\sigma_i,
$$

so

$$
\sigma_{i+1}^2-\sigma_i^2<0.
$$

The current direction is

$$
\frac{x-D}{\sigma^2}=-s_\sigma(x).
$$

Multiplying the negative variance step by the negative score gives motion in
the positive score direction:

$$
\frac{\sigma_{i+1}^2-\sigma_i^2}{2}
\frac{x-D}{\sigma^2}
=
\frac{\sigma_i^2-\sigma_{i+1}^2}{2}
s_\sigma(x).
$$

Because

$$
s_\sigma(x)
=
\frac{D-x}{\sigma^2},
$$

this moves the particle toward the denoised estimate $D$.

## 8. The proposed expression

The expression

$$
-\frac{x-D}{\sigma^2}\,d\sigma
$$

is equal to

$$
s_\sigma(x)\,d\sigma.
$$

That uses the score as though it were the derivative with respect to sigma.
It is not: the correct probability-flow equations are

$$
\boxed{
dx=-\sigma s_\sigma(x)\,d\sigma
}
$$

or, equivalently,

$$
\boxed{
dx=-\frac12s_\sigma(x)\,d(\sigma^2).
}
$$

Since $d\sigma<0$ during denoising, the proposed
$s_\sigma(x)\,d\sigma$ would move opposite to the score. The required
$-\sigma$ factor both corrects the coordinate scaling and gives the correct
denoising direction.

## 9. What this means for the guided sampler

The base transport in the current sampler is a valid variance-coordinate
Euler discretization:

$$
x_{i+1}
=
\hat x_i
+
\frac{\sigma_{i+1}^2-\hat\sigma_i^2}{2}
\frac{\hat x_i-D(\hat x_i,\hat\sigma_i)}{\hat\sigma_i^2}.
$$

The separate concern is the potential-guidance correction added inside
$D$. That correction was derived for the old sigma-space update and must be
rederived if the variance-space transport is retained. The base score
transport itself is consistent with the probability-flow ODE.

## 10. Add reward guidance

The desired tilted marginal is

$$
\pi_\sigma(x)
\propto
q_\sigma(x)
\exp\left(\beta_\sigma(\sigma)r(x)\right).
$$

Its score is

$$
\nabla_x\log\pi_\sigma(x)
=
s_\sigma(x)
+
\beta_\sigma(\sigma)\nabla r(x).
$$

The reheating operation adds Gaussian variance

$$
\hat\sigma_i^2-\sigma_i^2.
$$

The reward-guidance strength is simply

$$
\boxed{
\beta_\sigma(\hat\sigma_i)
(\hat\sigma_i^2-\sigma_i^2).
}
$$

The two factors have direct meanings:

- $\beta_\sigma(\hat\sigma_i)$ is the reward strength at the reheated noise
  level;
- $\hat\sigma_i^2-\sigma_i^2$ is the variance added by reheating.

There is no additional factor of $\hat\sigma_i$ in the desired particle
motion. Probability flow contributes a factor of $1/2$, and the scalar
strength acts in the reward direction $\nabla r(\hat x_i)$. The resulting
guidance displacement is

$$
\boxed{
\delta x_{\mathrm{guidance}}
=
\frac12
\beta_\sigma(\hat\sigma_i)
(\hat\sigma_i^2-\sigma_i^2)
\nabla r(\hat x_i).
}
$$

The base-score part is already included in the denoising transport from
$\hat\sigma_i$ to $\sigma_{i+1}$. Thus the clean guided update in
$\sigma^2$ is

$$
\boxed{
\begin{aligned}
x_{i+1}
=
\hat x_i
&+
\frac{\hat\sigma_i^2-\sigma_{i+1}^2}{2}
s_{\hat\sigma_i}(\hat x_i)\\
&+
\frac{\hat\sigma_i^2-\sigma_i^2}{2}
\beta_\sigma(\hat\sigma_i)
\nabla r(\hat x_i).
\end{aligned}
}
$$

This equation has a direct interpretation:

- the first added term transports the reheated particle toward the base
  density;
- the second added term transports it toward high reward;
- only the reward term uses the variance added by reheating.

### Express this update as a drift

The sampler now has the same shape as a conventional first-order solver:

$$
x_{i+1}
=
\hat x_i
+
\frac{\sigma_{i+1}^2-\hat\sigma_i^2}{2}
b_i.
$$

The callback returns the complete drift $b_i$ with respect to
$\sigma^2/2$. Matching the clean guided update above gives

$$
\boxed{
\begin{aligned}
b_i
=
&-s_{\hat\sigma_i}(\hat x_i)\\
&-
\frac{\hat\sigma_i^2-\sigma_i^2}
{\hat\sigma_i^2-\sigma_{i+1}^2}
\beta_\sigma(\hat\sigma_i)
\nabla r(\hat x_i).
\end{aligned}
}
$$

This naturally separates into a score drift and a guidance drift:

$$
b_i^{\mathrm{score}}
=
-s_{\hat\sigma_i}(\hat x_i),
\qquad
b_i^{\mathrm{guidance}}
=
-
\frac{\hat\sigma_i^2-\sigma_i^2}
{\hat\sigma_i^2-\sigma_{i+1}^2}
\beta_\sigma(\hat\sigma_i)
\nabla r(\hat x_i).
$$

Multiplying the guidance drift by the solver step size cancels the full
transport interval:

$$
\frac{\sigma_{i+1}^2-\hat\sigma_i^2}{2}
b_i^{\mathrm{guidance}}
=
\frac12
\beta_\sigma(\hat\sigma_i)
(\hat\sigma_i^2-\sigma_i^2)
\nabla r(\hat x_i).
$$

## 11. Calculate the FKC weights

The tilted target is

$$
\pi_\sigma(x)
\propto
q_\sigma(x)
\exp\left(\beta_\sigma(\sigma)r(x)\right).
$$

The proposal moves particles using the guided dynamics from section 10.
FKC weights correct the remaining change of measure so that the weighted
particle population represents $\pi_\sigma$.

For the reward used by the test, $r(x)$ has no explicit dependence on
$\sigma$. The continuous log-weight differential in $\sigma^2$ is

$$
\boxed{
d\log w
=
r(x)\,d\beta_\sigma
-
\frac12
\beta_\sigma(\sigma)
\left\langle
\nabla r(x),
s_\sigma(x)
\right\rangle
d(\sigma^2).
}
$$

There are two contributions.

### Change in tilt strength

As the sampler moves from $\sigma_i$ to $\sigma_{i+1}$, the reward tilt
changes from $\beta_\sigma(\sigma_i)$ to
$\beta_\sigma(\sigma_{i+1})$. Holding the particle fixed at the reheated
state gives

$$
\left[
\beta_\sigma(\sigma_{i+1})
-
\beta_\sigma(\sigma_i)
\right]
r(\hat x_i).
$$

This term increases the relative weight of particles with high reward as
the target tilt becomes stronger.

### Reward-score alignment

The second term is integrated over the decreasing base variance interval:

$$
\begin{aligned}
&-
\frac12
\beta_\sigma(\hat\sigma_i)
\left\langle
\nabla r(\hat x_i),
s_{\hat\sigma_i}(\hat x_i)
\right\rangle
\left(\sigma_{i+1}^2-\sigma_i^2\right)\\
&=
\frac{\sigma_i^2-\sigma_{i+1}^2}{2}
\beta_\sigma(\hat\sigma_i)
\left\langle
\nabla r(\hat x_i),
s_{\hat\sigma_i}(\hat x_i)
\right\rangle.
\end{aligned}
$$

This measures whether the reward gradient and the base-density score point
in similar directions:

- a positive inner product increases the log weight;
- a negative inner product decreases the log weight.

### Complete discrete weight increment

Combining both contributions gives exactly the callback used by the test:

$$
\boxed{
\begin{aligned}
\Delta\log w_i
=
&\left[
\beta_\sigma(\sigma_{i+1})
-
\beta_\sigma(\sigma_i)
\right]
r(\hat x_i)\\
&+
\frac{\sigma_i^2-\sigma_{i+1}^2}{2}
\beta_\sigma(\hat\sigma_i)
\left\langle
\nabla r(\hat x_i),
s_{\hat\sigma_i}(\hat x_i)
\right\rangle.
\end{aligned}
}
$$

The implementation is:

```python
score = gmm.score(x, schedule.time(sigma_hat))

potential = (
    beta_sigma(sigma_next) - beta_sigma(sigma_curr)
) * reward(x)

alignment = (
    beta_sigma(sigma_hat)
    * grad_reward(x)
    * score
    * (sigma_curr**2 - sigma_next**2)
    / 2.0
)

delta_log_weight = potential + alignment
```

The callback is evaluated at the reheated state $\hat x_i$, but the
alignment term uses the base variance difference

$$
\sigma_i^2-\sigma_{i+1}^2.
$$

It does not use

$$
\hat\sigma_i^2-\sigma_{i+1}^2,
$$

because reheating is an auxiliary proposal operation. It changes the state
at which the FKC increment is evaluated, but the weight differential still
belongs to the base progression from $\sigma_i$ to $\sigma_{i+1}$.

### Accumulate and normalize

The sampler accumulates the increments in log space:

$$
\log w
\leftarrow
\log w+\Delta\log w_i.
$$

The normalized weights are

$$
\widetilde w
=
\operatorname{softmax}(\log w).
$$

The normalized effective sample size is

$$
\frac{\operatorname{ESS}}{N}
=
\frac{1}
{N\lVert\widetilde w\rVert_2^2}.
$$

If ESS becomes too small, systematic resampling converts the nonuniform
weights into particle multiplicities and resets the log weights to zero.
Between resampling events, the **weighted** particle population, rather than
the unweighted histogram, is the approximation of the tilted target.


Sources: Skreta et al., *Feynman-Kac Correctors in Diffusion* (ICML 2025,
arXiv:2503.02819); Karras et al., *Elucidating the Design Space of
Diffusion-Based Generative Models* (NeurIPS 2022).