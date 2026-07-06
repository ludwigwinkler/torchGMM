# Karras Noise Schedule and Its Implied Forward Process

This note describes the `KarrasSchedule` used in `torchGMM`. It only covers the
noise schedule and the Euler-Maruyama SDE it implies. It deliberately ignores the
EDM/Karras denoiser parameterization, preconditioning, and sampler corrections.

## Schedule definition

`KarrasSchedule` is a variance-exploding (VE) schedule:

$$\alpha(t) \equiv 1,\qquad t\in[0,1].$$

The marginal noise standard deviation is

$$\bar\sigma(t)
= \sigma_{\mathrm{data}}\left(
\sigma_{\min}^{1/\rho}
+ t\left(\sigma_{\max}^{1/\rho}-\sigma_{\min}^{1/\rho}\right)
\right)^\rho.$$

Equivalently, if

$$u(t)=\sigma_{\min}^{1/\rho}
+ t\left(\sigma_{\max}^{1/\rho}-\sigma_{\min}^{1/\rho}\right),$$

then

$$\bar\sigma(t)=\sigma_{\mathrm{data}}u(t)^\rho.$$

So time is linear in $\bar\sigma^{1/\rho}$, not in $\bar\sigma$. The parameter
$\rho$ controls how the grid concentrates. For $\rho=1$, the schedule is linear
in $\bar\sigma$; for the common EDM/AF3 value $\rho=7$, the schedule spends more
resolution near low noise.

The derivative used by the SDE is

$$\dot{\bar\sigma}(t)
= \sigma_{\mathrm{data}}\,\rho\,u(t)^{\rho-1}
\left(\sigma_{\max}^{1/\rho}-\sigma_{\min}^{1/\rho}\right).$$

## Implied marginal path

For a clean data distribution $q_{\mathrm{data}}$, the schedule defines the
Gaussian-convolved marginal

$$q_t = q_{\mathrm{data}} * \mathcal{N}\!\left(0,\bar\sigma(t)^2I\right).$$

For a GMM, this stays exact:

$$q_t(x)=\sum_k \pi_k\,
\mathcal{N}\!\left(x;\mu_k,\Sigma_k+\bar\sigma(t)^2I\right).$$

The important endpoint detail is that `KarrasSchedule` has
$\bar\sigma(0)=\sigma_{\mathrm{data}}\sigma_{\min}$, not zero. Thus `t=0` is a
small-noise endpoint, not mathematically exact clean data unless
$\sigma_{\min}=0$. In practice $\sigma_{\min}$ is chosen tiny enough that this is
treated as the data endpoint.

## Implied forward SDE

Start from the desired noising path itself. Couple all times with the same clean
sample $X_{\mathrm{data}}$ and the same Gaussian noise $\epsilon$:

$$X_t = X_{\mathrm{data}} + \bar\sigma(t)\epsilon,
\qquad \epsilon\sim\mathcal{N}(0,I),$$

where $X_{\mathrm{data}}\sim q_{\mathrm{data}}$. At each fixed time $t$, this
implies the marginal density

$$q_t = q_{\mathrm{data}} * \mathcal{N}\!\left(0,\bar\sigma(t)^2I\right).$$

Now take the time derivative of the marginal variance. The clean-data variance
does not change with $t$, and the only time-dependent part is the added noise:

$$\frac{d}{dt}\mathrm{Var}[X_t]
= \frac{d}{dt}\mathrm{Var}\!\left[X_{\mathrm{data}}+\bar\sigma(t)\epsilon\right]
= \frac{d}{dt}\mathrm{Var}\!\left[\bar\sigma(t)\epsilon\right].$$

Since $\epsilon\sim\mathcal{N}(0,I)$,

$$\mathrm{Var}\!\left[\bar\sigma(t)\epsilon\right]=\bar\sigma(t)^2I,$$

so

$$\frac{d}{dt}\mathrm{Var}[X_t]
=\frac{d}{dt}\bar\sigma(t)^2I
=2\bar\sigma(t)\dot{\bar\sigma}(t)I.$$

A driftless VE SDE,

$$dX_t = g(t)\,dW_t,$$

Brownian motion contributes covariance $g(t)^2I\,dt$ over an infinitesimal
interval $dt$, so

$$\frac{d}{dt}\mathrm{Var}(X_t)=g(t)^2I.$$

Therefore, to make the SDE have the same marginal variance growth as the
differentiated noising path, choose

$$g(t)^2=\frac{d}{dt}\bar\sigma(t)^2
=2\bar\sigma(t)\dot{\bar\sigma}(t).$$

The final equality is just the chain rule. Thus $\bar\sigma(t)$ is the marginal
noise standard deviation, while $g(t)$ is the instantaneous SDE diffusion
coefficient; they are related by accumulated variance, not by equality.

The critical notation pitfall is:

$$\bar\sigma(t)^2 \neq g(t)^2.$$

$\bar\sigma(t)^2$ is the **marginal noise variance already accumulated** by time
$t$. $g(t)^2$ is the **instantaneous variance injection rate** at time $t$. Its
time derivative, $\frac{d}{dt}g(t)^2$, is the curvature of that injection rate and
is not the quantity needed to match the marginal path. The correct direction is

$$\bar\sigma(t)\;\longrightarrow\;\bar\sigma(t)^2
\;\longrightarrow\;\frac{d}{dt}\bar\sigma(t)^2
=2\bar\sigma(t)\dot{\bar\sigma}(t)
\;\longrightarrow\;g(t)^2.$$

Conversely, if we start from the instantaneous diffusion coefficient, the
marginal noise variance is obtained by integration:

$$\bar\sigma(t)^2-\bar\sigma(t_0)^2
=\int_{t_0}^{t}g(s)^2\,ds.$$

Using $g(s)^2=2\bar\sigma(s)\dot{\bar\sigma}(s)$, this is just a change of
variables:

$$\int_{t_0}^{t}g(s)^2\,ds
=\int_{t_0}^{t}2\bar\sigma(s)\dot{\bar\sigma}(s)\,ds
=\int_{\bar\sigma(t_0)}^{\bar\sigma(t)}2\sigma\,d\sigma
=\bar\sigma(t)^2-\bar\sigma(t_0)^2.$$

So integrating $2\sigma\,d\sigma$ recovers the **marginal variance increment**,
not a marginal value of $g(t)^2$. The quantity $g(t)^2$ is the derivative of that
marginal variance with respect to time.

This is exactly what `KarrasSchedule.diffusion_coeff(t)` implements:

$$g(t)=\sqrt{2\bar\sigma(t)\dot{\bar\sigma}(t)}.$$

Thus the implied forward SDE is

$$dX_t = \sqrt{2\bar\sigma(t)\dot{\bar\sigma}(t)}\,dW_t.$$

Its Fokker-Planck equation is

$$\partial_t q_t = \frac{1}{2}g(t)^2\Delta q_t
= \frac{1}{2}\frac{d}{dt}\bar\sigma(t)^2\Delta q_t.$$

This agrees with the heat-kernel identity for Gaussian convolution:

$$\partial_t\left(q_{\mathrm{data}} * \mathcal{N}(0,\bar\sigma(t)^2I)\right)
= \frac{1}{2}\frac{d}{dt}\bar\sigma(t)^2
\Delta\left(q_{\mathrm{data}} * \mathcal{N}(0,\bar\sigma(t)^2I)\right).$$

Strictly speaking, if the SDE starts at $X_0\sim q_0$, then
$q_0=q_{\mathrm{data}} * \mathcal{N}(0,\bar\sigma(0)^2I)$. Starting from exactly
clean data instead requires an initial noising step of variance
$\bar\sigma(0)^2$ before integrating the SDE.

## Forward Euler-Maruyama in continuous notation

In infinitesimal form, the forward SDE is

$$dX_t = g(t)\,dW_t,\qquad dW_t \sim \mathcal{N}(0,dt\,I),\qquad dt>0.$$

There is no deterministic term because `KarrasSchedule.forward_drift` returns
zero. The conditional covariance of the infinitesimal increment is

$$\mathrm{Var}[dX_t\mid X_t]=g(t)^2dt\,I.$$

Equivalently, the Euler-Maruyama local update over a small positive time
increment $dt$ is

$$X_{t+dt}=X_t+g(t)\sqrt{dt}\,\epsilon,\qquad
\epsilon\sim\mathcal{N}(0,I).$$

The exact forward transition variance over the same interval is

$$\bar\sigma(t+dt)^2-\bar\sigma(t)^2
=\int_t^{t+dt}g(s)^2\,ds.$$

Euler-Maruyama uses the local approximation

$$\int_t^{t+dt}g(s)^2\,ds \approx g(t)^2dt.$$

For a finite interval where this approximation is too coarse, an exact
forward-noising transition can instead sample

$$X_{t+dt}=X_t
+\sqrt{\bar\sigma(t+dt)^2-\bar\sigma(t)^2}\,\epsilon,$$

which is available because the Karras marginal variance is known in closed form.

## Reverse SDE under the same schedule

Let

$$s_t(x)=\nabla_x\log q_t(x).$$

Since the forward drift is zero, the Anderson reverse-time SDE is

$$dX_t = -g(t)^2s_t(X_t)\,dt + g(t)\,d\bar W_t,$$

when integrated with decreasing time, so $dt<0$. In the repository's
`reverse_sampling` convention, the drift callable is

```python
def reverse_drift(x, t):
    g = schedule.diffusion_coeff(t)
    return -(g**2) * gmm.score(x, t)
```

and the infinitesimal Euler-Maruyama update is

$$X_{t+dt}
= X_t - g(t)^2s_t(X_t)\,dt
+ g(t)\sqrt{|dt|}\epsilon,\qquad dt<0.$$

Because $dt<0$, the deterministic part moves in the denoising direction
$+g(t)^2s_t(X_t)|dt|$.

## Conditional reverse SDE in terms of $\bar\sigma(t)$, $x_T$, and $x_0$

For a single clean point $x_0$, first assume the ideal clean-endpoint convention
$\bar\sigma(0)=0$. Then the VE marginal is

$$q_t(x\mid x_0)=\mathcal{N}\!\left(x;x_0,\bar\sigma(t)^2I\right),$$

so the conditional score is

$$s_t(x\mid x_0)=\nabla_x\log q_t(x\mid x_0)
=-\frac{x-x_0}{\bar\sigma(t)^2}.$$

Plugging this score into the reverse SDE gives the clean-point conditional
reverse process

$$dX_t
= g(t)^2 \frac{\bigl(X_t-x_0\bigr)}{\bar\sigma(t)^2} \,dt
+ g(t)\,d\bar W_t,\qquad dt<0.$$

Since $g(t)^2=2\bar\sigma(t)\dot{\bar\sigma}(t)$, this can be written purely in
terms of the schedule as

$$
\begin{align}
dX_t
&= g(t)^2 \frac{\bigl(X_t-x_0\bigr)}{\bar\sigma(t)^2} \,dt
+ g(t)\,d\bar W_t,\qquad dt<0 \\
&= 2\bar\sigma(t)\dot{\bar\sigma}(t) \frac{\bigl(X_t-x_0\bigr)}{\bar\sigma(t)^2} \,dt
+ g(t)\,d\bar W_t,\qquad dt<0 \\
&= 2\dot{\bar\sigma}(t)
\frac{X_t-x_0}{\bar\sigma(t)}\,dt
+ \sqrt{2\bar\sigma(t)\dot{\bar\sigma}(t)}\,d\bar W_t,\qquad dt<0
\end{align}$$

If the reverse chain is initialized at a terminal noisy sample $X_T=x_T$, then
this SDE is run backward from

$$X_T=x_T.$$

The deterministic part moves toward $x_0$ because $dt<0$:

$$2\dot{\bar\sigma}(t)\frac{X_t-x_0}{\bar\sigma(t)}\,dt
= -2\dot{\bar\sigma}(t)\frac{X_t-x_0}{\bar\sigma(t)}|dt|.$$

If one also wants the process to be **pinned** to both endpoints,
$X_0=x_0$ and $X_T=x_T$, that is a diffusion bridge, not the ordinary reverse
SDE above. The Gaussian bridge marginal at an intermediate time is

$$X_t\mid x_0,x_T
\sim \mathcal{N}\!\left(
x_0+\frac{\bar\sigma(t)^2}{\bar\sigma(T)^2}(x_T-x_0),
\;\bar\sigma(t)^2\left(1-\frac{\bar\sigma(t)^2}{\bar\sigma(T)^2}\right)I
\right).$$

This endpoint-conditioned bridge is useful conceptually, but it is a different
conditioning problem from the standard score-based reverse SDE, which conditions
on the current noisy state distribution and is initialized at $x_T$.

For the literal Karras schedule in this repository, $\bar\sigma(0)>0$. To keep
the clean endpoint exact, replace every occurrence of $\bar\sigma(t)^2$ in the
conditional formulas above by the variance clock

$$V(t)=\bar\sigma(t)^2-\bar\sigma(0)^2.$$

For example,

$$q_t(x\mid x_0)=\mathcal{N}\!\left(x;x_0,V(t)I\right),\qquad
s_t(x\mid x_0)=-\frac{x-x_0}{V(t)},$$

and the clean-point conditional reverse SDE becomes

$$dX_t
= \frac{g(t)^2}{V(t)}\bigl(X_t-x_0\bigr)\,dt
+ g(t)\,d\bar W_t,\qquad dt<0.$$

## Practical consequence

For large $\sigma_{\max} = 160$ and $\rho=7$,

$$g(t)^2=2\bar\sigma(t)\dot{\bar\sigma}(t)$$

can become very large near $t=1$. Euler-Maruyama simulations are therefore
sensitive to the high-noise endpoint: stable runs usually need a sufficiently
fine grid, or a start time slightly below $1$, or an exact-in-variance forward
transition when only forward noising is needed.

## Matching FKC Steering Drift

Now combine the Karras VE schedule with the FKC steering equations from
`docs/fkc_steering.md`. The reward-tilted marginal is

$$p_t(x)\propto q_t(x)\exp(\beta(t)r(x,t)).$$

For Karras,

$$f_t(x)\equiv 0,\qquad
g(t)^2=2\bar\sigma(t)\dot{\bar\sigma}(t),\qquad
s_t(x)=\nabla_x\log q_t(x).$$

The unsteered reverse-time dynamics, written in a positive reverse-time
increment $d\tau=-dt>0$, are

$$dX_t = g(t)^2s_t(X_t)\,d\tau + g(t)\,dW_\tau.$$

FKC adds the drift

$$\beta(t)\frac{g(t)^2}{2}\nabla r(X_t,t)
= \beta(t)\bar\sigma(t)\dot{\bar\sigma}(t)\nabla r(X_t,t),$$

so the Karras FKC steered reverse SDE is

$$
\begin{align}
dX_t
&= \left[
g(t)^2s_t(X_t)
+ \beta(t)\frac{g(t)^2}{2}\nabla r(X_t,t)
\right]d\tau
+ g(t)\,dW_\tau \\
&= 2\bar\sigma(t)\dot{\bar\sigma}(t)
\left[
s_t(X_t)+\frac{\beta(t)}{2}\nabla r(X_t,t) 
\right]d\tau
+ \sqrt{2\bar\sigma(t)\dot{\bar\sigma}(t)}\,dW_\tau
\end{align}
$$

The corresponding FKC log-weight increment is

$$
\begin{align}
dw_t
&= \left[
-\dot\beta(t)r(X_t,t)
-\beta(t)\partial_t r(X_t,t)
+ \left\langle\beta(t)\nabla r(X_t,t),
\frac{g(t)^2}{2}s_t(X_t)
\right\rangle
\right]d\tau \\
&= \left[
-\dot\beta(t)r(X_t,t)
-\beta(t)\partial_t r(X_t,t)
+ \beta(t)\bar\sigma(t)\dot{\bar\sigma}(t)
\left\langle\nabla r(X_t,t),s_t(X_t)\right\rangle
\right]d\tau
\end{align}$$

The repository's samplers use a decreasing forward-time grid directly, so
$dt<0$ and $d\tau=|dt|=-dt$. In that convention, the drift callable passed to
`steered_reverse_sampling` is the signed-forward-time version

```python
def guided_drift(x, t):
    g = schedule.diffusion_coeff(t)
    score = gmm.score(x, t)
    return -(g**2) * score - beta(t) * (g**2 / 2) * grad_r(x, t)
```

and the weight update returns the positive reverse-time increment

```python
def weight_update(x, t, dt):
    g = schedule.diffusion_coeff(t)
    score = gmm.score(x, t)
    integrand = (
        -dbeta_dt(t) * r(x, t)
        - beta(t) * partial_t_r(x, t)
        + beta(t) * ((g**2) / 2) * (grad_r(x, t) * score).sum(dim=-1)
    )
    return integrand * dt.abs()
```

If the reward has no explicit time dependence, the `partial_t_r` term is zero.

### Matching $\beta(t)$ to balance score and reward gradients

Consider the drift

$$
\begin{align}
\text{drift}: 
&\quad 2\bar\sigma(t)\dot{\bar\sigma}(t)
\left[
s_t(X_t)+\frac{\beta(t)}{2}\nabla r(X_t,t) 
\right]d\tau \\ 
&= 2\bar\sigma(t)\dot{\bar\sigma}(t)
\left[
\frac{(X_t - x0)}{\bar\sigma(t)^2}+\frac{\beta(t)}{2}\nabla r(X_t,t) 
\right]d \tau
\end{align}$$

Now imagine that $r(X_t, t)$ is a simple quadratic reward, such that we have $r(X_t, t)= (X_t - b)^2$ for some target $b$. Then $\nabla r(X_t, t) = 2(X_t - b)$. The drift becomes
$$
\begin{align}
\text{drift}:
&= 2\bar\sigma(t)\dot{\bar\sigma}(t)
\left[
\frac{(X_t - x0)}{\bar\sigma(t)^2} + \beta(t)(X_t - b) 
\right]d \tau
\end{align}$$

We can then choose $\beta(t)$ such that the two terms balance each other, i.e., we want
$$\beta(t) = \frac{1}{\bar\sigma(t)^2}$$