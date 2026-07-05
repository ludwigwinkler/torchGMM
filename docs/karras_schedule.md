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

## Forward Euler-Maruyama

On an increasing time grid $0\le t_0<t_1<\cdots<t_T\le1$, the repository's
Euler-Maruyama loop applies

$$X_{i+1}=X_i+g(t_i)\sqrt{\Delta t_i}\,\epsilon_i,
\qquad \Delta t_i=t_{i+1}-t_i,\quad
\epsilon_i\sim\mathcal{N}(0,I).$$

There is no deterministic update because `KarrasSchedule.forward_drift` returns
zero.

The exact transition variance over a step is

$$\bar\sigma(t_{i+1})^2-\bar\sigma(t_i)^2
=\int_{t_i}^{t_{i+1}}g(s)^2\,ds.$$

Euler-Maruyama approximates this by $g(t_i)^2\Delta t_i$. For coarse grids, an
exact forward noising transition can instead sample

$$X_{i+1}=X_i
+\sqrt{\bar\sigma(t_{i+1})^2-\bar\sigma(t_i)^2}\,\epsilon_i,$$

which is available because the Karras marginal variance is known in closed form.

## Reverse SDE under the same schedule

Let

$$s_t(x)=\nabla_x\log q_t(x).$$

Since the forward drift is zero, the Anderson reverse-time SDE is

$$dX_t = -g(t)^2s_t(X_t)\,dt + g(t)\,d\bar W_t,$$

when integrated on a decreasing grid. In the repository's `reverse_sampling`
convention, `dt = t_{i+1}-t_i < 0`, so the drift callable is

```python
def reverse_drift(x, t):
    g = schedule.diffusion_coeff(t)
    return -(g**2) * gmm.score(x, t)
```

and the solver step is

$$X_{i+1}
= X_i - g(t_i)^2s_{t_i}(X_i)\Delta t_i
+ g(t_i)\sqrt{|\Delta t_i|}\epsilon_i.$$

Because $\Delta t_i<0$, the deterministic part moves in the denoising direction
$+g(t_i)^2s_{t_i}(X_i)|\Delta t_i|$.

## Practical consequence

For large $\sigma_{\max}$ and $\rho=7$,

$$g(t)^2=2\bar\sigma(t)\dot{\bar\sigma}(t)$$

can become very large near $t=1$. Euler-Maruyama simulations are therefore
sensitive to the high-noise endpoint: stable runs usually need a sufficiently
fine grid, or a start time slightly below $1$, or an exact-in-variance forward
transition when only forward noising is needed.
