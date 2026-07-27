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
