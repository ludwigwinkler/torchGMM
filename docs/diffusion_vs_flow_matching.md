# Diffusion vs Flow Matching: Compact Reference

## 1. The Unifying Interpolation

Both frameworks define a Gaussian path from data to noise:

$$x_t = \alpha_t x_0 + \sigma_t \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, I)$$

The GMM marginal at time $t$ is exact in closed form for any schedule:

$$p_t(x) = \sum_k \pi_k \, \mathcal{N}\!\left(x;\, \alpha_t \mu_k,\; \sigma_t^2 I + \alpha_t^2 \Sigma_k\right)$$

---

## 2. Schedules

| Schedule | $\alpha_t$ | $\sigma_t$ | $\dot{\alpha}_t$ | $\dot{\sigma}_t$ | Constraint |
|---|---|---|---|---|---|
| VP-SDE | $\exp\!\left(-\tfrac{1}{2}\int_0^t \beta(s)\,ds\right)$ | $\sqrt{1-\alpha_t^2}$ | $-\tfrac{1}{2}\beta(t)\,\alpha_t$ | $\tfrac{1}{2}\beta(t)\,\alpha_t^2/\sigma_t$ | $\alpha_t^2+\sigma_t^2=1$ |
| Flow Matching | $1-t$ | $t$ | $-1$ | $1$ | $\alpha_t+\sigma_t=1$ |

VP-SDE uses a linear $\beta$ schedule: $\beta(t) = \beta_{\min} + t(\beta_{\max} - \beta_{\min})$.

---

## 3. Score Function

$$s_t(x) = \nabla_x \log p_t(x)$$

Exact for a GMM via the marginal log-prob. For a single data point $x_0$:

$$\nabla_x \log p_t(x \mid x_0) = -\frac{x - \alpha_t x_0}{\sigma_t^2} = -\frac{\varepsilon}{\sigma_t}$$

---

## 4. Deriving the Velocity Field from the Score

The marginal velocity field is the expected instantaneous rate of change, conditioned on the current state:

$$v_t(x) = \mathbb{E}\!\left[\frac{dx_t}{dt}\;\Big|\; x_t = x\right]$$

Since $x_t = \alpha_t x_0 + \sigma_t \varepsilon$, differentiating gives $\frac{dx_t}{dt} = \dot{\alpha}_t x_0 + \dot{\sigma}_t \varepsilon$, so:

$$v_t(x) = \dot{\alpha}_t\, \mathbb{E}[x_0 \mid x_t = x] \;+\; \dot{\sigma}_t\, \mathbb{E}[\varepsilon \mid x_t = x]$$

**Tweedie's identities** express both posterior expectations in terms of the score $s_t(x) = \nabla_x \log p_t(x)$.

Starting from the conditional score $\nabla_x \log p(x_t \mid x_0) = -(x_t - \alpha_t x_0)/\sigma_t^2$ and taking the expectation under $p(x_0 \mid x_t = x)$:

$$s_t(x) = \mathbb{E}\!\left[-\frac{x - \alpha_t x_0}{\sigma_t^2}\;\Big|\; x_t = x\right] = -\frac{x - \alpha_t\,\mathbb{E}[x_0 \mid x_t = x]}{\sigma_t^2}$$

Rearranging:

$$\mathbb{E}[x_0 \mid x_t = x] = \frac{x + \sigma_t^2\, s_t(x)}{\alpha_t}$$

For the noise, using $\varepsilon = (x_t - \alpha_t x_0)/\sigma_t$:

$$\mathbb{E}[\varepsilon \mid x_t = x] 
= \frac{x - \alpha_t\,\mathbb{E}[x_0 \mid x_t = x]}{\sigma_t} 
= - \left(- \frac{\sigma_t}{\sigma_t} \right) \frac{x - \alpha_t\,\mathbb{E}[x_0 \mid x_t = x]}{\sigma_t} 
= -\sigma_t\, s_t(x)
$$

**Substituting** both into the velocity:

$$v_t(x) = \dot{\alpha}_t \cdot \frac{x + \sigma_t^2\, s_t(x)}{\alpha_t} + \dot{\sigma}_t \cdot \bigl(-\sigma_t\, s_t(x)\bigr)$$

With the identities
$$
\alpha_t = 1-t \rightarrow \dot{\alpha}_t = -1, \qquad
\sigma_t = t \rightarrow \dot{\sigma}_t = 1
$$

$$\boxed{v_t(x) = \frac{\dot{\alpha}_t}{\alpha_t}\, x \;+\; \left(\frac{\dot{\alpha}_t\, \sigma_t^2}{\alpha_t} - \dot{\sigma}_t\, \sigma_t\right) s_t(x)}
$$

This simplifies to

$$
\begin{align*}
v_t(x) 
&= \frac{\dot{\alpha}_t}{\alpha_t}\, x + \left(\frac{\dot{\alpha}_t \sigma_t^2}{\alpha_t} - \dot{\sigma}_t \sigma_t \right) s_t(x) \\
&= \frac{-x}{1-t} + \left(\frac{-t^2}{1-t} - t\right) s_t(x)\\
&= \frac{-x}{1-t} + \left(\frac{-t^2}{1-t} - t\frac{1-t}{1-t}\right) s_t(x)
\end{align*}
$$

This is the general velocity–score identity. It holds for any schedule and any data distribution — for a GMM, $s_t(x)$ is exact via autograd on the closed-form marginal $\log p_t(x)$.

**Specialisations:**

$$\text{VP-SDE:} \qquad v_t(x) = -\tfrac{1}{2}\beta(t)\bigl(x + s_t(x)\bigr)$$

$$\text{Flow Matching:} \qquad v_t(x) = -\frac{x + t\, s_t(x)}{1-t}$$

---

## 5. Governing Equations

| Mode | Direction | Equation |
|---|---|---|
| Forward ODE | $0 \to 1$ | $dx = v_t(x)\,dt$ |
| Reverse ODE | $1 \to 0$ | $dx = v_t(x)\,dt \quad (dt < 0)$ |
| VP-SDE forward | $0 \to 1$ | $dX = -\tfrac{1}{2}\beta(t)\,X\,dt + \sqrt{\beta(t)}\,dW$ |
| VP-SDE reverse (Anderson) | $1 \to 0$ | $dX = \bigl[-\tfrac{1}{2}\beta(t)\,X - \frac{1}{2}\beta(t) \ (1+\gamma_t^2) s_t(X)\bigr]dt + \gamma_t \sqrt{\beta(t)}\,d\tilde{W}$ |
| Stochastic augmentation | $1 \to 0$ | $dX = \bigl[v_t(X) - \tfrac{1}{2}\gamma^2 s_t(X)\bigr]dt + \gamma\,d\tilde{W}$ |

$\gamma=0$ recovers the ODE; $\gamma=\sqrt{\beta(t)}$ recovers the Anderson reverse SDE expressed via $v_t$.

---

## 6. Euler-Maruyama Integration

Single update rule covering all cases:

$$x_{t+dt} = x_t + \mathrm{drift}(x_t, t)\cdot dt + \mathrm{diffusion}(t)\cdot\sqrt{|dt|}\cdot\varepsilon$$

`diffusion=None` → pure ODE (no noise term).

| Mode | drift | diffusion |
|---|---|---|
| Forward ODE | $v_t(x)$ | — |
| Reverse ODE | $v_t(x)$ | — |
| VP-SDE forward | $-\tfrac{1}{2}\beta(t)\,x$ | $\sqrt{\beta(t)}$ |
| VP-SDE reverse | $-\tfrac{1}{2}\beta(t)\,x - \beta(t)\,s_t(x)$ | $\sqrt{\beta(t)}$ |
| Stochastic augmentation | $v_t(x) - \tfrac{1}{2}\gamma^2 s_t(x)$ | $\gamma$ |

---

## 7. Initialisation

$$\text{BetaSchedule:} \qquad x_{\text{start}} \sim \mathcal{N}(0, I) \quad \text{at } t_{\text{start}} = 1$$

$$\text{FlowMatchingSchedule:} \qquad x_{\text{start}} \sim \mathcal{N}(0, I) \quad \text{at } t_{\text{start}} = 1 - \varepsilon \quad \text{(avoids } 1/(1-t) \text{ singularity)}$$
