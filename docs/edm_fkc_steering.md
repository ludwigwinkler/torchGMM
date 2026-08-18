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
This is of interest to the EDM solver used in Alphafold3 which for the last 30% or so of reverse diffusion switches to $\alpha=0$.

## FKC correctors in $\sigma$-space

For VE/EDM marginals $x_\sigma=x_0+\sigma\epsilon$, let

$$
\Delta_i=\sigma_i^2-\sigma_{i+1}^2>0,\qquad
s_\sigma(x)=\nabla_x\log q(x;\sigma)
=-\frac{x - D_\theta(x;\sigma)}{\sigma^2}.
\tag{3}
$$

For the Karras grid, with generation time $t\in[0,1]$,

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

Using the exact variance gap
$\Delta_i=\sigma_i^2-\sigma_{i+1}^2$, a first-order step for (9) is

$$
\boxed{\;
\begin{aligned}
x_{i+1}=x_i
&+\left[
\frac{1+\alpha^2}{2}s_{\sigma_i}(x_i)
+\frac{\alpha^2\beta_i}{2}\nabla r(x_i,\sigma_i)
\right]\Delta_i\\
&+\alpha\sqrt{\Delta_i}\,\epsilon_i,
\qquad \epsilon_i\sim\mathcal N(0,I).
\end{aligned}
\;}
\tag{11}
$$

Evaluating (10) over the same gap gives

$$
\boxed{\;
\Delta w_i=
\beta_{i+1}r(x_i,\sigma_{i+1})
-\beta_i r(x_i,\sigma_i)
+\frac{1}{2}
\left\langle\beta_i\nabla r(x_i,\sigma_i),s_{\sigma_i}(x_i)\right\rangle
\Delta_i.
\;}
\tag{12}
$$

For a time-independent reward, the first two terms in (12) reduce to
$r(x_i)(\beta_{i+1}-\beta_i)$. The standard reverse SDE is $\alpha=1$;
$\alpha=0$ is the probability-flow ODE, for which steering acts entirely
through the weights.

EDM churn uses an exact reheat followed by deterministic transport rather
than the Euler step (11). In that split scheme, evaluate (10) at the
reheated state and advance the weight by the net variance gap $\Delta_i$.
At the terminal $\sigma=0$ step, skip weight accumulation because the score
is singular.

Sources: Skreta et al., *Feynman-Kac Correctors in Diffusion* (ICML 2025,
arXiv:2503.02819); Karras et al., *Elucidating the Design Space of
Diffusion-Based Generative Models* (NeurIPS 2022).