# EDM, the Repository Sigma-Scale Sampler, and AlphaFold 3

This note separates four related but distinct constructions:

1. EDM Algorithm 2 as defined by Karras et al.
2. The sigma-grid sampler implemented by `steered_reverse_edm_sampling`.
3. AlphaFold 3 Supplementary Algorithm 18, **Sample Diffusion**.
4. The FKC-steered AF3 implementation,
   `steered_reverse_af3_sampling`.

All four use a variance-exploding marginal

$$
x_\sigma=x_0+\sigma\epsilon,\qquad \epsilon\sim\mathcal N(0,I),
$$

and integrate in the physical noise standard deviation $\sigma$, not in an
arbitrary normalized time coordinate.

## 1. EDM Algorithm 2

Let $\sigma_i>\sigma_{i+1}\geq0$ be consecutive levels on a decreasing noise
grid. EDM optionally reheats the current state before denoising:

$$
\gamma_i=
\begin{cases}
\min(S_{\rm churn}/N,\sqrt{2}-1),
&S_{\rm tmin}\leq\sigma_i\leq S_{\rm tmax},\\
0,&\text{otherwise},
\end{cases}
\qquad
\hat\sigma_i=(1+\gamma_i)\sigma_i,
$$

$$
\hat x_i=x_i+
S_{\rm noise}\sqrt{\hat\sigma_i^2-\sigma_i^2}\,\epsilon_i.
$$

For an exact VE process, the reheat is an exact forward transition when
$S_{\rm noise}=1$. Values other than one deliberately bias the injected
variance.

The denoiser defines the probability-flow direction

$$
d_i=\frac{\hat x_i-D_\theta(\hat x_i,\hat\sigma_i)}{\hat\sigma_i}.
$$

EDM first makes an Euler proposal

$$
x'_i=\hat x_i+(\sigma_{i+1}-\hat\sigma_i)d_i.
$$

Except on the terminal step to $\sigma=0$, Algorithm 2 evaluates the denoiser
again at $(x'_i,\sigma_{i+1})$ and applies a Heun correction:

$$
d'_i=\frac{x'_i-D_\theta(x'_i,\sigma_{i+1})}{\sigma_{i+1}},
\qquad
x_{i+1}=\hat x_i+
(\sigma_{i+1}-\hat\sigma_i)\frac{d_i+d'_i}{2}.
$$

The important points are:

- $\sigma$ is the integration coordinate, so the signed step is
  $\sigma_{i+1}-\hat\sigma_i<0$.
- Churn is a fractional increase of the current noise level.
- The usual $\sqrt{2}-1$ cap limits the injected variance to at most the
  current noise variance.
- The deterministic leg is normally second-order Heun and therefore uses
  approximately two denoiser evaluations per step.

See [Operator-Splitting Samplers](operatorsplitting_samplers.md) for the
probability-flow derivation.

## 2. How this repository uses the sigma scale

### Karras grid

`KarrasSchedule` constructs

$$
\sigma(t)=\sigma_{\rm data}
\left[
\sigma_{\min}^{1/\rho}
+t\left(\sigma_{\max}^{1/\rho}-\sigma_{\min}^{1/\rho}\right)
\right]^\rho,
\qquad t\in[0,1].
$$

Here $t$ only places the grid points. The sampler itself receives the
resulting strictly decreasing sigma values and steps directly in sigma space.
For reverse sampling, the schedule must therefore be evaluated from `t=1`
down to `t=0`.

For the AlphaFold 3 grid parameters:

```python
schedule = KarrasSchedule(
    sigma_min=4e-4,
    sigma_max=160.0,
    rho=7.0,
    sigma_data=16.0,
)
sigma = schedule.get_sigma_t(torch.linspace(1, 0, 201))
```

This produces $\sigma_0=2560$ and $\sigma_{200}=0.0064$. The factor
$\sigma_{\rm data}=16$ scales the actual coordinate noise; $\rho=7$ changes
only grid placement, concentrating steps near low noise.

### `steered_reverse_edm_sampling`

The repository's sigma-grid sampler uses

$$
\hat\sigma_i=
\begin{cases}
\sigma_i,&i=0,\\
\sigma_{i-1},&i>0,
\end{cases}
$$

$$
\hat x_i=x_i+
\lambda\sqrt{\hat\sigma_i^2-\sigma_i^2}\,\epsilon_i,
\qquad
d_i=\frac{\hat x_i-D_\theta(\hat x_i,\hat\sigma_i)}{\hat\sigma_i},
$$

$$
x_{i+1}=\hat x_i+
\eta(\sigma_{i+1}-\hat\sigma_i)d_i.
$$

The public arguments are `noise_scale` $=\lambda$ and `step_scale` $=\eta$.
The defaults $\lambda=\eta=1$ give an exact VE reheat followed by an ordinary
Euler probability-flow step.

This is a sigma-coordinate, EDM-style operator split, but it is not EDM
Algorithm 2:

- It reheats to the immediately preceding grid level rather than using
  $\hat\sigma=(1+\gamma)\sigma$.
- Its first step has no reheat because no preceding grid level exists.
- It uses one Euler denoiser evaluation and no Heun correction.
- It adds Feynman--Kac weights, ESS monitoring, and particle resampling, which
  are steering machinery rather than part of base EDM.

See [EDM FKC Steering](edm_fkc_steering.md) for the weight update.

## 3. AlphaFold 3 Supplementary Algorithm 18

AlphaFold 3 uses the decreasing schedule

$$
c_\tau=\sigma_{\rm data}
\left[
s_{\max}^{1/p}
+\frac{\tau}{T}\left(s_{\min}^{1/p}-s_{\max}^{1/p}\right)
\right]^p,
\qquad \tau=0,\ldots,T,
$$

with

$$
T=200,\quad
\sigma_{\rm data}=16,\quad
s_{\max}=160,\quad
s_{\min}=4\times10^{-4},\quad
p=7.
$$

The symbols $c_\tau$ and $\hat t$ in Algorithm 18 are noise standard
deviations in coordinate units (angstroms in AF3), not normalized diffusion
times. Thus $c_0=2560$ and $c_T=0.0064$.

AF3 initializes

$$
x\sim c_0\mathcal N(0,I)
$$

and, for each destination level $c_\tau$, applies a random centered rigid
augmentation followed by

$$
\gamma_\tau=
\begin{cases}
\gamma_0,&c_\tau>\gamma_{\min},\\
0,&c_\tau\leq\gamma_{\min},
\end{cases}
\qquad
\hat t=(1+\gamma_\tau)c_{\tau-1},
$$

$$
x_{\rm noisy}=x+
\lambda\sqrt{\hat t^2-c_{\tau-1}^2}\,\epsilon,
$$

$$
d=\frac{x_{\rm noisy}-D_\theta(x_{\rm noisy},\hat t)}{\hat t},
\qquad
x\leftarrow x_{\rm noisy}+\eta(c_\tau-\hat t)d.
$$

The published defaults are

$$
\gamma_0=0.8,\qquad
\gamma_{\min}=1.0,\qquad
\lambda=1.003,\qquad
\eta=1.5.
$$

Two details are easy to misread:

- $\gamma_{\min}$ is a threshold on the **destination noise level**. It is
  not a lower bound on $\gamma$ and the rule is not
  `max(c_tau, gamma_min)`.
- The released implementation computes the direction from `x_noisy`, matching
  the denoiser input.

With the published schedule, $c_\tau=1$ occurs at approximately
$\tau/T=0.801$. Churn is therefore active for roughly the first 80% of the
trajectory and disabled for the final 20%.

## 4. EDM and AF3 side by side

EDM calls the noise grid $t_i$; AF3 calls it $c_\tau$. Both are decreasing
noise standard deviations. The algorithms below retain the published line
numbers; AF3 line 1 also expands the schedule input using equation 7 of the
supplement.

$$
\begin{array}{rl|rl}
&
\boxed{\text{EDM Algorithm 2}}
&
&
\boxed{\text{AF3 Algorithm 18: Sample Diffusion}}
\\[8pt]
&
\textit{Inputs: }D_\theta,\ N,\ \sigma_{\min},\ \sigma_{\max},\ \rho
&
&
\textit{Inputs: }\operatorname{Denoiser},\ \text{conditioning}
\\[-1pt]
&
\hspace{30pt}S_{\rm churn},\ S_{\min},\ S_{\max},\ S_{\rm noise}
&
&
\hspace{30pt}[c_0,\ldots,c_T],\
\gamma_0=0.8,\ \gamma_{\min}=1.0
\\[-1pt]
&
&
&
\hspace{30pt}\lambda=1.003,\ \eta=1.5
\\[6pt]
\mathbf{1.}
&
\begin{gathered}
u_i=\sigma_{\max}^{1/\rho}
+\dfrac{i}{N-1}
\left(\sigma_{\min}^{1/\rho}-\sigma_{\max}^{1/\rho}\right),\\[-1pt]
t_i=u_i^\rho,\qquad i=0,\ldots,N-1
\end{gathered}
&
\mathbf{1.}
&
\begin{gathered}
c_\tau=\sigma_{\rm data}
\left[
s_{\max}^{1/p}
+\dfrac{\tau}{T}
\left(s_{\min}^{1/p}-s_{\max}^{1/p}\right)
\right]^p,\\[-1pt]
T=200,\quad
\sigma_{\rm data}=16,\quad
s_{\max}=160,\quad
s_{\min}=4\times10^{-4},\quad
p=7,\\[-1pt]
\vec x_l\sim c_0\mathcal N(\vec0,I_3)
\end{gathered}
\\[8pt]
\mathbf{2.}
&
t_N=0
&
\mathbf{2.}
&
\textbf{for }c_\tau\in[c_1,\ldots,c_T]\textbf{:}
\\[5pt]
\mathbf{3.}
&
x_0\sim\mathcal N(0,t_0^2I)
&
\mathbf{3.}
&
\quad\{\vec x_l\}\leftarrow
\operatorname{CentreRandomAugmentation}(\{\vec x_l\})
\\[5pt]
\mathbf{4.}
&
\textbf{for }i=0,\ldots,N-1\textbf{:}
&
\mathbf{4.}
&
\quad\gamma\leftarrow
\begin{cases}
\gamma_0,&c_\tau>\gamma_{\min},\\
0,&c_\tau\leq\gamma_{\min}
\end{cases}
\\[8pt]
\mathbf{5.}
&
\quad\hat x_i\leftarrow x_i
&
\mathbf{5.}
&
\quad\hat t\leftarrow c_{\tau-1}(1+\gamma)
\\[5pt]
\mathbf{6.}
&
\quad\gamma_i\leftarrow
\begin{cases}
\min(S_{\rm churn}/N,\sqrt2-1),
&S_{\min}\leq t_i\leq S_{\max},\\
0,&\text{otherwise}
\end{cases}
&
\mathbf{6.}
&
\quad\vec\xi_l\leftarrow
\lambda\sqrt{\hat t^2-c_{\tau-1}^2}\,
\mathcal N(\vec0,I_3)
\\[8pt]
\mathbf{7.}
&
\quad\hat t_i\leftarrow t_i(1+\gamma_i)
&
\mathbf{7.}
&
\quad\vec x_l^{\,\rm noisy}\leftarrow\vec x_l+\vec\xi_l
\\[5pt]
\mathbf{8.}
&
\begin{gathered}
\quad\epsilon_i\sim\mathcal N(0,I),\\[-1pt]
\quad\hat x_i\leftarrow
\hat x_i+S_{\rm noise}\sqrt{\hat t_i^2-t_i^2}\,\epsilon_i
\end{gathered}
&
\mathbf{8.}
&
\quad\{\vec x_l^{\,\rm denoised}\}\leftarrow
\operatorname{Denoiser}
\left(\{\vec x_l^{\,\rm noisy}\},\hat t,\text{conditioning}\right)
\\[8pt]
\mathbf{9.}
&
\quad d_i\leftarrow
\dfrac{\hat x_i-D_\theta(\hat x_i,\hat t_i)}{\hat t_i}
&
\mathbf{9.}
&
\quad\vec\delta_l\leftarrow
\dfrac{
\vec x_l^{\,\rm noisy}-\vec x_l^{\,\rm denoised}
}{\hat t}
\quad\text{(released implementation)}
\\[8pt]
\mathbf{10.}
&
\quad x_{i+1}\leftarrow
\hat x_i+(t_{i+1}-\hat t_i)d_i
&
\mathbf{10.}
&
\quad dt\leftarrow c_\tau-\hat t
\\[5pt]
\mathbf{11.}
&
\quad\textbf{if }t_{i+1}\neq0\textbf{:}
&
\mathbf{11.}
&
\quad\vec x_l\leftarrow
\vec x_l^{\,\rm noisy}+\eta\,dt\,\vec\delta_l
\\[5pt]
\mathbf{12.}
&
\qquad d'_i\leftarrow
\dfrac{x_{i+1}-D_\theta(x_{i+1},t_{i+1})}{t_{i+1}}
&
\mathbf{12.}
&
\textbf{end for}
\\[8pt]
\mathbf{13.}
&
\qquad x_{i+1}\leftarrow \hat x_i+(t_{i+1}-\hat t_i)\dfrac{d_i+d'_i}{2}
&
\mathbf{13.}
&
\textbf{return }\{\vec x_l\}
\\[8pt]
\mathbf{14.}
&
\quad\textbf{end if}
&
&
\\[5pt]
\mathbf{15.}
&
\textbf{end for}
&
&
\\[5pt]
\mathbf{16.}
&
\textbf{return }x_N
&
&
\end{array}
$$

### Line-by-line consequences

1. **Grid and endpoint:** EDM lines 1--2 construct a Karras grid and append an
   exact zero endpoint. AF3 line 1 uses the same $\rho=p=7$ grid shape, scales
   every noise level by $\sigma_{\rm data}=16$, and ends at
   $c_T=16(4\times10^{-4})=0.0064$, not zero.
2. **Rigid augmentation:** AF3 line 3 centers, rotates, and translates the
   coordinates before every step. EDM has no corresponding operation.
3. **Churn:** EDM lines 6--8 use a configurable sigma window and cap
   $\gamma_i$ at $\sqrt2-1\approx0.414$. AF3 lines 4--6 use
   $\gamma=0.8$ whenever the destination $c_\tau>1$, including the first
   step, so the denoiser is evaluated at the generally off-grid level
   $\hat t=1.8c_{\tau-1}$.
4. **Injected variance:** while AF3 churn is active, line 6 injects

   $$
   \lambda^2\left((1.8)^2-1\right)c_{\tau-1}^2
   \approx2.25c_{\tau-1}^2,
   $$

   which is more than twice the current noise variance. The multiplier
   $\lambda=1.003$ also makes the reheat slightly wider than the exact VE
   transition.
5. **Direction fix:** AF3 Algorithm 18 line 9 is printed with
   $\vec x_l-\vec x_l^{\,\rm denoised}$. The side-by-side algorithm uses the
   released DeepMind implementation instead:

   $$
   \vec\delta_l=
   \frac{\vec x_l^{\,\rm noisy}-\vec x_l^{\,\rm denoised}}{\hat t},
   $$

   matching the denoiser input and the EDM direction in line 9.
6. **Integrator:** EDM line 10 makes an Euler proposal and lines 11--13 apply
   a Heun correction, except at the zero endpoint. AF3 has no second denoiser
   evaluation: line 11 is a single Euler step multiplied by $\eta=1.5$.

Consequently, supplying the AF3 Karras grid and setting
`noise_scale=1.003, step_scale=1.5` does **not** turn
`steered_reverse_edm_sampling` into Algorithm 18. Matching AF3 also requires
its gated fractional reheat, first-step off-grid denoiser call, and per-step
rigid augmentation. Conversely, adding EDM's Heun correction to AF3 would
define a different sampler with approximately twice the denoiser cost.

## 5. Repository AF3 sampler

`steered_reverse_af3_sampling` implements the AF3 sigma-space loop with SMC
resampling and FKC weights:

```python
trajectory, ess_history, weight_history = steered_reverse_af3_sampling(
    denoise=denoise,
    weight_update=None,
    x=x,
    sigma=sigma,
    gamma_0=0.8,
    gamma_min=1.0,
    noise_scale=1.0,
    step_scale=1.0,
    ess_threshold=0.5,
    potential=log_potential,
    reheat_proposal=reheat_proposal,
)
```

The implementation deliberately differs from the publication defaults in two
empirical multipliers: both the extra reheat-noise multiplier and Euler-step
multiplier default to one. Thus `noise_scale=1` is the exact VE forward
transition and `step_scale=1` is the unscaled probability-flow Euler step.

For AF3's fixed finite reheat, the recommended weighting callback is the full
grid-marginal log potential:

```python
log_potential(x, sigma)
```

For a target

$$
p_i(x)\propto q_i(x)\exp(\rho_i(x)),
$$

the sampler applies the exact discrete Feynman--Kac increment after each full
AF3 step,

$$
\Delta w_i=\rho_{i+1}(x_{i+1})-\rho_i(x_i).
$$

This mode assumes `denoise` defines the unsteered base AF3 transition. It is
valid for the finite $\gamma_0=0.8$ reheat because it weights the complete
transition rather than approximating the jump with a local generator.

The solver also retains the repository EDM-style callback:

```python
weight_update(x_noisy, sigma_hat, sigma_curr, sigma_next)
```

This returns a local continuous-FKC increment and permits caller-provided
proposal guidance in `denoise`. It is accurate when churn vanishes with the
grid spacing, including `gamma_0=0`; applying it unchanged to AF3's fixed
$0.8$ reheat produces a systematic under-tilt.

Exactly one of `potential` and `weight_update` must be supplied. The denoiser
uses the released-code line-9 correction,

$$
d=\frac{x_{\rm noisy}-D_\theta(x_{\rm noisy},\hat\sigma)}{\hat\sigma}.
$$

### Guided reheating

`reheat_proposal` can replace the base Gaussian reheat with a guided proposal:

```python
x_noisy, log_base_over_proposal = reheat_proposal(
    x,
    sigma_hat,
    sigma_curr,
    sigma_next,
)
```

The second return value is the per-particle correction

$$
\log K(x_{\rm noisy}\mid x)-\log Q(x_{\rm noisy}\mid x),
$$

where $K$ is the exact AF3 reheat and $Q$ is the guided proposal. The sampler
adds this correction in both weighting modes.

For

$$
K=\mathcal N(x,vI),\qquad
Q=\mathcal N(x+vg(x),vI),\qquad
v=\hat\sigma^2-\sigma^2,
$$

sample

$$
x_{\rm noisy}=x+vg(x)+\sqrt v\,\epsilon
$$

and return

$$
\log\frac KQ
=-(x_{\rm noisy}-x)^\mathsf T g(x)
+\frac v2\lVert g(x)\rVert^2.
$$

This permits reward-directed reheating even for $\gamma_0=0.8$ while exact
discrete potential weights correct the proposal bias. `noise_scale` is ignored
when a custom reheat proposal is supplied. The subsequent denoiser should
remain the base denoiser; guiding that deterministic map would require its own
proposal/Jacobian correction.

The sampler omits `CentreRandomAugmentation`: arbitrary rotations and
translations do not preserve a generic GMM target. It also accepts the initial
particles and decreasing sigma grid from the caller rather than constructing
the AF3 schedule internally.

For TorchGMM, the denoiser callback must support arbitrary $\hat\sigma$.
In particular, the first AF3 call is at
$\hat\sigma=(1+\gamma_0)\sigma_0=1.8\sigma_0$, outside the supplied Karras
grid, so `schedule.time(sigma_hat)` cannot be used for that callback. The
analytic arbitrary-sigma GMM example is in `notebooks/af3_steering.py`, and
`tests/test_af3_steering.py` verifies both the base marginal family and the
reward-tilted FKC family at intermediate sigma levels.

## Sources

- Karras et al., [*Elucidating the Design Space of Diffusion-Based Generative Models*](https://arxiv.org/abs/2206.00364), Algorithm 2.
- Abramson et al., [*Accurate structure prediction of biomolecular interactions with AlphaFold 3*](https://doi.org/10.1038/s41586-024-07487-w), [Supplementary Information](https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-07487-w/MediaObjects/41586_2024_7487_MOESM1_ESM.pdf) §3.7, Algorithm 18 and equation 7.
- Google DeepMind, [`diffusion_head.py`](https://github.com/google-deepmind/alphafold3/blob/97d20234c6eb89e8d05376e9eecc9321e60a559b/src/alphafold3/model/network/diffusion_head.py#L88-L371), released AF3 sampler implementation.
