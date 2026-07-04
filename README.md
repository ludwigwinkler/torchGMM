# torchGMM

Analytical diffusion on Gaussian Mixture Models in PyTorch when you have a diffusion idea and want a diffusion sandbox to test out stuff.

<p align="center">
  <img src="notebooks/forward_backward_diffusion.gif" alt="Forward and reverse diffusion on a 1-D GMM" width="800">
</p>

torchGMM provides **time-dependent GMMs** with closed-form log-probabilities, scores, and sampling under a forward SDE diffusion process — no neural network required. Because the GMM family is closed under Gaussian convolution, every quantity (density, score, energy) stays exact at every noise level $t \in [0, 1]$.

## Mathematics

The forward SDE follows the Variance-Preserving (VP) formulation:

$$dX_t = -\tfrac{1}{2}\,\beta(t)\,X_t\,dt + \sqrt{\beta(t)}\,dW_t$$

with linear schedule $\beta(t) = \beta_\text{min} + t  (\beta_\text{max} - \beta_\text{min})$.

The marginal at time $t$ of a GMM $p_0(x) = \sum_k \pi_k\,\mathcal{N}(x;\mu_k, \Sigma_k)$ is again a GMM:

$$p_t(x) = \sum_k \pi_k\,\mathcal{N}\!\bigl(x;\,\alpha_t\,\mu_k,\;\sigma_t^2 I + \alpha_t^2\,\Sigma_k\bigr)$$

where $\alpha_t = \exp\!\bigl(-\tfrac{1}{2}\int_0^t \beta(s)\,ds\bigr)$ and $\sigma_t = \sqrt{1 - \alpha_t^2}$.


## Features

- **Fully batched** — parameters are `[*B, K, D]` (arbitrary batch × components × dimensions). All ops broadcast over batch and sample dims.
- **Exact score** $\nabla_x \log p_t(x)$ via autograd on the analytical log-density.
- **Forward & reverse SDE** simulation (Euler–Maruyama) with the linear $\beta$-schedule from VP-SDE.
- **Conditional process** — collapse the mixture to a single Dirac at $x_0$ for conditional sampling / inference.
- **Marginalisation & mode dropping** — extract 1-D marginals or remove components on the fly.
- **Pure PyTorch** — differentiable end-to-end, GPU-friendly, no custom C++/CUDA.
- **Steering** — compute exact importance weights and ESS for steering the reverse process towards a target distribution.

## Steering

torchGMM uses [FeynmanKac-Correctors](https://arxiv.org/pdf/2503.02819) to steer the reverse SDE towards an arbitrary target distribution $p(x) \propto q(x) \exp(\beta r(x))$, using the theory developed in the FKC paper.
This allows you to sample from very particular regions of the sampling space with the correct importance weights.

<p align="center">
  <img src="notebooks/steered_diffusion.gif" alt="Forward and reverse diffusion on a 1-D GMM" width="800">
</p>

## Installation

```bash
# editable install
pip install -e .

# or with uv
uv pip install -e .
```

Requires Python ≥ 3.10 and PyTorch ≥ 2.7.

## Quick Start

```python
import torch
from torchGMM import GMM, BetaSchedule

# 2-component mixture in 2D
mu     = torch.tensor([[-2.0, 0.0],
                        [ 2.0, 0.0]]).unsqueeze(0)   # [1, K=2, D=2]
sigma  = torch.ones(1, 2, 2) * 0.5                    # [1, K=2, D=2]
weight = torch.tensor([[0.3, 0.7]])                    # [1, K=2]

schedule = BetaSchedule(beta_min=0.1, beta_max=20.0)
gmm = GMM(mu, sigma, weight, schedule=schedule)

# Exact log-probability at noise level t = 0.4
x = torch.randn(1000, 1, 2)          # [N, *B, D]
lp = gmm.log_prob(x, t=0.4)          # [N, *B]

# Exact score (gradient of log-density)
s = gmm.score(x, t=0.4)              # [N, *B, D]

# Ancestral sampling at t = 0 (clean data)
samples = gmm.sample(5000)           # [N, *B, D]
```

### Running the Forward & Reverse SDE

`forward_sampling` / `reverse_sampling` accept `drift` and `diffusion` callables, so
the schedule and the GMM score combine explicitly into the reverse SDE drift:

$$dX_t = \bigl[f(t)\,X_t - g(t)^2\,\nabla_x \log p_t(x)\bigr]\,dt + g(t)\,dW_t.$$

```python
from torchGMM import forward_sampling, reverse_sampling

eps = 1e-3
t_fwd = torch.linspace(eps, 1.0 - eps, 500)
t_rev = torch.linspace(1.0 - eps, eps, 500)

# Forward: data → noise (drift and diffusion come straight from the schedule)
x0 = gmm.sample(512)                                                  # [512, 1, 2]
traj_fwd = forward_sampling(
    schedule.forward_drift, schedule.diffusion_coeff, x0, t_fwd,
)                                                                     # [T, 512, 1, 2]

# Reverse: noise → data using the exact GMM score
reverse_drift = lambda x, t: (
    schedule.forward_drift(x, t) - schedule.diffusion_coeff(t) ** 2 * gmm.score(x, t)
)
x_noise = torch.randn_like(x0)
traj_rev = reverse_sampling(
    reverse_drift, schedule.diffusion_coeff, x_noise, t_rev,
).detach()                                                            # [T, 512, 1, 2]
```

Pass `diffusion=None` to either sampler for the deterministic probability-flow ODE.

### Conditional Process

```python
from torchGMM import Conditional

# Conditional on a single starting point x0
x0 = torch.tensor([[1.0, -1.0]])          # [B=1, D=2]
cond = Conditional(x0, schedule=schedule) # single-component GMM at x0

# Score of the conditional forward process
s = cond.score(x, t=0.6)
```

### Shape Convention

| Symbol | Meaning |
|---|---|
| `*B` | Batch dimensions (from GMM init, e.g. number of parallel GMMs) |
| `K` | Number of mixture components |
| `D` | Data dimensionality |
| `*N` | Sample dimensions (optional leading dims on inputs) |

Inputs are `[*N, *B, D]`. Scalar outputs (`log_prob`, `energy`) are `[*N, *B]`. Vector outputs (`score`, `sample`) are `[*N, *B, D]`.
