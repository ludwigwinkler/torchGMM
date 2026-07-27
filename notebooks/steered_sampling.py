"""Steered reverse sampling on a 1-D 3-mode GMM — matches bioemu em_fkc_simple setup.

Same hyperparameters as the bioemu validation script
(`bioemu2/worktrees/steering/notebooks/toy_gmm_validation.py`):

- GMM: 3 modes at μ = [-3, 0, 3], σ = 0.5, uniform weights
- Potential: E(x) = (x − 1.5)²  →  log-reward r(x) = −λ · E(x), λ = 1
- Target tilted distribution: p(x) · exp(−E(x))
- Sampler: VESchedule (σ_min=0.0004, σ_max=10.0), N=32768 particles, T=200 steps
- FK: ESS-adaptive resampling at threshold 0.9, systematic resample
- Pure-bootstrap (no drift twisting; all steering is via weight reweighting)

Output: `torchgmm_compare.png` — single histogram panel with unsteered prior,
analytic biased target, actual steered samples, and target line.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from torchGMM import GMM, VESchedule
from torchGMM.sampling import steered_reverse_sampling

plt.style.use("default")

torch.manual_seed(0)
device = torch.device("cpu")
torch.set_default_device(device)

SIGMA_MAX_CLI = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
OUT = Path(__file__).resolve().parent / f"torchgmm_compare_sigma{int(SIGMA_MAX_CLI)}.png"

# ---------------------------------------------------------------------------
# Hyperparameters (kept identical to bioemu's toy_gmm_validation.py)
# ---------------------------------------------------------------------------
TARGET = 1.5
LAMBDA = 1.0
N, T, ESS = 32_768, 200, 0.9
T_MAX, EPS_R = 0.999, 1e-3
N_BIASED = 100_000

ve_sched = VESchedule(sigma_min=0.0004, sigma_max=SIGMA_MAX_CLI)
gmm = GMM(
    mu=torch.tensor([[[-3.0], [0.0], [3.0]]]),
    sigma=torch.tensor([[[0.5], [0.5], [0.5]]]),
    weight=torch.tensor([[1.0 / 3, 1.0 / 3, 1.0 / 3]]),
    schedule=ve_sched,
)


def r(x):
    """log-reward = −λ · E(x) with E(x) = (x − TARGET)²."""
    return -LAMBDA * (x - TARGET) ** 2


def x0_hat(x, t):
    """Tweedie one-step denoiser for VE (α_t ≡ 1):  E[x_0 | x_t] = x_t + σ_t² · score."""
    sigma_t = ve_sched.get_sigma_t(t)
    return x + sigma_t**2 * gmm.score(x, t)


def reverse_drift(x_, t_):
    g = ve_sched.diffusion_coeff(t_)
    return -(g**2) * gmm.score(x_, t_)


def weight_update(x_, t_, dt):
    """Stateless FKC potential: Δlog_w = r(x̂₀(x_t, t)) · |dt|."""
    return r(x0_hat(x_, t_)).squeeze(-1).squeeze(-1) * dt.abs()


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
print(f"GMM: 3 modes at x = [-3, 0, 3], σ = 0.5")
print(f"target = {TARGET}, λ = {LAMBDA}")
print(f"N = {N} particles, T = {T} steps, ESS threshold = {ESS}")

t_rev = torch.linspace(T_MAX, EPS_R, T)
x_init = gmm.sample(shape=N, t=T_MAX)

traj, ess_hist, _ = steered_reverse_sampling(
    drift=reverse_drift,
    diffusion=ve_sched.diffusion_coeff,
    weight_update=weight_update,
    x=x_init,
    t=t_rev,
    ess_threshold=ESS,
)
steered = traj[-1, :, 0, 0].detach()
unsteered = gmm.sample(shape=N, t=torch.tensor(EPS_R)).reshape(-1).detach()

# Analytic biased target via importance sampling.
proposed = gmm.sample(shape=N_BIASED * 10, t=torch.tensor(EPS_R)).reshape(-1)
log_w = -LAMBDA * (proposed - TARGET) ** 2
idx = torch.multinomial((log_w - log_w.max()).exp(), N_BIASED, replacement=True)
biased = proposed[idx]


def wasserstein2_1d(x, y):
    if x.numel() == y.numel():
        return (x.sort().values - y.sort().values).pow(2).mean().sqrt()
    n = min(x.numel(), y.numel())
    qs = torch.linspace(0.5 / n, 1.0 - 0.5 / n, n, dtype=torch.float32, device=x.device)
    return (torch.quantile(x.float(), qs) - torch.quantile(y.float(), qs)).pow(2).mean().sqrt()


w2 = wasserstein2_1d(steered, biased).item()
print(f"\nW₂(steered, biased) = {w2:.4f}")
print(f"n_resamples = {sum(1 for e in ess_hist if e < ESS)}")

# ---------------------------------------------------------------------------
# Plot — same layout as bioemu's toy_gmm_em_fkc.png
# ---------------------------------------------------------------------------
bins = np.linspace(-5.0, 5.0, 80)
fig, ax = plt.subplots(figsize=(7, 4.4))
ax.hist(unsteered.numpy(), bins=bins, color="0.75", alpha=0.7,
        density=True, label="unsteered (p_GMM)")
ax.hist(biased.numpy(), bins=bins, histtype="step", color="k",
        linewidth=2, density=True, label="analytic biased  p₀·exp(−λE)")
ax.hist(steered.numpy(), bins=bins, color="C1", alpha=0.55,
        density=True, label="actual steered")
ax.axvline(TARGET, color="C3", linestyle="--", linewidth=1,
           label=f"target = {TARGET}")
ax.set_xlabel("x")
ax.set_ylabel("density")
ax.set_ylim(0, 1)
ax.set_title(f"torchGMM steered_reverse_sampling · σ_max={SIGMA_MAX_CLI} · W₂ = {w2:.3f}")
ax.legend(loc="upper right", fontsize=9)
fig.tight_layout()
fig.savefig(OUT, dpi=120)
print(f"saved {OUT}")
