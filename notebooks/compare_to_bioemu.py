"""Run torchGMM steered_reverse_sampling on the same 1-D problem as the
bioemu em_fkc_simple validation, and report W₂(steered, analytic_biased).

Problem: 3-mode 1-D GMM at μ = [-3, 0, 3], σ = [0.5, 0.5, 0.5].
Potential: E(x) = (x − 1.5)² (matches bioemu UmbrellaPotential
slope=1 order=2 weight=1).
Target tilted: p(x) · exp(−E(x))  (fk_lambda = 1).
"""

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

# --- model: same GMM as bioemu's notebook ---
ve_sched = VESchedule(sigma_min=0.0004, sigma_max=10.0)
gmm = GMM(
    mu=torch.tensor([[[-3.0], [0.0], [3.0]]]),       # [B=1, K=3, D=1]
    sigma=torch.tensor([[[0.5], [0.5], [0.5]]]),
    weight=torch.tensor([[1.0 / 3, 1.0 / 3, 1.0 / 3]]),
    schedule=ve_sched,
)

TARGET = 1.5
LAMBDA = 1.0


def r(x):
    """log-reward = −λ · E(x) with E(x) = (x − TARGET)²."""
    return -LAMBDA * (x - TARGET) ** 2


def x0_hat(x, t):
    """Tweedie one-step denoiser for VE (α_t ≡ 1)."""
    sigma_t = ve_sched.get_sigma_t(t)
    return x + sigma_t**2 * gmm.score(x, t)


N, T, ESS = 32_768, 200, 0.9   # N matches bioemu's B*K = 4096 * 8
T_MAX, EPS_R = 0.999, 1e-3
t_rev = torch.linspace(T_MAX, EPS_R, T)
x_init = gmm.sample(shape=N, t=T_MAX)  # [N, 1, 1]


def reverse_drift(x_, t_):
    g = ve_sched.diffusion_coeff(t_)
    return -(g**2) * gmm.score(x_, t_)


def weight_update(x_, t_, dt):
    return r(x0_hat(x_, t_)).squeeze(-1).squeeze(-1) * dt.abs()


print(f"GMM: 3 modes at x = [-3, 0, 3], σ = 0.5")
print(f"target = {TARGET}, λ = {LAMBDA}")
print(f"N = {N} particles, T = {T} steps, ESS threshold = {ESS}")

traj, ess_hist, _ = steered_reverse_sampling(
    drift=reverse_drift,
    diffusion=ve_sched.diffusion_coeff,
    weight_update=weight_update,
    x=x_init,
    t=t_rev,
    ess_threshold=ESS,
)
steered = traj[-1, :, 0, 0].detach()  # final-time scalar samples

# --- analytic biased target via importance sampling ---
N_BIASED = 100_000
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

# --- unsteered baseline for the plot ---
unsteered = gmm.sample(shape=N, t=torch.tensor(EPS_R)).reshape(-1).detach()

# --- plot, same layout as bioemu notebook ---
OUT = Path(__file__).resolve().parent / "torchgmm_compare.png"
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
ax.set_title(f"torchGMM reference on 1-D GMM    W₂(steered, biased) = {w2:.3f}")
ax.legend(loc="upper right", fontsize=9)
fig.tight_layout()
fig.savefig(OUT, dpi=120)
print(f"saved {OUT}")
