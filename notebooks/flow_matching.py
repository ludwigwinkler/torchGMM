# %% [markdown]
# # Flow Matching: Marginal Velocity Field (1D)
#
# Visualise the exact marginal velocity field v_t(x) for a 1D GMM
# under the flow matching (conditional OT) schedule: α_t = 1−t, σ_t = t.
#
# The velocity is computed from the exact score via:
#   v_t(x) = −x/(1−t) − t/(1−t) · s_t(x)

# %%
import torch
import matplotlib.pyplot as plt
import numpy as np
from _utils import plt_show

from torchGMM import TimeDependentGMM, FlowMatchingSchedule, BetaSchedule
from torchGMM.diffusion import reverse_sampling

torch.manual_seed(42)

# %%[markdown]
# ## Define a 1D GMM with flow matching schedule

# %%
mu = torch.tensor([[-2.0], [1.5], [4.0]]).unsqueeze(0)  # [B=1, K=3, D=1]
sigma = torch.tensor([[0.4], [0.3], [0.5]]).unsqueeze(0)  # [B=1, K=3, D=1]
weight = torch.tensor([0.4, 0.35, 0.25]).unsqueeze(0)  # [B=1, K=3]

schedule = FlowMatchingSchedule()
gmm = TimeDependentGMM(mu=mu, sigma=sigma, weight=weight, schedule=schedule)

# %%[markdown]
# ## Score-to-velocity conversion for the linear schedule
#
# For α_t = 1−t, σ_t = t:
#   v_t(x) = −x/(1−t) − t/(1−t) · s_t(x)


# %%
def velocity_from_score(gmm: TimeDependentGMM, x: torch.Tensor, t: float) -> torch.Tensor:
    """Compute marginal velocity v_t(x) from the exact score for the flow matching schedule.
    Note: gmm.velocity(x, t) computes this for any schedule. This is kept for illustration."""
    s = gmm.score(x, t)  # [N, B, D]
    return -x / (1 - t) - t / (1 - t) * s


# %%[markdown]
# ## Plot density and velocity at multiple time steps

# %%
grid_lim = 6.0
n_pts = 300
x_grid = torch.linspace(-grid_lim, grid_lim, n_pts).reshape(-1, 1, 1)  # [N, B=1, D=1]

times = [0.01, 0.1, 0.25, 0.5, 0.75, 0.95]

fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

cmap = plt.colormaps.get_cmap("viridis").resampled(len(times))

for idx, t in enumerate(times):
    color = cmap(idx)

    # Density p_t(x)
    with torch.no_grad():
        log_p = gmm.log_prob(x_grid, t=t).squeeze()  # [N]
        density = log_p.exp().numpy()
    axes[0].plot(x_grid[:, 0, 0].numpy(), density, color=color, label=f"t={t:.2f}", linewidth=1.5)

    # Velocity v_t(x)
    v = velocity_from_score(gmm, x_grid, t=t).squeeze().detach().numpy()  # [N]
    axes[1].plot(x_grid[:, 0, 0].numpy(), v, color=color, label=f"t={t:.2f}", linewidth=1.5)

axes[0].set_ylabel("p_t(x)")
axes[0].set_title("Marginal density under flow matching schedule")
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

axes[1].set_ylabel("v_t(x)")
axes[1].set_xlabel("x")
axes[1].set_title("Marginal velocity field v_t(x) = −x/(1−t) − t/(1−t) · s_t(x)")
axes[1].legend(fontsize=9)
axes[1].axhline(0, color="k", linewidth=0.5, linestyle="--")
axes[1].grid(True, alpha=0.3)

fig.suptitle("Flow Matching on a 1D GMM", fontsize=14, y=1.01)
plt.tight_layout()
plt_show()

# %%[markdown]
# ## Compare schedule coefficients: VP-SDE vs Flow Matching

# %%
t_range = torch.linspace(0.001, 0.999, 200)
vp = BetaSchedule()
fm = FlowMatchingSchedule()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(t_range, vp.get_alpha_t(t_range).numpy(), label="VP-SDE", linewidth=2)
axes[0].plot(t_range, fm.get_alpha_t(t_range).numpy(), label="Flow Matching", linewidth=2, linestyle="--")
axes[0].set_title("Signal coefficient α_t")
axes[0].set_xlabel("t")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(t_range, vp.get_sigma_t(t_range).numpy(), label="VP-SDE", linewidth=2)
axes[1].plot(t_range, fm.get_sigma_t(t_range).numpy(), label="Flow Matching", linewidth=2, linestyle="--")
axes[1].set_title("Noise coefficient σ_t")
axes[1].set_xlabel("t")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

snr_vp = torch.log(vp.get_alpha_t(t_range) ** 2 / vp.get_sigma_t(t_range) ** 2)
snr_fm = torch.log(fm.get_alpha_t(t_range) ** 2 / fm.get_sigma_t(t_range) ** 2)
axes[2].plot(t_range, snr_vp.numpy(), label="VP-SDE", linewidth=2)
axes[2].plot(t_range, snr_fm.numpy(), label="Flow Matching", linewidth=2, linestyle="--")
axes[2].set_title("Log-SNR λ_t = log(α²/σ²)")
axes[2].set_xlabel("t")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt_show()

# %%[markdown]
# ## ODE sampling: integrate dx/dt = v_t(x) from t=1 → t=0.05
#
# We stop at t=0.05 to avoid the 1/(1−t) singularity near t=0.

# %%
n_samples = 5_000
n_steps = 200
t_min = 0.05

# Sample initial noise at t=1
x0_noise = torch.randn(n_samples, 1, 1)  # [N, B=1, D=1]

# Time grid from t=1 down to t_min
t_ode = torch.linspace(1.0 - 1e-4, t_min, n_steps)

# ODE integration: dx/dt = v_t(x), no noise
traj_full = reverse_sampling(gmm.velocity, None, x0_noise, t_ode)  # [T, N, B=1, D=1]
trajectory = traj_full[:, :, 0, 0]  # [T, N]

# %%[markdown]
# ## Plot trajectories over time with target density

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [3, 1]})

# Left: trajectories in (t, x) space
ax = axes[0]
for j in range(n_samples):
    ax.plot(t_ode.numpy(), trajectory[:, j].numpy(), color="steelblue", alpha=0.1, linewidth=0.8)

ax.set_xlabel("t", fontsize=12)
ax.set_ylabel("x", fontsize=12)
ax.set_title("ODE trajectories: t = 1 (noise) → t = 0.05 (data)", fontsize=13)
ax.set_xlim(1.0, 0.0)
ax.axvline(t_min, color="red", linestyle="--", alpha=0.5, label=f"t_min = {t_min}")
ax.legend()
ax.set_ylim(-6, 6)
ax.grid(True, alpha=0.3)

# Right: histogram of final samples vs target density
ax2 = axes[1]
final_samples = trajectory[-1].numpy()
ax2.hist(
    final_samples,
    bins=100,
    density=True,
    orientation="horizontal",
    color="steelblue",
    alpha=0.6,
    edgecolor="white",
    label="ODE samples",
)

# Overlay target density at t=t_min
x_plot = torch.linspace(-grid_lim, grid_lim, 300).reshape(-1, 1, 1)
with torch.no_grad():
    target_density = gmm.log_prob(x_plot, t=t_min).squeeze().exp().numpy()
ax2.plot(target_density, x_plot[:, 0, 0].numpy(), color="red", linewidth=2, alpha=0.5, label=f"p_{{t={t_min}}}(x)")

ax2.set_xlabel("density", fontsize=12)
ax2.set_ylim(ax.get_ylim())
ax2.set_title("Final distribution", fontsize=13)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt_show()

# %%[markdown]
# ## Stochastic sampler: reverse SDE trajectories
#
# We add a time-dependent noise injection γ(t) = γ_min + t·(γ_max − γ_min)
# to the flow matching ODE, turning it into a stochastic sampler:
#   dX = [v_t(X) − ½γ(t)²s_t(X)] dt + γ(t) dW̃
#
# γ(t) = 0 recovers the deterministic ODE. Larger γ gives more stochastic paths.

# %%
gamma_min = 0.0
gamma_max = 1.0


def gamma_t(t: float, gamma_min: float = gamma_min, gamma_max: float = gamma_max) -> float:
    """Linear noise schedule: γ(t) = γ_min + t·(γ_max − γ_min)"""
    return 1 - (gamma_min + t * (gamma_max - gamma_min))


# Stochastic flow matching: Euler-Maruyama from t≈1 to t_min
t_sde = torch.linspace(1.0 - 1e-4, t_min, n_steps)

x_sde = torch.randn(n_samples, 1, 1)  # initial noise


# dx = [v_t(x) - ½γ(t)²·s_t(x)] dt + γ(t) dW
def sde_drift(x_, t_):
    v = gmm.velocity(x_, t_)
    score = gmm.score(x_, t_)
    gamma = gamma_t(t_.item())
    return v - 0.5 * gamma**2 * score


def sde_diffusion(t_):
    return torch.tensor(gamma_t(t_.item()))


traj_sde_full = reverse_sampling(sde_drift, sde_diffusion, x_sde, t_sde)  # [T, N, B=1, D=1]
traj_sde = traj_sde_full[:, :, 0, 0]  # [T, N]

# %%[markdown]
# ## Plot stochastic trajectories with target density

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [3, 1]})

ax = axes[0]
for j in range(n_samples):
    ax.plot(t_sde.numpy(), traj_sde[:, j].numpy(), color="darkorange", alpha=0.1, linewidth=0.8)

ax.set_xlabel("t", fontsize=12)
ax.set_ylabel("x", fontsize=12)
ax.set_title(f"Stochastic flow matching: γ(t) = {gamma_min} + t·{gamma_max - gamma_min:.0f}", fontsize=13)
ax.set_xlim(1.0, 0.0)
ax.axvline(t_min, color="red", linestyle="--", alpha=0.5, label=f"t_min = {t_min}")
ax.legend()
ax.set_ylim(-6, 6)
ax.grid(True, alpha=0.3)

ax2 = axes[1]
final_sde = traj_sde[-1].numpy()
ax2.hist(
    final_sde,
    bins=100,
    density=True,
    orientation="horizontal",
    color="darkorange",
    alpha=0.6,
    edgecolor="white",
    label="SDE samples",
)

# Target density under flow matching schedule at t_min
with torch.no_grad():
    target_vp = gmm.log_prob(x_plot, t=t_min).squeeze().exp().numpy()
ax2.plot(target_vp, x_plot[:, 0, 0].numpy(), color="red", linewidth=2, alpha=0.5, label=f"p_{{t={t_min}}}(x)")

ax2.set_xlabel("density", fontsize=12)
ax2.set_ylim(-6, 6)
ax2.set_title("Final distribution", fontsize=13)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
# plt.savefig("notebooks/flow_matching_sde_trajectories.png", dpi=150, bbox_inches="tight")
plt_show()

# %%[markdown]
# ## Combined: ODE vs SDE trajectories

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [3, 1]})

ax = axes[0]
for j in range(n_samples):
    ax.plot(t_sde.numpy(), traj_sde[:, j].numpy(), color="darkorange", alpha=0.08, linewidth=0.6)
for j in range(n_samples):
    ax.plot(t_ode.numpy(), trajectory[:, j].numpy(), color="steelblue", alpha=0.08, linewidth=0.6)

# Legend proxies
ax.plot([], [], color="steelblue", linewidth=2, label="Flow matching (ODE)")
ax.plot([], [], color="darkorange", linewidth=2, label="Reverse SDE (VP)")

ax.set_xlabel("t", fontsize=12)
ax.set_ylabel("x", fontsize=12)
ax.set_title("ODE (deterministic) vs SDE (stochastic) trajectories", fontsize=13)
ax.set_xlim(1.0, 0.0)
ax.axvline(t_min, color="red", linestyle="--", alpha=0.5, label=f"t_min = {t_min}")
ax.legend(fontsize=10)
ax.set_ylim(-6, 6)
ax.grid(True, alpha=0.3)

# Right: overlaid histograms
ax2 = axes[1]
ax2.hist(
    trajectory[-1].numpy(),
    bins=100,
    density=True,
    orientation="horizontal",
    color="steelblue",
    alpha=0.5,
    edgecolor="white",
    label="ODE",
)
ax2.hist(
    traj_sde[-1].numpy(),
    bins=100,
    density=True,
    orientation="horizontal",
    color="darkorange",
    alpha=0.5,
    edgecolor="white",
    label="SDE",
)

with torch.no_grad():
    target_fm = gmm.log_prob(x_plot, t=t_min).squeeze().exp().numpy()
ax2.plot(target_fm, x_plot[:, 0, 0].numpy(), color="red", linewidth=2, alpha=0.5, label="target")

ax2.set_xlabel("density", fontsize=12)
ax2.set_ylim(-6, 6)
ax2.set_title("Final distributions", fontsize=13)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt_show()
