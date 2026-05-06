# %%
"""Forward process visualization for the Karras VE schedule.

Top row: B=4 single-component GMMs overlaid in a single trajectory panel with
flanking data / noise marginals.
Bottom row: same trajectories split into five time segments [0, 0.2, …, 1.0],
each with its own y-range adapted to the min/max in that segment.
"""

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import torch
from _utils import plt_show

from torchGMM import GMM, KarrasSchedule, VESchedule, forward_sampling
from torchGMM.sampling import reverse_sampling, steered_reverse_sampling

plt.style.use("default")
plt.rcdefaults()

torch.manual_seed(0)
device = torch.device("cpu")
torch.set_default_device(device)

# %% [markdown]
# # Variance Exploding Forward Process

# --- model: B=4 separate GMMs, each with K=1 component ---
schedule = KarrasSchedule()
mu = torch.tensor([[[-3.0]], [[-0.5]], [[2.0]], [[4.0]]])  # [B=4, K=1, D=1]
sigma = torch.tensor([[[0.3]], [[0.25]], [[0.4]], [[0.2]]])  # [B=4, K=1, D=1]
weight = torch.ones(4, 1)
gmm = GMM(mu=mu, sigma=sigma, weight=weight, schedule=schedule)
B = mu.shape[0]

# --- forward simulation ---
EPS, N, T = 1e-3, 5000, 400
t = torch.linspace(EPS, 1 - EPS, T)
x0 = gmm.sample(shape=N, t=EPS)  # [N, B, D=1]
traj = forward_sampling(schedule.forward_drift, schedule.diffusion_coeff, x0, t).detach()  # [T, N, B, D=1]

# --- limits and marginals (top row) ---
data_lim = float(mu.abs().max()) + 4 * float(sigma.max())
traj_lim = float(traj[:, :, :, 0].abs().max()) * 1.05
noise_lim = float(schedule.get_sigma_t(torch.tensor(1 - EPS))) * 4

x_grid_data = torch.linspace(-data_lim, data_lim, 300).reshape(-1, 1, 1).expand(-1, B, -1)
x_grid_noise = torch.linspace(-noise_lim, noise_lim, 300).reshape(-1, 1, 1).expand(-1, B, -1)
p_data = gmm.log_prob(x_grid_data, t=EPS).exp().detach()
p_noise = gmm.log_prob(x_grid_noise, t=1 - EPS).exp().detach()

# --- segments (bottom row) ---
seg_edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
n_seg = len(seg_edges) - 1
colors = ["steelblue", "seagreen", "darkorange", "purple"]

# --- figure: two rows ---
fig = plt.figure(figsize=(18, 9))
outer = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.35)

# Top: data marginal | overlaid trajectories | noise marginal
top_gs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0], width_ratios=[1, 8, 1], wspace=0.15)
ax_l = fig.add_subplot(top_gs[0])
ax_m = fig.add_subplot(top_gs[1])
ax_r = fig.add_subplot(top_gs[2])

t_np = t.numpy()
for b in range(B):
    c = colors[b % len(colors)]
    label = rf"$\mu={float(mu[b, 0, 0]):.1f}$"

    ax_l.plot(p_data[:, b], x_grid_data[:, b, 0], color=c, lw=1.5)
    ax_l.fill_betweenx(x_grid_data[:, b, 0], 0, p_data[:, b], color=c, alpha=0.2)

    ax_m.plot(t_np, traj[:, :, b, 0], color=c, alpha=0.04, lw=0.5)
    ax_m.plot([], [], color=c, lw=2, label=label)

    ax_r.plot(p_noise[:, b], x_grid_noise[:, b, 0], color=c, lw=1.5)
    ax_r.fill_betweenx(x_grid_noise[:, b, 0], 0, p_noise[:, b], color=c, alpha=0.2)

ax_l.invert_xaxis()
ax_l.set_title(r"$p(x, t \approx 0)$")
ax_l.set_ylabel("x")
ax_l.set_xticks([])
ax_l.set_ylim(-data_lim, data_lim)

ax_m.set_title(r"$\rightarrow$ Karras VE Forward Diffusion $\rightarrow$")
ax_m.set_xlabel("t")
ax_m.set_xlim(0, 1)
ax_m.set_ylim(-traj_lim, traj_lim)
ax_m.legend(loc="upper left", fontsize=9)

ax_r.set_title(r"$p(x, t \approx 1)$")
ax_r.set_xticks([])
ax_r.set_ylim(-noise_lim, noise_lim)

# Bottom: 5 segmented trajectory windows with per-segment y-range
bot_gs = gridspec.GridSpecFromSubplotSpec(1, n_seg, subplot_spec=outer[1], wspace=0.3)
for s in range(n_seg):
    ax = fig.add_subplot(bot_gs[s])
    lo, hi = seg_edges[s], seg_edges[s + 1]
    mask = (t >= lo) & (t <= hi)
    ts = t_np[mask.numpy()]
    seg = traj[mask, :, :, 0]

    for b in range(B):
        ax.plot(ts, seg[:, :, b], color=colors[b], alpha=0.01, lw=0.5)
        if s == 0:
            ax.plot([], [], color=colors[b], lw=2, label=rf"$\mu={float(mu[b, 0, 0]):.1f}$")

    y_min, y_max = float(seg.min()), float(seg.max())
    pad = 0.05 * (y_max - y_min) if y_max > y_min else 1.0
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_xlim(lo, hi)
    ax.set_title(rf"$t \in [{lo:.1f}, {hi:.1f}]$")
    ax.set_xlabel("t")
    ax.grid(True, alpha=0.2)
    if s == 0:
        ax.set_ylabel("x")
        ax.legend(loc="upper left", fontsize=8)

plt_show()

# %% [markdown]
# # Variance Exploding FKC Steering

# %%
# ============================================================================
# FKC-steered reverse sampling on a B=1, K=4 mixture
# A Gaussian potential is placed on a single target mode; we compare the
# unguided reverse process (recovers the full mixture) against the steered
# one (concentrates mass on the target mode).
# ============================================================================

# VE schedule with a moderate σ_max so trajectories are visualisable.
ve_sched = VESchedule(sigma_min=0.01, sigma_max=10.0)
mu_mix = torch.tensor([[[-3.0], [-0.5], [2.0], [4.0]]])  # [B=1, K=4, D=1]
sigma_mix = torch.tensor([[[0.3], [0.25], [0.4], [0.2]]])  # [B=1, K=4, D=1]
weight_mix = torch.tensor([[0.25, 0.25, 0.25, 0.25]])  # [B=1, K=4]
gmm_mix = GMM(mu=mu_mix, sigma=sigma_mix, weight=weight_mix, schedule=ve_sched)

TARGET_K = 3  # mode at μ = +4.0
target_c = 3
target_s = 1.0  # potential width


def r(x):
    return -0.5 * (x - target_c) ** 2 / target_s**2


def grad_r(x):
    return -(x - target_c) / target_s**2


def x0_hat(x, t):
    """Tweedie one-step denoiser for VE (α_t ≡ 1):  E[x_0 | x_t] = x_t + σ_t² · score."""
    sigma_t = ve_sched.get_sigma_t(t)
    return x + sigma_t**2 * gmm_mix.score(x, t)


# --- reverse sampling setup ---
# Tiny cutoff away from t=1 (the SDE coefficients are singular there) but still
# essentially the prior. Initial particles ~ marginal at t_max (analytic).
# Tiny cutoff away from t=1 (singular SDE coefficients) — still essentially the prior.
T_MAX, EPS_R, N_R, T_R, ESS = 0.99, 1e-3, 4_000, 400, 0.9
t_rev = torch.linspace(T_MAX, EPS_R, T_R)
x_init = gmm_mix.sample(shape=N_R, t=T_MAX)


def reverse_drift(x_, t_):
    # VE: f = 0, so reverse drift is just −g²·score.
    g = ve_sched.diffusion_coeff(t_)
    return -(g**2) * gmm_mix.score(x_, t_)


# Pure-bootstrap FKC: plain reverse drift, all steering comes from weight
# reweighting + resampling. Twisting the drift at high σ_t requires Jacobian
# corrections AND a finer integrator than Euler can comfortably handle; dropping
# the twist is the robust choice and the weights still target the tilted
# distribution exactly.
guided_drift = reverse_drift


# Stateless FKC potential: Δ log w = r(x̂_0(x_t, t)) · |dt|. Cumulative log_w
# is then ∫₀^T r(x̂_0) dt — a valid Feynman-Kac potential targeting a tilted
# distribution that concentrates on the reward mode. Stateless is essential:
# `steered_reverse_sampling` reshuffles `x = x[idx]` on resample but cannot
# reshuffle external closure state, so any telescoping `log φ_curr − log φ_prev`
# trick gets corrupted after the first resample.
def weight_update(x_, t_, dt):
    return r(x0_hat(x_, t_)).squeeze(-1).squeeze(-1) * dt.abs()


traj_unguided = reverse_sampling(reverse_drift, ve_sched.diffusion_coeff, x_init.clone(), t_rev).detach()
traj_steered, ess_hist = steered_reverse_sampling(
    drift=guided_drift,
    diffusion=ve_sched.diffusion_coeff,
    weight_update=weight_update,
    x=x_init.clone(),
    t=t_rev,
    ess_threshold=ESS,
)
traj_steered = traj_steered.detach()

# --- analytical reference densities ---
xs = torch.linspace(-6, 6, 400).reshape(-1, 1, 1)
log_p = gmm_mix.log_prob(xs, t=torch.tensor(EPS_R)).squeeze().detach()
p_data = log_p.exp()
p_data = p_data / torch.trapezoid(p_data, xs.squeeze())
log_p_tilt = log_p + r(xs).squeeze()
p_tilt = (log_p_tilt - log_p_tilt.max()).exp()
p_tilt = p_tilt / torch.trapezoid(p_tilt, xs.squeeze())

# --- figure: trajectories side-by-side, plus densities & ESS ---
N_PLOT = 600  # subset of particles to render for legibility
idx_plot = torch.randperm(N_R)[:N_PLOT]

fig2 = plt.figure(figsize=(16, 11))
gs2 = gridspec.GridSpec(2, 2, height_ratios=[3, 1.3], hspace=0.3, wspace=0.18)
ax_un = fig2.add_subplot(gs2[0, 0])
ax_st = fig2.add_subplot(gs2[0, 1], sharey=ax_un)
ax_dens = fig2.add_subplot(gs2[1, 0])
ax_ess = fig2.add_subplot(gs2[1, 1])

t_rev_np = t_rev.numpy()
ax_un.plot(t_rev_np, traj_unguided[:, idx_plot, 0, 0], color="steelblue", alpha=0.08, lw=0.5)
ax_un.axhline(target_c, color="firebrick", ls="--", lw=1.2, alpha=0.8, label=f"target μ={target_c}")
ax_un.set_title(r"$\leftarrow$ Unguided reverse sampling $\leftarrow$")
ax_un.set_xlabel("t")
ax_un.set_ylabel("x")
y_lim_traj = float(traj_unguided[:, idx_plot, 0, 0].abs().max()) * 1.05
ax_un.set_ylim(-y_lim_traj, y_lim_traj)
ax_un.legend(loc="upper left", fontsize=10)

ax_st.plot(t_rev_np, traj_steered[:, idx_plot, 0, 0], color="darkorange", alpha=0.08, lw=0.5)
ax_st.axhline(target_c, color="firebrick", ls="--", lw=1.2, alpha=0.8)
ax_st.set_title(r"$\leftarrow$ FKC-steered reverse sampling $\leftarrow$")
ax_st.set_xlabel("t")

# --- bottom-left: empirical histograms vs analytical ---
xs_np = xs.squeeze().numpy()
dx = float(xs[1, 0, 0] - xs[0, 0, 0])
edges = torch.cat([xs[0:1, 0, 0] - dx / 2, xs[:, 0, 0] + dx / 2]).numpy()

h_un, _ = torch.histogram(traj_unguided[-1, :, 0, 0], bins=torch.tensor(edges), density=True)
h_st, _ = torch.histogram(traj_steered[-1, :, 0, 0], bins=torch.tensor(edges), density=True)

ax_dens.plot(xs_np, p_data, color="steelblue", lw=1.5, label="data $p$")
ax_dens.plot(xs_np, p_tilt, color="firebrick", lw=1.5, label=r"tilted $p \cdot e^{r}$")
ax_dens.bar(xs_np, h_un.numpy(), width=dx, alpha=0.35, color="steelblue", label="unguided hist", align="center")
ax_dens.bar(xs_np, h_st.numpy(), width=dx, alpha=0.45, color="darkorange", label="steered hist", align="center")
ax_dens.axvline(target_c, color="firebrick", ls="--", lw=1, alpha=0.7)
ax_dens.set_xlim(-6, 6)
ax_dens.set_xlabel("x")
ax_dens.set_ylabel("density")
ax_dens.set_title(f"final-time marginals  (potential on K={TARGET_K}, μ={target_c})")
ax_dens.legend(fontsize=8)

# --- bottom-right: ESS history on the same time axis as the trajectories ---
ess_t = t_rev_np[: len(ess_hist)]  # ess_hist[k] corresponds to t_rev[k]
ax_ess.plot(ess_t, ess_hist, color="darkorange", lw=1.0, label="ESS / N")
ax_ess.axhline(ESS, color="red", ls="--", lw=1, label=f"threshold={ESS}")
ax_ess.set_ylim(0, 1.05)
ax_ess.set_xlim(0, 1)
ax_ess.set_xlabel("t")
ax_ess.set_ylabel("ESS / N")
n_resamples = sum(1 for e in ess_hist if e < ESS)
ax_ess.set_title(f"ESS history  ({n_resamples} resamples)")
ax_ess.legend(fontsize=8)

plt_show()
