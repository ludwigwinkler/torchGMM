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

from torchGMM import GMM, KarrasSchedule, forward_sampling

plt.style.use("default")
plt.rcdefaults()

torch.manual_seed(0)
device = torch.device("cpu")
torch.set_default_device(device)

# --- model: B=4 separate GMMs, each with K=1 component ---
schedule = KarrasSchedule()
mu = torch.tensor([[[-3.0]], [[-0.5]], [[2.0]], [[4.0]]])  # [B=4, K=1, D=1]
sigma = torch.tensor([[[0.3]], [[0.25]], [[0.4]], [[0.2]]])  # [B=4, K=1, D=1]
weight = torch.ones(4, 1)
gmm = GMM(mu=mu, sigma=sigma, weight=weight, schedule=schedule)
B = mu.shape[0]

# --- forward simulation ---
EPS, N, T = 1e-3, 500, 400
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
        ax.plot(ts, seg[:, :, b], color=colors[b], alpha=0.05, lw=0.5)
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
