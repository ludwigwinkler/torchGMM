"""
Ping-pong GIF of forward & reverse diffusion on a 1-D multi-mode GMM.

x-axis = time (1 → 0),  y-axis = sample value.
Phase 1 (blue):   forward process draws trajectories right-to-left  (data → noise).
Phase 2 (red):    reverse process overdraws left-to-right  (noise → data).
Phase 3 (green):  steered reverse process overdraws left-to-right  (noise → reward-tilted data).

Forward trajectories stay visible so the overlap with the reverse is apparent.
Marginal densities are shown on the left (noise, t=1) and right (data, t=0) edges.
The GIF loops seamlessly: forward ↔ reverse ↔ steered.
"""

from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.stats import norm
from torchGMM import GMM, BetaSchedule, forward_sampling, reverse_sampling
from torchGMM.sampling import steered_reverse_sampling

plt.style.use("default")
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    }
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_SAMPLES = 300
N_STEPS = 500
SEED = 42
GIF_PATH = Path(__file__).parent / "steered_diffusion.gif"
FPS = 15
DPI = 120
STEPS_PER_FRAME = 2

# Reward config
REWARD_CENTER = -2.5
REWARD_SIGMA = 1.5
ESS_THRESHOLD = 0.9

# ---------------------------------------------------------------------------
# Setup 1-D multi-mode GMM
# ---------------------------------------------------------------------------
torch.manual_seed(SEED)

mu = torch.tensor([[[2], [-2.0]]])  # [1, K=2, D=1]
sigma = torch.tensor([[[1.0], [1.0]]])  # [1, K=2, D=1]
weight = torch.tensor([[0.7, 0.3]])  # [1, K=2]

schedule = BetaSchedule(beta_min=0.1, beta_max=20.0)
gmm = GMM(mu, sigma, weight, schedule=schedule)

# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------


def r(x):
    return -0.5 * (x - REWARD_CENTER) ** 2 / REWARD_SIGMA**2


def grad_r(x):
    return -(x - REWARD_CENTER) / REWARD_SIGMA**2


# ---------------------------------------------------------------------------
# Forward diffusion  (t: 0 → 1,  data → noise)
# ---------------------------------------------------------------------------
print("Simulating forward diffusion ...")
t_fwd = torch.linspace(0, 1, N_STEPS)
x0 = gmm.sample(N_SAMPLES)  # [N, 1, 1]
traj_fwd = forward_sampling(schedule.forward_drift, schedule.diffusion_coeff, x0, t_fwd)  # [T, N, 1, 1]

# ---------------------------------------------------------------------------
# Reverse diffusion  (t: 1 → 0,  noise → data)
# ---------------------------------------------------------------------------
print("Simulating reverse diffusion ...")
t_rev = torch.linspace(1, 0, N_STEPS)
x_noise = torch.randn_like(x0)


def reverse_drift(x_, t_):
    f = schedule.forward_drift(x_, t_)
    g = schedule.diffusion_coeff(t_)
    return f - g**2 * gmm.score(x_, t_)


traj_rev = reverse_sampling(reverse_drift, schedule.diffusion_coeff, x_noise, t_rev)  # [T, N, 1, 1]

# ---------------------------------------------------------------------------
# Steered reverse diffusion  (t: 1 → 0,  noise → reward-tilted data)
# ---------------------------------------------------------------------------
print("Simulating steered reverse diffusion ...")
EPS = 0.001
t_steer = torch.linspace(1 - EPS, EPS, N_STEPS)
x_noise_steer = torch.randn(N_SAMPLES, 1, 1)


def guided_drift(x_, t_):
    f = schedule.forward_drift(x_, t_)
    g = schedule.diffusion_coeff(t_)
    sc = gmm.score(x_, t_)
    beta = 1.0 - t_
    return f - g**2 * sc - beta * (g**2 / 2) * grad_r(x_)


def weight_update(x_, t_, dt):
    f = schedule.forward_drift(x_, t_)
    g = schedule.diffusion_coeff(t_)
    sc = gmm.score(x_, t_)
    rg, rv = grad_r(x_), r(x_)
    beta = 1.0 - t_
    return (rv - beta * rg * f + beta * rg * (g**2 / 2) * sc).squeeze(-1).squeeze(-1) * dt.abs()


traj_steer, ess_hist = steered_reverse_sampling(
    guided_drift, schedule.diffusion_coeff, weight_update, x_noise_steer, t_steer, ess_threshold=ESS_THRESHOLD
)

# Squeeze to [T, N]
traj_fwd_np = traj_fwd[:, :, 0, 0].detach().numpy()
traj_rev_np = traj_rev[:, :, 0, 0].detach().numpy()
traj_steer_np = traj_steer[:, :, 0, 0].detach().numpy()
t_fwd_np = t_fwd.numpy()  # 0 → 1
t_rev_np = t_rev.numpy()  # 1 → 0
t_steer_np = t_steer.numpy()  # 1 → 0

# Plot x-axis: plot_x = 1 - t so noise (t=1) on left, data (t=0) on right.
t_plot_fwd = 1.0 - t_fwd_np  # 1 → 0  (right → left)
t_plot_rev = 1.0 - t_rev_np  # 0 → 1  (left → right)
t_plot_steer = 1.0 - t_steer_np  # 0 → 1  (left → right)

Y_LIM = max(float(np.abs(traj_fwd_np).max()), float(np.abs(traj_rev_np).max()), float(np.abs(traj_steer_np).max()), 6.0)

# ---------------------------------------------------------------------------
# Compute marginal densities for side panels
# ---------------------------------------------------------------------------
y_grid = np.linspace(-Y_LIM, Y_LIM, 400)

# Data distribution (t=0): GMM density
mu_np = mu[0, :, 0].numpy()
sigma_np = sigma[0, :, 0].numpy()
weight_np = (weight[0] / weight[0].sum()).numpy()
p_data = np.zeros_like(y_grid)
for k in range(len(mu_np)):
    p_data += weight_np[k] * norm.pdf(y_grid, mu_np[k], sigma_np[k])

# Reward-tilted distribution
y_grid_t = torch.tensor(y_grid, dtype=torch.float32).reshape(-1, 1, 1)
log_p = gmm.log_prob(y_grid_t, t=torch.tensor(EPS)).squeeze().detach().numpy()
log_p_rew = log_p + r(y_grid_t).squeeze().numpy()
log_p_rew -= log_p_rew.max()
p_rew = np.exp(log_p_rew)
p_rew /= np.trapezoid(p_rew, y_grid)

# Noise distribution (t=1): standard normal
p_noise = norm.pdf(y_grid, 0, 1)

# Precompute marginals at every time step for the animated right panel
print("Precomputing marginals ...")
marginals_unsteered = np.zeros((N_STEPS, len(y_grid)))
marginals_steered = np.zeros((N_STEPS, len(y_grid)))
r_np = r(y_grid_t).squeeze().numpy()
for ti in range(N_STEPS):
    t_val = t_steer[ti]
    beta_t = 1.0 - t_val.item()
    log_p_t = gmm.log_prob(y_grid_t, t=t_val).squeeze().detach().numpy()
    # Unsteered marginal
    log_p_un = log_p_t - log_p_t.max()
    marginals_unsteered[ti] = np.exp(log_p_un)
    marginals_unsteered[ti] /= np.trapezoid(marginals_unsteered[ti], y_grid)
    # Steered (reward-tilted) marginal
    log_p_tilted = log_p_t + beta_t * r_np
    log_p_tilted -= log_p_tilted.max()
    marginals_steered[ti] = np.exp(log_p_tilted)
    marginals_steered[ti] /= np.trapezoid(marginals_steered[ti], y_grid)

# ---------------------------------------------------------------------------
# Animate
# ---------------------------------------------------------------------------
n_anim_steps = N_STEPS // STEPS_PER_FRAME
normal_dur = 1000 // FPS
fast_dur = normal_dur // 2  # 2x faster

# Forward: 2x fast. Reverse first half: 2x fast. Reverse second half: normal.
frame_indices = list(range(n_anim_steps))  # forward
reverse_mid = n_anim_steps + n_anim_steps // 2
frame_indices += list(range(n_anim_steps, 2 * n_anim_steps))  # reverse

frame_durations = []
frame_durations += [fast_dur] * n_anim_steps  # forward: fast
frame_durations += [fast_dur] * (reverse_mid - n_anim_steps)  # reverse first half: fast
frame_durations += [normal_dur] * (2 * n_anim_steps - reverse_mid)  # reverse second half: normal
N_FRAMES = len(frame_indices)

print(f"Rendering {N_FRAMES} frames ...")

fig = plt.figure(figsize=(12, 6), dpi=DPI)
fig.patch.set_facecolor("white")
gs = gridspec.GridSpec(2, 3, width_ratios=[1, 8, 1], height_ratios=[3, 1], wspace=0.05, hspace=0.35)

ax_left = fig.add_subplot(gs[0, 0])  # noise density (t=1, left)
ax_main = fig.add_subplot(gs[0, 1])  # trajectories
ax_right = fig.add_subplot(gs[0, 2])  # data density (t=0, right)
ax_ess = fig.add_subplot(gs[1, 1])  # ESS below centre

# --- Main axis ---
ax_main.set_xlim(0, 1)
ax_main.set_ylim(-Y_LIM, Y_LIM)
ax_main.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax_main.set_xticklabels([])  # hide — shared with ESS below
ax_main.grid(True, alpha=0.15)
ax_main.set_yticklabels([])
title = ax_main.set_title("", fontsize=13, fontweight="bold")

# --- Left panel: noise distribution (horizontal density, y shared) ---
ax_left.plot(p_noise, y_grid, color="firebrick", lw=1.2)
ax_left.fill_betweenx(y_grid, 0, p_noise, color="firebrick", alpha=0.15)
ax_left.set_ylim(-Y_LIM, Y_LIM)
ax_left.set_xlim(ax_left.get_xlim()[::-1])  # flip so density grows toward main
ax_left.set_yticklabels([])
ax_left.set_xticks([])
ax_left.set_title(r"Noise", fontsize=10, color="firebrick")

# --- Right panel: data + reward-tilted distribution (solid = targets) ---
ax_right.plot(p_data, y_grid, color="steelblue", lw=1.2, ls="-")
ax_right.fill_betweenx(y_grid, 0, p_data, color="steelblue", alpha=0.15)
ax_right.plot(p_rew, y_grid, color="seagreen", lw=1.2, ls="-")
ax_right.fill_betweenx(y_grid, 0, p_rew, color="seagreen", alpha=0.1)
ax_right.set_ylim(-Y_LIM, Y_LIM)
ax_right.set_xlim(0, max(p_data.max(), p_rew.max()) * 1.1)
ax_right.set_yticklabels([])
ax_right.set_xticks([])
ax_right.set_title(r"Data", fontsize=10, color="steelblue")

# --- ESS panel (below trajectories) ---
# Plot full ESS curve as faint background, animated line reveals progressively
t_ess_plot = 1.0 - t_steer_np[:-1]  # ESS has N_STEPS-1 entries, x = 1-t
ax_ess.set_xlim(0, 1)
ax_ess.set_ylim(0.8, 1.01)
ax_ess.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax_ess.set_xticklabels([r"$1$", r"$0.75$", r"$0.5$", r"$0.25$", r"$0$"])
ax_ess.set_xlabel("$t$", fontsize=12)
ax_ess.set_ylabel("ESS/N", fontsize=10)
ax_ess.axhline(ESS_THRESHOLD, color="gray", ls="--", lw=0.8, alpha=0.5)
ax_ess.plot(t_ess_plot, ess_hist, color="seagreen", lw=0.5, alpha=0.2)  # faint full curve
ax_ess.grid(True, alpha=0.15)

# Hide the two bottom corner cells
for side_ax in [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 2])]:
    side_ax.set_visible(False)

fig.tight_layout()

# Animated marginal lines on the right panel (dashed = dynamic)
(marginal_unsteered,) = ax_right.plot([], [], color="steelblue", lw=1.5, ls="--", alpha=0.8)
(marginal_steered,) = ax_right.plot([], [], color="seagreen", lw=1.5, ls="--", alpha=0.8)

# Animated ESS line
(ess_line,) = ax_ess.plot([], [], color="seagreen", lw=1.5)

# Pre-allocate line artists
fwd_lines = []
for j in range(N_SAMPLES):
    (ln,) = ax_main.plot([], [], color="steelblue", alpha=0.07, lw=0.5)
    fwd_lines.append(ln)

rev_lines = []
for j in range(N_SAMPLES):
    (ln,) = ax_main.plot([], [], color="firebrick", alpha=0.07, lw=0.5)
    rev_lines.append(ln)

steer_lines = []
for j in range(N_SAMPLES):
    (ln,) = ax_main.plot([], [], color="seagreen", alpha=0.07, lw=0.5)
    steer_lines.append(ln)

all_artists = fwd_lines + rev_lines + steer_lines + [title, marginal_unsteered, marginal_steered, ess_line]


def _frame(i: int):
    if i < n_anim_steps:
        # Phase 1: Forward — progressively reveal blue, hide red and green
        idx = min((i + 1) * STEPS_PER_FRAME, N_STEPS)
        for j in range(N_SAMPLES):
            fwd_lines[j].set_data(t_plot_fwd[:idx], traj_fwd_np[:idx, j])
            rev_lines[j].set_data([], [])
            steer_lines[j].set_data([], [])
        t_val = t_fwd_np[idx - 1]
        title.set_text(r"$\longleftarrow$ Forward Diffusion $\longleftarrow$  $t = " + f"{t_val:.2f}" + r"$")
        marginal_unsteered.set_data([], [])
        marginal_steered.set_data([], [])
        ess_line.set_data([], [])

    else:
        # Phase 2: Reverse + Steered simultaneously — keep forward, reveal red and green together
        for j in range(N_SAMPLES):
            fwd_lines[j].set_data(t_plot_fwd, traj_fwd_np[:, j])
        ri = i - n_anim_steps
        idx = min((ri + 1) * STEPS_PER_FRAME, N_STEPS)
        for j in range(N_SAMPLES):
            rev_lines[j].set_data(t_plot_rev[:idx], traj_rev_np[:idx, j])
            steer_lines[j].set_data(t_plot_steer[:idx], traj_steer_np[:idx, j])
        t_val = t_rev_np[idx - 1]
        title.set_text(
            r"$\longrightarrow$ Reverse (red) vs Steered (green) $\longrightarrow$  $t = " + f"{t_val:.2f}" + r"$"
        )
        # Update right panel: current marginals at this time step (dashed)
        marginal_unsteered.set_data(marginals_unsteered[idx - 1], y_grid)
        marginal_steered.set_data(marginals_steered[idx - 1], y_grid)
        # Update ESS panel: progressively reveal
        ess_idx = min(idx - 1, len(ess_hist))
        ess_line.set_data(t_ess_plot[:ess_idx], ess_hist[:ess_idx])

    return all_artists


# Render frames manually with per-frame durations for smooth slowdown
from io import BytesIO

from PIL import Image

pil_frames = []
for fi in range(N_FRAMES):
    _frame(frame_indices[fi])
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight")
    buf.seek(0)
    pil_frames.append(Image.open(buf).copy())
    buf.close()

pil_frames[0].save(
    str(GIF_PATH),
    save_all=True,
    append_images=pil_frames[1:],
    duration=frame_durations,
    loop=0,
)
plt.close(fig)

total_dur = sum(frame_durations) / 1000
print(f"GIF saved to {GIF_PATH}  ({N_FRAMES} frames, {total_dur:.1f}s)")
