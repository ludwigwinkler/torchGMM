"""
Ping-pong GIF of forward & reverse diffusion on a 1-D multi-mode GMM.

x-axis = time (1 → 0),  y-axis = sample value.
Phase 1 (blue):   forward process draws trajectories right-to-left  (data → noise).
Phase 2 (red):    reverse process overdraws left-to-right  (noise → data).

Forward trajectories stay visible so the overlap with the reverse is apparent.
Marginal densities are shown on the left (noise, t=1) and right (data, t=0) edges.
The GIF loops seamlessly: forward ↔ reverse.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.stats import norm
from pathlib import Path
from torchGMM import GMM, BetaSchedule, forward_sampling, reverse_sampling

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
N_STEPS = 200
SEED = 42
GIF_PATH = Path(__file__).parent / "forward_backward_diffusion.gif"
FPS = 15  # slowed down another 20% from 19
DPI = 120
STEPS_PER_FRAME = 2

# ---------------------------------------------------------------------------
# Setup 1-D multi-mode GMM
# ---------------------------------------------------------------------------
torch.manual_seed(SEED)

mu = torch.tensor([[[-3.0], [-0.5], [2.0], [4.0]]])  # [1, K=4, D=1]
sigma = torch.tensor([[[0.3], [0.25], [0.4], [0.2]]])  # [1, K=4, D=1]
weight = torch.tensor([[0.3, 0.25, 0.3, 0.15]])  # [1, K=4]

schedule = BetaSchedule(beta_min=0.1, beta_max=20.0)
gmm = GMM(mu, sigma, weight, schedule=schedule)

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

# Squeeze to [T, N]
traj_fwd_np = traj_fwd[:, :, 0, 0].detach().numpy()
traj_rev_np = traj_rev[:, :, 0, 0].detach().numpy()
t_fwd_np = t_fwd.numpy()  # 0 → 1
t_rev_np = t_rev.numpy()  # 1 → 0

# Plot x-axis: plot_x = 1 - t_diffusion  so data (t=0) is on the right.
t_plot_fwd = 1.0 - t_fwd_np  # 1 → 0  (right → left)
t_plot_rev = 1.0 - t_rev_np  # 0 → 1  (left → right)

Y_LIM = max(float(np.abs(traj_fwd_np).max()), float(np.abs(traj_rev_np).max()), 6.0)

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

# Noise distribution (t=1): standard normal
p_noise = norm.pdf(y_grid, 0, 1)

# ---------------------------------------------------------------------------
# Animate
# ---------------------------------------------------------------------------
n_anim_steps = N_STEPS // STEPS_PER_FRAME
N_FRAMES = 2 * n_anim_steps  # forward + reverse

print(f"Rendering {N_FRAMES} frames ...")

fig = plt.figure(figsize=(12, 4.5), dpi=DPI)
fig.patch.set_facecolor("white")
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 8, 1], wspace=0.05)

ax_left = fig.add_subplot(gs[0])  # noise density
ax_main = fig.add_subplot(gs[1])  # trajectories
ax_right = fig.add_subplot(gs[2])  # data density

# --- Main axis ---
ax_main.set_xlim(0, 1)
ax_main.set_ylim(-Y_LIM, Y_LIM)
ax_main.set_xlabel("$t$", fontsize=12)
ax_main.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax_main.set_xticklabels([r"$0$", r"$0.25$", r"$0.5$", r"$0.75$", r"$1$"])
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

# --- Right panel: data distribution ---
ax_right.plot(p_data, y_grid, color="steelblue", lw=1.2)
ax_right.fill_betweenx(y_grid, 0, p_data, color="steelblue", alpha=0.15)
ax_right.set_ylim(-Y_LIM, Y_LIM)
ax_right.set_yticklabels([])
ax_right.set_xticks([])
ax_right.set_title(r"Data", fontsize=10, color="steelblue")

fig.tight_layout()

# Pre-allocate line artists: one set for forward (persistent), one for reverse
fwd_lines = []
for j in range(N_SAMPLES):
    (ln,) = ax_main.plot([], [], color="steelblue", alpha=0.07, lw=0.5)
    fwd_lines.append(ln)

rev_lines = []
for j in range(N_SAMPLES):
    (ln,) = ax_main.plot([], [], color="firebrick", alpha=0.07, lw=0.5)
    rev_lines.append(ln)

all_artists = fwd_lines + rev_lines + [title]


def _frame(i: int):
    is_forward = i < n_anim_steps

    if is_forward:
        # Forward phase: progressively reveal blue trajectories, hide red
        idx = min((i + 1) * STEPS_PER_FRAME, N_STEPS)
        for j in range(N_SAMPLES):
            fwd_lines[j].set_data(t_plot_fwd[:idx], traj_fwd_np[:idx, j])
            rev_lines[j].set_data([], [])
        t_val = t_fwd_np[idx - 1]
        title.set_text(r"$\longleftarrow$ Forward Diffusion $\longleftarrow$  $t = " + f"{t_val:.2f}" + r"$")
    else:
        # Reverse phase: keep forward lines fully visible, progressively reveal red
        # Ensure forward lines show the complete trajectory
        for j in range(N_SAMPLES):
            fwd_lines[j].set_data(t_plot_fwd, traj_fwd_np[:, j])

        ri = i - n_anim_steps
        idx = min((ri + 1) * STEPS_PER_FRAME, N_STEPS)
        for j in range(N_SAMPLES):
            rev_lines[j].set_data(t_plot_rev[:idx], traj_rev_np[:idx, j])
        t_val = t_rev_np[idx - 1]
        title.set_text(r"$\longrightarrow$ Reverse Diffusion $\longrightarrow$  $t = " + f"{t_val:.2f}" + r"$")

    return all_artists


anim = FuncAnimation(fig, _frame, frames=N_FRAMES, interval=1000 // FPS, blit=True)
anim.save(str(GIF_PATH), writer=PillowWriter(fps=FPS), dpi=DPI)
plt.close(fig)

print(f"GIF saved to {GIF_PATH}  ({N_FRAMES} frames, {N_FRAMES / FPS:.1f}s)")
