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
T_MAX, EPS_R, N_R, T_R, ESS = 0.99, 1e-3, 4_000, 400, 0.95
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

# %% [markdown]
# # AF3 Forward Perturbation Kernel
#
# AF3 (supplement §3.7.1, eq. 7) defines the inference noise schedule
#     σ̂(t) = σ_data · (s_max^{1/p} + t · (s_min^{1/p} − s_max^{1/p}))^p,
# with s_max=160, s_min=4·10⁻⁴, p=7, t∈[0,1] in steps of 1/200.
# Because the schedule is variance-exploding (α_t ≡ 1), the perturbation kernel is
#     p(x_t | x_0) = N(x_t; x_0, σ̂(t)² I),
# i.e. samples can be drawn analytically as x_t = x_0 + σ̂(t) · ε without integrating
# any SDE. This is the same kernel AF3 uses to corrupt training structures during
# diffusion training (with σ̂ sampled from σ_data·exp(−1.2 + 1.5·N(0,1))).

# %%
# AF3 paper schedule values: σ_max=160, σ_min=4·10⁻⁴, ρ=7 (eq. 7 of §3.7.1).
# σ_data is kept at 1 because our GMM modes live at unit scale rather than the
# 16 Å atomic-coordinate scale AF3 was tuned for; this preserves the schedule
# *shape* (and the γ_min=1 crossing at τ/T≈0.66) while keeping the data at the
# scale of our toy mixture. Plotting uses a symlog y-axis to span both σ_max
# (≈160) and the unit-scale data simultaneously.
af3_sched = KarrasSchedule(sigma_min=4e-4, sigma_max=160.0, rho=7.0, sigma_data=1.0)

# Re-use the B=1, K=4 mixture but bind it to the AF3 schedule.
gmm_af3 = GMM(mu=mu_mix, sigma=sigma_mix, weight=weight_mix, schedule=af3_sched)

# t grid: AF3 inference uses 200 uniform steps in [0,1].
T_AF3 = 200
t_grid = torch.linspace(0.0, 1.0, T_AF3 + 1)
sigma_grid = af3_sched.get_sigma_t(t_grid)  # σ̂(t), descending order if we reverse t

N_KERNEL = 4_000
x0_kernel = gmm_af3.sample(shape=N_KERNEL, t=torch.tensor(0.0))  # [N, 1, 1]

# Pick a few representative noise levels to visualise the perturbation kernel.
# In our `KarrasSchedule`, σ̂(t=0)=σ_min and σ̂(t=1)=σ_max — i.e. t parametrises
# diffusion *progress* from clean (t=0) to noisy (t=1).
t_show = torch.tensor([0.0, 0.05, 0.15, 0.35, 0.6, 0.85, 1.0])
sigma_show = af3_sched.get_sigma_t(t_show)

# Direct kernel sample: x_t = x_0 + σ̂(t) · ε,  ε ~ N(0, I)
eps = torch.randn(t_show.numel(), N_KERNEL, 1, 1)
x_kernel = x0_kernel.unsqueeze(0) + sigma_show.view(-1, 1, 1, 1) * eps  # [T_show, N, 1, 1]

# Analytical reference density at each t via the GMM's closed-form marginal.
xs_kernel = torch.linspace(-50, 50, 600).reshape(-1, 1, 1)
p_marg = [gmm_af3.log_prob(xs_kernel, t=t_).exp().squeeze().detach() for t_ in t_show]

fig3, axes = plt.subplots(1, t_show.numel(), figsize=(20, 3.2), sharey=False)
for k, ax in enumerate(axes):
    samples_k = x_kernel[k, :, 0, 0].numpy()
    sig_k = float(sigma_show[k])
    # auto x-range: cover ~3σ̂ around the data span
    rng = max(6.0, 3.0 * sig_k)
    ax.hist(samples_k, bins=80, range=(-rng, rng), density=True, color="steelblue", alpha=0.55)
    mask = (xs_kernel[:, 0, 0] >= -rng) & (xs_kernel[:, 0, 0] <= rng)
    ax.plot(xs_kernel[mask, 0, 0], p_marg[k][mask], color="firebrick", lw=1.4, label="analytical $p(x,t)$")
    ax.set_title(rf"$t={float(t_show[k]):.2f},\ \hat\sigma={sig_k:.3g}$", fontsize=10)
    ax.set_xlim(-rng, rng)
    ax.grid(True, alpha=0.2)
    if k == 0:
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
fig3.suptitle("AF3 perturbation kernel: $x_t = x_0 + \\hat\\sigma(t)\\,\\varepsilon$", y=1.02)
plt_show()

# %% [markdown]
# # AF3 Reverse Diffusion (Algorithm 18)
#
# Implements AF3 supplement Algorithm 18 verbatim, minus the 3D-only
# `CentreRandomAugmentation` (no-op for 1D scalars). The DiffusionModule role is
# played by the analytical Tweedie denoiser
#     D(x, σ̂) = E[x_0 | x_σ̂] = x + σ̂² · ∇log p_{σ̂}(x),
# where the score is computed in closed form by the GMM at the time t(σ̂)
# obtained by inverting the Karras σ-schedule.
#
# Hyperparameters from the paper:
#     γ_0 = 0.8, γ_min = 1.0,  noise_scale λ = 1.003,  step_scale η = 1.5.
#
# Note on line 9 of Algorithm 18 (`δ = (x_l − x_denoised)/t̂`): as printed it
# uses the pre-noise-injection `x_l`. Standard EDM (Karras 2022) and the AF3
# reference implementation use `(x_noisy − x_denoised)/t̂` — the consensus is
# that the supplement has a typo. The flag `USE_PAPER_DELTA` toggles between
# the literal-paper form and the EDM-corrected form.

# %%
USE_PAPER_DELTA = False  # False -> EDM-style; True -> verbatim Algorithm 18 line 9


def af3_denoiser(x: torch.Tensor, sigma_hat: torch.Tensor, gmm_: GMM) -> torch.Tensor:
    """Analytical Tweedie denoiser at arbitrary σ̂, bypassing any schedule t-grid.

    For a VE process (α≡1) the noised GMM at level σ̂ is itself a GMM with
    component variances σ_k² + σ̂², so its score is closed-form. This avoids
    the `Schedule._clamp_t` saturation that occurs whenever γ-churn inflates
    σ̂ above the schedule's σ_max (which would otherwise compute the score at
    the wrong noise level — the bug that produced the spurious contraction).
    """
    # Effective component σ at the current noise level.
    sigma_eff = (gmm_.sigma**2 + sigma_hat**2).sqrt()  # [*B, K, D]
    # Build a noised GMM with identity schedule: querying it at t=0 returns
    # exactly the σ̂-marginal score. Re-using the parent's schedule is fine
    # because score(x, t=0) only uses (μ, σ_eff, weight) at α=1, σ=0.
    noised = GMM(mu=gmm_.mu.clone(), sigma=sigma_eff, weight=gmm_.weight.clone(), schedule=gmm_.schedule)
    score_hat = noised.score(x, t=torch.tensor(0.0))  # [*N, *B, D]
    return x + sigma_hat**2 * score_hat


def sample_diffusion(
    gmm_: GMM,
    sigma_grid_desc: torch.Tensor,  # [c_0, c_1, ..., c_T] descending; c_0 = σ_max, c_T = σ_min
    n_particles: int,
    gamma_0: float = 0.8,
    gamma_min: float = 1.0,
    noise_scale: float = 1.003,
    step_scale: float = 1.5,
    use_paper_delta: bool = USE_PAPER_DELTA,
) -> torch.Tensor:
    """AF3 Algorithm 18, returning the full trajectory [T+1, N, *B, D]."""
    *B, _, D = gmm_.mu.shape  # *batch, K, D
    c0 = sigma_grid_desc[0]
    x = c0 * torch.randn(n_particles, *B, D)  # line 1
    traj = [x.clone()]
    for tau in range(1, sigma_grid_desc.numel()):  # line 2
        c_prev = sigma_grid_desc[tau - 1]
        c_curr = sigma_grid_desc[tau]
        # line 3: CentreRandomAugmentation — identity in 1D / unconditional toy.
        gamma = gamma_0 if c_curr > gamma_min else 0.0  # line 4 (paper text)
        t_hat = c_prev * (gamma + 1.0)  # line 5
        # line 6-7: noise injection — guarded sqrt for the γ=0 case.
        var_inject = (t_hat**2 - c_prev**2).clamp(min=0.0)
        xi = noise_scale * torch.sqrt(var_inject) * torch.randn_like(x)
        x_noisy = x + xi
        # line 8: denoise at the inflated noise level t̂ (analytical, schedule-free).
        x_denoised = af3_denoiser(x_noisy, t_hat, gmm_)
        # line 9: δ = (x_? − x_denoised)/t̂  (see USE_PAPER_DELTA note).
        ref = x if use_paper_delta else x_noisy
        delta = (ref - x_denoised) / t_hat
        # lines 10-11: Heun-style update with step scaling η.
        dt = c_curr - t_hat
        x = x_noisy + step_scale * dt * delta
        traj.append(x.clone())
    return torch.stack(traj)


# AF3 eq. 7 places σ_max at AF3-t=0 and σ_min at AF3-t=1 — i.e. opposite to our
# `KarrasSchedule`'s forward convention. Build the descending grid by flipping.
# Also keep an "AF3 step-progress" axis going noise (0) → data (1) for plotting.
af3_step = torch.linspace(0.0, 1.0, T_AF3 + 1)  # τ/T
sigma_grid_desc = af3_sched.get_sigma_t(t_grid).flip(0)  # c_0 = σ_max … c_T = σ_min
assert sigma_grid_desc[0] > sigma_grid_desc[-1]

N_AF3 = 2_000
torch.manual_seed(1)
traj_af3 = sample_diffusion(
    gmm_af3,
    sigma_grid_desc=sigma_grid_desc,
    n_particles=N_AF3,
).detach()  # [T_AF3+1, N, 1, 1]

# Reference densities: clean data marginal (t=0) and noise prior (t=1).
xs_af3 = torch.linspace(-8, 8, 500).reshape(-1, 1, 1)
p_data_af3 = gmm_af3.log_prob(xs_af3, t=torch.tensor(0.0)).exp().squeeze().detach()
p_data_af3 = p_data_af3 / torch.trapezoid(p_data_af3, xs_af3.squeeze())

# Noise marginal at σ_max≈160 — needs ~3·σ_max range to be visible.
noise_lim_af3 = float(sigma_grid_desc[0]) * 3.0  # ≈ 480
xs_noise_af3 = torch.linspace(-noise_lim_af3, noise_lim_af3, 500).reshape(-1, 1, 1)
p_noise_af3 = gmm_af3.log_prob(xs_noise_af3, t=torch.tensor(1.0)).exp().squeeze().detach()
# Data-panel half-width (linear, tight around the modes).
data_lim_af3 = 8.0

# --- figure: top = noise | traj | data, mid = per-segment traj panels (as in
# the section-1 forward plot), bottom = σ-schedule | final-marginal histogram.
fig4 = plt.figure(figsize=(18, 13))
outer4 = gridspec.GridSpec(3, 1, height_ratios=[3, 1.3, 1.3], hspace=0.4)

top4 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer4[0], width_ratios=[1, 8, 1], wspace=0.15)
# Reverse-process layout: noise on the left (start, τ/T=0), data on the right
# (end, τ/T=1). Each marginal panel uses a y-extent matched to its own
# distribution — ±3σ_max for the noise prior, ±data_lim for the data modes.
ax_n = fig4.add_subplot(top4[0])
ax_t = fig4.add_subplot(top4[1])
ax_d = fig4.add_subplot(top4[2])

# Left: noise prior at σ_max (start of reverse process).
ax_n.plot(p_noise_af3.numpy(), xs_noise_af3[:, 0, 0].numpy(), color="darkorange", lw=1.5)
ax_n.fill_betweenx(xs_noise_af3[:, 0, 0].numpy(), 0, p_noise_af3.numpy(), color="darkorange", alpha=0.2)
ax_n.invert_xaxis()  # density peaks toward the trajectory panel
ax_n.set_title(r"$p(x, t \approx 1)$  (noise prior)")
ax_n.set_ylabel("x")
ax_n.set_xticks([])
ax_n.set_ylim(-noise_lim_af3, noise_lim_af3)

# Centre: trajectories vs AF3 step progress, going noise (left) → data (right).
N_PLOT_AF3 = 600
idx_p = torch.randperm(N_AF3)[:N_PLOT_AF3]
ax_t.plot(af3_step.numpy(), traj_af3[:, idx_p, 0, 0].numpy(), color="darkorange", alpha=0.04, lw=0.5)
for m in mu_mix.flatten().tolist():
    ax_t.axhline(m, color="firebrick", ls="--", lw=0.8, alpha=0.5)
ax_t.set_title(r"$\rightarrow$ AF3 SampleDiffusion (Algorithm 18) $\rightarrow$")
ax_t.set_xlabel(r"AF3 step  $\tau / T$  (noise $\to$ data)")
ax_t.set_xlim(0, 1)
# Linear y on the noise scale — the trajectory cloud at τ/T=0 is exactly the
# noise prior shown in the left panel, viewed at 90°.
ax_t.set_ylim(-noise_lim_af3, noise_lim_af3)
gamma_cutoff_step = float(af3_step[(sigma_grid_desc <= 1.0).nonzero().min()])
ax_t.axvline(
    gamma_cutoff_step,
    color="black",
    ls=":",
    lw=0.9,
    alpha=0.7,
    label=rf"$\hat\sigma=\gamma_{{\min}}$ at $\tau/T\!\approx\!{gamma_cutoff_step:.2f}$",
)
ax_t.legend(loc="upper right", fontsize=9)

# Right: ground-truth GMM density + empirical histogram of the AF3 final
# samples (τ/T = 1). Histogram is horizontal so it shares the x-axis (= x).
ax_d.hist(
    traj_af3[-1, :, 0, 0].numpy(),
    bins=80,
    range=(-data_lim_af3, data_lim_af3),
    density=True,
    orientation="horizontal",
    color="darkorange",
    alpha=0.45,
    label="AF3 samples",
)
ax_d.plot(p_data_af3.numpy(), xs_af3[:, 0, 0].numpy(), color="firebrick", lw=1.5, label="data $p(x,0)$")
ax_d.fill_betweenx(xs_af3[:, 0, 0].numpy(), 0, p_data_af3.numpy(), color="firebrick", alpha=0.15)
ax_d.set_title(r"$p(x, t \approx 0)$  (data vs samples)")
ax_d.set_xticks([])
ax_d.set_ylim(-data_lim_af3, data_lim_af3)
ax_d.yaxis.tick_right()
ax_d.legend(loc="lower right", fontsize=7)

# Middle: 5 segmented trajectory windows with per-segment y-range, mirroring
# the forward-process bottom row of section 1. Each window auto-scales y to its
# own min/max, exposing the smooth contraction-then-fan-out shape that the
# fixed-axis top panel can't show.
seg_edges_af3 = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
n_seg_af3 = len(seg_edges_af3) - 1
mid4 = gridspec.GridSpecFromSubplotSpec(1, n_seg_af3, subplot_spec=outer4[1], wspace=0.35)
af3_step_np = af3_step.numpy()
traj_xy = traj_af3[:, :, 0, 0]  # [T+1, N]
mode_levels = [float(m) for m in mu_mix.flatten().tolist()]

for s in range(n_seg_af3):
    ax_seg = fig4.add_subplot(mid4[s])
    lo, hi = seg_edges_af3[s], seg_edges_af3[s + 1]
    mask = (af3_step >= lo) & (af3_step <= hi)
    ts_seg = af3_step_np[mask.numpy()]
    seg = traj_xy[mask, :]  # [T_seg, N]

    ax_seg.plot(ts_seg, seg[:, idx_p].numpy(), color="darkorange", alpha=0.04, lw=0.5)

    y_min, y_max = float(seg.min()), float(seg.max())
    pad = 0.05 * (y_max - y_min) if y_max > y_min else 1.0
    ax_seg.set_ylim(y_min - pad, y_max + pad)
    # Draw mode lines only when they fall inside the auto-scaled segment range.
    for m in mode_levels:
        if y_min - pad <= m <= y_max + pad:
            ax_seg.axhline(m, color="firebrick", ls="--", lw=0.7, alpha=0.5)
    # Draw the γ-cutoff line in the segment that contains it.
    if lo <= gamma_cutoff_step <= hi:
        ax_seg.axvline(gamma_cutoff_step, color="black", ls=":", lw=0.9, alpha=0.7)
    ax_seg.set_xlim(lo, hi)
    ax_seg.set_title(rf"$\tau/T \in [{lo:.1f}, {hi:.1f}]$")
    ax_seg.set_xlabel(r"$\tau/T$")
    ax_seg.grid(True, alpha=0.2)
    if s == 0:
        ax_seg.set_ylabel("x")

# Bottom: σ-schedule (eq. 7) | final-marginal histogram vs analytical p_data.
bot4 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer4[2], width_ratios=[1, 1.4], wspace=0.25)
ax_sch = fig4.add_subplot(bot4[0])
ax_hist = fig4.add_subplot(bot4[1])

ax_sch.semilogy(af3_step.numpy(), sigma_grid_desc.numpy(), color="black", lw=1.4)
ax_sch.axhline(1.0, color="firebrick", ls="--", lw=0.8, alpha=0.7, label=r"$\gamma_{\min}=1$ (churn cutoff)")
ax_sch.set_xlabel(r"AF3 step  $\tau / T$")
ax_sch.set_ylabel(r"$\hat\sigma_\tau$  (log)")
ax_sch.set_title(r"AF3 $\sigma$-schedule (eq. 7, $\rho=7$)")
ax_sch.grid(True, alpha=0.2, which="both")
ax_sch.legend(fontsize=8)

# Match section 2's bar+line density comparison.
xs_np_af3 = xs_af3.squeeze().numpy()
dx = float(xs_af3[1, 0, 0] - xs_af3[0, 0, 0])
edges = torch.cat([xs_af3[0:1, 0, 0] - dx / 2, xs_af3[:, 0, 0] + dx / 2]).numpy()
h_af3, _ = torch.histogram(traj_af3[-1, :, 0, 0], bins=torch.tensor(edges), density=True)
ax_hist.plot(xs_np_af3, p_data_af3.numpy(), color="firebrick", lw=1.5, label="data $p(x,0)$")
ax_hist.bar(xs_np_af3, h_af3.numpy(), width=dx, alpha=0.45, color="darkorange", label="AF3 samples", align="center")
ax_hist.set_xlim(-8, 8)
ax_hist.set_xlabel("x")
ax_hist.set_ylabel("density")
ax_hist.set_title(f"final marginal  ($T={T_AF3}$, $\\eta={1.5}$, $\\lambda={1.003}$, $\\gamma_0={0.8}$)")
ax_hist.legend(fontsize=8)

plt_show()

# %% [markdown]
# # AF3 schedule scalars — sanity inspection
#
# Plot the AF3 inference σ-schedule (eq. 7) at the paper's scales (s_max=160,
# s_min=4·10⁻⁴, p=7) and the per-step scalars used inside Algorithm 18 — γ_τ,
# t̂_τ, ε_τ — to verify the values are doing what we expect.

# %%
import numpy

t = numpy.linspace(0, 1, 200)

s_max = 160
s_min = 4 * 10**-4
p = 7
c = (s_max ** (1 / p) + t * (s_min ** (1 / p) - s_max ** (1 / p))) ** p
fig = plt.figure(figsize=(8, 4))
plt.plot(t, c)
plt.axhline(y=1, color="r", linestyle="--")
plt.xlabel("t")
plt.ylabel("c")
plt.title("c vs t")
plt_show()

y0 = 0.8
ymin = 1.0
lam = 1.003
eta = 1.5

y = lambda c: y0 if c > ymin else 0.0

gammas = []
t_hats = []
epsilons = []

for i, c_ in enumerate(c):
    if i == 0:
        continue
    gamma = ymin if c_ > ymin else 0.0
    t_hat = c[i - 1] * (gamma + 1)
    epsilon = lam * (t_hat**2 - c[i - 1] ** 2) ** 0.5

    gammas.append(gamma)
    t_hats.append(t_hat)
    epsilons.append(epsilon)


fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs = axs.flatten()
axs[0].plot(t, c, label="c")
axs[0].plot(t[1:], gammas, label="gamma")
axs[0].plot(t[1:], t_hats, label="t_hat")
axs[0].plot(t[1:], epsilons, label="epsilon")
axs[0].legend()
axs[1].plot(t, c, label="c")
axs[1].plot(t[1:], gammas, label="gamma")
axs[1].plot(t[1:], t_hats, label="t_hat")
axs[1].plot(t[1:], epsilons, label="epsilon")
axs[1].legend()
axs[1].set_ylim(0, 2)

plt_show()
