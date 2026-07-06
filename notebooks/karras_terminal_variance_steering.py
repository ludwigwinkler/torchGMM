# %%
"""FKC-steered reverse sampling under the Karras schedule, swept over the
terminal noise level σ_max and the number of ODE-denoiser substeps
N_DENOISE_STEPS used to build x̂_0 for the reward gradient.

Reuses the experiment from ve_steering.py (B=1, K=4 mixture with a Gaussian
reward placed on a single target mode) but replaces VESchedule with
KarrasSchedule and produces one 2×2 figure per (σ_max, N_denoise_steps)
combination so the effect of the terminal variance and denoiser depth on
unguided / steered reverse sampling can be compared.

σ_min = 4e-4, ρ = 7, σ_data = 1 are fixed (Karras / AF3 defaults). Unguided
reverse sampling doesn't depend on the denoiser, so it is only run once per
σ_max and reused across the N_DENOISE_STEPS_SWEEP sweep.
"""

from pathlib import Path
import sys

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from _utils import plt_show

OUT_DIR = Path(__file__).resolve().parent

from torchGMM import GMM, KarrasSchedule
from torchGMM.sampling import reverse_sampling, steered_reverse_sampling

plt.style.use("default")
plt.rcdefaults()

torch.manual_seed(0)
device = torch.device("cpu")
torch.set_default_device(device)

# --- fixed mixture (B=1, K=4) ---
mu_mix = torch.tensor([[[-1.0], [-0.25], [0.25], [1.0]]])  # [B=1, K=4, D=1]
sigma_mix = torch.tensor([[[0.3], [0.2], [0.3], [0.1]]])  # [B=1, K=4, D=1]
weight_mix = torch.tensor([[0.25, 0.25, 0.25, 0.25]])  # [B=1, K=4]

# --- fixed reward: Gaussian potential on mode 3 (x=2) ---
target_c = 2
target_s = 1


def r(x_0):
    return -0.5 * (x_0 - target_c) ** 2 / target_s**2


# --- fixed reverse-sampling hyperparameters ---
T_NOISE = 0.99  # reverse-sampling start time (near pure noise)
T_DATA = 1e-3  # reverse-sampling end time (near data)
N_PARTICLES = 1_000  # SMC particle count
N_STEPS = 200  # number of reverse-time integrator steps
ESS_THRESHOLD = 0.8  # systematic-resample when ESS/N drops below this
N_PLOT = 600
SIGMA_MAX_SWEEP = [160.0]  # 160 is the AF3 default


# %%
# --- visualize the Karras marginal noise schedule and induced variance rate ---
t_schedule = torch.linspace(0.0, 1.0, 400)
colors_schedule = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, len(SIGMA_MAX_SWEEP)))

fig_karras, (ax_sigma, ax_dsigma, ax_prod) = plt.subplots(1, 3, figsize=(15, 4))
for color, sigma_max in zip(colors_schedule, SIGMA_MAX_SWEEP):
    schedule_plot = KarrasSchedule(sigma_min=4e-4, sigma_max=sigma_max, rho=7.0, sigma_data=1.0)
    sigma_t = schedule_plot.get_sigma_t(t_schedule)
    dsigma_dt = schedule_plot.get_dsigma_dt(t_schedule)
    sigma_dsigma_dt = sigma_t * dsigma_dt
    label = rf"$\sigma_{{\max}}={sigma_max:g}$"

    ax_sigma.plot(t_schedule.cpu(), sigma_t.cpu(), color=color, lw=2, label=label)
    ax_dsigma.plot(t_schedule.cpu(), dsigma_dt.cpu(), color=color, lw=2, label=label)
    ax_prod.plot(t_schedule.cpu(), sigma_dsigma_dt.cpu(), color=color, lw=2, label=label)

ax_sigma.set_xlabel("t")
ax_sigma.set_ylabel(r"$\bar\sigma(t)$")
ax_sigma.set_title(r"Karras marginal noise $\bar\sigma(t)$")
# ax_sigma.set_yscale("log")
ax_sigma.grid(True, which="both", alpha=0.25)
ax_sigma.legend(fontsize=8)

ax_dsigma.set_xlabel("t")
ax_dsigma.set_ylabel(r"$d\bar\sigma(t)/dt$")
ax_dsigma.set_title(r"Time derivative $d\bar\sigma/dt$")
# ax_dsigma.set_yscale("log")
ax_dsigma.grid(True, which="both", alpha=0.25)

ax_prod.set_xlabel("t")
ax_prod.set_ylabel(r"$\bar\sigma(t)\,d\bar\sigma(t)/dt$")
ax_prod.set_title(r"Half variance rate: $g(t)^2/2$")
# ax_prod.set_yscale("log")
ax_prod.grid(True, which="both", alpha=0.25)

fig_karras.tight_layout()
fig_karras.savefig(OUT_DIR / "karras_variance_schedule.png", dpi=120, bbox_inches="tight")
plt_show()

sys.exit()

# --- tilt schedule beta_t (0 at noise, 1 at data) ---
# Cosine ramp: smooth at both endpoints, full tilt only near data time.
def beta_fn(t):
    return torch.cos(0.5 * torch.pi * t) ** 2


def dbeta_dt(t):
    return -0.5 * torch.pi * torch.sin(torch.pi * t)


# # Qudratic ramp: smooth at noise, linear near data time.
def beta_fn(t):
    return (1 - t) ** 2


def dbeta_dt(t):
    return -2 * (1 - t)

# # Linear ramp: smooth at noise, linear near data time.
# def beta_fn(t):
#     return 1 - t

# def dbeta_dt(t):
#     return - torch.ones_like(t)

# %%
# --- visualize the active tilt schedule β(t) and dβ/dt ---
t_plot = torch.linspace(0.0, 1.0, 200)
fig_beta, (ax_beta, ax_dbeta) = plt.subplots(1, 2, figsize=(10, 4))
ax_beta.plot(t_plot.numpy(), beta_fn(t_plot).numpy(), color="darkorange", lw=2)
ax_beta.set_xlabel("t")
ax_beta.set_ylabel(r"$\beta(t)$")
ax_beta.set_title(r"Tilt schedule $\beta(t)$")
ax_beta.grid(True, alpha=0.25)

ax_dbeta.plot(t_plot.numpy(), dbeta_dt(t_plot).numpy(), color="steelblue", lw=2)
ax_dbeta.set_xlabel("t")
ax_dbeta.set_ylabel(r"$\dot\beta(t)$")
ax_dbeta.set_title(r"$d\beta/dt$")
ax_dbeta.grid(True, alpha=0.25)

fig_beta.tight_layout()
fig_beta.savefig(OUT_DIR / "karras_beta_schedule.png", dpi=120, bbox_inches="tight")
plt_show()


N_DENOISE_STEPS_SWEEP = [1, 25, 50, 100]  # ODE substeps from current t down to T_DATA, swept


def denoise(x_t, t, score_fn, sigma_fn, n_denoise_steps):
    """ODE denoiser: integrate the probability-flow ODE in Karras's denoiser
    parameterization
        dx/dσ = (x - D(x; σ)) / σ,    D(x; σ) = x + σ² · score(x, σ)
    from σ(t) down to σ(T_DATA) in n_denoise_steps Euler substeps. Each substep
    is a Tweedie blend  x_next = (σ_next/σ_curr)·x + (1 - σ_next/σ_curr)·D, so
    the integrator is stable at any σ-step (unlike plain Euler in t, which
    overshoots when d(log σ)/dt · dt is O(1)). Substep grid is linear in t
    (equivalent to σ^{1/ρ}-uniform for the Karras schedule). n_denoise_steps=1
    reduces to the single-step Tweedie estimate x̂_0 = D(x_t; σ(t)). Returns
    x̂_0 plus leaf tensors so autograd can backprop ∂r(x̂_0)/∂x_t and
    ∂r(x̂_0)/∂t through the unrolled solver."""
    x_t_leaf = x_t.detach().requires_grad_(True)
    t_leaf = (
        torch.as_tensor(t, dtype=x_t.dtype, device=x_t.device)
        .clone()
        .detach()
        .requires_grad_(True)
    )
    with torch.enable_grad():
        t_min = torch.as_tensor(T_DATA, dtype=x_t.dtype, device=x_t.device)
        x = x_t_leaf
        t_prev = t_leaf
        sigma_prev = sigma_fn(t_prev)
        for step_k in range(1, n_denoise_steps + 1):
            t_next = t_leaf + (t_min - t_leaf) * (step_k / n_denoise_steps)
            sigma_next = sigma_fn(t_next)
            D = x + sigma_prev**2 * score_fn(x, t_prev)
            ratio = sigma_next / sigma_prev
            x = ratio * x + (1.0 - ratio) * D
            t_prev = t_next
            sigma_prev = sigma_next
        x0 = x
    return x0, x_t_leaf, t_leaf


def wasserstein2_1d(samples, xs_grid, p):
    """W₂ between empirical samples and an analytic 1D density on a uniform
    grid via quantile matching: W₂² = ∫ (F⁻¹_emp − F⁻¹_p)² du."""
    s = np.sort(samples.detach().cpu().numpy().ravel())
    xs_arr = xs_grid.detach().cpu().numpy().ravel()
    p_arr = p.detach().cpu().numpy().ravel()
    dx_grid = xs_arr[1] - xs_arr[0]
    cdf = np.cumsum(p_arr) * dx_grid
    cdf = cdf / cdf[-1]
    levels = (np.arange(s.size) + 0.5) / s.size
    q = np.interp(levels, cdf, xs_arr)
    return float(np.sqrt(np.mean((s - q) ** 2)))


def run_unguided(sigma_max):
    """Build a Karras-scheduled K=4 GMM with the given σ_max and run unguided
    reverse sampling. Independent of the denoiser used for FKC steering, so
    this is computed once per σ_max and reused across N_DENOISE_STEPS_SWEEP."""
    schedule = KarrasSchedule(
        sigma_min=4e-4, sigma_max=sigma_max, rho=7.0, sigma_data=1.0
    )
    gmm_mix = GMM(mu=mu_mix, sigma=sigma_mix, weight=weight_mix, schedule=schedule)

    def reverse_drift(x_, t_):
        g = schedule.diffusion_coeff(t_)
        return -(g**2) * gmm_mix.score(x_, t_)

    torch.manual_seed(0)
    t_rev = torch.linspace(T_NOISE, T_DATA, N_STEPS)
    x_init = gmm_mix.sample(shape=N_PARTICLES, t=T_NOISE)

    traj_unguided = reverse_sampling(
        reverse_drift, schedule.diffusion_coeff, x_init.clone(), t_rev
    ).detach()

    # analytic reference densities at data time
    xs = torch.linspace(-6, 6, 200).reshape(-1, 1, 1)
    log_p = gmm_mix.log_prob(xs, t=torch.tensor(T_DATA)).squeeze().detach()
    p_data = log_p.exp()
    p_data = p_data / torch.trapezoid(p_data, xs.squeeze())
    log_p_tilt = log_p + r(xs).squeeze()
    p_tilt = (log_p_tilt - log_p_tilt.max()).exp()
    p_tilt = p_tilt / torch.trapezoid(p_tilt, xs.squeeze())

    W2_unguided = wasserstein2_1d(traj_unguided[-1, :, 0, 0], xs.squeeze(), p_data)

    return {
        "schedule": schedule,
        "gmm_mix": gmm_mix,
        "t_rev": t_rev,
        "x_init": x_init,
        "traj_unguided": traj_unguided,
        "xs": xs,
        "p_data": p_data,
        "p_tilt": p_tilt,
        "W2_unguided": W2_unguided,
    }


def run_steered(ctx, n_denoise_steps):
    """FKC-steered reverse sampling reusing the schedule/GMM/init samples
    from `run_unguided(sigma_max)`, with the ODE denoiser unrolled for
    n_denoise_steps substeps."""
    schedule = ctx["schedule"]
    gmm_mix = ctx["gmm_mix"]
    t_rev = ctx["t_rev"]
    x_init = ctx["x_init"]

    def _reward_and_grads(x_, t_):
        x0, x_leaf, t_leaf = denoise(
            x_, t_, gmm_mix.score, schedule.get_sigma_t, n_denoise_steps
        )
        rv = r(x0)
        grad_x, grad_t = torch.autograd.grad(rv.sum(), (x_leaf, t_leaf))
        # Score at the current (x_t, t) — needed for the alignment term. With an
        # ODE denoiser the Tweedie identity no longer recovers it from x̂_0.
        sc = gmm_mix.score(x_, t_).detach()
        return rv.detach(), grad_x.detach(), grad_t.detach(), sc

    def guided_drift(x_, t_):
        g = schedule.diffusion_coeff(t_)
        beta = beta_fn(t_)
        _, grad_x, _, sc = _reward_and_grads(x_, t_)
        return -(g**2) * sc - beta * (g**2 / 2) * grad_x

    def weight_update(x_, t_, dt):
        g = schedule.diffusion_coeff(t_)
        beta = beta_fn(t_)
        dbeta = dbeta_dt(t_)
        rv, grad_x, grad_t, sc = _reward_and_grads(x_, t_)
        # Eq. (276'): dw = [dot_beta r + beta d_t r + <beta grad r, sigma^2/2 s - f>] dt
        # Reverse sampler passes signed dt < 0; using |dt| flips the sign of the dt
        # factor, so the integrand below is the negation of the eq-(276') one
        # (same convention as ve_steering.py). grad_t is d_t r at fixed x_t through
        # hat_x0(x_t, t); the convective piece is already carried by the guided
        # drift and the alignment inner product.
        integrand = -dbeta * rv - beta * grad_t + beta * grad_x * (g**2 / 2) * sc
        return integrand.squeeze(-1).squeeze(-1) * dt.abs()

    # Fixed seed so the SDE noise is identical across n_denoise_steps — the
    # only knob that varies within a σ_max group.
    torch.manual_seed(0)
    traj_steered, ess_hist = steered_reverse_sampling(
        drift=guided_drift,
        diffusion=schedule.diffusion_coeff,
        weight_update=weight_update,
        x=x_init.clone(),
        t=t_rev,
        ess_threshold=ESS_THRESHOLD,
    )
    traj_steered = traj_steered.detach()

    W2_steered = wasserstein2_1d(
        traj_steered[-1, :, 0, 0], ctx["xs"].squeeze(), ctx["p_tilt"]
    )
    n_resamples = sum(1 for e in ess_hist if e < ESS_THRESHOLD)

    return {
        "traj_steered": traj_steered,
        "ess_hist": ess_hist,
        "W2_steered": W2_steered,
        "n_resamples": n_resamples,
    }


def plot_run(ctx, steered, sigma_max, n_denoise_steps):
    traj_unguided = ctx["traj_unguided"]
    traj_steered = steered["traj_steered"]
    ess_hist = steered["ess_hist"]
    t_rev = ctx["t_rev"]
    xs = ctx["xs"]
    p_data = ctx["p_data"]
    p_tilt = ctx["p_tilt"]

    W2_steered = steered["W2_steered"]
    W2_unguided = ctx["W2_unguided"]
    n_resamples = steered["n_resamples"]
    print(
        f"[σ_max={sigma_max:>5g}, N_denoise={n_denoise_steps:>2d}]  "
        f"W2(steered ‖ p_tilt) = {W2_steered:.4f}   "
        f"W2(unguided ‖ p_data) = {W2_unguided:.4f}   "
        f"resamples = {n_resamples}"
    )

    idx_plot = torch.randperm(N_PARTICLES)[:N_PLOT]
    fig = plt.figure(figsize=(16, 11))
    gs2 = gridspec.GridSpec(2, 2, height_ratios=[3, 1.3], hspace=0.3, wspace=0.18)
    ax_un = fig.add_subplot(gs2[0, 0])
    ax_st = fig.add_subplot(gs2[0, 1], sharey=ax_un)
    ax_dens = fig.add_subplot(gs2[1, 0])
    ax_ess = fig.add_subplot(gs2[1, 1])

    fig.suptitle(
        rf"Karras schedule, $\sigma_{{\max}}={sigma_max:g}$, $N_{{\mathrm{{denoise}}}}={n_denoise_steps}$"
        rf"  ($\rho=7$, $\sigma_{{\min}}=4\!\times\!10^{{-4}}$)",
        fontsize=13,
    )

    t_rev_np = t_rev.cpu().numpy()
    ax_un.plot(
        t_rev_np,
        traj_unguided[:, idx_plot, 0, 0].cpu(),
        color="steelblue",
        alpha=0.08,
        lw=0.5,
    )
    ax_un.axhline(
        target_c,
        color="firebrick",
        ls="--",
        lw=1.2,
        alpha=0.8,
        label=f"target μ={target_c}",
    )
    ax_un.set_title(r"$\leftarrow$ Unguided reverse sampling $\leftarrow$")
    ax_un.set_xlabel("t")
    ax_un.set_ylabel("x")
    y_lim_traj = float(traj_unguided[:, idx_plot, 0, 0].abs().max()) * 1.05
    ax_un.set_ylim(-y_lim_traj, y_lim_traj)
    ax_un.legend(loc="upper left", fontsize=10)

    ax_st.plot(
        t_rev_np,
        traj_steered[:, idx_plot, 0, 0].cpu(),
        color="darkorange",
        alpha=0.08,
        lw=0.5,
    )
    ax_st.axhline(target_c, color="firebrick", ls="--", lw=1.2, alpha=0.8)
    ax_st.set_title(r"$\leftarrow$ FKC-steered reverse sampling $\leftarrow$")
    ax_st.set_xlabel("t")

    # final-time histograms vs analytic densities
    xs_np = xs.squeeze().cpu().numpy()
    bin_w = float(xs[1, 0, 0] - xs[0, 0, 0])
    xs_centers = xs[:, 0, 0].cpu()
    edges = (
        torch.cat([xs_centers[0:1] - bin_w / 2, xs_centers + bin_w / 2]).cpu().numpy()
    )
    xs_centers_np = xs_centers.numpy()

    edges_cpu = torch.tensor(edges, device="cpu")
    h_un, _ = torch.histogram(
        traj_unguided[-1, :, 0, 0].cpu(), bins=edges_cpu, density=True
    )
    h_st, _ = torch.histogram(
        traj_steered[-1, :, 0, 0].cpu(), bins=edges_cpu, density=True
    )

    ax_dens.plot(xs_np, p_data.cpu(), color="steelblue", lw=1.5, label="data $p$")
    ax_dens.plot(
        xs_np, p_tilt.cpu(), color="firebrick", lw=1.5, label=r"tilted $p \cdot e^{r}$"
    )
    ax_dens.bar(
        xs_centers_np,
        h_un.numpy(),
        width=bin_w,
        alpha=0.35,
        color="steelblue",
        label="unguided hist",
        align="center",
    )
    ax_dens.bar(
        xs_centers_np,
        h_st.numpy(),
        width=bin_w,
        alpha=0.45,
        color="darkorange",
        label="steered hist",
        align="center",
    )
    ax_dens.axvline(target_c, color="firebrick", ls="--", lw=1, alpha=0.7)
    ax_dens.set_xlim(-3, 3)
    ax_dens.set_xlabel("x")
    ax_dens.set_ylabel("density")
    ax_dens.set_title(f"final-time marginals  (potential on {target_c})")
    ax_dens.legend(fontsize=8)

    ess_t = t_rev_np[: len(ess_hist)]
    ax_ess.plot(ess_t, ess_hist, color="darkorange", lw=1.0, label="ESS / N")
    ax_ess.axhline(
        ESS_THRESHOLD, color="red", ls="--", lw=1, label=f"threshold={ESS_THRESHOLD}"
    )
    ax_ess.set_ylim(0, 1)
    ax_ess.set_xlim(0, 1)
    ax_ess.set_xlabel("t")
    ax_ess.set_ylabel("ESS / N")
    n_resamples = sum(1 for e in ess_hist if e < ESS_THRESHOLD)
    ax_ess.set_title(f"ESS history  ({n_resamples} resamples)")
    ax_ess.legend(fontsize=8)

    fig.savefig(
        OUT_DIR / f"karras_sweep_sigma_max_{sigma_max:g}_ndenoise_{n_denoise_steps}.png",
        dpi=120,
        bbox_inches="tight",
    )


# %%
# Unguided reverse sampling does not depend on the denoiser, so run it once
# per σ_max and reuse it across the N_DENOISE_STEPS_SWEEP sweep below.
unguided_ctx = {sigma_max: run_unguided(sigma_max) for sigma_max in SIGMA_MAX_SWEEP}

results = []  # (n_denoise_steps, sigma_max, W2_steered, W2_unguided, n_resamples)
for sigma_max in SIGMA_MAX_SWEEP:
    ctx = unguided_ctx[sigma_max]
    for n_denoise_steps in N_DENOISE_STEPS_SWEEP:
        steered = run_steered(ctx, n_denoise_steps)
        plot_run(ctx, steered, sigma_max, n_denoise_steps)
        results.append(
            (
                n_denoise_steps,
                sigma_max,
                steered["W2_steered"],
                ctx["W2_unguided"],
                steered["n_resamples"],
            )
        )

# --- summary: W2 and resample count vs σ_max, colored by N_denoise_steps ---
sm = np.array(SIGMA_MAX_SWEEP)
w2_un = np.array([unguided_ctx[s]["W2_unguided"] for s in SIGMA_MAX_SWEEP])
colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, len(N_DENOISE_STEPS_SWEEP)))

fig_sum, ax_w2 = plt.subplots(figsize=(9, 5))
ax_rs = ax_w2.twinx()

for color, n_denoise_steps in zip(colors, N_DENOISE_STEPS_SWEEP):
    w2_st = np.array([r[2] for r in results if r[0] == n_denoise_steps])
    n_rs = np.array([r[4] for r in results if r[0] == n_denoise_steps])
    ax_w2.plot(
        sm,
        w2_st,
        "o-",
        color=color,
        lw=1.8,
        label=rf"$W_2$(steered $\Vert$ $p\cdot e^{{r}}$), $N_{{\mathrm{{denoise}}}}={n_denoise_steps}$",
    )
    ax_rs.plot(sm, n_rs, "^--", color=color, lw=1.2, alpha=0.5)

ax_w2.plot(
    sm, w2_un, "s-", color="black", lw=1.8, label=r"$W_2$(unguided $\Vert$ $p$)"
)
ax_w2.set_xscale("log")
ax_w2.set_xlabel(r"terminal noise $\sigma_{\max}$ (log scale)")
ax_w2.set_ylabel(r"$W_2$")
ax_w2.grid(True, which="both", alpha=0.25)

ax_rs.set_ylabel("# resamples (steered, dashed)")
ax_rs.set_ylim(bottom=0)

lines, labels = ax_w2.get_legend_handles_labels()
ax_w2.legend(lines, labels, loc="upper left", fontsize=8)
ax_w2.set_title(
    r"FKC reverse-sampling quality vs Karras $\sigma_{\max}$ and denoiser substeps ($\rho=7$)"
)

fig_sum.savefig(OUT_DIR / "karras_sweep_summary.png", dpi=120, bbox_inches="tight")

plt_show()
