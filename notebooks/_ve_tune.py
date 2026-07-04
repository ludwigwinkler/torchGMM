"""Throwaway harness for tuning ve_steering FKC params.
Reports W2 for both wide (target_s=2.0) and tight (target_s=0.5) potentials.
"""

import os
import sys
import time

import numpy as np
import torch

from torchGMM import GMM, VESchedule
from torchGMM.sampling import reverse_sampling, steered_reverse_sampling

torch.manual_seed(0)
torch.set_default_device("cpu")

# ---------------- model ----------------
SIGMA_MAX = float(os.environ.get("SIGMA_MAX", 8.0))
ve_sched = VESchedule(sigma_min=0.01, sigma_max=SIGMA_MAX)
mu_mix = torch.tensor([[[-3.0], [-0.5], [2.0], [4.0]]])
sigma_mix = torch.tensor([[[0.3], [0.25], [0.4], [0.2]]])
weight_mix = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
gmm_mix = GMM(mu=mu_mix, sigma=sigma_mix, weight=weight_mix, schedule=ve_sched)

target_c = 3.0


# ---------------- β schedule (configurable via env) ----------------
BETA_KIND = os.environ.get("BETA_KIND", "exp_neg2t2")  # "linear" or "exp_neg2t2"


def beta_fn(t):
    if BETA_KIND == "linear":
        return 1.0 - t
    if BETA_KIND == "exp_neg2t2":
        return torch.exp(-2.0 * t**2)
    raise ValueError(BETA_KIND)


def neg_dbeta_dt(t):
    """-∂_t β.  For β=1-t: returns 1.  For β=exp(-2t²): returns 4t·β."""
    if BETA_KIND == "linear":
        return torch.ones_like(torch.as_tensor(t))
    if BETA_KIND == "exp_neg2t2":
        return 4.0 * t * beta_fn(t)
    raise ValueError(BETA_KIND)


# ---------------- denoiser w/ autograd ----------------
def denoise(x_t, t, score_fn, sigma_fn):
    x_leaf = x_t.detach().requires_grad_(True)
    t_leaf = torch.as_tensor(t, dtype=x_t.dtype, device=x_t.device).clone().detach().requires_grad_(True)
    with torch.enable_grad():
        sigma_t = sigma_fn(t_leaf)
        sc = score_fn(x_leaf, t_leaf)
        x0 = x_leaf + sigma_t**2 * sc
    return x0, x_leaf, t_leaf


def wasserstein2_1d(samples, xs_grid, p):
    s = np.sort(samples.detach().cpu().numpy().ravel())
    xs_arr = xs_grid.detach().cpu().numpy().ravel()
    p_arr = p.detach().cpu().numpy().ravel()
    dx = xs_arr[1] - xs_arr[0]
    cdf = np.cumsum(p_arr) * dx
    cdf = cdf / cdf[-1]
    levels = (np.arange(s.size) + 0.5) / s.size
    q = np.interp(levels, cdf, xs_arr)
    return float(np.sqrt(np.mean((s - q) ** 2)))


# ---------------- run one configuration ----------------
def run(target_s: float, T_MAX=0.99, EPS_R=1e-3, N_R=5_000, T_R=2_000, ESS=0.9, seed=0):
    torch.manual_seed(seed)

    def r(x):
        return -0.5 * (x - target_c) ** 2 / target_s**2

    def reverse_drift(x_, t_):
        g = ve_sched.diffusion_coeff(t_)
        return -(g**2) * gmm_mix.score(x_, t_)

    def reward_grads(x_, t_):
        """Compute r(x̂_0) and ∇r evaluated AT x̂_0 (no chain-rule through the
        score Jacobian — the Tweedie Jacobian ∂x̂_0/∂x_t collapses to ~0 at
        high σ_t for narrow data, killing the steering signal). This is the
        standard DPS / classifier-guidance form."""
        sigma_t = ve_sched.get_sigma_t(t_)
        sc = gmm_mix.score(x_, t_).detach()
        x0 = (x_ + sigma_t**2 * sc).detach().requires_grad_(True)
        with torch.enable_grad():
            rv = r(x0)
        (grad_at_x0,) = torch.autograd.grad(rv.sum(), x0)
        return rv.detach(), grad_at_x0.detach(), sc

    W_GUIDE = float(os.environ.get("W_GUIDE", 1.0))
    MODE = os.environ.get("MODE", "fkc")  # "fkc" | "bootstrap" | "dps"

    def guided_drift(x_, t_):
        g = ve_sched.diffusion_coeff(t_)
        beta = beta_fn(t_)
        _, grad_x, sc = reward_grads(x_, t_)
        if MODE == "bootstrap":
            return -(g**2) * sc
        if MODE == "dps":
            # classifier-guidance form: drift twist independent of σ², scaled by w
            return -(g**2) * sc - W_GUIDE * grad_x
        # FKC literal form, optionally amplified by W_GUIDE
        return -(g**2) * sc - W_GUIDE * beta * (g**2 / 2) * grad_x

    def weight_update(x_, t_, dt):
        g = ve_sched.diffusion_coeff(t_)
        beta = beta_fn(t_)
        nbeta = neg_dbeta_dt(t_)
        rv, grad_x, sc = reward_grads(x_, t_)
        if MODE == "bootstrap":
            # cumulative reward at Tweedie estimate
            return rv.squeeze(-1).squeeze(-1) * dt.abs()
        if MODE == "dps":
            # weight increment matching the DPS drift twist
            return (rv + W_GUIDE * grad_x * sc).squeeze(-1).squeeze(-1) * dt.abs()
        return (nbeta * rv + W_GUIDE * beta * grad_x * (g**2 / 2) * sc).squeeze(-1).squeeze(-1) * dt.abs()

    t_rev = torch.linspace(T_MAX, EPS_R, T_R)
    x_init = gmm_mix.sample(shape=N_R, t=T_MAX)

    t0 = time.time()
    traj_unguided = reverse_sampling(reverse_drift, ve_sched.diffusion_coeff, x_init.clone(), t_rev).detach()
    traj_steered, ess_hist = steered_reverse_sampling(
        drift=guided_drift,
        diffusion=ve_sched.diffusion_coeff,
        weight_update=weight_update,
        x=x_init.clone(),
        t=t_rev,
        ess_threshold=ESS,
    )
    elapsed = time.time() - t0
    traj_steered = traj_steered.detach()

    xs = torch.linspace(-6, 6, 400).reshape(-1, 1, 1)
    log_p = gmm_mix.log_prob(xs, t=torch.tensor(EPS_R)).squeeze().detach()
    p_data = log_p.exp()
    p_data = p_data / torch.trapezoid(p_data, xs.squeeze())
    log_p_tilt = log_p + r(xs).squeeze()
    p_tilt = (log_p_tilt - log_p_tilt.max()).exp()
    p_tilt = p_tilt / torch.trapezoid(p_tilt, xs.squeeze())

    W2_steered = wasserstein2_1d(traj_steered[-1, :, 0, 0], xs.squeeze(), p_tilt)
    W2_unguided = wasserstein2_1d(traj_unguided[-1, :, 0, 0], xs.squeeze(), p_data)

    n_resamples = sum(1 for e in ess_hist if e < ESS)
    min_ess = float(min(ess_hist))
    return W2_steered, W2_unguided, n_resamples, min_ess, elapsed


if __name__ == "__main__":
    target_s_list = [2.0, 0.5]
    print(f"σ_max={SIGMA_MAX}  β={BETA_KIND}")
    print(f"{'tgt_s':>6} {'W2_st':>8} {'W2_un':>8} {'#resmp':>6} {'min_ess':>8} {'elapsed_s':>10}")
    for ts in target_s_list:
        W2_st, W2_un, n_res, min_ess, t = run(target_s=ts)
        flag = "✓" if W2_st < 0.01 else "✗"
        print(f"{ts:>6} {W2_st:>8.4f} {W2_un:>8.4f} {n_res:>6} {min_ess:>8.3f} {t:>10.1f}  {flag}")
