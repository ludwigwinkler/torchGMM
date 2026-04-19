import einops
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from _utils import plt_show
from matplotlib.cm import get_cmap

# --- plotting ---
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from torch.distributions import MultivariateNormal
from tqdm import tqdm

from torchGMM import GMM, Conditional
from torchGMM.sampling import forward_sampling, reverse_sampling
from torchGMM.schedule import BetaSchedule, LinearSchedule

torch.manual_seed(100)


# %%[markdown]
# # Marginal Distributions
#
# Marginal distributions we're going to use in the diffusion

x0 = torch.linspace(-3, 3, 5).reshape(-1, 1)
mu = torch.tensor([-2, 0, 2]).reshape(1, 3, 1)
sigma = torch.tensor([0.3, 0.1, 0.2]).reshape(1, 3, 1)
weight = torch.tensor([0.33, 0.5, 0.1]).reshape(1, 3)

schedule = LinearSchedule()
gmm = GMM(mu=mu, sigma=sigma, weight=weight, schedule=schedule)
x_grid_b = torch.linspace(-5, 5, 100).reshape(-1, 1, 1)  # [N, B=1, D=1]
plt.plot(x_grid_b.squeeze(), gmm.log_prob(x_grid_b, t=0.0).exp().squeeze())
plt_show()

conditional_gmm = Conditional(x0=x0)
x_grid_cond = torch.linspace(-5, 5, 100).reshape(-1, 1, 1).expand(-1, 5, -1)  # [N, B=5, D=1]

t = [0.01, 0.1, 0.2, 0.25, 0.35, 0.5, 0.75, 1.0]
colors = plt.cm.get_cmap("viridis", len(t))
# for idx, t in enumerate(t):
#     log_prob = conditional_gmm.log_prob(x_grid_cond, t=t)
#     fig, axs = plt.subplots(1, 1, figsize=(10, 10))
#     for i in range(5):
#         axs.plot(x_grid_cond[:, i, 0], log_prob[:, i].exp(), label=f"t={t}", color=colors(idx))
#         axs.legend()
# plt.legend()
# plt_show()

gmm = Conditional(x0=x0)
x_grid_cond = torch.linspace(-5, 5, 100).reshape(-1, 1, 1).expand(-1, 5, -1)
log_prob = gmm.log_prob(x_grid_cond, t=0.0)
assert log_prob.shape == (100, 5)
score = gmm.score(x_grid_cond, t=0.0)
assert score.shape == (100, 5, 1)


# %%[markdown]
# # Forward Flow (ODE)
# Flow matching uses the velocity field v_t(x) for ODE integration: dx/dt = v_t(x).
# The SDE formulation has f(x,t) = -x/(1-t) and g(t) = sqrt(2t/(1-t)) which blow up
# near t=1, making Euler-Maruyama unstable. The ODE is well-behaved everywhere.
eps = 1e-3
gmm = GMM(mu=mu, sigma=sigma, weight=weight, schedule=schedule)
colors = plt.cm.get_cmap("viridis", x0.shape[0])

t = torch.linspace(eps, 1.0 - eps, 200)
# Sample from the marginal at t=eps so particles start with some spread
x_init = gmm.sample(shape=1_000, t=eps)  # [N=50, B=1, D=1]
xt = forward_sampling(gmm.velocity, None, x_init, t)  # [T, N, B=1, D=1]
_ = plt.plot(t, xt[:, :, 0, 0], color="steelblue", alpha=0.05)
plt.title("$\\rightarrow$ Forward Flow (ODE) $\\rightarrow$")
plt.xlabel("t")
plt.ylabel("x")
plt_show()

# %%[markdown]
# # Reverse Flow Matching
schedule = LinearSchedule()
gmm = GMM(mu=mu, sigma=sigma, weight=weight, schedule=schedule)

x = torch.randn(10_000, 1, 1)  # [N, B=1, D=1]
colors = plt.cm.get_cmap("viridis", x0.shape[0])

t = torch.linspace(1.0 - eps, eps, 100)

xt = reverse_sampling(gmm.velocity, None, x, t).detach()
print(xt.shape)
fig, axs = plt.subplots(1, 2, figsize=(20, 10), sharey=True)
axs[1].plot(t, xt[:, :, 0, 0], color="steelblue", alpha=0.1)
axs[1].set_title("$\\leftarrow$ Reverse Flow (ODE) $\\leftarrow$")
axs[0].invert_xaxis()
axs[0].hist(xt[-1, :, 0, 0].flatten(), bins=100, density=True, orientation="horizontal")
x_grid_b = torch.linspace(-5, 5, 100).reshape(-1, 1, 1)
axs[0].plot(gmm.log_prob(x_grid_b, t=eps).exp().squeeze(), x_grid_b.squeeze())
axs[0].set_title("$\\leftarrow$ Target of Reverse Flow $\\leftarrow$")
axs[0].set_xlabel("t")
axs[0].set_ylabel("x")
plt_show()

# %%[markdown]
# # Reverse Flow Matching with SDE sampling
schedule = LinearSchedule()
gmm = GMM(mu=mu, sigma=sigma, weight=weight, schedule=schedule)

x = torch.randn(10_000, 1, 1)  # [N, B=1, D=1]
colors = plt.cm.get_cmap("viridis", x0.shape[0])

t = torch.linspace(1.0 - eps, eps, 200)

diffusion = lambda t: torch.max(torch.scalar_tensor(0.5), t)
score = lambda x, t: gmm.score(x, t)

drift = lambda x, t: gmm.velocity(x, t) - 1 / 2 * diffusion(t) ** 2 * gmm.score(x, t)

xt = reverse_sampling(drift, diffusion, x, t).detach()
print(xt.shape)
fig, axs = plt.subplots(1, 2, figsize=(20, 10), sharey=True)
axs[1].plot(t, xt[:, :, 0, 0], color="steelblue", alpha=0.1)
axs[1].set_title("$\\leftarrow$ Reverse Flow (ODE) $\\leftarrow$")
axs[0].invert_xaxis()
axs[0].hist(xt[-1, :, 0, 0].flatten(), bins=100, density=True, orientation="horizontal")
x_grid_b = torch.linspace(-5, 5, 100).reshape(-1, 1, 1)
axs[0].plot(gmm.log_prob(x_grid_b, t=eps).exp().squeeze(), x_grid_b.squeeze())
axs[0].set_title("$\\leftarrow$ Target of Reverse Flow $\\leftarrow$")
axs[0].set_xlabel("t")
axs[0].set_ylabel("x")
plt_show()

# %%[markdown]
# # Reverse BetaSchedule with SDE sampling

schedule = BetaSchedule()
gmm = GMM(mu=mu, sigma=sigma, weight=weight, schedule=schedule)

x = torch.randn(10_000, 1, 1)  # [N, B=1, D=1]
colors = plt.cm.get_cmap("viridis", x0.shape[0])

t = torch.linspace(1.0 - eps, eps, 200)

diffusion = lambda t: torch.max(torch.scalar_tensor(1.0), t)
score = lambda x, t: gmm.score(x, t)

drift = lambda x, t: gmm.velocity(x, t) - 1 / 2 * diffusion(t) ** 2 * gmm.score(x, t)

xt = reverse_sampling(drift, diffusion, x, t).detach()
print(xt.shape)
fig, axs = plt.subplots(1, 2, figsize=(20, 10), sharey=True)
axs[1].plot(t, xt[:, :, 0, 0], color="steelblue", alpha=0.1)
axs[1].set_title("$\\leftarrow$ Reverse Flow (ODE) $\\leftarrow$")
axs[0].invert_xaxis()
axs[0].hist(xt[-1, :, 0, 0].flatten(), bins=100, density=True, orientation="horizontal")
x_grid_b = torch.linspace(-5, 5, 100).reshape(-1, 1, 1)
axs[0].plot(gmm.log_prob(x_grid_b, t=eps).exp().squeeze(), x_grid_b.squeeze())
axs[0].set_title("$\\leftarrow$ Target of Reverse Flow $\\leftarrow$")
axs[0].set_xlabel("t")
axs[0].set_ylabel("x")
plt_show()

# %%[markdown]
# # All Samplers: Forward + Reverse Trajectories Overlaid
# Rows: FlowMatching ODE, FlowMatching SDE, BetaSchedule ODE, BetaSchedule SDE
# Columns: p(x, t≈0) | trajectories (blue=forward 0→1, orange=reverse 1→0) | p(x, t≈1)
plt.style.use("default")
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "text.color": "black",
        "grid.color": "#cccccc",
    }
)

_eps = 1e-2
_n_samples = 2_000
_n_steps = 200
_gamma = 1.0

_mu = torch.tensor([-2.0, 0.0, 2.0]).reshape(1, 3, 1)
_sigma = torch.tensor([0.3, 0.1, 0.2]).reshape(1, 3, 1)
_weight = torch.tensor([0.33, 0.5, 0.17]).reshape(1, 3)
_x_grid = torch.linspace(-5, 5, 200).reshape(-1, 1, 1)


def _fwd_traj(mode, sched, gmm_, gam, eps_, ns, nsteps):
    t_ = torch.linspace(eps_, 1 - eps_, nsteps)
    x0_ = gmm_.sample(shape=ns, t=eps_)
    if mode == "ODE":
        return t_, forward_sampling(gmm_.velocity, None, x0_, t_)
    if isinstance(sched, BetaSchedule):

        def drift_(x__, t__):
            g = sched.diffusion_coeff(t__)
            return gmm_.velocity(x__, t__) + 0.5 * gam**2 * g**2 * gmm_.score(x__, t__)

        def diff_(t__):
            return gam * sched.diffusion_coeff(t__)

    else:

        def drift_(x__, t__):
            return gmm_.velocity(x__, t__) + 0.5 * gam**2 * gmm_.score(x__, t__)

        def diff_(t__):
            return gam

    return t_, forward_sampling(drift_, diff_, x0_, t_)


def _rev_traj(mode, sched, gmm_, gam, eps_, ns, nsteps):
    t_ = torch.linspace(1 - eps_, eps_, nsteps)
    x1_ = torch.randn(ns, 1, 1)
    if mode == "ODE":
        if isinstance(sched, BetaSchedule):

            def drift_(x__, t__):
                f = sched.forward_drift(x__, t__)
                g = sched.diffusion_coeff(t__)
                return f - 0.5 * g**2 * gmm_.score(x__, t__)

        else:
            drift_ = gmm_.velocity
        return t_, reverse_sampling(drift_, None, x1_, t_)
    if isinstance(sched, BetaSchedule):

        def drift_(x__, t__):
            f = sched.forward_drift(x__, t__)
            g = sched.diffusion_coeff(t__)
            return f - 0.5 * g**2 * (1 + gam**2) * gmm_.score(x__, t__)

        def diff_(t__):
            return gam * sched.diffusion_coeff(t__)

    else:

        def drift_(x__, t__):
            return gmm_.velocity(x__, t__) - 0.5 * gam**2 * gmm_.score(x__, t__)

        def diff_(t__):
            return gam

    return t_, reverse_sampling(drift_, diff_, x1_, t_)


_cases = [
    ("Flow Matching — ODE", LinearSchedule, "ODE"),
    ("Flow Matching — SDE", LinearSchedule, "SDE"),
    ("Beta Schedule — ODE", BetaSchedule, "ODE"),
    ("Beta Schedule — SDE", BetaSchedule, "SDE"),
]

_fig = plt.figure(figsize=(20, 40))
_gs = gridspec.GridSpec(4, 3, figure=_fig, width_ratios=[1, 6, 1], wspace=0.05, hspace=0.5)

for _row, (_title, _sched_cls, _mode) in enumerate(_cases):
    _sched = _sched_cls()
    _gmm = GMM(mu=_mu, sigma=_sigma, weight=_weight, schedule=_sched)

    _t_fwd, _fwd = _fwd_traj(_mode, _sched, _gmm, _gamma, _eps, _n_samples, _n_steps)
    _t_rev, _rev = _rev_traj(_mode, _sched, _gmm, _gamma, _eps, _n_samples, _n_steps)
    _fwd, _rev = _fwd.detach(), _rev.detach()

    _ax_l = _fig.add_subplot(_gs[_row, 0])
    _ax_t = _fig.add_subplot(_gs[_row, 1], sharey=_ax_l)
    _ax_r = _fig.add_subplot(_gs[_row, 2], sharey=_ax_l)

    # Left marginal: p(x, t≈0), forward start samples, reverse end samples
    _p0 = _gmm.log_prob(_x_grid, t=_eps).exp().squeeze().detach()
    _ax_l.plot(_p0, _x_grid.squeeze(), color="gray", lw=1.5)
    _ax_l.hist(_fwd[0, :, 0, 0], bins=50, density=True, orientation="horizontal", alpha=0.5, color="steelblue")
    _ax_l.hist(_rev[-1, :, 0, 0], bins=50, density=True, orientation="horizontal", alpha=0.5, color="darkorange")
    _ax_l.invert_xaxis()
    _ax_l.set_ylabel("x")
    if _row == 0:
        _ax_l.set_title("p(x, t≈0)")

    # Overlaid forward (blue) + reverse (orange) trajectories
    _ax_t.plot(_t_fwd, _fwd[:, :, 0, 0], color="steelblue", alpha=0.05, lw=0.5)
    _ax_t.plot(_t_rev, _rev[:, :, 0, 0], color="darkorange", alpha=0.05, lw=0.5)
    _ax_t.set_title(_title)
    _ax_t.set_xlim(0, 1)
    _ax_t.yaxis.set_visible(False)
    if _row == 3:
        _ax_t.set_xlabel("t")

    # Right marginal: p(x, t≈1), forward end samples, reverse start samples
    _p1 = _gmm.log_prob(_x_grid, t=1 - _eps).exp().squeeze().detach()
    _ax_r.plot(_p1, _x_grid.squeeze(), color="gray", lw=1.5)
    _ax_r.hist(_fwd[-1, :, 0, 0], bins=50, density=True, orientation="horizontal", alpha=0.5, color="steelblue")
    _ax_r.hist(_rev[0, :, 0, 0], bins=50, density=True, orientation="horizontal", alpha=0.5, color="darkorange")
    _ax_r.yaxis.set_visible(False)
    if _row == 0:
        _ax_r.set_title("p(x, t≈1)")

_fig.legend(
    handles=[
        Line2D([0], [0], color="steelblue", lw=2, label="Forward 0→1"),
        Line2D([0], [0], color="darkorange", lw=2, label="Reverse 1→0"),
        Line2D([0], [0], color="gray", lw=2, label="Analytical"),
    ],
    loc="upper center",
    ncol=3,
    fontsize=11,
    bbox_to_anchor=(0.5, 1.02),
)
plt_show()
