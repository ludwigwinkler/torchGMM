import torch, einops
from torchGMM import TimeDependentGMM, Conditional
from torchGMM.diffusion import forward_sampling, reverse_sampling
from torchGMM.schedule import FlowMatchingSchedule, BetaSchedule
from torch.distributions import MultivariateNormal
import matplotlib.pyplot as plt

# --- plotting ---
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap
from _utils import plt_show
from tqdm import tqdm
import numpy as np

torch.manual_seed(100)


# %%[markdown]
# # Marginal Distributions
#
# Marginal distributions we're going to use in the diffusion

x0 = torch.linspace(-3, 3, 5).reshape(-1, 1)
mu = torch.tensor([-2, 0, 2]).reshape(1, 3, 1)
sigma = torch.tensor([0.3, 0.1, 0.2]).reshape(1, 3, 1)
weight = torch.tensor([0.33, 0.5, 0.1]).reshape(1, 3)

schedule = FlowMatchingSchedule()
gmm = TimeDependentGMM(mu=mu, sigma=sigma, weight=weight, schedule=schedule)
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
gmm = TimeDependentGMM(mu=mu, sigma=sigma, weight=weight, schedule=schedule)
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
schedule = FlowMatchingSchedule()
gmm = TimeDependentGMM(mu=mu, sigma=sigma, weight=weight, schedule=schedule)

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
schedule = FlowMatchingSchedule()
gmm = TimeDependentGMM(mu=mu, sigma=sigma, weight=weight, schedule=schedule)

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
gmm = TimeDependentGMM(mu=mu, sigma=sigma, weight=weight, schedule=schedule)

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
