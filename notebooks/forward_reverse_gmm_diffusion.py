import torch, einops
from torchGMM import TimeDependentGMM, Conditional
from torchGMM.diffusion import forward_sampling, reverse_sampling
from torchGMM.schedule import BetaSchedule
from torch.distributions import MultivariateNormal
import matplotlib.pyplot as plt

# --- plotting ---
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap
from tqdm import tqdm

torch.manual_seed(100)

# %%[markdown]
# # Marginal Distributions
#
# Marginal distributions we're going to use in the diffusion

x0 = torch.linspace(-3, 3, 5).reshape(-1, 1)
mu = torch.tensor([-2, 0, 2]).reshape(1, 3, 1)
sigma = torch.tensor([0.3, 0.1, 0.2]).reshape(1, 3, 1)
weight = torch.tensor([0.33, 0.5, 0.1]).reshape(1, 3)

gmm = TimeDependentGMM(mu=mu, sigma=sigma, weight=weight)
x_grid_b = torch.linspace(-5, 5, 100).reshape(-1, 1, 1)  # [N, B=1, D=1]
plt.plot(x_grid_b.squeeze(), gmm.log_prob(x_grid_b, t=0.0).exp().squeeze())
plt.show()

conditional_gmm = Conditional(x0=x0)
x_grid_cond = torch.linspace(-5, 5, 100).reshape(-1, 1, 1).expand(-1, 5, -1)  # [N, B=5, D=1]

t = [0.01, 0.1, 0.2, 0.25, 0.35, 0.5, 0.75, 1.0]
colors = plt.cm.get_cmap("viridis", len(t))
for idx, t in enumerate(t):
    log_prob = conditional_gmm.log_prob(x_grid_cond, t=t)
    fig, axs = plt.subplots(1, 1, figsize=(10, 10))
    for i in range(5):
        axs.plot(x_grid_cond[:, i, 0], log_prob[:, i].exp(), label=f"t={t}", color=colors(idx))
        axs.legend()
plt.legend()
plt.show()

gmm = Conditional(x0=x0)
x_grid_cond = torch.linspace(-5, 5, 100).reshape(-1, 1, 1).expand(-1, 5, -1)
log_prob = gmm.log_prob(x_grid_cond, t=0.0)
assert log_prob.shape == (100, 5)
score = gmm.score(x_grid_cond, t=0.0)
assert score.shape == (100, 5, 1)


# %%[markdown]
# # Forward Diffusion
schedule = BetaSchedule()
x = einops.repeat(x0, "B 1 -> N B 1", N=50)
colors = plt.cm.get_cmap("viridis", x0.shape[0])

t = torch.linspace(0.01, 1.0, 100)
xt = forward_sampling(schedule.forward_drift, schedule.diffusion_coeff, x, t)  # [T, N, B, D=1]
for x0_idx in range(x0.shape[0]):
    _ = plt.plot(t, xt[:, :, x0_idx, 0], color=colors(x0_idx), alpha=0.2)
_ = plt.legend()
plt.title("$\u2192$ Forward Diffusion $\u2192$")
plt.xlabel("t")
plt.ylabel("x")
plt.show()

# %%[markdown]
# # Reverse Diffusion
x = torch.randn(50, 5, 1)
colors = plt.cm.get_cmap("viridis", x0.shape[0])

t = torch.linspace(1.0, 0.01, 100)


def reverse_drift(x_, t_):
    f = schedule.forward_drift(x_, t_)
    g = schedule.diffusion_coeff(t_)
    return f - g**2 * gmm.score(x_, t_)


xt = reverse_sampling(reverse_drift, schedule.diffusion_coeff, x, t).detach()
print(xt.shape)
for x0_idx in range(x0.shape[0]):
    plt.plot(t, xt[:, :, x0_idx, 0], color=colors(x0_idx), alpha=0.2)
plt.legend()
plt.title("$\u2190$ Reverse Diffusion $\u2190$")
plt.xlabel("t")
plt.ylabel("x")
plt.show()

# %%[markdown]
# # Reverse Diffusion with Transition Kernel Score
gmm = TimeDependentGMM(mu=mu, sigma=sigma, weight=weight)

x = torch.randn(1_000, 1, 1)  # [N, B=1, D=1]
colors = plt.cm.get_cmap("viridis", x0.shape[0])

t = torch.linspace(1.0, 0.01, 100)


def reverse_drift_gmm(x_, t_):
    f = schedule.forward_drift(x_, t_)
    g = schedule.diffusion_coeff(t_)
    return f - g**2 * gmm.score(x_, t_)


xt = reverse_sampling(reverse_drift_gmm, schedule.diffusion_coeff, x, t).detach()
print(xt.shape)
fig, axs = plt.subplots(1, 2, figsize=(20, 10), sharey=True)
axs[1].plot(t, xt[:, :, 0, 0], color=colors(x0_idx), alpha=0.2)
axs[1].legend()
axs[1].set_title("$\u2190$ Reverse Diffusion $\u2190$")
axs[0].invert_xaxis()
axs[0].hist(xt[-1, :, 0, 0].flatten(), bins=100, density=True, orientation="horizontal")
x_grid_b = torch.linspace(-5, 5, 100).reshape(-1, 1, 1)
axs[0].plot(gmm.log_prob(x_grid_b, t=0.0).exp().squeeze(), x_grid_b.squeeze())
axs[0].set_title("$\u2190$ Target of Reverse Diffusion $\u2190$")
axs[0].set_xlabel("t")
axs[0].set_ylabel("x")
plt.show()
