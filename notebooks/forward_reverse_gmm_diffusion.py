import torch, einops
from torchGMM import TimeDependentGMM, Conditional
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
x = einops.repeat(x0, "B 1 -> N B 1", N=50)
colors = plt.cm.get_cmap("viridis", x0.shape[0])
xt = []

dt = 1 / 100
for t in torch.linspace(0.01, 1.0, 100):
    beta_t = gmm.schedule.beta(t)
    dx = -0.5 * beta_t * x * dt + torch.sqrt(beta_t) * torch.randn_like(x) * dt**0.5
    x = x + dx
    xt.append(x)
xt = torch.stack(xt)
t = torch.linspace(0.01, 1.0, 100)
# print(xt.shape)
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
xt = []

"""
x = [N1, N2, ..., Nk, D]
gmm is of shape [BS, k, D]
gmm.score(x, t) is of shape [N1, N2, ..., Nk, BS, D]
gmm(x,t) is of shape [N1, N2, ..., Nk, BS]
we need to extract the diagonal of the score function
for each batch element, we need to extract the diagonal of the score function
for each batch element, we need to extract the diagonal of the score function
"""

dt = 1 / 100
for t in torch.linspace(1.0, 0.01, 100):
    beta_t = gmm.schedule.beta(t)
    dx = (
        -(-0.5 * beta_t * x * dt) + (beta_t * gmm.score(x, t)) * dt + torch.sqrt(beta_t) * torch.randn_like(x) * dt**0.5
    )
    x = x + dx
    xt.append(x)
xt = torch.stack(xt).detach()
t = torch.linspace(1, 0.01, 100)
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

x = torch.randn(1_000, 1)
colors = plt.cm.get_cmap("viridis", x0.shape[0])
xt = []

dt = 1 / 100
for t in tqdm(torch.linspace(1.0, 0.01, 100)):
    beta_t = gmm.schedule.beta(t)
    x0_ = torch.randn_like(x)
    with torch.enable_grad():
        x.requires_grad_(True)
        x_b = x.unsqueeze(-1)  # [N, B=1, D=1] for gmm
        logpx_x0 = Conditional(x0=x0_).log_prob(x, t)
        logpx = gmm.log_prob(x_b, t)
        grad_logpx_x0 = torch.autograd.grad(logpx_x0.sum(), x, create_graph=False)[0]
        grad_logpx = gmm.score(x_b, t).squeeze(-1)
        score = grad_logpx_x0 - (grad_logpx_x0 - grad_logpx)
    dx = -(-0.5 * beta_t * x * dt) + (beta_t * score) * dt + torch.sqrt(beta_t) * torch.randn_like(x) * dt**0.5
    x = x + dx
    xt.append(x)
xt = torch.stack(xt).detach()
t = torch.linspace(1, 0.01, 100)
print(xt.shape)
fig, axs = plt.subplots(1, 2, figsize=(20, 10), sharey=True)
axs[1].plot(t, xt[:, :, 0], color=colors(x0_idx), alpha=0.2)
axs[1].legend()
axs[1].set_title("$\u2190$ Reverse Diffusion $\u2190$")
axs[0].invert_xaxis()
axs[0].hist(xt[-1, :, 0].flatten(), bins=100, density=True, orientation="horizontal")
x_grid_b = torch.linspace(-5, 5, 100).reshape(-1, 1, 1)
axs[0].plot(gmm.log_prob(x_grid_b, t=0.0).exp().squeeze(), x_grid_b.squeeze())
axs[0].set_title("$\u2190$ Target of Reverse Diffusion $\u2190$")
axs[0].set_xlabel("t")
axs[0].set_ylabel("x")
plt.show()
