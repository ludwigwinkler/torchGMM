import torch
from torchGMM.diffusion import forward_diffusion, reverse_diffusion
from torchGMM.gmm import TimeDependentGMM
from torchGMM.schedule import BetaSchedule
import matplotlib.pyplot as plt

schedule = BetaSchedule()

mu = torch.tensor([-2, 0, 2]).reshape(1, 3, 1)
sigma = torch.tensor([0.3, 0.3, 0.2]).reshape(1, 3, 1)
weight = torch.tensor([0.33, 0.5, 0.1]).reshape(1, 3)
gmm = TimeDependentGMM(mu=mu, sigma=sigma, weight=weight)

x = gmm.sample(500, t=0.0)
t = torch.linspace(0.00, 1.0, 100)
trajectory = forward_diffusion(schedule, x, t)


# Trajectory: [n_steps+1, n_samples, 1]

# Visualize some example trajectories for a subset of particles
if False:
    n_show = 100  # Number of trajectories to plot
    fig = plt.figure(figsize=(5, 3))
    for i in range(n_show):
        plt.plot(t, trajectory[:, i, 0].numpy(), color="red", alpha=0.2)
    plt.title("$\u2192$ Forward Diffusion $\u2192$")

# %%[markdown]
# # Standard Reverse Diffusion
x = torch.randn(1000, 1)
t = torch.linspace(1.0, 0.00, 100)
x_grid = torch.linspace(-5, 5, 100).reshape(-1, 1)
target_dist = gmm.log_prob(x_grid, t=0.0).exp()
print(f"{target_dist.shape=}")
trajectory = reverse_diffusion(schedule, lambda x, t: gmm.score(x, t), x, t)
print(trajectory.shape)

# Visualize some example trajectories for a subset of particles
n_show = 100  # Number of trajectories to plot
fig, axs = plt.subplots(1, 2, figsize=(7, 3), gridspec_kw={"width_ratios": [1, 2]}, sharey=True)
for i in range(n_show):
    axs[1].plot(t, trajectory[:, i, 0].numpy(), color="red", alpha=0.2)
axs[1].set_ylim(-5, 5)
axs[0].hist(trajectory[-1, :, 0].numpy(), bins=50, density=True, orientation="horizontal")
axs[0].plot(target_dist, x_grid, label="Target Distribution")
axs[0].set_xlim(0, 1)
axs[0].invert_xaxis()
plt.title("$\u2190$ Reverse Diffusion $\u2190$")
