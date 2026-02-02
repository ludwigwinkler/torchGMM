import torch
from torchGMM import TimeDependentGMM
from matplotlib import pyplot as plt



mu = torch.randn(4, 3, 1) * 2
sigma = torch.rand(4,3,1).clamp(0.1, 0.5)
weight = torch.ones(4, 3)

gmm = TimeDependentGMM(mu, sigma, weight)

for t in torch.linspace(0., 1.0, 11):
    fig, axs = plt.subplots(4, 1, figsize=(10, 10))
    samples = gmm.sample(10_000, t=t)
    x_grid = torch.linspace(-5, 5, 100)
    log_prob = gmm.log_prob(x_grid.reshape(-1,1), t=t)
    for bs in range(4):
        for d in range(1):
            axs[bs].hist(samples[:, bs, d], bins=100, density=True)
            axs[bs].plot(x_grid, log_prob[:,bs].exp())
            axs[bs].set_xlim(-5, 5)
            axs[bs].set_ylim(0, 0.5)
    fig.suptitle(f't={t:.2f}')
    plt.show()


#%%
if False:
    mu = torch.randn(4, 3, 2) * 2
    sigma = torch.rand(4,3,2).clamp(0.1, 0.5)
    weight = torch.ones(4, 3)

    gmm = TimeDependentGMM(mu, sigma, weight)

    for t in torch.linspace(0., 1.0, 11):
        fig, axs = plt.subplots(4, 2, figsize=(10, 10))
        samples = gmm.sample(10_000, t=t)
        x_grid = torch.stack([torch.linspace(-5, 5, 100) for _ in range(2)], dim=-1)
        log_prob = gmm.log_prob(x_grid, t=t)
        for bs in range(4):
            for d in range(2):
                axs[bs, d].hist(samples[:, bs, d], bins=100, density=True)
                axs[bs, d].set_xlim(-5, 5)
                axs[bs, d].set_ylim(0, 0.5)
        fig.suptitle(f't={t:.2f}')
        plt.show()