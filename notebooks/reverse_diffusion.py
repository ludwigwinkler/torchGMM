import torch
import einops
from matplotlib import pyplot as plt

from torchGMM import TimeDependentGMM
from torchGMM.diffusion import forward_diffusion, reverse_diffusion, denoising

BS = 3
k = 3
d = 1
mu = torch.stack([torch.tensor([-2, 0, 1]).reshape(3, 1) + torch.randn(3,1) for _ in range(BS)])
std = torch.stack([torch.tensor([0.25, 0.25, 0.25]).reshape(-1, 1) + torch.rand(3,1) for _ in range(BS)])
weight = torch.stack([torch.tensor([0.5, 0.3, 0.3]) for _ in range(BS)])

gmm = TimeDependentGMM(mu, std, weight)

# %%
samples = gmm.sample(10_000)

x = torch.linspace(-5, 5, 1000)
x_ = einops.repeat(x, 'N -> B N 1', B=BS)
prob = gmm.log_prob(x_).exp()
plt.hist(samples, bins=50, density=True)
plt.plot(x, prob)
plt.show()

for t in [0, 0.1, 1]:
    samples = gmm.sample(10_000, t)
    prob = gmm.log_prob(x.reshape(-1, 1), t).exp()
    plt.hist(samples, bins=50, density=True, alpha=0.5)
    plt.plot(x, prob, label=f't={t}')
    plt.show()

# %%
x0 = gmm.sample(10_000)

xt, t = forward_diffusion(gmm.schedule, x0, t=0)

plt.figure()
for traj_ in xt.permute(1, 0, 2)[:100]:
    plt.plot(t, traj_, alpha=0.25, color='red')

# %%

x1 = torch.randn(10_000, 1)

model = gmm
xt_rev, t_rev, _, _ = reverse_diffusion(model=model, scheduler=gmm.schedule, x=x1, t=1, denoising_and_resample_fn=denoising)


plt.figure()
for idx, traj_ in enumerate(xt_rev.permute(1, 0, 2)[:100]):
    plt.plot(t_rev, traj_, alpha=0.25, color='blue')
    plt.plot(t, xt[:, idx], alpha=0.25, color='red')
