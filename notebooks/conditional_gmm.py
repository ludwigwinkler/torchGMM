import torch
from torchGMM import TimeDependentGMM
from torch.distributions import MultivariateNormal
import matplotlib.pyplot as plt

torch.manual_seed(100)

x_grid = torch.linspace(-5, 5, 100).reshape(-1, 1)
x0 = torch.linspace(-3, 3, 5).reshape(-1, 1)
mu = torch.tensor([-2, 0, 2]).reshape(3,1)
sigma = torch.tensor([0.3, 0.5, 0.2]).reshape(3,1)
weight = torch.tensor([0.33, 0.5, 0.1])

gmm = TimeDependentGMM(mu=mu, sigma=sigma, weight=weight)
plt.plot(x_grid.squeeze(), gmm(x_grid).exp())
plt.show()

conditional_gmm = TimeDependentGMM(mu=x0)



t = [0.01, 0.1, 0.2, 0.25, 0.35, 0.5, 0.75, 1.0]
colors = plt.cm.get_cmap('viridis', len(t))
for idx, t in enumerate(t):
    log_prob = conditional_gmm(x_grid, t)
    print(log_prob.shape)
    plt.plot(x_grid, log_prob.exp(), label=f't={t}', color=colors(idx))
plt.legend()
plt.show()

x_grid = torch.linspace(-5, 5, 100).reshape(-1, 1)

gmm = TimeDependentGMM(mu=x0)

# --- test log_prob and score ---
log_prob = gmm.log_prob(x_grid, t=0.0)
assert log_prob.shape == (5, 100)
score = gmm.score(x_grid, t=0.0)
assert score.shape == (5, 100, 1)

# --- plotting ---
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap

t = torch.linspace(0.01, 1.0, 5)
colors = plt.cm.get_cmap('viridis', len(t))

for idx, t_val in enumerate(t):
    log_prob = gmm.log_prob(x_grid, t=t_val)
    print(log_prob.shape)
    for i in range(log_prob.shape[0]):
        plt.plot(x_grid.numpy(), log_prob[i].exp().numpy(), label=f't={t_val:.2f}, mu={x0[i].item():.2f}', color=colors(idx), alpha=0.5)

plt.legend()
plt.grid()
plt.show()

multivariate_normal = MultivariateNormal(loc=torch.zeros(1), covariance_matrix=torch.eye(1))
