import matplotlib.pyplot as plt
import torch
from _utils import plt_show
from tqdm import tqdm

from torchGMM import GMM
from torchGMM.sampling import steered_reverse_sampling
from torchGMM.schedule import BetaSchedule, LinearSchedule

torch.manual_seed(0)
device = torch.device("cpu")
print(f"Using device: {device}")
torch.set_default_device(device)

# --- model ---
sched = LinearSchedule()
gmm = GMM(
    mu=torch.tensor([[[-2.5], [2.5]]]),
    sigma=torch.tensor([[[0.8], [0.8]]]),
    weight=torch.tensor([[0.2, 0.8]]),
    schedule=sched,
)

EPS, N, T = 0.001, 25_000, 250
ESS = 0.9
t = torch.linspace(1 - EPS, EPS, T)
xs = torch.linspace(-6, 6, 100).reshape(-1, 1, 1)

# --- four reward configurations ---
rewards = [(-2.0, 1), (-1.0, 1.0), (0.5, 1.0), (2.5, 3.0)]

fig, axes = plt.subplots(len(rewards), 2, figsize=(12, 3.5 * len(rewards)))

for row, (rc, rs) in tqdm(enumerate(rewards)):
    r = lambda x, c=rc, s=rs: -0.5 * (x - c) ** 2 / s**2
    grad_r = lambda x, c=rc, s=rs: -(x - c) / s**2

    def guided_drift(x, t_):
        f, g, sc = sched.forward_drift(x, t_), sched.diffusion_coeff(t_), gmm.score(x, t_)
        return f - g**2 * sc - (1 - t_) * (g**2 / 2) * grad_r(x)

    def weight_update(x, t_, dt):
        f, g, sc = sched.forward_drift(x, t_), sched.diffusion_coeff(t_), gmm.score(x, t_)
        rg, rv = grad_r(x), r(x)
        beta = 1 - t_
        return (rv - beta * rg * f + beta * rg * (g**2 / 2) * sc).squeeze(-1).squeeze(-1) * dt.abs()

    traj, ess_hist = steered_reverse_sampling(
        drift=guided_drift,
        diffusion=sched.diffusion_coeff,
        weight_update=weight_update,
        x=torch.randn(N, 1, 1),
        t=t,
        ess_threshold=ESS,
    )

    # analytical densities
    log_p = gmm.log_prob(xs, t=torch.tensor(EPS)).squeeze()
    log_p_rew = log_p + r(xs).squeeze()
    p_data = log_p.exp()
    p_data /= torch.trapezoid(p_data, xs.squeeze())
    p_rew = (log_p_rew - log_p_rew.max()).exp()
    p_rew /= torch.trapezoid(p_rew, xs.squeeze())

    # histogram of final samples
    x_final = traj[-1, :, 0, 0]
    dx = xs[1, 0, 0] - xs[0, 0, 0]
    edges = torch.cat([xs[0:1, 0, 0] - dx / 2, xs[:, 0, 0] + dx / 2])
    hist, _ = torch.histogram(x_final, bins=edges, density=True)

    # L2 metrics (same as test assertions)
    l2_rew = ((hist - p_rew) ** 2).mean().sqrt().item()
    l2_data = ((hist - p_data) ** 2).mean().sqrt().item()

    # --- left: density plot ---
    ax_d = axes[row, 0]
    ax_d.plot(xs[:, 0, 0], p_data, color="steelblue", lw=1.5, label="unguided")
    ax_d.plot(xs[:, 0, 0], p_rew, color="firebrick", lw=1.5, label="tilted GT")
    ax_d.bar(xs[:, 0, 0], hist, width=dx, alpha=0.4, color="seagreen", label="steered", align="center")
    ax_d.axvline(rc, color="firebrick", lw=1, ls="--")
    ax_d.set_title(f"center={rc}  σ={rs}  |  L2_rew={l2_rew:.4f}  L2_data={l2_data:.4f}")
    ax_d.legend(fontsize=8)
    ax_d.set_ylabel("density")

    # --- right: ESS history ---
    ax_e = axes[row, 1]
    steps = list(range(len(ess_hist)))
    ax_e.plot(steps, ess_hist, color="darkorange", lw=1.2, label="ESS/N")
    ax_e.axhline(ESS, color="red", lw=1, ls="--", label=f"threshold={ESS}")
    ax_e.set_ylim(0, 1.05)
    n_resamples = sum(1 for e in ess_hist if e < ESS)
    ax_e.set_title(f"ESS history  |  {n_resamples} resamples")
    ax_e.set_xlabel("step")
    ax_e.set_ylabel("ESS / N")
    ax_e.legend(fontsize=8)

fig.suptitle("Steered reverse sampling — four reward functions")
plt.tight_layout()
plt_show()
