import math
from pathlib import Path

import pytest
import torch
from test_steering import _wasserstein1, _weighted_wasserstein1, plot_marginal_density_comparison

from torchGMM import GMM, KarrasSchedule, steered_reverse_edm_sampling

PLOT = True
PLOT_DIR = Path(__file__).parent / "plots"


def plot_edm_sampling_diagnostics(
    trajectory,
    sigma,
    ess_history,
    beta_sigma,
    reward_center,
    title,
    out_path,
    n_trajectories=200,
    alpha=0.08,
):
    """Save vertically stacked particle-trajectory, ESS, and tilt-schedule diagnostics."""
    import matplotlib.pyplot as plt

    n_particles = trajectory.shape[1]
    n_plot = min(n_trajectories, n_particles)
    idx_plot = torch.linspace(0, n_particles - 1, n_plot, dtype=torch.long, device=trajectory.device)
    ess = torch.as_tensor(ess_history, dtype=sigma.dtype, device=sigma.device)
    beta = beta_sigma(sigma).detach()

    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)
    axes[0].plot(sigma.cpu(), trajectory[:, idx_plot, 0, 0].cpu(), color="darkorange", alpha=alpha, linewidth=0.7)
    axes[0].axhline(reward_center, color="firebrick", linestyle="--", linewidth=1.2, label="reward center")
    axes[0].set_ylabel(r"$x_\sigma$")
    axes[0].set_title(title)
    axes[0].legend(fontsize=9)

    axes[1].plot(sigma[:-1].cpu(), ess.cpu(), color="darkorange", linewidth=1.2)
    axes[1].set_ylabel("ESS / N")
    axes[1].set_ylim(0, 1.05)

    axes[2].plot(sigma.cpu(), beta.cpu(), color="firebrick", linewidth=1.5)
    axes[2].set_xlabel(r"$\sigma$")
    axes[2].set_ylabel(r"$\beta_\sigma(\sigma)$")

    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlim(sigma[0].item(), sigma[-1].item())
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _reward(x: torch.Tensor, sigma: torch.Tensor, center: float = 0.3) -> torch.Tensor:
    return -0.5 * (x - center).square()


def _grad_reward(x: torch.Tensor, sigma: torch.Tensor, center: float = 0.3) -> torch.Tensor:
    return -(x - center)


def _partial_sigma_reward(x: torch.Tensor, sigma: torch.Tensor, center: float = 0.3) -> torch.Tensor:
    return torch.zeros_like(x)


@pytest.fixture
def karras_setup():
    schedule = KarrasSchedule()
    gmm = GMM(
        mu=torch.tensor([[[-1.0], [1.2]]], dtype=torch.float64),
        sigma=torch.tensor([[[0.35], [0.5]]], dtype=torch.float64),
        weight=torch.tensor([[0.4, 0.6]], dtype=torch.float64),
        schedule=schedule,
    )
    return gmm, schedule


class TestEdmSigmaIdentities:
    @pytest.mark.parametrize("rho", [1.0, 3.0, 7.0])
    def test_karras_map_inverse_and_derivative(self, rho):
        schedule = KarrasSchedule(sigma_min=0.05, sigma_max=3.0, rho=rho)
        tau = torch.linspace(0.05, 0.95, 9, dtype=torch.float64)

        u_min = schedule.sigma_min ** (1.0 / rho)
        u_max = schedule.sigma_max ** (1.0 / rho)
        expected_sigma = schedule.sigma_data * (u_max + tau * (u_min - u_max)) ** rho
        sigma = schedule.get_sigma_t(1.0 - tau)
        torch.testing.assert_close(sigma, expected_sigma)
        torch.testing.assert_close(1.0 - schedule.time(sigma), tau)
        torch.testing.assert_close(schedule.time(schedule.get_sigma_t(tau)), tau)

        sigma_leaf = sigma.detach().requires_grad_(True)
        tau_from_sigma = 1.0 - schedule.time(sigma_leaf)
        (actual_derivative,) = torch.autograd.grad(tau_from_sigma.sum(), sigma_leaf)
        expected_derivative = sigma_leaf ** (1.0 / rho - 1.0) / (
            rho
            * schedule.sigma_data ** (1.0 / rho)
            * (schedule.sigma_min ** (1.0 / rho) - schedule.sigma_max ** (1.0 / rho))
        )
        torch.testing.assert_close(actual_derivative, expected_derivative, rtol=1e-10, atol=1e-10)

    @pytest.mark.parametrize("sigma", [-0.1, 3.1])
    def test_karras_time_rejects_out_of_range_sigma(self, sigma):
        schedule = KarrasSchedule(sigma_min=0.05, sigma_max=3.0, rho=3.0)
        with pytest.raises(ValueError, match="sigma must be within"):
            schedule.time(torch.tensor(sigma))

    def test_beta_schedule_composition_and_chain_rule(self, karras_setup):
        _, schedule = karras_setup
        sigma = torch.linspace(0.1, 2.8, 11, dtype=torch.float64).requires_grad_(True)
        tau = 1.0 - schedule.time(sigma)
        beta_sigma = 1.0 - tau

        (actual_derivative,) = torch.autograd.grad(beta_sigma.sum(), sigma)
        sigma_for_tau = sigma.detach().requires_grad_(True)
        tau_for_derivative = 1.0 - schedule.time(sigma_for_tau)
        (dtau_dsigma,) = torch.autograd.grad(tau_for_derivative.sum(), sigma_for_tau)

        torch.testing.assert_close(beta_sigma, schedule.time(sigma))
        torch.testing.assert_close(actual_derivative, -dtau_dsigma)

    def test_ve_diffusion_identity_in_generation_time(self, karras_setup):
        _, schedule = karras_setup
        tau = torch.linspace(0.05, 0.95, 13, dtype=torch.float64).requires_grad_(True)
        repo_time = 1.0 - tau
        sigma = schedule.get_sigma_t(repo_time)
        (dsigma_dtau,) = torch.autograd.grad(sigma.sum(), tau)

        g2 = schedule.diffusion_coeff(repo_time).square()
        torch.testing.assert_close(g2, -2.0 * sigma * dsigma_dtau, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.4])
def test_time_and_sigma_guided_equations_are_equivalent(karras_setup, alpha):
    gmm, schedule = karras_setup
    tau = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)
    repo_time = 1.0 - tau
    sigma = schedule.get_sigma_t(repo_time)
    (dsigma_dtau,) = torch.autograd.grad(sigma, tau, create_graph=True)

    x = torch.tensor([[[-1.2]], [[0.1]], [[1.7]]], dtype=torch.float64)
    score = gmm.score(x, repo_time)
    reward = _reward(x, sigma)
    grad_reward = _grad_reward(x, sigma)
    partial_sigma_reward = _partial_sigma_reward(x, sigma)

    beta = 0.2 + 0.6 * tau.square()
    dbeta_dtau = 1.2 * tau
    dbeta_dsigma = dbeta_dtau / dsigma_dtau
    g2 = schedule.diffusion_coeff(repo_time).square()
    alpha_sq = alpha**2

    time_drift = 0.5 * g2 * ((1.0 + alpha_sq) * score + alpha_sq * beta * grad_reward)
    sigma_drift = -sigma * ((1.0 + alpha_sq) * score + alpha_sq * beta * grad_reward)
    torch.testing.assert_close(time_drift, sigma_drift * dsigma_dtau, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(alpha_sq * g2, alpha_sq * (-2.0 * sigma * dsigma_dtau))

    time_weight_rate = (
        dbeta_dtau * reward + beta * partial_sigma_reward * dsigma_dtau + beta * grad_reward * (g2 / 2.0) * score
    )
    sigma_weight_rate = dbeta_dsigma * reward + beta * partial_sigma_reward - sigma * beta * grad_reward * score
    torch.testing.assert_close(time_weight_rate, sigma_weight_rate * dsigma_dtau, rtol=1e-10, atol=1e-10)


class TestSteeredReverseEdmSamplingContract:
    def test_reheats_to_next_larger_grid_level(self):
        x0 = torch.tensor([[1.0], [2.0]])
        sigma = torch.tensor([10.0, 4.0, 2.0, 0.5])
        callback_levels = []
        drift_states = []
        weight_states = []

        def drift(x, sigma_hat, sigma_curr, sigma_next):
            drift_states.append(x.clone())
            return x / sigma_hat.square()

        def weight_update(x, sigma_hat, sigma_curr, sigma_next):
            weight_states.append(x.clone())
            callback_levels.append((sigma_hat.item(), sigma_curr.item(), sigma_next.item()))
            return torch.tensor([1.0, -1.0])

        torch.manual_seed(7)
        noise = torch.randn_like(x0)
        torch.manual_seed(7)
        trajectory, _, weight_history = steered_reverse_edm_sampling(
            drift=drift,
            weight_update=weight_update,
            x=x0,
            sigma=sigma,
            ess_threshold=1_000,
        )

        torch.testing.assert_close(trajectory[0], x0)
        first_step_scale = 1.0 + (sigma[1].square() - sigma[0].square()) / (2.0 * sigma[0].square())
        torch.testing.assert_close(trajectory[1], first_step_scale * x0)
        x_noisy = trajectory[1] + math.sqrt(10.0**2 - 4.0**2) * noise
        torch.testing.assert_close(drift_states[1], x_noisy)
        torch.testing.assert_close(weight_states[1], x_noisy)
        second_step_scale = 1.0 + (sigma[2].square() - sigma[0].square()) / (2.0 * sigma[0].square())
        torch.testing.assert_close(trajectory[2], second_step_scale * x_noisy)
        torch.testing.assert_close(weight_history[1], torch.softmax(torch.tensor([1.0, -1.0]), dim=0))
        assert callback_levels[0] == (10.0, 10.0, 4.0)
        assert callback_levels[1] == (10.0, 4.0, 2.0)
        assert callback_levels[2] == (4.0, 2.0, 0.5)

    def test_first_step_has_no_reheat(self):
        x0 = torch.tensor([[1.0], [2.0]])
        sigma = torch.tensor([3.0, 2.0, 1.0])

        trajectory, _, _ = steered_reverse_edm_sampling(
            drift=lambda x, sigma_hat, sigma_curr, sigma_next: x / sigma_hat.square(),
            weight_update=lambda x, sigma_hat, sigma_curr, sigma_next: torch.zeros(x.shape[0]),
            x=x0,
            sigma=sigma,
            noise_scale=1.0,
            step_scale=1.0,
            ess_threshold=1_000,
        )

        first_step_scale = 1.0 + (sigma[1].square() - sigma[0].square()) / (2.0 * sigma[0].square())
        torch.testing.assert_close(trajectory[1], first_step_scale * x0)

    def test_rejects_non_decreasing_sigma(self):
        with pytest.raises(ValueError, match="strictly decreasing"):
            steered_reverse_edm_sampling(
                drift=lambda x, sigma_hat, sigma_curr, sigma_next: torch.zeros_like(x),
                weight_update=lambda x, sigma_hat, sigma_curr, sigma_next: torch.zeros(x.shape[0]),
                x=torch.zeros(3, 1),
                sigma=torch.tensor([1.0, 2.0]),
            )


@pytest.mark.slow
class TestSteeredReverseEdmSamplingMarginals:
    N_PARTICLES = 20_000
    N_STEPS = 1000

    @pytest.fixture
    def setup(self):
        schedule = KarrasSchedule()
        gmm = GMM(
            mu=torch.tensor([[[-1.0], [1.0]]]),
            sigma=torch.tensor([[[0.4], [0.4]]]),
            weight=torch.tensor([[0.35, 0.65]]),
            schedule=schedule,
        )
        return gmm, schedule

    def test_zero_potential_matches_base_gmm(self, setup):
        gmm, schedule = setup

        def drift(x, sigma_hat, sigma_curr, sigma_next):
            return -gmm.score(x, schedule.time(sigma_hat))

        def zero_weight_update(x, sigma_hat, sigma_curr, sigma_next):
            return torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)

        tau = torch.linspace(0.0, 1.0, self.N_STEPS)
        sigma = schedule.get_sigma_t(1.0 - tau)
        torch.manual_seed(0)
        x0 = gmm.sample(shape=self.N_PARTICLES, t=torch.tensor(1.0))
        torch.manual_seed(1)
        trajectory, ess_history, weight_history = steered_reverse_edm_sampling(
            drift=drift,
            weight_update=zero_weight_update,
            x=x0,
            sigma=sigma,
            noise_scale=1.0,
            step_scale=1.0,
            ess_threshold=self.N_STEPS + 1,
        )

        plot_sigma_indices = torch.linspace(0, self.N_STEPS - 1, 11, dtype=torch.int64).tolist()
        if PLOT:
            plot_edm_sampling_diagnostics(
                trajectory=trajectory,
                sigma=sigma,
                ess_history=ess_history,
                beta_sigma=torch.zeros_like,
                reward_center=0.0,
                title="EDM sigma-space zero potential | KarrasSchedule",
                out_path=(
                    PLOT_DIR.joinpath(
                        "test_edm_sigma_steering",
                        "TestSteeredReverseEdmSamplingMarginals",
                        "test_zero_potential_matches_base_gmm",
                    )
                    / "diagnostics.png"
                ),
            )
            for sigma_idx in plot_sigma_indices:
                sigma_i = sigma[sigma_idx]
                t = schedule.time(sigma_i)
                radius = max(4.0, 6.0 * sigma_i.item())
                xs = torch.linspace(-radius, radius, 500)
                p_data = gmm.log_prob(xs.reshape(-1, 1, 1), t=t).squeeze().exp()
                p_data = p_data / torch.trapezoid(p_data, xs)
                w1 = _wasserstein1(trajectory[sigma_idx, :, 0, 0], xs, p_data)
                plot_marginal_density_comparison(
                    xs_flat=xs,
                    p_data=p_data,
                    p_rew=p_data,
                    samples=trajectory[sigma_idx, :, 0, 0],
                    weights=weight_history[sigma_idx],
                    reward_center=0.0,
                    title=(f"EDM sigma-space zero potential | KarrasSchedule\nsigma={sigma_i:.3f} | W1={w1:.4f}"),
                    out_path=(
                        PLOT_DIR.joinpath(
                            "test_edm_sigma_steering",
                            "TestSteeredReverseEdmSamplingMarginals",
                            "test_zero_potential_matches_base_gmm",
                        )
                        / f"sigma_idx{sigma_idx:03d}.png"
                    ),
                    min_x=-radius,
                    max_x=radius,
                )

        for sigma_idx in [25, 50, 75, 100, 125, 150, 175, 200, 225, self.N_STEPS - 1]:
            sigma_i = sigma[sigma_idx]
            t = schedule.time(sigma_i)
            radius = max(4.0, 6.0 * sigma_i.item())
            xs = torch.linspace(-radius, radius, 500)
            p_data = gmm.log_prob(xs.reshape(-1, 1, 1), t=t).squeeze().exp()
            p_data = p_data / torch.trapezoid(p_data, xs)
            w1 = _wasserstein1(trajectory[sigma_idx, :, 0, 0], xs, p_data)
            torch.testing.assert_close(
                weight_history[sigma_idx],
                torch.full_like(weight_history[sigma_idx], 1.0 / self.N_PARTICLES),
            )
            tolerance = 0.08 * (1.0 + sigma_i.item())
            assert w1 < tolerance, f"zero potential sigma={sigma_i:.3f}: W1={w1:.4f} >= {tolerance:.4f}"

    @pytest.mark.parametrize("reward_center", [-3, -1.0, 0.75], ids=lambda value: f"center={value}")
    @pytest.mark.parametrize("ess_threshold", [0.85, 1_000], ids=["adaptive", "final-only"])
    def test_weighted_intermediate_marginals(self, setup, reward_center, ess_threshold):
        gmm, schedule = setup
        reward_sigma = 0.9

        def reward(x):
            return -0.5 * (x - reward_center).square() / reward_sigma**2

        def grad_reward(x):
            return -(x - reward_center) / reward_sigma**2

        def beta_sigma(sigma):
            return (1.0 - schedule.time(sigma))**4 / (1 + 0.1 * sigma**2)

        def drift(x, sigma_hat, sigma_curr, sigma_next):
            t = schedule.time(sigma_hat)
            score_drift = -gmm.score(x, t)
            reheat_variance = sigma_hat**2 - sigma_curr**2
            guidance_drift = (
                -beta_sigma(sigma_hat) * reheat_variance  * grad_reward(x)
            )
            guidance_drift /= sigma_hat**2 - sigma_next**2
            return score_drift + guidance_drift

        def weight_update(x, sigma_hat, sigma_curr, sigma_next):
            t = schedule.time(sigma_hat)
            score = gmm.score(x, t)
            dsigma = sigma_curr**2 - sigma_next**2
            potential = (beta_sigma(sigma_next) - beta_sigma(sigma_curr)) * reward(x)
            alignment = beta_sigma(sigma_hat) * grad_reward(x) * score * dsigma / 2.0
            return (potential + alignment).squeeze(-1).squeeze(-1)

        tau = torch.linspace(0.0, 1.0, self.N_STEPS)
        sigma = schedule.get_sigma_t(1.0 - tau)
        torch.manual_seed(0)
        xT = gmm.sample(shape=self.N_PARTICLES, t=torch.tensor(1.0))
        torch.manual_seed(1)
        trajectory, ess_history, weight_history = steered_reverse_edm_sampling(
            drift=drift,
            weight_update=weight_update,
            x=xT,
            sigma=sigma,
            noise_scale=1.0,
            step_scale=1.0,
            ess_threshold=ess_threshold,
        )

        assert trajectory.shape == (self.N_STEPS, self.N_PARTICLES, 1, 1)
        assert weight_history.shape == (self.N_STEPS, self.N_PARTICLES)

        plot_sigma_indices = torch.linspace(0, self.N_STEPS - 1, 11, dtype=torch.int64).tolist()
        if PLOT:
            plot_edm_sampling_diagnostics(
                trajectory=trajectory,
                sigma=sigma,
                ess_history=ess_history,
                beta_sigma=beta_sigma,
                reward_center=reward_center,
                title=(f"EDM sigma-space FKC | KarrasSchedule\ness={ess_threshold}, center={reward_center}"),
                out_path=(
                    PLOT_DIR.joinpath(
                        "test_edm_sigma_steering",
                        "TestSteeredReverseEdmSamplingMarginals",
                        "test_weighted_intermediate_marginals",
                        f"ess{ess_threshold}",
                        f"center{reward_center}",
                    )
                    / "diagnostics.png"
                ),
            )
            for sigma_idx in plot_sigma_indices:
                sigma_i = sigma[sigma_idx]
                t = schedule.time(sigma_i)
                radius = max(4.0, 6.0 * sigma_i.item())
                xs = torch.linspace(-radius, radius, 500).reshape(-1, 1, 1)
                log_tilt = gmm.log_prob(xs, t=t).squeeze() + beta_sigma(sigma_i) * reward(xs).squeeze()
                target = torch.exp(log_tilt - log_tilt.max())
                target = target / torch.trapezoid(target, xs.squeeze())
                w1 = _weighted_wasserstein1(
                    trajectory[sigma_idx, :, 0, 0],
                    weight_history[sigma_idx],
                    xs.squeeze(),
                    target,
                )
                p_data = gmm.log_prob(xs, t=t).squeeze().exp()
                p_data = p_data / torch.trapezoid(p_data, xs.squeeze())
                plot_marginal_density_comparison(
                    xs_flat=xs.squeeze(),
                    p_data=p_data,
                    p_rew=target,
                    samples=trajectory[sigma_idx, :, 0, 0],
                    weights=weight_history[sigma_idx],
                    reward_center=reward_center,
                    title=(
                        "EDM sigma-space FKC | KarrasSchedule\n"
                        f"ess={ess_threshold}, center={reward_center}, "
                        f"sigma={sigma_i:.3f} | weighted W1={w1:.4f}"
                    ),
                    out_path=(
                        PLOT_DIR.joinpath(
                            "test_edm_sigma_steering",
                            "TestSteeredReverseEdmSamplingMarginals",
                            "test_weighted_intermediate_marginals",
                            f"ess{ess_threshold}",
                            f"center{reward_center}",
                        )
                        / f"sigma_idx{sigma_idx:03d}.png"
                    ),
                    min_x=-radius,
                    max_x=radius,
                )

        sigma_indices = [25, 50, 75, 100, 125, 150, 175, 200, self.N_STEPS - 2, self.N_STEPS - 1]
        for sigma_idx in sigma_indices:
            sigma_i = sigma[sigma_idx]
            t = schedule.time(sigma_i)
            radius = max(4.0, 6.0 * sigma_i.item())
            xs = torch.linspace(-radius, radius, 500).reshape(-1, 1, 1)
            log_tilt = gmm.log_prob(xs, t=t).squeeze() + beta_sigma(sigma_i) * reward(xs).squeeze()
            target = torch.exp(log_tilt - log_tilt.max())
            target = target / torch.trapezoid(target, xs.squeeze())
            w1 = _weighted_wasserstein1(
                trajectory[sigma_idx, :, 0, 0],
                weight_history[sigma_idx],
                xs.squeeze(),
                target,
            )
            tolerance = 0.10 * (1.0 + sigma_i.item())
            assert w1 < tolerance, (
                f"center={reward_center} ess={ess_threshold} sigma={sigma_i:.3f}: W1={w1:.4f} >= {tolerance:.4f}"
            )
