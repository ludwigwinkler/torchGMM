from pathlib import Path

import pytest
import torch
from test_steering import _wasserstein1, _weighted_wasserstein1, plot_marginal_density_comparison

from torchGMM import GMM, KarrasSchedule, steered_reverse_af3_sampling

PLOT = True
PLOT_DIR = Path(__file__).parent / "plots"


@pytest.fixture
def af3_setup():
    schedule = KarrasSchedule(sigma_min=4e-4, sigma_max=160.0, rho=7.0, sigma_data=16.0)
    gmm = GMM(
        mu=schedule.sigma_data * torch.tensor([[[-1.0], [1.0]]]),
        sigma=schedule.sigma_data * torch.tensor([[[0.4], [0.4]]]),
        weight=torch.tensor([[0.35, 0.65]]),
        schedule=schedule,
    )
    return gmm, schedule


class TestSteeredReverseAf3SamplingContract:
    def test_matches_af3_reheat_and_corrected_noisy_direction(self):
        x0 = torch.tensor([[1.0], [2.0]])
        sigma = torch.tensor([3.0, 2.0, 0.5])
        callback_levels = []
        denoise_states = []
        weight_states = []

        def denoise(x, sigma_hat, sigma_curr, sigma_next):
            denoise_states.append(x.clone())
            return torch.zeros_like(x)

        def weight_update(x, sigma_hat, sigma_curr, sigma_next):
            weight_states.append(x.clone())
            callback_levels.append((sigma_hat.item(), sigma_curr.item(), sigma_next.item()))
            return torch.zeros(x.shape[0])

        torch.manual_seed(7)
        noise = torch.randn_like(x0)
        torch.manual_seed(7)
        trajectory, _, weight_history = steered_reverse_af3_sampling(
            denoise=denoise,
            weight_update=weight_update,
            x=x0,
            sigma=sigma,
            ess_threshold=1_000,
        )

        sigma_hat = sigma[0] * 1.8
        x_noisy = x0 + torch.sqrt(sigma_hat.square() - sigma[0].square()) * noise
        expected_first = x_noisy + (sigma[1] - sigma_hat) * x_noisy / sigma_hat

        torch.testing.assert_close(trajectory[0], x0)
        torch.testing.assert_close(denoise_states[0], x_noisy)
        torch.testing.assert_close(weight_states[0], x_noisy)
        torch.testing.assert_close(trajectory[1], expected_first)
        torch.testing.assert_close(denoise_states[1], expected_first)
        torch.testing.assert_close(weight_states[1], expected_first)
        torch.testing.assert_close(weight_history[1], torch.full_like(weight_history[1], 0.5))
        assert callback_levels[0] == pytest.approx((5.4, 3.0, 2.0))
        assert callback_levels[1] == pytest.approx((2.0, 2.0, 0.5))

    def test_rejects_non_decreasing_sigma(self):
        with pytest.raises(ValueError, match="strictly decreasing"):
            steered_reverse_af3_sampling(
                denoise=lambda x, sigma_hat, sigma_curr, sigma_next: torch.zeros_like(x),
                weight_update=lambda x, sigma_hat, sigma_curr, sigma_next: torch.zeros(x.shape[0]),
                x=torch.zeros(3, 1),
                sigma=torch.tensor([1.0, 2.0]),
            )

    @pytest.mark.parametrize(
        ("argument", "value", "match"),
        [
            ("gamma_0", -0.1, "gamma_0 must be non-negative"),
            ("gamma_min", -0.1, "gamma_min must be non-negative"),
            ("noise_scale", -0.1, "noise_scale must be non-negative"),
            ("step_scale", 0.0, "step_scale must be positive"),
        ],
    )
    def test_rejects_invalid_af3_parameters(self, argument, value, match):
        kwargs = {argument: value}
        with pytest.raises(ValueError, match=match):
            steered_reverse_af3_sampling(
                denoise=lambda x, sigma_hat, sigma_curr, sigma_next: torch.zeros_like(x),
                weight_update=lambda x, sigma_hat, sigma_curr, sigma_next: torch.zeros(x.shape[0]),
                x=torch.zeros(3, 1),
                sigma=torch.tensor([2.0, 1.0]),
                **kwargs,
            )

    def test_requires_exactly_one_weighting_mode(self):
        common = {
            "denoise": lambda x, sigma_hat, sigma_curr, sigma_next: torch.zeros_like(x),
            "x": torch.zeros(3, 1),
            "sigma": torch.tensor([2.0, 1.0]),
        }
        with pytest.raises(ValueError, match="exactly one"):
            steered_reverse_af3_sampling(weight_update=None, **common)
        with pytest.raises(ValueError, match="exactly one"):
            steered_reverse_af3_sampling(
                weight_update=lambda x, sigma_hat, sigma_curr, sigma_next: torch.zeros(x.shape[0]),
                potential=lambda x, sigma: torch.zeros(x.shape[0]),
                **common,
            )

    def test_guided_reheat_adds_log_proposal_correction(self):
        x0 = torch.tensor([[1.0], [2.0]])
        sigma = torch.tensor([2.0, 1.5, 0.5])

        def reheat_proposal(x, sigma_hat, sigma_curr, sigma_next):
            return x + 1.0, torch.tensor([0.5, -0.5])

        trajectory, _, weight_history = steered_reverse_af3_sampling(
            denoise=lambda x, sigma_hat, sigma_curr, sigma_next: x,
            weight_update=lambda x, sigma_hat, sigma_curr, sigma_next: torch.zeros(x.shape[0]),
            x=x0,
            sigma=sigma,
            ess_threshold=1_000,
            reheat_proposal=reheat_proposal,
        )

        torch.testing.assert_close(trajectory[1], x0 + 1.0)
        torch.testing.assert_close(weight_history[1], torch.softmax(torch.tensor([0.5, -0.5]), dim=0))


@pytest.mark.slow
class TestSteeredReverseAf3SamplingMarginals:
    N_PARTICLES = 10_000
    N_STEPS = 1000

    def test_zero_potential_matches_base_gmm(self, af3_setup):
        gmm, schedule = af3_setup

        def denoise(x, sigma_hat, sigma_curr, sigma_next):
            sigma_floor = gmm.schedule.get_sigma_t(torch.zeros((), dtype=x.dtype, device=x.device))
            noised_gmm = GMM(
                mu=gmm.mu,
                sigma=torch.sqrt(gmm.sigma.square() + sigma_hat.square() - sigma_floor.square()),
                weight=gmm.weight,
                schedule=gmm.schedule,
            )
            return x + sigma_hat.square() * noised_gmm.score(x, t=0.0)

        def zero_weight_update(x, sigma_hat, sigma_curr, sigma_next):
            return torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)

        tau = torch.linspace(0.0, 1.0, self.N_STEPS)
        sigma = schedule.get_sigma_t(1.0 - tau)
        torch.manual_seed(0)
        x0 = gmm.sample(shape=self.N_PARTICLES, t=torch.tensor(1.0))
        torch.manual_seed(1)
        trajectory, _, weight_history = steered_reverse_af3_sampling(
            denoise=denoise,
            weight_update=zero_weight_update,
            x=x0,
            sigma=sigma,
            noise_scale=1.0,
            step_scale=1.0,
            ess_threshold=self.N_STEPS + 1,
        )

        for sigma_idx in torch.linspace(0, self.N_STEPS - 1, 12, dtype=torch.int64)[1:].tolist():
            sigma_i = sigma[sigma_idx]
            t = schedule.time(sigma_i)
            radius = max(4.0 * schedule.sigma_data, 6.0 * sigma_i.item())
            xs = torch.linspace(-radius, radius, 500)
            target = gmm.log_prob(xs.reshape(-1, 1, 1), t=t).squeeze().exp()
            target = target / torch.trapezoid(target, xs)
            w1 = _wasserstein1(trajectory[sigma_idx, :, 0, 0], xs, target)

            torch.testing.assert_close(
                weight_history[sigma_idx],
                torch.full_like(weight_history[sigma_idx], 1.0 / self.N_PARTICLES),
            )
            tolerance = 0.085 * (schedule.sigma_data + sigma_i.item())
            assert w1 < tolerance, f"zero potential sigma={sigma_i:.3f}: W1={w1:.4f} >= {tolerance:.4f}"

    @pytest.mark.parametrize("reward_center", [-1.0, 3.0], ids=lambda value: f"center={value}")
    @pytest.mark.parametrize("ess_threshold", [0.85, 50, 1001], ids=["adaptive", "interval", "final-only"])
    def test_discrete_potential_intermediate_marginals(self, af3_setup, reward_center, ess_threshold):
        gmm, schedule = af3_setup
        reward_center = reward_center * schedule.sigma_data
        reward_sigma = 0.9 * schedule.sigma_data
        sigma_max = schedule.sigma_data * schedule.sigma_max
        sigma_min = schedule.sigma_data * schedule.sigma_min

        def reward(x):
            return -0.5 * (x - reward_center).square() / reward_sigma**2

        def beta_sigma(sigma):
            sigma_on_grid = sigma.clamp(min=sigma_min, max=sigma_max)
            denoising_progress = 1.0 - schedule.time(sigma_on_grid)
            return denoising_progress / (1.0 + 0.1 * (sigma / schedule.sigma_data).square())

        def denoise(x, sigma_hat, sigma_curr, sigma_next):
            sigma_floor = gmm.schedule.get_sigma_t(torch.zeros((), dtype=x.dtype, device=x.device))
            noised_gmm = GMM(
                mu=gmm.mu,
                sigma=torch.sqrt(gmm.sigma.square() + sigma_hat.square() - sigma_floor.square()),
                weight=gmm.weight,
                schedule=gmm.schedule,
            )
            return x + sigma_hat.square() * noised_gmm.score(x, t=0.0)

        def potential(x, sigma):
            return (beta_sigma(sigma) * reward(x)).squeeze(-1).squeeze(-1)

        tau = torch.linspace(0.0, 1.0, self.N_STEPS)
        sigma = schedule.get_sigma_t(1.0 - tau)
        torch.manual_seed(0)
        x0 = gmm.sample(shape=self.N_PARTICLES, t=torch.tensor(1.0))
        torch.manual_seed(1)
        trajectory, _, weight_history = steered_reverse_af3_sampling(
            denoise=denoise,
            weight_update=None,
            x=x0,
            sigma=sigma,
            noise_scale=1.0,
            step_scale=1.0,
            ess_threshold=ess_threshold,
            potential=potential,
        )

        for sigma_idx in torch.linspace(0, self.N_STEPS - 1, 12, dtype=torch.int64)[1:].tolist():
            sigma_i = sigma[sigma_idx]
            t = schedule.time(sigma_i)
            radius = max(4.0 * schedule.sigma_data, 6.0 * sigma_i.item())
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
            if PLOT:
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
                        "AF3 discrete-potential FKC | KarrasSchedule\n"
                        f"ess={ess_threshold}, center={reward_center}, "
                        f"sigma={sigma_i:.3f} | weighted W1={w1:.4f}"
                    ),
                    out_path=(
                        PLOT_DIR.joinpath(
                            "test_af3_steering",
                            "TestSteeredReverseAf3SamplingMarginals",
                            "test_discrete_potential_intermediate_marginals",
                            f"ess{ess_threshold}",
                            f"center{reward_center}",
                        )
                        / f"sigma_idx{sigma_idx:03d}.png"
                    ),
                    min_x=-radius,
                    max_x=radius,
                )
            tolerance = 0.08 * (schedule.sigma_data + sigma_i.item())
            assert w1 < tolerance, (
                f"tilted center={reward_center} ess={ess_threshold} sigma={sigma_i:.3f}: W1={w1:.4f} >= {tolerance:.4f}"
            )

    @pytest.mark.parametrize("reward_center", [-1.0, 3.0], ids=lambda value: f"center={value}")
    def test_guided_reheat_matches_tilted_marginals(self, af3_setup, reward_center):
        gmm, schedule = af3_setup
        reward_center = reward_center * schedule.sigma_data
        reward_sigma = 0.9 * schedule.sigma_data
        sigma_max = schedule.sigma_data * schedule.sigma_max
        sigma_min = schedule.sigma_data * schedule.sigma_min

        def reward(x):
            return -0.5 * (x - reward_center).square() / reward_sigma**2

        def beta_sigma(sigma):
            sigma_on_grid = sigma.clamp(min=sigma_min, max=sigma_max)
            denoising_progress = 1.0 - schedule.time(sigma_on_grid)
            return denoising_progress / (1.0 + 0.1 * (sigma / schedule.sigma_data).square())

        def denoise(x, sigma_hat, sigma_curr, sigma_next):
            sigma_floor = gmm.schedule.get_sigma_t(torch.zeros((), dtype=x.dtype, device=x.device))
            noised_gmm = GMM(
                mu=gmm.mu,
                sigma=torch.sqrt(gmm.sigma.square() + sigma_hat.square() - sigma_floor.square()),
                weight=gmm.weight,
                schedule=gmm.schedule,
            )
            return x + sigma_hat.square() * noised_gmm.score(x, t=0.0)

        def potential(x, sigma):
            return (beta_sigma(sigma) * reward(x)).squeeze(-1).squeeze(-1)

        def reheat_proposal(x, sigma_hat, sigma_curr, sigma_next):
            variance = sigma_hat.square() - sigma_curr.square()
            transport_scale = sigma_next / sigma_hat
            tilt_precision = beta_sigma(sigma_next) * transport_scale.square() / reward_sigma**2
            precision_scale = 1.0 + variance * tilt_precision
            proposal_variance = variance / precision_scale
            proposal_mean = (
                x + variance * beta_sigma(sigma_next) * transport_scale * reward_center / reward_sigma**2
            ) / precision_scale
            x_noisy = proposal_mean + torch.sqrt(proposal_variance) * torch.randn_like(x)

            base_log_prob = -0.5 * (torch.log(2.0 * torch.pi * variance) + (x_noisy - x).square() / variance).flatten(
                start_dim=1
            ).sum(dim=-1)
            proposal_log_prob = -0.5 * (
                torch.log(2.0 * torch.pi * proposal_variance) + (x_noisy - proposal_mean).square() / proposal_variance
            ).flatten(start_dim=1).sum(dim=-1)
            return x_noisy, base_log_prob - proposal_log_prob

        tau = torch.linspace(0.0, 1.0, self.N_STEPS)
        sigma = schedule.get_sigma_t(1.0 - tau)
        torch.manual_seed(0)
        x0 = gmm.sample(shape=self.N_PARTICLES, t=torch.tensor(1.0))
        torch.manual_seed(1)
        trajectory, _, weight_history = steered_reverse_af3_sampling(
            denoise=denoise,
            weight_update=None,
            x=x0,
            sigma=sigma,
            ess_threshold=0.85,
            potential=potential,
            reheat_proposal=reheat_proposal,
        )

        for sigma_idx in torch.linspace(0, self.N_STEPS - 1, 12, dtype=torch.int64)[1:].tolist():
            sigma_i = sigma[sigma_idx]
            t = schedule.time(sigma_i)
            radius = max(4.0 * schedule.sigma_data, 6.0 * sigma_i.item())
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
            tolerance = 0.08 * (schedule.sigma_data + sigma_i.item())
            assert w1 < tolerance, (
                f"guided reheat center={reward_center} sigma={sigma_i:.3f}: W1={w1:.4f} >= {tolerance:.4f}"
            )

    @pytest.mark.parametrize("reward_center", [0.0, 0.75], ids=lambda value: f"center={value}")
    def test_continuous_weight_update_matches_tilted_marginals_without_finite_churn(self, af3_setup, reward_center):
        gmm, schedule = af3_setup
        reward_center = reward_center * schedule.sigma_data
        reward_sigma = 0.9 * schedule.sigma_data
        sigma_max = schedule.sigma_data * schedule.sigma_max
        sigma_min = schedule.sigma_data * schedule.sigma_min

        def reward(x):
            return -0.5 * (x - reward_center).square() / reward_sigma**2

        def grad_reward(x):
            return -(x - reward_center) / reward_sigma**2

        def beta_sigma(sigma):
            sigma_on_grid = sigma.clamp(min=sigma_min, max=sigma_max)
            denoising_progress = 1.0 - schedule.time(sigma_on_grid)
            return denoising_progress / (1.0 + 0.1 * (sigma / schedule.sigma_data).square())

        def denoise(x, sigma_hat, sigma_curr, sigma_next):
            sigma_floor = gmm.schedule.get_sigma_t(torch.zeros((), dtype=x.dtype, device=x.device))
            noised_gmm = GMM(
                mu=gmm.mu,
                sigma=torch.sqrt(gmm.sigma.square() + sigma_hat.square() - sigma_floor.square()),
                weight=gmm.weight,
                schedule=gmm.schedule,
            )
            denoised = x + sigma_hat.square() * noised_gmm.score(x, t=0.0)
            reheat_variance = sigma_hat.square() - sigma_curr.square()
            guidance = (
                sigma_hat * beta_sigma(sigma_hat) * grad_reward(x) * reheat_variance / (2.0 * (sigma_hat - sigma_next))
            )
            return denoised + guidance

        def weight_update(x, sigma_hat, sigma_curr, sigma_next):
            sigma_floor = gmm.schedule.get_sigma_t(torch.zeros((), dtype=x.dtype, device=x.device))
            noised_gmm = GMM(
                mu=gmm.mu,
                sigma=torch.sqrt(gmm.sigma.square() + sigma_hat.square() - sigma_floor.square()),
                weight=gmm.weight,
                schedule=gmm.schedule,
            )
            score = noised_gmm.score(x, t=0.0)
            delta = sigma_curr.square() - sigma_next.square()
            potential_increment = (beta_sigma(sigma_next) - beta_sigma(sigma_curr)) * reward(x)
            alignment = beta_sigma(sigma_hat) * grad_reward(x) * score * delta / 2.0
            return (potential_increment + alignment).squeeze(-1).squeeze(-1)

        tau = torch.linspace(0.0, 1.0, self.N_STEPS)
        sigma = schedule.get_sigma_t(1.0 - tau)
        torch.manual_seed(0)
        x0 = gmm.sample(shape=self.N_PARTICLES, t=torch.tensor(1.0))
        torch.manual_seed(1)
        trajectory, _, weight_history = steered_reverse_af3_sampling(
            denoise=denoise,
            weight_update=weight_update,
            x=x0,
            sigma=sigma,
            gamma_0=0.0,
            noise_scale=1.0,
            step_scale=1.0,
            ess_threshold=self.N_STEPS + 1,
        )

        for sigma_idx in torch.linspace(0, self.N_STEPS - 1, 12, dtype=torch.int64)[1:].tolist():
            sigma_i = sigma[sigma_idx]
            t = schedule.time(sigma_i)
            radius = max(4.0 * schedule.sigma_data, 6.0 * sigma_i.item())
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
            tolerance = 0.09 * (schedule.sigma_data + sigma_i.item())
            assert w1 < tolerance, (
                f"continuous FKC center={reward_center} sigma={sigma_i:.3f}: W1={w1:.4f} >= {tolerance:.4f}"
            )
