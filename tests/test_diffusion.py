import sys

import pytest
import torch

torch.set_printoptions(sci_mode=False)
from torchGMM.diffusion import forward_sampling, reverse_sampling
from torchGMM.gmm import TimeDependentGMM
from torchGMM.schedule import BetaSchedule, FlowMatchingSchedule


def get_local_device():
    """Fixture that returns the best available device (MPS, CUDA, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


@pytest.fixture
def gmm_model():
    """Fixture for the GMM model used in diffusion tests."""
    mu = torch.tensor([-2, 0, 2]).reshape(1, 3, 1)
    sigma = torch.tensor([0.3, 0.3, 0.2]).reshape(1, 3, 1)
    weight = torch.tensor([0.33, 0.5, 0.17]).reshape(1, 3)
    return TimeDependentGMM(mu=mu, sigma=sigma, weight=weight)


@pytest.fixture
def batched_gmm_model():
    """Fixture for the GMM model used in diffusion tests."""
    mu = torch.tensor([[-1, 0.5, 1.5], [-2, 0, 2]]).reshape(2, 3, 1)
    sigma = torch.tensor([[0.3, 0.3, 0.2], [0.3, 0.3, 0.2]]).reshape(2, 3, 1)
    weight = torch.tensor([[1 / 3, 1 / 2, 1 / 6], [0.33, 0.5, 0.17]]).reshape(2, 3)
    return TimeDependentGMM(mu=mu, sigma=sigma, weight=weight)


class TestForwardSampling:

    def test_forward_vp_sde(self, gmm_model):
        """Forward VP-SDE histogram matches analytical marginal at each time step."""
        schedule = BetaSchedule()

        samples = gmm_model.sample(shape=10_000, t=0.0)  # [N, B=1, D=1]
        x = samples.squeeze(1)  # [N, D=1]
        t = torch.linspace(0.00, 1.0, 100)

        def drift(x_, t_):
            return schedule.forward_drift(x_, t_)

        def diffusion(t_):
            return schedule.diffusion_coeff(t_)

        trajectory = forward_sampling(drift, diffusion, x, t)

        assert trajectory.shape == (len(t), x.shape[0], 1)
        assert torch.allclose(trajectory[0], x, atol=1e-5)

        x_grid = torch.linspace(-5, 5, 51).reshape(-1, 1, 1)
        dx = x_grid[1, 0, 0] - x_grid[0, 0, 0]
        x_flat = x_grid.squeeze()
        bin_edges = torch.cat([(x_flat[0] - dx / 2).unsqueeze(0), x_flat + dx / 2])

        for t_idx in range(t.numel())[::3]:
            t_ = t[t_idx]
            target_dist = gmm_model.log_prob(x_grid, t=t_).exp().squeeze(-1)
            hist, _ = torch.histogram(trajectory[t_idx, :, 0], bins=bin_edges, density=True)
            assert target_dist.shape == hist.shape
            assert (
                hist - target_dist
            ).abs().max() < 0.05, f"hist-target_dist.abs().max() @ t={t_}: {(hist-target_dist).abs().max()}"

    def test_t_must_be_increasing(self):
        x = torch.randn(10, 2)
        t = torch.linspace(1.0, 0.0, 10)
        with pytest.raises(ValueError, match="strictly increasing"):
            forward_sampling(lambda x_, t_: x_, None, x, t)


class TestReverseSampling:

    def test_reverse_vp_sde_with_score(self, batched_gmm_model):
        """Reverse VP-SDE using score: dx = [f - g² score] dt + g dW."""
        schedule = BetaSchedule()

        x = torch.randn(50_000, 2, 1)
        t = torch.linspace(1.0, 0.00, 300)

        def drift(x_, t_):
            f = schedule.forward_drift(x_, t_)
            g = schedule.diffusion_coeff(t_)
            return f - g**2 * batched_gmm_model.score(x_, t_)

        def diffusion(t_):
            return schedule.diffusion_coeff(t_)

        trajectory = reverse_sampling(drift, diffusion, x, t)

        assert trajectory.shape == (len(t), x.shape[0], 2, 1)
        assert torch.allclose(trajectory[0], x, atol=1e-5)

        x_grid = torch.linspace(-5, 5, 51).reshape(-1, 1, 1).repeat(1, 2, 1)
        dx = x_grid[1, 0, 0] - x_grid[0, 0, 0]
        x_flat = x_grid[:, 0, 0].squeeze()
        bin_edges = torch.cat([(x_flat[0] - dx / 2).unsqueeze(0), x_flat + dx / 2])

        for t_idx in range(t.numel())[::3]:
            t_ = t[t_idx]
            target_dist = batched_gmm_model.log_prob(x_grid, t=t_).exp().squeeze(-1)
            for dim in range(batched_gmm_model.dim):
                trajectory_dim = trajectory[t_idx, :, dim, 0].unsqueeze(-2)
                hist, _ = torch.histogram(trajectory_dim, bins=bin_edges, density=True)
                assert target_dist.shape[:1] == hist.shape
                assert (hist - target_dist[:, dim]).abs().max() < 0.05

    def test_reverse_vp_sde_with_velocity(self, batched_gmm_model):
        """Reverse VP-SDE using velocity: dx = [2v - f] dt + g dW."""
        schedule = BetaSchedule()

        x = torch.randn(50_000, 2, 1)
        t = torch.linspace(1.0, 0.00, 300)

        def drift(x_, t_):
            f = schedule.forward_drift(x_, t_)
            v = batched_gmm_model.velocity(x_, t_)
            return 2 * v - f

        def diffusion(t_):
            return schedule.diffusion_coeff(t_)

        trajectory = reverse_sampling(drift, diffusion, x, t)

        assert trajectory.shape == (len(t), x.shape[0], 2, 1)
        assert torch.allclose(trajectory[0], x, atol=1e-5)

        x_grid = torch.linspace(-5, 5, 51).reshape(-1, 1, 1).repeat(1, 2, 1)
        dx = x_grid[1, 0, 0] - x_grid[0, 0, 0]
        x_flat = x_grid[:, 0, 0].squeeze()
        bin_edges = torch.cat([(x_flat[0] - dx / 2).unsqueeze(0), x_flat + dx / 2])

        t_final = t[-1]
        target_dist = batched_gmm_model.log_prob(x_grid, t=t_final).exp().squeeze(-1)
        for dim in range(batched_gmm_model.dim):
            trajectory_dim = trajectory[-1, :, dim, 0].unsqueeze(-2)
            hist, _ = torch.histogram(trajectory_dim, bins=bin_edges, density=True)
            assert (hist - target_dist[:, dim]).abs().max() < 0.05

    def test_reverse_ode_with_velocity(self, batched_gmm_model):
        """Probability flow ODE: dx = v(x,t) dt (no noise)."""
        x = torch.randn(50_000, 2, 1)
        t = torch.linspace(1.0, 0.00, 500)

        trajectory = reverse_sampling(batched_gmm_model.velocity, None, x, t)

        assert trajectory.shape == (len(t), x.shape[0], 2, 1)

        # ODE is deterministic — same input gives same output
        trajectory2 = reverse_sampling(batched_gmm_model.velocity, None, x, t)
        torch.testing.assert_close(trajectory, trajectory2)

    def test_t_must_be_decreasing(self):
        x = torch.randn(10, 2)
        t = torch.linspace(0.0, 1.0, 10)
        with pytest.raises(ValueError, match="strictly decreasing"):
            reverse_sampling(lambda x_, t_: x_, None, x, t)

    @pytest.mark.slow
    @pytest.mark.integration
    def test_flow_matching_ode_generation(self):
        """Flow matching ODE: integrate velocity from t~0.5 to t~0, check histogram matches GMM."""
        schedule = FlowMatchingSchedule()
        mu = torch.tensor([-2.0, 0.0, 2.0]).reshape(1, 3, 1)
        sigma = torch.tensor([0.3, 0.3, 0.2]).reshape(1, 3, 1)
        weight = torch.tensor([0.33, 0.5, 0.17]).reshape(1, 3)
        gmm = TimeDependentGMM(mu=mu, sigma=sigma, weight=weight, schedule=schedule)

        n_samples = 50_000
        t_start = 0.5
        t_end = 0.01
        x_start = gmm.sample(shape=n_samples, t=t_start)

        t_grid = torch.linspace(t_start, t_end, 500)

        trajectory = reverse_sampling(gmm.velocity, None, x_start, t_grid)
        x_generated = trajectory[-1]

        x_grid = torch.linspace(-5, 5, 51).reshape(-1, 1, 1)
        dx = x_grid[1, 0, 0] - x_grid[0, 0, 0]
        x_flat = x_grid.squeeze()
        bin_edges = torch.cat([(x_flat[0] - dx / 2).unsqueeze(0), x_flat + dx / 2])

        target_dist = gmm.log_prob(x_grid, t=t_end).exp().squeeze(-1)
        hist, _ = torch.histogram(x_generated[:, 0, 0], bins=bin_edges, density=True)

        assert (
            hist - target_dist
        ).abs().max() < 0.15, f"Max histogram deviation: {(hist - target_dist).abs().max():.3f}"


class TestValidation:
    """Input validation tests for forward_sampling and reverse_sampling."""

    def test_non_monotonic_rejected(self):
        x = torch.randn(10, 2)
        t = torch.tensor([0.0, 0.5, 0.3, 0.8, 1.0])
        with pytest.raises(ValueError):
            forward_sampling(lambda x_, t_: x_, None, x, t)

    def test_t_outside_range_rejected(self):
        x = torch.randn(10, 2)
        t = torch.tensor([-0.1, 0.5, 1.0])
        with pytest.raises(ValueError, match="within \\[0, 1\\]"):
            forward_sampling(lambda x_, t_: x_, None, x, t)

    def test_t_too_few_points_rejected(self):
        x = torch.randn(10, 2)
        t = torch.tensor([0.5])
        with pytest.raises(ValueError, match="at least two"):
            forward_sampling(lambda x_, t_: x_, None, x, t)

    def test_t_non_finite_rejected(self):
        x = torch.randn(10, 2)
        t = torch.tensor([0.0, float("nan"), 1.0])
        with pytest.raises(ValueError, match="finite"):
            forward_sampling(lambda x_, t_: x_, None, x, t)

    def test_ode_deterministic(self):
        """No diffusion -> deterministic."""
        x = torch.randn(20, 2)
        t = torch.linspace(0.0, 1.0, 50)

        def drift(x_, t_):
            return -x_

        traj1 = forward_sampling(drift, None, x, t)
        traj2 = forward_sampling(drift, None, x, t)
        torch.testing.assert_close(traj1, traj2)

    def test_sde_stochastic(self):
        """With diffusion -> stochastic."""
        x = torch.randn(100, 2)
        t = torch.linspace(0.0, 1.0, 50)

        def drift(x_, t_):
            return -x_

        def diffusion(t_):
            return torch.tensor(1.0)

        traj1 = forward_sampling(drift, diffusion, x, t)
        traj2 = forward_sampling(drift, diffusion, x, t)
        assert not torch.allclose(traj1[-1], traj2[-1])


class TestDeviceHandling:
    """Test device handling for sampling functions."""

    def test_forward_cpu(self, gmm_model):
        schedule = BetaSchedule()
        x = gmm_model.sample(shape=100, t=0.0)[:, 0, :]
        t = torch.linspace(0.0, 1.0, 50)
        trajectory = forward_sampling(schedule.forward_drift, schedule.diffusion_coeff, x, t)
        assert trajectory.device.type == "cpu"
        assert trajectory.shape == (len(t), 100, 1)

    def test_forward_on_accelerator(self, gmm_model):
        device = get_local_device()
        schedule = BetaSchedule()
        gmm = gmm_model.to(device)
        x = gmm.sample(shape=100, t=0.0)[:, 0, :]
        t = torch.linspace(0.0, 1.0, 50, device=device)
        trajectory = forward_sampling(schedule.forward_drift, schedule.diffusion_coeff, x, t)
        assert trajectory.device == x.device
        assert trajectory.shape == (len(t), 100, 1)

    def test_reverse_cpu(self, gmm_model):
        schedule = BetaSchedule()
        x = torch.randn(100, 1, 1)
        t = torch.linspace(1.0, 0.0, 50)

        def drift(x_, t_):
            f = schedule.forward_drift(x_, t_)
            g = schedule.diffusion_coeff(t_)
            return f - g**2 * gmm_model.score(x_, t_)

        trajectory = reverse_sampling(drift, schedule.diffusion_coeff, x, t)
        assert trajectory.device.type == "cpu"
        assert trajectory.shape == (len(t), 100, 1, 1)

    def test_reverse_on_accelerator(self, gmm_model):
        device = get_local_device()
        schedule = BetaSchedule()
        gmm = gmm_model.to(device)
        x = torch.randn(100, 1, 1, device=device)
        t = torch.linspace(1.0, 0.0, 50, device=device)

        def drift(x_, t_):
            f = schedule.forward_drift(x_, t_)
            g = schedule.diffusion_coeff(t_)
            return f - g**2 * gmm.score(x_, t_)

        trajectory = reverse_sampling(drift, schedule.diffusion_coeff, x, t)
        assert trajectory.device == x.device
        assert trajectory.shape == (len(t), 100, 1, 1)
