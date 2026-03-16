from sympy.utilities.lambdify import implemented_function
import sys

import pytest
import torch

torch.set_printoptions(sci_mode=False)
from torchGMM.sampling import forward_sampling, reverse_sampling
from torchGMM.gmm import TimeDependentGMM
from torchGMM.schedule import BetaSchedule, LinearSchedule


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


def _histogram_setup():
    """51 evenly-spaced bins over [-5, 5]. Returns (x_grid [51,1,1], bin_edges [52])."""
    x_grid = torch.linspace(-5, 5, 51).reshape(-1, 1, 1)
    dx = x_grid[1, 0, 0] - x_grid[0, 0, 0]
    x_flat = x_grid.squeeze()
    bin_edges = torch.cat([(x_flat[0] - dx / 2).unsqueeze(0), x_flat + dx / 2])
    return x_grid, bin_edges


class TestForwardSampling:

    t_eps = 0.01

    @pytest.mark.parametrize(
        "schedule_cls, t_start, t_end",
        [
            (BetaSchedule, 0.01, 0.99),
            (LinearSchedule, 0.01, 0.99),
        ],
    )
    def test_forward_ode_marginals(self, schedule_cls, t_start, t_end):
        """Forward ODE histogram matches analytical GMM marginal at every sampled time step."""
        mu = torch.tensor([-2.0, 0.0, 2.0]).reshape(1, 3, 1)
        sigma = torch.tensor([0.3, 0.3, 0.2]).reshape(1, 3, 1)
        weight = torch.tensor([0.33, 0.5, 0.17]).reshape(1, 3)
        schedule = schedule_cls()
        gmm = TimeDependentGMM(mu=mu, sigma=sigma, weight=weight, schedule=schedule)

        n_samples, n_steps = 10_000, 200
        x = gmm.sample(shape=n_samples, t=t_start)  # [N, B=1, D=1]
        t = torch.linspace(t_start, t_end, n_steps)
        trajectory = forward_sampling(gmm.velocity, None, x, t)  # [T, N, B=1, D=1]

        assert trajectory.shape == (n_steps, n_samples, 1, 1)

        x_grid, bin_edges = _histogram_setup()
        for t_idx in range(n_steps)[::10]:
            t_ = t[t_idx]
            target = gmm.log_prob(x_grid, t=t_).exp().squeeze(-1)  # [nsteps]
            hist, _ = torch.histogram(trajectory[t_idx, :, 0, 0], bins=bin_edges, density=True)
            import matplotlib.pyplot as plt

            # plt.plot(x_grid[:, 0, 0], target, label="Target")
            # plt.hist(
            #     trajectory[t_idx, :, 0, 0].cpu(), bins=bin_edges.cpu(), density=True, alpha=0.5, label="Trajectory"
            # )
            # plt.title(f"{schedule_cls.__name__} forward ODE @ t={t_:.3f}")
            # plt.legend()
            # plt.ylim(0, 1)
            # plt.show()
            assert (hist - target).abs().max() < 0.05, (
                f"{schedule_cls.__name__} forward ODE @ t={t_:.3f}: " f"max deviation {(hist - target).abs().max():.3f}"
            )

    @pytest.mark.parametrize("schedule_cls", [BetaSchedule, LinearSchedule])
    @pytest.mark.parametrize("gamma", [0.1, 0.5, 1.0, 1.5], ids=lambda x: f"gamma={x}")
    def test_forward_sde_marginals(self, schedule_cls, gamma):
        """Forward SDE marginals match analytical GMM marginals at every 5 steps."""
        mu = torch.tensor([[-2.0, 0.0, 2.0], [-1.5, 0.5, 2.5]]).reshape(2, 3, 1)
        sigma = torch.tensor([[0.3, 0.3, 0.2], [0.2, 0.4, 0.3]]).reshape(2, 3, 1)
        weight = torch.tensor([[0.33, 0.5, 0.17], [0.25, 0.5, 0.25]]).reshape(2, 3)
        schedule = schedule_cls()
        gmm = TimeDependentGMM(mu=mu, sigma=sigma, weight=weight, schedule=schedule)

        n_samples, n_steps = 50_000, 100
        t = torch.linspace(self.t_eps, 1 - self.t_eps, n_steps)
        x = gmm.sample(shape=n_samples, t=self.t_eps)  # [N, B=2, D=1]

        if schedule_cls is BetaSchedule:

            def drift_fn(x_, t_):
                return schedule.forward_drift(x_, t_)

            def diffusion_fn(t_):
                return schedule.diffusion_coeff(t_)

        elif schedule_cls is LinearSchedule:

            def drift_fn(x_, t_):
                return gmm.velocity(x_, t_) + 0.5 * gamma**2 * gmm.score(x_, t_)

            def diffusion_fn(t_):
                return gamma

        trajectory = forward_sampling(drift_fn, diffusion_fn, x, t)  # [T, N, B=2, D=1]

        assert trajectory.shape == (len(t), n_samples, 2, 1)
        assert torch.allclose(trajectory[0], x, atol=1e-5)

        x_grid = torch.linspace(-5, 5, 51).reshape(-1, 1, 1).repeat(1, 2, 1)
        dx = x_grid[1, 0, 0] - x_grid[0, 0, 0]
        x_flat = x_grid[:, 0, 0].squeeze()
        bin_edges = torch.cat([(x_flat[0] - dx / 2).unsqueeze(0), x_flat + dx / 2])

        for t_idx in range(t.numel())[::5]:
            t_ = t[t_idx]
            target_dist = gmm.log_prob(x_grid, t=t_).exp().squeeze(-1)  # [51, 2]
            for dim in range(gmm.dim):
                trajectory_dim = trajectory[t_idx, :, dim, 0].unsqueeze(-2)
                hist, _ = torch.histogram(trajectory_dim, bins=bin_edges, density=True)
                assert target_dist.shape[:1] == hist.shape
                assert (hist - target_dist[:, dim]).abs().max() < 0.05, (
                    f"{schedule_cls.__name__} forward SDE gamma={gamma} @ t={t_:.3f} dim={dim}: "
                    f"max deviation {(hist - target_dist[:, dim]).abs().max():.3f}"
                )


class TestReverseSampling:
    """
    At t_start ≈ 1 − eps both schedules have σ_{t_start} ≈ 1 and α_{t_start} ≈ 0,
    so p_{t_start} ≈ N(0, I) and torch.randn is an accurate noise initialiser.
    This lets us check intermediate marginals at every 5 steps — the trajectory
    should stay on the analytical GMM marginal throughout the reverse integration.

    t ranges avoid singularities:
      BetaSchedule  : velocity has 1/σ_t singularity at t=0 → t_end = 0.01
      FlowMatching  : velocity has 1/(1−t) singularity at t=1 → t_start = 1−eps
                      marginal std < histogram bin width (0.2) for t < 0.1 → t_end = 0.1
    """

    eps = 1e-2  # offset from singularities shared by all reverse tests

    @pytest.mark.parametrize(
        "schedule_cls",
        [BetaSchedule, LinearSchedule],
    )
    def test_reverse_ode_marginals(self, schedule_cls):
        """Probability flow ODE: dx = v(x,t) dt. Histogram matches analytical GMM marginal every 5 steps."""
        mu = torch.tensor([-2.0, 0.0, 2.0]).reshape(1, 3, 1)
        sigma = torch.tensor([0.3, 0.3, 0.2]).reshape(1, 3, 1)
        weight = torch.tensor([0.33, 0.5, 0.17]).reshape(1, 3)
        schedule = schedule_cls()
        gmm = TimeDependentGMM(mu=mu, sigma=sigma, weight=weight, schedule=schedule)

        n_samples, n_steps = 10_000, 200
        t_start = 1.0 - self.eps  # p_{t_start} ≈ N(0, I) for both schedules
        x = torch.randn(n_samples, 1, 1)  # [N, B=1, D=1] — draws from p_{t_start} ≈ N(0,I)
        t = torch.linspace(1 - self.eps, self.eps, n_steps)
        if schedule_cls is BetaSchedule:
            # Anderson reverse SDE: dx = [f − g² score] dt + g dW. Histograms match analytical marginals every 5 steps.
            def drift_fn(x_, t_):
                f = schedule.forward_drift(x_, t_)
                g = schedule.diffusion_coeff(t_)
                return f - 1 / 2 * g**2 * gmm.score(x_, t_)

        elif schedule_cls is LinearSchedule:
            # Flow matching ODE: dx = v(x,t) dt. Histograms match analytical marginals every 5 steps.
            drift_fn = gmm.velocity

        trajectory = reverse_sampling(drift_fn, None, x, t)  # [T, N, B=1, D=1]

        assert trajectory.shape == (n_steps, n_samples, 1, 1)

        # ODE is deterministic — same input gives same output
        trajectory2 = reverse_sampling(drift_fn, None, x, t)
        torch.testing.assert_close(trajectory, trajectory2)

        x_grid, bin_edges = _histogram_setup()
        for t_idx in range(n_steps)[::5]:
            t_ = t[t_idx]
            target = gmm.log_prob(x_grid, t=t_).exp().squeeze(-1)  # [51]
            hist, _ = torch.histogram(trajectory[t_idx, :, 0, 0], bins=bin_edges, density=True)
            assert (
                hist - target
            ).abs().max() < 0.05, (
                f"{schedule_cls.__name__} ODE @ t={t_:.3f}: max deviation {(hist - target).abs().max():.3f}"
            )

    @pytest.mark.parametrize("schedule_cls", [BetaSchedule, LinearSchedule])
    @pytest.mark.parametrize("gamma", [0.1, 0.5, 1.0, 1.5], ids=lambda x: f"gamma={x}")
    def test_reverse_sde_marginals(self, schedule_cls, gamma):
        """BetaSchedule Anderson reverse SDE: dx = [f − g² score] dt + g dW. Histograms match analytical marginals every 5 steps."""
        mu = torch.tensor([[-2.0, 0.0, 2.0], [-1.5, 0.5, 2.5]]).reshape(2, 3, 1)
        sigma = torch.tensor([[0.3, 0.3, 0.2], [0.2, 0.4, 0.3]]).reshape(2, 3, 1)
        weight = torch.tensor([[0.33, 0.5, 0.17], [0.25, 0.5, 0.25]]).reshape(2, 3)
        schedule = schedule_cls()
        gmm = TimeDependentGMM(mu=mu, sigma=sigma, weight=weight, schedule=schedule)

        # BetaSchedule: no singularity at t=1. FlowMatching: 1/(1-t) singularity → start at 1-eps.
        x = torch.randn(50_000, 2, 1)
        t = torch.linspace(1 - self.eps, self.eps, 100)
        if schedule_cls is BetaSchedule:

            def drift_fn(x_, t_):
                f = schedule.forward_drift(x_, t_)
                g = schedule.diffusion_coeff(t_)
                return f - 1 / 2 * g**2 * (1 + gamma**2) * gmm.score(x_, t_)

            def diffusion_fn(t_):
                return gamma * schedule.diffusion_coeff(t_)

        elif schedule_cls is LinearSchedule:

            def drift_fn(x_, t_):
                return gmm.velocity(x_, t_) - 1 / 2 * gamma**2 * gmm.score(x_, t_)

            def diffusion_fn(t_):
                return gamma

        trajectory = reverse_sampling(drift_fn, diffusion_fn, x, t)

        assert trajectory.shape == (len(t), x.shape[0], 2, 1)
        assert torch.allclose(trajectory[0], x, atol=1e-5)

        x_grid = torch.linspace(-5, 5, 51).reshape(-1, 1, 1).repeat(1, 2, 1)
        dx = x_grid[1, 0, 0] - x_grid[0, 0, 0]
        x_flat = x_grid[:, 0, 0].squeeze()
        bin_edges = torch.cat([(x_flat[0] - dx / 2).unsqueeze(0), x_flat + dx / 2])

        for t_idx in range(t.numel())[::5]:
            t_ = t[t_idx]
            target_dist = gmm.log_prob(x_grid, t=t_).exp().squeeze(-1)
            for dim in range(gmm.dim):
                trajectory_dim = trajectory[t_idx, :, dim, 0].unsqueeze(-2)
                hist, _ = torch.histogram(trajectory_dim, bins=bin_edges, density=True)
                import matplotlib.pyplot as plt

                # plt.plot(x_grid[:, dim, 0], target_dist[:, dim], label="Target")
                # plt.plot(x_grid[:, dim, 0], hist, label="Trajectory")
                # plt.title(f"{schedule_cls.__name__} reverse SDE @ t={t_:.3f} dim={dim}")
                # plt.legend()
                # plt.ylim(0, 1)
                # plt.show()
                assert target_dist.shape[:1] == hist.shape
                assert (hist - target_dist[:, dim]).abs().max() < 0.05

    def test_t_must_be_decreasing(self):
        x = torch.randn(10, 2)
        t = torch.linspace(0.0, 1.0, 10)
        with pytest.raises(ValueError, match="strictly decreasing"):
            reverse_sampling(lambda x_, t_: x_, None, x, t)


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
        x = gmm_model.sample(shape=100, t=0.0)  # [100, 1, 1] — keep full [N, B, D] shape
        t = torch.linspace(0.0, 1.0, 50)
        trajectory = forward_sampling(schedule.forward_drift, schedule.diffusion_coeff, x, t)
        assert trajectory.device.type == "cpu"
        assert trajectory.shape == (len(t), 100, 1, 1)

    def test_forward_on_accelerator(self, gmm_model):
        device = get_local_device()
        schedule = BetaSchedule()
        gmm = gmm_model.to(device)
        x = gmm.sample(shape=100, t=0.0)  # [100, 1, 1] — keep full [N, B, D] shape
        t = torch.linspace(0.0, 1.0, 50, device=device)
        trajectory = forward_sampling(schedule.forward_drift, schedule.diffusion_coeff, x, t)
        assert trajectory.device == x.device
        assert trajectory.shape == (len(t), 100, 1, 1)

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
