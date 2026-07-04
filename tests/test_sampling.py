import os
from pathlib import Path

import pytest
import torch
from conftest import get_local_device

from torchGMM.gmm import GMM
from torchGMM.sampling import _ess_ratio, forward_sampling, reverse_sampling, steered_reverse_sampling
from torchGMM.schedule import BetaSchedule, KarrasSchedule, LinearSchedule

torch.set_printoptions(sci_mode=False)


@pytest.fixture
def gmm_model():
    """Fixture for the GMM model used in diffusion tests."""
    mu = torch.tensor([-2, 0, 2]).reshape(1, 3, 1)
    sigma = torch.tensor([0.3, 0.3, 0.2]).reshape(1, 3, 1)
    weight = torch.tensor([0.33, 0.5, 0.17]).reshape(1, 3)
    return GMM(mu=mu, sigma=sigma, weight=weight)


@pytest.fixture
def batched_gmm_model():
    """Fixture for the GMM model used in diffusion tests."""
    mu = torch.tensor([[-1, 0.5, 1.5], [-2, 0, 2]]).reshape(2, 3, 1)
    sigma = torch.tensor([[0.3, 0.3, 0.2], [0.3, 0.3, 0.2]]).reshape(2, 3, 1)
    weight = torch.tensor([[1 / 3, 1 / 2, 1 / 6], [0.33, 0.5, 0.17]]).reshape(2, 3)
    return GMM(mu=mu, sigma=sigma, weight=weight)


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
        torch.manual_seed(1)
        mu = torch.tensor([-2.0, 0.0, 2.0]).reshape(1, 3, 1)
        sigma = torch.tensor([0.3, 0.3, 0.2]).reshape(1, 3, 1)
        weight = torch.tensor([0.33, 0.5, 0.17]).reshape(1, 3)
        schedule = schedule_cls()
        gmm = GMM(mu=mu, sigma=sigma, weight=weight, schedule=schedule)

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
                f"{schedule_cls.__name__} forward ODE @ t={t_:.3f}: max deviation {(hist - target).abs().max():.3f}"
            )

    @pytest.mark.slow
    @pytest.mark.parametrize("schedule_cls", [BetaSchedule, LinearSchedule])
    @pytest.mark.parametrize("gamma", [0.1, 0.5, 1.0, 1.5], ids=lambda x: f"gamma={x}")
    def test_forward_sde_marginals(self, schedule_cls, gamma):
        """Forward SDE marginals match analytical GMM marginals at every 5 steps."""
        torch.manual_seed(0)
        mu = torch.tensor([[-2.0, 0.0, 2.0], [-1.5, 0.5, 2.5]]).reshape(2, 3, 1)
        sigma = torch.tensor([[0.3, 0.3, 0.2], [0.2, 0.4, 0.3]]).reshape(2, 3, 1)
        weight = torch.tensor([[0.33, 0.5, 0.17], [0.25, 0.5, 0.25]]).reshape(2, 3)
        schedule = schedule_cls()
        gmm = GMM(mu=mu, sigma=sigma, weight=weight, schedule=schedule)

        n_samples, n_steps = 50_000, 200
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
        torch.manual_seed(0)
        mu = torch.tensor([-2.0, 0.0, 2.0]).reshape(1, 3, 1)
        sigma = torch.tensor([0.3, 0.3, 0.2]).reshape(1, 3, 1)
        weight = torch.tensor([0.33, 0.5, 0.17]).reshape(1, 3)
        schedule = schedule_cls()
        gmm = GMM(mu=mu, sigma=sigma, weight=weight, schedule=schedule)

        n_samples, n_steps = 10_000, 200
        x = torch.randn(n_samples, 1, 1)  # [N, B=1, D=1] — draws from p_{t_start=1-eps} ≈ N(0,I)
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
            assert (hist - target).abs().max() < 0.05, (
                f"{schedule_cls.__name__} ODE @ t={t_:.3f}: max deviation {(hist - target).abs().max():.3f}"
            )

    @pytest.mark.slow
    @pytest.mark.parametrize("schedule_cls", [BetaSchedule, LinearSchedule])
    @pytest.mark.parametrize("gamma", [0.1, 0.5, 1.0, 1.5], ids=lambda x: f"gamma={x}")
    def test_reverse_sde_marginals(self, schedule_cls, gamma):
        """Anderson reverse SDE: dx = [f − g² score] dt + g dW.

        Histograms should match analytical marginals every 5 steps.
        """
        torch.manual_seed(0)
        mu = torch.tensor([[-2.0, 0.0, 2.0], [-1.5, 0.5, 2.5]]).reshape(2, 3, 1)
        sigma = torch.tensor([[0.3, 0.3, 0.2], [0.2, 0.4, 0.3]]).reshape(2, 3, 1)
        weight = torch.tensor([[0.33, 0.5, 0.17], [0.25, 0.5, 0.25]]).reshape(2, 3)
        schedule = schedule_cls()
        gmm = GMM(mu=mu, sigma=sigma, weight=weight, schedule=schedule)

        # BetaSchedule: no singularity at t=1. FlowMatching: 1/(1-t) singularity → start at 1-eps.
        x = torch.randn(50_000, 2, 1)
        t = torch.linspace(1 - self.eps, self.eps, 200)
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


class TestSteeredSamplingResampleModes:
    """Deterministic control-flow verification for the ess_threshold resampling modes.

    Harness: zero drift + diffusion=None means particle *values* never change except
    via resampling's index-selection (gather) — so every value in the returned
    trajectory must come from the original `x0` pool, in every mode. The weight_update
    returns a fixed per-slot bias independent of x/t, so with a constant-dt time grid
    the whole ESS trace is a deterministic function of "steps since last reset" —
    reproducible in the test via the same `_ess_ratio` helper (this tests the
    resample-timing control flow, not `_ess_ratio`'s own math). No `torch.manual_seed`
    is needed anywhere: every assertion holds regardless of `_systematic_resample`'s
    internal `torch.rand(1)` draw, so these tests can't flake under xdist.
    """

    N = 50
    N_STEPS = 21  # -> 20 integration steps
    K = 3.0

    @staticmethod
    def _zero_drift(x, t):
        return torch.zeros_like(x)

    def _weight_update(self, x, t, dt):
        bias = torch.linspace(-self.K, self.K, x.shape[0])
        return bias * dt.abs()

    def _predict_ess_history(self, t, trigger_fn):
        """Replay the same reset-on-trigger recursion the implementation runs.

        Mirrors the real loop's `dt = t_next - t_curr` scaling exactly, since
        `weight_update` multiplies the fixed bias by `dt.abs()` every step.
        """
        bias = torch.linspace(-self.K, self.K, self.N)
        log_w = torch.zeros(self.N)
        history = []
        for t_curr, t_next in zip(t[:-1], t[1:]):
            dt = t_next - t_curr
            log_w = log_w + bias * dt.abs()
            ess = _ess_ratio(log_w)
            history.append(ess)
            if trigger_fn(len(history) - 1, ess):
                log_w = torch.zeros(self.N)
        return history

    def test_adaptive_mode_matches_prediction(self):
        x0 = torch.randn(self.N, 1)
        t = torch.linspace(1.0 - 1e-3, 1e-3, self.N_STEPS)
        threshold = 0.7
        traj, ess_hist = steered_reverse_sampling(
            self._zero_drift, None, self._weight_update, x0, t, ess_threshold=threshold
        )
        predicted = self._predict_ess_history(t, lambda step, ess: ess < threshold)
        assert ess_hist == pytest.approx(predicted, abs=1e-6)
        assert torch.isin(traj, x0).all()

    def test_interval_mode_matches_prediction(self):
        x0 = torch.randn(self.N, 1)
        t = torch.linspace(1.0 - 1e-3, 1e-3, self.N_STEPS)  # 20 integration steps
        interval = 5
        traj, ess_hist = steered_reverse_sampling(
            self._zero_drift, None, self._weight_update, x0, t, ess_threshold=interval
        )
        # sanity check independent of the implementation's own formula: with 20 steps
        # and interval=5, triggers land at completed-step counts 5, 10, 15, 20.
        expected_trigger_steps = [4, 9, 14, 19]  # 0-indexed step at which (step+1) % 5 == 0
        assert [s for s in range(20) if (s + 1) % interval == 0] == expected_trigger_steps

        predicted = self._predict_ess_history(t, lambda step, ess: (step + 1) % interval == 0)
        assert ess_hist == pytest.approx(predicted, abs=1e-6)
        assert len(ess_hist) == self.N_STEPS - 1
        assert torch.isin(traj, x0).all()

    def test_interval_larger_than_steps_is_final_only(self):
        x0 = torch.randn(self.N, 1)
        t = torch.linspace(1.0 - 1e-3, 1e-3, self.N_STEPS)  # 20 integration steps
        traj, ess_hist = steered_reverse_sampling(
            self._zero_drift, None, self._weight_update, x0, t, ess_threshold=10_000
        )
        # interval >> total steps -> no intermittent trigger ever fires
        predicted = self._predict_ess_history(t, lambda step, ess: False)
        assert ess_hist == pytest.approx(predicted, abs=1e-6)
        assert torch.isin(traj, x0).all()

    def test_ess_equal_one_resamples_every_step(self):
        x0 = torch.randn(self.N, 1)
        t = torch.linspace(1.0 - 1e-3, 1e-3, self.N_STEPS)
        traj, ess_hist = steered_reverse_sampling(self._zero_drift, None, self._weight_update, x0, t, ess_threshold=1)
        # ess_threshold == 1 selects interval mode with resample_every=1 (every step),
        # not adaptive mode (which would resample whenever ess < 1, i.e. also nearly
        # every step here -- so cross-check against the interval prediction specifically).
        predicted = self._predict_ess_history(t, lambda step, ess: (step + 1) % 1 == 0)
        assert ess_hist == pytest.approx(predicted, abs=1e-6)
        assert torch.isin(traj, x0).all()

    @pytest.mark.parametrize("bad_threshold", [0, -1, -0.5])
    def test_non_positive_threshold_rejected(self, bad_threshold):
        x0 = torch.randn(4, 1)
        t = torch.linspace(1.0 - 1e-3, 1e-3, 5)
        with pytest.raises(ValueError, match="positive"):
            steered_reverse_sampling(self._zero_drift, None, self._weight_update, x0, t, ess_threshold=bad_threshold)

    @pytest.mark.parametrize("bad_threshold", [2.5, 1.5, 10.25])
    def test_non_integer_interval_threshold_rejected(self, bad_threshold):
        x0 = torch.randn(4, 1)
        t = torch.linspace(1.0 - 1e-3, 1e-3, 5)
        with pytest.raises(ValueError, match="whole number"):
            steered_reverse_sampling(self._zero_drift, None, self._weight_update, x0, t, ess_threshold=bad_threshold)


@pytest.mark.slow
class TestSteeredSampling:
    """SMC-steered reverse sampling via FKC weight update."""

    EPS = 0.001
    N_PARTICLES = 20_000
    N_STEPS = 500

    @pytest.fixture
    def setup(self):
        sched = BetaSchedule(beta_min=0.1, beta_max=20.0)
        gmm = GMM(
            mu=torch.tensor([[[-2.5], [2.5]]]),
            sigma=torch.tensor([[[0.8], [0.8]]]),
            weight=torch.tensor([[0.2, 0.8]]),
        )
        return gmm, sched

    @pytest.mark.parametrize(
        "reward_center,reward_sigma,ess_threshold",
        [
            (-2.0, 1.0, 0.95),
            (-1.5, 1.0, 0.95),
            (-1.5, 1.5, 0.95),
            (-1.0, 1.0, 0.95),
            (-1.0, 1.5, 0.95),
            (-1.5, 1.0, 25),  # fixed-interval: resample every 25 of the 499 integration steps
            (-1.5, 1.0, 100_000),  # final-only: interval far exceeds 499 integration steps
        ],
        ids=lambda v: f"{v}",
    )
    def test_steered_sampling(self, setup, reward_center, reward_sigma, ess_threshold):
        gmm, sched = setup

        def r(x):
            return -0.5 * (x - reward_center) ** 2 / reward_sigma**2

        def grad_r(x):
            return -(x - reward_center) / reward_sigma**2

        def guided_drift(x, t):
            f = sched.forward_drift(x, t)
            sigma = sched.diffusion_coeff(t)
            score = gmm.score(x, t)
            beta = 1.0 - t
            return f - sigma**2 * score - beta * (sigma**2 / 2) * grad_r(x)

        def fkc_weight_update(x, t, dt):
            f = sched.forward_drift(x, t)
            sigma = sched.diffusion_coeff(t)
            score = gmm.score(x, t)
            rg, rv = grad_r(x), r(x)
            beta = 1.0 - t
            term1 = rv
            term2 = -(beta * rg) * f
            term3 = (beta * rg) * (sigma**2 / 2) * score
            return (term1 + term2 + term3).squeeze(-1).squeeze(-1) * dt.abs()

        t = torch.linspace(1 - self.EPS, self.EPS, self.N_STEPS)
        x0 = torch.randn(self.N_PARTICLES, 1, 1)
        traj, ess_hist = steered_reverse_sampling(
            guided_drift, sched.diffusion_coeff, fkc_weight_update, x0, t, ess_threshold=ess_threshold
        )

        # 1. Trajectory shape
        assert traj.shape == (self.N_STEPS, self.N_PARTICLES, 1, 1)

        # 2. ESS/N history within [0, 1] at every step
        assert len(ess_hist) == self.N_STEPS - 1
        for ess in ess_hist:
            assert 0.0 <= ess <= 1.0

        # Ground truth: reward-tilted density
        xs = torch.linspace(-6, 6, 500).reshape(-1, 1, 1)
        log_p_data = gmm.log_prob(xs, t=self.EPS).squeeze()
        log_p_rew = log_p_data + (1.0 - self.EPS) * r(xs).squeeze()
        log_p_rew = log_p_rew - log_p_rew.max()
        p_rew = log_p_rew.exp()
        p_rew = p_rew / torch.trapezoid(p_rew, xs.squeeze())

        p_data = log_p_data.exp()
        p_data = p_data / torch.trapezoid(p_data, xs.squeeze())

        # Build histogram of final samples
        x_final = traj[-1, :, 0, 0]
        xs_flat = xs.squeeze()
        dx = xs_flat[1] - xs_flat[0]
        bin_edges = torch.cat([(xs_flat[0] - dx / 2).unsqueeze(0), xs_flat + dx / 2])
        hist, _ = torch.histogram(x_final, bins=bin_edges, density=True)

        # 4. L2 vs reward-tilted target < 0.1
        l2_rew = torch.sqrt(((hist - p_rew) ** 2).mean()).item()
        assert l2_rew < 0.1, f"center={reward_center} sigma={reward_sigma}: L2 vs reward-tilted={l2_rew:.4f}"

        # 5. Steered samples closer to reward-tilted target than unguided
        l2_data = torch.sqrt(((hist - p_data) ** 2).mean()).item()
        assert l2_rew < l2_data, (
            f"center={reward_center} sigma={reward_sigma}: L2 reward={l2_rew:.4f} should be < L2 unguided={l2_data:.4f}"
        )

        plot_dir = os.environ.get("TORCHGMM_PLOT_DIR")
        if plot_dir:
            plot_steering_result(
                traj=traj,
                t=t,
                ess_hist=ess_hist,
                xs_flat=xs_flat,
                p_data=p_data,
                p_rew=p_rew,
                hist=hist,
                reward_center=reward_center,
                reward_sigma=reward_sigma,
                ess_threshold=ess_threshold,
                n_steps=self.N_STEPS,
                out_path=Path(plot_dir) / f"steered_center{reward_center}_sigma{reward_sigma}_ess{ess_threshold}.png",
            )


@pytest.mark.slow
class TestSteeredSamplingKarras:
    """SMC-steered reverse sampling via FKC weight update, under the (variance-exploding)
    KarrasSchedule instead of BetaSchedule — same resampling-mode coverage as
    TestSteeredSampling, but exercising a schedule with forward_drift ≡ 0 and a
    non-affine, ρ-warped σ(t).

    At the AF3-default terminal variance (sigma_max=160), the diffusion coefficient
    g(t)² explodes near t≈1 (ρ=7 concentrates curvature there), so tilting the drift
    directly on the noisy x_t (as TestSteeredSampling does) blows up numerically —
    beta(t)*(g(t)²/2)*grad_r(x_t) diverges even though beta(t)→0 there, because g(t)²
    grows faster than beta(t) shrinks. Instead this mirrors
    notebooks/karras_terminal_variance_steering.py: an unrolled ODE/Tweedie denoiser
    estimates x̂_0 = D(x_t; σ(t)) (bounded, on the GMM's data manifold), the reward is
    evaluated on x̂_0, and its gradient is backpropagated through the denoiser to build
    guided_drift/weight_update — the standard FKC pattern for reward models defined on
    clean data rather than noisy latents.

    The tilt schedule beta(t) also needed retuning at this scale: a linear/quadratic
    ramp still leaves the tilt too large while g(t)² is enormous (t≈1), inflating
    importance-weight variance — worst for final-only resampling, which has no
    intermediate correction to absorb it. A sweep over candidate beta(t) shapes,
    scored by W1 (Wasserstein-1) distance to the analytic reward-tilted target, found
    a steeper sextic ramp (1-t)^6 ~14x better than (1-t)^2 (mean W1 0.019 vs 0.27
    across the three modes), which is what test_steered_sampling_karras uses below.
    """

    EPS = 0.001
    T_NOISE = 1 - EPS
    N_PARTICLES = 20_000
    N_STEPS = 500
    N_DENOISE_STEPS = 10

    @pytest.fixture
    def setup(self):
        sched = KarrasSchedule(sigma_min=4e-4, sigma_max=160.0, rho=7.0, sigma_data=1.0)
        gmm = GMM(
            mu=torch.tensor([[[-2.5], [2.5]]]),
            sigma=torch.tensor([[[0.8], [0.8]]]),
            weight=torch.tensor([[0.2, 0.8]]),
            schedule=sched,
        )
        return gmm, sched

    def _denoise(self, gmm, sched, x_t, t):
        """Tweedie-blend ODE denoiser: integrates the probability-flow ODE from
        sigma(t) down to sigma(EPS) in N_DENOISE_STEPS Euler substeps, returning a
        bounded x̂_0 estimate plus leaf tensors for backprop (see
        notebooks/karras_terminal_variance_steering.py:denoise for the derivation).
        """
        x_t_leaf = x_t.detach().requires_grad_(True)
        t_leaf = torch.as_tensor(t, dtype=x_t.dtype, device=x_t.device).clone().detach().requires_grad_(True)
        with torch.enable_grad():
            t_min = torch.as_tensor(self.EPS, dtype=x_t.dtype, device=x_t.device)
            x = x_t_leaf
            t_prev = t_leaf
            sigma_prev = sched.get_sigma_t(t_prev)
            for step_k in range(1, self.N_DENOISE_STEPS + 1):
                t_next = t_leaf + (t_min - t_leaf) * (step_k / self.N_DENOISE_STEPS)
                sigma_next = sched.get_sigma_t(t_next)
                D = x + sigma_prev**2 * gmm.score(x, t_prev)
                ratio = sigma_next / sigma_prev
                x = ratio * x + (1.0 - ratio) * D
                t_prev = t_next
                sigma_prev = sigma_next
            x0 = x
        return x0, x_t_leaf, t_leaf

    @pytest.mark.parametrize(
        "ess_threshold",
        [
            0.95,  # adaptive
            25,  # fixed-interval: resample every 25 of the 499 integration steps
            100_000,  # final-only: interval far exceeds 499 integration steps
        ],
        ids=lambda v: f"{v}",
    )
    def test_steered_sampling_karras(self, setup, ess_threshold):
        gmm, sched = setup
        reward_center, reward_sigma = -1.5, 1.0

        def r(x0_hat):
            return -0.5 * (x0_hat - reward_center) ** 2 / reward_sigma**2

        def beta_fn(t):
            # Sextic ramp: at sigma_max=160 the Karras diffusion coeff g(t)^2 explodes
            # near t=1 faster than a linear/quadratic beta(t) can suppress it (see the
            # class docstring), causing large importance-weight variance. A sweep over
            # candidate tilt shapes (linear/quadratic/cubic/quartic/.../cosine variants,
            # scored by W1 distance to the analytic reward-tilted target) found (1-t)^6
            # ~14x better than the quadratic ramp used at smaller sigma_max, consistently
            # across all three resampling modes.
            return (1 - t) ** 6

        def dbeta_dt(t):
            return -6 * (1 - t) ** 5

        def reward_and_grads(x, t):
            x0_hat, x_leaf, t_leaf = self._denoise(gmm, sched, x, t)
            rv = r(x0_hat)
            grad_x, grad_t = torch.autograd.grad(rv.sum(), (x_leaf, t_leaf))
            score = gmm.score(x, t).detach()
            return rv.detach(), grad_x.detach(), grad_t.detach(), score

        def guided_drift(x, t):
            g = sched.diffusion_coeff(t)
            beta = beta_fn(t)
            _, grad_x, _, score = reward_and_grads(x, t)
            return -(g**2) * score - beta * (g**2 / 2) * grad_x

        def fkc_weight_update(x, t, dt):
            g = sched.diffusion_coeff(t)
            beta = beta_fn(t)
            dbeta = dbeta_dt(t)
            rv, grad_x, grad_t, score = reward_and_grads(x, t)
            integrand = -dbeta * rv - beta * grad_t + beta * grad_x * (g**2 / 2) * score
            return integrand.squeeze(-1).squeeze(-1) * dt.abs()

        torch.manual_seed(0)
        t = torch.linspace(self.T_NOISE, self.EPS, self.N_STEPS)
        x0 = gmm.sample(shape=self.N_PARTICLES, t=self.T_NOISE)
        traj, ess_hist = steered_reverse_sampling(
            guided_drift, sched.diffusion_coeff, fkc_weight_update, x0, t, ess_threshold=ess_threshold
        )

        # 1. Trajectory shape
        assert traj.shape == (self.N_STEPS, self.N_PARTICLES, 1, 1)

        # 2. ESS/N history within [0, 1] at every step (small float32 tolerance: with
        # near-uniform weights, logsumexp-based ESS can round fractionally above 1.0)
        assert len(ess_hist) == self.N_STEPS - 1
        for ess in ess_hist:
            assert -1e-6 <= ess <= 1.0 + 1e-4

        # Ground truth: reward-tilted density. beta_fn(EPS) ≈ 1 (data time), matching
        # the tilt weight the sampler itself applies at the end of the reverse pass.
        xs = torch.linspace(-6, 6, 500).reshape(-1, 1, 1)
        log_p_data = gmm.log_prob(xs, t=self.EPS).squeeze()
        log_p_rew = log_p_data + beta_fn(torch.tensor(self.EPS)) * r(xs).squeeze()
        log_p_rew = log_p_rew - log_p_rew.max()
        p_rew = log_p_rew.exp()
        p_rew = p_rew / torch.trapezoid(p_rew, xs.squeeze())

        p_data = log_p_data.exp()
        p_data = p_data / torch.trapezoid(p_data, xs.squeeze())

        # Build histogram of final samples
        x_final = traj[-1, :, 0, 0]
        xs_flat = xs.squeeze()
        dx = xs_flat[1] - xs_flat[0]
        bin_edges = torch.cat([(xs_flat[0] - dx / 2).unsqueeze(0), xs_flat + dx / 2])
        hist, _ = torch.histogram(x_final, bins=bin_edges, density=True)

        # 4. L2 vs reward-tilted target < 0.1. The sextic beta_fn (see above) keeps this
        # comfortably under 0.02 in all three modes, including final-only — no special
        # casing needed once the tilt is suppressed correctly for this sigma_max.
        l2_rew = torch.sqrt(((hist - p_rew) ** 2).mean()).item()
        assert l2_rew < 0.1, f"KarrasSchedule ess_threshold={ess_threshold}: L2 vs reward-tilted={l2_rew:.4f}"

        # 5. Steered samples closer to reward-tilted target than unguided
        l2_data = torch.sqrt(((hist - p_data) ** 2).mean()).item()
        assert l2_rew < l2_data, (
            f"KarrasSchedule ess_threshold={ess_threshold}: "
            f"L2 reward={l2_rew:.4f} should be < L2 unguided={l2_data:.4f}"
        )

        plot_dir = os.environ.get("TORCHGMM_PLOT_DIR")
        if plot_dir:
            plot_steering_result(
                traj=traj,
                t=t,
                ess_hist=ess_hist,
                xs_flat=xs_flat,
                p_data=p_data,
                p_rew=p_rew,
                hist=hist,
                reward_center=reward_center,
                reward_sigma=reward_sigma,
                ess_threshold=ess_threshold,
                n_steps=self.N_STEPS,
                out_path=Path(plot_dir) / f"steered_karras_ess{ess_threshold}.png",
            )


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


def _resample_mode_label(ess_threshold, n_steps):
    """Human-readable label for the ess_threshold resampling mode (mirrors sampling.py semantics)."""
    if ess_threshold < 1:
        return f"adaptive (ESS threshold={ess_threshold})"
    if ess_threshold >= n_steps - 1:
        return f"final-only (interval={int(ess_threshold)} ≥ {n_steps - 1} steps)"
    return f"fixed-interval (every {int(ess_threshold)} steps)"


def _count_resamples(ess_threshold, ess_hist, n_steps):
    """Number of intermittent resamples implied by ess_threshold (mirrors the
    should_resample logic in steered_reverse_sampling; doesn't count the mandatory
    final resample after the loop, which isn't reflected in ess_hist)."""
    if ess_threshold < 1:
        return sum(1 for e in ess_hist if e < ess_threshold)
    resample_every = int(ess_threshold)
    return sum(1 for step in range(n_steps - 1) if (step + 1) % resample_every == 0)


def plot_steering_result(
    traj, t, ess_hist, xs_flat, p_data, p_rew, hist, reward_center, reward_sigma, ess_threshold, n_steps, out_path
):
    """Debug/inspection plot for FKC-steered reverse sampling, inspired by
    ``notebooks/karras_terminal_variance_steering.py``'s ``plot_run``: a trajectory
    spaghetti-plot of the steered particles, the true (unguided) density, the true
    reward-tilted target density and the steered samples' empirical density, and the
    ESS/N history over the reverse pass.

    Args:
        traj:           [T, N, *rest, D] steered trajectory (as returned by
                        steered_reverse_sampling)
        t:              [T] reverse-time grid used for the trajectory's x-axis
        ess_hist:       ESS/N history, len = T - 1 (one entry per integration step)
        xs_flat:        [G] evaluation grid for the analytic densities
        p_data:         [G] analytic unguided GMM density on xs_flat
        p_rew:          [G] analytic reward-tilted target density on xs_flat
        hist:           [G] empirical density histogram of the final steered samples
        reward_center:  center of the Gaussian reward potential (for the title/marker)
        reward_sigma:   width of the Gaussian reward potential (for the title)
        ess_threshold:  the resampling-mode parameter used for this run (for the title)
        n_steps:        total number of trajectory time points (len(t))
        out_path:       where to save the figure (parent dir created if missing)
    """
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt

    n_particles = traj.shape[1]
    n_plot = min(400, n_particles)
    idx_plot = torch.randperm(n_particles)[:n_plot]

    fig = plt.figure(figsize=(18, 5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1.15, 1.15], wspace=0.3)
    ax_traj = fig.add_subplot(gs[0])
    ax_dens = fig.add_subplot(gs[1])
    ax_ess = fig.add_subplot(gs[2])

    ax_traj.plot(t.cpu().numpy(), traj[:, idx_plot, 0, 0].cpu().numpy(), color="darkorange", alpha=0.06, lw=0.5)
    ax_traj.axhline(reward_center, color="firebrick", ls="--", lw=1.2, label=f"reward center={reward_center}")
    ax_traj.set_xlabel("t")
    ax_traj.set_ylabel("x")
    ax_traj.set_title(r"FKC-steered reverse trajectories $\leftarrow$")
    ax_traj.legend(loc="upper left", fontsize=9)

    bin_w = xs_flat[1] - xs_flat[0]
    bin_edges = torch.cat([xs_flat[:1] - bin_w / 2, xs_flat + bin_w / 2])
    ax_dens.plot(xs_flat.cpu(), p_data.cpu(), label="True unguided density $p$", color="steelblue", linewidth=2)
    ax_dens.plot(
        xs_flat.cpu(), p_rew.cpu(), label=r"True tilted density $p \cdot e^{r}$", color="firebrick", linewidth=2
    )
    ax_dens.stairs(hist.cpu(), bin_edges.cpu(), label="Steered sample density", fill=True, alpha=0.4, color="seagreen")
    ax_dens.axvline(reward_center, color="firebrick", linestyle="--", linewidth=1)
    ax_dens.set_xlabel("x")
    ax_dens.set_ylabel("density")
    ax_dens.set_title("Final-time marginal vs. analytic targets")
    ax_dens.legend(fontsize=9)

    n_resamples = _count_resamples(ess_threshold, ess_hist, n_steps)
    t_ess = t[:-1].cpu().numpy()  # ess_hist has one entry per completed step, len(t) - 1
    ax_ess.plot(t_ess, ess_hist, color="darkorange", lw=1.2, label="ESS / N")
    if ess_threshold < 1:
        ax_ess.axhline(ess_threshold, color="red", ls="--", lw=1, label=f"threshold={ess_threshold}")
    ax_ess.set_ylim(0, 1.05)
    ax_ess.set_xlabel("t")
    ax_ess.set_ylabel("ESS / N")
    ax_ess.set_title(f"ESS history ({n_resamples} intermittent resamples)")
    ax_ess.legend(fontsize=9)

    mode_label = _resample_mode_label(ess_threshold, n_steps)
    fig.suptitle(
        f"FKC steering — reward_center={reward_center}, reward_sigma={reward_sigma} — resampling: {mode_label}",
        fontsize=12,
    )
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
