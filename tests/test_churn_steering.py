"""Continuous FKC steering tests for `steered_reverse_churn_sampling`.

The churn sampler evaluates the same FKC callback as the Euler-Maruyama sampler at its
reheated state and time. The tests verify callback semantics, resampling control flow, and
weighted tilted marginals for guided probability-flow transport.
"""

import pytest
import torch
from test_steering import (
    _dbeta_dt,
    _plot_dir,
    _tilted_density,
    _wasserstein1,
    _weighted_wasserstein1,
    plot_marginal_density_comparison,
)

from torchGMM.gmm import GMM
from torchGMM.schedule import BetaSchedule, KarrasSchedule
from torchGMM.steering import _ess_ratio, steered_reverse_churn_sampling

torch.set_printoptions(sci_mode=False)

PLOT = True


class TestChurnSteeringProperties:
    """Resampling control flow, verified without a Monte-Carlo statistic.

    With zero velocity and an identity transition, particle values change only through
    resampling. A zero churn value permanently disables that resampling.
    The FKC callback assigns each fixed particle a deterministic log-weight rate, so its
    solver-scaled ESS trace and reset behavior are predictable for every resampling draw.
    """

    N = 50
    N_STEPS = 21
    K = 3.0

    @staticmethod
    def _zero_velocity(x, t):
        return torch.zeros_like(x)

    @staticmethod
    def _no_transition(x, t, s):
        raise AssertionError("churn=0 must not call the transition kernel")

    @staticmethod
    def _identity_transition(x, t, s):
        return x

    @staticmethod
    def _weight_update(x, t):
        return x.square().squeeze(-1)

    def test_shapes_and_normalisation(self):
        x0 = torch.linspace(-self.K, self.K, self.N).reshape(self.N, 1)
        t = torch.linspace(1.0 - 1e-3, 1e-3, self.N_STEPS)
        traj, ess_hist, weight_hist = steered_reverse_churn_sampling(
            self._zero_velocity,
            self._no_transition,
            self._weight_update,
            x0,
            t,
            churn=0.0,
            ess_threshold=0.7,
        )
        assert traj.shape == (self.N_STEPS, self.N, 1)
        assert weight_hist.shape == (self.N_STEPS, self.N)
        assert len(ess_hist) == self.N_STEPS - 1
        assert torch.allclose(weight_hist.sum(dim=1), torch.ones(self.N_STEPS), atol=1e-6)
        assert torch.isin(traj, x0).all()

    def test_weight_update_is_required(self):
        x0 = torch.tensor([[-1.0], [0.0], [1.0]])
        t = torch.tensor([0.8, 0.6, 0.4])
        with pytest.raises(ValueError, match="weight_update must be provided"):
            steered_reverse_churn_sampling(self._zero_velocity, self._no_transition, None, x0, t, churn=0.0)

    def test_weight_update_uses_the_reheated_state(self):
        x0 = torch.tensor([[-1.0], [0.0], [1.0]])
        t = torch.tensor([0.8, 0.6, 0.4])
        spans = []

        def transition(x, t_curr, t_hat):
            return x + (t_hat - t_curr)

        def fkc_weight_update(x, t_hat):
            spans.append(t_hat.item())
            return -x.squeeze(-1)

        _, _, weight_history = steered_reverse_churn_sampling(
            self._zero_velocity,
            transition,
            fkc_weight_update,
            x0,
            t,
            churn=0.5,
            ess_threshold=1_000,
        )

        assert spans == pytest.approx([0.9, 0.7])
        x_hat = x0 + 0.1
        expected_weights = torch.softmax(x_hat.squeeze(-1) * -0.2, dim=0)
        torch.testing.assert_close(weight_history[1], expected_weights)

    def test_transition_takes_precedence_over_diffusion(self):
        x0 = torch.zeros(3, 1)
        t = torch.tensor([0.8, 0.6])

        def transition(x, t_curr, t_hat):
            return x + (t_hat - t_curr)

        def diffusion(_):
            raise AssertionError("diffusion must not run when transition is supplied")

        trajectory, _, _ = steered_reverse_churn_sampling(
            drift=self._zero_velocity,
            transition=transition,
            diffusion=diffusion,
            weight_update=lambda x, t: torch.zeros(x.shape[0]),
            x=x0,
            t=t,
            churn=0.5,
            ess_threshold=1_000,
        )

        torch.testing.assert_close(trajectory[1], x0 + 0.1)

    def test_ess_trace_matches_analytic_prediction_before_first_resample(self):
        x0 = torch.linspace(-self.K, self.K, self.N).reshape(self.N, 1)
        t = torch.linspace(1.0 - 1e-3, 1e-3, self.N_STEPS)
        _, ess_hist, _ = steered_reverse_churn_sampling(
            self._zero_velocity,
            self._no_transition,
            self._weight_update,
            x0,
            t,
            churn=0.0,
            ess_threshold=1_000,
        )
        energy = torch.linspace(-self.K, self.K, self.N).square()
        log_w = torch.zeros(self.N)
        predicted = []
        for t_curr, t_next in zip(t[:-1], t[1:]):
            log_w = log_w + energy * (t_curr - t_next)
            predicted.append(_ess_ratio(log_w))
        assert ess_hist == pytest.approx(predicted, abs=1e-6)

    def test_interval_mode_resets_weights_on_schedule(self):
        interval = 5
        x0 = torch.linspace(-self.K, self.K, self.N).reshape(self.N, 1)
        t = torch.linspace(1.0 - 1e-3, 1e-3, self.N_STEPS)
        _, _, weight_hist = steered_reverse_churn_sampling(
            self._zero_velocity,
            self._identity_transition,
            self._weight_update,
            x0,
            t,
            churn=1.0,
            ess_threshold=interval,
        )
        uniform = torch.full((self.N,), 1 / self.N)
        for step in range(1, self.N_STEPS):
            expect_uniform = step % interval == 0 or step == self.N_STEPS - 1
            is_uniform = torch.allclose(weight_hist[step], uniform, atol=1e-6)
            assert is_uniform == expect_uniform, f"step {step}: uniform={is_uniform}, expected {expect_uniform}"

    def test_threshold_one_resamples_every_step(self):
        x0 = torch.linspace(-self.K, self.K, self.N).reshape(self.N, 1)
        t = torch.linspace(1.0 - 1e-3, 1e-3, self.N_STEPS)
        _, _, weight_hist = steered_reverse_churn_sampling(
            self._zero_velocity,
            self._identity_transition,
            self._weight_update,
            x0,
            t,
            churn=1.0,
            ess_threshold=1,
        )
        uniform = torch.full((self.N,), 1 / self.N)
        assert torch.allclose(weight_hist, uniform.expand_as(weight_hist), atol=1e-6)

    def test_final_resample_fires_while_churn_remains_positive(self):
        x0 = torch.linspace(-self.K, self.K, self.N).reshape(self.N, 1)
        t = torch.linspace(1.0 - 1e-3, 1e-3, self.N_STEPS)
        _, _, weight_hist = steered_reverse_churn_sampling(
            self._zero_velocity,
            self._identity_transition,
            self._weight_update,
            x0,
            t,
            churn=1.0,
            ess_threshold=1_000,
        )
        uniform = torch.full((self.N,), 1 / self.N)
        assert not torch.allclose(weight_hist[-2], uniform, atol=1e-6)
        assert torch.allclose(weight_hist[-1], uniform, atol=1e-6)

    def test_zero_churn_permanently_disables_resampling_and_accumulates_weights(self):
        x0 = torch.linspace(-self.K, self.K, self.N).reshape(self.N, 1)
        t = torch.tensor([0.9, 0.7, 0.5, 0.3])

        def churn(t_curr):
            return float(t_curr > 0.7)

        trajectory, _, weight_hist = steered_reverse_churn_sampling(
            self._zero_velocity,
            self._identity_transition,
            self._weight_update,
            x0,
            t,
            churn=churn,
            ess_threshold=1,
        )

        uniform = torch.full((self.N,), 1 / self.N)
        energy = trajectory[1].squeeze(-1).square()
        assert torch.allclose(weight_hist[1], uniform, atol=1e-6)
        torch.testing.assert_close(weight_hist[2], torch.softmax(energy * 0.2, dim=0))
        torch.testing.assert_close(weight_hist[-1], torch.softmax(energy * 0.4, dim=0))

    @pytest.mark.parametrize("bad_threshold", [0, -1, -0.5])
    def test_non_positive_threshold_rejected(self, bad_threshold):
        x0 = torch.linspace(-self.K, self.K, self.N).reshape(self.N, 1)
        t = torch.linspace(1.0 - 1e-3, 1e-3, self.N_STEPS)
        with pytest.raises(ValueError, match="positive"):
            steered_reverse_churn_sampling(
                drift=self._zero_velocity,
                diffusion=self._no_transition,
                weight_update=self._weight_update,
                x=x0,
                t=t,
                churn=0.0,
                ess_threshold=bad_threshold,
            )

    @pytest.mark.parametrize("bad_threshold", [2.5, 1.5, 10.25])
    def test_non_integer_interval_threshold_rejected(self, bad_threshold):
        x0 = torch.linspace(-self.K, self.K, self.N).reshape(self.N, 1)
        t = torch.linspace(1.0 - 1e-3, 1e-3, self.N_STEPS)
        with pytest.raises(ValueError, match="whole number"):
            steered_reverse_churn_sampling(
                self._zero_velocity,
                self._no_transition,
                self._weight_update,
                x0,
                t,
                churn=0.0,
                ess_threshold=bad_threshold,
            )

    def test_t_must_be_decreasing(self):
        x0 = torch.linspace(-self.K, self.K, self.N).reshape(self.N, 1)
        t = torch.linspace(1e-3, 1.0 - 1e-3, self.N_STEPS)
        with pytest.raises(ValueError, match="strictly decreasing"):
            steered_reverse_churn_sampling(
                self._zero_velocity, self._no_transition, self._weight_update, x0, t, churn=0.0
            )

    def test_negative_churn_rejected(self):
        x0 = torch.linspace(-self.K, self.K, self.N).reshape(self.N, 1)
        t = torch.linspace(1.0 - 1e-3, 1e-3, self.N_STEPS)
        with pytest.raises(ValueError, match="non-negative"):
            steered_reverse_churn_sampling(
                self._zero_velocity, self._no_transition, self._weight_update, x0, t, churn=-0.5
            )


@pytest.mark.slow
class TestSteeredChurnSampler:
    """Karras churn with FKC-guided probability-flow transport.

    Churn supplies the effective diffusion and therefore fixes the guidance displacement.
    It carries no potential-ratio weight; the bounded-variation FKC update compensates the
    guided deterministic leg.
    """

    EPS = 0.001
    T_NOISE = 1 - EPS
    N_PARTICLES = 20_000
    N_STEPS = 1000
    REWARD_SIGMA = 0.5

    @pytest.fixture
    def setup(self):
        sched = KarrasSchedule()
        gmm = GMM(
            mu=torch.tensor([[[-1.0], [1.0]]]),
            sigma=torch.tensor([[[0.4], [0.4]]]),
            weight=torch.tensor([[0.3, 0.7]]),
            schedule=sched,
        )
        return gmm, sched

    @pytest.mark.parametrize("churn", [0.5, 2.0], ids=lambda v: f"churn={v}")
    @pytest.mark.parametrize("reward_center", [-3, 1.0], ids=lambda v: f"center={v}")
    @pytest.mark.parametrize("ess_threshold", [0.9, 2000], ids=lambda v: f"ess={v}")
    def test_guided_steered_karras_churn_sampler_intermediate_marginals(
        self, setup, reward_center, churn, ess_threshold
    ):
        gmm, sched = setup

        def energy(x):
            return (0.5 * (x - reward_center) ** 2 / self.REWARD_SIGMA**2).squeeze(-1).squeeze(-1)

        def beta_fn(t):
            bar_sigma_t = sched.get_sigma_t(t)
            return (1 - t) / (1 + 0.1 * bar_sigma_t**2)

        def plot_energy(x, t):
            return energy(x)

        def grad_energy(x):
            return (x - reward_center) / self.REWARD_SIGMA**2

        def guided_drift(x, t_hat):
            # guidance needs the 1/(1+churn) to rescale from transport_dt back to base_step
            g2 = sched.diffusion_coeff(t_hat).square()
            guidance_weight = beta_fn(t_hat) * churn * g2 / 2
            guidance_weight /= 1 + churn  # we integrate with (1+churn)* dt, but FKC get's integrated with just dt
            return gmm.velocity(x, t_hat) + guidance_weight * grad_energy(x)

        def fkc_weight_update(x, t):
            g2 = sched.diffusion_coeff(t) ** 2
            dbeta = _dbeta_dt(beta_fn, t)
            annealing = dbeta * energy(x)
            alignment = (-beta_fn(t) * grad_energy(x) * (g2 / 2) * gmm.score(x, t)).squeeze(-1).squeeze(-1)
            return annealing + alignment

        torch.manual_seed(0)
        t = torch.linspace(self.T_NOISE, self.EPS, self.N_STEPS)
        x0 = gmm.sample(shape=self.N_PARTICLES, t=self.T_NOISE)
        torch.manual_seed(1)
        traj, _, weight_hist = steered_reverse_churn_sampling(
            drift=guided_drift,
            diffusion=sched.diffusion_coeff,
            transition=sched.transition,
            weight_update=fkc_weight_update,
            x=x0,
            t=t,
            churn=churn,
            ess_threshold=ess_threshold,
        )

        if churn == 1.0 and reward_center == -2.0 and ess_threshold == 2000:
            # Fixed-interval mode does not resample before the endpoint. With identical
            # transition noise, guidance must therefore improve the proposal itself rather
            # than relying on particle replication to make the cloud look tilted.
            torch.manual_seed(1)
            base_traj, _, _ = steered_reverse_churn_sampling(
                drift=gmm.velocity,
                diffusion=sched.diffusion_coeff,
                weight_update=fkc_weight_update,
                x=x0,
                t=t,
                churn=churn,
                ess_threshold=ess_threshold,
            )
            for t_idx in [350, self.N_STEPS - 2]:
                t_ = t[t_idx]
                grid_radius = max(4.0, 6.0 * sched.get_sigma_t(t_).item())
                xs = torch.linspace(-grid_radius, grid_radius, 600).reshape(-1, 1, 1)
                p_tilt = _tilted_density(gmm, xs, t_, beta_fn, lambda x, t__: -energy(x))
                guided_w1 = _wasserstein1(traj[t_idx, :, 0, 0], xs.squeeze(), p_tilt)
                base_w1 = _wasserstein1(base_traj[t_idx, :, 0, 0], xs.squeeze(), p_tilt)
                assert guided_w1 < 0.9 * base_w1, (
                    f"guidance did not improve its proposal at t={t_:.3f}: "
                    f"guided W1={guided_w1:.4f}, base W1={base_w1:.4f}"
                )

        if PLOT:
            for t_idx in [*range(50, self.N_STEPS, 100), self.N_STEPS - 2, self.N_STEPS - 1]:
                t_ = t[t_idx]
                sigma_t = sched.get_sigma_t(t_)
                grid_radius = max(4.0, 6.0 * sigma_t.item())
                xs = torch.linspace(-grid_radius, grid_radius, 600).reshape(-1, 1, 1)
                xs_flat = xs.squeeze()
                p_data = gmm.log_prob(xs, t=t_).squeeze().exp()
                p_data = p_data / torch.trapezoid(p_data, xs_flat)
                p_rew = _tilted_density(gmm, xs, t_, beta_fn, lambda x, t__: -energy(x))
                samples = traj[t_idx, :, 0, 0]
                weights = weight_hist[t_idx]
                w1 = _weighted_wasserstein1(samples, weights, xs_flat, p_rew)
                plot_marginal_density_comparison(
                    xs_flat=xs_flat,
                    p_data=p_data,
                    p_rew=p_rew,
                    samples=samples,
                    weights=weights,
                    reward_center=reward_center,
                    title=(
                        f"Guided churn FKC | KarrasSchedule\n"
                        f"churn={churn}, ess={ess_threshold}, center={reward_center}, "
                        f"t={t_:.3f} | weighted W1={w1:.4f}"
                    ),
                    out_path=(
                        _plot_dir(
                            "test_churn_steering",
                            "TestSteeredChurnGuidedFlow",
                            "test_guided_steered_karras_churn_sampler_intermediate_marginals",
                            f"churn{churn}",
                            f"ess{ess_threshold}",
                            f"reward{reward_center}",
                        )
                        / f"t{t_idx}.png"
                    ),
                    min_x=-grid_radius,
                    max_x=grid_radius,
                    energy=plot_energy,
                    energy_time=t_,
                )

        for t_idx in range(50, self.N_STEPS, 10):
            t_ = t[t_idx]
            sigma_t = sched.get_sigma_t(t_)
            grid_radius = max(4.0, 6.0 * sigma_t.item())
            xs = torch.linspace(-grid_radius, grid_radius, 600).reshape(-1, 1, 1)
            p_tilt = _tilted_density(gmm, xs, t_, beta_fn, lambda x, t__: -energy(x))
            w1 = _weighted_wasserstein1(traj[t_idx, :, 0, 0], weight_hist[t_idx], xs.squeeze(), p_tilt)
            # The bounded-variation FKC transport update is first-order accurate.
            tol = 0.08 * (1.0 + sigma_t.item())
            assert w1 < tol, f"guided Karras churn={churn} center={reward_center} t={t_:.3f}: W1={w1:.4f} >= {tol:.4f}"
