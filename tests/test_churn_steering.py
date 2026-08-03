"""FKC steering on the churn sampler — the same assertions test_steering.py makes of the
Euler-Maruyama steered sampler, retargeted at `steered_reverse_churn_sampling`.

The whole point of the comparison is that the two samplers reach the *same* tilted
marginals p_t ∝ q_t·exp(-β_t U_t) by structurally different routes: the EM sampler guides the
drift and carries a continuous-time weight (β̇U, ∂_t U, score-alignment; Prop. D.6), while
the churn sampler leaves its probability flow untouched and carries only the endpoint
difference ρ_{t+dt}(x_{t+dt}) − ρ_t(x_t) (docs/fkc_churn_steering.md §4). So the energy
plumbing here is deliberately much smaller — no gradients, no β̇ — and the marginal
assertions are deliberately identical.
"""

import os
from pathlib import Path

import pytest
import torch
from test_steering import (
    KarrasDenoiseMixin,
    _dbeta_dt,
    _plot_dir,
    _tilted_density,
    _wasserstein1,
    _weighted_wasserstein1,
    plot_marginal_density_comparison,
)

PLOT = os.getenv("TORCHGMM_PLOT_TESTS", "0") == "1"

from torchGMM.gmm import GMM
from torchGMM.sampling import _ess_ratio, reverse_churn_sampling, steered_reverse_churn_sampling
from torchGMM.schedule import BetaSchedule, KarrasSchedule

torch.set_printoptions(sci_mode=False)


class TestChurnSteeringControlFlow:
    """Resampling control flow, verified without any Monte-Carlo statistic.

    Harness: `churn=0` plus a zero velocity means particle *values* never change except
    through resampling's gather, so every trajectory value must come from the original
    `x0` pool. With β(t) = t and positive quadratic energy U(x, t) = x², the sampler's
    ρ_t(x) = -β(t)·U(x, t) is -x²·t, whose endpoint difference is exactly x²·|dt|
    — i.e. each particle carries its own fixed bias in its *value*, so the gather moves
    the bias with the particle and the recursion stays predictable. No
    `torch.manual_seed` is needed: every assertion below holds for any draw inside
    `_systematic_resample`, so these cannot flake under xdist.
    """

    N = 50
    N_STEPS = 21  # -> 20 integration steps
    K = 3.0

    @staticmethod
    def _zero_velocity(x, t):
        return torch.zeros_like(x)

    @staticmethod
    def _no_transition(x, t, s):
        raise AssertionError("churn=0 must not call the transition kernel")

    @staticmethod
    def _beta(t):
        return t

    @staticmethod
    def _energy(x, t):
        return x.square().squeeze(-1)

    def test_shapes_and_normalisation(self):
        x0 = torch.linspace(-self.K, self.K, self.N).reshape(self.N, 1)
        t = torch.linspace(1.0 - 1e-3, 1e-3, self.N_STEPS)
        traj, ess_hist, weight_hist = steered_reverse_churn_sampling(
            self._zero_velocity,
            self._no_transition,
            None,
            x0,
            t,
            churn=0.0,
            ess_threshold=0.7,
            beta=self._beta,
            energy=self._energy,
        )
        assert traj.shape == (self.N_STEPS, self.N, 1)
        assert weight_hist.shape == (self.N_STEPS, self.N)
        assert len(ess_hist) == self.N_STEPS - 1
        assert torch.allclose(weight_hist.sum(dim=1), torch.ones(self.N_STEPS), atol=1e-6)
        assert torch.isin(traj, x0).all()

    def test_energy_updates_at_each_operator_boundary(self):
        """A nonzero churn applies U at t̂ and the alpha=0 update at the step endpoint."""
        x0 = torch.tensor([[-1.0], [0.0], [1.0]])
        t = torch.tensor([0.8, 0.6, 0.4])
        energy_times = []

        def transition(x, t_curr, t_hat):
            return x + (t_hat - t_curr)

        def beta(t_):
            return t_

        def energy(x, t_):
            energy_times.append(t_.item())
            return x.square().squeeze(-1)

        _, _, weight_history = steered_reverse_churn_sampling(
            self._zero_velocity,
            transition,
            None,
            x0,
            t,
            churn=0.5,
            ess_threshold=1_000,
            beta=beta,
            energy=energy,
        )

        assert energy_times == pytest.approx([0.8, 0.9, 0.6, 0.7, 0.4])
        x_hat = x0 + 0.1
        expected_weights = torch.softmax(-x_hat.square().squeeze(-1) * t[1] + x0.square().squeeze(-1) * t[0], dim=0)
        torch.testing.assert_close(weight_history[1], expected_weights)

    def test_weight_update_uses_the_reheated_state(self):
        """The FKC route evaluates its increment at the churned state and time."""
        x0 = torch.tensor([[-1.0], [0.0], [1.0]])
        t = torch.tensor([0.8, 0.6, 0.4])
        spans = []

        def transition(x, t_curr, t_hat):
            return x + (t_hat - t_curr)

        def fkc_weight_update(x, t_hat, dt):
            spans.append((t_hat.item(), dt.item()))
            return x.squeeze(-1) * dt

        _, _, weight_history = steered_reverse_churn_sampling(
            self._zero_velocity,
            transition,
            fkc_weight_update,
            x0,
            t,
            churn=0.5,
            ess_threshold=1_000,
        )

        assert [t_hat for t_hat, _ in spans] == pytest.approx([0.9, 0.7])
        assert [dt for _, dt in spans] == pytest.approx([-0.2, -0.2])
        x_hat = x0 + 0.1
        fkc_update = x_hat.squeeze(-1) * -0.2
        expected_weights = torch.softmax(fkc_update, dim=0)
        torch.testing.assert_close(weight_history[1], expected_weights)

    def test_energy_and_weight_update_are_mutually_exclusive(self):
        x0 = torch.tensor([[-1.0], [0.0], [1.0]])
        t = torch.tensor([0.8, 0.6, 0.4])

        with pytest.raises(ValueError, match="mutually exclusive"):
            steered_reverse_churn_sampling(
                self._zero_velocity,
                self._no_transition,
                lambda x, t_, dt: torch.zeros(x.shape[0]),
                x0,
                t,
                churn=0.0,
                beta=self._beta,
                energy=self._energy,
            )

    def test_zero_churn_skips_the_redundant_midpoint_energy(self):
        x0 = torch.tensor([[-1.0], [0.0], [1.0]])
        t = torch.tensor([0.8, 0.6, 0.4])
        energy_times = []

        def beta(t_):
            return t_

        def energy(x, t_):
            energy_times.append(t_.item())
            return x.square().squeeze(-1)

        steered_reverse_churn_sampling(
            self._zero_velocity,
            self._no_transition,
            None,
            x0,
            t,
            churn=0.0,
            ess_threshold=1_000,
            beta=beta,
            energy=energy,
        )

        assert energy_times == pytest.approx([0.8, 0.6, 0.4])

    def test_ess_trace_matches_analytic_prediction_before_first_resample(self):
        """Until the first gather the particles are untouched, so the ESS trace is a closed
        form: log w accumulates U(x)·|dt| from a fixed linspace particle value."""
        x0 = torch.linspace(-self.K, self.K, self.N).reshape(self.N, 1)
        t = torch.linspace(1.0 - 1e-3, 1e-3, self.N_STEPS)
        _, ess_hist, _ = steered_reverse_churn_sampling(
            self._zero_velocity,
            self._no_transition,
            None,
            x0,
            t,
            churn=0.0,
            ess_threshold=1_000,  # interval >> steps: no intermittent resample
            beta=self._beta,
            energy=self._energy,
        )
        energy = torch.linspace(-self.K, self.K, self.N).square()
        log_w = torch.zeros(self.N)
        predicted = []
        for t_curr, t_next in zip(t[:-1], t[1:]):
            log_w = log_w + energy * (t_next - t_curr).abs()
            predicted.append(_ess_ratio(log_w))
        assert ess_hist == pytest.approx(predicted, abs=1e-6)

    def test_interval_mode_resets_weights_on_schedule(self):
        interval = 5
        x0 = torch.linspace(-self.K, self.K, self.N).reshape(self.N, 1)
        t = torch.linspace(1.0 - 1e-3, 1e-3, self.N_STEPS)
        _, _, weight_hist = steered_reverse_churn_sampling(
            self._zero_velocity,
            self._no_transition,
            None,
            x0,
            t,
            churn=0.0,
            ess_threshold=interval,
            beta=self._beta,
            energy=self._energy,
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
            self._no_transition,
            None,
            x0,
            t,
            churn=0.0,
            ess_threshold=1,
            beta=self._beta,
            energy=self._energy,
        )
        uniform = torch.full((self.N,), 1 / self.N)
        assert torch.allclose(weight_hist, uniform.expand_as(weight_hist), atol=1e-6)

    def test_final_resample_always_fires(self):
        x0 = torch.linspace(-self.K, self.K, self.N).reshape(self.N, 1)
        t = torch.linspace(1.0 - 1e-3, 1e-3, self.N_STEPS)
        _, _, weight_hist = steered_reverse_churn_sampling(
            self._zero_velocity,
            self._no_transition,
            None,
            x0,
            t,
            churn=0.0,
            ess_threshold=1_000,  # interval >> steps: only the mandatory final resample fires
            beta=self._beta,
            energy=self._energy,
        )
        uniform = torch.full((self.N,), 1 / self.N)
        assert not torch.allclose(weight_hist[-2], uniform, atol=1e-6)
        assert torch.allclose(weight_hist[-1], uniform, atol=1e-6)

    @pytest.mark.parametrize("bad_threshold", [0, -1, -0.5])
    def test_non_positive_threshold_rejected(self, bad_threshold):
        x0 = torch.linspace(-self.K, self.K, self.N).reshape(self.N, 1)
        t = torch.linspace(1.0 - 1e-3, 1e-3, self.N_STEPS)
        with pytest.raises(ValueError, match="positive"):
            steered_reverse_churn_sampling(
                self._zero_velocity,
                self._no_transition,
                None,
                x0,
                t,
                churn=0.0,
                ess_threshold=bad_threshold,
                beta=self._beta,
                energy=self._energy,
            )

    @pytest.mark.parametrize("bad_threshold", [2.5, 1.5, 10.25])
    def test_non_integer_interval_threshold_rejected(self, bad_threshold):
        x0 = torch.linspace(-self.K, self.K, self.N).reshape(self.N, 1)
        t = torch.linspace(1.0 - 1e-3, 1e-3, self.N_STEPS)
        with pytest.raises(ValueError, match="whole number"):
            steered_reverse_churn_sampling(
                self._zero_velocity,
                self._no_transition,
                None,
                x0,
                t,
                churn=0.0,
                ess_threshold=bad_threshold,
                beta=self._beta,
                energy=self._energy,
            )

    def test_t_must_be_decreasing(self):
        x0 = torch.linspace(-self.K, self.K, self.N).reshape(self.N, 1)
        t = torch.linspace(1e-3, 1.0 - 1e-3, self.N_STEPS)
        with pytest.raises(ValueError, match="strictly decreasing"):
            steered_reverse_churn_sampling(
                self._zero_velocity, self._no_transition, None, x0, t, churn=0.0, beta=self._beta, energy=self._energy
            )

    def test_negative_churn_rejected(self):
        x0 = torch.linspace(-self.K, self.K, self.N).reshape(self.N, 1)
        t = torch.linspace(1.0 - 1e-3, 1e-3, self.N_STEPS)
        with pytest.raises(ValueError, match="non-negative"):
            steered_reverse_churn_sampling(
                self._zero_velocity, self._no_transition, None, x0, t, churn=-0.5, beta=self._beta, energy=self._energy
            )


@pytest.mark.slow
class TestSteeredChurnGuidedFlow:
    """Karras churn with FKC-guided probability-flow transport.

    Churn supplies the effective diffusion and therefore fixes the guidance displacement.
    It carries no potential-ratio weight; the bounded-variation FKC update compensates the
    guided deterministic leg.
    """

    EPS = 0.001
    T_NOISE = 1 - EPS
    N_PARTICLES = 10_000
    N_STEPS = 500
    REWARD_SIGMA = 1.0
    GUIDANCE_BOOST = 20.0

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

    @pytest.mark.parametrize("churn", [0.25, 0.5, 1.0], ids=lambda v: f"churn={v}")
    @pytest.mark.parametrize("reward_center", [-2.0, 1.0], ids=lambda v: f"center={v}")
    @pytest.mark.parametrize('ess_threshold', [0.9, 700], ids=lambda v: f"ess={v}")
    def test_guided_steered_karras_churn_sampler_intermediate_marginals(
        self, setup, reward_center, churn, ess_threshold
    ):
        gmm, sched = setup
        step_size = (self.T_NOISE - self.EPS) / (self.N_STEPS - 1)

        def energy(x):
            return (0.5 * (x - reward_center) ** 2 / self.REWARD_SIGMA**2).squeeze(-1).squeeze(-1)

        def beta_fn(t):
            g = sched.diffusion_coeff(t)
            bar_sigma_t = sched.get_sigma_t(t)
            return (1 - t) / (1 + 0.1* bar_sigma_t**2)

        def plot_energy(x, t):
            return energy(x)

        def grad_energy(x):
            return (x - reward_center) / self.REWARD_SIGMA**2

        def guided_drift(x, t_hat):
            # The sampler calls drift at the reheated time. Churn injects the variance
            # between t_before_churn and t_hat; the guided flow transports over the
            # remaining -(1 + churn) grid steps.
            t_before_churn = t_hat - churn * step_size
            injected_variance = sched.get_sigma_t(t_hat).square() - sched.get_sigma_t(t_before_churn).square()
            transport_dt = -(1.0 + churn) * step_size
            guidance = -injected_variance * beta_fn(t_hat) / (2 * transport_dt) * grad_energy(x)
            return gmm.velocity(x, t_hat) + guidance

        def fkc_weight_update(x, t, dt):
            g2 = sched.diffusion_coeff(t) ** 2
            dbeta = _dbeta_dt(beta_fn, t)
            annealing = dbeta * energy(x)
            alignment = (-beta_fn(t) * grad_energy(x) * (g2 / 2) * gmm.score(x, t)).squeeze(-1).squeeze(-1)
            return (annealing + alignment) * dt.abs()

        torch.manual_seed(0)
        t = torch.linspace(self.T_NOISE, self.EPS, self.N_STEPS)
        x0 = gmm.sample(shape=self.N_PARTICLES, t=self.T_NOISE)
        torch.manual_seed(1)
        traj, _, weight_hist = steered_reverse_churn_sampling(
            drift=guided_drift,
            transition=sched.transition,
            weight_update=fkc_weight_update,
            x=x0,
            t=t,
            churn=churn,
            ess_threshold=ess_threshold,
        )

        if churn == 1.0 and reward_center == -2.0 and ess_threshold == 700:
            # Fixed-interval mode does not resample before the endpoint. With identical
            # transition noise, guidance must therefore improve the proposal itself rather
            # than relying on particle replication to make the cloud look tilted.
            torch.manual_seed(1)
            base_traj, _, _ = steered_reverse_churn_sampling(
                drift=gmm.velocity,
                transition=sched.transition,
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


@pytest.mark.slow
class TestSteeredChurnBetaIntermediateMarginals:
    """Weighted intermediate-time checks under BetaSchedule, direct energy U(x_t).

    Mirrors TestSteeredSamplingBetaIntermediateMarginals, but asserts (that one's W1
    assertion is commented out) across 49 intermediate time slots.
    """

    EPS = 0.001
    N_PARTICLES = 10_000
    N_STEPS = 500

    @pytest.fixture
    def setup(self):
        sched = BetaSchedule(beta_min=0.1, beta_max=20.0)
        gmm = GMM(
            mu=torch.tensor([[[-2.5], [2.5]]]),
            sigma=torch.tensor([[[0.8], [0.8]]]),
            weight=torch.tensor([[0.2, 0.8]]),
            schedule=sched,
        )
        return gmm, sched

    @pytest.mark.parametrize("churn", [0.5, 1.0], ids=lambda v: f"churn={v}")
    @pytest.mark.parametrize("ess_threshold", [0.9, 25, 1000], ids=lambda v: f"ess={v}")
    @pytest.mark.parametrize("reward_center", [-2.0, -0.25, 1.0], ids=lambda v: f"center={v}")
    def test_weighted_beta_churn_matches_tilted_intermediate_marginals(
        self, setup, reward_center, ess_threshold, churn
    ):
        gmm, sched = setup
        reward_sigma = 1.0

        def energy(x, t_):
            return (0.5 * (x - reward_center) ** 2 / reward_sigma**2).squeeze(-1).squeeze(-1)

        def beta_fn(t):
            return 1.0 - t

        torch.manual_seed(0)
        t = torch.linspace(1 - self.EPS, self.EPS, self.N_STEPS)
        x0 = gmm.sample(shape=self.N_PARTICLES, t=1 - self.EPS)
        traj, ess_hist, weight_hist = steered_reverse_churn_sampling(
            gmm.velocity,
            sched.transition,
            None,
            x0,
            t,
            churn=churn,
            ess_threshold=ess_threshold,
            beta=beta_fn,
            energy=energy,
        )
        assert traj.shape == (self.N_STEPS, self.N_PARTICLES, 1, 1)
        assert weight_hist.shape == (self.N_STEPS, self.N_PARTICLES)

        xs = torch.linspace(-8, 8, 400).reshape(-1, 1, 1)
        xs_flat = xs.squeeze()
        for t_idx in range(10, self.N_STEPS, 10):  # 49 intermediate slots
            t_ = t[t_idx]
            sigma_t = sched.get_sigma_t(t_)
            p_tilt = _tilted_density(gmm, xs, t_, beta_fn, lambda x, t__: -energy(x, t__))
            w1 = _weighted_wasserstein1(traj[t_idx, :, 0, 0], weight_hist[t_idx], xs_flat, p_tilt)
            tol = 0.05 * (1.0 + sigma_t.item())
            assert w1 < tol, (
                f"BetaSchedule churn={churn} center={reward_center} ess={ess_threshold} "
                f"t={t_:.3f}: W1={w1:.4f} >= {tol:.4f}"
            )


@pytest.mark.slow
class TestSteeredChurnKarrasIntermediateMarginals:
    """Weighted intermediate-time checks under KarrasSchedule, direct energy U(x_t).

    Mirrors TestSteeredSamplingKarrasIntermediateMarginals, tolerance included.
    """

    EPS = 0.001
    T_NOISE = 1 - EPS
    # 10k rather than the EM suite's 5k: the terminal-W1 spread over seeds is sd ~0.010 at
    # 5k against a ~0.05 tolerance, so single configs sit within one sigma of the bound and
    # pass or fail on the draw. 10k halves that and makes the assertion about the sampler.
    N_PARTICLES = 10_000
    N_STEPS = 600

    @pytest.fixture
    def setup(self):
        sched = KarrasSchedule(sigma_min=0.01, sigma_max=160, rho=7.0, sigma_data=1.0)
        gmm = GMM(
            mu=torch.tensor([[[-1.0], [1.0]]]),
            sigma=torch.tensor([[[0.4], [0.4]]]),
            weight=torch.tensor([[0.3, 0.7]]),
            schedule=sched,
        )
        return gmm, sched

    @pytest.mark.parametrize("churn", [0.1, 0.5, 1.0], ids=lambda v: f"churn={v}")
    @pytest.mark.parametrize("ess_threshold", [0.9, 25, 1000], ids=lambda v: f"ess={v}")
    @pytest.mark.parametrize("reward_center", [-1.0, -0.5, 0.5], ids=lambda v: f"center={v}")
    def test_weighted_karras_churn_matches_tilted_intermediate_marginals(
        self, setup, reward_center, ess_threshold, churn
    ):
        gmm, sched = setup
        reward_sigma = 1.0

        def energy(x, t_):
            return (0.5 * (x - reward_center) ** 2 / reward_sigma**2).squeeze(-1).squeeze(-1)

        def beta_fn(t):
            g = sched.diffusion_coeff(t)
            return (1 - t) ** 2 / (1 + g**2)

        torch.manual_seed(0)
        t = torch.linspace(self.T_NOISE, self.EPS, self.N_STEPS)
        x0 = gmm.sample(shape=self.N_PARTICLES, t=self.T_NOISE)
        traj, _, weight_hist = steered_reverse_churn_sampling(
            drift=gmm.velocity,
            transition=sched.transition,
            weight_update=None,
            x=x0,
            t=t,
            churn=churn,
            ess_threshold=ess_threshold,
            beta=beta_fn,
            energy=energy,
        )
        assert traj.shape == (self.N_STEPS, self.N_PARTICLES, 1, 1)

        for t_idx in range(50, self.N_STEPS, 10):  # 56 intermediate slots
            t_ = t[t_idx]
            sigma_t = sched.get_sigma_t(t_)
            grid_radius = max(4.0, 6.0 * sigma_t.item())
            xs = torch.linspace(-grid_radius, grid_radius, 400).reshape(-1, 1, 1)
            p_tilt = _tilted_density(gmm, xs, t_, beta_fn, lambda x, t__: -energy(x, t__))
            w1 = _weighted_wasserstein1(traj[t_idx, :, 0, 0], weight_hist[t_idx], xs.squeeze(), p_tilt)
            # Low churn plus interval resampling has a measured Monte Carlo tail just
            # above 0.05 at 10k particles; this remains far below a biased marginal.
            tol = 0.06 * (1.0 + sigma_t.item())
            assert w1 < tol, (
                f"KarrasSchedule churn={churn} center={reward_center} ess={ess_threshold} "
                f"t={t_:.3f}: W1={w1:.4f} >= {tol:.4f}"
            )

        if PLOT:
            # Regular slices across the reverse pass, plus the last two. The mandatory final
            # resample fires only at index N_STEPS-1, so the penultimate panel shows the tilt
            # still living in the *weights* (unweighted histogram off the target, weighted on
            # it) while the final panel shows the two coinciding after the gather.
            for t_idx in [*range(50, self.N_STEPS, 100), self.N_STEPS - 2, self.N_STEPS - 1]:
                # Grid tracks sigma_t exactly as the W1 loop above does, so the high-noise
                # panels are not truncated.
                t_ = t[t_idx]
                grid_radius = max(4.0, 6.0 * sched.get_sigma_t(t_).item())
                xs = torch.linspace(-grid_radius, grid_radius, 400).reshape(-1, 1, 1)
                xs_flat = xs.squeeze()
                p_data = gmm.log_prob(xs, t=t_).squeeze().exp()
                p_data = p_data / torch.trapezoid(p_data, xs_flat)
                p_rew = _tilted_density(gmm, xs, t_, beta_fn, lambda x, t__: -energy(x, t__))
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
                        f"Churn-FKC | KarrasSchedule, direct energy U(x_t)\n"
                        f"churn={churn}, ess={ess_threshold}, center={reward_center}, "
                        f"t={t_:.3f} | weighted W1={w1:.4f}"
                    ),
                    out_path=(
                        _plot_dir(
                            "test_churn_steering",
                            "TestSteeredChurnKarrasIntermediateMarginals",
                            "test_weighted_karras_churn_matches_tilted_intermediate_marginals",
                            f"churn{churn}",
                            f"ess{ess_threshold}",
                            f"reward{reward_center}",
                        )
                        / f"t{t_idx}.png"
                    ),
                    min_x=-grid_radius,
                    max_x=grid_radius,
                    energy=energy,
                    energy_time=t_,
                )


@pytest.mark.slow
class TestSteeredChurnDenoisingKarras(KarrasDenoiseMixin):
    """KarrasSchedule with an energy on denoised x̂₀ — the pattern real energy models use.

    Mirrors TestSteeredSamplingDenoisingKarrasIntermediateMarginals. Note how much smaller
    the steering plumbing is than the EM equivalent: the endpoint potential needs only
    ρ *values*, so the denoiser is never backpropagated through — no `torch.autograd.grad`,
    no ∂_t r, no β̇. Both the intermediate marginals and the terminal marginal are checked.
    """

    EPS = 0.001
    T_NOISE = 1 - EPS
    N_PARTICLES = 10_000  # see the note on TestSteeredChurnKarrasIntermediateMarginals
    N_STEPS = 610

    @pytest.fixture
    def setup(self):
        sched = KarrasSchedule(sigma_min=0.01, sigma_max=160, rho=7.0, sigma_data=1.0)
        gmm = GMM(
            mu=torch.tensor([[[-1.0], [1.0]]]),
            sigma=torch.tensor([[[0.4], [0.4]]]),
            weight=torch.tensor([[0.3, 0.7]]),
            schedule=sched,
        )
        return gmm, sched

    # Matches the EM denoising test's 3x3 grid. churn is fixed at 1.0 (the EDM default, and
    # the value whose continuous limit is the ordinary reverse SDE); churn=0.5 coverage comes
    # from the direct-energy class above, which is far cheaper — no denoiser in the loop.
    @pytest.mark.parametrize("ess_threshold", [0.9, 25, 1000], ids=lambda v: f"ess={v}")
    @pytest.mark.parametrize("reward_center", [-1.0, -0.5, 0.5], ids=lambda v: f"center={v}")
    @pytest.mark.parametrize("n_denoise_steps", [10], ids=lambda v: f"{v}")
    def test_weighted_denoising_karras_churn_matches_tilted_marginals(
        self, setup, reward_center, ess_threshold, n_denoise_steps
    ):
        gmm, sched = setup
        reward_sigma = 1.0
        churn = 1.0

        def energy(x0_hat):
            return (0.5 * (x0_hat - reward_center) ** 2 / reward_sigma**2).squeeze(-1).squeeze(-1)

        def beta_fn(t):
            g = sched.diffusion_coeff(t)
            return (1 - t) / (1 + g**2)

        def energy_on(x, t_):
            return energy(self._denoise(gmm, sched, x, t_, n_denoise_steps)[0].detach())

        torch.manual_seed(0)
        t = torch.linspace(self.T_NOISE, self.EPS, self.N_STEPS)
        x0 = gmm.sample(shape=self.N_PARTICLES, t=self.T_NOISE)
        traj, _, weight_hist = steered_reverse_churn_sampling(
            gmm.velocity,
            sched.transition,
            None,
            x0,
            t,
            churn=churn,
            ess_threshold=ess_threshold,
            beta=beta_fn,
            energy=energy_on,
        )
        assert traj.shape == (self.N_STEPS, self.N_PARTICLES, 1, 1)

        for t_idx in range(50, self.N_STEPS, 10):  # 56 intermediate slots
            t_ = t[t_idx]
            sigma_t = sched.get_sigma_t(t_)
            grid_radius = max(4.0, 6.0 * sigma_t.item())
            xs = torch.linspace(-grid_radius, grid_radius, 400).reshape(-1, 1, 1)
            p_tilt = _tilted_density(gmm, xs, t_, beta_fn, lambda x, t__: -energy_on(x, t__))
            w1 = _weighted_wasserstein1(traj[t_idx, :, 0, 0], weight_hist[t_idx], xs.squeeze(), p_tilt)
            # Higher floor than the direct-energy tests: the unrolled denoiser adds its own
            # discretization error on top of the sampler's.
            tol = 0.08 * (1.0 + sigma_t.item())
            assert w1 < tol, (
                f"Karras denoising churn={churn} center={reward_center} ess={ess_threshold} "
                f"t={t_:.3f}: W1={w1:.4f} >= {tol:.4f}"
            )

        if PLOT:
            # Regular slices across the reverse pass, plus the last two. The mandatory final
            # resample fires only at index N_STEPS-1, so the penultimate panel shows the tilt
            # still living in the *weights* (unweighted histogram off the target, weighted on
            # it) while the final panel shows the two coinciding after the gather.
            for t_idx in [*range(50, self.N_STEPS, 100), self.N_STEPS - 2, self.N_STEPS - 1]:
                # Grid tracks sigma_t exactly as the W1 loop above does, so the high-noise
                # panels are not truncated.
                t_ = t[t_idx]
                grid_radius = max(4.0, 6.0 * sched.get_sigma_t(t_).item())
                xs = torch.linspace(-grid_radius, grid_radius, 400).reshape(-1, 1, 1)
                xs_flat = xs.squeeze()
                p_data = gmm.log_prob(xs, t=t_).squeeze().exp()
                p_data = p_data / torch.trapezoid(p_data, xs_flat)
                p_rew = _tilted_density(gmm, xs, t_, beta_fn, lambda x, t__: -energy_on(x, t__))
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
                        f"Churn-FKC | KarrasSchedule, denoised energy U(D(x_t, t))\n"
                        f"churn={churn}, ess={ess_threshold}, center={reward_center}, "
                        f"t={t_:.3f} | weighted W1={w1:.4f}"
                    ),
                    out_path=(
                        _plot_dir(
                            "test_churn_steering",
                            "TestSteeredChurnDenoisingKarras",
                            "test_weighted_denoising_karras_churn_matches_tilted_marginals",
                            f"churn{churn}",
                            f"ess{ess_threshold}",
                            f"reward{reward_center}",
                        )
                        / f"t{t_idx}.png"
                    ),
                    min_x=-grid_radius,
                    max_x=grid_radius,
                    energy=energy_on,
                    energy_time=t_,
                )

        # Terminal marginal: the headline number, directly comparable to the EM sampler's
        # `assert w1_rew < 0.075` in TestSteeredSamplingKarrasFinalMarginal.
        xs = torch.linspace(-3, 3, 100).reshape(-1, 1, 1)
        xs_flat = xs.squeeze()
        p_rew = _tilted_density(gmm, xs, torch.tensor(self.EPS), beta_fn, lambda x, t__: -energy_on(x, t__))
        w1_terminal = _wasserstein1(traj[-1, :, 0, 0], xs_flat, p_rew)

        if PLOT:
            p_data = gmm.log_prob(xs, t=self.EPS).squeeze().exp()
            p_data = p_data / torch.trapezoid(p_data, xs_flat)
            plot_marginal_density_comparison(
                xs_flat=xs_flat,
                p_data=p_data,
                p_rew=p_rew,
                samples=traj[-1, :, 0, 0],
                reward_center=reward_center,
                title=(
                    f"Churn FKC terminal marginal | KarrasSchedule, denoised energy\n"
                    f"churn={churn}, ess={ess_threshold}, center={reward_center} | W1={w1_terminal:.4f}"
                ),
                out_path=(
                    _plot_dir(
                        "test_churn_steering",
                        "TestSteeredChurnDenoisingKarras",
                        "test_weighted_denoising_karras_churn_matches_tilted_marginals",
                        f"churn{churn}",
                        f"ess{ess_threshold}",
                        f"reward{reward_center}",
                    )
                    / "terminal.png"
                ),
                min_x=-3,
                max_x=3,
            )
        assert w1_terminal < 0.075, (
            f"Karras denoising terminal churn={churn} center={reward_center} ess={ess_threshold}: W1={w1_terminal:.4f}"
        )


class TestChurnOperatorWeights:
    """Check endpoint reweighting across an exact forward transition."""

    def test_churn_operator_weights(self):
        """A quadratic tilt remains correct after transitioning from ``t`` to ``s``.

        Starting with q_t and the current tilt rho_t, the incremental correction is
        rho_s(x_s) - rho_t(x_t). The resulting weighted particles must therefore
        represent q_s exp(rho_s), while the accumulated normalizer estimates Z_s / Z_t.
        """
        torch.manual_seed(0)
        schedule = BetaSchedule()
        gmm = GMM(
            mu=torch.tensor([[[-2.0], [0.0], [2.0]]], dtype=torch.float64),
            sigma=torch.tensor([[[0.3], [0.3], [0.2]]], dtype=torch.float64),
            weight=torch.tensor([[0.33, 0.5, 0.17]], dtype=torch.float64),
            schedule=schedule,
        )

        n_particles = 10_000
        t, s = torch.tensor(0.01, dtype=torch.float64), torch.tensor(0.15, dtype=torch.float64)
        center_t = torch.tensor(1.5, dtype=torch.float64)
        center_s = torch.tensor(-1.5, dtype=torch.float64)
        reward_sigma = torch.tensor(1.5, dtype=torch.float64)

        def energy(x, center, slope=1.0):
            return (slope * (x - center).square() / reward_sigma**2).squeeze()

        x_t = gmm.sample(shape=n_particles, t=t)
        rho_t = -(1.0 - t) * energy(x_t, center_t, slope=1.0)
        x_s = schedule.transition(x_t, t, s)
        rho_s = -(1.0 - s) * energy(x_s, center_s, slope=3.0)
        log_weight = rho_t + (rho_s - rho_t)
        weights = torch.softmax(log_weight, dim=0)

        # The endpoint correction must telescope exactly to the endpoint potential.
        torch.testing.assert_close(
            weights,
            torch.softmax(rho_s, dim=0),
            atol=1e-12,
            rtol=1e-12,
        )

        # Compare the weighted cloud with the analytic tilted q_s on a dense grid.
        x_grid = torch.linspace(-10.0, 10.0, 4_001, dtype=x_t.dtype).reshape(-1, 1, 1)
        log_density = gmm.log_prob(x_grid, t=s).squeeze() + (-(1.0 - s) * energy(x_grid, center_s, slope=3.0))
        density = log_density.exp()
        normalizer = torch.trapezoid(density, x_grid.squeeze())
        target_density = density / normalizer
        base_density_s = gmm.log_prob(x_grid, t=s).squeeze().exp()
        base_density_s = base_density_s / torch.trapezoid(base_density_s, x_grid.squeeze())
        weighted_w1_s = _weighted_wasserstein1(x_s.squeeze(), weights, x_grid.squeeze(), target_density)

        log_density_t = gmm.log_prob(x_grid, t=t).squeeze() + (-(1.0 - t) * energy(x_grid, center_t, slope=1.0))
        density_t = log_density_t.exp()
        target_density_t = density_t / torch.trapezoid(density_t, x_grid.squeeze())
        base_density_t = gmm.log_prob(x_grid, t=t).squeeze().exp()
        base_density_t = base_density_t / torch.trapezoid(base_density_t, x_grid.squeeze())
        initial_weights = torch.softmax(rho_t, dim=0)
        weighted_w1_t = _weighted_wasserstein1(x_t.squeeze(), initial_weights, x_grid.squeeze(), target_density_t)

        if PLOT:
            plot_marginal_density_comparison(
                xs_flat=x_grid.squeeze(),
                p_data=base_density_t,
                p_rew=target_density_t,
                samples=x_t.squeeze(),
                weights=initial_weights,
                reward_center=center_t.item(),
                title=f"Churn reweighting at t={t.item():.2f}, W1={weighted_w1_t:.4f}",
                out_path=(
                    _plot_dir(
                        "test_churn_steering",
                        "TestChurnOperatorWeights",
                        "test_churn_operator_weights",
                    )
                    / "reweighting_t.png"
                ),
                min_x=-5.0,
                max_x=5.0,
                energy=lambda x, time: energy(x, center_t, slope=1.0),
                energy_time=t,
            )
            plot_marginal_density_comparison(
                xs_flat=x_grid.squeeze(),
                p_data=base_density_s,
                p_rew=target_density,
                samples=x_s.squeeze(),
                weights=weights,
                reward_center=center_s.item(),
                title=f"Churn reweighting at s={s.item():.2f}, W1={weighted_w1_s:.4f}",
                out_path=(
                    _plot_dir(
                        "test_churn_steering",
                        "TestChurnOperatorWeights",
                        "test_churn_operator_weights",
                    )
                    / "reweighting_s.png"
                ),
                min_x=-5.0,
                max_x=5.0,
                energy=lambda x, time: energy(x, center_s, slope=3.0),
                energy_time=s,
            )
        assert weighted_w1_t < 0.05
        assert weighted_w1_s < 0.05


@pytest.mark.slow
class TestChurnSteeringWithEulerMaruyamaWeight:
    """The Euler-Maruyama FKC weight, reused verbatim on the churn sampler.

    This is what the `(drift, transition, weight_update)` signature buys: `weight_update`
    means the same thing here as in `steered_reverse_sampling`, so the *same closure*
    steers both samplers. It is also an empirical check on the κ-invariance derived in
    `docs/fkc_churn_steering.md` §7 — redoing Prop. D.6 Step 5 with the churn's effective
    generator (b_κ, g_κ) leaves the weight equal to Eq. (276) with the base g², carrying no
    κ at all. If that were wrong, sweeping churn here would walk off the tilted marginals.

    Only the *drift* carries κ. The churn sampler transports over −(1+κ)|dt| rather than
    −|dt|, so a guidance field must be divided by (1+κ) to land the same displacement per
    grid step, and the base transport is the probability-flow velocity rather than the
    reverse-SDE drift, since the churn supplies the stochasticity.
    """

    EPS = 0.001
    N_PARTICLES = 10_000
    N_STEPS = 500
    REWARD_SIGMA = 1.0

    @pytest.fixture
    def setup(self):
        sched = BetaSchedule(beta_min=0.1, beta_max=20.0)
        gmm = GMM(
            mu=torch.tensor([[[-2.5], [2.5]]]),
            sigma=torch.tensor([[[0.8], [0.8]]]),
            weight=torch.tensor([[0.2, 0.8]]),
            schedule=sched,
        )
        return gmm, sched

    @pytest.mark.parametrize("churn", [0.5, 1.0, 2.0], ids=lambda v: f"churn={v}")
    @pytest.mark.parametrize("reward_center", [-2.0, 1.0], ids=lambda v: f"center={v}")
    def test_em_weight_update_steers_the_churn_sampler(self, setup, reward_center, churn):
        gmm, sched = setup

        def energy(x):
            return (0.5 * (x - reward_center) ** 2 / self.REWARD_SIGMA**2).squeeze(-1).squeeze(-1)

        def grad_energy(x):
            return (x - reward_center) / self.REWARD_SIGMA**2

        def beta_fn(t):
            return 1.0 - t

        # The FKC update for the Boltzmann tilt ρ = -βU; no churn appears in it.
        def fkc_weight_update(x, t, dt):
            f = sched.forward_drift(x, t)
            sigma = sched.diffusion_coeff(t)
            score = gmm.score(x, t)
            energy_gradient, energy_value = grad_energy(x), energy(x)
            beta = beta_fn(t)
            term1 = _dbeta_dt(beta_fn, t) * energy_value
            term2 = (beta * energy_gradient * f).squeeze(-1).squeeze(-1)
            term3 = (-beta * energy_gradient * (sigma**2 / 2) * score).squeeze(-1).squeeze(-1)
            return (term1 + term2 + term3) * dt.abs()

        def guided_drift(x, t):
            """PF velocity (the churn supplies the noise) plus the D.6 guidance field.

            Two churn-dependent factors, and both are load-bearing. The magic constant is
            built from the *effective* diffusion g_κ² = κ·g², not from g² — a = β·κ·g²/2,
            per §7 — and the field is divided by (1+κ) because the transport spans
            −(1+κ)|dt| rather than −|dt|. They coincide only at κ=1, which is precisely
            why an implementation missing the κ in `a` passes at churn=1 and fails either
            side of it.
            """
            g2 = sched.diffusion_coeff(t) ** 2
            a = beta_fn(t) * churn * g2 / 2
            return gmm.velocity(x, t) + a * grad_energy(x) / (1.0 + churn)

        torch.manual_seed(0)
        t = torch.linspace(1 - self.EPS, self.EPS, self.N_STEPS)
        x0 = gmm.sample(shape=self.N_PARTICLES, t=1 - self.EPS)
        traj, _, weight_hist = steered_reverse_churn_sampling(
            guided_drift, sched.transition, fkc_weight_update, x0, t, churn=churn, ess_threshold=25
        )

        xs = torch.linspace(-8, 8, 400).reshape(-1, 1, 1)
        for t_idx in range(10, self.N_STEPS, 10):  # 49 slots
            t_ = t[t_idx]
            p_tilt = _tilted_density(gmm, xs, t_, beta_fn, lambda x, t__: -energy(x))
            w1 = _weighted_wasserstein1(traj[t_idx, :, 0, 0], weight_hist[t_idx], xs.squeeze(), p_tilt)
            tol = 0.05 * (1.0 + sched.get_sigma_t(t_).item())
            assert w1 < tol, (
                f"EM weight on churn sampler, churn={churn} center={reward_center} t={t_:.3f}: W1={w1:.4f} >= {tol:.4f}"
            )
