"""FKC steering on the churn sampler — the same assertions test_steering.py makes of the
Euler-Maruyama steered sampler, retargeted at `steered_reverse_churn_sampling`.

The whole point of the comparison is that the two samplers reach the *same* tilted
marginals p_t ∝ q_t·exp(ρ_t) by structurally different routes: the EM sampler guides the
drift and carries a continuous-time weight (β̇r, ∂_t r, score-alignment; Prop. D.6), while
the churn sampler leaves its probability flow untouched and carries only the endpoint
difference ρ_{t+dt}(x_{t+dt}) − ρ_t(x_t) (docs/fkc_churn_steering.md §4). So the reward
plumbing here is deliberately much smaller — no gradients, no β̇ — and the marginal
assertions are deliberately identical.
"""

from pathlib import Path

import pytest
import torch
from test_steering import (
    PLOT,
    KarrasDenoiseMixin,
    _dbeta_dt,
    _plot_dir,
    _tilted_density,
    _wasserstein1,
    _weighted_wasserstein1,
    plot_marginal_density_comparison,
)

from torchGMM.gmm import GMM
from torchGMM.sampling import _ess_ratio, reverse_churn_sampling, steered_reverse_churn_sampling
from torchGMM.schedule import BetaSchedule, KarrasSchedule

torch.set_printoptions(sci_mode=False)


class TestChurnSteeringControlFlow:
    """Resampling control flow, verified without any Monte-Carlo statistic.

    Harness: `churn=0` plus a zero velocity means particle *values* never change except
    through resampling's gather, so every trajectory value must come from the original
    `x0` pool. The potential is ρ_t(x) = −x·t, whose endpoint difference is exactly
    x·|dt| — i.e. each particle carries its own fixed bias in its *value*, so the gather
    moves the bias with the particle and the recursion stays predictable. No
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
    def _potential(x, t):
        return -(x.squeeze(-1) * t)

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
            potential=self._potential,
        )
        assert traj.shape == (self.N_STEPS, self.N, 1)
        assert weight_hist.shape == (self.N_STEPS, self.N)
        assert len(ess_hist) == self.N_STEPS - 1
        assert torch.allclose(weight_hist.sum(dim=1), torch.ones(self.N_STEPS), atol=1e-6)
        assert torch.isin(traj, x0).all()

    def test_ess_trace_matches_analytic_prediction_before_first_resample(self):
        """Until the first gather the particles are untouched, so the ESS trace is a closed
        form: log w accumulates bias·|dt| from a fixed linspace bias."""
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
            potential=self._potential,
        )
        bias = torch.linspace(-self.K, self.K, self.N)
        log_w = torch.zeros(self.N)
        predicted = []
        for t_curr, t_next in zip(t[:-1], t[1:]):
            log_w = log_w + bias * (t_next - t_curr).abs()
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
            potential=self._potential,
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
            potential=self._potential,
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
            potential=self._potential,
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
                potential=self._potential,
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
                potential=self._potential,
            )

    def test_t_must_be_decreasing(self):
        x0 = torch.linspace(-self.K, self.K, self.N).reshape(self.N, 1)
        t = torch.linspace(1e-3, 1.0 - 1e-3, self.N_STEPS)
        with pytest.raises(ValueError, match="strictly decreasing"):
            steered_reverse_churn_sampling(
                self._zero_velocity, self._no_transition, None, x0, t, churn=0.0, potential=self._potential
            )

    def test_negative_churn_rejected(self):
        x0 = torch.linspace(-self.K, self.K, self.N).reshape(self.N, 1)
        t = torch.linspace(1.0 - 1e-3, 1e-3, self.N_STEPS)
        with pytest.raises(ValueError, match="non-negative"):
            steered_reverse_churn_sampling(
                self._zero_velocity, self._no_transition, None, x0, t, churn=-0.5, potential=self._potential
            )


class TestChurnSteeringReducesToUnsteered:
    """A flat potential must leave the base churn sampler bit-for-bit unchanged.

    This is the invariant tying the steered sampler to the unsteered one, the analogue of
    `steered_reverse_sampling` with a zero weight_update reducing to `reverse_sampling`.
    With ρ ≡ 0 every log weight stays 0, so ESS/N ≡ 1, no adaptive resample ever triggers,
    and the mandatory final resample is the identity permutation on uniform weights.
    """

    @pytest.mark.parametrize("churn", [0.0, 0.5, 1.0, 2.0], ids=lambda v: f"churn={v}")
    def test_zero_potential_matches_reverse_churn_sampling(self, churn):
        schedule = BetaSchedule()
        gmm = GMM(
            mu=torch.tensor([-2.0, 0.0, 2.0]).reshape(1, 3, 1),
            sigma=torch.tensor([0.3, 0.3, 0.2]).reshape(1, 3, 1),
            weight=torch.tensor([0.33, 0.5, 0.17]).reshape(1, 3),
            schedule=schedule,
        )
        x0 = torch.randn(256, 1, 1)
        t = torch.linspace(1 - 1e-2, 1e-2, 40)

        def zero_potential(x, t_):
            return torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)

        torch.manual_seed(0)
        unsteered = reverse_churn_sampling(gmm.velocity, schedule.transition, x0, t, churn=churn)
        torch.manual_seed(0)
        steered, ess_hist, weight_hist = steered_reverse_churn_sampling(
            gmm.velocity, schedule.transition, None, x0, t, churn=churn, ess_threshold=0.5, potential=zero_potential
        )

        assert torch.allclose(unsteered, steered, atol=1e-6)
        assert all(e == pytest.approx(1.0, abs=1e-5) for e in ess_hist)
        assert torch.allclose(weight_hist, torch.full_like(weight_hist, 1 / 256), atol=1e-6)


class TestChurnSteeringFinalResampleOnly:
    """Accumulate log weights over the whole trajectory; resample once, at the very end.

    This mirrors the `ess_threshold=1_000` arm that test_steering.py sweeps for the
    Euler-Maruyama sampler (and that the three marginal classes below also sweep), but
    makes a much stronger assertion than those can. With no intermediate resample there is
    no reset and no gather, so the increments telescope across the *entire* trajectory and
    the accumulated weight collapses to a closed form:

        log w(t_i) = ρ_{t_i}(x_i) − ρ_{t_0}(x_0)                        (unguided)
        log w(t_i) = ρ_{t_i}(x_i) − ρ_{t_0}(x_0) + Σ_{j<i} corr_j·span_j  (guided)

    Checking that identity is exact arithmetic rather than a Monte-Carlo statistic, so it
    pins down the weight bookkeeping — the telescoping, the carried ρ, and the
    accumulation of the guidance compensation — to floating-point precision. A W1 check
    cannot do that: it only sees the weights through a resampled cloud, and intermediate
    resampling repeatedly resets log w and hides any drift in the accumulation.

    Run in float64 so "exact" means ~1e-12 rather than "within the float32 noise".
    """

    N_PARTICLES = 512
    N_STEPS = 60
    EPS = 1e-2
    CHURN = 1.0
    REWARD_CENTER = -2.0
    REWARD_SIGMA = 1.0
    # Interval far exceeding the step count: no intermittent resample can ever fire, so
    # only the mandatory final one does.
    FINAL_ONLY = 10_000

    @pytest.fixture
    def setup(self):
        sched = BetaSchedule(beta_min=0.1, beta_max=20.0)
        gmm = GMM(
            mu=torch.tensor([[[-2.5], [0], [2.5]]], dtype=torch.float64),
            sigma=torch.tensor([[[0.8], [0.8], [0.8]]], dtype=torch.float64),
            weight=torch.tensor([[0.2, 0.6, 0.2]], dtype=torch.float64),
            schedule=sched,
        )
        return gmm, sched

    def test_unguided_log_weights_telescope_to_the_endpoint_difference(self, setup):
        gmm, sched = setup

        def r(x):
            return -0.5 * (x - self.REWARD_CENTER) ** 2 / self.REWARD_SIGMA**2

        def beta_fn(t_):
            return 1.0 - t_

        def rho(x, t_):
            return (beta_fn(t_) * r(x)).squeeze(-1).squeeze(-1)

        torch.manual_seed(0)
        t = torch.linspace(1 - self.EPS, self.EPS, self.N_STEPS, dtype=torch.float64)
        x0 = gmm.sample(shape=self.N_PARTICLES, t=1 - self.EPS)
        traj, _, weight_hist = steered_reverse_churn_sampling(
            gmm.velocity,
            sched.transition,
            None,
            x0,
            t,
            churn=self.CHURN,
            ess_threshold=self.FINAL_ONLY,
            potential=rho,
        )
        rho_0 = rho(traj[0], t[0])
        # Index -1 is overwritten by the mandatory final resample, so check up to -2.
        for i in range(1, self.N_STEPS - 1):
            expected = torch.softmax(rho(traj[i], t[i]) - rho_0, dim=0)
            assert torch.allclose(weight_hist[i], expected, atol=1e-12), (
                f"step {i}: accumulated weights do not equal softmax(ρ_i − ρ_0); "
                f"max deviation {(weight_hist[i] - expected).abs().max():.3e}"
            )

    def test_guided_log_weights_add_the_accumulated_compensation(self, setup):
        """Same identity with a guidance field, which adds Σ corr·span on top.

        Recording `corr` from inside the callable is what makes this checkable: the
        compensation is evaluated at the reheated state x̂, which never appears in the
        returned trajectory, so it cannot be reconstructed after the fact.
        """
        gmm, sched = setup
        c = 0.5
        recorded = []

        def r(x):
            return -0.5 * (x - self.REWARD_CENTER) ** 2 / self.REWARD_SIGMA**2

        def beta_fn(t_):
            return 1.0 - t_

        def rho(x, t_):
            return (beta_fn(t_) * r(x)).squeeze(-1).squeeze(-1)

        def _grad_rho(x, t_hat):
            return -(1.0 - t_hat) * (x - self.REWARD_CENTER) / self.REWARD_SIGMA**2

        def guided_drift(x, t_hat):
            g2 = sched.diffusion_coeff(t_hat) ** 2
            return gmm.velocity(x, t_hat) + c * (g2 / 2) * _grad_rho(x, t_hat)

        def compensation(x, t_hat, dt):
            g2 = sched.diffusion_coeff(t_hat) ** 2
            grad_rho = _grad_rho(x, t_hat)
            lap_rho = -(1.0 - t_hat) / self.REWARD_SIGMA**2  # D = 1
            corr = c * (g2 / 2) * (lap_rho + (gmm.score(x, t_hat) * grad_rho).sum(-1).squeeze(-1))
            span = dt - self.CHURN * dt.abs()
            recorded.append((corr * span).clone())
            return corr * span

        torch.manual_seed(0)
        t = torch.linspace(1 - self.EPS, self.EPS, self.N_STEPS, dtype=torch.float64)
        x0 = gmm.sample(shape=self.N_PARTICLES, t=1 - self.EPS)
        traj, _, weight_hist = steered_reverse_churn_sampling(
            guided_drift,
            sched.transition,
            compensation,
            x0,
            t,
            churn=self.CHURN,
            ess_threshold=self.FINAL_ONLY,
            potential=rho,
        )
        assert len(recorded) == self.N_STEPS - 1, "compensation must be called once per step"

        rho_0 = rho(traj[0], t[0])
        accumulated = torch.zeros(self.N_PARTICLES, dtype=torch.float64)
        for i in range(1, self.N_STEPS - 1):
            accumulated = accumulated + recorded[i - 1]  # corr·span for that step
            expected = torch.softmax(rho(traj[i], t[i]) - rho_0 + accumulated, dim=0)
            assert torch.allclose(weight_hist[i], expected, atol=1e-12), (
                f"step {i}: guided weights do not equal softmax(ρ_i − ρ_0 + Σ corr·span); "
                f"max deviation {(weight_hist[i] - expected).abs().max():.3e}"
            )

    def test_weights_are_never_reset_before_the_end(self, setup):
        """Guard on the premise: if an intermittent resample fired, the identities above
        would hold vacuously on a uniform row."""
        gmm, sched = setup

        def r(x):
            return -0.5 * (x - self.REWARD_CENTER) ** 2 / self.REWARD_SIGMA**2

        def beta_fn(t_):
            return 1.0 - t_

        def rho(x, t_):
            return (beta_fn(t_) * r(x)).squeeze(-1).squeeze(-1)

        torch.manual_seed(0)
        t = torch.linspace(1 - self.EPS, self.EPS, self.N_STEPS, dtype=torch.float64)
        x0 = gmm.sample(shape=self.N_PARTICLES, t=1 - self.EPS)
        traj, _, weight_hist = steered_reverse_churn_sampling(
            gmm.velocity,
            sched.transition,
            None,
            x0,
            t,
            churn=self.CHURN,
            ess_threshold=self.FINAL_ONLY,
            potential=rho,
        )

        if PLOT:
            # The weighted cloud is the object the identity is about: with no intermittent
            # resample the particles themselves stay on the *base* marginal q_t and only the
            # accumulated weights carry the tilt, so the green histogram tracking the red
            # curve is the visual form of the closed-form check above.
            plot_churn_marginal_comparison(
                runs=[("final resample only", t, traj, weight_hist)],
                gmm=gmm,
                beta_fn=beta_fn,
                reward_fn=lambda x, t_: r(x),
                time_indices=[15, 30, 45, self.N_STEPS - 1],
                reward_center=self.REWARD_CENTER,
                out_dir=_plot_dir(
                    "test_churn_steering",
                    "TestChurnSteeringFinalResampleOnly",
                    "test_weights_are_never_reset_before_the_end",
                ),
                title_prefix=(
                    f"Churn-FKC, weights accumulated over the whole trajectory | BetaSchedule, "
                    f"churn={self.CHURN}, {self.N_PARTICLES} particles"
                ),
            )

        uniform = torch.full((self.N_PARTICLES,), 1 / self.N_PARTICLES, dtype=torch.float64)
        for i in range(1, self.N_STEPS - 1):
            assert not torch.allclose(weight_hist[i], uniform, atol=1e-9), f"weights reset at step {i}"
        assert torch.allclose(weight_hist[-1], uniform, atol=1e-12), "final resample must restore uniform weights"


@pytest.mark.slow
class TestSteeredChurnGuidedFlow:
    """Guiding the deterministic half, and weighting the churn half explicitly.

    With `guidance` the probability flow integrates `velocity + u`, moving part of the tilt
    out of the weights and into the dynamics. A deterministic half carries no diffusion, so
    unlike Prop. D.6 the reward Laplacian does not cancel and the compensation

        ∇·u + ⟨s_t, u⟩   (integrated over the transport span)

    must be applied explicitly. Here u = c·(g²/2)·∇ρ, for which ∇·u = c·(g²/2)·Δρ is
    analytic (Δρ = −β(t)·D/ς² for a Gaussian reward). c is a free knob: the construction is
    exact for any field, so these must all land on the same tilted marginals.
    """

    EPS = 0.001
    N_PARTICLES = 10_000
    N_STEPS = 500
    REWARD_SIGMA = 1.0
    CHURN = 1.0

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

    @pytest.mark.parametrize("c", [0.25, 0.5, 1.0], ids=lambda v: f"c={v}")
    @pytest.mark.parametrize("reward_center", [-2.0, 1.0], ids=lambda v: f"center={v}")
    def test_guided_flow_matches_tilted_intermediate_marginals(self, setup, reward_center, c):
        gmm, sched = setup

        def r(x):
            return -0.5 * (x - reward_center) ** 2 / self.REWARD_SIGMA**2

        def grad_rho(x, t):
            return -(1.0 - t) * (x - reward_center) / self.REWARD_SIGMA**2

        def beta_fn(t):
            return 1.0 - t

        def potential(x, t):
            return (beta_fn(t) * r(x)).squeeze(-1).squeeze(-1)

        def guided_drift(x, t):
            g2 = sched.diffusion_coeff(t) ** 2
            return gmm.velocity(x, t) + c * (g2 / 2) * grad_rho(x, t)

        def compensation(x, t, dt):
            """[∇·u + ⟨s,u⟩]·span, evaluated at the reheated state. The sampler hands us dt
            for the whole grid step, so recover the transport span as it does: dt − churn·|dt|."""
            g2 = sched.diffusion_coeff(t) ** 2
            lap_rho = -beta_fn(t) / self.REWARD_SIGMA**2  # D = 1
            corr = c * (g2 / 2) * (lap_rho + (gmm.score(x, t) * grad_rho(x, t)).sum(-1).squeeze(-1))
            return corr * (dt - self.CHURN * dt.abs())

        torch.manual_seed(0)
        t = torch.linspace(1 - self.EPS, self.EPS, self.N_STEPS)
        x0 = gmm.sample(shape=self.N_PARTICLES, t=1 - self.EPS)
        traj, _, weight_hist = steered_reverse_churn_sampling(
            guided_drift,
            sched.transition,
            compensation,
            x0,
            t,
            churn=self.CHURN,
            ess_threshold=25,
            potential=potential,
        )
        assert traj.shape == (self.N_STEPS, self.N_PARTICLES, 1, 1)
        assert weight_hist.shape == (self.N_STEPS, self.N_PARTICLES)

        if PLOT:
            plot_churn_marginal_comparison(
                runs=[(f"guided c={c}", t, traj, weight_hist)],
                gmm=gmm,
                beta_fn=beta_fn,
                reward_fn=lambda x, t__: r(x),
                time_indices=[100, 250, 400, self.N_STEPS - 1],
                reward_center=reward_center,
                out_dir=_plot_dir(
                    "test_churn_steering",
                    "TestSteeredChurnGuidedFlow",
                    "test_guided_flow_matches_tilted_intermediate_marginals",
                    f"center{reward_center}",
                ),
                title_prefix=f"Guided churn-FKC | BetaSchedule, churn=1.0, c={c}, reward center={reward_center}",
            )

        xs = torch.linspace(-8, 8, 400).reshape(-1, 1, 1)
        for t_idx in range(10, self.N_STEPS, 10):  # 49 intermediate slots
            t_ = t[t_idx]
            sigma_t = sched.get_sigma_t(t_)
            p_tilt = _tilted_density(gmm, xs, t_, beta_fn, lambda x, t__: r(x))
            w1 = _weighted_wasserstein1(traj[t_idx, :, 0, 0], weight_hist[t_idx], xs.squeeze(), p_tilt)
            tol = 0.05 * (1.0 + sigma_t.item())
            assert w1 < tol, f"guided c={c} center={reward_center} t={t_:.3f}: W1={w1:.4f} >= {tol:.4f}"

    @pytest.mark.parametrize("c", [0.5, 1.0], ids=lambda v: f"c={v}")
    def test_dropping_the_compensation_breaks_the_marginals(self, setup, c):
        """Guard: the ∇·u + ⟨s,u⟩ term must be load-bearing, not decorative.

        Without it the sampler still produces high-reward samples — it just targets the
        wrong distribution, which is exactly the failure mode a reward-only check misses.
        Deleting the term must therefore make W1 much worse, not marginally worse.
        """
        gmm, sched = setup
        reward_center = -2.0

        def r(x):
            return -0.5 * (x - reward_center) ** 2 / self.REWARD_SIGMA**2

        def grad_rho(x, t):
            return -(1.0 - t) * (x - reward_center) / self.REWARD_SIGMA**2

        def beta_fn(t):
            return 1.0 - t

        def potential(x, t):
            return (beta_fn(t) * r(x)).squeeze(-1).squeeze(-1)

        def guided_drift(x, t):
            g2 = sched.diffusion_coeff(t) ** 2
            return gmm.velocity(x, t) + c * (g2 / 2) * grad_rho(x, t)

        def compensation(x, t, dt):
            g2 = sched.diffusion_coeff(t) ** 2
            lap_rho = -beta_fn(t) / self.REWARD_SIGMA**2  # D = 1
            corr = c * (g2 / 2) * (lap_rho + (gmm.score(x, t) * grad_rho(x, t)).sum(-1).squeeze(-1))
            return corr * (dt - self.CHURN * dt.abs())

        def no_compensation(x, t, dt):
            return torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)

        xs = torch.linspace(-8, 8, 400).reshape(-1, 1, 1)
        t = torch.linspace(1 - self.EPS, self.EPS, self.N_STEPS)

        def run(weight_update):
            torch.manual_seed(0)
            x0 = gmm.sample(shape=self.N_PARTICLES, t=1 - self.EPS)
            traj, _, weight_hist = steered_reverse_churn_sampling(
                guided_drift,
                sched.transition,
                weight_update,
                x0,
                t,
                churn=self.CHURN,
                ess_threshold=25,
                potential=potential,
            )
            w1s = []
            for t_idx in range(10, self.N_STEPS, 10):
                p_tilt = _tilted_density(gmm, xs, t[t_idx], beta_fn, lambda x, t__: r(x))
                w1s.append(_weighted_wasserstein1(traj[t_idx, :, 0, 0], weight_hist[t_idx], xs.squeeze(), p_tilt))
            return traj, weight_hist, sum(w1s) / len(w1s), len(w1s)

        traj_ok, w_ok, mean_ok, n_slots = run(compensation)
        traj_bad, w_bad, mean_bad, _ = run(no_compensation)

        if PLOT:
            plot_churn_marginal_comparison(
                runs=[
                    (f"guided c={c}, compensation ON", t, traj_ok, w_ok),
                    (f"guided c={c}, compensation OFF", t, traj_bad, w_bad),
                ],
                gmm=gmm,
                beta_fn=beta_fn,
                reward_fn=lambda x, t__: r(x),
                time_indices=[100, 250, 400, self.N_STEPS - 1],
                reward_center=reward_center,
                out_dir=_plot_dir(
                    "test_churn_steering",
                    "TestSteeredChurnGuidedFlow",
                    "test_dropping_the_compensation_breaks_the_marginals",
                    f"c{c}",
                ),
                title_prefix=(
                    f"Guidance compensation ∇·u + ⟨s,u⟩ ablation | u = {c}·(g²/2)·∇ρ | "
                    f"mean W1 over {n_slots} slots: {mean_ok:.4f} with it, {mean_bad:.4f} without"
                ),
            )

        assert mean_bad > 4 * mean_ok, (
            f"c={c}: dropping the compensation changed mean W1 only {mean_ok:.4f} -> "
            f"{mean_bad:.4f}; the term is not being exercised by this configuration"
        )


@pytest.mark.slow
class TestSteeredChurnBetaIntermediateMarginals:
    """Weighted intermediate-time checks under BetaSchedule, direct reward r(x_t).

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

        def r(x):
            return -0.5 * (x - reward_center) ** 2 / reward_sigma**2

        def beta_fn(t):
            return 1.0 - t

        def potential(x, t):
            return (beta_fn(t) * r(x)).squeeze(-1).squeeze(-1)

        torch.manual_seed(0)
        t = torch.linspace(1 - self.EPS, self.EPS, self.N_STEPS)
        x0 = gmm.sample(shape=self.N_PARTICLES, t=1 - self.EPS)
        traj, ess_hist, weight_hist = steered_reverse_churn_sampling(
            gmm.velocity, sched.transition, None, x0, t, churn=churn, ess_threshold=ess_threshold, potential=potential
        )
        assert traj.shape == (self.N_STEPS, self.N_PARTICLES, 1, 1)
        assert weight_hist.shape == (self.N_STEPS, self.N_PARTICLES)

        xs = torch.linspace(-8, 8, 400).reshape(-1, 1, 1)
        xs_flat = xs.squeeze()
        for t_idx in range(10, self.N_STEPS, 10):  # 49 intermediate slots
            t_ = t[t_idx]
            sigma_t = sched.get_sigma_t(t_)
            p_tilt = _tilted_density(gmm, xs, t_, beta_fn, lambda x, t__: r(x))
            w1 = _weighted_wasserstein1(traj[t_idx, :, 0, 0], weight_hist[t_idx], xs_flat, p_tilt)
            tol = 0.05 * (1.0 + sigma_t.item())
            assert w1 < tol, (
                f"BetaSchedule churn={churn} center={reward_center} ess={ess_threshold} "
                f"t={t_:.3f}: W1={w1:.4f} >= {tol:.4f}"
            )


@pytest.mark.slow
class TestSteeredChurnKarrasIntermediateMarginals:
    """Weighted intermediate-time checks under KarrasSchedule, direct reward r(x_t).

    Mirrors TestSteeredSamplingKarrasIntermediateMarginals, tolerance included.
    """

    EPS = 0.001
    T_NOISE = 1 - EPS
    # 10k rather than the EM suite's 5k: the terminal-W1 spread over seeds is sd ~0.010 at
    # 5k against a ~0.05 tolerance, so single configs sit within one sigma of the bound and
    # pass or fail on the draw. 10k halves that and makes the assertion about the sampler.
    N_PARTICLES = 10_000
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

    @pytest.mark.parametrize("churn", [0.5, 1.0], ids=lambda v: f"churn={v}")
    @pytest.mark.parametrize("ess_threshold", [0.9, 25, 1000], ids=lambda v: f"ess={v}")
    @pytest.mark.parametrize("reward_center", [-1.0, -0.5, 0.5], ids=lambda v: f"center={v}")
    def test_weighted_karras_churn_matches_tilted_intermediate_marginals(
        self, setup, reward_center, ess_threshold, churn
    ):
        gmm, sched = setup
        reward_sigma = 1.0

        def r(x):
            return -0.5 * (x - reward_center) ** 2 / reward_sigma**2

        def beta_fn(t):
            g = sched.diffusion_coeff(t)
            return (1 - t) ** 2 / (1 + g**2)

        def potential(x, t):
            return (beta_fn(t) * r(x)).squeeze(-1).squeeze(-1)

        torch.manual_seed(0)
        t = torch.linspace(self.T_NOISE, self.EPS, self.N_STEPS)
        x0 = gmm.sample(shape=self.N_PARTICLES, t=self.T_NOISE)
        traj, _, weight_hist = steered_reverse_churn_sampling(
            gmm.velocity, sched.transition, None, x0, t, churn=churn, ess_threshold=ess_threshold, potential=potential
        )
        assert traj.shape == (self.N_STEPS, self.N_PARTICLES, 1, 1)

        for t_idx in range(50, self.N_STEPS, 10):  # 56 intermediate slots
            t_ = t[t_idx]
            sigma_t = sched.get_sigma_t(t_)
            grid_radius = max(4.0, 6.0 * sigma_t.item())
            xs = torch.linspace(-grid_radius, grid_radius, 400).reshape(-1, 1, 1)
            p_tilt = _tilted_density(gmm, xs, t_, beta_fn, lambda x, t__: r(x))
            w1 = _weighted_wasserstein1(traj[t_idx, :, 0, 0], weight_hist[t_idx], xs.squeeze(), p_tilt)
            tol = 0.05 * (1.0 + sigma_t.item())
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
                grid_radius = max(4.0, 6.0 * sched.get_sigma_t(t[t_idx]).item())
                plot_churn_marginal_comparison(
                    runs=[(f"churn={churn}", t, traj, weight_hist)],
                    gmm=gmm,
                    beta_fn=beta_fn,
                    reward_fn=lambda x, t__: r(x),
                    time_indices=[t_idx],
                    reward_center=reward_center,
                    out_dir=_plot_dir(
                        "test_churn_steering",
                        "TestSteeredChurnKarrasIntermediateMarginals",
                        "test_weighted_karras_churn_matches_tilted_intermediate_marginals",
                        f"center{reward_center}",
                        f"ess{ess_threshold}",
                    ),
                    title_prefix=(
                        f"Churn-FKC | KarrasSchedule, direct reward r(x_t), churn={churn}, "
                        f"ess={ess_threshold}, center={reward_center}"
                    ),
                    grid_radius=grid_radius,
                )


@pytest.mark.slow
class TestSteeredChurnDenoisingKarras(KarrasDenoiseMixin):
    """KarrasSchedule with a reward on denoised x̂₀ — the pattern real reward models use.

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
    # from the direct-reward class above, which is far cheaper — no denoiser in the loop.
    @pytest.mark.parametrize("ess_threshold", [0.9, 25, 1000], ids=lambda v: f"ess={v}")
    @pytest.mark.parametrize("reward_center", [-1.0, -0.5, 0.5], ids=lambda v: f"center={v}")
    @pytest.mark.parametrize("n_denoise_steps", [10], ids=lambda v: f"{v}")
    def test_weighted_denoising_karras_churn_matches_tilted_marginals(
        self, setup, reward_center, ess_threshold, n_denoise_steps
    ):
        gmm, sched = setup
        reward_sigma = 1.0
        churn = 1.0

        def r(x0_hat):
            return -0.5 * (x0_hat - reward_center) ** 2 / reward_sigma**2

        def beta_fn(t):
            g = sched.diffusion_coeff(t)
            return (1 - t) / (1 + g**2)

        def reward_on(x, t_):
            return r(self._denoise(gmm, sched, x, t_, n_denoise_steps)[0].detach())

        def potential(x, t_):
            return (beta_fn(t_) * reward_on(x, t_)).squeeze(-1).squeeze(-1)

        torch.manual_seed(0)
        t = torch.linspace(self.T_NOISE, self.EPS, self.N_STEPS)
        x0 = gmm.sample(shape=self.N_PARTICLES, t=self.T_NOISE)
        traj, _, weight_hist = steered_reverse_churn_sampling(
            gmm.velocity, sched.transition, None, x0, t, churn=churn, ess_threshold=ess_threshold, potential=potential
        )
        assert traj.shape == (self.N_STEPS, self.N_PARTICLES, 1, 1)

        for t_idx in range(50, self.N_STEPS, 10):  # 56 intermediate slots
            t_ = t[t_idx]
            sigma_t = sched.get_sigma_t(t_)
            grid_radius = max(4.0, 6.0 * sigma_t.item())
            xs = torch.linspace(-grid_radius, grid_radius, 400).reshape(-1, 1, 1)
            p_tilt = _tilted_density(gmm, xs, t_, beta_fn, reward_on)
            w1 = _weighted_wasserstein1(traj[t_idx, :, 0, 0], weight_hist[t_idx], xs.squeeze(), p_tilt)
            # Higher floor than the direct-reward tests: the unrolled denoiser adds its own
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
                grid_radius = max(4.0, 6.0 * sched.get_sigma_t(t[t_idx]).item())
                plot_churn_marginal_comparison(
                    runs=[(f"churn={churn}", t, traj, weight_hist)],
                    gmm=gmm,
                    beta_fn=beta_fn,
                    reward_fn=reward_on,
                    time_indices=[t_idx],
                    reward_center=reward_center,
                    out_dir=_plot_dir(
                        "test_churn_steering",
                        "TestSteeredChurnDenoisingKarras",
                        "test_weighted_denoising_karras_churn_matches_tilted_marginals",
                        f"center{reward_center}",
                        f"ess{ess_threshold}",
                    ),
                    title_prefix=(
                        f"Churn-FKC | KarrasSchedule, denoised reward r(D(x_t,t)), churn={churn}, "
                        f"ess={ess_threshold}, center={reward_center}"
                    ),
                    grid_radius=grid_radius,
                )

        # Terminal marginal: the headline number, directly comparable to the EM sampler's
        # `assert w1_rew < 0.075` in TestSteeredSamplingKarrasFinalMarginal.
        xs = torch.linspace(-3, 3, 100).reshape(-1, 1, 1)
        xs_flat = xs.squeeze()
        p_rew = _tilted_density(gmm, xs, torch.tensor(self.EPS), beta_fn, reward_on)
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
                    f"Churn FKC terminal marginal | KarrasSchedule, denoised reward\n"
                    f"churn={churn}, ess={ess_threshold}, center={reward_center} | W1={w1_terminal:.4f}"
                ),
                out_path=(
                    _plot_dir(
                        "test_churn_steering",
                        "TestSteeredChurnDenoisingKarras",
                        "test_weighted_denoising_karras_churn_matches_tilted_marginals",
                    )
                    / f"churn{churn}_center{reward_center}_ess{ess_threshold}.png"
                ),
                min_x=-3,
                max_x=3,
            )
        assert w1_terminal < 0.075, (
            f"Karras denoising terminal churn={churn} center={reward_center} ess={ess_threshold}: W1={w1_terminal:.4f}"
        )


def plot_churn_marginal_comparison(
    runs,
    gmm,
    beta_fn,
    reward_fn,
    time_indices,
    reward_center,
    out_dir,
    title_prefix,
    grid_radius=8.0,
    n_grid=400,
):
    """Save one `plot_marginal_density_comparison` panel per (run, time index).

    Thin loop over test_steering.py's shared plotter — base q_t, analytic tilted pi_t,
    and both the unweighted and weighted empirical densities — so the churn plots look
    exactly like the Euler-Maruyama ones. Weighted matters: SMC particles are only
    correct as a weighted cloud, so between resamples the unweighted histogram is
    expected to be wrong (see tests/CLAUDE.md).

    Args:
        runs:          [(label, t, trajectory, weight_history)]; label goes in the filename.
        gmm, beta_fn:  build the analytic curves; reward_fn is (x, t) -> reward.
        time_indices:  trajectory indices to plot, one PNG each.
        reward_center: drawn as a vertical marker.
        out_dir:       directory for the PNGs; created if missing.
        title_prefix:  first title line, shared across panels.
    """
    dtype = runs[0][2].dtype  # the trajectory's dtype; some runs are float64
    xs = torch.linspace(-grid_radius, grid_radius, n_grid, dtype=dtype).reshape(-1, 1, 1)
    xs_flat = xs.squeeze()

    for label, t, traj, weight_hist in runs:
        for t_idx in time_indices:
            t_ = t[t_idx]
            p_base = gmm.log_prob(xs, t=t_).squeeze().exp()
            p_base = p_base / torch.trapezoid(p_base, xs_flat)
            p_tilt = _tilted_density(gmm, xs, t_, beta_fn, reward_fn)
            w1 = _weighted_wasserstein1(traj[t_idx, :, 0, 0], weight_hist[t_idx], xs_flat, p_tilt)
            slug = label.replace(" ", "").replace(",", "_").replace("=", "")
            plot_marginal_density_comparison(
                xs_flat=xs_flat,
                p_data=p_base,
                p_rew=p_tilt,
                samples=traj[t_idx, :, 0, 0],
                weights=weight_hist[t_idx],
                reward_center=reward_center,
                title=f"{title_prefix}\n{label} | t_idx={t_idx}, t={t_:.3f} | weighted W1={w1:.4f}",
                out_path=Path(out_dir) / f"{slug}_t{t_idx}.png",
                min_x=-grid_radius,
                max_x=grid_radius,
            )


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

        def r(x):
            return -0.5 * (x - reward_center) ** 2 / self.REWARD_SIGMA**2

        def grad_r(x):
            return -(x - reward_center) / self.REWARD_SIGMA**2

        def beta_fn(t):
            return 1.0 - t

        # Byte-for-byte the fkc_weight_update of test_steering.py -- no churn anywhere in it.
        def fkc_weight_update(x, t, dt):
            f = sched.forward_drift(x, t)
            sigma = sched.diffusion_coeff(t)
            score = gmm.score(x, t)
            rg, rv = grad_r(x), r(x)
            beta = beta_fn(t)
            term1 = -_dbeta_dt(beta_fn, t) * rv
            term2 = -(beta * rg) * f
            term3 = (beta * rg) * (sigma**2 / 2) * score
            return (term1 + term2 + term3).squeeze(-1).squeeze(-1) * dt.abs()

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
            return gmm.velocity(x, t) - a * grad_r(x) / (1.0 + churn)

        torch.manual_seed(0)
        t = torch.linspace(1 - self.EPS, self.EPS, self.N_STEPS)
        x0 = gmm.sample(shape=self.N_PARTICLES, t=1 - self.EPS)
        traj, _, weight_hist = steered_reverse_churn_sampling(
            guided_drift, sched.transition, fkc_weight_update, x0, t, churn=churn, ess_threshold=25
        )

        xs = torch.linspace(-8, 8, 400).reshape(-1, 1, 1)
        for t_idx in range(10, self.N_STEPS, 10):  # 49 slots
            t_ = t[t_idx]
            p_tilt = _tilted_density(gmm, xs, t_, beta_fn, lambda x, t__: r(x))
            w1 = _weighted_wasserstein1(traj[t_idx, :, 0, 0], weight_hist[t_idx], xs.squeeze(), p_tilt)
            tol = 0.05 * (1.0 + sched.get_sigma_t(t_).item())
            assert w1 < tol, (
                f"EM weight on churn sampler, churn={churn} center={reward_center} t={t_:.3f}: W1={w1:.4f} >= {tol:.4f}"
            )
