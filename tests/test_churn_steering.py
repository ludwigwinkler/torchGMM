"""FKC steering on the churn sampler — the same assertions test_steering.py makes of the
Euler-Maruyama steered sampler, retargeted at `steered_reverse_churn_sampling`.

The whole point of the comparison is that the two samplers reach the *same* tilted
marginals p_t ∝ q_t·exp(ρ_t) by structurally different routes: the EM sampler guides the
drift and carries a continuous-time weight (β̇r, ∂_t r, score-alignment; Prop. D.6), while
the churn sampler leaves its probability flow untouched and carries only the endpoint
difference ρ_{t+dt}(x_{t+dt}) − ρ_t(x_t) (docs/fkc_churn_steering.md §3). So the reward
plumbing here is deliberately much smaller — no gradients, no β̇ — and the marginal
assertions are deliberately identical.
"""

import pytest
import torch
from test_steering import (
    PLOT,
    KarrasDenoiseMixin,
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

    def _bias_x0(self):
        return torch.linspace(-self.K, self.K, self.N).reshape(self.N, 1)

    def _run(self, ess_threshold):
        x0 = self._bias_x0()
        t = torch.linspace(1.0 - 1e-3, 1e-3, self.N_STEPS)
        return (
            x0,
            t,
            steered_reverse_churn_sampling(
                self._zero_velocity,
                self._no_transition,
                self._potential,
                x0,
                t,
                churn=0.0,
                ess_threshold=ess_threshold,
            ),
        )

    def test_shapes_and_normalisation(self):
        x0, _, (traj, ess_hist, weight_hist) = self._run(0.7)
        assert traj.shape == (self.N_STEPS, self.N, 1)
        assert weight_hist.shape == (self.N_STEPS, self.N)
        assert len(ess_hist) == self.N_STEPS - 1
        assert torch.allclose(weight_hist.sum(dim=1), torch.ones(self.N_STEPS), atol=1e-6)
        assert torch.isin(traj, x0).all()

    def test_ess_trace_matches_analytic_prediction_before_first_resample(self):
        """Until the first gather the particles are untouched, so the ESS trace is a closed
        form: log w accumulates bias·|dt| from a fixed linspace bias."""
        _, t, (_, ess_hist, _) = self._run(1_000)  # interval >> steps: no intermittent resample
        bias = torch.linspace(-self.K, self.K, self.N)
        log_w = torch.zeros(self.N)
        predicted = []
        for t_curr, t_next in zip(t[:-1], t[1:]):
            log_w = log_w + bias * (t_next - t_curr).abs()
            predicted.append(_ess_ratio(log_w))
        assert ess_hist == pytest.approx(predicted, abs=1e-6)

    def test_interval_mode_resets_weights_on_schedule(self):
        interval = 5
        _, _, (traj, _, weight_hist) = self._run(interval)
        uniform = torch.full((self.N,), 1 / self.N)
        for step in range(1, self.N_STEPS):
            expect_uniform = step % interval == 0 or step == self.N_STEPS - 1
            is_uniform = torch.allclose(weight_hist[step], uniform, atol=1e-6)
            assert is_uniform == expect_uniform, f"step {step}: uniform={is_uniform}, expected {expect_uniform}"

    def test_threshold_one_resamples_every_step(self):
        _, _, (_, _, weight_hist) = self._run(1)
        uniform = torch.full((self.N,), 1 / self.N)
        assert torch.allclose(weight_hist, uniform.expand_as(weight_hist), atol=1e-6)

    def test_final_resample_always_fires(self):
        _, _, (_, _, weight_hist) = self._run(1_000)
        uniform = torch.full((self.N,), 1 / self.N)
        assert not torch.allclose(weight_hist[-2], uniform, atol=1e-6)
        assert torch.allclose(weight_hist[-1], uniform, atol=1e-6)

    @pytest.mark.parametrize("bad_threshold", [0, -1, -0.5])
    def test_non_positive_threshold_rejected(self, bad_threshold):
        with pytest.raises(ValueError, match="positive"):
            self._run(bad_threshold)

    @pytest.mark.parametrize("bad_threshold", [2.5, 1.5, 10.25])
    def test_non_integer_interval_threshold_rejected(self, bad_threshold):
        with pytest.raises(ValueError, match="whole number"):
            self._run(bad_threshold)

    def test_t_must_be_decreasing(self):
        x0 = self._bias_x0()
        t = torch.linspace(1e-3, 1.0 - 1e-3, self.N_STEPS)
        with pytest.raises(ValueError, match="strictly decreasing"):
            steered_reverse_churn_sampling(self._zero_velocity, self._no_transition, self._potential, x0, t, churn=0.0)

    def test_negative_churn_rejected(self):
        x0 = self._bias_x0()
        t = torch.linspace(1.0 - 1e-3, 1e-3, self.N_STEPS)
        with pytest.raises(ValueError, match="non-negative"):
            steered_reverse_churn_sampling(self._zero_velocity, self._no_transition, self._potential, x0, t, churn=-0.5)


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
            gmm.velocity, schedule.transition, zero_potential, x0, t, churn=churn, ess_threshold=0.5
        )

        assert torch.allclose(unsteered, steered, atol=1e-6)
        assert all(e == pytest.approx(1.0, abs=1e-5) for e in ess_hist)
        assert torch.allclose(weight_hist, torch.full_like(weight_hist, 1 / 256), atol=1e-6)


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

    def _pieces(self, gmm, sched, reward_center, c, drop_correction=False):
        def r(x):
            return -0.5 * (x - reward_center) ** 2 / self.REWARD_SIGMA**2

        def beta_fn(t):
            return 1.0 - t

        def potential(x, t):
            return (beta_fn(t) * r(x)).squeeze(-1).squeeze(-1)

        def guidance(x, t):
            g2 = sched.diffusion_coeff(t) ** 2
            grad_rho = -beta_fn(t) * (x - reward_center) / self.REWARD_SIGMA**2
            u = c * (g2 / 2) * grad_rho
            if drop_correction:
                return u, torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
            lap_rho = -beta_fn(t) / self.REWARD_SIGMA**2  # D = 1
            corr = c * (g2 / 2) * (lap_rho + (gmm.score(x, t) * grad_rho).sum(-1).squeeze(-1))
            return u, corr

        return r, beta_fn, potential, guidance

    def _run(self, gmm, sched, reward_center, c, resample_at_churn=False, drop_correction=False):
        _, _, potential, guidance = self._pieces(gmm, sched, reward_center, c, drop_correction)
        torch.manual_seed(0)
        t = torch.linspace(1 - self.EPS, self.EPS, self.N_STEPS)
        x0 = gmm.sample(shape=self.N_PARTICLES, t=1 - self.EPS)
        traj, ess, weight_hist = steered_reverse_churn_sampling(
            gmm.velocity,
            sched.transition,
            potential,
            x0,
            t,
            churn=1.0,
            ess_threshold=25,
            guidance=guidance,
            resample_at_churn=resample_at_churn,
        )
        return t, traj, weight_hist

    def _w1_profile(self, gmm, sched, reward_center, t, traj, weight_hist):
        r, beta_fn, _, _ = self._pieces(gmm, sched, reward_center, 0.0)
        xs = torch.linspace(-8, 8, 400).reshape(-1, 1, 1)
        out = []
        for t_idx in range(10, self.N_STEPS, 10):  # 49 slots
            p_tilt = _tilted_density(gmm, xs, t[t_idx], beta_fn, lambda x, t__: r(x))
            w1 = _weighted_wasserstein1(traj[t_idx, :, 0, 0], weight_hist[t_idx], xs.squeeze(), p_tilt)
            out.append((t[t_idx], sched.get_sigma_t(t[t_idx]).item(), w1))
        return out

    @pytest.mark.parametrize("resample_at_churn", [False, True], ids=lambda v: f"churn_resample={v}")
    @pytest.mark.parametrize("c", [0.25, 0.5, 1.0], ids=lambda v: f"c={v}")
    @pytest.mark.parametrize("reward_center", [-2.0, 1.0], ids=lambda v: f"center={v}")
    def test_guided_flow_matches_tilted_intermediate_marginals(self, setup, reward_center, c, resample_at_churn):
        gmm, sched = setup
        t, traj, weight_hist = self._run(gmm, sched, reward_center, c, resample_at_churn)
        for t_, sigma_t, w1 in self._w1_profile(gmm, sched, reward_center, t, traj, weight_hist):
            tol = 0.05 * (1.0 + sigma_t)
            assert w1 < tol, (
                f"guided c={c} center={reward_center} churn_resample={resample_at_churn} "
                f"t={t_:.3f}: W1={w1:.4f} >= {tol:.4f}"
            )

    @pytest.mark.parametrize("c", [0.5, 1.0], ids=lambda v: f"c={v}")
    def test_dropping_the_compensation_breaks_the_marginals(self, setup, c):
        """Guard: the ∇·u + ⟨s,u⟩ term must be load-bearing, not decorative.

        Without it the sampler still produces high-reward samples — it just targets the
        wrong distribution, which is exactly the failure mode a reward-only check misses.
        Deleting the term must therefore make W1 much worse, not marginally worse.
        """
        gmm, sched = setup
        t_ok, traj_ok, w_ok = self._run(gmm, sched, -2.0, c)
        t_bad, traj_bad, w_bad = self._run(gmm, sched, -2.0, c, drop_correction=True)
        prof_ok = self._w1_profile(gmm, sched, -2.0, t_ok, traj_ok, w_ok)
        prof_bad = self._w1_profile(gmm, sched, -2.0, t_bad, traj_bad, w_bad)
        mean_ok = sum(w for _, _, w in prof_ok) / len(prof_ok)
        mean_bad = sum(w for _, _, w in prof_bad) / len(prof_bad)
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
            gmm.velocity, sched.transition, potential, x0, t, churn=churn, ess_threshold=ess_threshold
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
            gmm.velocity, sched.transition, potential, x0, t, churn=churn, ess_threshold=ess_threshold
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
        self, setup, reward_center, ess_threshold, n_denoise_steps, churn=1.0
    ):
        gmm, sched = setup
        reward_sigma = 1.0

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
            gmm.velocity, sched.transition, potential, x0, t, churn=churn, ess_threshold=ess_threshold
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
                    _plot_dir("test_weighted_denoising_karras_churn_matches_tilted_marginals")
                    / f"churn{churn}_center{reward_center}_ess{ess_threshold}.png"
                ),
                min_x=-3,
                max_x=3,
            )
        assert w1_terminal < 0.075, (
            f"Karras denoising terminal churn={churn} center={reward_center} ess={ess_threshold}: W1={w1_terminal:.4f}"
        )
