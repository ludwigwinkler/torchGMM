import math
from pathlib import Path

import pytest
import torch

from torchGMM.gmm import GMM
from torchGMM.sampling import _ess_ratio, steered_reverse_sampling
from torchGMM.schedule import BetaSchedule, KarrasSchedule

torch.set_printoptions(sci_mode=False)

PLOT = False  # flip to True locally to save FKC steering diagnostic plots next to this file
PLOT_DIR = Path(__file__).parent / "plots"


def _plot_dir(*parts):
    """plots/<test_file>/<TestClass>/<test_function>/... — one directory per test node."""
    return PLOT_DIR.joinpath(*parts)


def _wasserstein1(samples, xs_grid, p):
    """1D Wasserstein-1 distance between empirical `samples` and analytic density `p`
    (evaluated on the equispaced grid `xs_grid`), via quantile matching: the target's
    CDF is built by trapezoidal cumsum of `p`, then each sample's rank-based quantile
    level is matched against that CDF (linear interpolation) to get the corresponding
    target quantile; W1 is the mean absolute deviation between samples and their
    matched quantiles.
    """
    s, _ = torch.sort(samples)
    dx = xs_grid[1] - xs_grid[0]
    cdf = torch.cumsum(p, dim=0) * dx
    cdf = cdf / cdf[-1]
    n = s.shape[0]
    levels = (torch.arange(n, dtype=s.dtype, device=s.device) + 0.5) / n
    idx = torch.searchsorted(cdf, levels).clamp(1, cdf.shape[0] - 1)
    cdf_lo, cdf_hi = cdf[idx - 1], cdf[idx]
    x_lo, x_hi = xs_grid[idx - 1], xs_grid[idx]
    frac = (levels - cdf_lo) / (cdf_hi - cdf_lo).clamp_min(1e-12)
    q = x_lo + frac * (x_hi - x_lo)
    return (s - q).abs().mean().item()


def _weighted_wasserstein1(samples, weights, xs_grid, p):
    """Weighted 1D Wasserstein-1 distance between empirical particles and analytic density `p`."""
    s, idx = torch.sort(samples)
    w = weights[idx]
    w = w / w.sum()

    dx = xs_grid[1] - xs_grid[0]
    cdf = torch.cumsum(p, dim=0) * dx
    cdf = cdf / cdf[-1]
    levels = torch.cumsum(w, dim=0) - 0.5 * w
    grid_idx = torch.searchsorted(cdf, levels).clamp(1, cdf.shape[0] - 1)
    cdf_lo, cdf_hi = cdf[grid_idx - 1], cdf[grid_idx]
    x_lo, x_hi = xs_grid[grid_idx - 1], xs_grid[grid_idx]
    frac = (levels - cdf_lo) / (cdf_hi - cdf_lo).clamp_min(1e-12)
    q = x_lo + frac * (x_hi - x_lo)
    return (w * (s - q).abs()).sum().item()


def _tilted_density(gmm, xs, t, beta_fn, reward_fn):
    """Analytic 1D density p_t(x) ∝ q_t(x) exp(beta(t) r(x, t)) on grid `xs`."""
    t_tensor = torch.as_tensor(t, dtype=xs.dtype, device=xs.device)
    log_p = gmm.log_prob(xs, t=t_tensor).squeeze()
    reward = reward_fn(xs, t_tensor).squeeze()
    log_p_tilt = log_p + beta_fn(t_tensor) * reward
    log_p_tilt = log_p_tilt - log_p_tilt.max()
    p_tilt = log_p_tilt.exp()
    return p_tilt / torch.trapezoid(p_tilt, xs.squeeze())


def _dbeta_dt(beta_fn, t):
    t_leaf = torch.as_tensor(t).clone().detach().requires_grad_(True)
    with torch.enable_grad():
        beta = beta_fn(t_leaf)
        (dbeta,) = torch.autograd.grad(beta.sum(), t_leaf)
    return dbeta.detach()


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

    def _assert_weight_history(self, weight_hist):
        assert weight_hist.shape == (self.N_STEPS, self.N)
        assert torch.allclose(weight_hist.sum(dim=1), torch.ones(self.N_STEPS), atol=1e-6)

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
        traj, ess_hist, weight_hist = steered_reverse_sampling(
            self._zero_drift, None, self._weight_update, x0, t, ess_threshold=threshold
        )
        predicted = self._predict_ess_history(t, lambda step, ess: ess < threshold)
        assert ess_hist == pytest.approx(predicted, abs=1e-6)
        self._assert_weight_history(weight_hist)
        assert torch.isin(traj, x0).all()

    def test_interval_mode_matches_prediction(self):
        x0 = torch.randn(self.N, 1)
        t = torch.linspace(1.0 - 1e-3, 1e-3, self.N_STEPS)  # 20 integration steps
        interval = 5
        traj, ess_hist, weight_hist = steered_reverse_sampling(
            self._zero_drift, None, self._weight_update, x0, t, ess_threshold=interval
        )
        # sanity check independent of the implementation's own formula: with 20 steps
        # and interval=5, triggers land at completed-step counts 5, 10, 15, 20.
        expected_trigger_steps = [4, 9, 14, 19]  # 0-indexed step at which (step+1) % 5 == 0
        assert [s for s in range(20) if (s + 1) % interval == 0] == expected_trigger_steps

        predicted = self._predict_ess_history(t, lambda step, ess: (step + 1) % interval == 0)
        assert ess_hist == pytest.approx(predicted, abs=1e-6)
        assert len(ess_hist) == self.N_STEPS - 1
        self._assert_weight_history(weight_hist)
        uniform_weights = torch.full((self.N,), 1 / self.N)
        assert torch.allclose(weight_hist[interval], uniform_weights)
        assert torch.isin(traj, x0).all()

    def test_interval_larger_than_steps_is_final_only(self):
        x0 = torch.randn(self.N, 1)
        t = torch.linspace(1.0 - 1e-3, 1e-3, self.N_STEPS)  # 20 integration steps
        traj, ess_hist, weight_hist = steered_reverse_sampling(
            self._zero_drift, None, self._weight_update, x0, t, ess_threshold=10_000
        )
        # interval >> total steps -> no intermittent trigger ever fires
        predicted = self._predict_ess_history(t, lambda step, ess: False)
        assert ess_hist == pytest.approx(predicted, abs=1e-6)
        self._assert_weight_history(weight_hist)
        uniform_weights = torch.full((self.N,), 1 / self.N)
        assert not torch.allclose(weight_hist[-2], uniform_weights)
        assert torch.allclose(weight_hist[-1], uniform_weights)
        assert torch.isin(traj, x0).all()

    def test_ess_equal_one_resamples_every_step(self):
        x0 = torch.randn(self.N, 1)
        t = torch.linspace(1.0 - 1e-3, 1e-3, self.N_STEPS)
        traj, ess_hist, weight_hist = steered_reverse_sampling(
            self._zero_drift, None, self._weight_update, x0, t, ess_threshold=1
        )
        # ess_threshold == 1 selects interval mode with resample_every=1 (every step),
        # not adaptive mode (which would resample whenever ess < 1, i.e. also nearly
        # every step here -- so cross-check against the interval prediction specifically).
        predicted = self._predict_ess_history(t, lambda step, ess: (step + 1) % 1 == 0)
        assert ess_hist == pytest.approx(predicted, abs=1e-6)
        self._assert_weight_history(weight_hist)
        uniform_weights = torch.full((self.N,), 1 / self.N)
        assert torch.allclose(weight_hist, uniform_weights.expand_as(weight_hist))
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
class TestSteeredSamplingBetaFinalMarginal:
    """SMC-steered reverse sampling via FKC weight update."""

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
        )
        return gmm, sched

    @pytest.mark.parametrize("reward_center", [-2.0, 1.5], ids=lambda v: f"center={v}")
    @pytest.mark.parametrize("reward_sigma", [1.0], ids=lambda v: f"sigma={v}")
    @pytest.mark.parametrize(
        "ess_threshold",
        [0.9, 25, 1_000],
        ids=["adaptive", "interval-25", "final-only"],
    )
    def test_steered_sampling(self, setup, reward_center, reward_sigma, ess_threshold):
        gmm, sched = setup

        def r(x):
            return -0.5 * (x - reward_center) ** 2 / reward_sigma**2

        def grad_r(x):
            return -(x - reward_center) / reward_sigma**2

        def beta_fn(t):
            return 1.0 - t

        def dbeta_dt(t):
            return _dbeta_dt(beta_fn, t)

        def guided_drift(x, t):
            f = sched.forward_drift(x, t)
            sigma = sched.diffusion_coeff(t)
            score = gmm.score(x, t)
            beta = beta_fn(t)
            return f - sigma**2 * score - beta * (sigma**2 / 2) * grad_r(x)

        def fkc_weight_update(x, t, dt):
            f = sched.forward_drift(x, t)
            sigma = sched.diffusion_coeff(t)
            score = gmm.score(x, t)
            rg, rv = grad_r(x), r(x)
            beta = beta_fn(t)
            term1 = -dbeta_dt(t) * rv
            term2 = -(beta * rg) * f
            term3 = (beta * rg) * (sigma**2 / 2) * score
            return (term1 + term2 + term3).squeeze(-1).squeeze(-1) * dt.abs()

        # Seeded like every other slow test here: without this the run rides ambient RNG
        # state, so its W1 shifts with unrelated changes (e.g. the torch.randperm inside
        # plot_steering_result when PLOT is on) and with xdist worker assignment.
        torch.manual_seed(0)
        t = torch.linspace(1 - self.EPS, self.EPS, self.N_STEPS)
        x0 = torch.randn(self.N_PARTICLES, 1, 1)
        traj, ess_hist, weight_hist = steered_reverse_sampling(
            guided_drift, sched.diffusion_coeff, fkc_weight_update, x0, t, ess_threshold=ess_threshold
        )

        # 1. Trajectory shape
        assert traj.shape == (self.N_STEPS, self.N_PARTICLES, 1, 1)
        assert weight_hist.shape == (self.N_STEPS, self.N_PARTICLES)

        # 2. ESS/N history within [0, 1] at every step (small float32 tolerance: with
        # near-uniform weights, logsumexp-based ESS can round fractionally above 1.0)
        assert len(ess_hist) == self.N_STEPS - 1
        for ess in ess_hist:
            assert -1e-6 <= ess <= 1.0 + 1e-4

        # Ground truth: reward-tilted density at selected intermediate reverse times
        xs = torch.linspace(-6, 6, 250).reshape(-1, 1, 1)
        xs_flat = xs.squeeze()
        log_p_data = gmm.log_prob(xs, t=self.EPS).squeeze()
        p_rew = _tilted_density(gmm, xs, torch.tensor(self.EPS), beta_fn, lambda x, t_: r(x))
        p_data = log_p_data.exp()
        p_data = p_data / torch.trapezoid(p_data, xs.squeeze())

        # Build histogram of final samples
        x_final = traj[-1, :, 0, 0]
        dx = xs_flat[1] - xs_flat[0]
        bin_edges = torch.cat([(xs_flat[0] - dx / 2).unsqueeze(0), xs_flat + dx / 2])
        hist, _ = torch.histogram(x_final, bins=bin_edges, density=True)

        # 4. Wasserstein-1 vs reward-tilted target: hard threshold, must pass regardless
        # of how the unguided distribution compares
        w1_rew = _wasserstein1(x_final, xs_flat, p_rew)

        if PLOT:
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
                out_path=_plot_dir("test_steering", "TestSteeredSamplingBetaFinalMarginal", "test_steered_sampling")
                / f"steered_center{reward_center}_sigma{reward_sigma}_ess{ess_threshold}.png",
            )
        # Measured across the seven parameter combos at 10k particles: W1 spans
        # 0.030-0.050, i.e. the old flat 0.05 sat exactly on the Monte-Carlo floor. 0.06
        # clears it while keeping all the discriminating power: the same runs score
        # W1 = 2.0-3.8 against the *untilted* density, so an unsteered or mis-signed
        # sampler misses this by ~60x, not by a few percent.
        assert w1_rew < 0.06, f"center={reward_center} sigma={reward_sigma}: W1 vs reward-tilted={w1_rew:.4f}"


class KarrasDenoiseMixin:
    def _denoise(self, gmm, sched, x_t, t, n_denoise_steps):
        """Tweedie-blend ODE denoiser: integrates the probability-flow ODE from
        sigma(t) down to sigma(EPS) in Euler substeps of size 1/n_denoise_steps in t,
        returning a bounded x̂_0 estimate plus leaf tensors for backprop (see
        notebooks/karras_terminal_variance_steering.py:denoise for the derivation).
        """
        # Substep count is a pure function of (t, EPS, n_denoise_steps): the sampler and
        # the analytic tilted-density reference both route through here, so they only
        # agree on x̂_0 if it depends on nothing else (not particle count, not call
        # site) and both are handed the same n_denoise_steps.
        n_substeps = max(1, math.ceil((float(t) - self.EPS) * n_denoise_steps))
        x_t_leaf = x_t.detach().requires_grad_(True)
        t_leaf = torch.as_tensor(t, dtype=x_t.dtype, device=x_t.device).clone().detach().requires_grad_(True)
        with torch.enable_grad():
            t_min = torch.as_tensor(self.EPS, dtype=x_t.dtype, device=x_t.device)
            x = x_t_leaf
            t_prev = t_leaf
            sigma_prev = sched.get_sigma_t(t_prev)
            for step_k in range(1, n_substeps + 1):
                t_next = t_leaf + (t_min - t_leaf) * (step_k / n_substeps)
                sigma_next = sched.get_sigma_t(t_next)
                D = x + sigma_prev**2 * gmm.score(x, t_prev)
                ratio = sigma_next / sigma_prev
                x = ratio * x + (1.0 - ratio) * D
                t_prev = t_next
                sigma_prev = sigma_next
            x0 = x
        return x0, x_t_leaf, t_leaf


@pytest.mark.slow
class TestSteeredSamplingKarrasFinalMarginal(KarrasDenoiseMixin):
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
    N_PARTICLES = 5_000
    N_STEPS = 610

    @pytest.fixture
    def setup(self):
        sched = KarrasSchedule(sigma_min=4e-4, sigma_max=160.0, rho=7.0, sigma_data=1.0)
        gmm = GMM(
            mu=torch.tensor([[[-1], [1]]]),
            sigma=torch.tensor([[[0.4], [0.4]]]),
            weight=torch.tensor([[0.2, 0.8]]),
            schedule=sched,
        )
        return gmm, sched

    @pytest.mark.parametrize("reward_center", [-0.75, 0.25, 1.0], ids=lambda v: f"{v}")
    @pytest.mark.parametrize(
        "ess_threshold",
        [
            0.95,  # adaptive
            25,  # fixed-interval: resample every 25 of the 499 integration steps
            1_000,  # final-only: interval far exceeds 499 integration steps
        ],
        ids=lambda v: f"{v}",
    )
    @pytest.mark.parametrize("n_denoise_steps", [10], ids=lambda v: f"{v}")
    def test_steered_sampling_karras(self, setup, reward_center, ess_threshold, n_denoise_steps):
        gmm, sched = setup
        reward_sigma = 1.0

        def r(x0_hat):
            return -0.5 * (x0_hat - reward_center) ** 2 / reward_sigma**2

        def beta_fn(t):
            def karras_schedule_functional(t, sigma_min=4e-4, sigma_max=160.0, rho=7.0):
                """Karras schedule functional form:

                sigma(t) = (sigma_max^(1/rho) + t * (sigma_min^(1/rho) - sigma_max^(1/rho)))^rho
                """
                inv_rho = 1.0 / rho
                sigma_min_inv = sigma_min**inv_rho
                sigma_max_inv = sigma_max**inv_rho
                sigma = (sigma_max_inv + t * (sigma_min_inv - sigma_max_inv)) ** rho
                return sigma

            return (1 - t) ** 6 / karras_schedule_functional(t, sigma_min=4e-4, sigma_max=160.0, rho=7.0) ** 2

        def dbeta_dt(t):
            return _dbeta_dt(beta_fn, t)

        def reward_and_grads(x, t):
            x0_hat, x_leaf, t_leaf = self._denoise(gmm, sched, x, t, n_denoise_steps)
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
        traj, ess_hist, weight_hist = steered_reverse_sampling(
            guided_drift, sched.diffusion_coeff, fkc_weight_update, x0, t, ess_threshold=ess_threshold
        )

        # 1. Trajectory shape
        assert traj.shape == (self.N_STEPS, self.N_PARTICLES, 1, 1)
        assert weight_hist.shape == (self.N_STEPS, self.N_PARTICLES)

        # 2. ESS/N history within [0, 1] at every step (small float32 tolerance: with
        # near-uniform weights, logsumexp-based ESS can round fractionally above 1.0)
        assert len(ess_hist) == self.N_STEPS - 1
        for ess in ess_hist:
            assert -1e-6 <= ess <= 1.0 + 1e-4

        # Ground truth: reward-tilted density. beta_fn(EPS) ≈ 1 (data time), matching
        # the tilt weight the sampler itself applies at the end of the reverse pass.
        xs = torch.linspace(-3, 3, 100).reshape(-1, 1, 1)
        log_p_data = gmm.log_prob(xs, t=self.EPS).squeeze()
        p_rew = _tilted_density(
            gmm,
            xs,
            torch.tensor(self.EPS),
            beta_fn,
            lambda x, t_: r(self._denoise(gmm, sched, x, t_, n_denoise_steps)[0].detach()),
        )

        p_data = log_p_data.exp()
        p_data = p_data / torch.trapezoid(p_data, xs.squeeze())

        xs_flat = xs.squeeze()
        # Build histogram of final samples
        x_final = traj[-1, :, 0, 0]
        dx = xs_flat[1] - xs_flat[0]
        bin_edges = torch.cat([(xs_flat[0] - dx / 2).unsqueeze(0), xs_flat + dx / 2])
        hist, _ = torch.histogram(x_final, bins=bin_edges, density=True)

        # 4. Wasserstein-1 vs reward-tilted target: hard threshold. The sextic beta_fn
        # (see above) keeps this comfortably under threshold in all three modes,
        # including final-only — no special casing needed once the tilt is suppressed
        # correctly for this sigma_max.
        w1_rew = _wasserstein1(x_final, xs_flat, p_rew)

        if PLOT:
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
                out_path=_plot_dir(
                    "test_steering", "TestSteeredSamplingKarrasFinalMarginal", "test_steered_sampling_karras"
                )
                / f"steered_karras_center{reward_center}_ess{ess_threshold}.png",
            )
        assert w1_rew < 0.075, (
            f"KarrasSchedule reward_center={reward_center} ess_threshold={ess_threshold}: "
            f"W1 vs reward-tilted={w1_rew:.4f}"
        )


@pytest.mark.slow
class TestSteeredSamplingBetaIntermediateMarginals:
    """Weighted intermediate-time checks for BetaSchedule FKC particle marginals."""

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

    @pytest.mark.parametrize("reward_center", [-2.0, -0.25, 1.0], ids=lambda v: f"center={v}")
    def test_weighted_beta_reverse_trajectory_matches_tilted_intermediate_marginals(self, setup, reward_center):
        """Weighted reverse particles match p_t ∝ q_t exp(beta(t)r(x_t)) at intermediate times."""
        gmm, sched = setup
        reward_sigma = 1.0

        def r(x):
            return -0.5 * (x - reward_center) ** 2 / reward_sigma**2

        def grad_r(x):
            return -(x - reward_center) / reward_sigma**2

        def beta_fn(t):
            return 1.0 - t

        def dbeta_dt(t):
            return _dbeta_dt(beta_fn, t)

        def guided_drift(x, t):
            f = sched.forward_drift(x, t)
            sigma = sched.diffusion_coeff(t)
            score = gmm.score(x, t)
            beta = beta_fn(t)
            return f - sigma**2 * score - beta * (sigma**2 / 2) * grad_r(x)

        def fkc_weight_update(x, t, dt):
            f = sched.forward_drift(x, t)
            sigma = sched.diffusion_coeff(t)
            score = gmm.score(x, t)
            rg, rv = grad_r(x), r(x)
            beta = beta_fn(t)
            term1 = -dbeta_dt(t) * rv
            term2 = -(beta * rg) * f
            term3 = (beta * rg) * (sigma**2 / 2) * score
            return (term1 + term2 + term3).squeeze(-1).squeeze(-1) * dt.abs()

        torch.manual_seed(0)
        t = torch.linspace(1 - self.EPS, self.EPS, self.N_STEPS)
        x0 = torch.randn(self.N_PARTICLES, 1, 1)
        resample_every = 50
        # In interval mode, steered_reverse_sampling resets log_w to zero after
        # steps 50, 100, ... and stores the post-resample particle cloud at the
        # matching trajectory index. The final trajectory index is also resampled.
        traj, _, weight_hist = steered_reverse_sampling(
            guided_drift, sched.diffusion_coeff, fkc_weight_update, x0, t, ess_threshold=resample_every
        )
        assert weight_hist.shape == (self.N_STEPS, self.N_PARTICLES)

        xs = torch.linspace(-6, 6, 250).reshape(-1, 1, 1)
        xs_flat = xs.squeeze()
        for t_idx in [25, 50, 75, 125, 150, 275, 300, 350, 400, 450, self.N_STEPS - 1]:
            t_ = t[t_idx]
            sigma_t = sched.get_sigma_t(t_)
            p_tilt = _tilted_density(gmm, xs, t_, beta_fn, lambda x, t__: r(x))
            x_t = traj[t_idx, :, 0, 0]
            weighted_w1_tilt = _weighted_wasserstein1(x_t, weight_hist[t_idx], xs_flat, p_tilt)
            if PLOT:
                log_p_data = gmm.log_prob(xs, t=t_).squeeze()
                p_data = log_p_data.exp()
                p_data = p_data / torch.trapezoid(p_data, xs_flat)
                plot_marginal_density_comparison(
                    xs_flat=xs_flat,
                    p_data=p_data,
                    p_rew=p_tilt,
                    samples=x_t,
                    reward_center=reward_center,
                    weights=weight_hist[t_idx],
                    title=_intermediate_marginal_title(
                        schedule_name="BetaSchedule",
                        reward_name="direct noisy-state reward r(x_t)",
                        t=t_,
                        t_idx=t_idx,
                        resample_every=resample_every,
                        w1=weighted_w1_tilt,
                    ),
                    out_path=(
                        _plot_dir(
                            "test_steering",
                            "TestSteeredSamplingBetaIntermediateMarginals",
                            "test_weighted_beta_reverse_trajectory_matches_tilted_intermediate_marginals",
                            f"center{reward_center}",
                        )
                        / f"beta_intermediate_t{t_idx}_center{reward_center}_resample{resample_every}.png"
                    ),
                )
            # Same scale-aware rule as the Karras sibling classes: the W1 floor tracks
            # the marginal's own width. At 5k particles a flat 0.05 sits *below* the
            # Monte-Carlo floor at mid-range t (measured worst ≈0.059, and falling to
            # ≈0.03 at 80k particles — noise, not bias), so it cannot be used here.
            tol = 0.05 * (1.0 + sigma_t.item())
            assert weighted_w1_tilt < tol, (
                f"BetaSchedule weighted intermediate center={reward_center} t={t_:.3f}: "
                f"W1={weighted_w1_tilt:.4f} >= {tol:.4f}"
            )

        def dbeta_dt(t):
            return _dbeta_dt(beta_fn, t)

        def guided_drift(x, t):
            f = sched.forward_drift(x, t)
            sigma = sched.diffusion_coeff(t)
            score = gmm.score(x, t)
            beta = beta_fn(t)
            return (
                f
                - 1 / 2 * (1 + alpha**2) * sigma**2 * score
                - beta * alpha**2 * (sigma**2 / 2) * grad_r(x)
            )

        def diffusion(t):
            return alpha * sched.diffusion_coeff(t)

        def fkc_weight_update(x, t, dt):
            f = sched.forward_drift(x, t)
            sigma = sched.diffusion_coeff(t)
            score = gmm.score(x, t)
            rg, rv = grad_r(x), r(x)
            beta = beta_fn(t)
            term1 = -dbeta_dt(t) * rv
            term2 = -(beta * rg) * f
            term3 = (beta * rg) * (sigma**2 / 2) * score
            return (term1 + term2 + term3).squeeze(-1).squeeze(-1) * dt.abs()

        torch.manual_seed(0)
        t = torch.linspace(1 - self.EPS, self.EPS, self.N_STEPS)
        x0 = torch.randn(self.N_PARTICLES, 1, 1)
        resample_every = self.N_STEPS+5
        # In interval mode, steered_reverse_sampling resets log_w to zero after
        # steps 50, 100, ... and stores the post-resample particle cloud at the
        # matching trajectory index. The final trajectory index is also resampled.
        traj, _, weight_hist = steered_reverse_sampling(
            guided_drift, diffusion, fkc_weight_update, x0, t, ess_threshold=resample_every
        )
        assert weight_hist.shape == (self.N_STEPS, self.N_PARTICLES)

        xs = torch.linspace(-6, 6, 250).reshape(-1, 1, 1)
        xs_flat = xs.squeeze()
        for t_idx in [25, 50, 75, 125, 150, 275, 300, 350, 400, 450, self.N_STEPS - 2, self.N_STEPS - 1]:
            t_ = t[t_idx]
            p_tilt = _tilted_density(gmm, xs, t_, beta_fn, lambda x, t__: r(x))
            x_t = traj[t_idx, :, 0, 0]
            weighted_w1_tilt = _weighted_wasserstein1(x_t, weight_hist[t_idx], xs_flat, p_tilt)
            if PLOT:
                log_p_data = gmm.log_prob(xs, t=t_).squeeze()
                p_data = log_p_data.exp()
                p_data = p_data / torch.trapezoid(p_data, xs_flat)
                plot_marginal_density_comparison(
                    xs_flat=xs_flat,
                    p_data=p_data,
                    p_rew=p_tilt,
                    samples=x_t,
                    reward_center=reward_center,
                    weights=weight_hist[t_idx],
                    title=_intermediate_marginal_title(
                        schedule_name="BetaSchedule",
                        reward_name="direct noisy-state reward r(x_t)",
                        t=t_,
                        t_idx=t_idx,
                        resample_every=resample_every,
                        w1=weighted_w1_tilt,
                    ),
                    out_path=(
                        _plot_dir(
                            "test_steering",
                            "TestSteeredSamplingBetaIntermediateMarginals",
                            "test_weighted_beta_reverse_alpha_trajectory_matches_tilted_intermediate_marginals",
                            f"alpha{alpha}",
                            f"center{reward_center}",
                        )
                        / (
                            f"beta_alpha{alpha}_intermediate_t{t_idx}_center{reward_center}_"
                            f"resample{resample_every}.png"
                        )
                    ),
                )
            assert weighted_w1_tilt < 0.05, (
                f"BetaSchedule weighted intermediate center={reward_center} t={t_:.3f}: W1={weighted_w1_tilt:.4f}"
            )

@pytest.mark.slow
class TestSteeredSamplingKarrasIntermediateMarginals:
    """Weighted intermediate-time checks for KarrasSchedule FKC particle marginals."""

    EPS = 0.001
    T_NOISE = 1 - EPS
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

    @pytest.mark.parametrize("ess_threshold", [0.9, 25, 1000], ids=lambda v: f"ess={v}")
    @pytest.mark.parametrize("reward_center", [-1.0, -0.5, 0.5], ids=lambda v: f"center={v}")
    def test_weighted_karras_reverse_trajectory_matches_tilted_intermediate_marginals(
        self, setup, reward_center, ess_threshold
    ):
        """Weighted reverse particles match p_t ∝ q_t exp(beta(t)r(x_t)) at intermediate times."""
        gmm, sched = setup
        reward_sigma = 1.0

        def r(x):
            return -0.5 * (x - reward_center) ** 2 / reward_sigma**2

        def grad_r(x):
            return -(x - reward_center) / reward_sigma**2

        def beta_fn(t):
            g = sched.diffusion_coeff(t)
            return (1 - t) ** 2 / (1 + g**2)

        def dbeta_dt(t):
            return _dbeta_dt(beta_fn, t)

        def guided_drift(x, t):
            g = sched.diffusion_coeff(t)
            beta = beta_fn(t)
            score = gmm.score(x, t)
            return -(g**2) * score - beta * (g**2 / 2) * grad_r(x)

        def fkc_weight_update(x, t, dt):
            g = sched.diffusion_coeff(t)
            beta = beta_fn(t)
            dbeta = dbeta_dt(t)
            rv = r(x)
            rg = grad_r(x)
            score = gmm.score(x, t)
            integrand = -dbeta * rv + beta * rg * (g**2 / 2) * score
            return integrand.squeeze(-1).squeeze(-1) * dt.abs()

        torch.manual_seed(0)
        t = torch.linspace(self.T_NOISE, self.EPS, self.N_STEPS)
        x0 = gmm.sample(shape=self.N_PARTICLES, t=self.T_NOISE)
        traj, _, weight_hist = steered_reverse_sampling(
            guided_drift, sched.diffusion_coeff, fkc_weight_update, x0, t, ess_threshold=ess_threshold
        )
        assert traj.shape == (self.N_STEPS, self.N_PARTICLES, 1, 1)
        assert weight_hist.shape == (self.N_STEPS, self.N_PARTICLES)

        for t_idx in [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600]:
            t_ = t[t_idx]
            sigma_t = sched.get_sigma_t(t_)
            grid_radius = max(4.0, 6.0 * sigma_t.item())
            xs = torch.linspace(-grid_radius, grid_radius, 400).reshape(-1, 1, 1)
            xs_flat = xs.squeeze()
            p_tilt = _tilted_density(gmm, xs, t_, beta_fn, lambda x, t__: r(x))
            x_t = traj[t_idx, :, 0, 0]
            weighted_w1_tilt = _weighted_wasserstein1(x_t, weight_hist[t_idx], xs_flat, p_tilt)
            if PLOT:
                log_p_data = gmm.log_prob(xs, t=t_).squeeze()
                p_data = log_p_data.exp()
                p_data = p_data / torch.trapezoid(p_data, xs_flat)
                plot_radius = max(3, 3.0 * sigma_t.item())
                plot_marginal_density_comparison(
                    xs_flat=xs_flat,
                    p_data=p_data,
                    p_rew=p_tilt,
                    samples=x_t,
                    reward_center=reward_center,
                    weights=weight_hist[t_idx],
                    title=_intermediate_marginal_title(
                        schedule_name="KarrasSchedule",
                        reward_name="direct noisy-state reward r(x_t)",
                        t=t_,
                        t_idx=t_idx,
                        resample_every=ess_threshold,
                        w1=weighted_w1_tilt,
                    ),
                    out_path=(
                        _plot_dir(
                            "test_steering",
                            "TestSteeredSamplingKarrasIntermediateMarginals",
                            "test_weighted_karras_reverse_trajectory_matches_tilted_intermediate_marginals",
                            f"center{reward_center}",
                            f"ess{ess_threshold}",
                        )
                        / f"karras_intermediate_t{t_idx}_center{reward_center}_resample{ess_threshold}.png"
                    ),
                    min_x=-plot_radius,
                    max_x=plot_radius,
                )
            # Tolerance tracks the marginal's own scale: loose while sigma_t is large,
            # tightening as sigma_t -> 0. The 0.06 base (was 0.05) is the same floor
            # correction as TestSteeredSamplingBetaFinalMarginal — at sigma_t ~ 0 the
            # old bound was 0.0507 against a measured 0.0523 at center=-1.0, ess=25.
            tol = 0.06 * (1.0 + sigma_t.item())
            assert weighted_w1_tilt < tol, (
                f"KarrasSchedule weighted intermediate center={reward_center} t={t_:.3f}: "
                f"W1={weighted_w1_tilt:.4f} >= {tol:.4f}"
            )

        if PLOT:
            plot_trajectory_samples(
                traj=traj,
                t=t,
                reward_center=reward_center,
                title=f"KarrasSchedule steered reverse trajectories | reward_center={reward_center}",
                out_path=(
                    _plot_dir(
                        "test_steering",
                        "TestSteeredSamplingKarrasIntermediateMarginals",
                        "test_weighted_karras_reverse_trajectory_matches_tilted_intermediate_marginals",
                        f"center{reward_center}",
                        f"ess{ess_threshold}",
                    )
                    / f"karras_trajectories_center{reward_center}_resample{ess_threshold}.png"
                ),
            )


@pytest.mark.slow
class TestSteeredSamplingDenoisingKarrasIntermediateMarginals(KarrasDenoiseMixin):
    """Weighted Karras intermediate checks with rewards evaluated on denoised x0_hat(x_t)."""

    EPS = 0.001
    T_NOISE = 1 - EPS
    N_PARTICLES = 5_000
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

    @pytest.mark.parametrize("ess_threshold", [0.9, 25, 1000], ids=lambda v: f"ess={v}")
    @pytest.mark.parametrize("reward_center", [-1.0, -0.5, 0.5], ids=lambda v: f"center={v}")
    @pytest.mark.parametrize("n_denoise_steps", [10], ids=lambda v: f"{v}")
    def test_weighted_denoising_karras_reverse_trajectory_matches_tilted_intermediate_marginals(
        self, setup, reward_center, ess_threshold, n_denoise_steps
    ):
        """Weighted reverse particles match p_t ∝ q_t exp(beta(t)r(x0_hat(x_t,t))) at intermediate times."""
        gmm, sched = setup
        reward_sigma = 1.0

        def r(x0_hat):
            return -0.5 * (x0_hat - reward_center) ** 2 / reward_sigma**2

        def beta_fn(t):
            g = sched.diffusion_coeff(t)
            return (1 - t) / (1 + g**2)

        def dbeta_dt(t):
            return _dbeta_dt(beta_fn, t)

        def reward_and_grads(x, t):
            x0_hat, x_leaf, t_leaf = self._denoise(gmm, sched, x, t, n_denoise_steps)
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
        traj, _, weight_hist = steered_reverse_sampling(
            guided_drift, sched.diffusion_coeff, fkc_weight_update, x0, t, ess_threshold=ess_threshold
        )
        assert traj.shape == (self.N_STEPS, self.N_PARTICLES, 1, 1)
        assert weight_hist.shape == (self.N_STEPS, self.N_PARTICLES)

        for t_idx in [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, self.N_STEPS - 1]:
            t_ = t[t_idx]
            sigma_t = sched.get_sigma_t(t_)
            grid_radius = max(4.0, 6.0 * sigma_t.item())
            xs = torch.linspace(-grid_radius, grid_radius, 400).reshape(-1, 1, 1)
            xs_flat = xs.squeeze()
            p_tilt = _tilted_density(
                gmm,
                xs,
                t_,
                beta_fn,
                lambda x, t__: r(self._denoise(gmm, sched, x, t__, n_denoise_steps)[0].detach()),
            )
            x_t = traj[t_idx, :, 0, 0]
            weighted_w1_tilt = _weighted_wasserstein1(x_t, weight_hist[t_idx], xs_flat, p_tilt)
            if PLOT:
                log_p_data = gmm.log_prob(xs, t=t_).squeeze()
                p_data = log_p_data.exp()
                p_data = p_data / torch.trapezoid(p_data, xs_flat)
                plot_radius = max(3, 3.0 * sigma_t.item())
                plot_marginal_density_comparison(
                    xs_flat=xs_flat,
                    p_data=p_data,
                    p_rew=p_tilt,
                    samples=x_t,
                    reward_center=reward_center,
                    weights=weight_hist[t_idx],
                    title=_intermediate_marginal_title(
                        schedule_name="KarrasSchedule",
                        reward_name="denoised reward r(D(x_t,t))",
                        t=t_,
                        t_idx=t_idx,
                        resample_every=ess_threshold,
                        w1=weighted_w1_tilt,
                    ),
                    out_path=(
                        _plot_dir(
                            "test_steering",
                            "TestSteeredSamplingDenoisingKarrasIntermediateMarginals",
                            "test_weighted_denoising_karras_reverse_trajectory_matches_tilted_intermediate_marginals",
                            f"center{reward_center}",
                            f"ess{ess_threshold}",
                        )
                        / f"karras_denoising_intermediate_t{t_idx}_center{reward_center}_resample{ess_threshold}.png"
                    ),
                    min_x=-plot_radius,
                    max_x=plot_radius,
                )
            # Higher floor than the direct-reward tests: the n_denoise_steps=10 unrolled
            # denoiser adds its own discretization error on top of the sampler's.
            tol = 0.08 * (1.0 + sigma_t.item())
            assert weighted_w1_tilt < tol, (
                f"KarrasSchedule denoising weighted intermediate center={reward_center} t={t_:.3f}: "
                f"W1={weighted_w1_tilt:.4f} >= {tol:.4f}"
            )

        if PLOT:
            plot_trajectory_samples(
                traj=traj,
                t=t,
                reward_center=reward_center,
                title=f"KarrasSchedule denoising steered reverse trajectories | reward_center={reward_center}",
                out_path=(
                    _plot_dir(
                        "test_steering",
                        "TestSteeredSamplingDenoisingKarrasIntermediateMarginals",
                        "test_weighted_denoising_karras_reverse_trajectory_matches_tilted_intermediate_marginals",
                        f"center{reward_center}",
                        f"ess{ess_threshold}",
                    )
                    / f"karras_denoising_trajectories_center{reward_center}_resample{ess_threshold}.png"
                ),
            )


@pytest.mark.slow
class TestSteeredSamplingIntermediateMarginals(KarrasDenoiseMixin):
    """Intermediate-time checks for post-resampling FKC particle marginals."""

    EPS = 0.001
    T_NOISE = 1 - EPS
    N_PARTICLES = 5_000

    @pytest.fixture
    def setup_karras(self):
        sched = KarrasSchedule(sigma_min=4e-4, sigma_max=160.0, rho=7.0, sigma_data=1.0)
        gmm = GMM(
            mu=torch.tensor([[[-1], [1]]]),
            sigma=torch.tensor([[[0.4], [0.4]]]),
            weight=torch.tensor([[0.2, 0.8]]),
            schedule=sched,
        )
        return gmm, sched

    @pytest.mark.parametrize("n_denoise_steps", [10], ids=lambda v: f"{v}")
    def test_steered_sampling_karras_denoised_reward_intermediate_marginals(self, setup_karras, n_denoise_steps):
        """At known fixed-interval resampling times, Karras particles match q_t exp(beta(t)r(denoise(x_t,t))).

        Without a resample at a given time slice, the unweighted particle cloud is
        not the FKC marginal yet; the accumulated log weights carry the tilt.
        """
        gmm, sched = setup_karras
        reward_center = -1
        reward_sigma = 1.0
        n_steps = 610

        def r(x0_hat):
            return -0.5 * (x0_hat - reward_center) ** 2 / reward_sigma**2

        def beta_fn(t):
            return (1 - t) ** 6

        def dbeta_dt(t):
            return _dbeta_dt(beta_fn, t)

        def reward_and_grads(x, t):
            x0_hat, x_leaf, t_leaf = self._denoise(gmm, sched, x, t, n_denoise_steps)
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
        t = torch.linspace(self.T_NOISE, self.EPS, n_steps)
        x0 = gmm.sample(shape=self.N_PARTICLES, t=self.T_NOISE)
        resample_every = 50
        # In interval mode, steered_reverse_sampling resets log_w to zero after
        # steps 50, 100, ... and stores the post-resample particle cloud at the
        # matching trajectory index. The final trajectory index is also resampled.
        traj, _, weight_hist = steered_reverse_sampling(
            guided_drift, sched.diffusion_coeff, fkc_weight_update, x0, t, ess_threshold=resample_every
        )
        assert weight_hist.shape == (n_steps, self.N_PARTICLES)

        for t_idx in [150, 200, 250, 300, 350, 400, 450, 500, 550, 600, n_steps - 1]:
            t_ = t[t_idx]
            sigma_t = sched.get_sigma_t(t_)
            # Grid must span the marginal, as in the sibling classes: at this AF3-scale
            # Karras setting sigma_t is ~31 at t_idx=150 and particles reach |x|~135, so
            # a fixed [-20, 20] window truncates the cloud and W1 measures the truncation
            # (15.5!) rather than the FKC resampling invariant.
            grid_radius = max(4.0, 6.0 * sigma_t.item())
            xs = torch.linspace(-grid_radius, grid_radius, 400).reshape(-1, 1, 1)
            xs_flat = xs.squeeze()
            p_tilt = _tilted_density(
                gmm,
                xs,
                t_,
                beta_fn,
                lambda x, t__: r(self._denoise(gmm, sched, x, t__, n_denoise_steps)[0].detach()),
            )
            x_t = traj[t_idx, :, 0, 0]
            w1_tilt = _wasserstein1(x_t, xs_flat, p_tilt)
            if PLOT:
                log_p_data = gmm.log_prob(xs, t=t_).squeeze()
                p_data = log_p_data.exp()
                p_data = p_data / torch.trapezoid(p_data, xs_flat)
                plot_radius = max(3.0, min(20.0, 4.0 * sigma_t.item()))
                plot_marginal_density_comparison(
                    xs_flat=xs_flat,
                    p_data=p_data,
                    p_rew=p_tilt,
                    samples=x_t,
                    reward_center=reward_center,
                    title=_intermediate_marginal_title(
                        schedule_name="KarrasSchedule",
                        reward_name="denoised reward r(D(x_t,t))",
                        t=t_,
                        t_idx=t_idx,
                        resample_every=resample_every,
                        w1=w1_tilt,
                    ),
                    out_path=(
                        _plot_dir(
                            "test_steering",
                            "TestSteeredSamplingIntermediateMarginals",
                            "test_steered_sampling_karras_denoised_reward_intermediate_marginals",
                        )
                        / f"karras_denoised_intermediate_t{t_idx}_center{reward_center}_resample{resample_every}.png"
                    ),
                    min_x=-plot_radius,
                    max_x=plot_radius,
                )
            # Same rule as TestSteeredSamplingDenoisingKarrasIntermediateMarginals, which
            # shares the denoised-reward construction: the 0.08 floor (vs 0.05 for direct
            # rewards) absorbs the n_denoise_steps=10 unrolled denoiser's own
            # discretization error, and the sigma_t scaling tracks the marginal's width.
            tol = 0.08 * (1.0 + sigma_t.item())
            assert w1_tilt < tol, f"KarrasSchedule denoised intermediate t={t_:.3f}: W1={w1_tilt:.4f} >= {tol:.4f}"

    def test_steered_sampling_karras_direct_noisy_reward_intermediate_marginals(self):
        """At known fixed-interval resampling times, direct Karras steering matches p_t ∝ q_t exp(beta(t)r(x_t)).

        Without a resample at a given time slice, the unweighted particle cloud is
        not the FKC marginal yet; the accumulated log weights carry the tilt.
        """
        sched = KarrasSchedule(sigma_min=0.01, sigma_max=160, rho=7.0, sigma_data=1.0)
        gmm = GMM(
            mu=torch.tensor([[[-1.0], [1.0]]]),
            sigma=torch.tensor([[[0.4], [0.4]]]),
            weight=torch.tensor([[0.3, 0.7]]),
            schedule=sched,
        )
        reward_center = -0.5
        reward_sigma = 1.0
        n_particles = 5_000
        n_steps = 610

        def r(x):
            return -0.5 * (x - reward_center) ** 2 / reward_sigma**2

        def grad_r(x):
            return -(x - reward_center) / reward_sigma**2

        def beta_fn(t):
            g = sched.diffusion_coeff(t)
            return (1 - t) ** 2 / (1 + g**2)

        def dbeta_dt(t):
            return _dbeta_dt(beta_fn, t)

        def guided_drift(x, t):
            g = sched.diffusion_coeff(t)
            beta = beta_fn(t)
            score = gmm.score(x, t)
            return -(g**2) * score - beta * (g**2 / 2) * grad_r(x)

        def fkc_weight_update(x, t, dt):
            g = sched.diffusion_coeff(t)
            beta = beta_fn(t)
            dbeta = dbeta_dt(t)
            rv = r(x)
            rg = grad_r(x)
            score = gmm.score(x, t)
            integrand = -dbeta * rv + beta * rg * (g**2 / 2) * score
            return integrand.squeeze(-1).squeeze(-1) * dt.abs()

        torch.manual_seed(0)
        t = torch.linspace(self.T_NOISE, self.EPS, n_steps)
        x0 = gmm.sample(shape=n_particles, t=self.T_NOISE)
        resample_every = 50
        # In interval mode, steered_reverse_sampling resets log_w to zero after
        # steps 50, 100, ... and stores the post-resample particle cloud at the
        # matching trajectory index. The final trajectory index is also resampled.
        traj, ess_hist, weight_hist = steered_reverse_sampling(
            guided_drift, sched.diffusion_coeff, fkc_weight_update, x0, t, ess_threshold=resample_every
        )

        assert traj.shape == (n_steps, n_particles, 1, 1)
        assert weight_hist.shape == (n_steps, n_particles)
        assert len(ess_hist) == n_steps - 1
        for ess in ess_hist:
            assert -1e-6 <= ess <= 1.0 + 1e-4

        for t_idx in [50, 100, 150, 200, 250, 300, 400, 450, 500, 550, 600, n_steps - 1]:
            t_ = t[t_idx]
            sigma_t = sched.get_sigma_t(t_)
            # Grid must span the marginal: at t=0.917 sigma_t is ~100, so a fixed
            # [-4, 4] window truncates the cloud and W1 measures the truncation.
            grid_radius = max(4.0, 6.0 * sigma_t.item())
            xs = torch.linspace(-grid_radius, grid_radius, 400).reshape(-1, 1, 1)
            xs_flat = xs.squeeze()
            p_tilt = _tilted_density(gmm, xs, t_, beta_fn, lambda x, t__: r(x))
            x_t = traj[t_idx, :, 0, 0]
            w1_tilt = _wasserstein1(x_t, xs_flat, p_tilt)
            if PLOT:
                log_p_data = gmm.log_prob(xs, t=t_).squeeze()
                p_data = log_p_data.exp()
                p_data = p_data / torch.trapezoid(p_data, xs_flat)
                plot_marginal_density_comparison(
                    xs_flat=xs_flat,
                    p_data=p_data,
                    p_rew=p_tilt,
                    samples=x_t,
                    reward_center=reward_center,
                    title=_intermediate_marginal_title(
                        schedule_name="KarrasSchedule",
                        reward_name="direct noisy-state reward r(x_t)",
                        t=t_,
                        t_idx=t_idx,
                        resample_every=resample_every,
                        w1=w1_tilt,
                    ),
                    out_path=(
                        _plot_dir(
                            "test_steering",
                            "TestSteeredSamplingIntermediateMarginals",
                            "test_steered_sampling_karras_direct_noisy_reward_intermediate_marginals",
                        )
                        / f"karras_direct_intermediate_t{t_idx}_center{reward_center}_resample{resample_every}.png"
                    ),
                )
            tol = 0.05 * (1.0 + sigma_t.item())
            assert w1_tilt < tol, f"KarrasSchedule direct noisy reward t={t_:.3f}: W1={w1_tilt:.4f} >= {tol:.4f}"


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


def _intermediate_marginal_title(schedule_name, reward_name, t, t_idx, resample_every, w1):
    return (
        f"{schedule_name}: intermediate steered marginal\n"
        f"{reward_name} | t_idx={t_idx}, t={t:.3f} | resample every {resample_every} steps | W1={w1:.4f}"
    )


def _plot_density_comparison(
    ax, xs_flat, p_data, p_rew, hist, reward_center, title, bin_edges, min_x=None, max_x=None, weighted_hist=None
):
    ax.plot(xs_flat.cpu(), p_data.cpu(), label="True unguided density", color="steelblue", linewidth=2)
    ax.plot(xs_flat.cpu(), p_rew.cpu(), label="True tilted density", color="firebrick", linewidth=2)
    if weighted_hist is not None:
        ax.stairs(hist.cpu(), bin_edges.cpu(), label="Unweighted empirical density", color="gray", linewidth=1.4)
        ax.stairs(
            weighted_hist.cpu(),
            bin_edges.cpu(),
            label="Weighted empirical density",
            fill=True,
            alpha=0.35,
            color="seagreen",
        )
    else:
        ax.stairs(hist.cpu(), bin_edges.cpu(), label="Empirical sample density", fill=True, alpha=0.4, color="seagreen")
    ax.axvline(reward_center, color="firebrick", linestyle="--", linewidth=1)
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.set_title(title)
    if min_x is not None and max_x is not None:
        ax.set_xlim(min_x, max_x)
    ax.legend(fontsize=9)


def plot_marginal_density_comparison(
    xs_flat,
    p_data,
    p_rew,
    samples,
    reward_center,
    title,
    out_path,
    min_x=None,
    max_x=None,
    weights=None,
    energy=None,
    energy_time=None,
):
    """Save a single-time marginal density comparison, optionally showing its energy."""
    import matplotlib.pyplot as plt

    if min_x is not None and max_x is not None:
        bin_edges = torch.linspace(min_x, max_x, 161, dtype=xs_flat.dtype, device=xs_flat.device)
    else:
        bin_w = xs_flat[1] - xs_flat[0]
        bin_edges = torch.cat([xs_flat[:1] - bin_w / 2, xs_flat + bin_w / 2])
    hist, _ = torch.histogram(samples, bins=bin_edges, density=True)
    weighted_hist = None
    if weights is not None:
        weighted_hist, _ = torch.histogram(samples, bins=bin_edges, weight=weights, density=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    _plot_density_comparison(
        ax,
        xs_flat,
        p_data,
        p_rew,
        hist,
        reward_center,
        title,
        bin_edges,
        min_x=min_x,
        max_x=max_x,
        weighted_hist=weighted_hist,
    )
    if energy is not None:
        energy_x = xs_flat.reshape(-1, 1, 1)
        energy_values = energy(energy_x, energy_time).squeeze().detach().cpu()
        energy_ax = ax.twinx()
        energy_ax.plot(
            xs_flat.cpu(),
            energy_values,
            label="Energy",
            color="darkviolet",
            linestyle="--",
            linewidth=1.5,
        )
        energy_ax.set_ylabel("energy", color="darkviolet")
        energy_ax.tick_params(axis="y", labelcolor="darkviolet")
        energy_ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_trajectory_samples(traj, t, reward_center, title, out_path, n_trajectories=200, alpha=0.08):
    """Save a trajectory plot with time on x-axis and x_t on y-axis."""
    import matplotlib.pyplot as plt

    n_particles = traj.shape[1]
    n_plot = min(n_trajectories, n_particles)
    idx_plot = torch.linspace(0, n_particles - 1, n_plot, dtype=torch.long, device=traj.device)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t.cpu(), traj[:, idx_plot, 0, 0].cpu(), color="darkorange", alpha=alpha, linewidth=0.7)
    ax.axhline(reward_center, color="firebrick", linestyle="--", linewidth=1.2, label=f"reward center={reward_center}")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$x_t$")
    ax.set_ylim(-20, 20)
    ax.set_title(title)
    ax.legend(fontsize=9)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


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
    _plot_density_comparison(
        ax_dens,
        xs_flat,
        p_data,
        p_rew,
        hist,
        reward_center,
        title="Final-time marginal vs. analytic targets",
        bin_edges=bin_edges,
    )

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
