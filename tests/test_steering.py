from pathlib import Path

import pytest
import torch

from torchGMM.gmm import GMM
from torchGMM.sampling import _ess_ratio, steered_reverse_sampling
from torchGMM.schedule import BetaSchedule, KarrasSchedule

torch.set_printoptions(sci_mode=False)

PLOT = False  # flip to True locally to save FKC steering diagnostic plots next to this file
PLOT_DIR = Path(__file__).parent


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
            (-2.0, 1.0, 0.9),
            (-1.5, 1.0, 0.9),
            (-1.5, 1.5, 0.9),
            (-1.0, 1.0, 0.9),
            (-1.0, 1.5, 0.9),
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
        xs = torch.linspace(-6, 6, 250).reshape(-1, 1, 1)
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

        # 4. Wasserstein-1 vs reward-tilted target: hard threshold, must pass regardless
        # of how the unguided distribution compares
        w1_rew = _wasserstein1(x_final, xs_flat, p_rew)
        assert w1_rew < 0.05, f"center={reward_center} sigma={reward_sigma}: W1 vs reward-tilted={w1_rew:.4f}"

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
                out_path=PLOT_DIR / f"steered_center{reward_center}_sigma{reward_sigma}_ess{ess_threshold}.png",
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
    N_STEPS = 200
    N_DENOISE_STEPS = 5

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
    @pytest.mark.parametrize("reward_center", [-1.5, -0.5, 1.0], ids=lambda v: f"{v}")
    def test_steered_sampling_karras(self, setup, reward_center, ess_threshold):
        gmm, sched = setup
        reward_sigma = 1.0

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
        xs = torch.linspace(-6, 6, 100).reshape(-1, 1, 1)
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
                out_path=PLOT_DIR / f"steered_karras_center{reward_center}_ess{ess_threshold}.png",
            )
        assert w1_rew < 0.075, (
            f"KarrasSchedule reward_center={reward_center} ess_threshold={ess_threshold}: "
            f"W1 vs reward-tilted={w1_rew:.4f}"
        )


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
