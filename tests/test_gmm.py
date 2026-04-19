import pytest
import torch
from conftest import get_local_device

from torchGMM.gmm import GMM, Conditional
from torchGMM.schedule import BetaSchedule, LinearSchedule


@pytest.fixture
def simple_gmm_2d():
    """Create a simple 2-component 2D GMM with known parameters"""
    mu = torch.tensor([[-1.0, 0.0], [1.0, 0.0]]).unsqueeze(0)  # [BS=1, k=2, d=2]
    sigma = torch.tensor([[0.5, 0.5], [0.5, 0.5]]).unsqueeze(0)  # [BS=1, k=2, d=2]
    weight = torch.tensor([0.5, 0.5]).unsqueeze(0)  # [BS=1, k=2]
    gmm = GMM(mu, sigma, weight)

    # Expected statistics for t=0 (no diffusion)
    # Mean: weighted average of component means = 0.5*[-1,0] + 0.5*[1,0] = [0,0]
    expected_mean = torch.tensor([0.0, 0.0])
    # Variance for mixture: E[X^2] - E[X]^2
    # E[X^2] = 0.5*(mu1^2 + sigma1^2) + 0.5*(mu2^2 + sigma2^2)
    # For dimension 0: 0.5*(1 + 0.25) + 0.5*(1 + 0.25) = 1.25
    # Var = 1.25 - 0 = 1.25, std = sqrt(1.25) ≈ 1.118
    expected_std = torch.tensor([1.118, 0.5])  # dim 0 has mixture variance, dim 1 just component variance

    return gmm, expected_mean, expected_std


class TestShapes:
    """Test shape validation for all GMM methods"""

    @pytest.mark.parametrize(
        "mu_shape, sigma_shape, weight_shape, expected_num_components, expected_dim, expected_batch_shape",
        [
            ((2, 3, 2), (2, 3, 2), (2, 3), 3, 2, (2,)),
            ((10, 9, 3, 2), (10, 9, 3, 2), (10, 9, 3), 3, 2, (10, 9)),
        ],
    )
    def test_gmm_initialization(
        self, mu_shape, sigma_shape, weight_shape, expected_num_components, expected_dim, expected_batch_shape
    ):
        """Test GMM can be initialized with various component counts and dimensions"""
        # 2 components, 2D, batch_shape=(2, 3)
        mu = torch.ones(mu_shape)
        sigma = torch.ones(sigma_shape) * 0.5
        weight = torch.ones(weight_shape)
        gmm = GMM(mu, sigma, weight)
        assert gmm.num_components == expected_num_components
        assert gmm.dim == expected_dim
        assert gmm.batch_shape == expected_batch_shape

    @pytest.mark.parametrize(
        "shape, t, expected_shape",
        [
            (None, 0.5, (5, 4, 2)),  # shape=None -> [*B,D], scalar t
            ((100, 5, 4), torch.ones(100, 5, 4), (100, 5, 4, 2)),  # shape [*N,*B], t [*N,*B]
            (None, None, (5, 4, 2)),
            (100, 0.5, (100, 5, 4, 2)),  # shape=int -> one sample dim, scalar t
            ((10, 20, 5, 4), 0.5, (10, 20, 5, 4, 2)),  # shape [*N,*B], scalar t
            ((10, 20, 5, 4), torch.ones(10, 20, 5, 4), (10, 20, 5, 4, 2)),  # shape [*N,*B], t [*N,*B]
        ],
    )
    def test_sample_shapes_various_inputs(self, shape, t, expected_shape):
        """Test sample(shape, t). shape: full [*N,*B]; t: [*N,*B] or scalar (broadcast). Output [*N,*B,D]."""
        mu = torch.randn(5, 4, 3, 2)  # [B1, B2, K, D]
        sigma = torch.ones(5, 4, 3, 2) * 0.5  # [B1, B2, K, D]
        weight = torch.ones(5, 4, 3)  # [B1, B2, K]
        gmm = GMM(mu, sigma, weight)
        samples = gmm.sample(shape=shape, t=t)
        assert samples.shape == expected_shape, f"Expected {expected_shape}, got {samples.shape}"

    @pytest.mark.parametrize(
        "x_shape, t, expected_shape",
        [
            ((50, 5, 4, 2), 0.5, (50, 5, 4)),  # x [*N,*B,D], scalar t
            ((50, 5, 4, 2), None, (50, 5, 4)),  # x [*N,*B,D], no t -> t=0
            ((5, 4, 2), 0.5, (5, 4)),  # x [*B,D] only
        ],
    )
    def test_logprob_shapes_various_inputs(self, x_shape, t, expected_shape):
        """Test log_prob(x, t) output shapes. x: [*N,*B,D], t: scalar or [*B] or [*N,*B]."""
        mu = torch.randn(5, 4, 3, 2)
        sigma = torch.ones(5, 4, 3, 2) * 0.5
        weight = torch.ones(5, 4, 3)
        gmm = GMM(mu, sigma, weight)
        x = torch.randn(*x_shape)
        log_p = gmm.log_prob(x, t=t)
        assert log_p.shape == expected_shape, f"Expected {expected_shape}, got {log_p.shape}"

    @pytest.mark.parametrize(
        "x_shape, t, expected_shape",
        [
            ((50, 5, 4, 2), 0.5, (50, 5, 4, 2)),  # x [*N,*B,D], t scalar -> t [*N,*B]
            ((5, 4, 2), 0.5, (5, 4, 2)),  # x [*B,D], t scalar -> t [*B]
            ((5, 10, 5, 4, 2), 0.5, (5, 10, 5, 4, 2)),  # x [*N,*B,D], t scalar -> t [*N,*B]
        ],
    )
    def test_score_shapes_various_inputs(self, x_shape, t, expected_shape):
        """Test score(x, t) output shapes. x: [*N,*B,D], t: scalar or [*B] or [*N,*B]."""
        mu = torch.randn(5, 4, 3, 2)
        sigma = torch.ones(5, 4, 3, 2) * 0.5
        weight = torch.ones(5, 4, 3)
        gmm = GMM(mu, sigma, weight)
        x = torch.randn(*x_shape)
        score = gmm.score(x, t=t)
        assert score.shape == expected_shape, f"Expected {expected_shape}, got {score.shape}"

    @pytest.mark.parametrize(
        "sample_shape, t, sample_shape_expected",
        [
            ((100,), None, (100, 5, 4)),
            ((100,), 0.5, (100, 5, 4)),
            ((100,), torch.tensor(0.5), (100, 5, 4)),
            ((10, 20), 0.5, (10, 20, 5, 4)),
            ((), None, (5, 4)),
            ((), 0.5, (5, 4)),
            ((), torch.rand(5, 4), (5, 4)),
        ],
    )
    def test_expand_t(self, sample_shape, t, sample_shape_expected):
        """Test _expand_t returns shape (*sample_shape, *batch_shape)."""
        mu = torch.randn(5, 4, 3, 2)
        sigma = torch.ones(5, 4, 3, 2) * 0.5
        weight = torch.ones(5, 4, 3)
        gmm = GMM(mu, sigma, weight)
        t_exp = gmm._expand_t(t, sample_shape)
        assert t_exp.shape == sample_shape_expected, f"Expected (*{sample_shape}, 5, 4), got {t_exp.shape}"

    def test_invalid_t_shape_raises(self):
        """Invalid t shape (neither [*B] nor [*N,*B]) raises."""
        mu = torch.randn(5, 4, 3, 2)
        sigma = torch.ones(5, 4, 3, 2) * 0.5
        weight = torch.ones(5, 4, 3)
        gmm = GMM(mu, sigma, weight)
        x = torch.randn(50, 5, 4, 2)  # [N, B, D], batch_shape=(5, 4)
        t_bad = torch.rand(50, 2)  # wrong trailing dims; need (5, 4) or (50, 5, 4)
        with pytest.raises(ValueError, match="t shape"):
            gmm.log_prob(x, t=t_bad)

    @pytest.mark.parametrize(
        "x_shape, expected_message",
        [
            ((50, 5, 4, 3), "x last dim must be"),  # wrong D
            ((50, 5, 3, 2), "x must have batch dims"),  # wrong batch dims
        ],
    )
    def test_invalid_x_shape_raises(self, x_shape, expected_message):
        """Invalid x shapes should raise for log_prob and score."""
        mu = torch.randn(5, 4, 3, 2)
        sigma = torch.ones(5, 4, 3, 2) * 0.5
        weight = torch.ones(5, 4, 3)
        gmm = GMM(mu, sigma, weight)
        x_bad = torch.randn(*x_shape)
        with pytest.raises(AssertionError, match=expected_message):
            gmm.log_prob(x_bad, t=0.5)
        with pytest.raises(AssertionError, match=expected_message):
            gmm.score(x_bad, t=0.5)

    def test_invalid_t_shape_raises_for_score(self):
        """Invalid t shape should raise for score."""
        mu = torch.randn(5, 4, 3, 2)
        sigma = torch.ones(5, 4, 3, 2) * 0.5
        weight = torch.ones(5, 4, 3)
        gmm = GMM(mu, sigma, weight)
        x = torch.randn(50, 5, 4, 2)  # [N, B, D], batch_shape=(5, 4)
        t_bad = torch.rand(5, 2)  # wrong trailing dims; need (5, 4) or (50, 5, 4)
        with pytest.raises(ValueError, match="t shape"):
            gmm.score(x, t=t_bad)

    def test_invalid_t_shape_raises_for_sample(self):
        """Invalid t shape should raise for sample."""
        mu = torch.randn(5, 4, 3, 2)
        sigma = torch.ones(5, 4, 3, 2) * 0.5
        weight = torch.ones(5, 4, 3)
        gmm = GMM(mu, sigma, weight)
        t_bad = torch.rand(5, 2)  # wrong batch dims; need (5, 4)
        with pytest.raises(ValueError, match="t shape"):
            gmm.sample(shape=None, t=t_bad)

    def test_single_batch_dim_dropping(self):
        """Test score output shapes: x [*N,*B,D] -> score [*N,*B,D]."""
        mu1 = torch.randn(1, 5, 2)
        sigma1 = torch.ones(1, 5, 2) * 0.5
        weight1 = torch.ones(1, 5)
        gmm_single_batch = GMM(mu1, sigma1, weight1)
        mu2 = torch.randn(2, 5, 2)
        sigma2 = torch.ones(2, 5, 2) * 0.5
        weight2 = torch.ones(2, 5)
        gmm_multiple_batch = GMM(mu2, sigma2, weight2)
        x_single = torch.randn(50, 1, 2)  # [N, B=1, D]
        x_multi = torch.randn(50, 2, 2)  # [N, B=2, D]
        score_single = gmm_single_batch.score(x_single, t=0.5)
        score_multi = gmm_multiple_batch.score(x_multi, t=0.5)
        assert score_single.shape == (50, 1, 2)
        assert score_multi.shape == (50, 2, 2)


class TestDistribution:
    """Test distribution properties"""

    @pytest.mark.parametrize("t", [0.0, 0.5, 0.9, 1.0])
    def test_conditional_vs_gmm(self, t):
        """Test distribution properties for different times.

        Initialize a conditional process with only mu=x0 and a GMM with mu, sigma, and weight
        imitating a conditional model. Compare forward-process statistics at different times t.
        """
        mu = torch.randn(4, 1, 2)
        sigma = torch.zeros(4, 1, 2) + 1e-10
        weight = torch.ones(4, 1)
        gmm = GMM(mu, sigma, weight)  # GMM with mu and superfluous sigma, weight
        conditional = GMM(mu)  # Conditional GMM with only mu=x0
        alpha_t, sigma_t = gmm.schedule.get_alpha_t_sigma_t(torch.scalar_tensor(t))
        # Compute the true mean and variance of the forward process [batch=4,component=1,dim=2) -> x0=[batch=4,dim=2]
        true_mean = alpha_t * mu.squeeze(1)
        true_std = (sigma_t**2 + alpha_t**2 * sigma.squeeze(1) ** 2) ** 0.5 * torch.ones_like(mu).squeeze(1)
        # Create Conditional GMM and GMM with the same parameters and sample from them

        gmm_samples = gmm.sample(shape=1_000_000, t=t)  # [1_000_000, BS=4, Dim=2]
        conditional_samples = conditional.sample(shape=500_000, t=t)  # [500_000, BS=4, Dim=2]
        # Compute moments are compare
        ground_truth_moments = [true_mean, true_std]
        gmm_moments = [gmm_samples.mean(dim=0), gmm_samples.std(dim=0)]  # [N, BS, Dim] -[mean,std](0)-> [BS, Dim]
        conditional_moments = [conditional_samples.mean(dim=0), conditional_samples.std(dim=0)]
        torch.testing.assert_close(ground_truth_moments, gmm_moments, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(gmm_moments, conditional_moments, atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize("t", [0.0, 0.5, 0.9, 1.0])
    def test_conditional_vs_gmm_score(self, t):
        """Test score agreement between Conditional and equivalent GMM at different times.

        Note: This test focuses on scalar time values to keep the logic simple.
        Each batch GMM is independent, so we evaluate samples from each batch separately.
        """
        mu = torch.randn(4, 1, 2)
        sigma = torch.zeros(4, 1, 2) + 1e-10
        weight = torch.ones(4, 1)
        gmm = GMM(mu, sigma, weight)  # GMM with mu and superfluous sigma, weight
        conditional = GMM(mu)  # Conditional GMM with only mu=x0

        # For scalar time, compute true parameters
        t_scalar = torch.scalar_tensor(t)
        alpha_t, sigma_t = gmm.schedule.get_alpha_t_sigma_t(t_scalar)
        true_mean = alpha_t * mu.squeeze(1)  # [BS, Dim]
        true_std = (sigma_t**2 + alpha_t**2 * sigma.squeeze(1) ** 2) ** 0.5 * torch.ones_like(mu).squeeze(1)

        # Sample: [50, BS, Dim]
        gmm_samples = gmm.sample(shape=50, t=t)
        gmm_score_all = gmm.score(gmm_samples, t=t)  # [50, BS, Dim]
        conditional_score_all = conditional.score(gmm_samples, t=t)

        # For each batch index b, extract samples for that batch and evaluate
        # We want to compare score of samples[n, b, :] evaluated at GMM b
        for b in range(gmm.batch_shape[0]):
            # Extract samples for this batch: [50, Dim]
            samples_b = gmm_samples[:, b, :]

            # Compute true score for this batch: [50, Dim]
            true_score_b = -(samples_b - true_mean[b : b + 1, :]) / true_std[b : b + 1, :] ** 2

            # Extract scores for this batch: [50, Dim]
            gmm_score_b = gmm_score_all[:, b, :]
            conditional_score_b = conditional_score_all[:, b, :]

            # Compare
            torch.testing.assert_close(true_score_b, gmm_score_b, atol=1e-2, rtol=1e-2)
            torch.testing.assert_close(gmm_score_b, conditional_score_b, atol=1e-2, rtol=1e-2)


class TestGMMProperties:
    """Test mathematical properties of the GMM"""

    def test_log_prob_vs_energy(self):
        """Test that energy = -log_prob"""
        mu = torch.randn(1, 3, 2)
        sigma = torch.ones(1, 3, 2) * 0.5
        weight = torch.ones(1, 3)
        gmm = GMM(mu, sigma, weight)

        x = torch.randn(50, 1, 2)  # [N, B, D]
        log_p = gmm.log_prob(x, t=0.5)
        energy = gmm.energy(x, t=0.5)

        torch.testing.assert_close(energy, -log_p)

    def test_score_is_gradient(self):
        """
        Test that score equals gradient of log_prob
        """
        mu = torch.randn(4, 3, 2)
        sigma = torch.ones(4, 3, 2) * 0.5
        weight = torch.ones(4, 3)
        gmm = GMM(mu, sigma, weight)

        x = torch.randn(10, 4, 2)  # [N, B, D]
        score = gmm.score(x, t=0.5)
        x_copy = x.clone().detach().requires_grad_(True)
        log_p = gmm.log_prob(x_copy, t=0.5)
        grad = torch.autograd.grad(log_p.sum(), x_copy)[0]

        # Score should be [10, 4, 2], grad should be [10, 4, 2]
        assert score.shape == (10, 4, 2), f"Expected (10, 4, 2), got {score.shape}"
        assert grad.shape == (10, 4, 2), f"Expected (10, 4, 2), got {grad.shape}"

        torch.testing.assert_close(score, grad, atol=1e-5, rtol=1e-4)


class Test2DMarginalizedDistributions:
    """Test marginal distribution extraction from GMM"""

    @pytest.mark.slow
    def test_marginal_2d_empirical_comparison(self):
        """Compare empirical histograms with analytical marginal distributions"""
        torch.manual_seed(0)
        # Create 3-component 2D GMM
        mu = torch.tensor([[2.0, -2.0], [-2.0, 3.0], [0.0, 0.0]]).unsqueeze(0)  # [1, 3, 2]
        sigma = torch.tensor([[0.5, 0.4], [0.5, 0.5], [0.3, 0.4]]).unsqueeze(0)  # [1, 3, 2]
        weight = torch.tensor([0.3, 0.4, 0.3]).unsqueeze(0)  # [1, 3]
        gmm_2d = GMM(mu, sigma, weight)

        # Sample from the GMM: [n_samples, BS=1, Dim=2]
        n_samples = 100_000
        samples = gmm_2d.sample(shape=n_samples, t=0.0)

        # Set up histogram parameters
        x_min, x_max = -5.0, 5.0
        n_bins = 100
        bin_edges = torch.linspace(x_min, x_max, n_bins + 1)  # [n_bins+1]
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])  # [n_bins]

        # Test dimension 0
        empirical_0 = samples[:, :, 0]  # [n_samples, BS=1]
        hist_empirical_0, _ = torch.histogram(empirical_0, bins=bin_edges, density=True)  # [n_bins]

        marginal_0 = gmm_2d.marginal_gmm(dim=0)
        bin_centers_reshaped = bin_centers.unsqueeze(-1).unsqueeze(-1)  # [n_bins, 1, 1]
        log_probs_0 = marginal_0.log_prob(bin_centers_reshaped, t=0.0)
        analytical_probs_0 = torch.exp(log_probs_0)  # [n_bins, BS=1]

        # Compare pointwise
        torch.testing.assert_close(hist_empirical_0, analytical_probs_0[:, 0], atol=0.01, rtol=0.1)

        # Test dimension 1
        empirical_1 = samples[:, :, 1]  # [n_samples, BS=1]
        hist_empirical_1, _ = torch.histogram(empirical_1, bins=bin_edges, density=True)  # [n_bins]

        marginal_1 = gmm_2d.marginal_gmm(dim=1)
        log_probs_1 = marginal_1.log_prob(bin_centers_reshaped, t=0.0)
        analytical_probs_1 = torch.exp(log_probs_1)  # [n_bins, BS=1]

        # Compare pointwise
        torch.testing.assert_close(hist_empirical_1, analytical_probs_1[:, 0], atol=0.01, rtol=0.1)


class TestTemperatureSampling:
    """Test temperature sampling from the GMM"""

    def test_temperature_probability(self):
        """
        Test temperature sampling from the GMM
        1D Gaussian N(mu, sigma^2)^\beta = N(mu, var=sigma^2/beta) = N(mu, std=sigma/sqrt(beta))
        We're testing in probability space (not log probability space)
        Log probability space has values that are too large and the tests trigger negatively
        """
        mu = torch.randn(1, 1, 1)
        sigma = torch.ones(1, 1, 1) * 0.5
        weight = torch.ones(1, 1)
        gmm = GMM(mu, sigma, weight)
        for temperature in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]:
            temperature_gmm = GMM(mu, sigma / temperature**0.5, weight)
            # Build a two dimensional meshgrid with points
            x_min, x_max, n_points = -3.0, 3.0, 100
            x_grid = torch.linspace(x_min, x_max, n_points)
            mesh_points = x_grid.unsqueeze(-1).unsqueeze(-1)  # [N, 1, 1]
            log_prob = gmm.log_prob(mesh_points, t=0.0)
            tempered_log_prob = temperature * log_prob
            temperature_log_prob = temperature_gmm.log_prob(mesh_points, t=0.0)
            tempered_prob = torch.exp(tempered_log_prob) / torch.exp(tempered_log_prob).sum()
            temperature_prob = torch.exp(temperature_log_prob) / torch.exp(temperature_log_prob).sum()
            torch.testing.assert_close(
                temperature_prob, tempered_prob, atol=0.01, rtol=0.1, msg=f"Temperature: {temperature}"
            )

    @pytest.mark.slow
    def test_temperature_importance_sampling(self):
        """
        Test importance sampling from a tempered GMM.
        Samples from p(x), reweights by w ∝ p(x)^(β-1), then resamples.
        The resulting histogram should match the tempered GMM N(mu, sigma²/β).
        """
        torch.manual_seed(0)
        mu = torch.randn(1, 1, 1)
        sigma = torch.ones(1, 1, 1) * 0.5
        weight = torch.ones(1, 1)
        gmm = GMM(mu, sigma, weight)

        for temperature in [0.5, 1.0, 2.0]:
            temperature_gmm = GMM(mu, sigma / temperature**0.5, weight)

            # Sample from p, reweight by p^(β-1), resample
            n_particles = 100_000
            samples = gmm.sample(shape=n_particles, t=0.0)  # [N, 1, 1]
            log_p = gmm.log_prob(samples, t=0.0)  # [N, 1]
            log_w = (temperature - 1) * log_p  # [N, 1]
            log_w = log_w - log_w.max(dim=0).values  # stabilise
            w = log_w.exp()
            w = w / w.sum(dim=0)  # normalise per batch
            idx = torch.multinomial(w[:, 0], n_particles, replacement=True)
            resampled = samples[idx, 0, 0]  # [N]

            # Compare histogram of resampled to tempered GMM density
            bin_edges = torch.linspace(-3.0, 3.0, 51)
            hist, _ = torch.histogram(resampled, bins=bin_edges, density=True)
            # Evaluate target at bin centres
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:]).reshape(-1, 1, 1)
            target_bins = temperature_gmm.log_prob(bin_centers, t=0.0).exp().squeeze()
            assert (hist - target_bins).abs().max() < 0.05, (
                f"IS resample @ temp={temperature}: max deviation {(hist - target_bins).abs().max():.3f}"
            )


class TestConditional:
    def test_conditional_shapes(self):
        """Test Conditional initialization and method shapes"""
        # Test with 2D input
        x0 = torch.randn(10, 2)
        cond_gmm = Conditional(x0)
        assert cond_gmm.mu.shape == (10, 1, 2)
        assert cond_gmm.sigma.shape == (10, 1, 2)
        assert cond_gmm.weight.shape == (10, 1)
        assert cond_gmm.batch_shape == (10,)
        assert cond_gmm.event_shape == (2,)

        # Test sampling with no shape -> [BS, Dim]
        samples = cond_gmm.sample()
        assert samples.shape == (10, 2), f"Expected (10, 2), got {samples.shape}"

        # Test sampling with shape
        samples = cond_gmm.sample(shape=50)
        assert samples.shape == (50, 10, 2), f"Expected (50, 10, 2), got {samples.shape}"

        x = torch.randn(11, 10, 2)  # [N, B, D]
        log_prob = cond_gmm.log_prob(x, t=0.0)
        assert log_prob.shape == (11, 10), f"Expected (11, 10), got {log_prob.shape}"
        score = cond_gmm.score(x, t=0.0)
        assert score.shape == (11, 10, 2), f"Expected (11, 10, 2), got {score.shape}"
        energy = cond_gmm.energy(x, t=0.0)
        assert energy.shape == (11, 10), f"Expected (11, 10), got {energy.shape}"


class TestVelocity:
    """Test velocity computation on GMM"""

    @pytest.mark.parametrize(
        "x_shape, t, expected_shape",
        [
            ((50, 5, 4, 2), 0.5, (50, 5, 4, 2)),
            ((5, 4, 2), 0.5, (5, 4, 2)),
            ((5, 10, 5, 4, 2), 0.5, (5, 10, 5, 4, 2)),
        ],
    )
    def test_velocity_shapes(self, x_shape, t, expected_shape):
        """Test velocity(x, t) output shapes match score shapes."""
        mu = torch.randn(5, 4, 3, 2)
        sigma = torch.ones(5, 4, 3, 2) * 0.5
        weight = torch.ones(5, 4, 3)
        gmm = GMM(mu, sigma, weight)
        x = torch.randn(*x_shape)
        v = gmm.velocity(x, t=t)
        assert v.shape == expected_shape, f"Expected {expected_shape}, got {v.shape}"

    @pytest.mark.parametrize("schedule_cls", [BetaSchedule, LinearSchedule])
    def test_score_velocity_consistency(self, schedule_cls):
        """Verify v = (dα/dt / α) x + (dα/dt σ/α - dσ/dt) σ score for both schedules — the Tweedie velocity formula."""
        schedule = schedule_cls() if schedule_cls == LinearSchedule else schedule_cls(beta_min=0.1, beta_max=20.0)
        mu = torch.randn(1, 3, 2)
        sigma = torch.ones(1, 3, 2) * 0.5
        weight = torch.ones(1, 3)
        gmm = GMM(mu, sigma, weight, schedule=schedule)

        t_val = 0.5
        x = torch.randn(20, 1, 2)
        v = gmm.velocity(x, t=t_val)

        # Manually compute velocity from score
        t_tensor = torch.full((20, 1), t_val)
        alpha_t = schedule.get_alpha_t(t_tensor)
        sigma_t = schedule.get_sigma_t(t_tensor)
        dalpha_dt = schedule.get_dalpha_dt(t_tensor)
        dsigma_dt = schedule.get_dsigma_dt(t_tensor)
        score = gmm.score(x, t=t_val)

        coeff_x = (dalpha_dt / alpha_t).unsqueeze(-1)
        coeff_score = ((dalpha_dt * sigma_t / alpha_t - dsigma_dt) * sigma_t).unsqueeze(-1)
        v_expected = coeff_x * x + coeff_score * score

        torch.testing.assert_close(v, v_expected, atol=1e-5, rtol=1e-4)

    def test_flow_matching_velocity_formula(self):
        """For flow matching (α=1-t, σ=t): v = -(x + t·score) / (1-t).

        Derivation: dα/dt=-1, dσ/dt=1, α=1-t, σ=t
          coeff_x = dα/dt / α = -1/(1-t)
          coeff_score = (dα/dt·σ/α - dσ/dt)·σ = (-t/(1-t) - 1)·t = -t/(1-t)
          v = -x/(1-t) - t·score/(1-t) = -(x + t·score)/(1-t)
        """
        schedule = LinearSchedule()
        mu = torch.randn(1, 3, 2)
        sigma = torch.ones(1, 3, 2) * 0.5
        weight = torch.ones(1, 3)
        gmm = GMM(mu, sigma, weight, schedule=schedule)

        t_val = 0.4
        x = torch.randn(20, 1, 2)
        v = gmm.velocity(x, t=t_val)

        score = gmm.score(x, t=t_val)
        v_expected = -(x + t_val * score) / (1 - t_val)

        torch.testing.assert_close(v, v_expected, atol=1e-5, rtol=1e-4)

    def test_beta_schedule_velocity_formula(self):
        """For BetaSchedule VP-SDE (α²+σ²=1): v = -½β(t)·(x + score).

        Derivation: dα/dt=-½β·α, dσ/dt=½β·α²/σ
          coeff_x = dα/dt / α = -½β
          coeff_score = (dα/dt·σ/α - dσ/dt)·σ = (-½β·σ - ½β·α²/σ)·σ = -½β·(σ²+α²) = -½β  (VP: α²+σ²=1)
          v = -½β·x - ½β·score = -½β·(x + score)
        """
        schedule = BetaSchedule()
        mu = torch.randn(1, 3, 2)
        sigma = torch.ones(1, 3, 2) * 0.5
        weight = torch.ones(1, 3)
        gmm = GMM(mu, sigma, weight, schedule=schedule)

        t_val = 0.5
        x = torch.randn(20, 1, 2)
        v = gmm.velocity(x, t=t_val)

        t_tensor = torch.full((20, 1), t_val)
        half_beta = (0.5 * schedule.beta(t_tensor)).unsqueeze(-1)  # [20, 1, 1]
        score = gmm.score(x, t=t_val)
        v_expected = -half_beta * (x + score)

        torch.testing.assert_close(v, v_expected, atol=1e-5, rtol=1e-4)


class TestDeviceHandling:
    """Test device handling for GMM"""

    @pytest.mark.parametrize("local_device", [torch.device("cpu"), get_local_device()])
    def test_log_prob_and_score_on_device(self, simple_gmm_2d, local_device):
        """
        Create GMM on the specified device, evaluate log_prob and score.
        """
        gmm, _, _ = simple_gmm_2d

        # Move GMM to the desired device
        gmm = gmm.to(local_device)

        x = torch.tensor([[0.0, 0.0]], device=local_device)  # [B=1, D=2]
        t = torch.tensor(0.0, device=local_device)
        log_prob = gmm.log_prob(x, t=t)
        score = gmm.score(x, t=t)
        assert log_prob.shape == (1,)  # [*B]
        assert score.shape == (1, 2)  # [*B, D]

        assert log_prob.device == x.device
        assert score.device == x.device
