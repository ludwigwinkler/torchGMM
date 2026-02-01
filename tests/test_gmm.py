import sys

import pytest
import torch
import einops

from torchGMM.gmm import TimeDependentGMM, Conditional
from torchGMM.schedule import BetaSchedule


def get_local_device():
    """Fixture that returns the best available device (MPS, CUDA, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


@pytest.fixture
def simple_gmm_2d():
    """Create a simple 2-component 2D GMM with known parameters"""
    mu = torch.tensor([[-1.0, 0.0], [1.0, 0.0]]).unsqueeze(0)  # [BS=1, k=2, d=2]
    sigma = torch.tensor([[0.5, 0.5], [0.5, 0.5]]).unsqueeze(0)  # [BS=1, k=2, d=2]
    weight = torch.tensor([0.5, 0.5]).unsqueeze(0)  # [BS=1, k=2]
    gmm = TimeDependentGMM(mu, sigma, weight)

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
        gmm = TimeDependentGMM(mu, sigma, weight)
        assert gmm.num_components == expected_num_components
        assert gmm.dim == expected_dim
        assert gmm.batch_shape == expected_batch_shape

    @pytest.mark.parametrize(
        "shape, t, expected_shape",
        [
            (None, 0.5, (5, 4, 2)),  # No sample dims, scalar t -> [*B, D]
            ((100, 5, 4), torch.ones(100, 5, 4), (100, 5, 4, 2)),  # No sample dims, [*B] t -> [*B, D]
            (None, None, (5, 4, 2)),
            (100, 0.5, (100, 5, 4, 2)),  # sample shape (100,), scalar t
            ((10, 20), 0.5, (10, 20, 5, 4, 2)),
            ((10, 20, 5, 4), torch.ones(10, 20, 5, 4), (10, 20, 5, 4, 2)),
        ],
    )
    def test_sample_shapes_various_inputs(self, shape, t, expected_shape):
        """Test sample(shape, t) output shapes. x: [*N,*B,D], t: scalar or [*B] or [*N,*B]."""
        mu = torch.randn(5, 4, 3, 2)  # [B1, B2, K, D]
        sigma = torch.ones(5, 4, 3, 2) * 0.5  # [B1, B2, K, D]
        weight = torch.ones(5, 4, 3)  # [B1, B2, K]
        gmm = TimeDependentGMM(mu, sigma, weight)
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
        gmm = TimeDependentGMM(mu, sigma, weight)
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
        gmm = TimeDependentGMM(mu, sigma, weight)
        x = torch.randn(*x_shape)
        score = gmm.score(x, t=t)
        assert score.shape == expected_shape, f"Expected {expected_shape}, got {score.shape}"

    @pytest.mark.parametrize(
        "sample_shape, t",
        [
            ((100,), None),
            ((100,), 0.5),
            ((100,), torch.tensor(0.5)),
            ((100,), torch.rand(5, 4)),  # t [*B]
            ((10, 20), 0.5),
            ((), None),
            ((), 0.5),
            ((), torch.rand(5, 4)),
        ],
    )
    def test_expand_t(self, sample_shape, t):
        """Test _expand_t returns shape (*sample_shape, *batch_shape)."""
        mu = torch.randn(5, 4, 3, 2)
        sigma = torch.ones(5, 4, 3, 2) * 0.5
        weight = torch.ones(5, 4, 3)
        gmm = TimeDependentGMM(mu, sigma, weight)
        t_exp = gmm._expand_t(t, sample_shape)
        assert t_exp.shape == (*sample_shape, 5, 4), f"Expected (*{sample_shape}, 2, 3), got {t_exp.shape}"

    def test_invalid_t_shape_raises(self):
        """Invalid t shape (neither [*B] nor [*N,*B]) raises."""
        mu = torch.randn(5, 4, 3, 2)
        sigma = torch.ones(5, 4, 3, 2) * 0.5
        weight = torch.ones(5, 4, 3)
        gmm = TimeDependentGMM(mu, sigma, weight)
        x = torch.randn(50, 5, 4, 2)  # [N, B, D], batch_shape=(5, 4)
        t_bad = torch.rand(50, 2)  # wrong trailing dims; need (5, 4) or (50, 5, 4)
        with pytest.raises(ValueError, match="t must be of shape"):
            gmm.log_prob(x, t=t_bad)

    def test_single_batch_dim_dropping(self):
        """Test score output shapes: x [*N,*B,D] -> score [*N,*B,D]."""
        mu1 = torch.randn(1, 5, 2)
        sigma1 = torch.ones(1, 5, 2) * 0.5
        weight1 = torch.ones(1, 5)
        gmm_single_batch = TimeDependentGMM(mu1, sigma1, weight1)
        mu2 = torch.randn(2, 5, 2)
        sigma2 = torch.ones(2, 5, 2) * 0.5
        weight2 = torch.ones(2, 5)
        gmm_multiple_batch = TimeDependentGMM(mu2, sigma2, weight2)
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
        """Test distribution properties for different times
        We initialize a conditional process with only mu=x0 and a GMM with mu, sigma, and weight imitating a conditional model.
        Then we compare the statistics of the sampled forward process at different times t.
        """
        mu = torch.randn(4, 1, 2)
        sigma = torch.zeros(4, 1, 2) + 1e-10
        weight = torch.ones(4, 1)
        gmm = TimeDependentGMM(mu, sigma, weight)  # GMM with mu and superfluous sigma, weight
        conditional = TimeDependentGMM(mu)  # Conditional GMM with only mu=x0
        alpha_t, sigma_t = gmm.schedule.get_alpha_t_sigma_t(torch.scalar_tensor(t))
        # Compute the true mean and variance of the forward process [batch=4,component=1,dim=2) -> x0=[batch=4,dim=2]
        true_mean = alpha_t * mu.squeeze(1)
        true_std = (sigma_t**2 + alpha_t * sigma.squeeze(1) ** 2) ** 0.5 * torch.ones_like(mu).squeeze(1)
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
        """Test distribution properties for different times
        We initialize a conditional process with only mu=x0 and a GMM with mu, sigma, and weight imitating a conditional model.
        Then we compare the statistics of the sampled forward process at different times t.

        Note: This test focuses on scalar time values to keep the logic simple.
        Each batch GMM is independent, so we evaluate samples from each batch separately.
        """
        mu = torch.randn(4, 1, 2)
        sigma = torch.zeros(4, 1, 2) + 1e-10
        weight = torch.ones(4, 1)
        gmm = TimeDependentGMM(mu, sigma, weight)  # GMM with mu and superfluous sigma, weight
        conditional = TimeDependentGMM(mu)  # Conditional GMM with only mu=x0

        # For scalar time, compute true parameters
        t_scalar = torch.scalar_tensor(t)
        alpha_t, sigma_t = gmm.schedule.get_alpha_t_sigma_t(t_scalar)
        true_mean = alpha_t * mu.squeeze(1)  # [BS, Dim]
        true_std = (sigma_t**2 + alpha_t * sigma.squeeze(1) ** 2) ** 0.5 * torch.ones_like(mu).squeeze(1)

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
        gmm = TimeDependentGMM(mu, sigma, weight)

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
        gmm = TimeDependentGMM(mu, sigma, weight)

        x = torch.randn(10, 4, 2)  # [N, B, D]
        score = gmm.score(x, t=0.5)
        x_copy = x.clone().detach().requires_grad_(True)
        log_p = gmm.log_prob(x_copy, t=0.5)
        grad = torch.autograd.grad(log_p.sum(), x_copy)[0]

        # Score should be [10, 4, 2], grad should be [10, 4, 2]
        assert score.shape == (10, 4, 2), f"Expected (10, 4, 2), got {score.shape}"
        assert grad.shape == (10, 4, 2), f"Expected (10, 4, 2), got {grad.shape}"

        torch.testing.assert_close(score, grad, atol=1e-5, rtol=1e-4)


class TestMarginalDistributions:
    """Test marginal distribution extraction from TimeDependentGMM"""

    def test_marginal_2d_empirical_comparison(self):
        """Compare empirical histograms with analytical marginal distributions"""
        # Create 3-component 2D GMM
        mu = torch.tensor([[2.0, -2.0], [-2.0, 3.0], [0.0, 0.0]]).unsqueeze(0)  # [1, 3, 2]
        sigma = torch.tensor([[0.5, 0.4], [0.5, 0.5], [0.3, 0.4]]).unsqueeze(0)  # [1, 3, 2]
        weight = torch.tensor([0.3, 0.4, 0.3]).unsqueeze(0)  # [1, 3]
        gmm_2d = TimeDependentGMM(mu, sigma, weight)

        # Sample from the GMM: [n_samples, BS=1, Dim=2]
        n_samples = 100_000
        samples = gmm_2d.sample(shape=n_samples, t=0.0)

        # Set up histogram parameters
        x_min, x_max = -5.0, 5.0
        n_bins = 100
        bin_edges = torch.linspace(x_min, x_max, n_bins + 1)  # [n_bins+1]
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])  # [n_bins]
        bin_width = (x_max - x_min) / n_bins

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
        gmm = TimeDependentGMM(mu, sigma, weight)
        for temperature in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]:
            temperature_gmm = TimeDependentGMM(mu, sigma / temperature**0.5, weight)
            # Build a two dimensional meshgrid with points
            x_min, x_max, n_points = -3.0, 3.0, 100
            x_grid = torch.linspace(x_min, x_max, n_points)
            mesh_points = x_grid.unsqueeze(-1).unsqueeze(-1)  # [N, 1, 1]
            log_prob = gmm.log_prob(mesh_points, t=0.0)
            tempered_log_prob = temperature * log_prob
            temperature_log_prob = temperature_gmm.log_prob(mesh_points, t=0.0)
            prob = torch.exp(log_prob) / torch.exp(log_prob).sum()
            tempered_prob = torch.exp(tempered_log_prob) / torch.exp(tempered_log_prob).sum()
            temperature_prob = torch.exp(temperature_log_prob) / torch.exp(temperature_log_prob).sum()
            torch.testing.assert_close(
                temperature_prob, tempered_prob, atol=0.01, rtol=0.1, msg=f"Temperature: {temperature}"
            )

    def test_temperature_sampling(self):
        """
        Test importance sampling from a tempered GMM
        We're sampling with weights
        1. w = p^beta / p = p^(beta - 1)
        2. w = w / sum_j w_j
        3. multinomial(w)

        Only valid for a 1D Gaussian
        """
        mu = torch.randn(1, 1, 1)
        sigma = torch.ones(1, 1, 1) * 0.5
        weight = torch.ones(1, 1)
        gmm = TimeDependentGMM(mu, sigma, weight)
        for temperature in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]:
            temperature_gmm = TimeDependentGMM(mu, sigma / temperature**0.5, weight)
            # Build a one dimensional meshgrid with points
            x_min, x_max, n_points = -3.0, 3.0, 100
            x_grid = torch.linspace(x_min, x_max, n_points)
            mesh_points = x_grid.unsqueeze(-1).unsqueeze(-1)  # [N, 1, 1]
            log_prob = gmm.log_prob(mesh_points, t=0.0)
            tempered_log_prob = temperature * log_prob
            temperature_log_prob = temperature_gmm.log_prob(mesh_points, t=0.0)
            prob = torch.exp(log_prob) / torch.exp(log_prob).sum()
            tempered_prob = torch.exp(tempered_log_prob) / torch.exp(tempered_log_prob).sum()
            temperature_prob = torch.exp(temperature_log_prob) / torch.exp(temperature_log_prob).sum()
            torch.testing.assert_close(
                temperature_prob, tempered_prob, atol=0.01, rtol=0.1, msg=f"Temperature: {temperature}"
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


class TestDeviceHandling:
    """Test device handling for TimeDependentGMM"""

    @pytest.mark.parametrize("local_device", [torch.device("cpu"), get_local_device()])
    def test_log_prob_and_score_on_device(self, simple_gmm_2d, local_device):
        """
        Create GMM on the specified device, evaluate log_prob and score.
        """
        gmm, expected_mean, expected_std = simple_gmm_2d

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
