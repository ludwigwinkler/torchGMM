import sys

import pytest
import torch
import einops


from torchGMM.diffusion import (
    denoising_and_resample_fkc,
    denoising_and_resample_fksmc,
    denoising_and_resample_smc,
    denoising_and_resample_smclangevin,
    reverse_diffusion,
    reverse_diffusion_with_regular_resampling,
)
from torchGMM.gmm import TimeDependentGMM
from torchGMM.schedule import BetaSchedule


@pytest.fixture
def simple_gmm_2d():
    """Create a simple 2-component 2D GMM with known parameters"""
    mu = torch.tensor([[-1.0, 0.0], [1.0, 0.0]])  # [k=2, d=2]
    sigma = torch.tensor([[0.5, 0.5], [0.5, 0.5]])  # [k=2, d=2]
    weight = torch.tensor([0.5, 0.5])  # [k=2]
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

    def test_gmm_initialization(self):
        """Test GMM can be initialized with various component counts and dimensions"""
        # 2 components, 2D
        mu = torch.randn(3, 2, 2)
        sigma = torch.ones(3, 2, 2) * 0.5
        weight = torch.ones(3, 2)
        gmm = TimeDependentGMM(mu, sigma, weight)
        assert gmm.num_components == 2
        assert gmm.dim == 2
        assert gmm.batch_size == 3

        # 5 components, 3D
        mu = torch.randn(10, 5, 3)
        sigma = torch.ones(10, 5, 3) * 0.5
        weight = torch.ones(10, 5)
        gmm = TimeDependentGMM(mu, sigma, weight)
        assert gmm.num_components == 5
        assert gmm.dim == 3
        assert gmm.batch_size == 10

    def test_sample_shapes_various_inputs(self):
        """Test sample() output shapes with different time inputs"""
        mu = torch.randn(4, 3, 2)
        sigma = torch.ones(4, 3, 2) * 0.5
        weight = torch.ones(4, 3)
        gmm = TimeDependentGMM(mu, sigma, weight)

        # Integer n_samples, no time
        samples = gmm.sample(100)
        assert samples.shape == (100, 4, 2), f"Expected (100, 4, 2), got {samples.shape}"

        # Scalar time (float)
        samples = gmm.sample(100, t=0.5)
        assert samples.shape == (100, 4, 2), f"Expected (100, 4, 2), got {samples.shape}"

        # Scalar time (tensor)
        samples = gmm.sample(100, t=torch.tensor(0.5))
        assert samples.shape == (100, 4, 2), f"Expected (100, 4, 2), got {samples.shape}"

        # Batch time [BS]
        samples = gmm.sample(100, t=torch.rand(100))
        assert samples.shape == (100, 4, 2), f"Expected (100, 4, 2), got {samples.shape}"

        # None time (default t=0)
        samples = gmm.sample(100, t=None)
        assert samples.shape == (100, 4, 2), f"Expected (100, 4, 2), got {samples.shape}"

    def test_time_processing(self):
        """Test time processing for all valid formats"""
        mu = torch.randn(4, 3, 2)
        sigma = torch.ones(4, 3, 2) * 0.5
        weight = torch.ones(4, 3)
        gmm = TimeDependentGMM(mu, sigma, weight)

        t = gmm._process_time(0.5, (100, 4))
        assert t.shape == (100, 4), f"Expected (100, 4), got {t.shape}"
        assert torch.allclose(t, torch.full((100, 4), 0.5))

        t = gmm._process_time(torch.tensor(0.5), (100, 4))
        assert t.shape == (100, 4), f"Expected (100, 4), got {t.shape}"
        assert torch.allclose(t, torch.full((100, 4), 0.5))

        t_input = torch.rand(100)
        t = gmm._process_time(t_input, (100, 4))
        assert t.shape == (100, 4), f"Expected (100, 4), got {t.shape}"
        assert torch.allclose(einops.repeat(t_input, "n -> n b", b=4), t)

        t_input = torch.rand(100, 4)
        t = gmm._process_time(t_input, (100, 4))
        assert t.shape == (100, 4), f"Expected (100, 4), got {t.shape}"
        assert torch.allclose(t_input, t)

    @pytest.mark.parametrize("x_shape", [(50, 2), (50, 4, 2)])
    def test_logprob_shapes(self, x_shape):
        """Test __call__() (log_prob) output shapes"""
        mu = torch.randn(4, 3, 2)
        sigma = torch.ones(4, 3, 2) * 0.5
        weight = torch.ones(4, 3)
        gmm = TimeDependentGMM(mu, sigma, weight)

        # [BS, D] input -> [BS] output
        x = torch.randn(50, 2)

        # No time
        log_p = gmm(x)
        assert log_p.shape == (50, 4), f"Expected (4, 50), got {log_p.shape}"

        # Scalar time
        log_p = gmm(x, t=0.5)
        assert log_p.shape == (50, 4), f"Expected (4, 50), got {log_p.shape}"

        # Batch time
        log_p = gmm(x, t=torch.rand(50))
        assert log_p.shape == (50, 4), f"Expected (4, 50), got {log_p.shape}"

    def test_energy_shapes(self):
        """Test energy() output shapes"""
        mu = torch.randn(4, 3, 2)
        sigma = torch.ones(4, 3, 2) * 0.5
        weight = torch.ones(4, 3)
        gmm = TimeDependentGMM(mu, sigma, weight)

        x = torch.randn(50, 4, 2)
        energy = gmm.energy(x, t=0.5)
        assert energy.shape == (50, 4), f"Expected (4, 50), got {energy.shape}"

    @pytest.mark.parametrize(
        "x_t_shape", [((2,), (1,), (1, 4, 2)), ((50, 2), (50,), (50, 4, 2)), ((50, 4, 2), (50, 4), (50, 4, 2))]
    )
    def test_score_shapes(self, x_t_shape):
        """Test score() output shapes (gradient)"""
        x_shape, t_shape, score_shape = x_t_shape
        mu = torch.randn(4, 3, 2)
        sigma = torch.ones(4, 3, 2) * 0.5
        weight = torch.ones(4, 3)
        gmm = TimeDependentGMM(mu, sigma, weight)

        # [BS, D] input -> [BS, D] gradient output
        x = torch.randn(x_shape)
        t = torch.randn(t_shape).clamp(min=0.001, max=0.999)
        score = gmm.score(x, t=t)
        assert score.shape == score_shape, f"Expected {score_shape}, got {score.shape}"


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

        gmm_samples = gmm.sample(1_000_000, t=t)
        conditional_samples = conditional.sample(500_000, t=t)
        # Compute moments are compare
        ground_truth_moments = [true_mean, true_std]
        gmm_moments = [gmm_samples.mean(dim=0), gmm_samples.std(dim=0)]  # [N, B, D] -[mean,std](0)-> [B, D]
        conditional_moments = [conditional_samples.mean(dim=0), conditional_samples.std(dim=0)]
        torch.testing.assert_close(ground_truth_moments, gmm_moments, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(gmm_moments, conditional_moments, atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize("t", [0.0, 0.5, 0.9, 1.0, torch.rand(50)])
    def test_conditional_vs_gmm_score(self, t):
        """Test distribution properties for different times
        We initialize a conditional process with only mu=x0 and a GMM with mu, sigma, and weight imitating a conditional model.
        Then we compare the statistics of the sampled forward process at different times t.
        """
        mu = torch.randn(4, 1, 2)
        sigma = torch.zeros(4, 1, 2) + 1e-10
        weight = torch.ones(4, 1)
        gmm = TimeDependentGMM(mu, sigma, weight)  # GMM with mu and superfluous sigma, weight
        conditional = TimeDependentGMM(mu)  # Conditional GMM with only mu=x0

        # Compute the true mean and variance of the forward process [batch=4,component=1,dim=2) -> x0=[batch=4,dim=2]
        if not isinstance(t, torch.Tensor):
            t = torch.scalar_tensor(t)
            alpha_t, sigma_t = gmm.schedule.get_alpha_t_sigma_t(t)
            true_mean = alpha_t * mu.squeeze(1)
            true_std = (sigma_t**2 + alpha_t * sigma.squeeze(1) ** 2) ** 0.5 * torch.ones_like(mu).squeeze(1)
        else:
            alpha_t, sigma_t = gmm.schedule.get_alpha_t_sigma_t(t)
            # true_std = (sigma_t ** 2 + alpha_t * sigma.squeeze(1) ** 2)**0.5 * torch.ones_like(mu).squeeze(1)
            decreasing_var_t = einops.einsum(alpha_t, sigma, "n, b k d -> n b k d") ** 2
            increasing_var_t = einops.repeat(
                sigma_t**2, "n -> n b k d", b=gmm.batch_size, k=gmm.num_components, d=gmm.dim
            )
            var_t = increasing_var_t + decreasing_var_t
            true_std = (var_t**0.5).squeeze(-2)
            true_mean = einops.einsum(alpha_t, mu, "n, b k d -> n b k d").squeeze(-2)

        # Create Conditional GMM and GMM with the same parameters and compute score on them
        gmm_samples = gmm.sample(50, t=t)
        true_score = -(gmm_samples - true_mean) / true_std**2
        gmm_score = gmm.score(gmm_samples, t=t)
        conditional_score = conditional.score(gmm_samples, t=t)
        torch.testing.assert_close(true_score, gmm_score, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(gmm_score, conditional_score, atol=1e-2, rtol=1e-2)


class TestTimeProcessing:
    """Test time processing for all valid formats"""

    def test_time_processing_formats(self):
        """Test that _process_time handles all valid time formats"""
        mu = torch.randn(1, 2, 2)
        sigma = torch.ones(1, 2, 2) * 0.5
        weight = torch.ones(1, 2)
        gmm = TimeDependentGMM(mu, sigma, weight)

        batch_size = (50, 1)  # Fifty evaluations for one data point: [N, BS]

        # None -> zeros
        t = gmm._process_time(None, batch_size)
        assert t.shape == batch_size
        assert torch.allclose(t, torch.zeros(batch_size))

        # Float -> broadcast
        t = gmm._process_time(0.5, batch_size)
        assert t.shape == batch_size
        assert torch.allclose(t, torch.full(batch_size, 0.5))

        # Scalar tensor -> broadcast
        t = gmm._process_time(torch.tensor(0.7), batch_size)
        assert t.shape == batch_size
        assert torch.allclose(t, torch.full(batch_size, 0.7))

        # [BS] tensor -> direct use
        t_input = torch.rand(batch_size)
        t = gmm._process_time(t_input, batch_size)
        assert t.shape == batch_size
        assert torch.allclose(t, t_input)


class TestGMMProperties:
    """Test mathematical properties of the GMM"""

    def test_log_prob_vs_energy(self):
        """Test that energy = -log_prob"""
        mu = torch.randn(1, 3, 2)
        sigma = torch.ones(1, 3, 2) * 0.5
        weight = torch.ones(1, 3)
        gmm = TimeDependentGMM(mu, sigma, weight)

        x = torch.randn(50, 2)
        log_p = gmm(x, t=0.5)
        energy = gmm.energy(x, t=0.5)

        torch.testing.assert_close(energy, -log_p)

    def test_score_is_gradient(self):
        """
        Test that score equals gradient of log_prob
        x has to be a 3D tensor [BS, N, d], as we're internally broadcasting over all batched GMM's [BS, d] -> [N, BS, d]
        """
        mu = torch.randn(4, 3, 2)
        sigma = torch.ones(4, 3, 2) * 0.5
        weight = torch.ones(4, 3)
        gmm = TimeDependentGMM(mu, sigma, weight)

        x = torch.randn(10, 4, 2)

        # Get score from gmm.score()
        score = gmm.score(x, t=0.5)

        # Compute gradient manually
        x_copy = x.clone().detach().requires_grad_(True)
        log_p = gmm(x_copy, t=0.5)
        grad = torch.autograd.grad(log_p.sum(), x_copy)[0]
        assert score.shape == grad.shape, f"Expected {score.shape=}, got {grad.shape=}"
        torch.testing.assert_close(score, grad, atol=1e-5, rtol=1e-4)


class TestMarginalDistributions:
    """Test marginal distribution extraction from TimeDependentGMM"""

    def test_marginal_2d_empirical_comparison(self):
        """Compare empirical histograms with analytical marginal distributions"""
        # Create 3-component 2D GMM
        mu = torch.tensor([[2.0, -2.0], [-2.0, 3.0], [0.0, 0.0]]).unsqueeze(0)  # [1, 3, 2]
        sigma = torch.tensor([[0.5, 0.4], [0.5, 0.5], [0.3, 0.4]]).unsqueeze(0)  # [1, 3, 2]
        weight = torch.tensor([0.3, 0.4, 0.3]).unsqueeze(0)  # [3]
        gmm_2d = TimeDependentGMM(mu, sigma, weight)

        # Sample from the GMM
        n_samples = 100_000
        samples = gmm_2d.sample(n_samples, t=0.0)  # [n_samples, 2]

        # Set up histogram parameters
        x_min, x_max = -5.0, 5.0
        n_bins = 100
        bin_edges = torch.linspace(x_min, x_max, n_bins + 1)  # [n_bins+1]
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])  # [n_bins]
        bin_width = (x_max - x_min) / n_bins

        # Test dimension 0
        empirical_0 = samples[:, :, 0]  # [n_samples, BS]
        hist_empirical_0, _ = torch.histogram(empirical_0, bins=bin_edges, density=True)  # [n_bins]

        marginal_0 = gmm_2d.marginal_gmm(dim=0)
        bin_centers_reshaped = bin_centers.unsqueeze(-1)  # [n_bins, 1]
        log_probs_0 = marginal_0.log_prob(bin_centers_reshaped)  # [n_bins, BS]
        analytical_probs_0 = torch.exp(log_probs_0)  # [n_bins, BS]

        # Compare pointwise
        torch.testing.assert_close(hist_empirical_0, analytical_probs_0[:, 0], atol=0.01, rtol=0.1)

        # Test dimension 1
        empirical_1 = samples[:, :, 1]  # [n_samples, BS]
        hist_empirical_1, _ = torch.histogram(empirical_1, bins=bin_edges, density=True)  # [n_bins]

        marginal_1 = gmm_2d.marginal_gmm(dim=1)
        log_probs_1 = marginal_1.log_prob(bin_centers_reshaped)  # [n_bins, BS]
        analytical_probs_1 = torch.exp(log_probs_1)  # [n_bins, BS]

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
            mesh_points = x_grid.unsqueeze(-1)
            log_prob = gmm(mesh_points, t=0.0)
            tempered_log_prob = temperature * log_prob
            temperature_log_prob = temperature_gmm(mesh_points, t=0.0)
            # torch.testing.assert_close(temperature * log_prob, temperature_log_prob, atol=0.01, rtol=0.1)
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
            mesh_points = x_grid.unsqueeze(-1)  # [BS]->[BS, 1]
            log_prob = gmm(mesh_points, t=0.0)
            tempered_log_prob = temperature * log_prob
            temperature_log_prob = temperature_gmm(mesh_points, t=0.0)
            # torch.testing.assert_close(temperature * log_prob, temperature_log_prob, atol=0.01, rtol=0.1)
            prob = torch.exp(log_prob) / torch.exp(log_prob).sum()
            tempered_prob = torch.exp(tempered_log_prob) / torch.exp(tempered_log_prob).sum()
            temperature_prob = torch.exp(temperature_log_prob) / torch.exp(temperature_log_prob).sum()
            torch.testing.assert_close(
                temperature_prob, tempered_prob, atol=0.01, rtol=0.1, msg=f"Temperature: {temperature}"
            )


# class TestConditionalGMM:

#     @pytest.mark.parametrize("dim", [1, 3])
#     def test_conditional_log_prob(self, dim):
#         x0 = torch.randn(10, dim)
#         cond_gmm = TimeDependentGMM(x0)
#         x = torch.randn(11, dim)

#         log_prob = cond_gmm(x, t=0.0)
#         assert log_prob.shape == (10, 11)

#     @pytest.mark.parametrize("dim", [1, 3])
#     def test_conditional_score(self, dim):
#         x0 = torch.randn(10, dim)
#         cond_gmm = TimeDependentGMM(x0)
#         x = torch.randn(11, dim)

#         score = cond_gmm.score(x, t=0.0)
#         assert score.shape == (10, 11, dim)

#     @pytest.mark.parametrize('t', [0.0, 0.1, 0.8, 1.0])
#     @pytest.mark.parametrize("dim", [1, 3])
#     def test_conditional_energy(self, dim, t):
#         x0 = torch.randn(10,dim)
#         cond_gmm = TimeDependentGMM(x0)
#         x = torch.randn(11, dim)
#         energy = cond_gmm.energy(x, t=t)

#     @pytest.mark.parametrize('t', [0.0, 0.1, 0.8, 1.0])
#     @pytest.mark.parametrize("dim", [1, 3])
#     def test_conditional_energy(self, dim, t):
#         x0 = torch.randn(10,dim)
#         cond_gmm = TimeDependentGMM(x0)
#         x = torch.randn(11, dim)
#         energy = cond_gmm.energy(x, t=t)
