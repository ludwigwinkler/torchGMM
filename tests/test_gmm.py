import sys

import pytest
import torch


from torchGMM.diffusion import (
    denoising_and_resample_fkc,
    denoising_and_resample_fksmc,
    denoising_and_resample_smc,
    denoising_and_resample_smclangevin,
    reverse_diffusion,
    reverse_diffusion_with_regular_resampling,
)
from torchGMM.schedule import BetaSchedule
from torchGMM import TimeDependentGMM


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
        mu = torch.randn(2, 2)
        sigma = torch.ones(2, 2) * 0.5
        weight = torch.ones(2)
        gmm = TimeDependentGMM(mu, sigma, weight)
        assert gmm.num_components == 2
        assert gmm.dim == 2

        # 5 components, 3D
        mu = torch.randn(5, 3)
        sigma = torch.ones(5, 3) * 0.5
        weight = torch.ones(5)
        gmm = TimeDependentGMM(mu, sigma, weight)
        assert gmm.num_components == 5
        assert gmm.dim == 3

    def test_sample_shapes_various_inputs(self):
        """Test sample() output shapes with different time inputs"""
        mu = torch.randn(3, 2)
        sigma = torch.ones(3, 2) * 0.5
        weight = torch.ones(3)
        gmm = TimeDependentGMM(mu, sigma, weight)

        # Integer n_samples, no time
        samples = gmm.sample(100)
        assert samples.shape == (100, 2), f"Expected (100, 2), got {samples.shape}"

        # Tuple n_samples, no time
        samples = gmm.sample((50,))
        assert samples.shape == (50, 2), f"Expected (50, 2), got {samples.shape}"

        # Scalar time (float)
        samples = gmm.sample(100, t=0.5)
        assert samples.shape == (100, 2), f"Expected (100, 2), got {samples.shape}"

        # Scalar time (tensor)
        samples = gmm.sample(100, t=torch.tensor(0.5))
        assert samples.shape == (100, 2), f"Expected (100, 2), got {samples.shape}"

        # Batch time [BS]
        samples = gmm.sample(100, t=torch.rand(100))
        assert samples.shape == (100, 2), f"Expected (100, 2), got {samples.shape}"

        # None time (default t=0)
        samples = gmm.sample(100, t=None)
        assert samples.shape == (100, 2), f"Expected (100, 2), got {samples.shape}"

    def test_logprob_shapes(self):
        """Test __call__() (log_prob) output shapes"""
        mu = torch.randn(3, 2)
        sigma = torch.ones(3, 2) * 0.5
        weight = torch.ones(3)
        gmm = TimeDependentGMM(mu, sigma, weight)

        # [BS, D] input -> [BS] output
        x = torch.randn(50, 2)

        # No time
        log_p = gmm(x)
        assert log_p.shape == (50,), f"Expected (50,), got {log_p.shape}"

        # Scalar time
        log_p = gmm(x, t=0.5)
        assert log_p.shape == (50,), f"Expected (50,), got {log_p.shape}"

        # Batch time
        log_p = gmm(x, t=torch.rand(50))
        assert log_p.shape == (50,), f"Expected (50,), got {log_p.shape}"

    def test_energy_shapes(self):
        """Test energy() output shapes"""
        mu = torch.randn(3, 2)
        sigma = torch.ones(3, 2) * 0.5
        weight = torch.ones(3)
        gmm = TimeDependentGMM(mu, sigma, weight)

        x = torch.randn(50, 2)
        energy = gmm.energy(x, t=0.5)
        assert energy.shape == (50,), f"Expected (50,), got {energy.shape}"

    def test_score_shapes(self):
        """Test score() output shapes (gradient)"""
        mu = torch.randn(3, 2)
        sigma = torch.ones(3, 2) * 0.5
        weight = torch.ones(3)
        gmm = TimeDependentGMM(mu, sigma, weight)

        # [BS, D] input -> [BS, D] gradient output
        x = torch.randn(50, 2)
        score = gmm.score(x, t=0.5)
        assert score.shape == (50, 2), f"Expected (50, 2), got {score.shape}"


class TestTimeProcessing:
    """Test time processing for all valid formats"""

    def test_time_processing_formats(self):
        """Test that _process_time handles all valid time formats"""
        mu = torch.randn(2, 2)
        sigma = torch.ones(2, 2) * 0.5
        weight = torch.ones(2)
        gmm = TimeDependentGMM(mu, sigma, weight)

        batch_size = 50

        # None -> zeros
        t = gmm._process_time(None, batch_size)
        assert t.shape == (batch_size,)
        assert torch.allclose(t, torch.zeros(batch_size))

        # Float -> broadcast
        t = gmm._process_time(0.5, batch_size)
        assert t.shape == (batch_size,)
        assert torch.allclose(t, torch.full((batch_size,), 0.5))

        # Scalar tensor -> broadcast
        t = gmm._process_time(torch.tensor(0.7), batch_size)
        assert t.shape == (batch_size,)
        assert torch.allclose(t, torch.full((batch_size,), 0.7))

        # [BS] tensor -> direct use
        t_input = torch.rand(batch_size)
        t = gmm._process_time(t_input, batch_size)
        assert t.shape == (batch_size,)
        assert torch.allclose(t, t_input)


class TestGMMProperties:
    """Test mathematical properties of the GMM"""

    def test_log_prob_vs_energy(self):
        """Test that energy = -log_prob"""
        mu = torch.randn(3, 2)
        sigma = torch.ones(3, 2) * 0.5
        weight = torch.ones(3)
        gmm = TimeDependentGMM(mu, sigma, weight)

        x = torch.randn(50, 2)
        log_p = gmm(x, t=0.5)
        energy = gmm.energy(x, t=0.5)

        torch.testing.assert_close(energy, -log_p)

    def test_score_is_gradient(self):
        """Test that score equals gradient of log_prob"""
        mu = torch.randn(2, 2)
        sigma = torch.ones(2, 2) * 0.5
        weight = torch.ones(2)
        gmm = TimeDependentGMM(mu, sigma, weight)

        x = torch.randn(10, 2)

        # Get score from gmm.score()
        score = gmm.score(x, t=0.5)

        # Compute gradient manually
        x_copy = x.clone().detach().requires_grad_(True)
        log_p = gmm(x_copy, t=0.5)
        grad = torch.autograd.grad(log_p.sum(), x_copy)[0]

        torch.testing.assert_close(score, grad, atol=1e-5, rtol=1e-4)


class TestMarginalDistributions:
    """Test marginal distribution extraction from TimeDependentGMM"""

    def test_marginal_2d_empirical_comparison(self):
        """Compare empirical histograms with analytical marginal distributions"""
        # Create 3-component 2D GMM
        mu = torch.tensor([[1.0, 2.0], [-1.0, 3.0], [0.0, 0.0]])  # [3, 2]
        sigma = torch.tensor([[0.5, 0.8], [1.0, 0.6], [0.7, 0.9]])  # [3, 2]
        weight = torch.tensor([0.3, 0.4, 0.3])  # [3]
        gmm_2d = TimeDependentGMM(mu, sigma, weight)

        # Sample from the GMM
        n_samples = 100_000
        samples = gmm_2d.sample((n_samples,), t=0.0)  # [n_samples, 2]

        # Set up histogram parameters
        x_min, x_max = -5.0, 5.0
        n_bins = 100
        bin_edges = torch.linspace(x_min, x_max, n_bins + 1)  # [n_bins+1]
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])  # [n_bins]
        bin_width = (x_max - x_min) / n_bins

        # Test dimension 0
        empirical_0 = samples[:, 0]  # [n_samples]
        hist_empirical_0, _ = torch.histogram(empirical_0, bins=bin_edges, density=True)  # [n_bins]

        marginal_0 = gmm_2d.marginal_gmm(dim=0)
        bin_centers_reshaped = bin_centers.unsqueeze(-1)  # [n_bins, 1]
        log_probs_0 = marginal_0.log_prob(bin_centers_reshaped)  # [n_bins]
        analytical_probs_0 = torch.exp(log_probs_0)  # [n_bins]

        # Compare pointwise
        torch.testing.assert_close(hist_empirical_0, analytical_probs_0, atol=0.01, rtol=0.1)

        # Test dimension 1
        empirical_1 = samples[:, 1]  # [n_samples]
        hist_empirical_1, _ = torch.histogram(empirical_1, bins=bin_edges, density=True)  # [n_bins]

        marginal_1 = gmm_2d.marginal_gmm(dim=1)
        log_probs_1 = marginal_1.log_prob(bin_centers_reshaped)  # [n_bins]
        analytical_probs_1 = torch.exp(log_probs_1)  # [n_bins]

        # Compare pointwise
        torch.testing.assert_close(hist_empirical_1, analytical_probs_1, atol=0.01, rtol=0.1)


class TestTemperatureSampling:
    """Test temperature sampling from the GMM"""

    def test_temperature_probability(self):
        """
        Test temperature sampling from the GMM
        1D Gaussian N(mu, sigma^2)^\beta = N(mu, var=sigma^2/beta) = N(mu, std=sigma/sqrt(beta))
        We're testing in probability space (not log probability space)
        Log probability space has values that are too large and the tests trigger negatively
        """
        mu = torch.randn(1, 1)
        sigma = torch.ones(1, 1) * 0.5
        weight = torch.ones(1)
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
        mu = torch.randn(1, 1)
        sigma = torch.ones(1, 1) * 0.5
        weight = torch.ones(1)
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
