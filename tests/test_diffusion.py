import sys

import pytest
import torch
import einops

import matplotlib.pyplot as plt

torch.set_printoptions(sci_mode=False)
from torchGMM.diffusion import (
    forward_diffusion,
    reverse_diffusion,
)
from torchGMM.gmm import TimeDependentGMM
from torchGMM.schedule import BetaSchedule


@pytest.fixture
def gmm_model():
    """Fixture for the GMM model used in diffusion tests."""
    mu = torch.tensor([-2, 0, 2]).reshape(1, 3, 1)
    sigma = torch.tensor([0.3, 0.3, 0.2]).reshape(1, 3, 1)
    weight = torch.tensor([0.33, 0.5, 0.1]).reshape(1, 3)
    return TimeDependentGMM(mu=mu, sigma=sigma, weight=weight)


class TestDiffusion:

    def test_forward_diffusion(self, gmm_model):
        schedule = BetaSchedule()

        # Sample from the GMM at t=0
        x = gmm_model.sample(50_000, t=0.0)[:, 0]  # [*N, *BS=1, D=1]
        t = torch.linspace(0.00, 1.0, 100)

        # Run forward diffusion
        trajectory = forward_diffusion(schedule, x, t)

        # Check trajectory shape: [n_steps+1, n_samples, dim]
        assert trajectory.shape == (len(t), x.shape[0], 1)

        # Check that trajectory starts at x
        assert torch.allclose(trajectory[0], x, atol=1e-5)

        # Create a set of bin_edges and the corresponding bin centers
        x_grid = torch.linspace(-5, 5, 51).reshape(-1, 1)
        dx = x_grid[1] - x_grid[0]
        bin_edges = torch.cat(
            [
                x_grid[0] - dx / 2,
                x_grid.squeeze() + dx / 2,
            ]
        ).flatten()

        # Bin edges: center the bins around x_grid points
        for t_idx in range(t.numel())[::3]:
            t_ = t[t_idx]
            target_dist = gmm_model.log_prob(x_grid, t=t_).exp()
            # Compute histogram of the trajectory at t_idx

            # Compare the histogram to the target distribution using bin centers
            hist, _ = torch.histogram(trajectory[t_idx, :, 0], bins=bin_edges, density=True)
            # target_dist is shape [100, 1] for x_grid of shape [100, 1]
            # Assert the empirical distribution matches the target within some tolerance

            # plt.plot(hist, label=f"t={t_}")
            # plt.plot(target_dist, label=f"t={t_}")
            # plt.legend()
            # plt.show()
            assert target_dist.shape == hist.shape, f"target_dist.shape: {target_dist.shape}, hist.shape: {hist.shape}"

            assert (
                hist - target_dist
            ).abs().max() < 0.05, f"hist-target_dist.abs().max() @ t={t_}: {hist-target_dist.abs().max()}"

    def test_reverse_diffusion(self, gmm_model):
        schedule = BetaSchedule()

        # Start from random noise
        x = torch.randn(50_000, 1).reshape(-1, 1)
        t = torch.linspace(1.0, 0.00, 300)

        # Run reverse diffusion using the GMM score function
        trajectory = reverse_diffusion(schedule, lambda x, t: gmm_model.score(x, t), x, t)

        # Check trajectory shape: [n_steps, n_samples, dim]
        assert trajectory.shape == (len(t), x.shape[0], 1)

        # Check that trajectory starts at x
        assert torch.allclose(trajectory[0], x, atol=1e-5)

        # Create a set of bin_edges and the corresponding bin centers
        x_grid = torch.linspace(-5, 5, 51).reshape(-1, 1)
        dx = x_grid[1] - x_grid[0]
        bin_edges = torch.cat(
            [
                x_grid[0] - dx / 2,
                x_grid.squeeze() + dx / 2,
            ]
        ).flatten()

        # Bin edges: center the bins around x_grid points
        for t_idx in range(t.numel())[::3]:
            t_ = t[t_idx]
            target_dist = gmm_model.log_prob(x_grid, t=t_).exp()
            # Compute histogram of the trajectory at t_idx

            # Compare the histogram to the target distribution using bin centers
            hist, _ = torch.histogram(trajectory[t_idx, :, 0], bins=bin_edges, density=True)
            # target_dist is shape [100, 1] for x_grid of shape [100, 1]
            # Assert the empirical distribution matches the target within some tolerance

            # plt.plot(hist, label=f"t={t_}")
            # plt.plot(target_dist, label=f"t={t_}")
            # plt.legend()
            # plt.show()
            assert target_dist.shape == hist.shape, f"target_dist.shape: {target_dist.shape}, hist.shape: {hist.shape}"

            assert (
                hist - target_dist
            ).abs().max() < 0.05, f"hist-target_dist.abs().max() @ t={t_}: {hist-target_dist.abs().max()}"
