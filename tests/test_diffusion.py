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


def get_local_device():
    """Fixture that returns the best available device (MPS, CUDA, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


@pytest.fixture
def gmm_model():
    """Fixture for the GMM model used in diffusion tests."""
    mu = torch.tensor([-2, 0, 2]).reshape(1, 3, 1)
    sigma = torch.tensor([0.3, 0.3, 0.2]).reshape(1, 3, 1)
    weight = torch.tensor([0.33, 0.5, 0.1]).reshape(1, 3)
    return TimeDependentGMM(mu=mu, sigma=sigma, weight=weight)


@pytest.fixture
def batched_gmm_model():
    """Fixture for the GMM model used in diffusion tests."""
    mu = torch.tensor([[-1, 0.5, 1.5], [-2, 0, 2]]).reshape(2, 3, 1)
    sigma = torch.tensor([[0.3, 0.3, 0.2], [0.3, 0.3, 0.2]]).reshape(2, 3, 1)
    weight = torch.tensor([[0.2, 0.3, 0.1], [0.33, 0.5, 0.1]]).reshape(2, 3)
    return TimeDependentGMM(mu=mu, sigma=sigma, weight=weight)


class TestDiffusion:

    def test_forward_diffusion(self, gmm_model):
        schedule = BetaSchedule()

        # Sample from the GMM at t=0; keep [N, B=1, D] for diffusion (diffusion uses x [N, D] in loop)
        samples = gmm_model.sample(shape=10_000, t=0.0)  # [N, B=1, D=1]
        x = samples.squeeze(1)  # [N] for forward_diffusion which expects [N, D]; D=1 -> [N, 1]
        t = torch.linspace(0.00, 1.0, 100)

        # Run forward diffusion
        trajectory = forward_diffusion(schedule, x, t)

        # Check trajectory shape: [n_steps+1, n_samples, dim]
        assert trajectory.shape == (len(t), x.shape[0], 1)

        # Check that trajectory starts at x
        assert torch.allclose(trajectory[0], x, atol=1e-5)

        x_grid = torch.linspace(-5, 5, 51).reshape(-1, 1, 1)  # [N, B=1, D=1]
        dx = x_grid[1, 0, 0] - x_grid[0, 0, 0]
        x_flat = x_grid.squeeze()
        bin_edges = torch.cat([(x_flat[0] - dx / 2).unsqueeze(0), x_flat + dx / 2])

        for t_idx in range(t.numel())[::3]:
            t_ = t[t_idx]
            target_dist = gmm_model.log_prob(x_grid, t=t_).exp().squeeze(-1)  # [51]
            hist, _ = torch.histogram(trajectory[t_idx, :, 0], bins=bin_edges, density=True)
            assert target_dist.shape == hist.shape, f"target_dist.shape: {target_dist.shape}, hist.shape: {hist.shape}"
            assert (
                hist - target_dist
            ).abs().max() < 0.05, f"hist-target_dist.abs().max() @ t={t_}: {(hist-target_dist).abs().max()}"

    def test_reverse_diffusion(self, batched_gmm_model):
        schedule = BetaSchedule()

        x = torch.randn(50_000, 2, 1)  # [N, B=1, D=1] for gmm.score
        t = torch.linspace(1.0, 0.00, 300)
        trajectory = reverse_diffusion(
            schedule, score_fn=lambda x_xt, t_xt: batched_gmm_model.score(x_xt, t_xt), x=x, t=t
        )

        assert trajectory.shape == (len(t), x.shape[0], 2, 1)

        assert torch.allclose(trajectory[0], x, atol=1e-5)

        x_grid = torch.linspace(-5, 5, 51).reshape(-1, 1, 1).repeat(1, 2, 1)  # [N, B=2, D=1]
        dx = x_grid[1, 0, 0] - x_grid[0, 0, 0]
        x_flat = x_grid[:, 0, 0].squeeze()
        bin_edges = torch.cat([(x_flat[0] - dx / 2).unsqueeze(0), x_flat + dx / 2])

        for t_idx in range(t.numel())[::3]:
            t_ = t[t_idx]
            target_dist = batched_gmm_model.log_prob(x_grid, t=t_).exp().squeeze(-1)
            for dim in range(batched_gmm_model.dim):
                trajectory_dim = trajectory[t_idx, :, dim, 0].unsqueeze(-2)
                hist, _ = torch.histogram(trajectory_dim, bins=bin_edges, density=True)
                assert target_dist.shape[:1] == hist.shape
                assert (hist - target_dist[:, dim]).abs().max() < 0.05


class TestDiffusionDeviceHandling:
    """Test device handling for diffusion functions"""

    def test_forward_diffusion_cpu(self, gmm_model):
        """Test forward diffusion works on CPU"""
        schedule = BetaSchedule()
        x = gmm_model.sample(shape=100, t=0.0)[:, 0, :]  # [N, D] for forward_diffusion
        t = torch.linspace(0.0, 1.0, 50)
        trajectory = forward_diffusion(schedule, x, t)
        assert trajectory.device.type == "cpu"
        assert trajectory.shape == (len(t), 100, 1)

    def test_forward_diffusion_on_accelerator(self, gmm_model):
        device = get_local_device()
        schedule = BetaSchedule()
        gmm = gmm_model.to(device)
        x = gmm.sample(shape=100, t=0.0)[:, 0, :]
        t = torch.linspace(0.0, 1.0, 50, device=device)
        trajectory = forward_diffusion(schedule.to(device), x, t)
        assert trajectory.device == x.device
        assert trajectory.shape == (len(t), 100, 1)

    def test_reverse_diffusion_cpu(self, gmm_model):
        """Test reverse diffusion works on CPU"""
        schedule = BetaSchedule()
        x = torch.randn(100, 1, 1)  # [N, B=1, D=1]
        t = torch.linspace(1.0, 0.0, 50)
        trajectory = reverse_diffusion(schedule, lambda x_xt, t_xt: gmm_model.score(x_xt, t_xt), x, t)
        assert trajectory.device.type == "cpu"
        assert trajectory.shape == (len(t), 100, 1, 1)

    def test_reverse_diffusion_on_accelerator(self, gmm_model):
        device = get_local_device()
        schedule = BetaSchedule()
        gmm = gmm_model.to(device)
        x = torch.randn(100, 1, 1, device=device)
        t = torch.linspace(1.0, 0.0, 50, device=device)
        trajectory = reverse_diffusion(schedule.to(device), lambda x_xt, t_xt: gmm.score(x_xt, t_xt), x, t)
        assert trajectory.device == x.device
        assert trajectory.shape == (len(t), 100, 1, 1)
