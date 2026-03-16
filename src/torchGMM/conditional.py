import torch

from torchGMM.gmm import TimeDependentGMM
from torchGMM.schedule import BetaSchedule, LinearSchedule, Schedule


class Conditional(TimeDependentGMM):
    """
    Conditional Process class
    Instead of simulating the full GMM, we only simulate the conditional process conditioned on the initial value x0.
    This is useful for conditional sampling and inference.

    Args:
            x0: [..., d] - initial value
            schedule: Schedule - schedule for the conditional process

    Returns:
            Conditional - Conditional process
    """

    def __init__(self, x0: torch.Tensor, schedule: Schedule = None):
        assert torch.isfinite(x0).all(), f"x0 must contain only finite values, got {x0}"
        mu = x0.unsqueeze(-2)  # [..., d] -> [..., 1, d]
        assert mu.dim() == x0.dim() + 1, f"mu must be a tensor [..., 1, d], got {mu.shape}"
        sigma = torch.zeros_like(mu) + 1e-10
        weight = mu.new_ones((*mu.shape[:-2], 1))
        super().__init__(mu, sigma, weight, schedule)
