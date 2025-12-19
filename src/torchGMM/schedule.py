import sys
from typing import Tuple

import torch


class BetaSchedule(torch.nn.Module):
    """
    Beta schedule for the forward SDE:
    dX_t = -1/2 * β(t) * X_t dt + √β(t) dW_t

    With β(t) = β_min + t(β_max - β_min)
    
    """

    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0):
        super().__init__()
        self.beta_min = beta_min
        self.beta_max = beta_max

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """β(t) = β_min + t(β_max - β_min)"""
        # Move time tensor to device and compute linear beta schedule
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def integrated_beta(self, t: torch.Tensor) -> torch.Tensor:
        """∫₀ᵗ β(s) ds = β_min * t + (β_max - β_min) * t²/2"""
        # Compute integral of beta schedule for analytical solutions
        return self.beta_min * t + (self.beta_max - self.beta_min) * t**2 / 2

    def get_alpha_t(self, t: torch.Tensor) -> torch.Tensor:
        """Signal coefficient: exp(-1/2 * ∫₀ᵗ β(s) ds)"""
        # Compute signal decay coefficient for forward SDE
        int_beta = self.integrated_beta(t)
        return torch.exp(-0.5 * int_beta)

    def get_sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        """Noise coefficient: √(1 - exp(-∫₀ᵗ β(s) ds))"""
        # Compute noise scaling coefficient for forward SDE
        int_beta = self.integrated_beta(t)
        return torch.sqrt(1 - torch.exp(-int_beta))

    def get_alpha_t_sigma_t(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Signal and noise coefficients: exp(-1/2 * ∫₀ᵗ β(s) ds) and √(1 - exp(-∫₀ᵗ β(s) ds))"""
        # Compute signal and noise coefficients for forward SDE
        return self.get_alpha_t(t), self.get_sigma_t(t)

    def get_t_from_lambda(self, lambda_t: torch.Tensor) -> torch.Tensor:
        """Used by DPMsolver. The formula comes from Section D.4 of the DPMsolver paper."""
        log_exp = 2 * torch.log(1 + torch.exp(-2 * lambda_t))
        sqrt_denom = torch.sqrt(self.beta_min**2 + (self.beta_max - self.beta_min) * log_exp)
        t_lambda = log_exp / (sqrt_denom + self.beta_min)
        return t_lambda
