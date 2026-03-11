import sys
from typing import Tuple

import torch


class Schedule(torch.nn.Module):
    """Base class for interpolation schedules x_t = α_t x₀ + σ_t ε.

    Any schedule must provide (α_t, σ_t) and their time derivatives (α̇_t, σ̇_t).
    Boundary conditions: α₀ = 1, σ₀ = 0 (data) and α₁ ≈ 0, σ₁ ≈ 1 (noise).
    """

    def get_alpha_t(self, t: torch.Tensor) -> torch.Tensor:
        """Signal coefficient α_t."""
        raise NotImplementedError

    def get_sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        """Noise coefficient σ_t."""
        raise NotImplementedError

    def get_alpha_t_sigma_t(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Signal and noise coefficients (α_t, σ_t)."""
        return self.get_alpha_t(t), self.get_sigma_t(t)

    def get_dalpha_dt(self, t: torch.Tensor) -> torch.Tensor:
        """Time derivative dα_t/dt — needed for velocity computation."""
        raise NotImplementedError

    def get_dsigma_dt(self, t: torch.Tensor) -> torch.Tensor:
        """Time derivative dσ_t/dt — needed for velocity computation."""
        raise NotImplementedError


class BetaSchedule(Schedule):
    """
    VP-SDE schedule derived from the forward SDE:
    dX_t = -1/2 * β(t) * X_t dt + √β(t) dW_t

    With β(t) = β_min + t(β_max - β_min)

    Satisfies the variance-preserving constraint: α_t² + σ_t² = 1.
    """

    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0):
        super().__init__()
        self.beta_min = beta_min
        self.beta_max = beta_max

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """β(t) = β_min + t(β_max - β_min)"""
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def integrated_beta(self, t: torch.Tensor) -> torch.Tensor:
        """∫₀ᵗ β(s) ds = β_min * t + (β_max - β_min) * t²/2"""
        return self.beta_min * t + (self.beta_max - self.beta_min) * t**2 / 2

    def get_alpha_t(self, t: torch.Tensor) -> torch.Tensor:
        """Signal coefficient: α_t = exp(-1/2 * ∫₀ᵗ β(s) ds)"""
        int_beta = self.integrated_beta(t)
        return torch.exp(-0.5 * int_beta)

    def get_sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        """Noise coefficient: σ_t = √(1 - exp(-∫₀ᵗ β(s) ds))"""
        int_beta = self.integrated_beta(t)
        return torch.sqrt(1 - torch.exp(-int_beta))

    def get_dalpha_dt(self, t: torch.Tensor) -> torch.Tensor:
        """dα_t/dt = -1/2 * β(t) * α_t"""
        return -0.5 * self.beta(t) * self.get_alpha_t(t)

    def get_dsigma_dt(self, t: torch.Tensor) -> torch.Tensor:
        """dσ_t/dt = 1/2 * β(t) * α_t² / σ_t

        Derived from σ_t² = 1 - α_t², so 2σ_t σ̇_t = -2α_t α̇_t = β(t) α_t².
        """
        alpha_t = self.get_alpha_t(t)
        sigma_t = self.get_sigma_t(t)
        return 0.5 * self.beta(t) * alpha_t**2 / sigma_t

    def get_alpha_t_sigma_t(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Signal and noise coefficients: (α_t, σ_t)"""
        return self.get_alpha_t(t), self.get_sigma_t(t)

    def get_t_from_lambda(self, lambda_t: torch.Tensor) -> torch.Tensor:
        """Used by DPMsolver. The formula comes from Section D.4 of the DPMsolver paper."""
        log_exp = 2 * torch.log(1 + torch.exp(-2 * lambda_t))
        sqrt_denom = torch.sqrt(self.beta_min**2 + (self.beta_max - self.beta_min) * log_exp)
        t_lambda = log_exp / (sqrt_denom + self.beta_min)
        return t_lambda


class FlowMatchingSchedule(Schedule):
    """Linear interpolation (conditional OT) schedule: α_t = 1 − t, σ_t = t.

    This is the schedule used by flow matching / rectified flow. The interpolation
    path x_t = (1 − t) x₀ + t ε is a straight line from data to noise.

    Satisfies α_t + σ_t = 1 (not variance-preserving).
    """

    def get_alpha_t(self, t: torch.Tensor) -> torch.Tensor:
        """Signal coefficient: α_t = 1 − t"""
        return 1 - t

    def get_sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        """Noise coefficient: σ_t = t"""
        return t

    def get_dalpha_dt(self, t: torch.Tensor) -> torch.Tensor:
        """dα_t/dt = −1"""
        return torch.full_like(t, -1.0)

    def get_dsigma_dt(self, t: torch.Tensor) -> torch.Tensor:
        """dσ_t/dt = 1"""
        return torch.ones_like(t)
