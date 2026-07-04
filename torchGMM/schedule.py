import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor


class Schedule(torch.nn.Module):
    """Base class for interpolation schedules x_t = α_t x₀ + σ_t ε.

    Any schedule must provide (α_t, σ_t) and their time derivatives (α̇_t, σ̇_t).
    Boundary conditions: α₀ = 1, σ₀ = 0 (data) and α₁ ≈ 0, σ₁ ≈ 1 (noise).

    All methods clamp t to [eps, 1-eps] to avoid division-by-zero singularities
    at the boundaries (e.g. 1/(1-t) at t=1, or 1/σ_t at t=0).
    """

    eps: float = 1e-2  # small offset to avoid boundary singularities

    def _clamp_t(self, t: Float[Tensor, "*batch"]) -> Float[Tensor, "*batch"]:
        return t.clamp(self.eps, 1.0 - self.eps)

    def get_alpha_t(self, t: Float[Tensor, "*batch"]) -> Float[Tensor, "*batch"]:
        """Signal coefficient α_t."""
        raise NotImplementedError

    def get_sigma_t(self, t: Float[Tensor, "*batch"]) -> Float[Tensor, "*batch"]:
        """Noise coefficient σ_t."""
        raise NotImplementedError

    def get_alpha_t_sigma_t(
        self, t: Float[Tensor, "*batch"]
    ) -> tuple[Float[Tensor, "*batch"], Float[Tensor, "*batch"]]:
        """Signal and noise coefficients (α_t, σ_t)."""
        return self.get_alpha_t(t), self.get_sigma_t(t)

    def get_dalpha_dt(self, t: Float[Tensor, "*batch"]) -> Float[Tensor, "*batch"]:
        """Time derivative dα_t/dt — needed for velocity computation."""
        raise NotImplementedError

    def get_dsigma_dt(self, t: Float[Tensor, "*batch"]) -> Float[Tensor, "*batch"]:
        """Time derivative dσ_t/dt — needed for velocity computation."""
        raise NotImplementedError

    @jaxtyped(typechecker=beartype)
    def forward_drift(
        self, x: Float[Tensor, "*batch D"], t: Float[Tensor, "*t"]
    ) -> Float[Tensor, "*batch D"]:
        """Forward SDE drift f(x,t) = (α̇_t / α_t) x. t broadcasts over x's batch dims."""
        t = self._clamp_t(t)
        return (self.get_dalpha_dt(t) / self.get_alpha_t(t)).unsqueeze(-1) * x

    @jaxtyped(typechecker=beartype)
    def diffusion_coeff(self, t: Float[Tensor, "*batch"]) -> Float[Tensor, "*batch"]:
        """Forward SDE diffusion g(t) where g²(t) = 2(σ̇_t σ_t − α̇_t σ_t² / α_t)."""
        t = self._clamp_t(t)
        alpha_t = self.get_alpha_t(t)
        sigma_t = self.get_sigma_t(t)
        dalpha_dt = self.get_dalpha_dt(t)
        dsigma_dt = self.get_dsigma_dt(t)
        g_sq = 2 * (dsigma_dt * sigma_t - dalpha_dt * sigma_t**2 / alpha_t)
        return torch.sqrt(g_sq)


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

    @jaxtyped(typechecker=beartype)
    def beta(self, t: Float[Tensor, "*batch"]) -> Float[Tensor, "*batch"]:
        """β(t) = β_min + t(β_max - β_min)"""
        return self.beta_min + t * (self.beta_max - self.beta_min)

    @jaxtyped(typechecker=beartype)
    def integrated_beta(self, t: Float[Tensor, "*batch"]) -> Float[Tensor, "*batch"]:
        """∫₀ᵗ β(s) ds = β_min * t + (β_max - β_min) * t²/2"""
        return self.beta_min * t + (self.beta_max - self.beta_min) * t**2 / 2

    @jaxtyped(typechecker=beartype)
    def get_alpha_t(self, t: Float[Tensor, "*batch"]) -> Float[Tensor, "*batch"]:
        """Signal coefficient: α_t = exp(-1/2 * ∫₀ᵗ β(s) ds)"""
        int_beta = self.integrated_beta(t)
        return torch.exp(-0.5 * int_beta)

    @jaxtyped(typechecker=beartype)
    def get_sigma_t(self, t: Float[Tensor, "*batch"]) -> Float[Tensor, "*batch"]:
        """Noise coefficient: σ_t = √(1 - exp(-∫₀ᵗ β(s) ds))"""
        int_beta = self.integrated_beta(t)
        return torch.sqrt(1 - torch.exp(-int_beta))

    @jaxtyped(typechecker=beartype)
    def get_dalpha_dt(self, t: Float[Tensor, "*batch"]) -> Float[Tensor, "*batch"]:
        """dα_t/dt = -1/2 * β(t) * α_t"""
        return -0.5 * self.beta(t) * self.get_alpha_t(t)

    @jaxtyped(typechecker=beartype)
    def get_dsigma_dt(self, t: Float[Tensor, "*batch"]) -> Float[Tensor, "*batch"]:
        """dσ_t/dt = 1/2 * β(t) * α_t² / σ_t

        Derived from σ_t² = 1 - α_t², so 2σ_t σ̇_t = -2α_t α̇_t = β(t) α_t².
        """
        t = self._clamp_t(t)
        alpha_t = self.get_alpha_t(t)
        sigma_t = self.get_sigma_t(t)
        return 0.5 * self.beta(t) * alpha_t**2 / sigma_t

    @jaxtyped(typechecker=beartype)
    def get_alpha_t_sigma_t(
        self, t: Float[Tensor, "*batch"]
    ) -> tuple[Float[Tensor, "*batch"], Float[Tensor, "*batch"]]:
        """Signal and noise coefficients: (α_t, σ_t)"""
        return self.get_alpha_t(t), self.get_sigma_t(t)

    @jaxtyped(typechecker=beartype)
    def forward_drift(
        self, x: Float[Tensor, "*batch D"], t: Float[Tensor, "*t"]
    ) -> Float[Tensor, "*batch D"]:
        """f(x,t) = -½ β(t) x. t broadcasts over x's batch dims."""
        return -0.5 * self.beta(t).unsqueeze(-1) * x

    @jaxtyped(typechecker=beartype)
    def diffusion_coeff(self, t: Float[Tensor, "*batch"]) -> Float[Tensor, "*batch"]:
        """g(t) = √β(t)"""
        return torch.sqrt(self.beta(t))


class LinearSchedule(Schedule):
    """Linear interpolation (conditional OT) schedule: α_t = 1 − t, σ_t = t.

    This is the schedule used by flow matching / rectified flow. The interpolation
    path x_t = (1 − t) x₀ + t ε is a straight line from data to noise.

    Satisfies α_t + σ_t = 1 (not variance-preserving).
    """

    @jaxtyped(typechecker=beartype)
    def get_alpha_t(self, t: Float[Tensor, "*batch"]) -> Float[Tensor, "*batch"]:
        """Signal coefficient: α_t = 1 − t"""
        return 1 - t

    @jaxtyped(typechecker=beartype)
    def get_sigma_t(self, t: Float[Tensor, "*batch"]) -> Float[Tensor, "*batch"]:
        """Noise coefficient: σ_t = t"""
        return t

    @jaxtyped(typechecker=beartype)
    def get_dalpha_dt(self, t: Float[Tensor, "*batch"]) -> Float[Tensor, "*batch"]:
        """dα_t/dt = −1"""
        return torch.full_like(t, -1.0)

    @jaxtyped(typechecker=beartype)
    def get_dsigma_dt(self, t: Float[Tensor, "*batch"]) -> Float[Tensor, "*batch"]:
        """dσ_t/dt = 1"""
        return torch.ones_like(t)

    @jaxtyped(typechecker=beartype)
    def forward_drift(
        self, x: Float[Tensor, "*batch D"], t: Float[Tensor, "*t"]
    ) -> Float[Tensor, "*batch D"]:
        """f(x,t) = -x / (1 − t). t broadcasts over x's batch dims."""
        t = self._clamp_t(t)
        return -x / (1 - t).unsqueeze(-1)

    @jaxtyped(typechecker=beartype)
    def diffusion_coeff(self, t: Float[Tensor, "*batch"]) -> Float[Tensor, "*batch"]:
        """g(t) = √(2t / (1 − t))"""
        t = self._clamp_t(t)
        return torch.sqrt(2 * t / (1 - t))


class VESchedule(Schedule):
    """Variance Exploding schedule (Song et al., SMLD / NCSN).

    Geometric noise scale: σ_t = σ_min · (σ_max / σ_min)^t, with α_t ≡ 1.

    Forward SDE has zero drift and diffusion g(t) = σ_t · √(2 ln(σ_max/σ_min)).
    Not variance-preserving: marginal variance grows from σ_min² to σ_max².
    """

    def __init__(self, sigma_min: float = 0.01, sigma_max: float = 50.0):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.log_ratio = torch.log(torch.tensor(sigma_max / sigma_min))

    @jaxtyped(typechecker=beartype)
    def get_alpha_t(self, t: Float[Tensor, "*batch"]) -> Float[Tensor, "*batch"]:
        """α_t = 1"""
        return torch.ones_like(t)

    @jaxtyped(typechecker=beartype)
    def get_sigma_t(self, t: Float[Tensor, "*batch"]) -> Float[Tensor, "*batch"]:
        """σ_t = σ_min · (σ_max / σ_min)^t"""
        return self.sigma_min * torch.exp(t * self.log_ratio)

    @jaxtyped(typechecker=beartype)
    def get_dalpha_dt(self, t: Float[Tensor, "*batch"]) -> Float[Tensor, "*batch"]:
        """dα_t/dt = 0"""
        return torch.zeros_like(t)

    @jaxtyped(typechecker=beartype)
    def get_dsigma_dt(self, t: Float[Tensor, "*batch"]) -> Float[Tensor, "*batch"]:
        """dσ_t/dt = σ_t · ln(σ_max/σ_min)"""
        return self.get_sigma_t(t) * self.log_ratio

    @jaxtyped(typechecker=beartype)
    def forward_drift(
        self, x: Float[Tensor, "*batch D"], t: Float[Tensor, "*t"]
    ) -> Float[Tensor, "*batch D"]:
        """f(x,t) = 0 — VE has no drift."""
        return torch.zeros_like(x)

    @jaxtyped(typechecker=beartype)
    def diffusion_coeff(self, t: Float[Tensor, "*batch"]) -> Float[Tensor, "*batch"]:
        """g(t) = σ_t · √(2 ln(σ_max/σ_min))"""
        return self.get_sigma_t(t) * torch.sqrt(2 * self.log_ratio)


class KarrasSchedule(Schedule):
    """Karras et al. (2022) / AlphaFold3-style VE schedule.

    Continuous form of the discrete EDM noise grid:
        σ(t) = (σ_min^{1/ρ} + t · (σ_max^{1/ρ} − σ_min^{1/ρ}))^ρ

    with α_t ≡ 1 and t ∈ [0, 1]. ρ controls step concentration: ρ=7 (the AF3 /
    EDM default) packs more low-noise steps near t=0, which is helpful for
    high-fidelity sampling. ρ=1 recovers a linear σ schedule.

    AF3 sets σ_data ≈ 16 Å, σ_max ≈ 160, σ_min ≈ 4e−4, ρ=7. σ_data is a
    preconditioning constant for the denoiser; here we expose it as a multiplier
    on the σ range so the schedule itself remains a pure noise schedule.
    """

    def __init__(
        self,
        sigma_min: float = 4e-4,
        sigma_max: float = 160.0,
        rho: float = 7.0,
        sigma_data: float = 1.0,
    ):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.sigma_data = sigma_data
        # cache the t=0/t=1 inverse-ρ endpoints
        self._u_min = sigma_min ** (1.0 / rho)
        self._u_max = sigma_max ** (1.0 / rho)

    @jaxtyped(typechecker=beartype)
    def get_alpha_t(self, t: Float[Tensor, "*batch"]) -> Float[Tensor, "*batch"]:
        """α_t = 1"""
        return torch.ones_like(t)

    @jaxtyped(typechecker=beartype)
    def get_sigma_t(self, t: Float[Tensor, "*batch"]) -> Float[Tensor, "*batch"]:
        """σ(t) = σ_data · (σ_min^{1/ρ} + t · (σ_max^{1/ρ} − σ_min^{1/ρ}))^ρ"""
        u = self._u_min + t * (self._u_max - self._u_min)
        return self.sigma_data * u**self.rho

    @jaxtyped(typechecker=beartype)
    def get_dalpha_dt(self, t: Float[Tensor, "*batch"]) -> Float[Tensor, "*batch"]:
        """dα_t/dt = 0"""
        return torch.zeros_like(t)

    @jaxtyped(typechecker=beartype)
    def get_dsigma_dt(self, t: Float[Tensor, "*batch"]) -> Float[Tensor, "*batch"]:
        """dσ/dt = σ_data · ρ · u(t)^{ρ−1} · (σ_max^{1/ρ} − σ_min^{1/ρ})"""
        u = self._u_min + t * (self._u_max - self._u_min)
        return (
            self.sigma_data
            * self.rho
            * u ** (self.rho - 1)
            * (self._u_max - self._u_min)
        )

    @jaxtyped(typechecker=beartype)
    def forward_drift(
        self, x: Float[Tensor, "*batch D"], t: Float[Tensor, "*t"]
    ) -> Float[Tensor, "*batch D"]:
        """f(x,t) = 0 — pure VE process."""
        return torch.zeros_like(x)

    @jaxtyped(typechecker=beartype)
    def diffusion_coeff(self, t: Float[Tensor, "*batch"]) -> Float[Tensor, "*batch"]:
        """g(t) = √(2 σ_t σ̇_t)"""
        sigma_t = self.get_sigma_t(t)
        dsigma_dt = self.get_dsigma_dt(t)
        return torch.sqrt(2 * sigma_t * dsigma_dt)
