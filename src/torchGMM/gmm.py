import torch
import einops
import numbers
from torch.distributions import MultivariateNormal, Normal, MixtureSameFamily, Categorical
from torchGMM.schedule import BetaSchedule

"""
TimeDependentGMM: GMM with params [*B, K, D]. Two shapes only:
- Batch shape B (from init). Sample shape N (optional leading dims).
- x: [*B, D] or [*N, *B, D]. t: [*B] or [*N, *B] (scalar per batch; no event dim).
- Outputs: log_prob/energy -> [*N, *B], score -> [*N, *B, D], sample(shape, t) -> [*N, *B, D].
"""


class TimeDependentGMM(torch.nn.Module):
    def __init__(
        self, mu: torch.Tensor, sigma: torch.Tensor = None, weight: torch.Tensor = None, schedule: BetaSchedule = None
    ):
        super().__init__()
        """
        Args:
            mu: [..., k, d] - means for batched GMMs, each with k components and d dimensions
            sigma: [..., k, d] - standard deviations for batched GMMs, each with k components and d dimensions
            weight: [..., k] - mixture weights for batched GMMs, each with k components
        """
        assert (mu is not None and sigma is not None and weight is not None) or (
            mu is not None
        ), "Mu, sigma, and weight must be provided"
        assert mu.dim() >= 2, f"mu must be at least 2D [*, k, d], got {mu.shape}"
        if sigma is None:
            sigma = torch.zeros_like(mu) + 1e-10
        if weight is None:
            weight = mu.new_ones((*mu.shape[:-2], 1))

        assert (
            mu.shape[:-1] == sigma.shape[:-1] == weight.shape
        ), f"mu/sigma/weight leading shapes must match: mu {mu.shape}, sigma {sigma.shape}, weight {weight.shape}"

        # If mu is [K, D], treat it as a single-batch GMM and expand to [1, K, D]
        if mu.dim() == 2:
            mu = mu.unsqueeze(0)
            sigma = sigma.unsqueeze(0)
            weight = weight.unsqueeze(0)

        assert sigma.shape == mu.shape, f"sigma must match mu shape, got {sigma.shape} vs {mu.shape}"
        assert weight.shape == mu.shape[:-1], f"weight must be [..., k], got {weight.shape}"
        assert (
            mu.shape[-2] == weight.shape[-1]
        ), f"Number of components must match: mu {mu.shape[-2]}, weight {weight.shape[-1]}"

        weight = weight / weight.sum(dim=-1, keepdim=True)
        self.register_buffer("mu", mu)
        self.register_buffer("sigma", sigma)
        self.register_buffer("weight", weight)

        self.batch_shape = mu.shape[:-2]  # [...BS, K, D]
        self.batch_ndim = len(self.batch_shape)  # #batch_dims
        self.num_components = mu.shape[-2]
        self.dim = mu.shape[-1]

        assert self.batch_ndim >= 1, "Batch dimension must be at least 1"

        # Shape metadata following PyTorch distribution conventions
        self.batch_shape = self.batch_shape
        self.event_shape = (self.dim,)  # [Dim]

        self.schedule = BetaSchedule(beta_min=0.1, beta_max=20.0) if not schedule else schedule

    def _expand_t(self, t: numbers.Number | torch.Tensor | None, sample_shape: tuple) -> torch.Tensor:
        """Return t with shape [*sample_shape, *batch_shape]. Accept t scalar, [*B], or [*N,*B]."""
        if t is None:
            return self.mu.new_ones(*sample_shape, *self.batch_shape) * 0.0
        if not isinstance(t, torch.Tensor):
            t = self.mu.new_ones(*self.batch_shape, device=self.mu.device) * float(t)
        elif t.dim() == 0:
            t = self.mu.new_ones(*self.batch_shape, device=t.device) * t.item()
        if t.shape == self.batch_shape:
            return t.broadcast_to((*sample_shape, *self.batch_shape)).clone()
        if t.shape == sample_shape:
            return t
        raise ValueError(f"t must be of shape {t.shape=} must be {sample_shape=} or {self.batch_shape=}")

    def _gmm_t(self, t: torch.Tensor) -> MixtureSameFamily:
        """Marginal GMM at time t. t: [*N, *B]. Returns MixtureSameFamily with batch_shape=t.shape, event_shape=(D,)."""
        assert (
            t.shape[-self.batch_ndim :] == self.batch_shape
        ), f"t must have trailing dims batch_shape {self.batch_shape}, got {t.shape}"
        # t [*N, *B], self.mu / self.sigma / self.weight [*B, K, D] or [*B, K]
        alpha_t, sigma_t = self.schedule.get_alpha_t_sigma_t(t)  # [*N, *B]
        # [*N,*B,1,1] * [*B,K,D] -> [*N,*B,K,D]
        alpha_t_nk = alpha_t.unsqueeze(-1).unsqueeze(-1)  # [*N, *B, 1, 1]
        mu_t = alpha_t_nk * self.mu  # [*N, *B, K, D]
        increasing_var_t = (sigma_t**2).unsqueeze(-1).unsqueeze(-1)  # [*N, *B, 1, 1]
        decreasing_var_t = (alpha_t_nk * self.sigma) ** 2  # [*N, *B, K, D]
        var_t = increasing_var_t + decreasing_var_t  # [*N, *B, K, D]
        covar_t = torch.diag_embed(var_t)  # [*N, *B, K, D, D]
        component = MultivariateNormal(loc=mu_t, covariance_matrix=covar_t)
        batched_probs = torch.ones(t.shape, device=self.weight.device).unsqueeze(-1) * self.weight  # [*N, *B, K]
        mix = Categorical(batched_probs)
        return MixtureSameFamily(mix, component)

    def marginal_gmm(self, dim) -> "TimeDependentGMM":
        """
        Get marginal GMM distribution for batched GMMs.

        Args:
            dim: int - dimension to marginalize out

        Returns:
            TimeDependentGMM with a single dimension for the marginal GMM
        Notes
        -----
        For a 2D Gaussian mixture with independent coordinates per component,
            p(x, y) = ∑_k π_k N([x, y]; μ_k, Σ_k),
        where Σ_k = diag(σ_{x,k}², σ_{y,k}²).

        Because each component factorizes as
            N([x, y]; μ_k, Σ_k) = N(x; μ_{x,k}, σ_{x,k}²) * N(y; μ_{y,k}, σ_{y,k}²),
        we can integrate out one variable analytically:
            p(x) = ∫ p(x, y) dy = ∑_k π_k N(x; μ_{x,k}, σ_{x,k}²)
            p(y) = ∫ p(x, y) dx = ∑_k π_k N(y; μ_{y,k}, σ_{y,k}²)
        Hence, each marginal is itself a 1D Gaussian mixture with the same component
        weights but the corresponding mean and variance from the chosen axis.
        """
        # Extract dimension dim from [..., k, d] -> [..., k, 1]
        mu_marginal = self.mu[..., dim].unsqueeze(-1)  # [..., k, 1]
        sigma_marginal = self.sigma[..., dim].unsqueeze(-1)  # [..., k, 1]

        return TimeDependentGMM(mu_marginal, sigma_marginal, self.weight, self.schedule)

    def drop_mode(self, component_index: int) -> "TimeDependentGMM":
        """
        Drop a component from all GMMs in the batch.
        """
        assert self.num_components > 1, "Cannot drop mode from a single component GMM"

        def _pop_batched(thing_to_pop, component_index, component_dim):
            # Remove index component_index from component_dim for all batches
            indices = torch.arange(thing_to_pop.shape[component_dim], dtype=torch.long, device=thing_to_pop.device)
            mask = indices != component_index
            return thing_to_pop.index_select(component_dim, mask.nonzero(as_tuple=True)[0])

        mu_new = _pop_batched(self.mu, component_index, -2)  # [..., k-1, d]
        sigma_new = _pop_batched(self.sigma, component_index, -2)  # [..., k-1, d]
        weight_new = _pop_batched(self.weight, component_index, -1)  # [..., k-1]

        # Renormalize weights for each GMM in batch
        weight_new = weight_new / weight_new.sum(dim=-1, keepdim=True)

        return TimeDependentGMM(mu_new, sigma_new, weight_new, self.schedule)

    def log_prob(self, x: torch.Tensor, t: numbers.Number | torch.Tensor | None = None) -> torch.Tensor:
        """log_prob(x, t) -> [*N, *B]. x: [*N, *B, D], t: [*B] or [*N, *B] (or scalar)."""
        assert x.shape[-1] == self.dim, f"x last dim must be {self.dim}, got {x.shape[-1]}"
        assert (
            x.shape[-(self.batch_ndim + 1) : -1] == self.batch_shape
        ), f"x must have batch dims {self.batch_shape} before last, got {x.shape}"
        sample_shape = x.shape[: -(self.batch_ndim + 1)]
        t_exp = self._expand_t(t, sample_shape)
        return self._gmm_t(t_exp).log_prob(x)

    def __call__(self, x: torch.Tensor, t: numbers.Number | torch.Tensor | None = None) -> torch.Tensor:
        """Alias for log_prob(x, t)."""
        return self.log_prob(x, t)

    def energy(self, x: torch.Tensor, t: numbers.Number | torch.Tensor | None = None) -> torch.Tensor:
        """energy(x, t) -> [*N, *B]. Returns -log_prob(x, t)."""
        return -self.log_prob(x, t)

    def cdf(self, x: torch.Tensor, t: numbers.Number | torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError("CDF not implemented for time dependent GMMs")
        """cdf(x, t) -> [*N, *B]. Only 1D (D=1). x: [*N, *B, 1], t: [*B] or [*N, *B] (or scalar)."""
        assert x.shape[-1] == self.dim == 1, f"CDF only supports 1D, got x.shape={x.shape}, self.dim={self.dim}"
        assert (
            x.shape[-(self.batch_ndim + 1) : -1] == self.batch_shape
        ), f"x must have batch dims {self.batch_shape} before last, got {x.shape}"
        sample_shape = x.shape[: -(self.batch_ndim + 1)]
        t_exp = self._expand_t(t, sample_shape)
        x_flat = x.reshape(-1, *self.batch_shape, 1)
        t_flat = t_exp.reshape(-1, *self.batch_shape)
        alpha_t, sigma_t = self.schedule.get_alpha_t_sigma_t(t_flat)
        mu_t = einops.einsum(alpha_t, self.mu, "n ... , ... k d -> n ... k d")
        increasing_var_t = einops.einsum(
            sigma_t**2, self.mu.new_ones(self.num_components, 1), "n ... , k d -> n ... k d"
        )
        decreasing_var_t = einops.einsum(alpha_t, self.sigma, "n ... , ... k d -> n ... k d") ** 2
        std_t = torch.sqrt(increasing_var_t + decreasing_var_t)
        x_expanded = einops.repeat(x_flat, "n ... d -> n ... k d", k=self.num_components)
        component_cdf = Normal(mu_t.squeeze(-1), std_t.squeeze(-1)).cdf(x_expanded.squeeze(-1))
        batched_probs = einops.repeat(self.weight, "... k -> n ... k", n=t_flat.shape[0])
        mix_cdf_flat = einops.einsum(component_cdf, batched_probs, "n ... k, n ... k -> n ...")
        return mix_cdf_flat.reshape(*sample_shape, *self.batch_shape)

    @torch.enable_grad()
    def score(self, x: torch.Tensor, t: numbers.Number | torch.Tensor | None = None) -> torch.Tensor:
        """score(x, t) -> [*N, *B, D]. ∇_x log p(x). x: [*N, *B, D], t: [*B] or [*N, *B] (or scalar)."""
        assert x.shape[-1] == self.dim, f"x last dim must be {self.dim}, got {x.shape[-1]}"
        assert (
            x.shape[-(self.batch_ndim + 1) : -1] == self.batch_shape
        ), f"x must have batch dims {self.batch_shape} before last, got {x.shape}"
        sample_shape = x.shape[: -(self.batch_ndim + 1)]
        t_exp = self._expand_t(t, sample_shape)
        x = x.requires_grad_(True)
        return torch.autograd.grad(self._gmm_t(t_exp).log_prob(x).sum(), x, create_graph=False)[0]

    def sample(self, shape: tuple | int | None = None, t: numbers.Number | torch.Tensor | None = None) -> torch.Tensor:
        """sample(shape, t) -> [*N, *B, D]. shape: int or tuple (sample shape N); t: [*B] or [*N,*B] (or scalar)."""
        if shape is None:
            sample_shape = ()
        elif isinstance(shape, int):
            sample_shape = (shape,)
        elif isinstance(shape, tuple):
            sample_shape = shape
        else:
            raise ValueError(f"shape must be int, tuple, or None, got {type(shape)}")
        t_exp = self._expand_t(t, sample_shape)
        return self._gmm_t(t_exp).sample()

    def __repr__(self):
        return f"TimeDependentGMM(mu={self.mu.shape}, sigma={self.sigma.shape}, weight={self.weight.shape})"


class Conditional(TimeDependentGMM):
    """
    Conditional Process class
    Instead of simulating the full GMM, we only simulate the conditional process conditioned on the initial value x0.
    This is useful for conditional sampling and inference.

    Args:
        x0: [..., d] - initial value
        schedule: BetaSchedule - schedule for the conditional process

    Returns:
        Conditional - Conditional process
    """

    def __init__(self, x0: torch.Tensor, schedule: BetaSchedule = None):
        mu = x0.unsqueeze(-2)  # [..., d] -> [..., 1, d]
        assert mu.dim() == x0.dim() + 1, f"mu must be a tensor [..., 1, d], got {mu.shape}"
        sigma = torch.zeros_like(mu) + 1e-10
        weight = mu.new_ones((*mu.shape[:-2], 1))
        super().__init__(mu, sigma, weight, schedule)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    for _ in range(3):
        num_components = 5
        # Create batched GMM with batch_shape=(1,) (single GMM)
        mu = torch.ones(1, num_components, 2).uniform_(-3, 3)  # [1, k=5, d=2]
        sigma = torch.ones(1, num_components, 2).uniform_(0.5, 1.2)  # [1, k=5, d=2]
        weight = torch.ones(1, num_components).uniform_(0.3, 1.0)  # [1, k=5]
        gmm = TimeDependentGMM(mu, sigma, weight)
        samples = gmm.sample(1_000_000)  # [n_samples=1_000_000, 1, d=2]
        # Extract samples for plotting: [1_000_000, 2]
        samples_plot = samples[0]  # [1_000_000, 2]
        plt.hist2d(samples_plot[:, 0], samples_plot[:, 1], bins=100)
        plt.grid()
        plt.show()

    # from mcmc.metropolis_hasting_nd import batch_mh, batch_langevin

    def sample_fn(x):
        return gmm.energy(x), x

    init_samples = torch.randn(500, 2)
