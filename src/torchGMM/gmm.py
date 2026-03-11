import numbers

import einops
import torch
from torch.distributions import Categorical, MixtureSameFamily, MultivariateNormal, Normal

from torchGMM.schedule import BetaSchedule, FlowMatchingSchedule, Schedule

"""
TimeDependentGMM: GMM with params [*B, K, D]. Two shapes only:
- Batch shape B (from init). Sample shape N (optional leading dims).
- x: [*B, D] or [*N, *B, D]. t: [*B] or [*N, *B] (scalar per batch; no event dim).
- Outputs: log_prob/energy -> [*N, *B], score -> [*N, *B, D], sample(shape, t) -> [*N, *B, D].
"""


class TimeDependentGMM(torch.nn.Module):
    def __init__(
        self, mu: torch.Tensor, sigma: torch.Tensor = None, weight: torch.Tensor = None, schedule: Schedule = None
    ):
        super().__init__()
        """
        Args:
            mu: [..., k, d] - means for batched GMMs, each with k components and d dimensions
            sigma: [..., k, d] - standard deviations for batched GMMs, each with k components and d dimensions
            weight: [..., k] - mixture weights for batched GMMs, each with k components
            schedule: Optional Schedule. If not provided, defaults to BetaSchedule(beta_min=0.1, beta_max=20.0).
        """
        assert (mu is not None and sigma is not None and weight is not None) or (
            mu is not None
        ), "Mu, sigma, and weight must be provided or just mu"
        mu = torch.tensor(mu)
        assert mu.dim() >= 2, f"mu must be at least 2D [*, k, d], got {mu.shape}"
        sigma = torch.zeros_like(mu) + 1e-10 if sigma is None else torch.tensor(sigma)
        weight = mu.new_ones((*mu.shape[:-2], 1)) if weight is None else torch.tensor(weight)

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

        self._validate_positive_tensor("sigma", sigma)
        self._validate_positive_tensor("weight", weight)
        weight_sum = weight.sum(dim=-1, keepdim=True)
        if not torch.all(weight_sum > 0):
            raise ValueError("weight sum must be > 0 for all batches")
        weight = weight / weight_sum
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

        self.schedule = BetaSchedule(beta_min=0.1, beta_max=20.0) if schedule is None else schedule

    def _expand_t(self, t: numbers.Number | torch.Tensor | None, sample_shape: tuple) -> torch.Tensor:
        """Return t with shape [*sample_shape, *batch_shape]. Accept t scalar, [*B], or [*N,*B]."""
        if t is None:
            # None initializes to 0.0
            return torch.ones(*(sample_shape + self.batch_shape), device=self.mu.device, dtype=self.mu.dtype) * 0.0
        elif not isinstance(t, torch.Tensor):
            t_value = float(t)
            self._validate_time_range(t_value)
            return torch.ones(*(sample_shape + self.batch_shape), device=self.mu.device, dtype=self.mu.dtype) * t_value
        elif t.dim() == 0:
            t_value = t.item()
            self._validate_time_range(t_value)
            return torch.ones(*(sample_shape + self.batch_shape), device=t.device, dtype=t.dtype) * t_value
        elif t.shape == sample_shape + self.batch_shape:
            self._validate_time_range(t)
            return t
        elif t.shape == self.batch_shape:
            self._validate_time_range(t)
            if len(sample_shape) == 0:
                return t
            return t.expand(sample_shape + self.batch_shape)
        raise ValueError(
            f"t must be of shape {t.shape if isinstance(t, torch.Tensor) else t} must be {sample_shape+self.batch_shape=}, got {t.shape if isinstance(t, torch.Tensor) else t}"
        )

    @staticmethod
    def _validate_positive_tensor(name: str, tensor: torch.Tensor) -> None:
        if not torch.all(torch.isfinite(tensor)):
            raise ValueError(f"{name} must be finite")
        if not torch.all(tensor > 0):
            raise ValueError(f"{name} must be > 0")

    @staticmethod
    def _validate_time_range(t: numbers.Number | torch.Tensor) -> None:
        if isinstance(t, torch.Tensor):
            if not torch.all(torch.isfinite(t)):
                raise ValueError("t must be finite")
            if not torch.all((t >= 0) & (t <= 1)):
                raise ValueError("t must be within [0, 1]")
        else:
            if not (0 <= t <= 1):
                raise ValueError("t must be within [0, 1]")

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
        """log_prob(x, t) -> [*N, *B]. x: [*N, *B, D], t: scalar or shape x.shape[:-1], with t in [0, 1]."""
        assert x.shape[-1] == self.dim, f"x last dim must be {self.dim}, got {x.shape[-1]}"
        assert (
            x.shape[-(self.batch_ndim + 1) : -1] == self.batch_shape
        ), f"x must have batch dims {self.batch_shape} before last, got {x.shape}"
        sample_shape = x.shape[: -(self.batch_ndim + 1)]
        t_exp = self._expand_t(t, sample_shape)
        assert t_exp.shape == x.shape[:-1], f"t_exp must have shape {x.shape[:-1]}, got {t_exp.shape}"
        return self._gmm_t(t_exp).log_prob(x)

    def __call__(self, x: torch.Tensor, t: numbers.Number | torch.Tensor | None = None) -> torch.Tensor:
        """Alias for log_prob(x, t)."""
        return self.log_prob(x, t)

    def energy(self, x: torch.Tensor, t: numbers.Number | torch.Tensor | None = None) -> torch.Tensor:
        """energy(x, t) -> [*N, *B]. Returns -log_prob(x, t). t: scalar or shape x.shape[:-1] in [0, 1]."""
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
        """score(x, t) -> [*N, *B, D]. ∇_x log p(x). x: [*N, *B, D], t: scalar or shape x.shape[:-1] in [0, 1]."""
        assert x.shape[-1] == self.dim, f"x last dim must be {self.dim}, got {x.shape[-1]}"
        assert (
            x.shape[-(self.batch_ndim + 1) : -1] == self.batch_shape
        ), f"x must have batch dims {self.batch_shape} before last, got {x.shape}"
        sample_shape = x.shape[: -(self.batch_ndim + 1)]
        t_exp = self._expand_t(t, sample_shape)
        assert t_exp.shape == x.shape[:-1], f"t_exp must have shape {x.shape[:-1]}, got {t_exp.shape}"
        x_grad = x.detach().clone().requires_grad_(True)
        score = torch.autograd.grad(self._gmm_t(t_exp).log_prob(x_grad).sum(), x_grad, create_graph=False)[0].detach()
        return score

    def velocity(self, x: torch.Tensor, t: numbers.Number | torch.Tensor | None = None) -> torch.Tensor:
        """velocity(x, t) -> [*N, *B, D]. Marginal velocity field.

        Derived from v_t(x) = (dα/dt) E[x_0|x_t=x] + (dσ/dt) E[ε|x_t=x]
        using Tweedie: E[x_0|x_t] = (x + σ² score) / α, E[ε|x_t] = -σ score.
        """
        assert x.shape[-1] == self.dim, f"x last dim must be {self.dim}, got {x.shape[-1]}"
        assert (
            x.shape[-(self.batch_ndim + 1) : -1] == self.batch_shape
        ), f"x must have batch dims {self.batch_shape} before last, got {x.shape}"
        sample_shape = x.shape[: -(self.batch_ndim + 1)]
        t_exp = self._expand_t(t, sample_shape)  # [*N, *B]
        assert t_exp.shape == x.shape[:-1], f"t_exp must have shape {x.shape[:-1]}, got {t_exp.shape}"

        alpha_t = self.schedule.get_alpha_t(t_exp)  # [*N, *B]
        sigma_t = self.schedule.get_sigma_t(t_exp)  # [*N, *B]
        dalpha_dt = self.schedule.get_dalpha_dt(t_exp)  # [*N, *B]
        dsigma_dt = self.schedule.get_dsigma_dt(t_exp)  # [*N, *B]

        score = self.score(x, t)  # [*N, *B, D]

        # v = (dα/dt / α) x + (dα/dt σ/α - dσ/dt) σ score
        coeff_x = (dalpha_dt / alpha_t).unsqueeze(-1)  # [*N, *B, 1]
        coeff_score = ((dalpha_dt * sigma_t / alpha_t - dsigma_dt) * sigma_t).unsqueeze(-1)  # [*N, *B, 1]

        return coeff_x * x + coeff_score * score

    def sample(self, shape: tuple | int | None = None, t: numbers.Number | torch.Tensor | None = None) -> torch.Tensor:
        """sample(shape, t) -> [*N, *B, D]. shape: full [*N,*B] (tuple must end with batch_shape), or int (N single dim), or None -> [*B,D]. t: scalar or shape [*N,*B] in [0, 1]."""
        if shape is None:
            sample_shape = ()
        elif isinstance(shape, int):
            sample_shape = (shape,)
        elif isinstance(shape, tuple):
            assert (
                shape[-self.batch_ndim :] == self.batch_shape
            ), f"shape must end with batch_shape {self.batch_shape}, got {shape}"
            sample_shape = shape[: -self.batch_ndim]
        t_exp = self._expand_t(t, sample_shape)
        assert (
            t_exp.shape == sample_shape + self.batch_shape
        ), f"t_exp must have shape {sample_shape+self.batch_shape}, got {t_exp.shape}"
        return self._gmm_t(t_exp).sample()

    def __repr__(self):
        return f"TimeDependentGMM(mu={self.mu.shape}, sigma={self.sigma.shape}, weight={self.weight.shape})"


from torchGMM.conditional import Conditional

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
