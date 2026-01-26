import torch
import einops
import numbers
from torch.distributions import Independent, MultivariateNormal, Normal, MixtureSameFamily, Categorical
from torchGMM.schedule import BetaSchedule

LENGTH_SCALE = 5.0
ENERGY_SCALE = 0.1  # 0.2


class FirstDimension(torch.nn.Module):
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        return positions[..., 0:1]


f"""
GMM: [..., K, D] with arbitrary batch_shape
x: [..., D] or [..., *batch_shape, D]
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
            weight = torch.ones((*mu.shape[:-2], 1), device=mu.device, dtype=mu.dtype)

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

        self.register_buffer("mu", mu)
        self.register_buffer("sigma", sigma)
        self.register_buffer("weight", weight)

        # Normalize weights per GMM: [..., k] -> [..., k]
        weight_normalized = weight / weight.sum(dim=-1, keepdim=True)

        # Create batched Categorical distributions: batch_shape=[*batch_shape], event_shape=[]
        self.mix = Categorical(weight_normalized)

        # Create batched MultivariateNormal components
        # sigma: [..., k, d] -> covar: [..., k, d, d]
        covar = torch.diag_embed(sigma.pow(2))
        self.comp = MultivariateNormal(mu, covar)  # batch_shape=[*batch_shape, k], event_shape=[d]

        # Create batched MixtureSameFamily: batch_shape=[*batch_shape], event_shape=[d]
        self.gmm = MixtureSameFamily(self.mix, self.comp)

        self.batch_shape = mu.shape[:-2]  # [...BS, K, D]
        self.batch_ndim = len(self.batch_shape)  # #batch_dims
        self.num_components = mu.shape[-2]
        self.dim = mu.shape[-1]

        assert self.batch_ndim >= 1, "Batch dimension must be at least 1"

        # Shape metadata following PyTorch distribution conventions
        self.batch_shape = self.batch_shape
        self.event_shape = (self.dim,)  # [Dim]

        self.schedule = BetaSchedule(beta_min=0.1, beta_max=20.0) if not schedule else schedule
        self.schedule.to(mu.device)

    def __call__(self, x: torch.Tensor, t: torch.Tensor | None = None, batched_data: bool = False) -> torch.Tensor:
        """
        Compute log probability.

        Args:
            x: [N1, ..., Nk, Dim] if batched_data=False, or [N1, ..., Nk, *batch_shape, Dim] if batched_data=True
            t: scalar, tensor matching [N1, ..., Nk] or [N1, ..., Nk, B1, ... , Bk]
            batched_data: whether x is batched data in the shape of [N1, ..., Nk, *batch_shape, D] or [N1, ..., Nk, D]
        Returns:
            log_prob: [..., *batch_shape] - log probabilities with shape [*sample_event_shape, *batch_shape]
        """
        # Validate input has correct event dimension
        assert x.shape[-1] == self.dim, f"x must have last dimension {self.dim}, got {x.shape[-1]}"

        if batched_data:
            # Check that the data is [..., B1, ..., Bk, D] where [B1,...,Bk] == self.batch_shape and D == self.dim
            assert (
                x.shape[-(self.batch_ndim + 1) : -1] == self.batch_shape
            ), f"x must have batch dims {self.batch_shape} before last dimension, got {x.shape}"
            sample_event_shape = x.shape[:1]
            x_batched = x
        else:
            # x[N1, ..., Nk, D] -> x[N1, ..., Nk, B1, ... , Bk, D]
            sample_event_shape = x.shape[:-1]
            x_batched = x.reshape(*sample_event_shape, *([1] * self.batch_ndim), self.dim).expand(
                *sample_event_shape, *self.batch_shape, self.dim
            )

        # Process time: returns [N_total, *batch_shape]
        t_shaped = self._process_time(t, sample_event_shape)

        # Get batched GMM distribution with batch_shape=[N_total, *batch_shape]
        gmm_t = self._get_gmm_t(t_shaped, sample_shape=sample_event_shape)

        # Compute log probability: [N1, ... Nk, B1, ... , Bk]
        log_prob = gmm_t.log_prob(x_batched)

        assert log_prob.shape == (
            *sample_event_shape,
            *self.batch_shape,
        ), f"log_prob shape {log_prob.shape} must match sample_event_shape {sample_event_shape} and batch_shape {self.batch_shape}"

        if self.batch_shape == (1,) and not batched_data:
            log_prob = log_prob.squeeze(-1)  # [*N, BS=1] -> [*N]

        return log_prob

    def _process_time(self, t: numbers.Number | torch.Tensor | None, sample_event_shape: tuple) -> torch.Tensor:
        """
        Process time input to ensure it's in the correct format.

        Args:
            t: scalar, or tensor matching sample_event_shape, batch_shape, or [*sample_event_shape, *batch_shape], or None
            sample_event_shape: shape of sample events, e.g., (N1, N2, N3)

        Returns:
            t: [N1, ..., Nk, *batch_shape] - time tensor where N1, ..., Nk = sample_event_shape and *batch_shape = self.batch_shape
        """
        if t is None:
            # Default to t=0 for all sample events
            t = torch.zeros(*sample_event_shape, *self.batch_shape, device=self.mu.device)
        elif not isinstance(t, torch.Tensor):
            # t is a scalar -> broadcast to [*sample_event_shape, *batch_shape]
            t = torch.full((*sample_event_shape, *self.batch_shape), float(t), device=self.mu.device)
        elif t.dim() == 0:
            # Scalar tensor -> broadcast to sample_event_shape
            t = torch.full((*sample_event_shape, *self.batch_shape), t.item(), device=self.mu.device)
        elif t.shape == sample_event_shape:
            # Broadcast from [N1, ... , Nk] to [N1, ... , Nk, BS1, ... , BSk]
            expand_shape = (*sample_event_shape, *self.batch_shape)
            t = t.view(*sample_event_shape, *([1] * len(self.batch_shape))).expand(expand_shape)
        elif t.shape == (*sample_event_shape, *self.batch_shape):
            # Already correct shape [N1, ... , Nk, BS1, ..., Bk]
            pass
        else:
            raise ValueError(
                f"Time shape {t.shape} doesn't match sample_event_shape {sample_event_shape}, "
                f"batch_shape={self.batch_shape}, or (*sample_event_shape, *batch_shape) = "
                f"{(*sample_event_shape, *self.batch_shape)}"
            )

        assert t.shape == (
            *sample_event_shape,
            *self.batch_shape,
        ), f"t shape {t.shape} must match sample_event_shape {sample_event_shape} and batch_shape {self.batch_shape}"

        return t

    def _get_gmm_t(self, t: torch.Tensor, sample_shape: tuple):
        """
        Get marginal GMM distribution at time t with proper batching support.

        Args:
            t: [*N, *batch_shape] - flattened batch of time values

        Returns:
            MixtureSameFamily distribution for the marginal at time t
            with batch_shape=[*N, *BS], event_shape=[Dim]
        """

        assert t.shape == (
            *sample_shape,
            *self.batch_shape,
        ), f"t must have batch shape {self.batch_shape} at the end, got {t.shape[-self.batch_ndim:]}"
        # Get schedule parameters: [N_total, *batch_shape]
        alpha_t, sigma_t = self.schedule.get_alpha_t_sigma_t(t)
        assert (
            alpha_t.shape == sigma_t.shape == t.shape
        ), f"alpha_t shape {alpha_t.shape} must match sigma_t shape {sigma_t.shape} and t shape {t.shape}"

        # Compute time-dependent parameters using einsum
        # alpha_t[*N, *BS] mu_t: [*BS, Components, Dim]
        mu_t = alpha_t.unsqueeze(-1).unsqueeze(-1) * self.mu

        # var_t: [N_total, *batch_shape, Components, Dim]
        # Increasing variance term from noise
        increasing_var_t = (sigma_t**2).unsqueeze(-1).unsqueeze(-1)
        # Decreasing variance term from signal
        decreasing_var_t = (alpha_t.unsqueeze(-1).unsqueeze(-1) * self.sigma) ** 2
        var_t = increasing_var_t + decreasing_var_t

        # Create covariance matrices: [*N, *BS, Components, Dim, Dim]
        covar_t = torch.diag_embed(var_t)

        # Create batched MultivariateNormal components
        # batch_shape=[N_total, *batch_shape, Components], event_shape=[Dim]
        component = MultivariateNormal(loc=mu_t, covariance_matrix=covar_t)

        # Create batched Categorical mixture
        # batch_shape=[N_total, *batch_shape], event_shape=[]
        batched_probs = torch.ones(t.shape).unsqueeze(-1) * self.mix.probs
        mix = Categorical(batched_probs)

        # Create batched MixtureSameFamily
        # batch_shape=[N_total, *batch_shape], event_shape=[Dim]
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

        return TimeDependentGMM(mu_marginal, sigma_marginal, self.mix.probs, self.schedule)

    def drop_mode(self, component_index: int) -> "TimeDependentGMM":
        """
        Drop a component from all GMMs in the batch.
        """
        assert self.num_components > 1, "Cannot drop mode from a single component GMM"

        def _pop_batched(thing_to_pop, component_index, component_dim):
            # Remove index component_index from component_dim for all batches
            indices = torch.arange(thing_to_pop.shape[component_dim], device=thing_to_pop.device)
            mask = indices != component_index
            return thing_to_pop.index_select(component_dim, mask.nonzero(as_tuple=True)[0])

        mu_new = _pop_batched(self.mu, component_index, -2)  # [..., k-1, d]
        sigma_new = _pop_batched(self.sigma, component_index, -2)  # [..., k-1, d]
        probs_new = _pop_batched(self.mix.probs, component_index, -1)  # [..., k-1]

        # Renormalize weights for each GMM in batch
        probs_new = probs_new / probs_new.sum(dim=-1, keepdim=True)

        return TimeDependentGMM(mu_new, sigma_new, probs_new, self.schedule)

    def log_prob(self, x: torch.Tensor, t: torch.Tensor | None = None, batched_data: bool = False) -> torch.Tensor:
        """
        Compute log probability.

        Args:
            x: [..., Dim] if batched_data=False, or [..., *batch_shape, Dim] if batched_data=True
            t: scalar, tensor matching x.shape[:-1] (unbatched), or
                scalar/[batch_shape]/[...]/[...,batch_shape] (batched), or None

        Returns:
            log_prob: [..., *batch_shape] - log probabilities with shape [*sample_event_shape, *batch_shape]
        """
        return self.__call__(x, t, batched_data=batched_data)

    def cdf(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        """
        Compute cumulative distribution function.

        Note: Currently only supports 1D case (d=1).

        Args:
            x: [..., *batch_shape, 1] - data points with arbitrary leading dimensions (only Dim=1 supported)
            t: scalar, tensor matching x.shape[:-1], batch_shape, or [..., *batch_shape], or None

        Returns:
            cdf: [..., *batch_shape] - CDF values with shape [*sample_event_shape, *batch_shape]
        """
        assert x.shape[-1] == self.dim == 1, f"CDF only supports 1D (Dim=1), got x.shape={x.shape}, self.dim={self.dim}"
        if self.batch_ndim > 0:
            batch_slice = x.shape[-(self.batch_ndim + 1) : -1]
            assert batch_slice == self.batch_shape, f"CDF expects batch shape {self.batch_shape}, got {batch_slice}"

        # Extract sample_event_shape and flatten
        sample_event_shape = x.shape[: -(self.batch_ndim + 1)] if self.batch_ndim > 0 else x.shape[:-1]
        x_flat = x.reshape(-1, *self.batch_shape, 1)  # [N_total, *batch_shape, 1]

        # Process time: returns [N_total, *batch_shape]
        t_flat = self._process_time(t, sample_event_shape)

        # Get time-dependent parameters using einsum
        alpha_t, sigma_t = self.schedule.get_alpha_t_sigma_t(t_flat)  # [N_total, *batch_shape]

        # mu_t: [N_total, *batch_shape, Components, 1]
        mu_t = einops.einsum(alpha_t, self.mu, "n ... , ... k d -> n ... k d")

        # std_t: [N_total, *batch_shape, Components, 1]
        increasing_var_t = einops.einsum(
            sigma_t**2, torch.ones(self.num_components, 1, device=self.mu.device), "n ... , k d -> n ... k d"
        )
        decreasing_var_t = einops.einsum(alpha_t, self.sigma, "n ... , ... k d -> n ... k d") ** 2
        var_t = increasing_var_t + decreasing_var_t
        std_t = torch.sqrt(var_t)

        # Expand x for broadcasting: [N_total, *batch_shape, 1] -> [N_total, *batch_shape, Components, 1]
        x_expanded = einops.repeat(x_flat, "n ... d -> n ... k d", k=self.num_components)

        # Compute component CDFs: [N_total, *batch_shape, Components]
        component_cdf = Normal(mu_t.squeeze(-1), std_t.squeeze(-1)).cdf(x_expanded.squeeze(-1))

        # Mix CDFs using einsum: [N_total, *batch_shape, Components] * [*batch_shape, Components] -> [N_total, *batch_shape]
        batched_probs = einops.repeat(self.mix.probs, "... k -> n ... k", n=t_flat.shape[0])
        mix_cdf_flat = einops.einsum(component_cdf, batched_probs, "n ... k, n ... k -> n ...")

        # Reshape to [*sample_event_shape, *batch_shape]
        mix_cdf = mix_cdf_flat.reshape(*sample_event_shape, *self.batch_shape)

        return mix_cdf

    def energy(self, x: torch.Tensor, t: torch.Tensor | None = None, batched_data: bool = False) -> torch.Tensor:
        """
        Compute energy in the form of -log_prob.

        Args:
            x: [..., Dim] if batched_data=False, or [..., *batch_shape, Dim] if batched_data=True
            t: scalar, tensor matching x.shape[:-1] (unbatched), or
                scalar/[batch_shape]/[...]/[...,batch_shape] (batched), or None

        Returns:
            energy: [..., *batch_shape] - energy with shape [*sample_event_shape, *batch_shape]
        """
        return -self.__call__(x, t, batched_data=batched_data)

    @torch.enable_grad()
    def score(self, x: torch.Tensor, t: torch.Tensor | None = None, batched_data: bool = False) -> torch.Tensor:
        """
        Compute score function using autograd: ∇_x log p(x)

        Args:
            x: [..., Dim] if batched_data=False, or [..., *batch_shape, Dim] if batched_data=True
            t: scalar, tensor matching x.shape[:-1] (unbatched), or
                scalar/[batch_shape]/[...]/[...,batch_shape] (batched), or None

        Returns:
            score: [..., *batch_shape, Dim] - score with shape [*sample_event_shape, *batch_shape, *event_shape]
        """
        # Validate input has correct event dimension
        assert x.shape[-1] == self.dim, f"x must have last dimension {self.dim}, got {x.shape[-1]}"

        if batched_data:
            assert (
                x.shape[-(self.batch_ndim + 1) : -1] == self.batch_shape
            ), f"x must have batch shape {self.batch_shape} before last dimension, got {x.shape[-(self.batch_ndim + 1) : -1]}"
            sample_event_shape = x.shape[: -(self.batch_ndim + 1)]  # [N1, ..., Nk] from [N1, ..., Nk, *BS, D]
            x_batched = x
        else:
            sample_event_shape = x.shape[:-1]

            x_batched = x.reshape(*sample_event_shape, *([1] * self.batch_ndim), self.dim).expand(
                *sample_event_shape, *self.batch_shape, self.dim
            )

        # Flatten to [N_total, *batch_shape, Dim]
        x_batched.requires_grad_(True)

        # Process time: returns [N_total, *batch_shape]
        t = self._process_time(t, sample_event_shape)

        # Get batched GMM distribution with batch_shape=[(N), *BS]
        gmm_t = self._get_gmm_t(t, sample_shape=sample_event_shape)

        # Compute log probability: [N_total, *batch_shape]
        log_prob = gmm_t.log_prob(x_batched)

        # Compute gradient: [N_total, *batch_shape, Dim]
        score = torch.autograd.grad(log_prob.sum(), x_batched, create_graph=False)[0]
        assert score.shape == (
            *sample_event_shape,
            *self.batch_shape,
            self.dim,
        ), f"score shape {score.shape} must match sample_event_shape {sample_event_shape} and batch_shape {self.batch_shape} and dim {self.dim}"

        if self.batch_shape == (1,) and not batched_data:
            score = score.squeeze(-2)  # [N_total, *batch_shape, 1] -> [N_total, *batch_shape]
        return score

    def sample(self, shape: tuple | int | None = None, t: torch.Tensor | float | None = None) -> torch.Tensor:
        """
        Sample from the batched GMMs at time t.

        Args:
            shape: tuple, int, or None - shape of sample events to generate
                   - None: no sample_event_shape, returns [*batch_shape, Dim]
                   - int: sample_event_shape=(shape,), returns [shape, *batch_shape, Dim]
                   - tuple: sample_event_shape=shape, returns [*shape, *batch_shape, Dim]
            t: scalar, tensor matching shape, batch_shape, or [*shape, *batch_shape], or None
               - None: assume t=0 for all
               - scalar: same time for all sample events
               - tensor: must match sample_event_shape

        Returns:
            samples: [..., *batch_shape, Dim] - samples with shape [*sample_event_shape, *batch_shape, *event_shape]
        """
        # Determine sample_event_shape
        if shape is None:
            sample_event_shape = ()
        elif isinstance(shape, int):
            sample_event_shape = (shape,)
        elif isinstance(shape, tuple):
            sample_event_shape = shape
        else:
            raise ValueError(f"Invalid shape: {shape}")
        t = self._process_time(t, sample_event_shape=sample_event_shape)

        # Get batched GMM distribution with batch_shape=[N_total, *batch_shape]
        gmm_t = self._get_gmm_t(t, sample_shape=sample_event_shape)

        # Sample: [N_total, *batch_shape, Dim]
        samples = gmm_t.sample()

        return samples

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
        weight = torch.ones((*mu.shape[:-2], 1), device=mu.device, dtype=mu.dtype)
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
