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
GMM: [BS, K, D]
x: [..., D] or [..., BS, D]
"""


class TimeDependentGMM(torch.nn.Module):
    def __init__(
        self, mu: torch.Tensor, sigma: torch.Tensor = None, weight: torch.Tensor = None, schedule: BetaSchedule = None
    ):
        super().__init__()
        """
        Args:
            mu: [BS, k, d] - means for BS batched GMMs, each with k components and d dimensions
            sigma: [BS, k, d] - standard deviations for BS batched GMMs, each with k components and d dimensions  
            weight: [BS, k] - mixture weights for BS batched GMMs, each with k components
        """
        assert (mu is not None and sigma is not None and weight is not None) or (
            mu is not None
        ), "Either mu, sigma, and weight must be provided or mu only"
        assert mu.dim() == 3, f"mu must be a 3D tensor [BS, k, d], got {mu.shape}"

        # If sigma and weight not provided, create defaults
        if sigma is None:
            sigma = torch.zeros_like(mu) + 1e-10
        if weight is None:
            weight = torch.ones(mu.shape[0], mu.shape[1], device=mu.device, dtype=mu.dtype)

        assert sigma.dim() == 3, f"sigma must be a 3D tensor [BS, k, d], got {sigma.shape}"
        assert weight.dim() == 2, f"weight must be a 2D tensor [BS, k], got {weight.shape}"
        assert (
            mu.shape[0] == sigma.shape[0] == weight.shape[0]
        ), f"Batch size must match: mu {mu.shape[0]}, sigma {sigma.shape[0]}, weight {weight.shape[0]}"
        assert (
            mu.shape[1] == sigma.shape[1] == weight.shape[1]
        ), f"Number of components must match: mu {mu.shape[1]}, sigma {sigma.shape[1]}, weight {weight.shape[1]}"
        assert mu.shape[2] == sigma.shape[2], f"Dimension must match: mu {mu.shape[2]}, sigma {sigma.shape[2]}"

        self.register_buffer("mu", mu)
        self.register_buffer("sigma", sigma)
        self.register_buffer("weight", weight)

        # Normalize weights per GMM: [BS, k] -> [BS, k]
        weight_normalized = weight / weight.sum(dim=1, keepdim=True)  # [BS, k]

        # Create batched Categorical distributions: batch_shape=[BS], event_shape=[]
        self.mix = Categorical(weight_normalized)

        # Create batched MultivariateNormal components
        # sigma: [BS, k, d] -> covar: [BS, k, d, d]
        covar = torch.diag_embed(sigma.pow(2))  # [BS, k, d, d]
        self.comp = MultivariateNormal(mu, covar)  # batch_shape=[BS, k], event_shape=[d]

        # Create batched MixtureSameFamily: batch_shape=[BS], event_shape=[d]
        self.gmm = MixtureSameFamily(self.mix, self.comp)

        self.batch_size = mu.shape[-3]
        self.BS = mu.shape[-3]
        self.num_components = mu.shape[-2]
        self.dim = mu.shape[-1]

        # Shape metadata following PyTorch distribution conventions
        self.batch_shape = (self.BS,)  # [BS]
        self.event_shape = (self.dim,)  # [Dim]

        self.schedule = BetaSchedule(beta_min=0.1, beta_max=20.0) if not schedule else schedule
        self.schedule.to(mu.device)

    def check_is_batched(self, x: torch.Tensor) -> None:
        assert x.shape[-2] == self.BS, f"x must have batch dimension {self.BS}, got {x.shape[-2]}"

    def __call__(self, x: torch.Tensor, t: torch.Tensor | None = None, batched_data: bool = False) -> torch.Tensor:
        """
        Compute log probability.

        Args:
            x: [..., Dim] if batched_data=False, or [..., BS, Dim] if batched_data=True
            t: scalar, tensor matching x.shape[:-1] (unbatched), or
               scalar/[BS]/[...]/[...,BS] (batched), or None
            batched_data: whether x is batched data in the shape of [..., BS, D] or [..., D]
        Returns:
            log_prob: [..., BS] - log probabilities with shape [*sample_event_shape, *batch_shape]
        """
        # Validate input has correct event dimension
        assert x.shape[-1] == self.dim, f"x must have last dimension {self.dim}, got {x.shape[-1]}"

        if batched_data:
            self.check_is_batched(x)
            sample_event_shape = x.shape[:-2]
            x_batched = x
        else:
            sample_event_shape = x.shape[:-1]
            x_batched = einops.repeat(x, "... d -> ... b d", b=self.BS)

        # Flatten to [N_total, BS, Dim]
        x_flat = x_batched.reshape(-1, self.BS, self.dim)

        # Process time: returns [N_total, BS]
        t_flat = self._process_time(t, sample_event_shape)

        # Get batched GMM distribution with batch_shape=[N_total, BS]
        gmm_t = self._get_gmm_t(t_flat)

        # Compute log probability: [N_total, BS]
        log_prob_flat = gmm_t.log_prob(x_flat)

        # Reshape to [*sample_event_shape, *batch_shape]
        log_prob = log_prob_flat.reshape(*sample_event_shape, *self.batch_shape)

        return log_prob

    def _process_time(self, t: numbers.Number | torch.Tensor | None, sample_event_shape: tuple) -> torch.Tensor:
        """
        Process time input to ensure it's in the correct format.

        Args:
            t: scalar, or tensor matching sample_event_shape, [BS], or [*sample_event_shape, BS], or None
            sample_event_shape: shape of sample events, e.g., (N1, N2, N3)

        Returns:
            t: [N_total, BS] - flattened time tensor where N_total = prod(sample_event_shape)
        """
        if t is None:
            # Default to t=0 for all sample events
            t = torch.zeros(*sample_event_shape, *self.batch_shape, device=self.mu.device)
        elif not isinstance(t, torch.Tensor):
            # Convert float/int to tensor and broadcast
            t = torch.full((*sample_event_shape, *self.batch_shape), float(t), device=self.mu.device)
        elif t.dim() == 0:
            # Scalar tensor -> broadcast to sample_event_shape
            t = torch.full((*sample_event_shape, *self.batch_shape), t.item(), device=self.mu.device)
        elif t.shape == (self.BS,):
            # Broadcast to [*sample_event_shape, BS]
            expand_shape = (*sample_event_shape, self.BS)
            t = t.view(*([1] * len(sample_event_shape)), self.BS).expand(expand_shape)
        elif t.shape == sample_event_shape:
            # Expand using einsum: [N1, N2, ...] -> [N1, N2, ..., BS]
            t = einops.repeat(t, "... -> ... b", b=self.BS)
        elif t.shape == (*sample_event_shape, *self.batch_shape):
            # Already correct shape
            pass
        else:
            raise ValueError(
                f"Time shape {t.shape} doesn't match sample_event_shape {sample_event_shape}, "
                f"[BS]={self.batch_shape}, or (*sample_event_shape, *batch_shape) = "
                f"{(*sample_event_shape, *self.batch_shape)}"
            )

        assert t.shape == (
            *sample_event_shape,
            *self.batch_shape,
        ), f"t shape {t.shape} must match sample_event_shape {sample_event_shape} and batch_shape {self.batch_shape}"

        # Flatten: [N1, N2, ..., Nk, BS] -> [N_total, BS]
        return t.reshape(-1, self.BS)

    def _get_gmm_t(self, t: torch.Tensor):
        """
        Get marginal GMM distribution at time t with proper batching support.

        Args:
            t: [N_total, BS] - flattened batch of time values

        Returns:
            MixtureSameFamily distribution for the marginal at time t
            with batch_shape=[N_total, BS], event_shape=[Dim]
        """
        assert t.dim() == 2, f"t must be a [N_total, BS] tensor, got {t.shape}"
        assert (
            t.shape[1] == self.batch_size
        ), f"Time batch size {t.shape[1]} must match GMM batch size {self.batch_size}"

        # Get schedule parameters: [N_total, BS]
        alpha_t, sigma_t = self.schedule.get_alpha_t_sigma_t(t)
        assert (
            alpha_t.shape == sigma_t.shape == t.shape
        ), f"alpha_t shape {alpha_t.shape} must match sigma_t shape {sigma_t.shape} and t shape {t.shape}"

        # Compute time-dependent parameters using einsum
        # mu_t: [N_total, BS, Components, Dim]
        mu_t = einops.einsum(alpha_t, self.mu, "n b, b k d -> n b k d")

        # var_t: [N_total, BS, Components, Dim]
        # Increasing variance term from noise
        increasing_var_t = einops.einsum(
            sigma_t**2, torch.ones(self.num_components, self.dim, device=self.mu.device), "n b, k d -> n b k d"
        )
        # Decreasing variance term from signal
        decreasing_var_t = einops.einsum(alpha_t, self.sigma, "n b, b k d -> n b k d") ** 2
        var_t = increasing_var_t + decreasing_var_t

        # Create covariance matrices: [N_total, BS, Components, Dim, Dim]
        covar_t = torch.diag_embed(var_t)

        # Create batched MultivariateNormal components
        # batch_shape=[N_total, BS, Components], event_shape=[Dim]
        component = MultivariateNormal(loc=mu_t, covariance_matrix=covar_t)

        # Create batched Categorical mixture
        # batch_shape=[N_total, BS], event_shape=[]
        batched_probs = einops.repeat(self.mix.probs, "b k -> n b k", n=t.shape[0])
        mix = Categorical(batched_probs)

        # Create batched MixtureSameFamily
        # batch_shape=[N_total, BS], event_shape=[Dim]
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
        # Extract dimension dim from [BS, k, d] -> [BS, k, 1]
        mu_marginal = self.mu[:, :, dim : dim + 1]  # [BS, k, 1]
        sigma_marginal = self.sigma[:, :, dim : dim + 1]  # [BS, k, 1]

        return TimeDependentGMM(mu_marginal, sigma_marginal, self.mix.probs, self.schedule)

    def drop_mode(self, component_index: int) -> "TimeDependentGMM":
        """
        Drop a component from all GMMs in the batch.
        """
        assert self.num_components > 1, "Cannot drop mode from a single component GMM"

        def _pop_batched(thing_to_pop, component_index):
            # Remove index component_index from dimension 1 (components) for all batches
            # thing_to_pop: [BS, k, ...]
            # Return: [BS, k-1, ...]
            indices = torch.arange(thing_to_pop.shape[1], device=thing_to_pop.device)
            mask = indices != component_index
            return thing_to_pop[:, mask, ...]

        mu_new = _pop_batched(self.mu, component_index)  # [BS, k-1, d]
        sigma_new = _pop_batched(self.sigma, component_index)  # [BS, k-1, d]
        probs_new = _pop_batched(self.mix.probs, component_index)  # [BS, k-1]

        # Renormalize weights for each GMM in batch
        probs_new = probs_new / probs_new.sum(dim=1, keepdim=True)  # [BS, k-1]

        return TimeDependentGMM(mu_new, sigma_new, probs_new, self.schedule)

    def log_prob(self, x: torch.Tensor, t: torch.Tensor | None = None, batched_data: bool = False) -> torch.Tensor:
        """
        Compute log probability.

        Args:
            x: [..., Dim] if batched_data=False, or [..., BS, Dim] if batched_data=True
            t: scalar, tensor matching x.shape[:-1] (unbatched), or
               scalar/[BS]/[...]/[...,BS] (batched), or None

        Returns:
            log_prob: [..., BS] - log probabilities with shape [*sample_event_shape, *batch_shape]
        """
        return self.__call__(x, t, batched_data=batched_data)

    def cdf(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        """
        Compute cumulative distribution function.

        Note: Currently only supports 1D case (d=1).

        Args:
            x: [..., BS, 1] - data points with arbitrary leading dimensions (only Dim=1 supported)
            t: scalar, tensor matching x.shape[:-2], [BS], or [..., BS], or None

        Returns:
            cdf: [..., BS] - CDF values with shape [*sample_event_shape, *batch_shape]
        """
        assert x.shape[-1] == self.dim == 1, f"CDF only supports 1D (Dim=1), got x.shape={x.shape}, self.dim={self.dim}"
        assert x.shape[-2] == self.BS, f"CDF expects batch dimension {self.BS}, got {x.shape[-2]}"

        # Extract sample_event_shape and flatten
        sample_event_shape = x.shape[:-2]
        x_flat = x.reshape(-1, self.BS, 1)  # [N_total, BS, 1]

        # Process time: returns [N_total, BS]
        t_flat = self._process_time(t, sample_event_shape)

        # Get time-dependent parameters using einsum
        alpha_t, sigma_t = self.schedule.get_alpha_t_sigma_t(t_flat)  # [N_total, BS]

        # mu_t: [N_total, BS, Components, 1]
        mu_t = einops.einsum(alpha_t, self.mu, "n b, b k d -> n b k d")

        # std_t: [N_total, BS, Components, 1]
        increasing_var_t = einops.einsum(
            sigma_t**2, torch.ones(self.num_components, 1, device=self.mu.device), "n b, k d -> n b k d"
        )
        decreasing_var_t = einops.einsum(alpha_t, self.sigma, "n b, b k d -> n b k d") ** 2
        var_t = increasing_var_t + decreasing_var_t
        std_t = torch.sqrt(var_t)

        # Expand x for broadcasting: [N_total, BS, 1] -> [N_total, BS, Components, 1]
        x_expanded = einops.repeat(x_flat, "n b d -> n b k d", k=self.num_components)

        # Compute component CDFs: [N_total, BS, Components]
        component_cdf = Normal(mu_t.squeeze(-1), std_t.squeeze(-1)).cdf(x_expanded.squeeze(-1))

        # Mix CDFs using einsum: [N_total, BS, Components] * [BS, Components] -> [N_total, BS]
        batched_probs = einops.repeat(self.mix.probs, "b k -> n b k", n=t_flat.shape[0])
        mix_cdf_flat = einops.einsum(component_cdf, batched_probs, "n b k, n b k -> n b")

        # Reshape to [*sample_event_shape, *batch_shape]
        mix_cdf = mix_cdf_flat.reshape(*sample_event_shape, *self.batch_shape)

        return mix_cdf

    def energy(self, x: torch.Tensor, t: torch.Tensor | None = None, batched_data: bool = False) -> torch.Tensor:
        """
        Compute energy in the form of -log_prob.

        Args:
            x: [..., Dim] if batched_data=False, or [..., BS, Dim] if batched_data=True
            t: scalar, tensor matching x.shape[:-1] (unbatched), or
               scalar/[BS]/[...]/[...,BS] (batched), or None

        Returns:
            energy: [..., BS] - energy with shape [*sample_event_shape, *batch_shape]
        """
        return -self.__call__(x, t, batched_data=batched_data)

    @torch.enable_grad()
    def score(self, x: torch.Tensor, t: torch.Tensor | None = None, batched_data: bool = False) -> torch.Tensor:
        """
        Compute score function using autograd: ∇_x log p(x)

        Args:
            x: [..., Dim] if batched_data=False, or [..., BS, Dim] if batched_data=True
            t: scalar, tensor matching x.shape[:-1] (unbatched), or
               scalar/[BS]/[...]/[...,BS] (batched), or None

        Returns:
            score: [..., BS, Dim] - score with shape [*sample_event_shape, *batch_shape, *event_shape]
        """
        # Validate input has correct event dimension
        assert x.shape[-1] == self.dim, f"x must have last dimension {self.dim}, got {x.shape[-1]}"

        if batched_data:
            self.check_is_batched(x)
            sample_event_shape = x.shape[:-2]
            x_batched = x
        else:
            sample_event_shape = x.shape[:-1]
            x_batched = einops.repeat(x, "... d -> ... b d", b=self.BS)

        # Flatten to [N_total, BS, Dim]
        x_flat = x_batched.reshape(-1, self.BS, self.dim).requires_grad_(True)

        # Process time: returns [N_total, BS]
        t_flat = self._process_time(t, sample_event_shape)

        # Get batched GMM distribution with batch_shape=[N_total, BS]
        gmm_t = self._get_gmm_t(t_flat)

        # Compute log probability: [N_total, BS]
        log_prob_flat = gmm_t.log_prob(x_flat)

        # Compute gradient: [N_total, BS, Dim]
        score_flat = torch.autograd.grad(log_prob_flat.sum(), x_flat, create_graph=False)[0]

        # Reshape to [*sample_event_shape, *batch_shape, *event_shape]
        score = score_flat.reshape(*sample_event_shape, *self.batch_shape, *self.event_shape)

        return score

    def sample(self, shape: tuple | int | None = None, t: torch.Tensor | float | None = None) -> torch.Tensor:
        """
        Sample from the batched GMMs at time t.

        Args:
            shape: tuple, int, or None - shape of sample events to generate
                   - None: no sample_event_shape, returns [BS, Dim]
                   - int: sample_event_shape=(shape,), returns [shape, BS, Dim]
                   - tuple: sample_event_shape=shape, returns [*shape, BS, Dim]
            t: scalar, tensor matching shape, [BS], or [*shape, BS], or None
               - None: assume t=0 for all
               - scalar: same time for all sample events
               - tensor: must match sample_event_shape

        Returns:
            samples: [..., BS, Dim] - samples with shape [*sample_event_shape, *batch_shape, *event_shape]
        """
        # Determine sample_event_shape
        if shape is None and (t is None or not isinstance(t, torch.Tensor) or t.dim() == 0):
            # No shape, scalar/None time -> single sample per GMM
            sample_event_shape = ()
        elif shape is None and isinstance(t, torch.Tensor) and t.dim() > 0:
            # Infer from time tensor
            if t.shape == (self.BS,):
                sample_event_shape = ()
            elif t.shape[-1] == self.BS:
                sample_event_shape = t.shape[:-1]
            else:
                sample_event_shape = t.shape
        elif isinstance(shape, int):
            sample_event_shape = (shape,)
        elif isinstance(shape, tuple):
            sample_event_shape = shape
        else:
            raise ValueError(f"Invalid shape: {shape}")

        # If both shape and tensor t provided, verify compatibility
        if shape is not None and isinstance(t, torch.Tensor) and t.dim() > 0:
            valid_shapes = {sample_event_shape, (self.BS,), (*sample_event_shape, self.BS)}
            assert (
                t.shape in valid_shapes
            ), f"Time shape {t.shape} must match sample_event_shape {sample_event_shape}, [BS]={self.BS}, or (*sample_event_shape, BS)"

        # Process time: returns [N_total, BS]
        t_flat = self._process_time(t, sample_event_shape)

        # Get batched GMM distribution with batch_shape=[N_total, BS]
        gmm_t = self._get_gmm_t(t_flat)

        # Sample: [N_total, BS, Dim]
        samples_flat = gmm_t.sample()

        # Reshape to [*sample_event_shape, *batch_shape, *event_shape]
        samples = samples_flat.reshape(*sample_event_shape, *self.batch_shape, *self.event_shape)

        return samples

    def __repr__(self):
        return f"TimeDependentGMM(mu={self.mu.shape}, sigma={self.sigma.shape}, weight={self.weight.shape})"


class Conditional(TimeDependentGMM):
    """
    Conditional Process class
    Instead of simulating the full GMM, we only simulate the conditional process conditioned on the initial value x0.
    This is useful for conditional sampling and inference.

    Args:
        x0: [BS, d] - initial value
        schedule: BetaSchedule - schedule for the conditional process

    Returns:
        Conditional - Conditional process
    """

    def __init__(self, x0: torch.Tensor, schedule: BetaSchedule = None):
        mu = x0.unsqueeze(-2)  # [BS, d] -> [BS, 1, d]
        assert mu.dim() == x0.dim() + 1, f"mu must be a 3D tensor [..., 1, d], got {mu.shape}"
        sigma = torch.zeros_like(mu) + 1e-10
        weight = torch.ones((mu.shape[0], 1), device=mu.device, dtype=mu.dtype)
        super().__init__(mu, sigma, weight, schedule)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    for _ in range(3):
        num_components = 5
        # Create batched GMM with BS=1 (single GMM)
        mu = torch.ones(1, num_components, 2).uniform_(-3, 3)  # [BS=1, k=5, d=2]
        sigma = torch.ones(1, num_components, 2).uniform_(0.5, 1.2)  # [BS=1, k=5, d=2]
        weight = torch.ones(1, num_components).uniform_(0.3, 1.0)  # [BS=1, k=5]
        gmm = TimeDependentGMM(mu, sigma, weight)
        samples = gmm.sample(1_000_000)  # [BS=1, n_samples=1_000_000, d=2]
        # Extract samples for plotting: [1_000_000, 2]
        samples_plot = samples[0]  # [1_000_000, 2]
        plt.hist2d(samples_plot[:, 0], samples_plot[:, 1], bins=100)
        plt.grid()
        plt.show()

    # from mcmc.metropolis_hasting_nd import batch_mh, batch_langevin

    def sample_fn(x):
        return gmm.energy(x), x

    init_samples = torch.randn(500, 2)
