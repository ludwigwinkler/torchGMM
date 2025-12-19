import torch
import einops
import numbers
from torch.distributions import Independent,MultivariateNormal, Normal, MixtureSameFamily, Categorical
from torchGMM.schedule import BetaSchedule

LENGTH_SCALE = 5.0
ENERGY_SCALE = 0.1  # 0.2


class FirstDimension(torch.nn.Module):
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        return positions[..., 0:1]


class TimeDependentGMM(torch.nn.Module):
    def __init__(self, mu: torch.Tensor, sigma: torch.Tensor=None, weight: torch.Tensor=None, schedule: BetaSchedule = None):
        super().__init__()
        """
        Args:
            mu: [BS, k, d] - means for BS batched GMMs, each with k components and d dimensions
            sigma: [BS, k, d] - standard deviations for BS batched GMMs, each with k components and d dimensions  
            weight: [BS, k] - mixture weights for BS batched GMMs, each with k components
        """
        assert (mu is not None and sigma is not None and weight is not None) or (mu is not None), "Either mu, sigma, and weight must be provided or mu only"
        if mu is not None and sigma is None and weight is None:
            sigma = torch.zeros_like(mu) + 1e-10
            weight = torch.ones((mu.shape[0],1), device=mu.device, dtype=mu.dtype)
        assert mu.dim() == 3, f"mu must be a 3D tensor [BS, k, d], got {mu.shape}"
        assert sigma.dim() == 3, f"sigma must be a 3D tensor [BS, k, d], got {sigma.shape}"
        assert weight.dim() == 2, f"weight must be a 2D tensor [BS, k], got {weight.shape}"
        assert (
            mu.shape[0] == sigma.shape[0] == weight.shape[0]
        ), f"Batch size must match: mu {mu.shape[0]}, sigma {sigma.shape[0]}, weight {weight.shape[0]}"
        assert (
            mu.shape[1] == sigma.shape[1] == weight.shape[1]
        ), f"Number of components must match: mu {mu.shape[1]}, sigma {sigma.shape[1]}, weight {weight.shape[1]}"
        assert (
            mu.shape[2] == sigma.shape[2]
        ), f"Dimension must match: mu {mu.shape[2]}, sigma {sigma.shape[2]}"

        
        self.register_buffer("mu", mu)
        self.register_buffer("sigma", sigma)
        
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

        self.batch_size = mu.shape[0]
        self.BS = mu.shape[0]
        self.num_components = mu.shape[1]
        self.dim = mu.shape[2]

        self.schedule = BetaSchedule(beta_min=0.1, beta_max=20.0) if not schedule else schedule
        self.schedule.to(mu.device)

    def __call__(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        """
        Compute log probability.

        Args:
            x: [N, d] or [BS, N, d] - batch of data points
               - [N, d]: same samples evaluated in each of the [BS] GMMs
               - [BS, N, d]: each GMM batch evaluates its own samples
            t: scalar, [N], [BS, N], or None - time value(s)
               - None: assume t=0 for all
               - scalar: same time for all (GMM, sample) pairs
               - [N]: different time per sample, broadcast to all GMMs → [BS, N]
               - [BS, N]: different time for each (GMM, sample) pair

        Returns:
            log_prob: [BS, N] - log probabilities for each GMM and each data point
        """
        if x.dim() == 1:
            assert x.shape[0] == self.dim, f"x must have dimension {self.dim}, got {x.shape[0]}"
            N = 1
            x = einops.repeat(x, 'd -> N B d', B=self.batch_size, N=N)
        elif x.dim() == 2:
            assert x.shape[1] == self.dim, f"x must have dimension {self.dim}, got {x.shape[1]}"
            N = x.shape[0]
            x = einops.repeat(x, 'N d -> N B d', B=self.batch_size)  # [BS, N, d]
        elif x.dim() == 3:
            assert x.shape[1] == self.batch_size, f"x batch size {x.shape[0]} must match GMM batch size {self.batch_size}"
            assert x.shape[2] == self.dim, f"x must have dimension {self.dim}, got {x.shape[2]}"
            N = x.shape[0]
            x = x  # [BS, N, d]

        assert x.dim() == 3, f"x must be a 3D tensor [BS, N, d], got {x.shape}"
        # Process time for GMM batch dimension
        t = self._process_time(t, x.shape[:2])  # [BS, N]
        
        # Get batched GMM distribution: batch_shape=[BS_gmm], event_shape=[d]
        gmm_t = self._get_gmm_t(t)
        
        # GMM: [BS, k, d]
        # Data: [BS, N, d]
        # Transpose N dimension to the first position [N, BS, d]
        # N is implicit batch dimension, BS is explicit batch dimension
        assert x.shape[1] == self.batch_size, f"x batch size {x_expanded.shape[0]} must match GMM batch size {self.batch_size}"
        log_prob = gmm_t.log_prob(x) # [N, BS, d] -> [BS, N]
        assert log_prob.shape == (N, self.batch_size), f"log_prob shape {log_prob.shape} must match (BS, N) = ({self.batch_size}, {N})"
        return log_prob  # [BS, N]

    def _process_time(self, t: torch.Tensor | numbers.Number | None, batch_size: tuple) -> torch.Tensor:
        """
        Process time input to ensure it's in the correct format [BS_gmm].

        Args:
            t: scalar, [BS_gmm], or None - time value(s) for each GMM in batch
            batch_size: GMM batch size (BS_gmm)

        Returns:
            t: [BS_gmm] - processed time tensor
        """
        assert type(batch_size) in [tuple, torch.Size], f"batch_size must be a tuple, got {type(batch_size)}"
        if t is None:
            # Default to t=0 for all GMMs in batch
            return torch.zeros(batch_size, device=self.mu.device)
        elif not isinstance(t, torch.Tensor):
            # Convert float/int to tensor scalar
            t = torch.tensor(t, device=self.mu.device)

        if t.dim() == 0:
            # Scalar -> expand to GMM batch size
            return einops.repeat(t, "-> n b", n=batch_size[0], b=batch_size[1])
        elif t.dim() == 1:
            # [N] -> [N, BS]
            return einops.repeat(t, "n -> n b", b=batch_size[1])
        else:
            # Already [N, BS]
            assert t.shape[0] == batch_size[0], f"Time batch size {t.shape[0]} must match GMM batch size {batch_size[0]}"
            assert t.shape[1] == batch_size[1], f"Time batch size {t.shape[1]} must match GMM batch size {batch_size[1]}"
            return t

    def _get_gmm_t(self, t: torch.Tensor):
        """
        Get marginal GMM distribution at time t with proper batching support.

        Args:
            t: [BS_gmm] - batch of time values for each GMM

        Returns:
            MixtureSameFamily distribution for the marginal at time t with batch_shape=[BS_gmm], event_shape=[d]
        """
        assert t.dim() == 2, f"t must be a [BS_gmm, N] tensor, got {t.shape}"
        assert t.shape[1] == self.batch_size, f"Time batch size {t.shape[0]} must match GMM batch size {self.batch_size}"

        alpha_t, sigma_t = self.schedule.get_alpha_t_sigma_t(t)  # [BS_gmm, N]
        assert alpha_t.shape == sigma_t.shape == t.shape, f"alpha_t shape {alpha_t.shape} must match sigma_t shape {sigma_t.shape} and t shape {t.shape}"

        # Compute time-dependent parameters
        # alpha_t: [BS_gmm], self.mu: [BS_gmm, k, d] -> mu_t: [BS_gmm, k, d]
        # Element-wise multiplication: alpha_t[b] * self.mu[b, k, d] for each b
        # mu_t = alpha_t * self.mu  # [N, BS] x [BS_gmm, k, d] -> [N, BS, k, d]
        mu_t = einops.einsum(alpha_t, self.mu, "n b, b k d -> n b k d")
        
        # var_t: [BS_gmm, k, d]
        increasing_var_t = einops.repeat(sigma_t ** 2, "n b -> n b k d", k=self.num_components, d=self.dim) # [N, BS, k, d]
        decreasing_var_t = einops.einsum(alpha_t, self.sigma, "n b, b k d -> n b k d") ** 2 # [N, BS, k, d]
        # var_t = sigma_t.unsqueeze(-1).unsqueeze(-1) ** 2 + (
        #     alpha_t.unsqueeze(-1).unsqueeze(-1) * self.sigma
        # ) ** 2  # [BS_gmm, k, d]
        var_t = increasing_var_t + decreasing_var_t
        
        # Create covariance matrices: [BS_gmm, k, d, d]
        covar_t = torch.diag_embed(var_t)  # [n, BS_gmm, k, d, d]

        # Create batched MultivariateNormal components
        # batch_shape=[BS_gmm, k], event_shape=[d]
        component = MultivariateNormal(loc=mu_t, covariance_matrix=covar_t)

        # Create batched Categorical mixture
        batched_probs = einops.repeat(self.mix.probs, "b k -> n b k", n=t.shape[0])
        mix = Categorical(batched_probs)  # batch_shape=[BS_gmm], event_shape=[]

        # Create batched MixtureSameFamily
        # batch_shape=[BS_gmm], event_shape=[d]
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
        mu_marginal = self.mu[:, :, dim:dim+1]  # [BS, k, 1]
        sigma_marginal = self.sigma[:, :, dim:dim+1]  # [BS, k, 1]
        
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


    def log_prob(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        """
        Compute log probability.

        Args:
            x: [N, d] - batch of data points
            t: scalar, [BS_gmm], or None - time value(s) for each GMM in batch

        Returns:
            log_prob: [BS_gmm, N] - log probabilities for each GMM and each data point
        """
        return self.__call__(x, t)

    def cdf(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        """
        Compute cumulative distribution function.
        
        Note: Currently only supports 1D case (d=1).

        Args:
            x: [N, 1] - batch of data points (only d=1 supported)
            t: scalar, [BS_gmm], or None - time value(s) for each GMM in batch

        Returns:
            cdf: [BS_gmm, N] - CDF values for each GMM and each data point
        """
        assert (
            x.dim() == 2 and x.shape[1] == self.dim == 1
        ), f"x must be a 2D tensor [N, 1] = [*,{self.dim}] for the moment, got {x.shape}"

        # Process time for GMM batch dimension
        t = self._process_time(t, self.batch_size)  # [BS_gmm]
        
        # Get time-dependent parameters
        alpha_t, sigma_t = self.schedule.get_alpha_t_sigma_t(t)  # [BS_gmm]
        mu_t = alpha_t.unsqueeze(-1).unsqueeze(-1) * self.mu  # [BS_gmm, k, 1]
        var_t = sigma_t.unsqueeze(-1).unsqueeze(-1) ** 2 + (
            alpha_t.unsqueeze(-1).unsqueeze(-1) * self.sigma
        ) ** 2  # [BS_gmm, k, 1]
        std_t = torch.sqrt(var_t)  # [BS_gmm, k, 1]
        
        # Expand x to [BS_gmm, N, 1] for broadcasting
        x_expanded = x.unsqueeze(0).expand(self.batch_size, -1, -1)  # [BS_gmm, N, 1]
        
        # Compute component CDFs: [BS_gmm, k, N, 1]
        # mu_t: [BS_gmm, k, 1], std_t: [BS_gmm, k, 1], x_expanded: [BS_gmm, N, 1]
        # We need to broadcast: [BS_gmm, k, 1] and [BS_gmm, N, 1] -> [BS_gmm, k, N, 1]
        mu_t_expanded = mu_t.unsqueeze(2)  # [BS_gmm, k, 1, 1]
        std_t_expanded = std_t.unsqueeze(2)  # [BS_gmm, k, 1, 1]
        x_for_cdf = x_expanded.unsqueeze(1)  # [BS_gmm, 1, N, 1]
        
        component_cdf = Normal(mu_t_expanded.squeeze(-1), std_t_expanded.squeeze(-1)).cdf(x_for_cdf.squeeze(-1))  # [BS_gmm, k, N]
        
        # Mix CDFs: [BS_gmm, k, N] * [BS_gmm, k] -> [BS_gmm, N]
        mix_cdf = torch.einsum("bkn,bk->bn", component_cdf, self.mix.probs)  # [BS_gmm, N]
        
        return mix_cdf

    def energy(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        """
        Compute energy in the form of -log_prob.

        Args:
            x: [N, d] - batch of data points
            t: scalar, [BS_gmm], or None - time value(s) for each GMM in batch

        Returns:
            energy: [BS_gmm, N] - energy for each GMM and each data point
        """
        return -self.__call__(x, t)  # [BS_gmm, N]

    @torch.enable_grad()
    def score(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        """
        Compute score function using autograd: ∇_x log p(x)

        Args:
            x: [N, d] or [BS, N, d] - batch of data points
               - [N, d]: same samples evaluated in each of the [BS] GMMs
               - [BS, N, d]: each GMM batch evaluates its own samples
            t: scalar, [N], [BS, N], or None - time value(s)
               - None: assume t=0 for all
               - scalar: same time for all (GMM, sample) pairs
               - [N]: different time per sample, broadcast to all GMMs → [BS, N]
               - [BS, N]: different time for each (GMM, sample) pair

        Returns:
            score: [BS, N, d] - score for each GMM and each data point
        """

        if x.dim() == 1:
            assert x.shape[0] == self.dim, f"x must have dimension {self.dim}, got {x.shape[0]}"
            N = 1
            x = einops.repeat(x, 'd -> N B d', B=self.batch_size, N=N)
        elif x.dim() == 2:
            assert x.shape[1] == self.dim, f"x must have dimension {self.dim}, got {x.shape[1]}"
            N = x.shape[0]
            x = einops.repeat(x, 'N d -> N B d', B=self.batch_size)  # [BS, N, d]
        elif x.dim() == 3:
            assert x.shape[1] == self.batch_size, f"x batch size {x.shape[0]} must match GMM batch size {self.batch_size}"
            assert x.shape[2] == self.dim, f"x must have dimension {self.dim}, got {x.shape[2]}"
            N = x.shape[0]
            x = x  # [BS, N, d]
        
        assert x.dim() == 3, f"x must be a 3D tensor [BS, N, d], got {x.shape}"
        # Process time for GMM batch dimension
        x = x.requires_grad_(True)
        t = self._process_time(t, x.shape[:2])  # [BS, N]

        gmm_t = self._get_gmm_t(t)
        
        # Evaluate each GMM separately
        log_prob = gmm_t.log_prob(x)  # [M]
        score = torch.autograd.grad(log_prob.sum(), x, create_graph=False)[0]  # [M, d]
        
        return score  # [BS, N, d]

    def sample(self, n_samples: int | tuple, t: torch.Tensor | None = None) -> torch.Tensor:
        """
        Sample from the batched GMMs at time t.

        Args:
            n_samples: int or tuple - number of samples to generate per GMM in batch
            t: scalar, [BS_gmm], [n_samples], or None - time value(s)
               - None: assume t=0 for all GMMs
               - scalar: same time for all GMMs
               - [BS_gmm]: time for each GMM
               - [n_samples]: if n_samples matches, use first value for all GMMs

        Returns:
            samples: [BS_gmm, n_samples, d] - generated samples for each GMM in batch
        """
        # Handle both int and tuple inputs for n_samples
        if isinstance(n_samples, int):
            batch_size= (n_samples, self.batch_size,)
        elif isinstance(n_samples, tuple):
            assert len(n_samples) == 2, f"n_samples must be a tuple of length 2, got {len(n_samples)}"
            assert n_samples[1] == self.batch_size, f"n_samples[1] must be {self.batch_size}, got {n_samples[1]}"
            batch_size= n_samples
        
        t = self._process_time(t, batch_size)  # [BS_gmm]
        
        # Get batched GMM distribution: batch_shape=[BS_gmm], event_shape=[d]
        gmm_t = self._get_gmm_t(t)
        
        # Sample from each GMM in batch
        # The distribution has batch_shape=[BS_gmm], so sample() returns [BS_gmm, d]
        # We need [BS_gmm, n_samples, d], so we sample n_samples times
        samples = gmm_t.sample()  # [BS_gmm, n_samples, d]
        
        
        return samples


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
    # samples, _, _ = batch_langevin(sample_fn, init_samples, n_steps=5_000, step_size=0.1, burn_in=200)
    # plt.hist2d(samples[:, 0], samples[:, 1], bins=100)
    # plt.grid()
    # plt.show()
    #
    # init_samples = torch.randn(500, 2)
    # samples, _, _ = batch_mh(sample_fn, init_samples, n_steps=5_000, step_sigma=0.1, burn_in=200)
    # plt.hist2d(samples[:, 0], samples[:, 1], bins=100)
    # plt.grid()
    # plt.show()
