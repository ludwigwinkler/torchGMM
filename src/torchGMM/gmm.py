from typing import Tuple

import torch
import einops
from torch.distributions import Normal, MultivariateNormal, MixtureSameFamily, Independent, Categorical, MultivariateNormal
from torchGMM.schedule import BetaSchedule

LENGTH_SCALE = 5.0
ENERGY_SCALE = 0.1  # 0.2


class FirstDimension(torch.nn.Module):
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        return positions[..., 0:1]


class TimeDependentGMM(torch.nn.Module):
    def __init__(self, mu: torch.Tensor, sigma: torch.Tensor = None, weight: torch.Tensor = None, schedule: BetaSchedule = None):
        super().__init__()
        """
        Args:
            mu: [k, d] - means for k components with d dimensions each
            sigma: [k, d] - standard deviations for k components with d dimensions each  
            weight: [k] - mixture weights for k components
        """
        self.is_conditional = weight is None and sigma is None

        if not self.is_conditional:
            # --- Mixture Model Initialization ---
            assert mu.dim() == 2, f"mu must be a 2D tensor [k, d], got {mu.shape}"
            assert sigma.dim() == 2, f"sigma must be a 2D tensor [k, d], got {sigma.shape}"
            assert weight.dim() == 1, f"weight must be a 1D tensor [k], got {weight.shape}"
            assert (
                mu.shape[0] == sigma.shape[0] == weight.shape[0]
            ), f"Number of components must match: mu {mu.shape[0]}, sigma {sigma.shape[0]}, weight {weight.shape[0]}"

            self.register_buffer("mu", mu)
            self.register_buffer("sigma", sigma)
            self.mix = Categorical(weight / weight.sum())
            self.comp = Independent(Normal(mu, sigma), 1)
            self.gmm = MixtureSameFamily(self.mix, self.comp)
        else:
            # --- Conditional Model Initialization ---
            assert mu.dim() == 2, f"For conditional model, mu must be a 2D tensor [BS, D], got {mu.shape}"
            sigma = torch.zeros_like(mu) + 1e-5
            # assert mu.shape == sigma.shape, f"For conditional model, mu and sigma must have the same shape"
            # assert torch.all(sigma == 0), "For conditional model, sigma must be a tensor of zeros."

            self.register_buffer("mu", mu)
            self.register_buffer("sigma", sigma) # still save sigma, for consistency
            self.mix = None
            self.comp = None
            self.gmm = None

        self.num_components = mu.shape[0]
        self.dim = mu.shape[1]

        self.schedule = BetaSchedule(beta_min=0.1, beta_max=20.0) if not schedule else schedule
        self.schedule.to(mu.device)

    def _process_time(self, t: torch.Tensor | None, batch_size: int) -> torch.Tensor:
        """
        Process time input to ensure it's in the correct format [BS].

        Args:
            t: scalar, [BS], or None - time value(s)
            batch_size: number of samples in batch

        Returns:
            t: [BS] - processed time tensor
        """
        if t is None:
            # Default to t=0
            return torch.zeros((batch_size,), device=self.mu.device)
        elif not isinstance(t, torch.Tensor):
            # Convert float/int to tensor scalar
            t = torch.tensor(t, device=self.mu.device)

        if t.dim() == 0:
            # Scalar -> expand to batch
            return t.unsqueeze(0).expand(batch_size)
        else:
            # Already [BS]
            return t

    def _get_gmm_t(self, t: torch.Tensor):
        """
        Get marginal GMM distribution at time t with proper batching support.

        Args:
            t: [BS] - batch of time values

        Returns:
            MixtureSameFamily or Independent(Normal) distribution for the marginal at time t
        """
        assert t.dim() == 1, f"t must be a [BS] tensor, got {t.shape}"

        batch_size = t.shape[0]
        alpha_t, sigma_t = self.schedule.get_alpha_t_sigma_t(t)  # [BS]

        if not self.is_conditional:
            # --- Mixture Model Logic ---
            # Broadcast: [BS, 1, 1] * [k, d] -> [BS, k, d]
            mu_t = einops.einsum(alpha_t, self.mu, "b, m ... -> b m ...")  # [BS, k, d]
            var_t = sigma_t.reshape(-1, 1, 1) ** 2 + (
                einops.einsum(alpha_t, self.sigma, "b, m ... -> b m ...") ** 2
            )  # [BS, k, d]
            std_t = torch.sqrt(var_t)

            # Create batched MixtureSameFamily distribution
            mix_probs = self.mix.probs.unsqueeze(0).expand(batch_size, -1).to(t.device)
            mix = Categorical(mix_probs)  # batch_shape=[BS]
            component = Independent(Normal(mu_t, std_t), 1)  # batch_shape=[BS, k], event_shape=[d]

            return MixtureSameFamily(mix, component)
        else:
            # --- Conditional Model Logic ---
            # self.mu is [C, D], alpha_t is [B] -> mu_t [B, C, D]
            mu_t = einops.einsum(alpha_t, self.mu, "b, c d -> b c d")

            # self.sigma is small, so variance is mostly sigma_t^2
            # var_t is [B, C, D]
            var_t = einops.einsum(sigma_t.pow(2), torch.ones_like(self.mu), "b, c d -> b c d") + einops.einsum(alpha_t, self.sigma, "b, c d -> b c d").pow(2)
            
            return MultivariateNormal(loc=mu_t, covariance_matrix=torch.diag_embed(var_t))

    def marginal_gmm(self, dim) -> "TimeDependentGMM":
        """
        Get marginal GMM distribution at time t with proper batching support.

        Args:
            dim: int - dimension to marginalize out

        Returns:
            MixtureSameFamily distribution with a single dimension for the marginal GMM at time t
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
        if self.is_conditional:
            raise NotImplementedError("marginal_gmm is not applicable to a conditional GMM.")

        return TimeDependentGMM(self.mu[:, dim].unsqueeze(-1), self.sigma[:, dim].unsqueeze(-1), self.mix.probs)

    def drop_mode(self, component_index: int) -> "TimeDependentGMM":
        """
        Drop a component from the GMM.
        """
        if self.is_conditional:
            raise NotImplementedError("drop_mode is not applicable to a conditional GMM.")
        assert self.mu.shape[0] > 1, "Cannot drop mode from a single component GMM"

        def pop(thing_to_pop, component_index):
            # Remove index component_index from 0th dimension (convert to list for arbitrary shape)
            items = [x for x in thing_to_pop]
            items.pop(component_index)
            return torch.stack(items, dim=0)

        mu_new = pop(self.mu, component_index)
        sigma_new = pop(self.sigma, component_index)
        probs_new = pop(self.mix.probs, component_index)
        probs_new = probs_new / probs_new.sum()  # renormalize
        return TimeDependentGMM(mu_new, sigma_new, probs_new)

    def __call__(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        """
        Compute log probability.

        Args:
            x: [BS, D] - batch of samples
            t: scalar, [BS], or None - time value(s)
               - None: assume t=0 for all samples
               - scalar: same time for all samples
               - [BS]: time for each sample

        Returns:
            log_prob: [BS] - log probabilities
        """
        t = self._process_time(t, x.shape[0])
        assert x.dim() == 2 and x.shape[1] == self.dim, f"x must be a 2D tensor [BS, D] = [*,{self.dim}], got {x.shape}"
        
        if self.is_conditional:
            dist = self._get_gmm_t(t) # MultivariateNormal with batch_shape [B, C]
            log_prob = dist.log_prob(x.unsqueeze(1))
            return log_prob.transpose(0, 1) # [C, B]

        gmm_t = self._get_gmm_t(t)
        return gmm_t.log_prob(x)

    def log_prob(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        """
        Compute log probability.
        """
        assert x.dim() == 2 and x.shape[1] == self.dim, f"x must be a 2D tensor [BS, D] = [*,{self.dim}], got {x.shape}"
        return self.__call__(x, t)

    def cdf(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        """
        Compute cumulative distribution function.
        """
        assert (
            x.dim() == 2 and x.shape[1] == self.dim == 1
        ), f"x must be a 2D tensor [BS, 1] = [*,{self.dim}] for the moment, got {x.shape}"

        t = self._process_time(t, x.shape[0])
        gmm_t = self._get_gmm_t(t)
        assert x.dim() == 2 and x.shape[1] == self.dim, f"x must be a 2D tensor [BS, D] = [*,{self.dim}], got {x.shape}"
        component_cdf = Normal(self.mu.unsqueeze(1), self.sigma.unsqueeze(1)).cdf(x)

        mix_cdf = torch.sum(component_cdf * self.mix.probs, dim=1)
        mix_cdf = torch.einsum("kbd,k->bd", component_cdf, self.mix.probs)
        return mix_cdf

    def energy(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        """
        Compute energy in the form of -log_prob.

        Args:
            x: [BS, D] - batch of samples
            t: scalar, [BS], or None - time value(s)
        """
        return -self.__call__(x, t)

    @torch.enable_grad()
    def score(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        """
        Compute score function using autograd: ∇_x log p(x)

        Args:
            x: [BS, D] - batch of samples
            t: scalar, [BS], or None - time value(s)
        """
        x.requires_grad_(True)

        if self.is_conditional:
            t = self._process_time(t, x.shape[0])
            dist = self._get_gmm_t(t)
            
            x_usq = x.unsqueeze(1) # [B, 1, D]
            
            precision = torch.linalg.inv(dist.covariance_matrix)
            score = torch.einsum("bcd,bcde->bce", -(x_usq - dist.loc), precision)
            
            return score.transpose(0, 1) # [C, B, D]

        log_prob = self.__call__(x, t)
        grad = torch.autograd.grad(log_prob.sum(), x)[0]
        return grad

    def sample(self, n_samples: int | tuple, t: torch.Tensor | None = None) -> torch.Tensor:
        """
        Sample from the GMM at time t.

        Args:
            n_samples: int or tuple - number of samples to generate
            t: scalar, [BS], or None - time value(s)

        Returns:
            samples: [BS, D] - generated samples
        """
        # Handle both int and tuple inputs for n_samples
        if isinstance(n_samples, int):
            batch_size = n_samples
        else:
            # n_samples is a tuple like (1000,)
            batch_size = n_samples[0] if len(n_samples) > 0 else 1

        # Process time input and get GMM distribution
        t = self._process_time(t, batch_size)
        gmm_t = self._get_gmm_t(t)

        return gmm_t.sample()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    for _ in range(3):
        num_components = 5
        mu = torch.ones(num_components, 2).uniform_(-3, 3)
        sigma = torch.ones(num_components, 2).uniform_(0.5, 1.2)
        weight = torch.ones(num_components).uniform_(0.3, 1.0)
        gmm = TimeDependentGMM(mu, sigma, weight)
        samples = gmm.sample((1_000_000,))
        plt.hist2d(samples[:, 0], samples[:, 1], bins=100)
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
