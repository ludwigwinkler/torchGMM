import torch
import einops
from torchGMM.schedule import BetaSchedule

class TimeDependentConditional(torch.nn.Module):

    def __init__(self, mu: torch.Tensor, sigma: torch.Tensor= None, schedule: BetaSchedule = None):
        super().__init__()
        assert mu.dim() == 2, f"For conditional model, mu must be a 2D tensor [BS, D], got {mu.shape}"
        self.mu = mu
        self.sigma = torch.zeros_like(mu) + 1e-5 if sigma is None else sigma
        self.schedule = BetaSchedule(beta_min=0.1, beta_max=20.0) if not schedule else schedule
        self.dim = mu.shape[1]

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

    def _get_dist_t(self, t: torch.Tensor):
        """
        Get marginal GMM distribution at time t with proper batching support.

        Args:
            t: [BS] - batch of time values

        Returns:
            MixtureSameFamily distribution for the marginal at time t
        """
        assert t.dim() == 1, f"t must be a [BS] tensor, got {t.shape}"

        batch_size = t.shape[0]
        alpha_t, sigma_t = self.schedule.get_alpha_t_sigma_t(t)  # [BS]

        # Broadcast: [BS, 1, 1] * [k, d] -> [BS, k, d]
        mu_t = einops.einsum(alpha_t, self.mu, "b, m ... -> b m ...")  # [BS, k, d]
        var_t = sigma_t.reshape(-1, 1, 1) ** 2 + (
            einops.einsum(alpha_t, self.sigma, "b, m ... -> b m ...") ** 2
        )  # [BS, k, d]


        
        covariance_matrix=torch.diag_embed(var_t)

        dist = torch.distributions.MultivariateNormal(loc=mu_t, covariance_matrix=covariance_matrix)

        return dist

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
        
        gmm_t = self._get_dist_t(t)
        return gmm_t.log_prob(x)
        