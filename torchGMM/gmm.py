"""
GMM: Gaussian Mixture Model with time-dependent diffusion schedule.
Params [*B, K, D]. Two shapes only:
- Batch shape B (from init). Sample shape N (optional leading dims).
- x: [*B, D] or [*N, *B, D]. t: [*B] or [*N, *B] (scalar per batch; no event dim).
- Outputs: log_prob/energy -> [*N, *B], score -> [*N, *B, D], sample(shape, t) -> [*N, *B, D].
"""

import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor
from torch.distributions import Categorical, MixtureSameFamily, MultivariateNormal

from torchGMM.schedule import BetaSchedule, LinearSchedule, Schedule


def validate_positive_tensor(name: str, tensor: torch.Tensor) -> None:
    if not torch.all(torch.isfinite(tensor)):
        raise ValueError(f"{name} must be finite")
    if not torch.all(tensor > 0):
        raise ValueError(f"{name} must be > 0")


def validate_time_range(t: int | float | torch.Tensor) -> None:
    if isinstance(t, torch.Tensor):
        # tensor path: check element-wise so the full batch is validated at once
        if not torch.all(torch.isfinite(t)):
            raise ValueError("t must be finite")
        if not torch.all((t >= 0) & (t <= 1)):
            raise ValueError("t must be within [0, 1]")
    else:
        # scalar path: plain Python comparison suffices
        if not (0 <= t <= 1):
            raise ValueError("t must be within [0, 1]")


class GMM(torch.nn.Module):
    # register_buffer stores tensors in nn.Module._buffers; these annotations expose them to type checkers.
    mu: torch.Tensor
    sigma: torch.Tensor
    weight: torch.Tensor
    batch_shape: torch.Size
    batch_ndim: int
    num_components: int
    dim: int
    event_shape: tuple[int, ...]
    schedule: Schedule

    def __init__(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor | None = None,
        weight: torch.Tensor | None = None,
        schedule: Schedule | None = None,
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
        mu = torch.as_tensor(mu)
        assert mu.dim() >= 2, f"mu must be at least 2D [*, k, d], got {mu.shape}"
        sigma = (
            torch.zeros_like(mu) + 1e-4 if sigma is None else torch.as_tensor(sigma)
        )  # default floor sits safely above float32 precision
        weight = (
            mu.new_ones((*mu.shape[:-2], 1)) if weight is None else torch.as_tensor(weight)
        )  # no weight → single uniform component

        assert sigma.shape == mu.shape, f"sigma must match mu shape, got {sigma.shape} vs {mu.shape}"
        assert weight.shape == mu.shape[:-1], f"weight must be [..., k], got {weight.shape}"

        # mu is [K, D]: no explicit batch dim provided → wrap into a single-element batch [1, K, D]
        if mu.dim() == 2:
            mu = mu.unsqueeze(0)
            sigma = sigma.unsqueeze(0)
            weight = weight.unsqueeze(0)

        validate_positive_tensor("sigma", sigma)
        validate_positive_tensor("weight", weight)
        weight_sum = weight.sum(dim=-1, keepdim=True)
        if not torch.all(weight_sum > 0):
            raise ValueError("weight sum must be > 0 for all batches")
        weight = weight / weight_sum
        self.register_buffer("mu", mu)
        self.register_buffer("sigma", sigma)
        self.register_buffer("weight", weight)

        # Convenience attributes for batch shape and dimensions
        self.batch_shape = mu.shape[:-2]  # [...BS, K, D]
        self.batch_ndim = len(self.batch_shape)  # #batch_dims
        self.num_components = mu.shape[-2]
        self.dim = mu.shape[-1]

        assert self.batch_ndim >= 1, "Batch dimension must be at least 1"

        # Shape metadata following PyTorch distribution conventions
        self.event_shape = (self.dim,)  # [Dim]

        self.schedule = BetaSchedule(beta_min=0.1, beta_max=20.0) if schedule is None else schedule

    def _expand_t(self, t: int | float | torch.Tensor | None, sample_shape: tuple) -> torch.Tensor:
        """Return t with shape [*sample_shape, *batch_shape]. Accept t scalar, [*B], or [*N,*B]."""
        if t is None:
            # no t provided → assume clean data at t=0
            return torch.ones(*(sample_shape + self.batch_shape), device=self.mu.device, dtype=self.mu.dtype) * 0.0
        elif not isinstance(t, torch.Tensor):
            # Python scalar (int/float) → broadcast to full [*sample_shape, *batch_shape]
            t_value = float(t)
            validate_time_range(t_value)
            return torch.ones(*(sample_shape + self.batch_shape), device=self.mu.device, dtype=self.mu.dtype) * t_value
        elif t.dim() == 0:
            # 0-dim tensor (e.g. torch.tensor(0.5)) → same as scalar, broadcast to full shape
            t_value = t.item()
            validate_time_range(t_value)
            return torch.ones(*(sample_shape + self.batch_shape), device=t.device, dtype=t.dtype) * t_value
        elif t.shape == sample_shape + self.batch_shape:
            # t already has the full [*sample_shape, *batch_shape] shape → use as-is
            validate_time_range(t)
            return t
        elif t.shape == self.batch_shape:
            # t has only batch shape (one t per GMM in the batch) → expand over sample dims
            validate_time_range(t)
            if len(sample_shape) == 0:
                # no sample dims requested → batch shape is the full shape already
                return t
            return t.expand(sample_shape + self.batch_shape)
        t_repr = t.shape if isinstance(t, torch.Tensor) else t
        raise ValueError(f"t shape {t_repr!r} incompatible with expected {sample_shape + self.batch_shape}")

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

    def marginal_gmm(self, dim) -> "GMM":
        """
        Get marginal GMM distribution for batched GMMs.

        Args:
            dim: int - dimension to marginalize out

        Returns:
            GMM with a single dimension for the marginal GMM
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

        return GMM(mu_marginal, sigma_marginal, self.weight, self.schedule)

    def drop_mode(self, component_index: int) -> "GMM":
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

        return GMM(mu_new, sigma_new, weight_new, self.schedule)

    @jaxtyped(typechecker=beartype)
    def log_prob(
        self, x: Float[Tensor, "*batch D"], t: int | float | torch.Tensor | None = None
    ) -> Float[Tensor, "*batch"]:
        """log_prob(x, t) -> [*N, *B]. x: [*N, *B, D], t: scalar or shape x.shape[:-1], with t in [0, 1]."""
        assert x.shape[-1] == self.dim, f"x last dim must be {self.dim}, got {x.shape[-1]}"
        assert (
            x.shape[-(self.batch_ndim + 1) : -1] == self.batch_shape
        ), f"x must have batch dims {self.batch_shape} before last, got {x.shape}"
        sample_shape = x.shape[: -(self.batch_ndim + 1)]
        t_exp = self._expand_t(t, sample_shape)
        assert t_exp.shape == x.shape[:-1], f"t_exp must have shape {x.shape[:-1]}, got {t_exp.shape}"
        return self._gmm_t(t_exp).log_prob(x)

    def forward(self, x: torch.Tensor, t: int | float | torch.Tensor | None = None) -> torch.Tensor:
        """Alias for log_prob(x, t). Dispatching via nn.Module.__call__ preserves forward-hook semantics."""
        return self.log_prob(x, t)

    @jaxtyped(typechecker=beartype)
    def energy(
        self, x: Float[Tensor, "*batch D"], t: int | float | torch.Tensor | None = None
    ) -> Float[Tensor, "*batch"]:
        """energy(x, t) -> [*N, *B]. Returns -log_prob(x, t). t: scalar or shape x.shape[:-1] in [0, 1]."""
        return -self.log_prob(x, t)

    @jaxtyped(typechecker=beartype)
    @torch.enable_grad()
    def score(
        self, x: Float[Tensor, "*batch D"], t: int | float | torch.Tensor | None = None
    ) -> Float[Tensor, "*batch D"]:
        """score(x, t) -> [*N, *B, D]. ∇_x log p(x). x: [*N, *B, D], t: scalar or shape x.shape[:-1] in [0, 1]."""
        assert x.shape[-1] == self.dim, f"x last dim must be {self.dim}, got {x.shape[-1]}"
        assert (
            x.shape[-(self.batch_ndim + 1) : -1] == self.batch_shape
        ), f"x must have batch dims {self.batch_shape} before last, got {x.shape}"
        sample_shape = x.shape[: -(self.batch_ndim + 1)]
        t_exp = self._expand_t(t, sample_shape)
        assert t_exp.shape == x.shape[:-1], f"t_exp must have shape {x.shape[:-1]}, got {t_exp.shape}"
        # If x (or t) already requires grad, the caller wants to differentiate
        # *through* the score — e.g. reconstruction/x̂0 guidance, where
        # x̂0 = x + σ²·score must carry the full denoiser Jacobian ∂score/∂x_t (and
        # ∂score/∂t). Compute the score on x itself with create_graph=True and do NOT
        # detach, so that Jacobian flows back. Otherwise (the usual drift-term use)
        # return score as a plain detached value — cheaper, no second-order graph.
        if x.requires_grad or (isinstance(t_exp, Tensor) and t_exp.requires_grad):
            score = torch.autograd.grad(
                self._gmm_t(t_exp).log_prob(x).sum(), x, create_graph=True
            )[0]
        else:
            x_grad = x.detach().clone().requires_grad_(True)
            score = torch.autograd.grad(
                self._gmm_t(t_exp).log_prob(x_grad).sum(), x_grad, create_graph=False
            )[0].detach()
        return score

    @jaxtyped(typechecker=beartype)
    def velocity(
        self, x: Float[Tensor, "*batch D"], t: int | float | torch.Tensor | None = None
    ) -> Float[Tensor, "*batch D"]:
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

        alpha_t = self.schedule.get_alpha_t(t_exp).clamp_min(1e-6)  # [*N, *B]; avoid 1/α→∞ at t=1 (LinearSchedule)
        sigma_t = self.schedule.get_sigma_t(t_exp)  # [*N, *B]
        dalpha_dt = self.schedule.get_dalpha_dt(t_exp)  # [*N, *B]
        dsigma_dt = self.schedule.get_dsigma_dt(t_exp)  # [*N, *B]

        score = self.score(x, t)  # [*N, *B, D]

        # v = (dα/dt / α) x + (dα/dt σ/α - dσ/dt) σ score
        coeff_x = (dalpha_dt / alpha_t).unsqueeze(-1)  # [*N, *B, 1]
        coeff_score = ((dalpha_dt * sigma_t / alpha_t - dsigma_dt) * sigma_t).unsqueeze(-1)  # [*N, *B, 1]

        return coeff_x * x + coeff_score * score

    def sample(self, shape: tuple | int | None = None, t: float | torch.Tensor | None = None) -> torch.Tensor:
        """sample(shape, t) -> [*N, *B, D].

        shape: full [*N,*B] tuple ending with batch_shape, int N, or None -> [*B, D].
        t: scalar or [*N, *B] in [0, 1].
        """
        if shape is None:
            # no shape → one sample per GMM in the batch, output is [*batch_shape, D]
            sample_shape = ()
        elif isinstance(shape, int):
            # single int N → N i.i.d. samples per batch element, output is [N, *batch_shape, D]
            sample_shape = (shape,)
        elif isinstance(shape, tuple):
            # full shape tuple → must end with batch_shape; leading dims become sample dims
            assert (
                shape[-self.batch_ndim :] == self.batch_shape
            ), f"shape must end with batch_shape {self.batch_shape}, got {shape}"
            sample_shape = shape[: -self.batch_ndim]
        t_exp = self._expand_t(t, sample_shape)
        assert (
            t_exp.shape == sample_shape + self.batch_shape
        ), f"t_exp must have shape {sample_shape + self.batch_shape}, got {t_exp.shape}"
        return self._gmm_t(t_exp).sample()

    def __repr__(self):
        return f"GMM(mu={self.mu.shape}, sigma={self.sigma.shape}, weight={self.weight.shape})"


class Conditional(GMM):
    """Wraps a single point x0 as a single-component GMM (near-delta distribution).

    Args:
        x0: [*B, D] — one point per batch element
        schedule: Optional Schedule. Defaults to BetaSchedule.
    """

    def __init__(self, x0: torch.Tensor, schedule: Schedule | None = None):
        x0 = torch.as_tensor(x0)
        assert x0.dim() >= 1, "x0 must be at least 1D [*B, D]"
        mu = x0.unsqueeze(-2)  # [*B, 1, D]
        sigma = torch.ones_like(mu) * 1e-4  # small but above float32 precision floor
        weight = x0.new_ones(*x0.shape[:-1], 1)  # [*B, 1]
        super().__init__(mu, sigma, weight, schedule)

    def __repr__(self):
        sched = type(self.schedule).__name__
        return f"Conditional(x0={self.mu.squeeze(-2).shape}, batch_shape={self.batch_shape}, schedule={sched})"
