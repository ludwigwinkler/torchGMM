"""
Analytical Diffusion with Time-Dependent Gaussian Mixture Model

This module implements a time-dependent Gaussian Mixture Model (GMM) that:
1. Starts with two differently weighted modes at t=0
2. Evolves through the forward SDE: dX_t = -1/2 * β(t) * X_t dt + √β(t) dW_t
3. Ends in a unimodal normal distribution at t=1

The score is computed using autograd of the log probability.
"""

import torch
from tqdm import tqdm

from typing import Any


def forward_diffusion(
    scheduler: torch.nn.Module,
    x: torch.Tensor,
    t: float,
    n_steps: int = 100,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Simulate the forward diffusion process using Euler-Maruyama:
    dX_t = -1/2 * β(t) * X_t dt + √β(t) dW_t

    Args:
        potential_sampler: Time-dependent GMM model
        x: Initial samples [n_samples, 1] at time t (optional if n_samples provided)
        t: Starting time (0 <= t <= 1) (optional if n_samples provided)
        n_steps: Number of integration steps

    Returns:
        trajectory: [n_steps+1, n_samples, 1] - trajectory to terminal distribution
    """

    x = x.clone()
    if x.dim() == 1:
        x = x.unsqueeze(-1)  # Ensure [n_samples, 1] shape

    if t >= 1.0:
        # Already at terminal time, return input as single timestep
        return x.unsqueeze(0), torch.tensor([1.0], device=x.device)

    # Prepare for forward diffusion simulation
    n_samples = x.shape[0]

    # Set up time discretization for Euler-Maruyama integration
    time_span = 1.0 - t
    dt = time_span / n_steps
    trajectory = [x.clone()]

    # Forward diffusion loop using Euler-Maruyama method
    for step in tqdm(range(n_steps)):
        current_t = t + step * dt
        t_tensor = torch.full((n_samples,), current_t, device=x.device)

        # Get beta value for current time step
        beta_t = scheduler.beta(t_tensor).unsqueeze(-1)  # [n_samples, 1]

        # Compute drift and diffusion terms for forward SDE
        drift = -0.5 * beta_t * x
        diffusion = torch.sqrt(beta_t)
        noise = torch.randn_like(x, device=x.device)

        # Apply Euler-Maruyama step: x_{t+dt} = x_t + drift*dt + diffusion*sqrt(dt)*noise
        x = x + drift * dt + diffusion * torch.sqrt(torch.tensor(dt, device=x.device)) * noise
        trajectory.append(x.clone())

    # Stack all trajectory steps and create time indices
    trajectory_tensor = torch.stack(trajectory)  # [n_steps+1, n_samples, 1]
    time_indices = torch.linspace(t, 1.0, n_steps + 1, device=x.device)  # [n_steps+1]

    return trajectory_tensor, time_indices


def reverse_diffusion_with_regular_resampling(
    model: torch.nn.Module,
    scheduler: torch.nn.Module,
    x: torch.Tensor,
    t: float,
    n_steps: int = 100,
    potentials: list = None,
    steering_config: dict = None,
    score_fn=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Simulate the reverse diffusion process using the score function:
    dX_t = [1/2 * β(t) * X_t + β(t) * ∇_x log p_t(x)] dt + √β(t) dW_t

    Args:
        potential_sampler: Time-dependent GMM model
        x: Initial samples [n_samples, 1] at time t (optional if n_samples provided)
        t: Starting time (0 <= t <= 1) (optional if n_samples provided)
        n_samples: Number of samples to generate from N(0,1) at t=1 (optional if x,t provided)
        n_steps: Number of integration steps
        potentials: List of potential functions for steering
        steering_config: Dict with keys: n_particles, start, resampling_freq

    Returns:
        trajectory: [n_steps+1, n_samples, 1] - trajectory to terminal distribution
    """
    x = x.clone()
    device = x.device
    if x.dim() == 1:
        x = x.unsqueeze(-1)  # Ensure [n_samples, 1] shape

    if t <= 0.0:
        # Already at terminal time, return input as single timestep
        return x.unsqueeze(0), torch.tensor([0.0], device=x.device)

    # Prepare for reverse diffusion simulation
    n_samples = x.shape[0]

    # Set up time discretization for reverse process (going backwards from t to 0)
    time_span = t
    dt = time_span / n_steps
    trajectory = [x.detach().clone()]
    previous_energy = None
    delta_x, universal_backward_score = None, None
    log_weight = torch.distributions.Normal(0, 1).log_prob(x).sum(dim=-1)
    x0 = None

    # Reverse diffusion loop using Euler-Maruyama method
    for step in tqdm(range(n_steps)):
        current_t = t - step * dt  # Go backwards in time
        t_tensor = torch.full((n_samples,), current_t, device=x.device)

        # Get beta value for current time step
        beta_t = scheduler.beta(t_tensor).unsqueeze(-1)  # [n_samples, 1]
        alpha_t, sigma_t = scheduler.get_alpha_t_sigma_t(t_tensor)
        alpha_t, sigma_t = alpha_t.unsqueeze(-1), sigma_t.unsqueeze(-1)

        # Compute score function for entire batch at once. If an external score_fn is provided
        # (e.g., a neural network approximation), use it; otherwise fall back to analytical score.
        if score_fn is None:
            # Use the model's score method (works for both TimeDependentGMM and TimeDependentGMM2)
            score = model.score(x, t_tensor)  # [n_samples, d]
        else:
            with torch.no_grad():
                score = score_fn(x, t_tensor)
            # Ensure gradient can flow if caller wants (typically not in sampling)
            if not score.requires_grad:
                score = score.to(x.dtype)

        # Extract clean data x0 from noisy observation using Tweedie's formula
        x0_pred = (x + sigma_t * score) / torch.sqrt(alpha_t)

        # Compute drift and diffusion terms for reverse SDE
        drift = 0.5 * beta_t * x + beta_t * score
        dt = torch.tensor(dt, device=device)
        diffusion = torch.sqrt(beta_t)
        noise = torch.randn_like(x, device=device)
        dW = noise * dt**0.5

        # Apply Euler-Maruyama step (backwards in time)
        update = drift * dt + diffusion * dW

        # Compute universal backward score
        if steering_config and steering_config["gradient_steering"]:
            delta_x = potential_gradient_minimization(x0_pred, potentials, learning_rate=0.2, num_steps=20)
            w_t_mod = torch.relu(3 * (torch.tensor(current_t) - 0.0))
            w_t_orig = 1.0
            w_t_mod /= w_t_orig + w_t_mod
            w_t_orig /= w_t_orig + w_t_mod
            universal_backward_score = -(x - alpha_t * (x0_pred + delta_x)) / sigma_t**2
            modified_score = w_t_orig * score + w_t_mod * universal_backward_score
            drift = 0.5 * beta_t * x + beta_t * modified_score
            update = drift * dt + diffusion * dW

            # Compute and accumulate log_weight for gradient steering
            # -(A + B)^2 + B^2 in the classic Gaussian form
            step_log_weight = -((beta_t * (modified_score - score) * dt + diffusion * dW) ** 2) + (diffusion * dW) ** 2
            # step_log_weight = -(beta_t * (modified_score - score) * dt + diffusion * dW)**2 + beta_t * dt
            step_log_weight = (step_log_weight / (2 * beta_t * dt)).squeeze(-1)
            step_log_weight = step_log_weight.squeeze(-1)  # Ensure 1D tensor
            # -A^2 - 2AB after calculating the square
            # step_log_weight = -0.5 * w_t**2 * beta_t * (universal_backward_score)**2 * dt -
            #                   w_t * universal_backward_score * diffusion * dW
            # step_log_weight = step_log_weight.squeeze(-1)
            log_weight = log_weight + step_log_weight
        else:
            # When not using gradient steering, add 0 (no change to log_weight)
            log_weight = log_weight + 0.0

        assert x.shape == update.shape
        x = x + update
        trajectory.append(x.clone())

        # Steering functionality using particle filtering
        if steering_config and potentials:

            # Evaluate potential energies on predicted clean data
            energies = [pot(x0_pred) for pot in potentials]  # TODO: why are we using a list here?
            total_energy = torch.stack(energies, dim=-1)  # [n_samples]

            # Apply particle filtering resampling if multiple particles per sample
            if steering_config["n_particles"] > 1:
                start_step = int(n_steps * steering_config["start"])
                resample_freq = steering_config["resampling_freq"]

                # Resample particles based on potential energy
                if start_step <= step < (n_steps - 2) and step % resample_freq == 0:
                    x, total_energy, log_weight = resample_particles(
                        x,
                        total_energy,
                        previous_energy=previous_energy,
                        log_weight=log_weight,
                        steering_config=steering_config,
                    )
                    previous_energy = total_energy
                elif step == n_steps - 1:  # Resample last step with clean samples, instead of x0_pred
                    print("Resampling particles in the last step...", step)

                    # Recompute the total energy based on x instead of x0_pred
                    # Evaluate potential energies on predicted clean data
                    energies = [pot(x) for pot in potentials]
                    total_energy = torch.stack(energies, dim=-1)  # [n_samples]

                    x, total_energy, log_weight = resample_particles(
                        x,
                        total_energy,
                        previous_energy=previous_energy,
                        log_weight=log_weight,
                        steering_config=steering_config,
                    )
                    previous_energy = total_energy
                    # Replace last step with resampled x
                    trajectory[-1] = x.clone()

    # Stack all trajectory steps and create time indices
    trajectory_tensor = torch.stack(trajectory)  # [n_steps+1, n_samples, 1]
    time_indices = torch.linspace(t, 0.0, n_steps + 1, device=x.device)  # [n_steps+1]

    return trajectory_tensor, time_indices


def reverse_diffusion(
    model: torch.nn.Module,
    scheduler: torch.nn.Module,
    x: torch.Tensor,
    t: float,
    n_steps: int = 100,
    potentials: list = None,
    steering_config: dict = None,
    score_fn=None,
    denoising_and_resample_fn: callable = None,
) -> tuple[torch.Tensor, torch.Tensor, list, int]:
    """
    Simulate the reverse diffusion process using the score function:
    dX_t = [1/2 * β(t) * X_t + β(t) * ∇_x log p_t(x)] dt + √β(t) dW_t

    Args:
        potential_sampler: Time-dependent GMM model
        x: Initial samples [n_samples, 1] at time t (optional if n_samples provided)
        t: Starting time (0 <= t <= 1) (optional if n_samples provided)
        n_samples: Number of samples to generate from N(0,1) at t=1 (optional if x,t provided)
        n_steps: Number of integration steps
        potentials: List of potential functions for steering
        steering_config: Dict with keys: n_particles, start, resampling_freq
        score_fn: Optional external score function (e.g., neural network approximation)
        denoising_and_resample_fn: Function to perform one denoising step and resampling.

    Returns:
        trajectory: [n_steps+1, n_samples, 1] - trajectory to terminal distribution
        time_indices: 1D tensor of corresponding times (descending from t to 0)
        ess_log: list of (time, ess_fraction) pairs recorded at each resampling step,
                 where ess_fraction = ESS / n_particles (range (0,1]). Empty if no resampling.
        nfe: Number of score function evaluations performed during reverse diffusion
    """
    x = x.clone()
    if x.dim() == 1:
        x = x.unsqueeze(-1)  # Ensure [n_samples, 1] shape

    if t <= 0.0:
        # Already at terminal time, return input as single timestep
        return x.unsqueeze(0), torch.tensor([0.0], device=x.device), [], 0

    # Prepare for reverse diffusion simulation
    n_samples, n_dim = x.shape

    # Set up time discretization for reverse process (going backwards from t to 0)
    time_span = t
    dt = time_span / n_steps
    trajectory = [x.detach().clone()]
    ess = 1.0
    ess_log = []  # list of (time, ESS_fraction)
    previous_energy = torch.zeros(n_samples, device=x.device)
    n_particles = steering_config["n_particles"] if steering_config else 1

    # Track log weights of samples
    log_weight = torch.zeros(n_samples, device=x.device)

    # Track number of score function evaluations
    nfe = 0

    # Reverse diffusion loop using Euler-Maruyama method
    for step in tqdm(range(n_steps)):
        current_t = t - step * dt  # Go backwards in time: 1 -> 0
        t_tensor = torch.full((n_samples,), current_t, device=x.device)

        # Get beta value for current time step
        beta_t = scheduler.beta(t_tensor).unsqueeze(-1)  # [n_samples, 1]
        alpha_t, sigma_t = scheduler.get_alpha_t_sigma_t(t_tensor)
        alpha_t, sigma_t = alpha_t.unsqueeze(-1), sigma_t.unsqueeze(-1)

        # Create score function wrapper for denoising functions
        # The denoising functions will call this with (x, t_tensor)
        if score_fn is None:

            def counted_score_func(x_, t_tensor_):
                nonlocal nfe
                nfe += 1
                return model.score(x_, t_tensor_)

        else:

            def counted_score_func(x_, t_tensor_):
                nonlocal nfe
                nfe += 1
                return score_fn(x_, t_tensor_)

        # Evolve one step of x and log_weight
        assert log_weight.shape == (n_samples,) and t_tensor.shape == (
            n_samples,
        ), f"log_weight.shape: {log_weight.shape}, t_tensor.shape: {t_tensor.shape}"
        x, log_weight, total_energy, ess = denoising_and_resample_fn(
            x=x,
            scheduler=scheduler,
            log_weight=log_weight,
            beta_t=beta_t,
            score_fn=counted_score_func,
            dt=dt,
            potentials=potentials,
            current_t=current_t,
            t=t,
            previous_energy=previous_energy,
            step=step,
            n_steps=n_steps,
            n_particles=n_particles,
            steering_config=steering_config,
        )
        # assert ess >= 0.0 and ess <= 1.0, f"ESS: {ess}"
        ess_log.append((float(current_t), float(ess.cpu())))
        previous_energy = total_energy
        trajectory.append(x.clone())
    print(f"Final t: {current_t:.3f}, ESS: {ess:.3f}")
    # trajectory = trajectory[1:]

    # Stack all trajectory steps and create time indices
    trajectory_tensor = torch.stack(trajectory)  # [n_steps+1, n_samples, 1]
    time_indices = torch.linspace(t, 0.0, n_steps + 1, device=x.device)  # [n_steps+1]

    return trajectory_tensor, time_indices, ess_log, nfe


def denoising_and_resample_smc(
    x: torch.Tensor,
    scheduler: torch.nn.Module,
    log_weight: torch.Tensor,
    beta_t: torch.Tensor,
    score_fn: callable,
    dt: float,
    potentials: list,
    current_t: float,
    t: float,
    previous_energy: torch.Tensor,
    step: int,
    n_steps: int,
    n_particles: int,
    steering_config: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Compute drift and diffusion terms for reverse SDE
    n_samples = x.shape[0]
    t_tensor = torch.full((n_samples,), current_t, device=x.device)
    score = score_fn(x, t_tensor)
    drift = 0.5 * beta_t * x + beta_t * score
    diffusion = torch.sqrt(beta_t)
    noise = torch.randn_like(x, device=x.device)

    # Apply Euler-Maruyama step (backwards in time)
    assert x.shape == drift.shape == noise.shape == (n_samples, x.shape[-1])
    x = x + drift * dt + diffusion * torch.sqrt(torch.tensor(dt, device=x.device)) * noise

    # Denoise
    alpha_t, sigma_t = scheduler.get_alpha_t_sigma_t(t_tensor)
    alpha_t, sigma_t = alpha_t.unsqueeze(-1), sigma_t.unsqueeze(-1)
    x_0 = (x + sigma_t * score) / torch.sqrt(alpha_t)

    # Steering functionality using particle filtering
    # Evaluate potential energies on predicted clean data
    pot = potentials[0]
    # linear interpolation of target center from 0 to final target. current_t goes from t to 0
    prefactor_t = 1.0 - current_t / t  # alpha_t.squeeze() #(1.0 - current_t / t)
    total_energy = pot(x_0) * prefactor_t  # [n_samples]
    assert (
        total_energy.shape == (n_samples,) == previous_energy.shape == log_weight.shape
    ), f"total_energy.shape: {total_energy.shape}, previous_energy.shape: {previous_energy.shape}, log_weight.shape: {log_weight.shape}"
    dlog_weight = -total_energy + previous_energy
    log_weight = log_weight + dlog_weight

    ess, normalized_weight = compute_ess_from_log_weights(log_weight, n_particles)
    if 0 < steering_config["ess_threshold"] < 1.0:
        resampling_condition = ess < steering_config["ess_threshold"]
    elif steering_config["ess_threshold"] > 1.0:
        resampling_condition = step % int(steering_config["ess_threshold"]) == 0
    else:
        raise ValueError(f"Invalid ess_threshold: {steering_config['ess_threshold']}")
    if resampling_condition or (step == n_steps - 1):
        # if step % 5 == 0:
        # print(f"ESS: {ess:.3f} at step {step}, triggering resampling.")
        indices = torch.multinomial(
            normalized_weight, num_samples=n_particles, replacement=True
        )  # [n_groups, n_particles]

        n_samples, n_dim = x.shape
        assert n_samples % n_particles == 0, "n_samples must be multiple of n_particles"
        n_groups = n_samples // n_particles

        # Resample particles, shuffle sample and energy
        x_grouped = x.view(n_groups, n_particles, n_dim)
        energy_grouped = total_energy.view(n_groups, n_particles)
        x = torch.stack([x_grouped[i, indices[i]] for i in range(n_groups)]).view(n_samples, n_dim)
        total_energy = torch.stack([energy_grouped[i, indices[i]] for i in range(n_groups)]).view(n_samples)

        # Reset log weights after resampling
        log_weight = torch.zeros(n_samples, device=x.device)
    assert log_weight.shape == (n_samples,) == total_energy.shape
    return x, log_weight, total_energy, ess


def denoising_and_resample_fkc(
    x: torch.Tensor,
    scheduler: torch.nn.Module,
    log_weight: torch.Tensor,
    beta_t: torch.Tensor,
    score_fn: callable,
    dt: float,
    potentials: list[torch.nn.Module],
    current_t: float,
    t: float,
    previous_energy: torch.Tensor,
    step: int,
    n_steps: int,
    n_particles: int,
    steering_config: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Feynman-Kac corrector denoising and resampling step.

    Implements guided reverse diffusion with reward-based steering:
    - Forward Process:    dx = - 0.5 * beta_t * x * dt + sqrt(beta_t) dW_t
    - Reverse Process:    dx = (0.5 * beta_t * x + beta_t * score(x, t) + lambda_t * beta_t/2 * grad_r(x)) * dt + sqrt(beta_t) dW_t
    - Reverse Weight ODE: dw = partial_t lambda_t + (lambda_t grad_r(x) * (0.5 * beta_t * x + beta_t * score(x, t))) * dt

    Args:
        x: Current particle positions [n_samples, d]
        scheduler: Diffusion scheduler providing beta(t) and noise schedule
        log_weight: Current log weights of particles [n_samples]
        beta_t: Beta values at current time [n_samples, 1]
        score_fn: Score function for reverse diffusion
        dt: Time step size
        potentials: List of potential functions for steering (only first is used)
        current_t: Current time value
        t: Total time span
        previous_energy: Previous energy values [n_samples]
        step: Current step number
        n_steps: Total number of steps
        n_particles: Number of particles per group
        steering_config: Configuration dict with resample_threshold

    Returns:
        tuple: (x, log_weight, total_energy, ess) - updated particles and diagnostics
    """

    assert len(potentials) <= 1, "Only one potential is supported in FKC currently."

    n_samples = x.shape[0]
    n_dimensions = x.shape[1]
    t_tensor = torch.full((n_samples,), current_t, device=x.device)
    score_batch = score_fn(x, t_tensor)
    fw_drift = -0.5 * beta_t * x  # forward process drift in t: 0 -> 1 direction
    drift = -fw_drift + beta_t * score_batch  # reverse drift in tau: 1 -> 0 direction
    diffusion = torch.sqrt(beta_t)
    noise = torch.randn_like(x, device=x.device)
    n_samples, x_dim = x.shape

    # Get dx_t
    # Gradient-based steering adjustment
    if len(potentials) > 1:
        raise NotImplementedError("Only one potential is supported in FKC currently.")

    lambda_t = (1.0 - current_t / t) * torch.ones([n_samples, 1], device=x.device)  # (1.0 - current_t / t)

    # target = potentials[0].target
    # order = potentials[0].order
    # slope = potentials[0].slope
    # reward_grad = -order * slope**order * (x - target) ** (order - 1)
    reward_grad = potentials[0].force(x)  # [n_samples, d]

    dx_t = (
        drift * dt
        + beta_t * lambda_t / 2 * reward_grad * dt
        + diffusion * torch.sqrt(torch.tensor(dt, device=x.device)) * noise
    )

    # Get dlog_weight_t
    total_energy = potentials[0](x)  # [n_samples]
    reward = -total_energy.reshape(n_samples, n_dimensions)  # [BS, D]

    # lambda: T -> 0: 0 -> 1, increases as we reverse difffuse, so positive sign
    dlambda_dt = 1.0 / t * torch.ones([n_samples, 1], device=x.device)

    dlog_weight_dt = dlambda_dt * reward + lambda_t * (reward_grad * (-fw_drift + 0.5 * beta_t * score_batch)).sum(
        dim=-1, keepdim=True
    )  # [BS, 1]*[BS, 1] + [BS, 1] * [BS, D].sum(dim=-1) = [BS, 1]
    dlog_weight_dt = dlog_weight_dt.reshape(n_samples)

    # Update x and log_weight
    x = x + dx_t
    log_weight = log_weight + dlog_weight_dt * dt

    # Compute ESS from log_weights for particles in a group
    assert n_samples % n_particles == 0, f"n_samples ({n_samples}) is not multiple of n_particles ({n_particles})"
    n_groups = n_samples // n_particles
    unnormalized_weight = torch.exp(torch.nn.functional.log_softmax(log_weight.view(n_groups, n_particles), dim=-1))
    normalized_weight = unnormalized_weight / (unnormalized_weight.sum(dim=-1, keepdim=True) + 1e-12)
    ess = 1.0 / (normalized_weight**2).sum(dim=-1)
    ess = (ess / n_particles).mean()  # average over groups

    # Resample particles based on log weights
    if steering_config is None:
        # Default behavior: resample every 5 steps
        resampling_condition = step % 5 == 0
    elif 0 < steering_config["ess_threshold"] < 1.0:
        resampling_condition = ess < steering_config["ess_threshold"]
    elif steering_config["ess_threshold"] > 1.0:
        resampling_condition = step % int(steering_config["ess_threshold"]) == 0
    else:
        raise ValueError(f"Invalid ess_threshold: {steering_config['ess_threshold']}")
    if resampling_condition or (step == n_steps - 1):
        indices = torch.multinomial(
            normalized_weight, num_samples=n_particles, replacement=True
        )  # [n_groups, n_particles]

        # Resample particles, shuffle sample and energy
        x_grouped = x.view(n_groups, n_particles, x_dim)
        # energy_grouped = total_energy.view(n_groups, n_particles)
        x = torch.stack([x_grouped[i, indices[i]] for i in range(n_groups)]).view(n_samples, x_dim)
        # total_energy = torch.stack([energy_grouped[i, indices[i]] for i in range(n_groups)]).view(n_samples)

        # Reset log weights after resampling
        log_weight = torch.zeros(n_samples, device=x.device)
    return x, log_weight, total_energy, ess


def denoising(
    x: torch.Tensor,
    scheduler: torch.nn.Module,
    log_weight: torch.Tensor,
    beta_t: torch.Tensor,
    score_fn: callable,
    dt: float,
    potentials: list,
    current_t: float,
    t: float,
    previous_energy: torch.Tensor,
    step: int,
    n_steps: int,
    n_particles: int,
    steering_config: dict = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Forward Process: 0 -> 1
    Reverse Process: 1 -> 0
    Forward Process:    dx = - 0.5 * beta_t * x * dt + sqrt(beta_t) dW_t
    Reverse Process:    dx = (0.5 * beta_t * x + beta_t * score(x, t)) * dt + sqrt(beta_t) dW_t
                        dx = (0.5 * beta_t * x + beta_t * score(x, t) + lambda_t * beta_t/2 * grad_r(x) * dt + sqrt(beta_t) dW_t
    Reverse Weight ODE: dw = partial_t lambda_t + ( lambda_t grad_r(x) * (0.5 * beta_t * x + beta_t * score(x, t))) * dt
    lambda_t is the papers beta_t
    """

    n_samples = x.shape[0]
    t_tensor = torch.full((n_samples,), current_t, device=x.device)
    score_batch = score_fn(x, t_tensor)
    fw_drift = -0.5 * beta_t * x  # forward process drift in t: 0 -> 1 direction
    drift = -fw_drift + beta_t * score_batch  # reverse drift in tau: 1 -> 0 direction
    diffusion = torch.sqrt(beta_t)
    noise = torch.randn_like(x, device=x.device)
    n_samples, x_dim = x.shape

    # Get dx_t
    # Gradient-based steering adjustment

    lambda_t = (1.0 - current_t / t) * torch.ones_like(beta_t)  # (1.0 - current_t)
    p = 1.0
    lambda_t = (1.0 - current_t / t) ** p * torch.ones_like(beta_t)

    assert lambda_t.shape == torch.Size([x.shape[0], 1])
    # reward_grad = -order * slope**order * (x - target) ** (order - 1)

    dx_t = drift * dt + diffusion * torch.sqrt(torch.tensor(dt, device=x.device)) * noise

    # Update x and log_weight
    assert x.shape == dx_t.shape, f"x.shape: {x.shape}, dx_t.shape: {dx_t.shape}"
    x = x + dx_t
    total_energy = torch.zeros(x.shape[0], device=x.device)
    log_weight = torch.zeros(x.shape[0], device=x.device)
    ess = torch.tensor(0.0, device=x.device)
    return x, log_weight, total_energy, ess


def denoising_and_resample_smclangevin(
    x: torch.Tensor,
    scheduler: torch.nn.Module,
    log_weight: torch.Tensor,
    beta_t: torch.Tensor,
    score_fn: callable,
    dt: float,
    potentials: list,
    current_t: float,
    t: float,
    previous_energy: torch.Tensor,
    step: int,
    n_steps: int,
    n_particles: int,
    ess_threshold: float = 0.5,
    steering_config: dict = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Extract method-specific parameters from steering_config
    n_langevin_steps = steering_config.get("n_langevin_steps", 10) if steering_config else 10
    langevin_stepsize = steering_config.get("langevin_stepsize", 0.25) if steering_config else 0.25
    # Compute drift and diffusion terms for reverse SDE
    n_samples = x.shape[0]
    t_tensor = torch.full((n_samples,), current_t, device=x.device)
    score = score_fn(x, t_tensor)
    alpha_t, sigma_t = scheduler.get_alpha_t_sigma_t(t_tensor)
    alpha_t, sigma_t = alpha_t.unsqueeze(-1), sigma_t.unsqueeze(-1)
    drift = 0.5 * beta_t * x + beta_t * score
    diffusion = torch.sqrt(beta_t)
    noise = torch.randn_like(x, device=x.device)

    # Apply Euler-Maruyama step (backwards in time)
    x = x + drift * dt + diffusion * torch.sqrt(torch.tensor(dt, device=x.device)) * noise

    # Steering functionality using particle filtering
    # Evaluate potential energies on predicted clean data
    pot = potentials[0]
    x0 = (x + sigma_t * score) / torch.sqrt(alpha_t)
    # linear interpolation of target center from 0 to final target. current_t goes from t to 0
    prefactor_t = 1.0 - current_t / t  # alpha_t.squeeze() #(1.0 - current_t / t)
    total_energy = pot(x0) * prefactor_t  # [n_samples]
    dlog_weight = -total_energy + previous_energy
    log_weight = log_weight + dlog_weight

    ess, normalized_weight = compute_ess_from_log_weights(log_weight, n_particles)
    # Resample particles based on log weights, and shuffle energies correspondingly
    if 0 < ess_threshold < 1.0:
        resampling_condition = ess < steering_config["ess_threshold"]
    elif steering_config["ess_threshold"] > 1.0:
        resampling_condition = step % int(steering_config["ess_threshold"]) == 0
    else:
        raise ValueError(f"Invalid ess_threshold: {steering_config['ess_threshold']}")
    if resampling_condition or (step == n_steps - 1):
        # print(f"ESS: {ess:.3f} at step {step}, triggering resampling.")
        indices = torch.multinomial(
            normalized_weight, num_samples=n_particles, replacement=True
        )  # [n_groups, n_particles]

        n_samples, n_dim = x.shape
        assert n_samples % n_particles == 0, "n_samples must be multiple of n_particles"
        n_groups = n_samples // n_particles

        # Resample particles, shuffle sample and energy
        x_grouped = x.view(n_groups, n_particles, n_dim)
        energy_grouped = total_energy.view(n_groups, n_particles)
        x = torch.stack([x_grouped[i, indices[i]] for i in range(n_groups)]).view(n_samples, n_dim)
        total_energy = torch.stack([energy_grouped[i, indices[i]] for i in range(n_groups)]).view(n_samples)
        log_weight = torch.zeros(n_samples, device=x.device)

        # Equilibriate distribution
        eta = langevin_stepsize
        for langevin_step in range(n_langevin_steps):
            z = torch.randn_like(x, device=x.device)
            score = score_fn(x, t_tensor)
            force = pot.force(x)  # - \nabla E(x)
            x = x + 0.5 * eta * eta * (score + prefactor_t * force) + eta * z

        # Reset log weights after resampling

    return x, log_weight, total_energy, ess


def denoising_and_resample_fksmc(
    x: torch.Tensor,
    scheduler: torch.nn.Module,
    log_weight: torch.Tensor,
    beta_t: torch.Tensor,
    score_fn: callable,
    dt: float,
    potentials: list,
    current_t: float,
    t: float,
    previous_energy: torch.Tensor,
    step: int,
    n_steps: int,
    n_particles: int,
    steering_config: dict = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Extract method-specific parameters from steering_config
    n_langevin_steps = steering_config.get("n_langevin_steps", 10) if steering_config else 10
    langevin_stepsize = steering_config.get("langevin_stepsize", 0.25) if steering_config else 0.25
    # Compute drift and diffusion terms for reverse SDE
    n_samples = x.shape[0]
    t_tensor = torch.full((n_samples,), current_t, device=x.device)
    score = score_fn(x, t_tensor)
    alpha_t, sigma_t = scheduler.get_alpha_t_sigma_t(t_tensor)
    alpha_t, sigma_t = alpha_t.unsqueeze(-1), sigma_t.unsqueeze(-1)
    drift = 0.5 * beta_t * x + beta_t * score_fn(x, t_tensor)
    diffusion = torch.sqrt(beta_t)
    noise = torch.randn_like(x, device=x.device)

    # Apply Euler-Maruyama step (backwards in time)
    x = x + drift * dt + diffusion * torch.sqrt(torch.tensor(dt, device=x.device)) * noise

    # Apply Langevin steps
    prefactor_t = 1.0 - current_t / t

    eta = langevin_stepsize
    for langevin_step in range(n_langevin_steps):
        z = torch.randn_like(x, device=x.device)
        score = score_fn(x, t_tensor)
        x0 = (x + sigma_t * score) / torch.sqrt(alpha_t)
        force = potentials[0].force(x0)  # -\nabla E(x0)
        x = x + 0.5 * eta * eta * (score + prefactor_t * force) + eta * z

    # Steering functionality using particle filtering
    # Evaluate potential energies on predicted clean data
    pot = potentials[0]
    x0 = (x + sigma_t * score) / torch.sqrt(alpha_t)
    # linear interpolation of target center from 0 to final target. current_t goes from t to 0
    prefactor_t = 1.0 - current_t / t  # T -> 0: 0 -> 1
    dbeta_t = dt
    total_energy = pot(x0)
    dlog_weight = -dbeta_t * total_energy
    log_weight = log_weight + dlog_weight

    ess, normalized_weight = compute_ess_from_log_weights(log_weight, n_particles)
    # Resample particles based on log weights, and shuffle energies correspondingly
    if 0 < steering_config["ess_threshold"] < 1.0:
        resampling_condition = ess < steering_config["ess_threshold"]
    elif steering_config["ess_threshold"] >= 1.0:
        resampling_condition = step % int(steering_config["ess_threshold"]) == 0
    else:
        raise ValueError(f"Invalid ess_threshold: {steering_config['ess_threshold']}")
    if resampling_condition or (step == n_steps - 1):
        # print(f"ESS: {ess:.3f} at step {step}, triggering resampling.")
        indices = torch.multinomial(
            normalized_weight, num_samples=n_particles, replacement=True
        )  # [n_groups, n_particles]

        n_samples, n_dim = x.shape
        assert n_samples % n_particles == 0, "n_samples must be multiple of n_particles"
        n_groups = n_samples // n_particles

        # Resample particles, shuffle sample and energy
        x_grouped = x.view(n_groups, n_particles, n_dim)
        energy_grouped = total_energy.view(n_groups, n_particles)
        x = torch.stack([x_grouped[i, indices[i]] for i in range(n_groups)]).view(n_samples, n_dim)
        total_energy = torch.stack([energy_grouped[i, indices[i]] for i in range(n_groups)]).view(n_samples)

        # Reset log weights after resampling
        log_weight = torch.zeros(n_samples, device=x.device)
    return x, log_weight, total_energy, ess


def compute_ess_from_log_weights(log_weight: torch.Tensor, n_particles: int) -> tuple[torch.Tensor, torch.Tensor]:
    # Compute ESS from log_weights for particles in a group
    n_samples = log_weight.shape[0]
    assert n_samples % n_particles == 0, "n_samples must be multiple of n_particles"
    n_groups = n_samples // n_particles
    unnormalized_weight = torch.exp(torch.nn.functional.log_softmax(log_weight.view(n_groups, n_particles), dim=-1))
    normalized_weight = unnormalized_weight / (unnormalized_weight.sum(dim=-1, keepdim=True) + 1e-12)
    ess = 1.0 / (normalized_weight**2).sum(dim=-1)
    ess = (ess / n_particles).mean()  # average over groups
    return ess, normalized_weight
