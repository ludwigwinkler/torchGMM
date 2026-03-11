# Diffusion Models vs Flow Matching: Schedules, Scores, and Velocities

> A technical reference for the torchGMM project.
> Assumes familiarity with probability, stochastic calculus, and basic generative modelling.

---

## Table of Contents

1. [Setup and Notation](#1-setup-and-notation)
2. [Diffusion (VP-SDE) Formulation](#2-diffusion-vp-sde-formulation)
3. [Flow Matching (Conditional OT) Formulation](#3-flow-matching-conditional-ot-formulation)
4. [How the Schedules Relate](#4-how-the-schedules-relate)
5. [Score vs Velocity](#5-score-vs-velocity)
6. [Implementation Plan for torchGMM](#6-implementation-plan-for-torcghmm)
7. [Summary Table](#7-summary-table)

---

## 1. Setup and Notation

Both diffusion models and flow matching solve the same high-level problem: **learn a
time-indexed family of transformations that maps a simple prior (noise) to a complex data
distribution.** They differ in *how* they parameterise the path from data to noise and
*what* they regress during training.

Throughout this document:

| Symbol | Meaning |
|--------|---------|
| x_0 | Data sample, drawn from data distribution q(x_0) |
| ε | Standard Gaussian noise, ε ~ N(0, I) |
| t | Time index, t ∈ [0, 1]. t = 0 is data, t = 1 is noise |
| D | Ambient dimension |
| β(t) | Instantaneous noise rate (diffusion schedule) |
| α_t | Signal coefficient at time t |
| σ_t | Noise coefficient at time t |
| p_t(x) | Marginal density at time t |
| p_t(x \| x_0) | Conditional density given a fixed starting point x_0 |

**Convention (matching torchGMM):** t = 0 corresponds to clean data and t = 1
corresponds to (approximately) pure noise.

---

## 2. Diffusion (VP-SDE) Formulation

### 2.1 The Forward SDE

The Variance-Preserving SDE (VP-SDE) defines a continuous-time noising process:

```
dX_t = -1/2 β(t) X_t dt + √β(t) dW_t
```

where W_t is a standard Wiener process and β(t) > 0 is a time-dependent noise rate.

This is an **Ornstein–Uhlenbeck process** with time-varying coefficients: the drift
shrinks the signal toward zero while the diffusion injects noise.

### 2.2 Linear Beta Schedule

torchGMM implements a linear schedule (the most common choice):

```
β(t) = β_min + t · (β_max − β_min)
```

with defaults β_min = 0.1 and β_max = 20.0. The integral is:

```
∫₀ᵗ β(s) ds = β_min · t + 1/2 (β_max − β_min) · t^2
```

### 2.3 Conditional Marginals

Because the forward SDE is linear in X_t with Gaussian noise, the conditional
distribution p_t(x | x_0) is Gaussian at every t:

```
p_t(x | x_0) = N(x; α_t x_0, σ_t² I)
```

where the **signal coefficient** α_t and **noise coefficient** σ_t are:

```
α_t = exp( -1/2 ∫₀ᵗ β(s) ds )

σ_t² = 1 − α_t²
```

**Derivation sketch.** The OU process dX = -1/2β X dt + √β dW has the explicit solution:

```
X_t = exp(-1/2 ∫₀ᵗ β(s) ds) · x_0  +  ∫₀ᵗ exp(-1/2 ∫ₛᵗ β(u) du) · √β(s) dW_s
```

The first term gives the conditional mean α_t x_0. The Itô isometry gives the
conditional variance:

```
Var[X_t | x_0] = ∫₀ᵗ exp(-∫ₛᵗ β(u) du) · β(s) ds
```

One can verify (by differentiating or direct computation) that this equals 1 − α_t²,
which is the "variance-preserving" identity: the total variance α_t² + σ_t² = 1 is
conserved when x_0 has unit variance.

### 2.4 Reparameterisation

Any sample from p_t(x | x_0) can be written as:

```
x_t = α_t x_0 + σ_t ε,    ε ~ N(0, I)
```

This is the basis for the "noise prediction" training objective in DDPM and related methods.

### 2.5 Boundary Behaviour

| t | α_t | σ_t | p_t(x \| x_0) |
|---|-----|-----|--------------|
| 0 | 1 | 0 | δ(x − x_0) |
| 1 | ≈ 0 | ≈ 1 | ≈ N(0, I) |

With the default schedule (β_min = 0.1, β_max = 20.0):
- α₁ = exp(-1/2 · 10.05) ≈ 0.0067, so α₁² ≈ 4.5 × 10⁻⁵
- σ₁ ≈ 0.99998

The data is almost, but not exactly, destroyed at t = 1.

### 2.6 Full Marginal for a GMM

When the data distribution is a Gaussian mixture q(x_0) = Σ_k π_k N(x_0; μ_k, Σ_k),
the marginal at time t is also a GMM (this is what torchGMM exploits):

```
p_t(x) = Σ_k π_k N(x; α_t μ_k, σ_t² I + α_t² Σ_k)
```

Each component's mean shrinks by α_t and its covariance receives an additive isotropic
contribution σ_t² I. This is computed exactly in `TimeDependentGMM._gmm_t(t)`.

**This formula works for any schedule** — the only requirement is that (α_t, σ_t) define
a Gaussian conditional path p_t(x | x_0) = N(x; α_t x_0, σ_t² I). This is the key
insight that makes flow matching "free" to add: `_gmm_t` already accepts arbitrary
(α_t, σ_t) from the schedule.

### 2.7 The Reverse SDE

The time-reversed process (Anderson, 1982) is:

```
dX_t = [ -1/2 β(t) X_t  −  β(t) ∇_x log p_t(X_t) ] dt + √β(t) dW̃_t
```

where W̃_t is a reverse-time Wiener process. This is implemented in
`reverse_diffusion()` using Euler-Maruyama with negative dt.

---

## 3. Flow Matching (Conditional OT) Formulation

### 3.1 Core Idea

Flow matching (Lipman et al., 2022; Liu et al., 2022; Albergo & Vanden-Eijnden, 2022)
replaces the stochastic differential equation with an **ordinary** differential equation
(ODE). Instead of adding noise via a Wiener process, we define a deterministic velocity
field that transports samples from the data distribution to noise.

The generative ODE is:

```
dx/dt = v_t(x)
```

where v_t : R^D → R^D is a time-dependent velocity field. The key question is: how do
we choose v_t so that integrating this ODE from t = 0 to t = 1 maps data to noise (or
vice versa)?

### 3.2 Conditional Optimal Transport Path

The simplest and most popular choice is the **conditional OT path** (also called the
rectified flow or linear interpolation):

```
x_t = (1 − t) x_0 + t ε,    ε ~ N(0, I)
```

This is a straight-line interpolation between the data point x_0 and a noise sample ε.

**Conditional marginal:**

```
p_t(x | x_0) = N(x; (1 − t) x_0, t² I)
```

The mean is (1 − t) x_0 and the variance is t² I.

### 3.3 The Conditional Velocity Field

Differentiating the interpolation x_t = (1 − t) x_0 + t ε with respect to t:

```
dx_t/dt = −x_0 + ε = ε − x_0
```

This is the **conditional velocity field**: u_t(x_t | x_0) = ε − x_0. Note that it is
constant in time — a straight-line path has constant velocity.

### 3.4 The Marginal Velocity Field

The marginal velocity field (what we actually regress) is obtained by averaging over
the data distribution and the noise:

```
v_t(x) = E[ u_t(x_t | x_0) | x_t = x ]
       = E[ ε − x_0 | x_t = x ]
```

The flow matching training objective is:

```
L_FM = E_{t, x_0, ε} [ ‖ v_θ(x_t, t) − (ε − x_0) ‖² ]
```

where x_t = (1 − t) x_0 + t ε and t ~ U[0, 1].

### 3.5 Generalised Flow Matching Schedules

The conditional OT path can be generalised using arbitrary differentiable schedule
functions:

```
x_t = α_t x_0 + σ_t ε
```

with the requirements α₀ = 1, σ₀ = 0 (start at data) and α₁ = 0, σ₁ = 1 (end at
noise). The conditional velocity becomes:

```
u_t(x_t | x_0) = α̇_t x_0 + σ̇_t ε
```

where α̇_t = dα_t/dt and σ̇_t = dσ_t/dt.

For the linear (conditional OT) schedule: α_t = 1 − t, σ_t = t, giving α̇_t = −1,
σ̇_t = 1, and u_t = −x_0 + ε.

### 3.6 Boundary Behaviour

| t | α_t = 1−t | σ_t = t | p_t(x \| x_0) |
|---|-----------|---------|--------------|
| 0 | 1 | 0 | δ(x − x_0) |
| 1 | 0 | 1 | N(0, I) |

Unlike diffusion, flow matching reaches **exact** pure noise at t = 1: α₁ = 0 exactly.

### 3.7 Full Marginal for a GMM under Flow Matching

**The same closure property holds.** For q(x_0) = Σ_k π_k N(x_0; μ_k, Σ_k):

```
p_t(x) = Σ_k π_k N(x; (1−t) μ_k, t² I + (1−t)² Σ_k)
```

This is just the general formula from §2.6 with α_t = 1−t and σ_t = t. The GMM
family remains closed — torchGMM's analytical advantage carries over to flow matching
without approximation.

### 3.8 Exact Marginal Velocity for a GMM

For a GMM data distribution, the marginal velocity v_t(x) = E[ε − x_0 | x_t = x]
can be computed **in closed form**. Given the posterior weights:

```
w_k(x, t) = π_k N(x; (1−t)μ_k, t²I + (1−t)²Σ_k) / p_t(x)
```

the conditional expectations are:

```
E[x_0 | x_t = x] = Σ_k w_k(x,t) · [(1−t)⁻¹ Σ_k' (Σ_k' + (t/(1−t))² I)⁻¹ (x − (1−t)μ_k) + μ_k]
```

For diagonal covariance (as in torchGMM), this simplifies considerably. The velocity is
then v_t(x) = (x − E[x_0 | x_t = x]) / t − E[x_0 | x_t = x], or equivalently derived
from the score via the conversion formula in §5.7.

**Simpler approach**: since torchGMM already computes the exact score s_t(x) = ∇_x log p_t(x)
for any schedule, the velocity can be obtained via the score-to-velocity conversion
(§5.7) without implementing E[x_0 | x_t = x] directly.

---

## 4. How the Schedules Relate

### 4.1 The Unifying Framework

Both diffusion and flow matching produce Gaussian conditional paths of the form:

```
p_t(x | x_0) = N(x; α_t x_0, σ_t² I)
```

with the reparameterisation x_t = α_t x_0 + σ_t ε. They differ **only** in the choice
of (α_t, σ_t):

| Framework | α_t | σ_t | Constraint |
|-----------|-----|-----|------------|
| **Diffusion (VP-SDE)** | exp(−1/2 ∫₀ᵗ β(s) ds) | √(1 − α_t²) | α_t² + σ_t² = 1 |
| **Flow Matching (OT)** | 1 − t | t | α_t + σ_t = 1 |

### 4.2 Geometric Comparison

The VP-SDE schedule traces a **quarter-circle** in (α, σ) space: since α² + σ² = 1,
the point (α_t, σ_t) moves along the unit circle from (1, 0) to (≈0, ≈1).

The flow matching schedule traces a **straight line**: since α + σ = 1, the point
(α_t, σ_t) moves along the line from (1, 0) to (0, 1).

```
σ_t
 1 ─┬─────────────────────*  ← both end here (t=1)
    │                  ╱  ╱
    │               ╱   ╱
    │  VP-SDE    ╱    ╱  ← Flow matching (straight line)
    │  (arc)  ╱     ╱
    │       ╱     ╱
    │     ╱    ╱
    │   ╱   ╱
    │  ╱ ╱
 0 ─*─────────────────────
    0                     1
                         α_t
```

### 4.3 Signal-to-Noise Ratio

The log signal-to-noise ratio (log-SNR) λ_t = log(α_t² / σ_t²) decreases monotonically
from +∞ to −∞ for both schedules, but at different rates:

**Diffusion (VP-SDE):**
```
λ_t = log(α_t² / σ_t²) = log( α_t² / (1 − α_t²) )
    = −∫₀ᵗ β(s) ds − log(1 − exp(−∫₀ᵗ β(s) ds))
```

This decreases slowly at first (the exponential barely changes near t = 0) and rapidly
near the end.

**Flow matching:**
```
λ_t = log((1−t)² / t²) = 2 log((1−t)/t)
```

This decreases linearly in log-space and passes through zero at t = 0.5 (equal signal
and noise).

The SNR profile matters because it determines where the model "spends its capacity":
regions of rapid SNR change require more accurate predictions.

### 4.4 The Key Insight

**Flow matching is the "straight line" interpolation; diffusion uses exponential decay.**

The VP-SDE arises from a *physical* process (Ornstein-Uhlenbeck diffusion) where the
signal decays exponentially. The signal hangs around near its initial value for a while,
then collapses rapidly.

Flow matching simply defines a *geometric* path — the shortest straight line from data
to noise. The signal decreases linearly and uniformly.

One can see the VP-SDE schedule as a "reparameterised" version of the flow matching
schedule, where time is warped so that the α_t curve follows an exponential instead of a
line. Formally, given any monotone time-change τ: [0,1] → [0,1], one can define:

```
α̃_t = α^(diff)_{τ(t)},    σ̃_t = σ^(diff)_{τ(t)}
```

and obtain a different schedule that produces the same *set* of marginals
{p_t(x | x_0)}_{t∈[0,1]} but traverses them at a different speed.

### 4.5 Variance-Preserving vs Variance-Exploding

The VP-SDE schedule satisfies α_t² + σ_t² = 1, so x_t always has unit variance when
x_0 has unit variance. The flow matching (OT) schedule does **not** preserve variance:

```
Var[x_t | x_0 = 0] = σ_t² = t²        (flow matching)
Var[x_t | x_0 = 0] = σ_t² = 1 − α_t²  (VP-SDE, unit variance preserved)
```

For flow matching at t = 0.5, the variance is only 0.25 (compared to the VP-SDE's ≈ 0.75
at its midpoint). This means the flow matching path spends more "time" near the data
manifold and passes through a lower-variance region in the middle. This can be beneficial
for training stability.

---

## 5. Score vs Velocity

### 5.1 Two Ways to Characterise a Generative Process

In **diffusion models**, the central object is the **score function**:

```
s_t(x) = ∇_x log p_t(x)
```

In **flow matching**, the central object is the **velocity field**:

```
v_t(x) = dx_t/dt    (the time derivative of the transport map)
```

These are different quantities, but they are related.

### 5.2 Score of the Conditional

For p_t(x | x_0) = N(x; α_t x_0, σ_t² I), the score is:

```
∇_x log p_t(x | x_0) = −(x − α_t x_0) / σ_t²
```

Since x = α_t x_0 + σ_t ε, this simplifies to:

```
∇_x log p_t(x | x_0) = −ε / σ_t
```

The score points from x back toward the mean α_t x_0, scaled by 1/σ_t.

### 5.3 Conditional Velocity in Terms of Score

The conditional velocity for the general schedule x_t = α_t x_0 + σ_t ε is:

```
u_t(x | x_0) = α̇_t x_0 + σ̇_t ε
```

We can express ε in terms of x and x_0: ε = (x − α_t x_0) / σ_t. Substituting:

```
u_t(x | x_0) = α̇_t x_0 + σ̇_t (x − α_t x_0) / σ_t
             = (α̇_t − σ̇_t α_t / σ_t) x_0 + (σ̇_t / σ_t) x
```

Alternatively, using the score ∇_x log p_t = −ε / σ_t:

```
ε = −σ_t ∇_x log p_t(x | x_0)
```

So:

```
u_t(x | x_0) = α̇_t x_0 − σ̇_t σ_t ∇_x log p_t(x | x_0)
```

### 5.4 Deriving the Marginal Velocity from the Continuity Equation

The marginal velocity v_t(x) can be derived directly from the **continuity equation**
(conservation of probability mass):

```
∂_t p_t(x) + ∇ · (p_t(x) v_t(x)) = 0
```

For the general Gaussian interpolation x_t = α_t x_0 + σ_t ε with marginal
p_t(x | x_0) = N(x; α_t x_0, σ_t² I), the marginal velocity that satisfies this
continuity equation is:

```
v_t(x) = (α̇_t / α_t) x + (σ̇_t − α̇_t σ_t / α_t) σ_t · ∇_x log p_t(x)
```

**Derivation.** The marginal p_t(x) = ∫ p_t(x | x_0) q(x_0) dx_0 evolves according to
the continuity equation. For a Gaussian conditional path, the velocity that generates
this evolution decomposes into two terms:

1. A **deterministic drift** (α̇_t / α_t) x that accounts for the shrinkage of the
   signal coefficient.
2. A **score-dependent correction** proportional to ∇_x log p_t(x) that accounts for
   the growth of the noise coefficient.

The coefficient of the score term, (σ̇_t − α̇_t σ_t / α_t) σ_t, ensures that the
velocity field is consistent with the time evolution of both the mean and covariance
of each Gaussian component.

**Specialisation to flow matching** (α_t = 1 − t, σ_t = t, so α̇_t = −1, σ̇_t = 1):

```
coeff_x = α̇_t / α_t = −1 / (1 − t)

coeff_score = (σ̇_t − α̇_t σ_t / α_t) σ_t
            = (1 − (−1) · t / (1 − t)) · t
            = (1 + t/(1 − t)) · t
            = t / (1 − t)
```

Therefore:

```
v_t(x) = −x/(1 − t) + t/(1 − t) · s_t(x)
```

Or equivalently, combining over a common denominator:

```
v_t(x) = (−x + t · s_t(x)) / (1 − t)
```

**The punchline for torchGMM:** Since p_t(x) is a GMM at every t (closure under
Gaussian convolution), the score s_t(x) = ∇_x log p_t(x) is available in closed form.
This gives the **exact** marginal velocity field — no neural network needed.

**Singularity note:** At t = 1 (for flow matching, α_t = 0), the 1/α_t term diverges.
At t = 0 (σ_t = 0), the score diverges. In practice, clamp t to [ε, 1 − ε].

### 5.5 Marginal Velocity via the Probability Flow ODE

An alternative route to the velocity–score relationship uses the **probability flow ODE**
(the deterministic counterpart of the reverse SDE):

```
dx/dt = f(x, t) − 1/2 g(t)² ∇_x log p_t(x)
```

where f(x, t) = −1/2 β(t) x and g(t) = √β(t) for the VP-SDE. This gives:

```
v_t^(PF-ODE)(x) = −1/2 β(t) x − 1/2 β(t) ∇_x log p_t(x)
```

For the flow matching ODE dx/dt = v_t(x) with linear schedule, there is no separate
drift/diffusion decomposition — the continuity equation derivation in §5.4 is the
natural route.

Let us state the conversion formulas cleanly.

### 5.6 Conversion Formulas

Given the general interpolation x_t = α_t x_0 + σ_t ε, define the noise prediction
ε_θ(x, t) ≈ ε. The three common parameterisations are related as follows:

**(a) Noise prediction ↔ Score:**
```
s_t(x) = −ε_θ(x, t) / σ_t
```

**(b) Noise prediction ↔ Velocity:**
```
v_t(x) = α̇_t x_0 + σ̇_t ε
```

Since x_0 = (x − σ_t ε) / α_t (assuming α_t ≠ 0):

```
v_t(x) = α̇_t (x − σ_t ε_θ) / α_t + σ̇_t ε_θ
        = (α̇_t / α_t) x + (σ̇_t − α̇_t σ_t / α_t) ε_θ(x, t)
```

**(c) Velocity ↔ Score (the key conversion):**

Substituting ε_θ = −σ_t s_t(x):

```
v_t(x) = (α̇_t / α_t) x − (σ̇_t − α̇_t σ_t / α_t) σ_t s_t(x)
        = (α̇_t / α_t) x − (σ_t σ̇_t − α̇_t σ_t² / α_t) s_t(x)
```

Or equivalently, solving for the score given the velocity:

```
s_t(x) = [ (α̇_t / α_t) x − v_t(x) ] / (σ_t σ̇_t − α̇_t σ_t² / α_t)
```

### 5.7 Specialisation to VP-SDE and Flow Matching

**VP-SDE** (probability flow ODE parameterisation):

With f(x,t) = −1/2 β(t) x and g(t)² = β(t):

```
v_t(x) = −1/2 β(t) x − 1/2 β(t) s_t(x)

⟹  s_t(x) = −[v_t(x) + 1/2 β(t) x] / [1/2 β(t)]
            = −v_t(x) / [1/2 β(t)] − x
```

Since α̇_t / α_t = −1/2 β(t) and σ_t σ̇_t = 1/2 β(t) α_t² (which can be verified by
differentiation), this is consistent with the general formula above.

**Flow matching (linear schedule):**

With α_t = 1 − t, σ_t = t, α̇_t = −1, σ̇_t = 1:

```
v_t(x) = −x/(1−t) + [1 + t/(1−t)] ε_θ(x, t)
        = −x/(1−t) + ε_θ(x, t) / (1−t)
        = (ε_θ(x, t) − x) / (1−t)
```

And the score–velocity relation:

```
v_t(x) = −x/(1−t) − t/(1−t) · s_t(x)

⟹  s_t(x) = −[(1−t) v_t(x) + x] / t
```

These two formulas are the bridge between the existing `score()` method and the new
`velocity()` method we want to implement.

---

## 6. Implementation Plan for torchGMM

### 6.1 Design Principle: Schedule as the Single Abstraction

The key insight: `TimeDependentGMM._gmm_t(t)` constructs the marginal GMM using only
`schedule.get_alpha_t_sigma_t(t)`. It never touches β(t) directly. This means **any
schedule that provides (α_t, σ_t) works out of the box** for `log_prob`, `score`, and
`sample`.

The only place β(t) is used is in `diffusion.py` (forward/reverse SDE simulation), which
is specific to the SDE formulation and not needed for flow matching (which uses an ODE).

### 6.2 Step 1: Abstract Schedule Base Class

Extract a common interface from `BetaSchedule`:

```python
# schedule.py

class Schedule(torch.nn.Module):
    """Base class for interpolation schedules x_t = α_t x_0 + σ_t ε."""

    def get_alpha_t(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_alpha_t_sigma_t(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.get_alpha_t(t), self.get_sigma_t(t)

    def get_dalpha_dt(self, t: torch.Tensor) -> torch.Tensor:
        """dα_t/dt — needed for velocity computation."""
        raise NotImplementedError

    def get_dsigma_dt(self, t: torch.Tensor) -> torch.Tensor:
        """dσ_t/dt — needed for velocity computation."""
        raise NotImplementedError
```

`BetaSchedule` becomes a subclass that adds `beta()`, `integrated_beta()`, and
`get_t_from_lambda()`.

### 6.3 Step 2: FlowMatchingSchedule

```python
class FlowMatchingSchedule(Schedule):
    """Linear interpolation schedule: α_t = 1 − t, σ_t = t."""

    def get_alpha_t(self, t: torch.Tensor) -> torch.Tensor:
        return 1 - t

    def get_sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        return t

    def get_dalpha_dt(self, t: torch.Tensor) -> torch.Tensor:
        return torch.full_like(t, -1.0)

    def get_dsigma_dt(self, t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(t)
```

With this schedule, `TimeDependentGMM` immediately supports:
- `log_prob(x, t)` — exact, using the flow matching marginal GMM
- `score(x, t)` — exact, via autograd on the flow matching marginal
- `sample(shape, t)` — exact samples from p_t(x)

No changes to `gmm.py` needed for these core methods.

### 6.4 Step 3: velocity() Method on TimeDependentGMM

Add a `velocity(x, t)` method that computes the exact marginal velocity field:

```python
def velocity(self, x: torch.Tensor, t: torch.Tensor | float) -> torch.Tensor:
    """Exact marginal velocity v_t(x) = α̇_t/α_t · x − (σ_t σ̇_t − α̇_t σ_t²/α_t) · s_t(x).

    For flow matching (α_t = 1−t, σ_t = t):
        v_t(x) = −x/(1−t) − t/(1−t) · s_t(x)
    """
    t = self._expand_t(x, t)
    s = self.score(x, t)                           # [*N, *B, D]

    alpha_t = self.schedule.get_alpha_t(t)          # [*N, *B]
    sigma_t = self.schedule.get_sigma_t(t)          # [*N, *B]
    dalpha_dt = self.schedule.get_dalpha_dt(t)      # [*N, *B]
    dsigma_dt = self.schedule.get_dsigma_dt(t)      # [*N, *B]

    # Reshape for broadcasting with D dimension
    alpha_t = alpha_t.unsqueeze(-1)                 # [*N, *B, 1]
    sigma_t = sigma_t.unsqueeze(-1)
    dalpha_dt = dalpha_dt.unsqueeze(-1)
    dsigma_dt = dsigma_dt.unsqueeze(-1)

    # v_t(x) = (α̇_t / α_t) x − (σ_t σ̇_t − α̇_t σ_t² / α_t) s_t(x)
    coeff_x = dalpha_dt / alpha_t
    coeff_s = sigma_t * dsigma_dt - dalpha_dt * sigma_t**2 / alpha_t
    return coeff_x * x - coeff_s * s
```

This works for **any** schedule. For the flow matching schedule specifically, it
simplifies to `v_t(x) = −x/(1−t) − t/(1−t) · s_t(x)`.

**Singularity note:** At t = 0 (α_t = 1, σ_t = 0), the score diverges. At t = 1
(α_t = 0 for flow matching), the velocity formula has a 1/α_t singularity. In practice,
clamp t to [ε, 1−ε] for numerical stability, consistent with how diffusion models handle
boundary times.

### 6.5 Step 4: ODE Integration for Generation

Flow matching generates samples by integrating the ODE backwards from t = 1 → 0:

```python
def ode_sample(
    velocity_fn: callable,       # (x, t) → velocity
    x: torch.Tensor,             # [*B, D] initial noise at t=1
    t: torch.Tensor,             # [T] decreasing times from 1 to 0
) -> torch.Tensor:               # [T, *B, D] trajectory
    """Euler integration of dx/dt = v_t(x) from t=1 to t=0."""
    trajectory = [x]
    for i in range(len(t) - 1):
        dt = t[i + 1] - t[i]    # negative (going backward)
        v = velocity_fn(x, t[i])
        x = x + v * dt
        trajectory.append(x)
    return torch.stack(trajectory)
```

This is the flow matching counterpart to `reverse_diffusion`. It is simpler because
there is no stochastic term — pure Euler forward integration with negative dt.

### 6.6 Step 5: Export and API Surface

Add to `__init__.py`:

```python
from .schedule import Schedule, BetaSchedule, FlowMatchingSchedule
from .ode import ode_sample  # or add to diffusion.py
```

### 6.7 What Does NOT Change

- `_gmm_t(t)` — already schedule-agnostic (uses `get_alpha_t_sigma_t` only)
- `log_prob(x, t)` — delegates to `_gmm_t`, works as-is
- `score(x, t)` — uses autograd on `log_prob`, works as-is
- `sample(shape, t)` — delegates to `_gmm_t`, works as-is
- `Conditional` — just a single-component `TimeDependentGMM`, works as-is
- `marginal_gmm()`, `drop_mode()` — propagate `self.schedule`, work as-is

### 6.8 What Changes

| Component | Change | Scope |
|-----------|--------|-------|
| `schedule.py` | Add `Schedule` base class, `FlowMatchingSchedule` | New code |
| `BetaSchedule` | Make subclass of `Schedule`, add `get_dalpha_dt`/`get_dsigma_dt` | Small refactor |
| `gmm.py` | Add `velocity()` method | ~15 lines |
| `gmm.py` | Type hint `schedule: Schedule` instead of `BetaSchedule` | 1-line change |
| `diffusion.py` | Add `ode_sample()` function | ~20 lines |
| `__init__.py` | Export new symbols | 2-3 lines |

### 6.9 Testing Strategy

**Analytical tests (exact, no tolerance needed):**
- `FlowMatchingSchedule`: verify α_t = 1−t, σ_t = t at several time points
- Boundary: `get_alpha_t(0) == 1`, `get_sigma_t(0) == 0`, etc.

**Numerical tests (tight tolerances, using GMM closed forms):**
- `velocity()` consistency: verify `v_t(x) = −x/(1−t) − t/(1−t) · score(x, t)` with flow matching schedule
- Score-velocity roundtrip: convert score → velocity → score, check match
- `ode_sample` with exact velocity on a known GMM: samples at t=0 should match the data distribution (test via log_prob or moment matching)

**Cross-schedule tests:**
- Same GMM with both schedules at same SNR should have proportional scores
- `velocity()` with VP-SDE schedule should match the probability flow ODE drift

---

## 7. Summary Table

| Property | Diffusion (VP-SDE) | Flow Matching (Cond. OT) |
|----------|-------------------|-----------------------------|
| **Process type** | Stochastic (SDE) | Deterministic (ODE) |
| **Forward equation** | dX = −1/2β X dt + √β dW | x_t = (1−t) x_0 + t ε |
| **α_t** | exp(−1/2 ∫₀ᵗ β(s) ds) | 1 − t |
| **σ_t** | √(1 − α_t²) | t |
| **Constraint** | α² + σ² = 1 | α + σ = 1 |
| **Path geometry** | Quarter-circle (unit sphere) | Straight line (simplex) |
| **Conditional velocity** | −1/2β x − 1/2β s_t(x) | (ε − x_0) |
| **Training target** | Score s_t or noise ε | Velocity v_t = ε − x_0 |
| **Loss weighting** | Often requires σ_t⁻² or SNR weighting | Uniform (target is O(1)) |
| **Boundary at t = 1** | Approximate (α₁ ≈ 0.007) | Exact (α₁ = 0) |
| **Variance at midpoint** | ≈ 0.75 (preserved) | 0.25 (compressed) |
| **Sampling** | SDE or ODE (many steps) | ODE (fewer steps, straighter) |
| **torchGMM: log_prob** | Exact | Exact (same `_gmm_t`) |
| **torchGMM: score** | Exact | Exact (same autograd) |
| **torchGMM: velocity** | Via PF-ODE conversion | Native (§5.7 formula) |
| **torchGMM: generation** | `reverse_diffusion` (SDE) | `ode_sample` (ODE) |

---

## References

- **Song et al. (2021).** *Score-Based Generative Modeling through Stochastic Differential Equations.* ICLR 2021. — Defines the VP-SDE framework.
- **Lipman et al. (2023).** *Flow Matching for Generative Modeling.* ICLR 2023. — Introduces conditional flow matching with optimal transport paths.
- **Liu et al. (2023).** *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow.* ICLR 2023. — Rectified flow / linear interpolation perspective.
- **Albergo & Vanden-Eijnden (2023).** *Building Normalizing Flows with Stochastic Interpolants.* ICLR 2023. — Stochastic interpolant framework unifying diffusion and flow matching.
- **Anderson (1982).** *Reverse-time diffusion equation models.* Stochastic Processes and their Applications. — Time reversal of diffusion processes.
