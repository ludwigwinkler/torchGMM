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
6. [Practical Implications](#6-practical-implications)
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
| x₀ | Data sample, drawn from data distribution q(x₀) |
| ε | Standard Gaussian noise, ε ~ N(0, I) |
| t | Time index, t ∈ [0, 1]. t = 0 is data, t = 1 is noise |
| D | Ambient dimension |
| β(t) | Instantaneous noise rate (diffusion schedule) |
| α_t | Signal coefficient at time t |
| σ_t | Noise coefficient at time t |
| p_t(x) | Marginal density at time t |
| p_t(x \| x₀) | Conditional density given a fixed starting point x₀ |

**Convention (matching torchGMM):** t = 0 corresponds to clean data and t = 1
corresponds to (approximately) pure noise.

---

## 2. Diffusion (VP-SDE) Formulation

### 2.1 The Forward SDE

The Variance-Preserving SDE (VP-SDE) defines a continuous-time noising process:

```
dX_t = -½ β(t) X_t dt + √β(t) dW_t
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
∫₀ᵗ β(s) ds = β_min · t + ½ (β_max − β_min) · t²
```

### 2.3 Conditional Marginals

Because the forward SDE is linear in X_t with Gaussian noise, the conditional
distribution p_t(x | x₀) is Gaussian at every t:

```
p_t(x | x₀) = N(x; α_t x₀, σ_t² I)
```

where the **signal coefficient** α_t and **noise coefficient** σ_t are:

```
α_t = exp( -½ ∫₀ᵗ β(s) ds )

σ_t² = 1 − α_t²
```

**Derivation sketch.** The OU process dX = -½β X dt + √β dW has the explicit solution:

```
X_t = exp(-½ ∫₀ᵗ β(s) ds) · X₀  +  ∫₀ᵗ exp(-½ ∫ₛᵗ β(u) du) · √β(s) dW_s
```

The first term gives the conditional mean α_t x₀. The Itô isometry gives the
conditional variance:

```
Var[X_t | X₀] = ∫₀ᵗ exp(-∫ₛᵗ β(u) du) · β(s) ds
```

One can verify (by differentiating or direct computation) that this equals 1 − α_t²,
which is the "variance-preserving" identity: the total variance α_t² + σ_t² = 1 is
conserved when x₀ has unit variance.

### 2.4 Reparameterisation

Any sample from p_t(x | x₀) can be written as:

```
x_t = α_t x₀ + σ_t ε,    ε ~ N(0, I)
```

This is the basis for the "noise prediction" training objective in DDPM and related methods.

### 2.5 Boundary Behaviour

| t | α_t | σ_t | p_t(x \| x₀) |
|---|-----|-----|--------------|
| 0 | 1 | 0 | δ(x − x₀) |
| 1 | ≈ 0 | ≈ 1 | ≈ N(0, I) |

With the default schedule (β_min = 0.1, β_max = 20.0):
- α₁ = exp(-½ · 10.05) ≈ 0.0067, so α₁² ≈ 4.5 × 10⁻⁵
- σ₁ ≈ 0.99998

The data is almost, but not exactly, destroyed at t = 1.

### 2.6 Full Marginal for a GMM

When the data distribution is a Gaussian mixture q(x₀) = Σ_k π_k N(x₀; μ_k, Σ_k),
the marginal at time t is also a GMM (this is what torchGMM exploits):

```
p_t(x) = Σ_k π_k N(x; α_t μ_k, σ_t² I + α_t² Σ_k)
```

Each component's mean shrinks by α_t and its covariance receives an additive isotropic
contribution σ_t² I. This is computed exactly in `TimeDependentGMM._gmm_t(t)`.

### 2.7 The Reverse SDE

The time-reversed process (Anderson, 1982) is:

```
dX_t = [ -½ β(t) X_t  −  β(t) ∇_x log p_t(X_t) ] dt + √β(t) dW̃_t
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
x_t = (1 − t) x₀ + t ε,    ε ~ N(0, I)
```

This is a straight-line interpolation between the data point x₀ and a noise sample ε.

**Conditional marginal:**

```
p_t(x | x₀) = N(x; (1 − t) x₀, t² I)
```

The mean is (1 − t) x₀ and the variance is t² I.

### 3.3 The Conditional Velocity Field

Differentiating the interpolation x_t = (1 − t) x₀ + t ε with respect to t:

```
dx_t/dt = −x₀ + ε = ε − x₀
```

This is the **conditional velocity field**: u_t(x_t | x₀) = ε − x₀. Note that it is
constant in time — a straight-line path has constant velocity.

### 3.4 The Marginal Velocity Field

The marginal velocity field (what we actually regress) is obtained by averaging over
the data distribution and the noise:

```
v_t(x) = E[ u_t(x_t | x₀) | x_t = x ]
       = E[ ε − x₀ | x_t = x ]
```

The flow matching training objective is:

```
L_FM = E_{t, x₀, ε} [ ‖ v_θ(x_t, t) − (ε − x₀) ‖² ]
```

where x_t = (1 − t) x₀ + t ε and t ~ U[0, 1].

### 3.5 Generalised Flow Matching Schedules

The conditional OT path can be generalised using arbitrary differentiable schedule
functions:

```
x_t = α_t x₀ + σ_t ε
```

with the requirements α₀ = 1, σ₀ = 0 (start at data) and α₁ = 0, σ₁ = 1 (end at
noise). The conditional velocity becomes:

```
u_t(x_t | x₀) = α̇_t x₀ + σ̇_t ε
```

where α̇_t = dα_t/dt and σ̇_t = dσ_t/dt.

For the linear (conditional OT) schedule: α_t = 1 − t, σ_t = t, giving α̇_t = −1,
σ̇_t = 1, and u_t = −x₀ + ε.

### 3.6 Boundary Behaviour

| t | α_t = 1−t | σ_t = t | p_t(x \| x₀) |
|---|-----------|---------|--------------|
| 0 | 1 | 0 | δ(x − x₀) |
| 1 | 0 | 1 | N(0, I) |

Unlike diffusion, flow matching reaches **exact** pure noise at t = 1: α₁ = 0 exactly.

---

## 4. How the Schedules Relate

### 4.1 The Unifying Framework

Both diffusion and flow matching produce Gaussian conditional paths of the form:

```
p_t(x | x₀) = N(x; α_t x₀, σ_t² I)
```

with the reparameterisation x_t = α_t x₀ + σ_t ε. They differ **only** in the choice
of (α_t, σ_t):

| Framework | α_t | σ_t | Constraint |
|-----------|-----|-----|------------|
| **Diffusion (VP-SDE)** | exp(−½ ∫₀ᵗ β(s) ds) | √(1 − α_t²) | α_t² + σ_t² = 1 |
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
{p_t(x | x₀)}_{t∈[0,1]} but traverses them at a different speed.

### 4.5 Variance-Preserving vs Variance-Exploding

The VP-SDE schedule satisfies α_t² + σ_t² = 1, so x_t always has unit variance when
x₀ has unit variance. The flow matching (OT) schedule does **not** preserve variance:

```
Var[x_t | x₀ = 0] = σ_t² = t²        (flow matching)
Var[x_t | x₀ = 0] = σ_t² = 1 − α_t²  (VP-SDE, unit variance preserved)
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

For p_t(x | x₀) = N(x; α_t x₀, σ_t² I), the score is:

```
∇_x log p_t(x | x₀) = −(x − α_t x₀) / σ_t²
```

Since x = α_t x₀ + σ_t ε, this simplifies to:

```
∇_x log p_t(x | x₀) = −ε / σ_t
```

The score points from x back toward the mean α_t x₀, scaled by 1/σ_t.

### 5.3 Conditional Velocity in Terms of Score

The conditional velocity for the general schedule x_t = α_t x₀ + σ_t ε is:

```
u_t(x | x₀) = α̇_t x₀ + σ̇_t ε
```

We can express ε in terms of x and x₀: ε = (x − α_t x₀) / σ_t. Substituting:

```
u_t(x | x₀) = α̇_t x₀ + σ̇_t (x − α_t x₀) / σ_t
             = (α̇_t − σ̇_t α_t / σ_t) x₀ + (σ̇_t / σ_t) x
```

Alternatively, using the score ∇_x log p_t = −ε / σ_t:

```
ε = −σ_t ∇_x log p_t(x | x₀)
```

So:

```
u_t(x | x₀) = α̇_t x₀ − σ̇_t σ_t ∇_x log p_t(x | x₀)
```

### 5.4 Marginal Velocity–Score Relationship

After marginalising over x₀, the relationship between the marginal velocity v_t(x) and
the marginal score s_t(x) = ∇_x log p_t(x) depends on the schedule. For the general
framework x_t = α_t x₀ + σ_t ε, the **probability flow ODE** (the deterministic
counterpart of the reverse SDE) is:

```
dx/dt = f(x, t) − ½ g(t)² ∇_x log p_t(x)
```

where f(x, t) = −½ β(t) x and g(t) = √β(t) for the VP-SDE. This gives:

```
v_t^(PF-ODE)(x) = −½ β(t) x − ½ β(t) ∇_x log p_t(x)
```

For the flow matching ODE dx/dt = v_t(x) with linear schedule, there is no separate
drift/diffusion decomposition. Instead:

```
v_t(x) = (σ̇_t / σ_t) x − (σ̇_t σ_t − α̇_t / α_t · σ_t²) / σ_t · (−σ_t s_t(x))
```

Let us state the conversion formulas cleanly.

### 5.5 Conversion Formulas

Given the general interpolation x_t = α_t x₀ + σ_t ε, define the noise prediction
ε_θ(x, t) ≈ ε. The three common parameterisations are related as follows:

**(a) Noise prediction ↔ Score:**
```
s_t(x) = −ε_θ(x, t) / σ_t
```

**(b) Noise prediction ↔ Velocity:**
```
v_t(x) = α̇_t x₀ + σ̇_t ε
```

Since x₀ = (x − σ_t ε) / α_t (assuming α_t ≠ 0):

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

### 5.6 Specialisation to VP-SDE and Flow Matching

**VP-SDE** (probability flow ODE parameterisation):

With f(x,t) = −½ β(t) x and g(t)² = β(t):

```
v_t(x) = −½ β(t) x − ½ β(t) s_t(x)

⟹  s_t(x) = −[v_t(x) + ½ β(t) x] / [½ β(t)]
            = −v_t(x) / [½ β(t)] − x
```

Since α̇_t / α_t = −½ β(t) and σ_t σ̇_t = ½ β(t) α_t² (which can be verified by
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
v_t(x) = −x/(1−t) − [t + t²/(1−t)] s_t(x)
        = −x/(1−t) − t/(1−t) s_t(x)

⟹  s_t(x) = −[(1−t) v_t(x) + x] / t
```

---

## 6. Practical Implications

### 6.1 When to Use Each Schedule

**Prefer diffusion (VP-SDE) when:**
- You have an existing infrastructure built around score matching or DDPM.
- You need the reverse SDE (with stochasticity), e.g. for temperature-scaled sampling.
- You want the variance-preserving property (all intermediate distributions have
  unit variance when the data does), which can simplify normalisation.
- Your model is a `TimeDependentGMM` in torchGMM — the entire library is built around
  the VP-SDE schedule, with exact score computation.

**Prefer flow matching (OT) when:**
- You want **faster sampling** — the straight-line paths require fewer ODE steps.
- You want **simpler training** — the velocity target ε − x₀ is constant in t and does
  not require weighting by σ_t.
- You want exact boundary conditions (α₁ = 0 exactly, not approximately).
- You are building a continuous normalizing flow (CNF) and want the simplest ODE.

### 6.2 Converting a Trained Model

If you have a network f_θ(x, t) trained under one framework and want to use it under the
other, the conversion requires knowing the schedule functions at inference time:

**Score model → Velocity model:**
```python
def score_to_velocity(score_fn, x, t, schedule):
    """Convert a VP-SDE score model to a velocity field."""
    beta_t = schedule.beta(t)
    s = score_fn(x, t)
    return -0.5 * beta_t * x - 0.5 * beta_t * s
```

**Velocity model → Score model:**
```python
def velocity_to_score(velocity_fn, x, t, schedule):
    """Convert a velocity field to a score function (VP-SDE schedule)."""
    beta_t = schedule.beta(t)
    v = velocity_fn(x, t)
    return -(v + 0.5 * beta_t * x) / (0.5 * beta_t)
```

These conversions are exact — no retraining needed. However, the **training objective
weights** differ between score matching and flow matching, so a model trained to minimise
one loss is not necessarily optimal for the other.

### 6.3 Schedule Reparameterisation

Given a trained diffusion model with schedule (α_t^diff, σ_t^diff), one can define an
equivalent flow matching model by time-warping. Define the reparameterisation
τ: [0,1] → [0,1] such that:

```
1 − τ(t) = α_t^diff    ⟹    τ(t) = 1 − α_t^diff
```

Then the flow matching velocity in the new time coordinate is:

```
ṽ_τ(x) = v_t(x) / (dτ/dt) = v_t(x) · (−1 / α̇_t^diff)
```

This is a useful trick when you want to reuse a VP-SDE model with an ODE solver that
assumes a linear schedule.

### 6.4 Impact on torchGMM

The `TimeDependentGMM` class uses the VP-SDE schedule internally. To use it with a flow
matching schedule, you would:

1. Define a custom schedule class with `get_alpha_t(t) = 1 - t` and `get_sigma_t(t) = t`.
2. Pass it to `TimeDependentGMM(mu, sigma, weight, schedule=custom_schedule)`.
3. Use `gmm.score(x, t)` as before — the score is exact regardless of the schedule.

The conversion to a velocity field would then use the formulas from Section 5.6.

---

## 7. Summary Table

| Property | Diffusion (VP-SDE) | Flow Matching (Cond. OT) |
|----------|-------------------|--------------------------|
| **Process type** | Stochastic (SDE) | Deterministic (ODE) |
| **Forward equation** | dX = −½β X dt + √β dW | x_t = (1−t) x₀ + t ε |
| **α_t** | exp(−½ ∫₀ᵗ β(s) ds) | 1 − t |
| **σ_t** | √(1 − α_t²) | t |
| **Constraint** | α² + σ² = 1 | α + σ = 1 |
| **Path geometry** | Quarter-circle (unit sphere) | Straight line (simplex) |
| **Conditional velocity** | −½β x − ½β s_t(x) | (ε − x₀) |
| **Training target** | Score s_t or noise ε | Velocity v_t = ε − x₀ |
| **Loss weighting** | Often requires σ_t⁻² or SNR weighting | Uniform (target is O(1)) |
| **Boundary at t = 1** | Approximate (α₁ ≈ 0.007) | Exact (α₁ = 0) |
| **Variance at midpoint** | ≈ 0.75 (preserved) | 0.25 (compressed) |
| **Sampling** | SDE or ODE (many steps) | ODE (fewer steps, straighter) |
| **torchGMM support** | Full (native schedule) | Via custom schedule |

---

## References

- **Song et al. (2021).** *Score-Based Generative Modeling through Stochastic Differential Equations.* ICLR 2021. — Defines the VP-SDE framework.
- **Lipman et al. (2023).** *Flow Matching for Generative Modeling.* ICLR 2023. — Introduces conditional flow matching with optimal transport paths.
- **Liu et al. (2023).** *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow.* ICLR 2023. — Rectified flow / linear interpolation perspective.
- **Albergo & Vanden-Eijnden (2023).** *Building Normalizing Flows with Stochastic Interpolants.* ICLR 2023. — Stochastic interpolant framework unifying diffusion and flow matching.
- **Anderson (1982).** *Reverse-time diffusion equation models.* Stochastic Processes and their Applications. — Time reversal of diffusion processes.
