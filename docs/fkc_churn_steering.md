# Combining Feynman-Kac Steering with Churn Sampling

**Status:** Design note

**Related documents:** [`fkc_steering.md`](fkc_steering.md), [`churn_sampler.md`](churn_sampler.md)

## 1. Summary

Feynman-Kac (FK) steering and churn sampling can be combined cleanly, but the finite-step churn
sampler should be treated as a **discrete proposal kernel**. Proposition D.6 in
[`fkc_steering.md`](fkc_steering.md) is a continuous-time generator identity and should not be
inserted unchanged into the reordered churn step.

Let

$$
\rho_t(x) = \beta_t r_t(x), \qquad
\pi_t(x) = \frac{q_t(x)e^{\rho_t(x)}}{Z_t}.
$$

If one complete churn step maps the base marginal $q_{t_i}$ to $q_{t_{i+1}}$, its exact
incremental FK weight is

$$
\boxed{
\Delta\log w_i
=
\rho_{t_{i+1}}(x_{i+1})-\rho_{t_i}(x_i).
}
$$

This construction does not require a score in the weight. The score is still needed by the
probability-flow ODE, and it must be evaluated at the reheated state and reheated time:

$$
s_{\hat t_i}(\hat x_i)=\nabla_{\hat x_i}\log q_{\hat t_i}(\hat x_i).
$$

The reordered score evaluation is therefore not an obstacle. It is part of the definition of
the churn proposal.

## 2. Churn as a discrete proposal kernel

Write one churn step as the composition

$$
x_i
\xrightarrow{K_{t_i\rightarrow\hat t_i}}
\hat x_i
\xrightarrow{\Phi_{\hat t_i\rightarrow t_{i+1}}}
x_{i+1}.
$$

Here:

- $K_{t_i\rightarrow\hat t_i}$ is the exact forward noising transition.
- $\Phi_{\hat t_i\rightarrow t_{i+1}}$ is the probability-flow ODE transport.

For the VP kernel in [`churn_sampler.md`](churn_sampler.md),

$$
\hat x_i
=
\frac{\alpha_{\hat t_i}}{\alpha_{t_i}}x_i
+
\sqrt{1-\frac{\alpha_{\hat t_i}^2}{\alpha_{t_i}^2}}\,\eta_i.
$$

With an exact score and exact ODE flow, both operators preserve the base marginal family:

$$
q_{t_i}K_{t_i\rightarrow\hat t_i}=q_{\hat t_i},
\qquad
q_{\hat t_i}\Phi_{\hat t_i\rightarrow t_{i+1}}=q_{t_{i+1}}.
$$

Consequently, their composition

$$
M_i=K_{t_i\rightarrow\hat t_i}\Phi_{\hat t_i\rightarrow t_{i+1}}
$$

satisfies $q_{t_i}M_i=q_{t_{i+1}}$ and is a valid base proposal for a discrete FK model.

## 3. Exact incremental potential

Suppose particles at $t_i$ represent

$$
\pi_{t_i}(x)\propto q_{t_i}(x)e^{\rho_{t_i}(x)}.
$$

Propagate them with the unmodified churn kernel $M_i$ and apply

$$
G_i(x_i,x_{i+1})
=
\exp\left(\rho_{t_{i+1}}(x_{i+1})-\rho_{t_i}(x_i)\right).
$$

Then

$$
\begin{aligned}
&\int q_{t_i}(x_i)e^{\rho_{t_i}(x_i)}
M_i(x_i,dx_{i+1})G_i(x_i,x_{i+1}) \\
&\quad =
e^{\rho_{t_{i+1}}(x_{i+1})}
\int q_{t_i}(x_i)M_i(x_i,dx_{i+1}) \\
&\quad =
q_{t_{i+1}}(x_{i+1})e^{\rho_{t_{i+1}}(x_{i+1})}.
\end{aligned}
$$

The unknown normalization constants cancel when particle weights are normalized.

### 3.1 Proposal propagation versus retargeting

Applying a transition does not mechanically change a particle's weight. If particles
$(x^n,w^n)$ are propagated through a Markov kernel, they initially retain the same $w^n$.
The weight changes below are **retargeting corrections**: they convert the pushforward of the
old tilted distribution into the desired tilted distribution at the new time.

If the desired target were instead defined as "the old tilted distribution propagated through
the transition," no weight update would be necessary. The correction is needed because the
chosen bridge is

$$
\pi_t(x)\propto q_t(x)e^{\rho_t(x)}
$$

at every time, which is generally different from propagating $\pi_t$ under the base dynamics.

### 3.2 Weight update through the forward kernel

Let $K_{t\rightarrow s}$ be the exact forward kernel, so

$$
q_tK_{t\rightarrow s}=q_s.
$$

Carrying the old weights unchanged produces the unnormalized measure

$$
\int q_t(x)e^{\rho_t(x)}K_{t\rightarrow s}(x,dy).
$$

This is the old tilted distribution diffused forward. It is not generally the desired bridge
$q_s(y)e^{\rho_s(y)}$. Apply the edge potential

$$
G_K(x,y)=\exp\left(\rho_s(y)-\rho_t(x)\right).
$$

Then

$$
\begin{aligned}
&\int q_t(x)e^{\rho_t(x)}
K_{t\rightarrow s}(x,dy)G_K(x,y) \\
&\quad =
e^{\rho_s(y)}
\int q_t(x)K_{t\rightarrow s}(x,dy) \\
&\quad =
q_s(y)e^{\rho_s(y)}.
\end{aligned}
$$

Thus the log-weight update is

$$
\boxed{
\Delta\log w_K=\rho_s(y)-\rho_t(x).
}
$$

It removes the ancestor's old tilt and applies the desired tilt at the newly diffused state.
The stochastic forward kernel itself introduces no additional likelihood ratio because it is
the base kernel that maps $q_t$ to $q_s$.

### 3.3 Weight update through a deterministic flow

Let

$$
y=\Phi_{t\rightarrow s}(x)
$$

and assume the exact probability-flow map satisfies

$$
\Phi_{t\rightarrow s\#}q_t=q_s.
$$

Carrying weights unchanged through this map produces

$$
q_s(y)e^{\rho_t(\Phi_{t\rightarrow s}^{-1}(y))}.
$$

Therefore the same endpoint correction gives the desired bridge:

$$
\boxed{
\Delta\log w_\Phi=\rho_s(y)-\rho_t(x).
}
$$

No explicit Jacobian appears because the Jacobian is already exactly what transforms $q_t$
into $q_s$ under $\Phi$; it cancels between the base proposal and base target.

If $\Phi$ does not exactly map $q_t$ to $q_s$, the full deterministic importance correction is

$$
\boxed{
\Delta\log w_\Phi
=
\rho_s(y)-\rho_t(x)
+
\log q_s(y)-\log q_t(x)
+
\log\left|\det D\Phi_{t\rightarrow s}(x)\right|.
}
$$

For a numerical PF-ODE solver, the Jacobian determinant and exact inverse-density correction
are usually unavailable. The practical endpoint-potential sampler therefore inherits the
base ODE integration error rather than correcting it.

This endpoint-potential form also supports a denoised reward. For example,

$$
\rho_t(x)=\beta_t r(D(x,t))
$$

defines a valid bridge as long as $\rho_t$ is reevaluated at every endpoint. No
$\dot\beta_t$, $\partial_t r$, reward Laplacian, or score-alignment term is needed for this
finite-step potential update.

## 4. Resampling before or after the ODE

The potential difference can be split at the reheated state:

$$
\Delta\log w_i^{\mathrm{churn}}
=
\rho_{\hat t_i}(\hat x_i)-\rho_{t_i}(x_i),
$$

$$
\Delta\log w_i^{\mathrm{ODE}}
=
\rho_{t_{i+1}}(x_{i+1})-\rho_{\hat t_i}(\hat x_i).
$$

The two increments telescope to the full-step weight. This gives two valid resampling
strategies:

1. Accumulate both increments and test ESS after the complete churn step.
2. Test ESS after reheating, resample at $\hat t_i$ if needed, then accumulate the ODE
   increment and test ESS again at $t_{i+1}$.

The second option is attractive because forward reheating restores particle diversity before
selection.

```text
for i in 0 .. N - 1:
    rho_i = potential(x_i, t_i)

    x_hat = forward_transition(x_i, t_i, t_hat)
    rho_hat = potential(x_hat, t_hat)
    log_w += rho_hat - rho_i
    optionally_resample(x_hat, log_w)

    x_next = probability_flow_heun(x_hat, t_hat, t_next)
    rho_next = potential(x_next, t_next)
    log_w += rho_next - rho_hat
    optionally_resample(x_next, log_w)
```

If there is no intermediate resampling, evaluating only the full-step endpoint difference is
equivalent and cheaper.

## 5. Correct score evaluation

The score follows the state through the split:

1. The exact forward transition $K_{t_i\rightarrow\hat t_i}$ requires no score.
2. The first PF-ODE evaluation uses $s_{\hat t_i}(\hat x_i)$.
3. A Heun corrector uses $s_{t_{i+1}}(x^{\mathrm{pred}}_{i+1})$.

Using $s_{t_i}(\hat x_i)$ is inconsistent: $\hat x_i$ is distributed according to
$q_{\hat t_i}$, not $q_{t_i}$. In particular, the implementation must call

```python
velocity(x_hat, t_hat)
```

rather than

```python
velocity(x_hat, t_curr)
```

after the forward transition.

The same rule applies to denoised rewards. If $r_t(x)=r(D(x,t))$, recompute
$D(\hat x_i,\hat t_i)$ after churn rather than reusing a denoiser or score evaluated before
reheating.

## 6. Lower-variance twisted churn proposal

The endpoint-potential construction is exact but can have high weight variance. Reward
information can instead be moved into the stochastic churn proposal while retaining an exact,
tractable correction.

Let the base churn transition be

$$
K_i(\hat x\mid x_i)=\mathcal N\bigl(\hat x;m_i(x_i),V_i\bigr)
$$

and sample from a reward-biased proposal

$$
\widetilde K_i(\hat x\mid x_i).
$$

The corrected full-step weight is

$$
\boxed{
\Delta\log w_i
=
\rho_{t_{i+1}}(x_{i+1})-\rho_{t_i}(x_i)
+
\log K_i(\hat x_i\mid x_i)
-
\log\widetilde K_i(\hat x_i\mid x_i).
}
$$

For a local linearization of the reheated potential, define

$$
a_i=\nabla_{\hat x}\rho_{\hat t_i}(\hat x)\big|_{\hat x=m_i(x_i)}
$$

and use the mean-shifted Gaussian

$$
\widetilde K_i
=
\mathcal N\bigl(m_i+V_i a_i,V_i\bigr).
$$

Its log-density correction is analytic:

$$
\log K_i-\log\widetilde K_i
=
-a_i^\top(\hat x_i-m_i)+\frac12a_i^\top V_i a_i.
$$

This is the discrete analogue of moving part of the FK correction into a guidance drift.
Unlike guiding the deterministic ODE, it preserves a straightforward likelihood-ratio
correction because both stochastic proposals have known Gaussian densities.

Changing the deterministic PF-ODE vector field is less convenient at finite step size. Two
different deterministic flows define singular transition kernels relative to one another
unless their inverse maps and Jacobian determinants are tracked. The safer design is therefore:

- keep the base PF-ODE unchanged;
- twist only the Gaussian churn transition;
- correct the twist with $\log K_i-\log\widetilde K_i$.

## 7. Relation to Proposition D.6

Proposition D.6 derives a continuous-time FK correction for a reverse SDE generator. Its score
term appears after part of the potential correction is moved into the drift. Generic
Feynman-Kac weighting itself does not require a score.

For a reverse grid step of size $\delta=|dt|$ and churn interval
$h=\kappa\delta$, the infinitesimal churn split converges to a reverse diffusion whose
decreasing-time drift callable and diffusion coefficient are

$$
b_\kappa(x,t)
=
f(x,t)-\frac{1+\kappa}{2}g(t)^2s_t(x),
\qquad
g_\kappa(t)=\sqrt{\kappa}\,g(t).
$$

At $\kappa=1$ this is the ordinary Anderson reverse SDE:

$$
b_1=f-g^2s,
\qquad
g_1=g.
$$

Thus the ordinary reverse generator assumed by Proposition D.6 is recovered when $\kappa=1$.
Applying its drift-versus-weight decomposition to that limit yields Eqs. (275)-(276). For
$\kappa\ne1$, the FK derivation must instead use the effective generator
$(b_\kappa,g_\kappa)$.

The untwisted endpoint potential is a different, discrete FK factorization. It does not
converge term by term to the finite-variation weight in Eq. (276): the endpoint difference
contains the stochastic increment of $\rho_t(X_t)$. Recovering the guided drift and weight of
Proposition D.6 requires the corresponding proposal change (or continuous-time Girsanov
argument), which moves that stochastic contribution from the weight into the dynamics.

For finite churn steps, the discrete endpoint-potential construction is simpler and avoids
approximating the churn split by a continuous SDE.

## 8. Exactness conditions

The discrete proof relies on $q_{t_i}M_i=q_{t_{i+1}}$. In practice:

- The exact forward kernel preserves the base marginal only when its prescribed noise scale is
  used. An EDM-style $S_{\mathrm{noise}}\ne1$ deliberately breaks this property.
- A numerical PF-ODE solver preserves the marginal only up to its integration error.
- A learned score introduces model error in the PF-ODE flow.
- A twisted churn proposal remains correct only when its density ratio is included.

Thus the steered sampler inherits the same base-model and ODE discretization error as the
unsteered churn sampler. The FK weighting adds no further bias when the endpoint potential and
any proposal-density correction are evaluated exactly.

## 9. Validation

Validation should compare the weighted particle cloud against

$$
\pi_t(x)\propto q_t(x)e^{\rho_t(x)}
$$

at every recorded time, not only at the terminal endpoint. The sweep should cover:

- churn strength, including $0$, $1$, and over-churn;
- resampling after the full step versus after the reheating sub-step;
- untwisted and mean-shifted Gaussian churn proposals;
- direct rewards $r(x)$ and denoised rewards $r(D(x,t))$.

The untwisted endpoint-potential sampler is the reference implementation. Any guided or
twisted variant should match its weighted intermediate marginals while improving ESS.
