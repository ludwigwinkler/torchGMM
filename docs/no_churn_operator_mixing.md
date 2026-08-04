# Continuous FKC Guidance for Churn Sampling

`steered_reverse_churn_sampling` is a continuous Feynman--Kac-Corrector
construction.  It does not assign separate target weights to its stochastic
reheat and deterministic transport operators.

For a decreasing schedule-time grid, each step re-noises from $t_i$ to

$$
\hat t_i=\min(t_i+\gamma(t_i-t_{i+1}),1)
$$

with the exact transition, then transports from $\hat t_i$ to $t_{i+1}$.
The state is distributed at the reheated time after the transition, so all
model-dependent work uses $(\hat x_i,\hat t_i)$.

The caller supplies two matched objects:

1. a probability-flow velocity plus any guidance field;
2. `weight_update(x_hat, t_hat)`, the continuous FKC log-weight rate for
   that guided process; the solver multiplies it by `t_curr - t_next`.

In the fine-grid effective process, churn strength $\gamma$ changes the
diffusion to $\sqrt{\gamma}\,g$.  For a reward tilt, use

$$
a=\frac{\beta\gamma g^2}{2}
$$

in the effective guidance.  Because the deterministic transport spans
$(1+\gamma)$ grid intervals, divide the field passed to the sampler by
$1+\gamma$.  The corresponding continuous FKC integrand is
gamma-invariant: it is evaluated at the reheated pair and multiplied by the
base reverse-grid magnitude, not by the transport duration.

This placement is essential.  The exact transition supplies the stochastic
part of the effective process, while the guided probability-flow leg supplies
the deterministic part.  Their FKC terms are derived together; a callback
that compensates only one operator does not describe the guided process.

See [`edm_fkc_steering.md`](edm_fkc_steering.md) for the complete sampler
contract, guidance scaling, and FKC weight semantics.
