# FKC Fokker--Planck equation and the churn split

This note replaces the earlier claim that churn-ratio reweighting and guided
denoising categorically cannot compose. That claim was too strong.

They **can** compose, but the weight for the guided deterministic operator is
not the Proposition D.6 / Eq. (276) FKC integrand by itself. It is a
compressibility correction. There are therefore two valid, distinct
factorisations:

1. the **continuous-time FKC factorisation**, which treats churn as the
   diffusion part of an effective reverse SDE and uses the usual FKC
   integrand; and
2. the **operator-local factorisation**, which retargets after the exact churn
   kernel and then corrects the guided deterministic flow by its
   compressibility.

Do not add the weights from these two factorisations. They describe the same
change of measure in different decompositions.

The notation follows the paper *Feynman-Kac Correctors in Diffusion:
Annealing, Guidance, and Product of Experts*, Proposition D.6, whose
derivation is transcribed in [`fkc_steering.md`](fkc_steering.md). The
operator-local guided-flow result is derived in
[`fkc_churn_steering.md`](fkc_churn_steering.md).

## 1. Base and tilted marginals

Let schedule time $t$ increase from data to noise. The base forward SDE is

$$
dX_t = f_t(X_t)\,dt + g_t\,dW_t,
\tag{1}
$$

so its marginal $q_t$ obeys

$$
\partial_t q_t
= -\nabla\!\cdot(f_t q_t) + \frac{g_t^2}{2}\Delta q_t.
\tag{2}
$$

The factor $g_t^2/2$ is the diffusion rate. It is **not** the marginal
variance $\sigma_t^2$, except in special parameterisations. Over a short
forward interval $h$, the diffusion part is exactly one additive term

$$
\frac{g_t^2}{2}\Delta q_t\,h.
\tag{3}
$$

Define an energy-tilted target bridge

$$
\pi_t(x) = \frac{q_t(x)e^{-\beta(t)U(x,t)}}{Z_t}.
\tag{4}
$$

All spatial derivatives below are derivatives in $x$. If the energy is
time-dependent, $\partial_t[\beta(t)U(x,t)]$ is its partial derivative at
fixed $x$.

## 2. The FKC Fokker--Planck equation

It is easiest to state Proposition D.6 in reverse time
$\tau=T-t$, which increases from noise to data. Write

$$
s_t(x)=\nabla\log q_t(x),
\qquad
b_t(x)=-f_t(x)+g_t^2s_t(x).
\tag{5}
$$

The ordinary reverse SDE has drift $b_t$ and diffusion $g_t$. Its
Fokker--Planck equation is the reverse-time version of (2).

FKC uses the positive guidance magnitude

$$
a_t=\beta(t)\frac{g_t^2}{2},
\qquad
\text{guidance drift}=-a_t\nabla_xU(x,t).
\tag{6}
$$

The tilted bridge (4) then satisfies

$$
\boxed{
\partial_\tau\pi_t
 =-\nabla\!\cdot\!\left[\pi_t(b_t-a_t\nabla_xU)\right]
   +\frac{g_t^2}{2}\Delta\pi_t
   +\pi_t\left(G_t-\mathbb E_{\pi_t}[G_t]\right),
}
\tag{7}
$$

where

$$
\boxed{
G_t(x)
 = -\partial_\tau[\beta(t)U(x,t)]
   -\left\langle\nabla_x[\beta(t)U(x,t)],
      \frac{g_t^2}{2}s_t(x)-f_t(x)\right\rangle .
}
\tag{8}
$$

Equation (7) is the Feynman--Kac Fokker--Planck equation. It says that a
particle process with guided drift $b-a\nabla_xU$, diffusion $g$, and incremental
log-weight $G_t\,d\tau$ represents $\pi_t$.

### Where the energy Laplacian went

Before choosing (6), adding the drift $-a\nabla_xU$ contributes

$$
-\nabla\!\cdot(-\pi_t a\nabla_x U)
=\nabla\!\cdot(\pi_t a\nabla_x U)
=\pi_t\left[a\Delta U+
  a\langle\nabla\log\pi_t,\nabla_x U\rangle\right]
\tag{9}
$$

to the weight equation. Independently, substituting
$\pi_t\propto q_te^{-\beta(t)U}$ into the diffusion term in (7) produces

$$
-\pi_t\,\beta(t)\frac{g_t^2}{2}\Delta U.
\tag{10}
$$

Choosing $a=\beta(t)g_t^2/2$ makes the guidance contribution in (9) cancel
the corresponding energy-Laplacian term when the tilted FPE is rearranged
into transport plus weight. This is
the paper's ``magic'' coefficient. The cancellation is between the
**guidance divergence** and the single diffusion term
$g_t^2\Delta\pi_t/2$ in the Fokker--Planck equation.

Thus the intuition in the question is correct: the diffusion is one additive
$g_t^2\Delta\pi_t/2$ term. The subtlety is only how a finite churn split
realises that term.

## 3. One churn cycle as two base operators

### Diffusion-only Fokker--Planck equation

For a drift-free diffusion step

$$
dX_t=\sigma_{\mathrm{diff}}(t)\,dW_t,
\tag{11}
$$

the density being diffused obeys the vanilla heat equation

$$
\boxed{
\partial_t\pi_t
=\frac{\sigma_{\mathrm{diff}}(t)^2}{2}\Delta\pi_t.
}
\tag{12}
$$

Thus the proposed equation

$$
\partial_t\pi_t=\frac{\sigma_t^2}{2}\Delta\pi_t
\tag{13}
$$

is correct when $\sigma_t$ denotes the **instantaneous SDE diffusion
coefficient**, as it does in the FKC paper. This document uses $g_t$ for that
quantity because elsewhere in this repository $\sigma_t$ denotes the
**marginal noise standard deviation** in
$X_t=\alpha_tX_0+\sigma_t\varepsilon$. In this notation the same equation is

$$
\boxed{
\partial_t\pi_t=\frac{g_t^2}{2}\Delta\pi_t.
}
\tag{14}
$$

Now substitute the energy-tilted density

$$
\pi_t(x)=\frac{q_t(x)e^{-\beta(t)U(x,t)}}{Z_t}.
\tag{15}
$$

At fixed $t$, both $\beta(t)$ and $Z_t$ are constant under spatial
differentiation. Therefore

$$
\nabla_x\pi_t
=\pi_t\left[s_t-\beta(t)\nabla_xU\right],
\qquad
s_t=\nabla_x\log q_t,
\tag{16}
$$

For the second spatial derivative, first apply the Laplacian to the product:

$$
\Delta\pi_t
=\frac{1}{Z_t}\Delta\left(q_t e^{-\beta(t)U}\right).
\tag{17a}
$$

The Laplacian product rule is

$$
\Delta(fg)=g\Delta f+2\langle\nabla_x f,\nabla_x g\rangle+f\Delta g,
\tag{17b}
$$

so

$$
\Delta\pi_t
=\frac{1}{Z_t}\left[
\Delta q_t e^{-\beta(t)U}
+2\left\langle\nabla_xq_t,\nabla_xe^{-\beta(t)U}\right\rangle
+q_t\Delta e^{-\beta(t)U}
\right].
\tag{17c}
$$

The gradient of the energy factor is

$$
\nabla_xe^{-\beta(t)U}
=-\beta(t)e^{-\beta(t)U}\nabla_xU.
\tag{17d}
$$

Taking its divergence gives

$$
\begin{aligned}
\Delta e^{-\beta(t)U}
&=\nabla_x\!\cdot\!\left[-\beta(t)e^{-\beta(t)U}\nabla_xU\right]\\
&=-\beta(t)e^{-\beta(t)U}\Delta U
  -\beta(t)\left\langle\nabla_xe^{-\beta(t)U},\nabla_xU\right\rangle\\
&=e^{-\beta(t)U}
\left[\beta(t)^2\left\|\nabla_xU\right\|^2-\beta(t)\Delta U\right].
\end{aligned}
\tag{17e}
$$

Substituting (17d) and (17e) into (17c) and factoring out the common
exponential yields

$$
\boxed{
\Delta\pi_t
=\frac{e^{-\beta(t)U}}{Z_t}
\left[
\Delta q_t
-2\beta(t)\left\langle\nabla_xq_t,\nabla_xU\right\rangle
-\beta(t)q_t\Delta U
+\beta(t)^2q_t\left\|\nabla_xU\right\|^2
\right].
}
\tag{17f}
$$

Hence the vanilla diffusion-only Fokker--Planck equation can be written
explicitly as

$$
\boxed{
\partial_t\pi_t
=\frac{g_t^2}{2}\frac{e^{-\beta(t)U}}{Z_t}
\left[
\Delta q_t
-2\beta(t)\left\langle\nabla_xq_t,\nabla_xU\right\rangle
-\beta(t)q_t\Delta U
+\beta(t)^2q_t\left\|\nabla_xU\right\|^2
\right].
}
\tag{18}
$$

The energy Laplacian therefore appears directly as
$-\pi_t\beta(t)g_t^2\Delta U/2$. The diffusion kernel itself remains
energy-blind; $\Delta U$ appears because the density being diffused already
contains $e^{-\beta(t)U}$.

For the infinitesimal effective churn process, the injected diffusion
coefficient is $\sqrt{\gamma}\,g_t$, so its diffusion-only contribution is

$$
\boxed{
\partial_\tau\pi_t\big|_{\mathrm{churn}}
=\frac{\gamma g_t^2}{2}\Delta\pi_t.
}
\tag{19}
$$

This equation describes only the stochastic churn contribution. The
probability-flow denoising operator contributes a separate deterministic
transport term. For VE schedules the churn kernel is literally additive
Gaussian diffusion; for a general VP schedule the exact forward transition
also includes its linear scaling/drift, so its full finite-step operator is
not diffusion-only even though its stochastic generator contribution has
the form above.

Let $h=t_i-t_{i+1}>0$ be a reverse-grid step. Churn first moves forwards from
$t_i$ to

$$
\hat t_i=t_i+\gamma h,
\tag{20}
$$

then probability-flow denoising moves from $\hat t_i$ down to
$t_{i+1}$. Let $K_i$ be the exact forward transition and $\Phi_i$ the exact
probability-flow map:

$$
x_i \xrightarrow{K_i} \hat x_i
\xrightarrow{\Phi_i} x_{i+1}.
\tag{21}
$$

They separately preserve the base family:

$$
q_{t_i}K_i=q_{\hat t_i},
\qquad
q_{\hat t_i}\Phi_i=q_{t_{i+1}}.
\tag{22}
$$

The probability-flow velocity in schedule time is

$$
v_t=f_t-\frac{g_t^2}{2}s_t,
\qquad
\partial_tq_t=-\nabla\!\cdot(q_tv_t).
\tag{23}
$$

The first equality in (22) is exact at finite step size; the second is exact
for the exact ODE flow and otherwise has the ODE solver's error.

### Infinitesimal generator of the split

For a small step, the churn kernel injects variance

$$
V_i=\gamma g_{t_i}^2h+O(h^2).
\tag{24}
$$

Hence churn supplies precisely the positive diffusion contribution

$$
\frac{\gamma g_t^2}{2}\Delta\mu\,d\tau
\tag{25}
$$

for any smooth density $\mu(x)$ acted on by the churn kernel, where
$\Delta\mu=\sum_{j=1}^D\partial_{x_j}^2\mu$. It is the spatial Laplacian of
the density, not the energy Laplacian $\Delta U$. The deterministic leg is longer: it travels
$(1+\gamma)h$ in reverse time. Combining its score transport with (25), the
split has the effective reverse SDE generator

$$
\boxed{
b_\gamma=-f+\frac{1+\gamma}{2}g^2s,
\qquad
g_\gamma^2=\gamma g^2.
}
\tag{26}
$$

Indeed,

$$
\begin{aligned}
-\nabla\!\cdot(q_tb_\gamma)
 +\frac{g_\gamma^2}{2}\Delta q_t
&=\nabla\!\cdot(f_tq_t)
 -\frac{1+\gamma}{2}g_t^2\Delta q_t
 +\frac{\gamma}{2}g_t^2\Delta q_t\\
&=\nabla\!\cdot(f_tq_t)-\frac{g_t^2}{2}\Delta q_t
=\partial_\tau q_t .
\end{aligned}
\tag{27}
$$

This is the requested allocation: churn contributes the one additive
$\gamma g^2\Delta/2$ diffusion term; the denoising leg supplies the remaining
score transport. Neither half alone is the reverse SDE.

## 4. Churn operator on a tilted distribution

At the beginning of a step, the weighted particle cloud represents

$$
\pi_t(x)=\frac{q_t(x)e^{-\beta(t)U(x,t)}}{Z_t}.
\tag{28}
$$

The churn kernel does not evaluate $U$ and contains no energy-dependent
coefficient. Infinitesimally, it is the energy-blind heat operator

$$
\mathcal C_h\mu
=\mu+\frac{\gamma g_t^2}{2}\Delta\mu\,h+O(h^2)
\tag{29}
$$

acting on whichever density $\mu$ is supplied to it. Supplying the tilted
density means setting $\mu=\pi_t$ in (29). Although the operator is
energy-blind, the spatial shape of its input is not.

At fixed $t$, $\beta(t)$ and $Z_t$ are constant with respect to $x$. The
first derivative of the tilted density is

$$
\nabla_x\pi_t
=\pi_t\left[s_t-\beta(t)\nabla_xU\right],
\qquad
s_t=\nabla_x\log q_t.
\tag{30}
$$

Applying the product rule a second time gives

$$
\boxed{
\Delta\pi_t
=\frac{e^{-\beta(t)U}}{Z_t}
\left[
\Delta q_t
-2\beta(t)\left\langle\nabla_xq_t,\nabla_xU\right\rangle
-\beta(t)q_t\Delta U
+\beta(t)^2q_t\left\|\nabla_xU\right\|^2
\right].
}
\tag{31}
$$

Substituting (31) into the churn diffusion term
$(\gamma g_t^2/2)\Delta\pi_t$ gives the churn contribution in standard
Fokker--Planck form:

with zero drift,

$$
\boxed{
\begin{aligned}
\left.\partial_\tau\pi_t\right|_{\mathrm{churn}}
&=-\nabla_x\!\cdot\!\left[0\cdot\pi_t\right]
  +\frac{\gamma g_t^2}{2}\Delta\pi_t,\\[4pt]
&=\frac{\gamma g_t^2}{2}\frac{e^{-\beta(t)U}}{Z_t}
  \left[
    \Delta q_t
    -2\beta(t)\left\langle\nabla_xq_t,\nabla_xU\right\rangle
    -\beta(t)q_t\Delta U
    +\beta(t)^2q_t\left\|\nabla_xU\right\|^2
  \right].
\end{aligned}
}
\tag{32}
$$

Thus applying churn to an already tilted distribution produces
energy-gradient, energy-score-alignment, energy-Laplacian, and squared
energy-gradient terms. These terms arise from differentiating the input
density $\pi_t$; they are not parameters of the churn kernel.

Equation (32) is the diffusion-only Fokker--Planck contribution with $t$
frozen; its drift is zero and its diffusion coefficient is
$\sqrt{\gamma}\,g_t$. A finite exact churn transition also changes the schedule time from
$t_i$ to $\hat t_i$. Its exact operator-local correction is the endpoint
ratio derived in Section 6, rather than a finite-step use of the expansion
above.

## 5. Continuous-time FKC applied to the effective SDE

Apply the Proposition D.6 algebra to (26). Its diffusion is
$g_\gamma^2=\gamma g^2$. Before choosing the guidance coefficient, let
$a(t)$ be an arbitrary scalar. The original tilted Fokker--Planck equation is

$$
\boxed{
\partial_\tau\pi_t
=-\nabla_x\!\cdot\!\left[
  \pi_t\left(b_\gamma-a(t)\nabla_xU\right)
\right]
+\frac{\gamma g_t^2}{2}\Delta\pi_t
+\pi_t\left(G_t^a-\mathbb E_{\pi_t}[G_t^a]\right).
}
\tag{33}
$$

The corresponding guided effective reverse SDE is

$$
dX_\tau
=\left[b_\gamma(X_\tau,t)-a(t)\nabla_xU(X_\tau,t)\right]d\tau
+\sqrt{\gamma}\,g_t\,dW_\tau,
\tag{34}
$$

and its full FKC log-weight potential before any cancellation is

$$
\begin{aligned}
G_t^a(x)
=\;&-\partial_\tau[\beta(t)U(x,t)] \\
&+\left\langle\nabla_xU(x,t),\;
  -\beta(t)\left[
    b_\gamma(x,t)-\gamma g_t^2s_t(x)
    +\beta(t)\frac{\gamma g_t^2}{2}\nabla_xU(x,t)
  \right]\right.\\
&\hspace{7.8em}\left.
  -a(t)\left[s_t(x)-\beta(t)\nabla_xU(x,t)\right]
\right\rangle\\
&+\left[-a(t)+\beta(t)\frac{\gamma g_t^2}{2}\right]\Delta U(x,t).
\end{aligned}
\tag{35}
$$

The last line is the energy-Laplacian contribution. It remains for every
general choice of $a(t)$. Now choose

$$
a(t)=\beta(t)\frac{\gamma g_t^2}{2},
\qquad
\text{guidance drift}=-a(t)\nabla_xU(x,t)
=-\beta(t)\frac{\gamma g_t^2}{2}\nabla_xU(x,t).
\tag{36}
$$

Equation (37) is obtained from Equation (31) by multiplying by
$\gamma g_t^2/2$ and retaining only the terms containing $U$. Equivalently,
these are the energy-dependent terms on the second line of Equation (32).
Thus the energy-dependent part of the churn diffusion term in (33) is

$$
\left.\frac{\gamma g_t^2}{2}\Delta\pi_t\right|_{U}
=\pi_t\frac{\gamma g_t^2}{2}
\left[
  -2\beta(t)\langle s_t,\nabla_xU\rangle
  -\beta(t)\Delta U
  +\beta(t)^2\|\nabla_xU\|^2
\right].
\tag{37}
$$

With the choice (36), the energy-dependent guided-flux contribution is

$$
\begin{aligned}
+\nabla_x\!\cdot\!\left[\pi_ta(t)\nabla_xU\right]
&=\pi_t\frac{\gamma g_t^2}{2}
\left[
  \beta(t)\Delta U
  +\beta(t)\langle s_t,\nabla_xU\rangle
  -\beta(t)^2\|\nabla_xU\|^2
\right],
\end{aligned}
\tag{38}
$$

where
$\nabla_x\log\pi_t=s_t-\beta(t)\nabla_xU$. Adding (37) and (38) gives

$$
\boxed{
\left.\frac{\gamma g_t^2}{2}\Delta\pi_t\right|_{U}
+\nabla_x\!\cdot\!\left[\pi_ta(t)\nabla_xU\right]
=-\pi_t\frac{\beta(t)\gamma g_t^2}{2}
  \langle s_t,\nabla_xU\rangle.
}
\tag{39}
$$

The $\Delta U$ and $\|\nabla_xU\|^2$ terms cancel exactly. The remaining
score-alignment term combines with the score transport already present in
$b_\gamma$ and with the target's explicit time derivative. After those
terms are collected, the FKC log-weight integrand is

$$
\boxed{
G_t
=-\partial_\tau[\beta(t)U(x,t)]
 -\left\langle\nabla_x[\beta(t)U(x,t)],\frac{g_t^2}{2}s_t-f_t\right\rangle ,
}
\tag{40}
$$

which is independent of $\gamma$. This is a generator statement: it applies
to the **whole effective SDE**, not separately to a reweighted churn kernel
and a deterministic map.

To deliver the reverse-time guidance displacement on a deterministic leg
of length $(1+\gamma)h$, that leg receives the split field

$$
a_{\rm split}(t)
=\beta(t)\frac{\gamma}{1+\gamma}\frac{g_t^2}{2},
\qquad
\text{split guidance drift}=-a_{\rm split}(t)\nabla_xU(x,t).
\tag{41}
$$

The field is smaller than (36) because the deterministic leg lasts
$(1+\gamma)$ times the grid interval.

This explains the implementation placement of the weight update. In the
continuous-time FKC factorisation, unweighted churn already supplies the
diffusion term in (33), while the deterministic leg supplies the matching
guidance flux. Their energy-Laplacian and squared-gradient terms cancel only
after the two generator contributions are added. The surviving potential
(40) is a bounded-variation update with no stochastic increment, so it can
be accumulated once per complete split step, conventionally immediately
before the deterministic denoising move. It should not be interpreted as an
operator-local weight belonging to denoising alone.

## 6. Operator-local factorisation: churn ratio plus guided flow

One may instead retarget explicitly after **each** operator. First apply the
exact churn ratio:

$$
\Delta\log w_{\rm churn}
=\beta(t_i)U(x_i,t_i)-\beta(\hat t_i)U(\hat x_i,\hat t_i).
\tag{42}
$$

Because $q_{t_i}K_i=q_{\hat t_i}$, this is exact and turns a weighted cloud
for $\pi_{t_i}$ into one for $\pi_{\hat t_i}$. This is exactly what
`TestChurnOperatorWeights::test_churn_operator_weights` proves.

Now guide the deterministic leg by an arbitrary schedule-time field $u_t$:

$$
\dot x_t=v_t(x_t)+u_t(x_t).
\tag{43}
$$

The guided flow no longer maps $q_{\hat t_i}$ to $q_{t_{i+1}}$. Its exact
operator-local correction is

$$
\boxed{
\Delta\log w_{\rm guided\ ODE}
=\beta(\hat t_i)U(\hat x_i,\hat t_i)-\beta(t_{i+1})U(x_{i+1},t_{i+1})
+\int_{\hat t_i}^{t_{i+1}}
\left[\nabla\!\cdot u_t(x_t)
      +\langle s_t(x_t),u_t(x_t)\rangle\right]dt .
}
\tag{44}
$$

The limits in (44) are decreasing, so the integral has a negative sign for a
positive integrand. The last two terms are the guided flow's compressibility
relative to $q_t$:

$$
\nabla\!\cdot u+\langle s,u\rangle
=q_t^{-1}\nabla\!\cdot(q_tu).
\tag{45}
$$

Adding (42) and (44) is valid:

$$
\boxed{
\Delta\log w_{\rm cycle}
=\beta(t_i)U(x_i,t_i)-\beta(t_{i+1})U(x_{i+1},t_{i+1})
+\int_{\hat t_i}^{t_{i+1}}
\left[\nabla\!\cdot u_t+\langle s_t,u_t\rangle\right]dt .
}
\tag{46}
$$

This answers the original question precisely: **yes**, churn reweighting can
be followed by a guided drift. What is required is (44), not the
continuous-time FKC integrand (40).

For $u=0$, (46) reduces to the endpoint-ratio weight. For a gradient guide,
$u=-c_t\beta(t)\nabla_xU$, its divergence contains $-c_t\beta(t)\Delta U$; therefore this
operator-local route generally requires an energy Laplacian.

## 7. What must not be added

The following is invalid:

$$
\left[\beta(t)U(x,t)-\beta(\hat t)U(\hat x,\hat t)\right]
+\left[G_t\,h\right]
\tag{47}
$$

when the particle also receives the guided deterministic field from (41).
The first term is part of the operator-local factorisation (42)--(46); the
second is from the effective-SDE factorisation (33)--(40). Equation (47)
uses two descriptions of the same churn diffusion without including the
compressibility term (44) that makes the actual guided map valid.

The valid alternatives are:

| Proposal and bookkeeping | Incremental log weight |
| --- | --- |
| Exact churn + unguided PF flow | Endpoint ratio $\beta(t_i)U(x_i,t_i)-\beta(t_{i+1})U(x_{i+1},t_{i+1})$ |
| Exact churn + guided PF flow $v+u$ | Equation (46): endpoint ratio plus guided-flow compressibility |
| Effective churn SDE + FKC guidance | Equation (40), using the matching effective guide (36) / split field (41) |
| Mean-shifted churn kernel + unguided PF flow | Endpoint ratio plus the exact Gaussian proposal-density ratio |

The first two are finite-step operator statements. The third is the
continuous-time FKC statement and is accurate to the order of the split.
They agree in the fine-grid limit but do not have equal per-step weights.
