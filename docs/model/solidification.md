---
title: Solidification Quantities
parent: User Guide
nav_order: 5
---

# Solidification Quantities

`Interface` mode detects a cooling crossing between one timestep's liquid
temperature and the next timestep's solid temperature. It refines that bracket
with a safeguarded secant iteration: a secant proposal is used when it remains
inside the bracket, otherwise the midpoint is used. Refinement stops when

$$
\left|1-\frac{T}{T_L}\right| < \epsilon_T
$$

or `max_iter` is reached. `dT_err` is $\epsilon_T$. The reported `tSol` is a
refined estimate, not an unconditional exact root.

## Gradient, cooling rate, and interface speed

For one quadrature contribution

$$
f=A\exp\!\left[-3\sum_i \Delta x_i^2\phi_i\right],
\qquad \phi_i=\Phi_i^{-1},
$$

the mathematically consistent derivatives are

$$
\frac{\partial f}{\partial x_i}=f(-6\Delta x_i\phi_i),
$$

$$
\frac{\partial^2 f}{\partial x_i^2}
=f\left[36\Delta x_i^2\phi_i^2-6\phi_i\right].
$$

After summing contributions, the intended solidification outputs are

$$
G=\lVert\nabla T\rVert \quad [\mathrm{K/m}],
\qquad
\widehat{\mathbf{G}}=\frac{\nabla T}{G},
$$

$$
\dot T=\alpha\nabla^2T+\dot T_{\mathrm{source}},
\qquad
\text{cooling rate}=|\dot T| \quad [\mathrm{K/s}],
$$

$$
V=\frac{|\dot T|}{G}\quad [\mathrm{m/s}].
$$

This $V$ follows from differentiating the isotherm condition
$T(\mathbf{x}(t),t)=T_L$ in the local normal direction.

## Melting, remelting, and stored fields

Standalone `Interface` output treats the solidification columns as the current
state of each tracked point, rather than as an archive of every crossing. Condor
buffers at most one pending update per point between transfers to the host:

- A solid-to-liquid transition increments `numMelt` and stores a molten marker.
  When that marker reaches the host, `tSol`, `G`, `V`, `dTdt`, and the requested
  gradient components are reset to zero without running the solidification
  refinement.
- A subsequent liquid-to-solid transition replaces the pending molten marker
  with its temperature bracket. Condor then refines the crossing and stores the
  new solidification quantities.
- If several transitions occur before the next transfer, the sparse entry holds
  the point's latest state while `numMelt` retains the total number of melting
  transitions.

Consequently, a liquid point has zero-valued solidification fields in ordinary
CSV output, and a resolidified point contains values from its most recent
crossing. `numMelt` counts solid-to-liquid transitions; it is not a count of
completed melt/solidification pairs.

Coupled RDF output remains event-oriented. Melting updates the point's melt time
and count, but RDF publishes the resulting completed solidification event rather
than a separate molten-marker event.

## Columnar-to-equiaxed estimate

The optional CET hook evaluates

$$
\phi_{eq}=1-\exp\!\left[
-\frac{4\pi N_0}{3}
\frac{(aV)^{3/n}}{[G(n+1)]^3}
\right].
$$

$N_0$ is the nucleation-site density [m$^{-3}$], $n$ is dimensionless, and
$a$ belongs to the convention $\Delta T_c=(aV)^{1/n}$, or
$V=(\Delta T_c)^n/a$. Its units are K$^n$ s m$^{-1}$. This is the convention
used by Condor's `pow(a*V, 3/n)` implementation and the cited scan-optimization
paper. The expression is an empirical, local CET estimate and
inherits the accuracy and assumptions of $G$ and $V$. Requesting `eqFrac`
automatically requests those two solidification fields internally. `eqFrac` is
meaningful only where a valid solidification crossing exists; values at liquid
points should not be interpreted as solidification conditions.
