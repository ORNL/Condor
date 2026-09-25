---
title: Thermal Model
parent: User Guide
nav_order: 4
---

# Thermal Model

## Governing problem

Condor models linear heat conduction in a homogeneous material,

$$
\rho c_p\frac{\partial T}{\partial t}=k\nabla^2T+q(\mathbf{x},t),
\qquad
\alpha=\frac{k}{\rho c_p},
$$

where $T$ is temperature [K], $k$ is conductivity [W m$^{-1}$ K$^{-1}$],
$\rho$ is density [kg m$^{-3}$], $c_p$ is specific heat
[J kg$^{-1}$ K$^{-1}$], and $\alpha$ is diffusivity [m$^2$ s$^{-1}$].
All four material quantities are spatially and temporally constant.

For absorbed power $\eta Q$, the ellipsoidal Gaussian Green-function
convolution used by Condor is

$$
T(\mathbf{x},t)-T_0 =
\frac{2\eta Q}{\rho c_p(\pi/3)^{3/2}}
\int_0^t
\frac{m(t')}{\sqrt{\Phi_x\Phi_y\Phi_z}}
\exp\!\left[-3\sum_{i\in\{x,y,z\}}
\frac{(x_i-x_{b,i}(t'))^2}{\Phi_i}\right]dt',
$$

with

$$
\Phi_i(t,t')=\sigma_i^2+12\alpha(t-t').
$$

$T_0$ is the initial temperature, $m(t')$ is the path power modifier,
$\mathbf{x}_b(t')$ is the beam location, and $\sigma_i$ is the input source
width along axis $i$ [m]. The `width_x`, `width_y`, and `width_z`
values are this model's Gaussian width parameters; they are not beam diameters.
At zero diffusion time, comparison with the conventional form
$\exp[-x_i^2/(2s_i^2)]$ gives $\sigma_i=\sqrt{6}\,s_i$.

## Source paths and superposition

Temperature is linear in source history. Contributions from every enabled
segment and every beam/path pair are summed. A line segment interpolates
$\mathbf{x}_b(t')$ from its previous endpoint to its endpoint at the specified
speed. A spot segment holds its endpoint for its dwell time. `Pmod` multiplies
absorbed power over that segment.

Configured boundary planes add reflected copies of the quadrature nodes. These
are same-sign image sources and therefore approximate insulating/zero-normal-flux
planes.

## Adaptive history integration

Condor uses the adaptive strategy developed by Stump and evaluates the time convolution
with Gauss--Legendre rules of order 2, 4, 8, or 16. It uses the nondimensional lag

$$
s=\frac{t-t'}{t_c},\qquad t_c=\frac{\sigma_x^2}{\alpha}.
$$

Near-source intervals receive higher order. Stationary-spot history intervals
grow by powers of two in $s$. For a moving line, the implemented physical step
scale is

$$
\Delta t' \sim
\frac{\sqrt{\ln\sqrt{2}}\,\sigma_x}{v}\sqrt{12s+1},
$$

clipped to segment boundaries; older history is smoother and can use wider,
lower-order intervals.

{: .note }
The settings `quadrature.cutoff_peak` and `cutoff_t0tl` are parsed and used in
spatial search-radius estimates, but the current `t0calc` implementation resets
the history start to zero. They do not currently truncate the temporal
convolution.

## Scope and assumptions

The solution assumes linear conduction, uniform constant properties, and the
Gaussian effective source. It omits fluid advection in the liquid, latent heat,
radiative and convective surface losses, evaporation, recoil pressure, powder
resolution, and deformation. The source is consequently an effective model:
calibration can absorb some omitted physics, but does not add that physics.

See [Papers]({{ '/about/papers/' | relative_url }}) for the derivation lineage.
