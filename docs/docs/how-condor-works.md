---
title: How Condor Works
parent: User Guide
nav_order: 3
---

# How Condor Works

Condor separates source-history integration from spatial temperature
evaluation. For each simulation time, it prepares a compact set of quadrature
nodes that represents the active heat-source history, then reuses those nodes
to evaluate the temperature at many spatial points.

The main calculation proceeds as follows:

1. At each requested time $t$, Condor divides every active path segment into
   adaptive history intervals. Gauss--Legendre nodes encode the emission time,
   source position, diffused source widths, power, and quadrature weight.
2. Condor transfers the prepared quadrature data from the CPU to the selected
   Kokkos execution device.
3. Kokkos team kernels independently sum the quadrature-node contributions at
   the selected grid points. This pointwise independence provides substantial
   parallelism on both CPUs and accelerators.
4. When interface tracking is enabled, Condor revisits previously liquid
   material and adds seed points along the path traveled since the preceding
   timestep. It then expands the search on the device to identify the points
   needed to describe the current melt pool.
5. When solidification output is requested, Condor refines each bracketed
   liquidus crossing on the CPU with a safeguarded secant iteration. This step
   requires repeated temperature evaluations to estimate the solidification
   time within the configured tolerance.

## Interface tracking

`Interface` mode advances through the scan path with a fixed timestep. At each
step, it begins with surface columns reached by the beam and liquid columns
retained from the previous step. Condor searches downward from the top surface
for points at or above the liquidus temperature $T_L$, then follows the
connected liquid region until its boundary is resolved.

This approach evaluates the active melt-pool region instead of reconstructing
the entire three-dimensional temperature field at every timestep. Additional
temperatures are reconstructed only when an output or a refined
liquidus-crossing calculation requires them. After the final scan segment,
Condor continues advancing until the tracked liquid region has solidified.

Phase changes also update a sparse per-point event buffer. Melting increments
the point's melt count and queues a marker that clears stale solidification
fields on the host. Resolidification replaces that marker with a liquid/solid
temperature bracket, so only the point's latest state is communicated while the
total number of melting transitions is retained.

Optional hooks use the tracked interface to calculate solidification
conditions, melt-pool dimensions, column depth, and columnar-to-equiaxed
transition (CET) estimates. The same mode also supports direct coupling to
Stork RDF and SRDF containers. See
[Solidification Quantities]({{ '/model/solidification/' | relative_url }}) for
the definitions and current implementation notes for $G$, $V$, cooling rate,
and CET.

## Snapshots

`Snapshots` mode evaluates the thermal field at explicitly requested times.
Times may be given in seconds or as percentages of the longest scan duration,
which makes it convenient to compare equivalent stages of different scan
paths.

Two evaluation strategies are available:

- `tracking: "None"` evaluates every point in the regular grid and produces a
  dense three-dimensional thermal profile. Use this option when temperatures
  outside the melted region are important.
- `tracking: "Interface"` applies the melt-pool tracker at each requested time
  and writes only the melted or tracked region. This reduces computation and
  output volume when the objective is a quick melt-pool estimate rather than a
  full-domain temperature field.

The optional melt-pool hook measures length, width, depth, and per-column melt
depth at each snapshot. Unlike `Interface` mode, snapshots do not march through
every fixed timestep between requested outputs; each requested time is prepared
and evaluated directly from the source history.

## Calibration

`Calibrate` mode adjusts the effective thermal-model parameters so that
predicted melt-pool dimensions match supplied targets. Each candidate scales
thermal diffusivity and volumetric heat capacity, runs an interface-tracked
snapshot calculation, measures the resulting melt pool, and evaluates the
configured loss for length, width, and/or depth.

The line evaluator compares candidates against target dimensions for a single
straight scan. The path evaluator uses the configured input path and can
compare measurements at multiple times. Targets may be weighted to emphasize
the dimensions that matter most for a particular experiment.

Condor can search the candidate space with an iteratively refined grid or with
finite-difference gradient descent. The selected result is written as an
updated material file, while the optional history records how the loss and
candidate parameters evolved. Calibration identifies effective parameters for
this model; it does not introduce omitted physics such as fluid flow or latent
heat. See [Modes and Hooks]({{ '/reference/modes/' | relative_url }}) for the
available evaluators, optimizers, and settings.

## MPI parallel model

Once the source history has been prepared, the temperature at a grid point does
not depend on temperatures at neighboring points. Condor exploits this
independence with Kokkos within each rank and decomposes the regular domain in
$x$ and $y$ across MPI ranks. Coupling metadata records each rank's local
bounds and its neighboring ranks. The coupled SRDF route uses the
one-dimensional decomposition required by that data path.

{: .warning }
Melt-pool dimensions and other nonlocal quantities depend on points that may be
owned by different MPI ranks. They are not generally valid under spatial
decomposition unless the calculation explicitly combines the required data
across ranks. Pointwise temperature and solidification calculations do not have
this limitation.
