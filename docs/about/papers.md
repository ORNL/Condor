---
title: Papers
nav_order: 4
---

# Papers

These are the Benjamin Stump papers directly relevant to Condor's thermal
kernel, integration, parallelization, solidification outputs, calibration/CET,
and sparse coupling. The list was cross-checked against Benjamin Stump's Google
Scholar link and ORNL publication record; publications on cellular automata,
alloys, tribology, and other topics are not part of Condor's mathematical
derivation.

## Core numerical method

1. B. Stump and A. Plotkowski, “An adaptive integration scheme for heat
   conduction in additive manufacturing,” *Applied Mathematical Modelling* 75,
   787–805 (2019). [doi:10.1016/j.apm.2019.07.008](https://doi.org/10.1016/j.apm.2019.07.008).
   This is the primary source for adaptive quadrature and selective melt-pool
   evaluation implemented in Condor.

2. B. Stump and A. Plotkowski, “Spatiotemporal parallelization of
   an analytical heat conduction model for additive manufacturing via a hybrid
   OpenMP + MPI approach,” *Computational Materials Science* 184, 109861
   (2020). [doi:10.1016/j.commatsci.2020.109861](https://doi.org/10.1016/j.commatsci.2020.109861).
   This provides the parallelization context for independent space-time work.

## Model fidelity and process use

3. B. Stump, A. Plotkowski, and J. Coleman, “Solidification dynamics in metal
   additive manufacturing: analysis of model assumptions,” *Modelling and
   Simulation in Materials Science and Engineering* 29, 035001 (2021).
   [doi:10.1088/1361-651X/abca19](https://doi.org/10.1088/1361-651X/abca19).
   It compares conduction-only predictions with added physics and motivates
   explicit model limitations and calibration.

4. B. Stump, “An algorithm for physics informed scan path optimization in
   additive manufacturing,” *Computational Materials Science* 212, 111566
   (2022). [doi:10.1016/j.commatsci.2022.111566](https://doi.org/10.1016/j.commatsci.2022.111566).
   This supplies the CET convention $\Delta T_c=(aV)^{1/n}$ used by Condor's
   equiaxed-fraction expression and provides scan-optimization context.

5. S. Bi, B. Stump, J. Zhang, Y. Lee, J. Coleman, M. Bement, and G. Zhang,
   “Blackbox optimization for approximating high-fidelity heat transfer
   calculations in metal additive manufacturing,” *Results in Materials* 13,
   100258 (2022).
   [doi:10.1016/j.rinma.2022.100258](https://doi.org/10.1016/j.rinma.2022.100258).
   This provides the model-calibration context. Condor's current `Calibrate`
   mode uses its own grid or gradient-descent search over two effective material
   factors rather than the paper's Bayesian or directional-smoothing methods.