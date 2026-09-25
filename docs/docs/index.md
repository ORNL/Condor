---
title: User Guide
nav_order: 2
has_children: true
---

# User Guide

Use this guide to build Condor, run the command-line application, and understand
how scan-path input becomes thermal and solidification output.

1. [Install Condor](installation/) and its Kokkos, MPI, OpenMP, Stork, and
   nlohmann/json dependencies.
2. Follow the [quick start](quick-start/) to configure and run a case.
3. Read [How Condor works](how-condor-works/) for the source-history,
   temperature-evaluation, and interface-tracking workflow.
4. Use the [thermal model]({{ '/model/thermal-model/' | relative_url }}) and
   [solidification quantities]({{ '/model/solidification/' | relative_url }})
   pages for the governing equations, assumptions, and derivative-field caveat.
5. Start from the [examples]({{ '/examples/' | relative_url }}) when preparing a
   new case.

The [reference]({{ '/reference/' | relative_url }}) documents input files, run
modes, output fields, and the C++ coupling API.
