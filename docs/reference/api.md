---
title: Library API and Coupling
parent: Reference
nav_order: 3
---

# Library API and Coupling

Include the installed public header and initialize MPI and Kokkos in the caller.
The caller must keep both runtimes alive for the entire Condor call.

## Standalone library call

```cpp
#include <Condor_Core.hpp>
#include <Kokkos_Core.hpp>
#include <mpi.h>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  {
    Condor::Run::Classic<double>(argc, argv);
  }
  Kokkos::finalize();
  MPI_Finalize();
}
```

`Classic` reads `argv[1]`, or `ParamInput.json` when it is absent, and dispatches
the selected run mode.

## Produce RDF in memory

```cpp
Stork::Structs::RDF_Dual<double> rdf;
Condor::Run::Coupled<double>(rdf, "ParamInput.json");
```

The input must select `Interface`. The caller-owned RDF receives solidification
events for downstream interpolation or microstructure simulation. Melting
updates the melt time and melt count associated with the next solidification;
it does not add a standalone molten-marker event to the RDF.

## Produce SRDF in memory

```cpp
Stork::Structs::SRDF_Dual<double> srdf;
Condor::Run::Coupled<double>(srdf, "ParamInput.json");
```

SRDF stores sparse interface snapshots. Stork can interpolate the resulting
container to RDF for downstream consumers.

## Ownership and behavior

- Condor does not initialize or finalize MPI/Kokkos in these entry points.
- Coupled overloads fill a container supplied by the caller.
- Coupled mode disables normal CSV output and user JSON hooks.
- `float` and `double` paths are instantiated.
- The source currently exposes only the two Stork container overloads as the
  public thermal-data coupling contract.
