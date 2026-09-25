# Condor

Condor is a performance-portable C++ application and library for
semi-analytical thermal simulation of metal additive manufacturing. It uses
Kokkos, OpenMP, and MPI to evaluate laser scan histories on regular grids,
track liquid/solid interfaces, and calculate melt-pool and solidification
quantities.

Condor supports interface tracking, requested-time thermal snapshots, and
material calibration. It can write standalone CSV results or populate Stork
RDF and SRDF containers in memory for coupled microstructure workflows.

The core numerical method and its parallelization are presented in
[Stump and Plotkowski (2019)](https://doi.org/10.1016/j.apm.2019.07.008) and
[Stump and Plotkowski (2020)](https://doi.org/10.1016/j.commatsci.2020.109861).

## Documentation

The full installation guide, quick start, input reference, model description,
output reference, and library API documentation are on the
[Condor documentation site](https://ornl.github.io/Condor/).

## Installation

Condor requires Kokkos, MPI, OpenMP, Stork, and nlohmann/json. See the
[installation guide](https://ornl.github.io/Condor/docs/installation/) for
dependency and configuration details.

After setting the dependency paths and architecture options in `make.sh`, build
and install with:

```sh
./make.sh
```

## Usage

Run Condor from a case directory so relative input paths resolve correctly:

```sh
cd /path/to/case
mpiexec -n <ranks> /path/to/Condor/build/apps/Main ParamInput.json
```

A downstream CMake project can import the installed library target with:

```cmake
find_package(Condor CONFIG REQUIRED)
target_link_libraries(my_target PRIVATE Condor::Condor)
```

The public API supports standalone execution and coupled overloads that fill
caller-owned Stork RDF or SRDF containers.

## Contributors

- [Benjamin Stump](https://www.ornl.gov/staff-profile/benjamin-stump)

## Contributing

Contributions are welcome, including new thermal models, scan strategies,
coupling workflows, validation cases, performance improvements, and output
capabilities.
