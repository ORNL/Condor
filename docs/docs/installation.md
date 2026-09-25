---
title: Installation
parent: User Guide
nav_order: 1
---

# Installation

Condor requires a C++17 compiler, CMake 3.16 or newer, MPI, OpenMP, Kokkos,
Stork, and `nlohmann_json`. All dependencies must have compatible compiler and
backend settings; in particular, Condor and Stork must use a Kokkos build that
targets the same execution space.

## Configure with CMake

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PWD/build/install" \
  -DCMAKE_PREFIX_PATH="/path/to/kokkos;/path/to/stork;/path/to/json"
cmake --build build -j
cmake --install build
```

This builds the `Condor` library and one executable for each source in `apps/`.
Executables are placed in `build/apps`; the installed CMake package and public
headers are placed under `build/install`.

The repository's `make.sh` is a machine-specific example. It deletes `build/`
and contains local dependency paths, CPU flags, and a CUDA architecture, so
review it before use.

## Consume the installed library

```cmake
find_package(Condor REQUIRED)
add_executable(my_thermal_driver main.cpp)
target_link_libraries(my_thermal_driver PRIVATE Condor::Condor)
```

Add Condor and dependency prefixes to `CMAKE_PREFIX_PATH` if CMake cannot locate
them.

## Supported scalar types

The library explicitly instantiates internal paths for `float` and `double`.
The supplied `Main` executable uses `double`.
