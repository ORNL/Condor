#!/bin/bash

# Clean previous build
rm -rf build
clear

# Set environment variables
export NVCC_WRAPPER_DEFAULT_COMPILER=mpic++

# Set environment variables for external library installation paths.
export STORK_DIR=$HOME/stork/build/install
export JSON_DIR=$HOME/build/install
export MPI_DIR=/usr/lib/x86_64-linux-gnu/openmpi

# Create build directory
mkdir -p build
pushd build

# CMake configuration
cmake \
  -D CMAKE_BUILD_TYPE="Release" \
  -D CMAKE_INSTALL_PREFIX=install \
  -D CMAKE_CXX_FLAGS="-fopenmp -O3 -ffast-math -march=znver3 -mtune=znver3" \
  -D CMAKE_PREFIX_PATH="${KOKKOS_DIR};${MPI_DIR};${STORK_DIR};${JSON_DIR}" \
  -D CMAKE_CUDA_ARCHITECTURES="86" \
  ..

# Build the project
make -j 8 install  # Use all available CPU cores for faster compilation

# Return to original directory
popd
