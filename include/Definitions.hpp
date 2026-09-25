#pragma once

// Library includes
#include <Kokkos_Core.hpp>
#include <Stork_Core.hpp>

// For better coding autocompletion
#ifdef __INTELLISENSE__
#define KOKKOS_INLINE_FUNCTION inline
#define KOKKOS_LAMBDA []
#endif

namespace Condor {

    // Space definitions
    using host_space = Kokkos::DefaultHostExecutionSpace;
    using device_space = Kokkos::DefaultExecutionSpace;

    // Kokkos host memory and execution spaces
    using host_memory = Kokkos::DefaultHostExecutionSpace::memory_space;
    using host_exe = Kokkos::DefaultHostExecutionSpace::execution_space;

    // Prefer pinned host buffers for transfer-heavy staging when CUDA is available.
    #ifdef KOKKOS_ENABLE_CUDA
    using host_transfer_memory = Kokkos::CudaHostPinnedSpace;
    #else
    using host_transfer_memory = host_memory;
    #endif

    // Kokkos device memory and exectution spaces
    using device_exe = Kokkos::DefaultExecutionSpace::execution_space;
    using device_memory = Kokkos::DefaultExecutionSpace::memory_space;

    // Set memory access layout
    typedef typename device_exe::array_layout layout;

    // Set quick access to views
    typedef Kokkos::View<bool*, layout, device_memory> bool_deviceView;
    typedef Kokkos::View<bool*, layout, host_memory> bool_hostView;

    typedef Kokkos::View<bool**, layout, device_memory> bool_2D_deviceView;
    typedef Kokkos::View<bool**, layout, host_memory> bool_2D_hostView;

    typedef Kokkos::View<bool*[5], layout, device_memory> bool_x5_deviceView;
    typedef Kokkos::View<bool*[5], layout, host_memory> bool_x5_hostView;

    typedef Kokkos::View<uint8_t*, layout, device_memory> uint8_deviceView;
    typedef Kokkos::View<uint8_t*, layout, host_memory> uint8_hostView;

    typedef Kokkos::View<int*, layout, device_memory> int_deviceView;
    typedef Kokkos::View<int*, layout, host_memory> int_hostView;

    typedef Kokkos::View<int[2], layout, device_memory> int_x2_deviceView;
    typedef Kokkos::View<int[2], layout, host_memory> int_x2_hostView;

    typedef Kokkos::View<int[3], layout, device_memory> int_x3_deviceView;
    typedef Kokkos::View<int[3], layout, host_memory> int_x3_hostView;

    typedef Kokkos::View<int[6], layout, device_memory> int_x6_deviceView;
    typedef Kokkos::View<int[6], layout, host_memory> int_x6_hostView;

    typedef Kokkos::View<int**, layout, device_memory> int_2D_deviceView;
    typedef Kokkos::View<int**, layout, host_memory> int_2D_hostView;

    typedef Kokkos::View<int**[3], layout, device_memory> int_2D_x3_deviceView;
    typedef Kokkos::View<int**[3], layout, host_memory> int_2D_x3_hostView;

    typedef Kokkos::View<uint32_t*, layout, device_memory> uint32_deviceView;
    typedef Kokkos::View<uint32_t*, layout, host_memory> uint32_hostView;

}