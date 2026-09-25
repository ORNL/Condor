#pragma once

#include "Definitions.hpp"
#include "impl/Calc/Structs.hpp"

namespace Condor::impl::Grid {

    KOKKOS_INLINE_FUNCTION int ijk_to_p(
        const int i,
        const int j,
        const int k,
        const int y_num,
        const int z_num)
    {
        return i * (y_num * z_num) + j * z_num + k;
    }

    template<typename FloatType>
    KOKKOS_INLINE_FUNCTION int ijk_to_p(
        const int i,
        const int j,
        const int k,
        const Calc::Device_Constants<FloatType>& const_d)
    {
        return ijk_to_p(
            i,
            j,
            k,
            const_d.ints(IntConstants::Y_NUM),
            const_d.ints(IntConstants::Z_NUM));
    }

    KOKKOS_INLINE_FUNCTION int p_to_i(
        const int p,
        const int y_num,
        const int z_num)
    {
        return p / (y_num * z_num);
    }

    KOKKOS_INLINE_FUNCTION int p_to_j(
        const int p,
        const int y_num,
        const int z_num)
    {
        return (p / z_num) % y_num;
    }

    KOKKOS_INLINE_FUNCTION int p_to_k(
        const int p,
        const int z_num)
    {
        return p % z_num;
    }

    template<typename FloatType>
    KOKKOS_INLINE_FUNCTION int p_to_i(
        const int p,
        const Calc::Device_Constants<FloatType>& const_d)
    {
        return p_to_i(p, const_d.ints(IntConstants::Y_NUM), const_d.ints(IntConstants::Z_NUM));
    }

    template<typename FloatType>
    KOKKOS_INLINE_FUNCTION int p_to_j(
        const int p,
        const Calc::Device_Constants<FloatType>& const_d)
    {
        return p_to_j(p, const_d.ints(IntConstants::Y_NUM), const_d.ints(IntConstants::Z_NUM));
    }

    template<typename FloatType>
    KOKKOS_INLINE_FUNCTION int p_to_k(
        const int p,
        const Calc::Device_Constants<FloatType>& const_d)
    {
        return p_to_k(p, const_d.ints(IntConstants::Z_NUM));
    }

    KOKKOS_INLINE_FUNCTION int ij_to_v2d(
        const int i,
        const int j,
        const int y_num)
    {
        return i * y_num + j;
    }

    KOKKOS_INLINE_FUNCTION int v2d_to_i(
        const int v2d,
        const int y_num)
    {
        return v2d / y_num;
    }

    KOKKOS_INLINE_FUNCTION int v2d_to_j(
        const int v2d,
        const int y_num)
    {
        return v2d % y_num;
    }

    template<typename FloatType>
    KOKKOS_INLINE_FUNCTION int top_surface_point(
        const int v2d,
        const Calc::Device_Constants<FloatType>& const_d)
    {
        const int y_num = const_d.ints(IntConstants::Y_NUM);
        return ijk_to_p(
            v2d_to_i(v2d, y_num),
            v2d_to_j(v2d, y_num),
            const_d.ints(IntConstants::Z_NUM) - 1,
            const_d);
    }

    template<typename FloatType>
    struct InitializeDevicePoints {
        using float_device_view_t = Kokkos::View<FloatType*, layout, device_memory>;

        float_device_view_t x_p;
        float_device_view_t y_p;
        float_device_view_t z_p;
        int y_num;
        int z_num;
        FloatType x0;
        FloatType y0;
        FloatType z0;
        FloatType dx;
        FloatType dy;
        FloatType dz;

        KOKKOS_INLINE_FUNCTION
        void operator()(const int p) const {
            const int i = p_to_i(p, y_num, z_num);
            const int j = p_to_j(p, y_num, z_num);
            const int k = p_to_k(p, z_num);

            x_p(p) = x0 + static_cast<FloatType>(i) * dx;
            y_p(p) = y0 + static_cast<FloatType>(j) * dy;
            z_p(p) = z0 + static_cast<FloatType>(k) * dz;
        }
    };

    template<typename FloatType>
    struct Device_Points {
        using float_device_view_t = Kokkos::View<FloatType*, layout, device_memory>;

        float_device_view_t x_p;
        float_device_view_t y_p;
        float_device_view_t z_p;

        Device_Points() = default;

        explicit Device_Points(const Simdat<FloatType>& sim)
            : x_p(Kokkos::ViewAllocateWithoutInitializing("grid_x_p"), sim.domain.pnum),
              y_p(Kokkos::ViewAllocateWithoutInitializing("grid_y_p"), sim.domain.pnum),
              z_p(Kokkos::ViewAllocateWithoutInitializing("grid_z_p"), sim.domain.pnum)
        {
            const int p_num = sim.domain.pnum;
            const int y_num = sim.domain.ynum;
            const int z_num = sim.domain.znum;
            const FloatType x0 = (sim.domain.xnum == 1) ? sim.domain.xmax : sim.domain.xmin;
            const FloatType y0 = (sim.domain.ynum == 1) ? sim.domain.ymax : sim.domain.ymin;
            const FloatType z0 = (sim.domain.znum == 1) ? sim.domain.zmax : sim.domain.zmin;
            const FloatType dx = (sim.domain.xnum == 1) ? static_cast<FloatType>(0.0) : sim.domain.xres;
            const FloatType dy = (sim.domain.ynum == 1) ? static_cast<FloatType>(0.0) : sim.domain.yres;
            const FloatType dz = (sim.domain.znum == 1) ? static_cast<FloatType>(0.0) : sim.domain.zres;

            Kokkos::parallel_for(
                "grid_initialize_points",
                Kokkos::RangePolicy<device_exe>(0, p_num),
                InitializeDevicePoints<FloatType>{x_p, y_p, z_p, y_num, z_num, x0, y0, z0, dx, dy, dz});
        }
    };

}
