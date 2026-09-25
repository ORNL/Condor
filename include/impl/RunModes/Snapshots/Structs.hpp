#pragma once

// Include the shared view aliases and Snapshots' prefetched quad slot types.
#include "Definitions.hpp"
#include "impl/Calc/Prefetch.hpp"
#include "impl/RunModes/Interface/Config.hpp"
#include "impl/RunModes/Interface/Structs.hpp"
#include "impl/RunModes/Snapshots/Config.hpp"
#include "impl/Structs/Simdat.hpp"
#include "impl/Structs/StateStructs.hpp"

namespace Condor::impl::Modes::Snapshots {

    // Bundle Snapshots' prefetch and temperature state in one reusable execution owner.
    template<typename FloatType>
    struct Snapshots_State : public Interface::Interface_State<FloatType> {
        using Base = Interface::Interface_State<FloatType>;

        int_deviceView full_depth_by_column; // Full-depth helper used when dense output writes all points.

        // Allocate Snapshots-owned execution state once so the timestep loop can reuse the shared state cheaply.
        Snapshots_State(const Simdat<FloatType>& sim, const Snapshots_Config<FloatType>& config)
            : Base(sim, MakeInterfaceConfig(config)),
              full_depth_by_column(Kokkos::ViewAllocateWithoutInitializing("Snapshots_full_depth_by_column"), sim.domain.xnum * sim.domain.ynum)
        {
            this->quad.Reset(config.settings.time_iterations);
            Kokkos::deep_copy(full_depth_by_column, sim.domain.znum);
        }

        // Advance all state variables to one explicit snapshot time.
        void Advance_To(const int snapshot_index, const FloatType t_prev, const FloatType t_cur) {
            this->step.iter_prev = snapshot_index;
            this->step.iter_cur = snapshot_index + 1;
            this->step.t_prev = t_prev;
            this->step.t_cur = t_cur;
            this->step.dt = t_cur - t_prev;

            this->thermal.Advance();
            this->quad.Advance(this->step.iter_cur);
        }

        private:
            static Interface::Interface_Config<FloatType> MakeInterfaceConfig(const Snapshots_Config<FloatType>& config) {
                Interface::Interface_Config<FloatType> interface_config;
                interface_config.settings.timestep = config.settings.fallback_timestep;
                interface_config.output.frequency = 1;
                interface_config.output.temperature = false;
                return interface_config;
            }
    };

    // Snapshot-local dense view of the sparse interface tracker depth.
    template<typename FloatType>
    struct Interface_Tracking_Reference {
        int active_count = 0;
        int_deviceView active_columns;
        int_2D_deviceView depth;

        explicit Interface_Tracking_Reference(const Simdat<FloatType>& sim)
            : active_columns(Kokkos::ViewAllocateWithoutInitializing("Snapshots_interface_active_columns"), 1),
              depth(Kokkos::ViewAllocateWithoutInitializing("Snapshots_interface_depth"), sim.domain.xnum, sim.domain.ynum)
        {
            Kokkos::deep_copy(depth, 0);
        }

        template<typename TrackingType>
        void BuildFromTracking(const Simdat<FloatType>& sim, const TrackingType& tracking) {
            active_count = tracking.ActiveColumnCount();
            active_columns = tracking.ActiveColumns();
            Kokkos::deep_copy(depth, 0);

            if (active_count <= 0) {
                return;
            }

            const int_deviceView active_columns_local = active_columns;
            const int_2D_deviceView source_depth = tracking.CurrentDepth();
            int_2D_deviceView dense_depth = depth;
            const int ynum = sim.domain.ynum;

            Kokkos::parallel_for(
                "Snapshots_interface_build_depth",
                Kokkos::RangePolicy<device_exe>(0, active_count),
                KOKKOS_LAMBDA(const int q) {
                    const int v2d = active_columns_local(q);
                    const int i = v2d / ynum;
                    const int j = v2d % ynum;
                    dense_depth(i, j) = source_depth(i, j);
                });
            Kokkos::fence();
        }

        int ActiveColumnCount() const { return active_count; }
        const int_deviceView& ActiveColumns() const { return active_columns; }
        const int_2D_deviceView& CurrentDepth() const { return depth; }
    };

    // Snapshot-local tracking reference built from a dense temperature field.
    template<typename FloatType>
    struct Dense_Tracking_Reference {
        int active_count = 0;
        int_deviceView active_count_d;
        int_hostView active_count_h;
        int_deviceView active_columns;
        int_2D_deviceView depth;

        explicit Dense_Tracking_Reference(const Simdat<FloatType>& sim)
            : active_count_d(Kokkos::ViewAllocateWithoutInitializing("Snapshots_dense_active_count"), 1),
              active_count_h(Kokkos::ViewAllocateWithoutInitializing("Snapshots_dense_active_count_h"), 1),
              active_columns(Kokkos::ViewAllocateWithoutInitializing("Snapshots_dense_active_columns"), sim.domain.xnum * sim.domain.ynum),
              depth(Kokkos::ViewAllocateWithoutInitializing("Snapshots_dense_depth"), sim.domain.xnum, sim.domain.ynum)
        {
            Kokkos::deep_copy(active_count_d, 0);
            Kokkos::deep_copy(active_count_h, 0);
            Kokkos::deep_copy(depth, 0);
        }

        void BuildFromState(const Simdat<FloatType>& sim, Snapshots_State<FloatType>& state) {
            Kokkos::deep_copy(active_count_d, 0);
            Kokkos::deep_copy(depth, 0);

            const Kokkos::View<FloatType*, layout, device_memory> T_cur = state.thermal.T_cur;
            bool_deviceView is_liq = state.thermal.isLiq_cur;
            int_deviceView active_columns_local = active_columns;
            int_deviceView active_count_local = active_count_d;
            int_2D_deviceView depth_local = depth;
            const int xnum = sim.domain.xnum;
            const int ynum = sim.domain.ynum;
            const int znum = sim.domain.znum;
            const FloatType T_liq = sim.material.T_liq;

            Kokkos::parallel_for(
                "Snapshots_dense_build_tracking",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {xnum, ynum}),
                KOKKOS_LAMBDA(const int i, const int j) {
                    int liquid_depth = 0;
                    bool still_top_connected = true;
                    for (int k = znum - 1; k >= 0; k--) {
                        const int p = i * (ynum * znum) + j * znum + k;
                        const bool liquid = T_cur(p) >= T_liq;
                        is_liq(p) = liquid;
                        if (still_top_connected && liquid) {
                            liquid_depth++;
                        }
                        else {
                            still_top_connected = false;
                        }
                    }

                    depth_local(i, j) = liquid_depth;
                    if (liquid_depth > 0) {
                        const int v2d = i * ynum + j;
                        const int q = Kokkos::atomic_fetch_add(&active_count_local(0), 1);
                        active_columns_local(q) = v2d;
                    }
                });
            Kokkos::fence();

            Kokkos::deep_copy(active_count_h, active_count_d);
            active_count = active_count_h(0);
        }

        int ActiveColumnCount() const { return active_count; }
        const int_deviceView& ActiveColumns() const { return active_columns; }
        const int_2D_deviceView& CurrentDepth() const { return depth; }
    };
}
