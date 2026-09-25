// Include parent .hpp file
#include "impl/RunModes/Snapshots/Mode.hpp"

// Internal Includes
#include "Definitions.hpp"
#include "impl/Meltpool/Beam.hpp"
#include "impl/Meltpool/Tracking.hpp"
#include "impl/RunModes/Snapshots/Config.hpp"
#include "impl/RunModes/Snapshots/Structs.hpp"
#include "impl/RunModes/Snapshots/Hooks.hpp"

// Include necessary functions.
#include "impl/Calc/Structs.hpp"
#include "impl/Calc/Temperature.hpp"
#include "impl/Output/Writer.hpp"

// Include external dependencies
#include <chrono>
#include <iostream>

namespace Condor::impl::Modes::Snapshots {

    // Calculate the full temperature field for a dense snapshot.
    template<typename FloatType>
    void Calculate_Dense_Temperatures(
        const Simdat<FloatType>& sim,
        Snapshots_State<FloatType>& state)
    {
        using dense_team_policy = Kokkos::TeamPolicy<device_exe>;
        using dense_team_member = typename dense_team_policy::member_type;

        const Grid::Device_Points<FloatType>& grid = state.grid;
        const Calc::Device_Nodes<FloatType>& nodes_cur_d = state.quad.CurrentSlot().device_nodes;
        const Calc::Device_Constants<FloatType> const_d(sim, state.step.dt);

        Kokkos::View<FloatType*, layout, device_memory> T_cur = state.thermal.T_cur;
        int_deviceView T_iter_cur = state.thermal.T_calc_iter_cur;
        const int iter_cur = state.step.iter_cur;
        const int pnum = sim.domain.pnum;

        if (pnum <= 0) {
            return;
        }

        Kokkos::parallel_for(
            "Snapshots_dense_temperature",
            dense_team_policy(pnum, Kokkos::AUTO, Kokkos::AUTO),
            KOKKOS_LAMBDA(const dense_team_member& team) {
                const int p = team.league_rank();
                const FloatType T = Calc::T_device(grid, nodes_cur_d, const_d, p, team);

                if (team.team_rank() == 0) {
                    T_cur(p) = T;
                    T_iter_cur(p) = iter_cur;
                }
            });
        Kokkos::fence();
    }

    // Fill every point that the interface snapshot writer will emit. The
    // tracker itself only probes enough points to recover the liquid boundary.
    template<typename FloatType, typename TrackingReference>
    void Calculate_Interface_Output_Temperatures(
        const Simdat<FloatType>& sim,
        Snapshots_State<FloatType>& state,
        const TrackingReference& tracking)
    {
        const int active_count = tracking.ActiveColumnCount();
        if (active_count <= 0) {
            return;
        }

        using team_policy = Kokkos::TeamPolicy<device_exe>;
        using team_member = typename team_policy::member_type;

        const Grid::Device_Points<FloatType>& grid = state.grid;
        const Calc::Device_Nodes<FloatType>& nodes_cur_d = state.quad.CurrentSlot().device_nodes;
        const Calc::Device_Constants<FloatType> const_d(sim, state.step.dt);

        const int_deviceView active_columns = tracking.ActiveColumns();
        const int_2D_deviceView depth = tracking.CurrentDepth();
        Kokkos::View<FloatType*, layout, device_memory> T_cur = state.thermal.T_cur;
        int_deviceView T_iter_cur = state.thermal.T_calc_iter_cur;
        const int iter_cur = state.step.iter_cur;
        const int ynum = sim.domain.ynum;
        const int znum = sim.domain.znum;

        Kokkos::parallel_for(
            "Snapshots_interface_output_temperature",
            team_policy(active_count, Kokkos::AUTO),
            KOKKOS_LAMBDA(const team_member& team) {
                const int v2d = active_columns(team.league_rank());
                const int i = v2d / ynum;
                const int j = v2d % ynum;
                const int column_depth = depth(i, j);

                for (int d = 0; d < column_depth; d++) {
                    const int p = v2d * znum + (znum - 1 - d);
                    const FloatType T = Calc::T_device(grid, nodes_cur_d, const_d, p, team);
                    if (team.team_rank() == 0) {
                        T_cur(p) = T;
                        T_iter_cur(p) = iter_cur;
                    }
                    team.team_barrier();
                }
            });
        Kokkos::fence();
    }

     // March Snapshots while calculating every point at every requested time.
    template<typename FloatType>
    void Run_Dense_Snapshots(
        const Simdat<FloatType>& sim,
        const Snapshots_Config<FloatType>& config,
        Snapshots_State<FloatType>& state,
        Snapshots_Hooks<FloatType>& hooks,
        Out::Writer<FloatType>& writer)
    {
        Dense_Tracking_Reference<FloatType> dense_tracking(sim);

        for (std::size_t snapshot = 0; snapshot < config.settings.times.size(); snapshot++) {
            const FloatType t_prev = (snapshot == 0)
                ? static_cast<FloatType>(0.0)
                : config.settings.times[snapshot - 1];
            const FloatType t_cur = config.settings.times[snapshot];

            // Advance one snapshot
            Kokkos::Profiling::pushRegion("Snapshots::Advance_State");
            state.Advance_To(static_cast<int>(snapshot), t_prev, t_cur);
            Kokkos::Profiling::popRegion();

            // Calculate every point at this snapshot time
            Calculate_Dense_Temperatures(sim, state);
            dense_tracking.BuildFromState(sim, state);

            // Run hooks
            Kokkos::Profiling::pushRegion("Snapshots::Hooks_PostStep");
            hooks.PostStep(
                dense_tracking.ActiveColumnCount(),
                dense_tracking.ActiveColumns(),
                dense_tracking.CurrentDepth(),
                sim,
                state);
            Kokkos::Profiling::popRegion();

            // Output this snapshot
            writer.Write(state.step.iter_cur, state.full_depth_by_column);

            // Output this iteration
            if (sim.settings.print) {
                std::cout << "Snapshot: " << (snapshot + 1) << " time: " << t_cur << "\n";
            }
        }
    }

    // March Snapshots while only tracking the liquid interface.
    template<typename FloatType>
    void Run_Tracking_Snapshots(
        const Simdat<FloatType>& sim,
        const Snapshots_Config<FloatType>& config,
        Snapshots_State<FloatType>& state,
        Snapshots_Hooks<FloatType>& hooks,
        Out::Writer<FloatType>& writer)
    {
        // Initialize beam trace
        Meltpool::BeamTracer<FloatType> beam_trace(sim, config.settings.fallback_timestep);
        beam_trace.Stop();

        // Make meltpool tracking object
        Meltpool::Tracking<FloatType> meltpool(sim);
        meltpool.Initialize_Solidification();
        Interface_Tracking_Reference<FloatType> snapshot_tracking(sim);

        for (std::size_t snapshot = 0; snapshot < config.settings.times.size(); snapshot++) {
            const FloatType t_prev = (snapshot == 0)
                ? static_cast<FloatType>(0.0)
                : config.settings.times[snapshot - 1];
            const FloatType t_cur = config.settings.times[snapshot];

            // Advance one snapshot iteration
            Kokkos::Profiling::pushRegion("Snapshots::Advance_State");
            state.Advance_To(static_cast<int>(snapshot), t_prev, t_cur);
            Kokkos::Profiling::popRegion();

            // Trace the beam over the snapshot window
            Kokkos::Profiling::pushRegion("Snapshots::Beam_Trace");
            beam_trace.TraceWindowDevice(t_prev, t_cur);
            Kokkos::Profiling::popRegion();

            // Meltpool tracking
            Kokkos::Profiling::pushRegion("Snapshots::Advance_Meltpool");
            meltpool.Advance(state, beam_trace);
            Kokkos::Profiling::popRegion();

            Kokkos::Profiling::pushRegion("Snapshots::Build_Interface_Depth");
            snapshot_tracking.BuildFromTracking(sim, meltpool);
            Kokkos::Profiling::popRegion();

            if (config.output.temperature) {
                Kokkos::Profiling::pushRegion("Snapshots::Interface_Output_Temperature");
                Calculate_Interface_Output_Temperatures(sim, state, snapshot_tracking);
                Kokkos::Profiling::popRegion();
            }

            // Run hooks
            Kokkos::Profiling::pushRegion("Snapshots::Hooks_PostStep");
            hooks.PostStep(
                meltpool.ActiveColumnCount(),
                meltpool.ActiveColumns(),
                meltpool.CurrentDepth(),
                sim,
                state);
            Kokkos::Profiling::popRegion();

            // Output this snapshot
            writer.Write(state.step.iter_cur, snapshot_tracking.CurrentDepth());

            // Output iteration
            if (sim.settings.print) {
                std::cout << "Snapshot: " << (snapshot + 1) << " time: " << t_cur << "\n";
            }
        }

        // Finalize meltpool
        meltpool.Finalize();
    }

    // March the standalone Snapshots calculator.
    template<typename FloatType>
    void Run(Simdat<FloatType>& sim) {
        Kokkos::Profiling::pushRegion("Snapshots::Initialization");

        // Initialize Snapshots Mode config
        const Snapshots_Config<FloatType> config(sim.files.mode_json, sim.util.allScansEndTime);

        // Initialize execution state (temperatures, calculation checks)
        Snapshots_State<FloatType> state(sim, config);

        // Initialize output object
        Out::Writer<FloatType> writer(sim);
        writer.SetOutputPrefix(sim.files.dataDir + "/" + sim.files.name);
        if (config.settings.tracking == Tracking_Mode::None) {
            writer.SetMode(Out::OutputMode::Volume3DFull);
        }

        // Make hooks
        Snapshots_Hooks<FloatType> hooks(config.hooks);

        // Initialize hooks
        hooks.Initialize_All(sim, writer);

        // Initialize output hooks
        if (config.output.temperature) { writer.AddView("T", state.thermal.T_cur); }
        Kokkos::Profiling::popRegion();

        Kokkos::Profiling::pushRegion("Snapshots::RunLoop");
        if (config.settings.tracking == Tracking_Mode::None) {
            Run_Dense_Snapshots(sim, config, state, hooks, writer);
        }
        else if (config.settings.tracking == Tracking_Mode::Interface) {
            Run_Tracking_Snapshots(sim, config, state, hooks, writer);
        }
        else {
            throw std::runtime_error("Unrecognized 'Snapshots' Mode");
        }
        Kokkos::Profiling::popRegion();

        Kokkos::Profiling::pushRegion("Snapshots::Finalize");

        // Finalize hooks
        hooks.Finalize_All(sim);

        // Stop the state (which runs asynchronously)
        state.Stop_All();

        Kokkos::Profiling::popRegion();
    }

    // Instantiate the Snapshots mode entry points for the supported scalar types.
    template void Run(Simdat<float>&);
    template void Run(Simdat<double>&);
}
