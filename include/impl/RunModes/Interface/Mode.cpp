// Include parent .hpp file
#include "impl/RunModes/Interface/Mode.hpp"

// Internal Includes
#include "Definitions.hpp"
#include "impl/Meltpool/Beam.hpp"
#include "impl/Meltpool/Tracking.hpp"
#include "impl/RunModes/Interface/Config.hpp"
#include "impl/RunModes/Interface/Structs.hpp"
#include "impl/RunModes/Interface/Hooks.hpp"

// Include necessary functions.
#include "impl/Structs/StateStructs.hpp"
#include "impl/Utility/Util.hpp"
#include "impl/Utility/Nodes.hpp"
#include "impl/Utility/Grid.hpp"
#include "impl/Calc/Prefetch.hpp"
#include "impl/Calc/Quad.hpp"
#include "impl/Calc/Structs.hpp"
#include "impl/Calc/Temperature.hpp"
#include "impl/Output/Progress.hpp"
#include "impl/Output/Writer.hpp"

// Include external dependencies
#include <algorithm>
#include <array>
#include <future>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace Condor::impl::Modes::Interface {

    // March Interface with optional SRDF or RDF export targets while reusing one shared timestep loop.
    template<typename FloatType>
    void Solidify_Interface_Impl(
        Simdat<FloatType>& sim,
        Stork::Structs::RDF_Dual<FloatType>* RDF,
        Stork::Structs::SRDF_Dual<FloatType>* SRDF
        )
    {
        Kokkos::Profiling::pushRegion("Interface::Initialization");

        // TODO::Have all JSON files be broadcast after being read (so only sync once in initialization)
        
        // Turn off output if doing RDF of SRDF
        if ((RDF != nullptr) || (SRDF != nullptr)){ sim.settings.output.noOutput=true;}

        // Initialize Interface Mode config
        const Interface_Config<FloatType> config(sim.files.mode_json, RDF, SRDF);

        // Initialize execution state (temperatures, calculation checks)
        Interface_State<FloatType> state(sim, config);

        // Initialize beam trace
        Meltpool::BeamTracer<FloatType> beam_trace(sim, config.settings.timestep);

        // Make meltpool tracking object
        Meltpool::Tracking<FloatType> meltpool(sim);

        // Initialize output object
        Out::Writer<FloatType> writer(sim);
        std::string output_prefix = sim.files.dataDir + "/" + sim.files.name;
        if (sim.settings.nproc > 1) {
            output_prefix += ".rank" + std::to_string(sim.settings.rank);
        }
        writer.SetOutputPrefix(output_prefix);

        // Make hooks
        Interface_Hooks<FloatType> hooks(config.hooks, RDF, SRDF);

        // Initialize hooks
        hooks.Initialize_All(sim, meltpool, writer);

        // Initialize meltpool tracker after hooks set calculation requirements
        meltpool.Initialize_Solidification();

        // Initialize output hooks
        if (config.output.temperature) {writer.AddView("T",state.thermal.T_cur);}
        Kokkos::Profiling::popRegion();

        // // Go until scan path is over and meltpools have solidified
        Kokkos::Profiling::pushRegion("Interface::RunLoop");
        while (state.step.t_cur < sim.util.allScansEndTime || meltpool.summary.active_columns > 0)
        {   

            // Trace the beam from this iteration to next
            Kokkos::Profiling::pushRegion("Beam_Trace");
            beam_trace.Step(state.step.iter_cur);
            Kokkos::Profiling::popRegion();
            
            // If next beam trace iteration is in the future and we have no liquid, reset at that iteraiton
            if (beam_trace.NextIteration() > state.step.iter_cur + 1 && meltpool.summary.active_columns == 0){
                state.Reset_All(beam_trace.NextIteration());
                beam_trace.Step(state.step.iter_cur);
            }
            
            // Advance one iteration
            Kokkos::Profiling::pushRegion("Advance_State");
            state.Advance_All();
            Kokkos::Profiling::popRegion();
            
            // Get quadrature for current time
            Kokkos::Profiling::pushRegion("Quadrature");
            Calc::AdaptiveQuadratureSlot<FloatType>& cur_quad = state.quad.CurrentSlot();
            Kokkos::Profiling::popRegion();

            // Meltpool tracking
            Kokkos::Profiling::pushRegion("Advance_Meltpool");
            meltpool.Advance(state, beam_trace);
            Kokkos::Profiling::popRegion();

            // TODO::Make MPI Capable
            if (sim.settings.print){
                std::cout << "Time step: " << state.step.iter_cur << "\n";
            }

            // Run hooks
            Kokkos::Profiling::pushRegion("Hooks_PostStep");
            hooks.PostStep_All(meltpool, sim, state);
            Kokkos::Profiling::popRegion();

            // Output if desired
            if (state.step.isOutputStep()){
                writer.Write(state.step.iter_cur, meltpool.MaxDepthByColumn());
            }
        }
        Kokkos::Profiling::popRegion();
        Kokkos::Profiling::pushRegion("Finalize");
        
        // Finalize meltpool
        meltpool.Finalize();

        // Finalize hooks
        hooks.Finalize_All(meltpool, sim, state.step);

        // Output final
        writer.Write(-1, meltpool.MaxDepthByColumn());

        // Stop the beam trace and state (which run asynchronously)
        // beam_trace.Stop();
        state.Stop_All();

        Kokkos::Profiling::popRegion();
    }

    // March the standalone Interface surface tracker by reusing the coupled-capable path without a coupled target.
    template<typename FloatType>
    void Run(Simdat<FloatType>& sim) {
        Solidify_Interface_Impl<FloatType>(sim, nullptr, nullptr);
    }

    // March the Interface surface tracker while publishing RDF events into the caller-owned container.
    template<typename FloatType>
    void Run(Simdat<FloatType>& sim, Stork::Structs::RDF_Dual<FloatType>& RDF) {
        Solidify_Interface_Impl<FloatType>(sim, &RDF, nullptr);
    }

    // March the Interface surface tracker while publishing SRDF snapshots into the caller-owned container.
    template<typename FloatType>
    void Run(Simdat<FloatType>& sim, Stork::Structs::SRDF_Dual<FloatType>& SRDF) {
        Solidify_Interface_Impl<FloatType>(sim, nullptr, &SRDF);
    }

    // Instantiate the Interface mode entry points for the supported scalar types.
    template void Run(Simdat<float>&);
    template void Run(Simdat<double>&);
    template void Run(Simdat<float>&, Stork::Structs::RDF_Dual<float>&);
    template void Run(Simdat<double>&, Stork::Structs::RDF_Dual<double>&);
    template void Run(Simdat<float>&, Stork::Structs::SRDF_Dual<float>&);
    template void Run(Simdat<double>&, Stork::Structs::SRDF_Dual<double>&);
}
