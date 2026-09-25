#pragma once

// Include the shared view aliases and Interface's prefetched quad slot types.
#include "Definitions.hpp"
#include "impl/Calc/Prefetch.hpp"
#include "impl/Structs/Simdat.hpp"
#include "impl/Structs/StateStructs.hpp"
#include "impl/RunModes/Interface/Config.hpp"
#include "impl/Utility/Grid.hpp"

namespace Condor::impl::Modes::Interface {

    // Bundle Interface's prefetch and temperature state in one reusable execution owner.
    template<typename FloatType>
    struct Interface_State {
        const FloatType timestep;

        Grid::Device_Points<FloatType> grid; // Device point coordinates shared by temperature kernels.
        Step_State<FloatType> step; // Timesteps and iterations
        Thermal_State<FloatType> thermal; // Current/previous temperatures plus calculation-iteration markers.
        Quad_State<FloatType> quad; // Prefetched current/previous device nodes used by Interface integration.
        
        // Allocate Interface-owned execution state once so the timestep loop can reuse the shared state cheaply.
        Interface_State(const Simdat<FloatType>& sim, const Interface_Config<FloatType>& config)
            : timestep(config.settings.timestep),
              grid(sim),
              step(timestep, config.output.frequency),
              thermal(sim),
              quad(sim, timestep)
        {
            quad.Start(timestep);
        }

        ~Interface_State(){
            Stop_All();
        }
        
        // Advance all states variables forward
        void Advance_All(){
            step.Advance_Iteration();
            thermal.Advance();
            quad.Advance(step.iter_cur);
        }

        // Reset is for "jumping" forward to another iteration.
        void Reset_All(const int itert){
            step.Reset(itert);
            thermal.Reset(itert);
            quad.Reset(timestep, step.iter_cur);
            quad.Acquire(step.iter_cur);
        }

        // Stop asynchronous quadrature prefetchers
        void Stop_All(){
            quad.Stop();
        }
    };
}
