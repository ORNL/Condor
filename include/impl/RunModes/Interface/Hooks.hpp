#pragma once

// Aggregate the Interface hook headers so callers can keep one include.
#include "impl/RunModes/Interface/Hooks/Solidification.hpp"
#include "impl/RunModes/Interface/Hooks/Meltpool.hpp"
#include "impl/RunModes/Interface/Hooks/CET.hpp"

// // Exclusive ones
#include "impl/RunModes/Interface/Hooks/RDF.hpp"
#include "impl/RunModes/Interface/Hooks/SRDF.hpp"


namespace Condor::impl::Modes::Interface {
    
    // A struct containing all the interface hooks
    template<typename FloatType>
    struct Interface_Hooks
    {
        // NOTE: The order of declaration here is the order of construction!
        // Non-exclusive hooks in order of dependency.
        Hooks::Solidification_Hook<FloatType> solidification;
        Hooks::Meltpool_Hook<FloatType> meltpool_hook;
        Hooks::CET_Hook<FloatType> cet;
        // Exclusive hooks (manages in-memory constructions, etc). Should always be last.
        Hooks::RDF_Hook<FloatType> rdf;
        Hooks::SRDF_Hook<FloatType> srdf;

        // Cache the hook config nodes and coupled output pointers in one place for the old Jamie path.
        Interface_Hooks(
            const typename Interface_Config<FloatType>::Hooks& nodes,
            Stork::Structs::RDF_Dual<FloatType>* RDF_ptr,
            Stork::Structs::SRDF_Dual<FloatType>* SRDF_ptr)
            :   
                // Initilize hooks in the order of dependency
                solidification(nodes.solidification),
                meltpool_hook(nodes.meltpool),
                cet(nodes.cet),
                // Exclusive hooks should always be last
                rdf(RDF_ptr),
                srdf(SRDF_ptr)
        {}

        // Hook initialization. This is where each hook will make it's views.
        // NOTE::Should be in opposite order as construction (Make sure initialized is only called AFTER everything else which needs it, uses it)
        void Initialize_All(
            const Simdat<FloatType>& sim, 
            Meltpool::Tracking<FloatType>& meltpool, 
            Out::Writer<FloatType>& writer)
        {
            // Exclusive hooks should be first here (opposite order)
            rdf.Initialize(sim, meltpool, solidification);
            srdf.Initialize(sim, meltpool);
            // Initialize dependent hooks (to be safe, do in opposite order)
            cet.Initialize(sim, solidification, writer);
            meltpool_hook.Initialize(sim, meltpool, writer);
            solidification.Initialize(sim, meltpool, writer);
        }

        // Post step function for all
        // NOTE::Should be in construction order
        void PostStep_All(
            const Meltpool::Tracking<FloatType>& meltpool,
            const Simdat<FloatType>& sim,
            const Step_State<FloatType>& step,
            const Thermal_State<FloatType>& thermal)
        {
            // Initilize hooks in the order of dependency
            solidification.PostStep(meltpool, sim, step.dt);
            meltpool_hook.PostStep(meltpool, sim, step, thermal);
            cet.PostStep(solidification, step, sim);
            // Exclusive hooks should always be last
            rdf.PostStep(solidification, meltpool, sim);
        }

        void PostStep_All(
            const Meltpool::Tracking<FloatType>& meltpool,
            const Simdat<FloatType>& sim,
            Interface_State<FloatType>& state)
        {
            // Initilize hooks in the order of dependency
            solidification.PostStep(meltpool, sim, state.step.dt);
            meltpool_hook.PostStep(meltpool, sim, state.step, state.thermal);
            cet.PostStep(solidification, state.step, sim);
            // Exclusive hooks should always be last
            rdf.PostStep(solidification, meltpool, sim);
            srdf.PostStep(meltpool, sim, state);
        }

        // Finalize function for all
        // NOTE::Should be in construction order
        void Finalize_All(
            const Meltpool::Tracking<FloatType>& meltpool,
            const Simdat<FloatType>& sim,
            const Step_State<FloatType>& step)
        {
            // Initilize hooks in the order of dependency
            solidification.Finalize(meltpool, sim, step.dt);
            meltpool_hook.Finalize(meltpool, sim, step);
            cet.Finalize(solidification, step, sim);
            // Exclusive hooks should always be last
            rdf.Finalize(solidification, meltpool, sim);
            srdf.Finalize(meltpool, sim, step);
        }
    };
}
