#pragma once

// Aggregate the Snapshots hook headers so callers can keep one include.
#include "impl/RunModes/Snapshots/Hooks/Meltpool.hpp"

namespace Condor::impl::Modes::Snapshots {

    // A struct containing all the snapshot hooks
    template<typename FloatType>
    struct Snapshots_Hooks {
        // NOTE: The order of declaration here is the order of construction!
        Hooks::Meltpool_Hook<FloatType> meltpool_hook;

        // Cache the hook config nodes in one place for the Snapshots path.
        explicit Snapshots_Hooks(const typename Snapshots_Config<FloatType>::Hooks& nodes)
            :
                // Initialize hooks in the order of dependency
                meltpool_hook(nodes.meltpool)
        {}

        // Hook initialization. This is where each hook will make it's views.
        void Initialize_All(const Simdat<FloatType>& sim, Out::Writer<FloatType>& writer) {
            meltpool_hook.Initialize(sim, writer);
        }

        // Post step function for snapshot hooks
        void PostStep(
            const int active_count,
            const int_deviceView& active_columns,
            const int_2D_deviceView& depth_view,
            const Simdat<FloatType>& sim,
            Snapshots_State<FloatType>& state)
        {
            meltpool_hook.PostStep(active_count, active_columns, depth_view, sim, state);
        }

        // Finalize function for all
        void Finalize_All(const Simdat<FloatType>& sim) {
            meltpool_hook.Finalize(sim);
        }
    };
}
