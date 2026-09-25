#pragma once

// Include the shared solver data structures used by the Snapshots mode subsystem.
#include "impl/Structs/Simdat.hpp"

namespace Condor::impl::Modes::Snapshots {

    // Run the standalone snapshot calculator.
    template<typename FloatType>
    void Run(Simdat<FloatType>& sim);
}
