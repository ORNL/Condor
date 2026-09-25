#pragma once

// Include the shared solver data structures used by the Interface mode subsystem.
#include "impl/Structs/Simdat.hpp"
#include "impl/Structs/MpiStructs.hpp"

namespace Condor::impl::Modes::Interface {

    // Run the standalone interface tracker.
    template<typename FloatType>
    void Run(Simdat<FloatType>& sim);

    // Run Interface while publishing RDF data into the caller-owned container.
    template<typename FloatType>
    void Run(Simdat<FloatType>& sim, Stork::Structs::RDF_Dual<FloatType>& RDF);

    // Run Interface while publishing SRDF data into the caller-owned container.
    template<typename FloatType>
    void Run(Simdat<FloatType>& sim, Stork::Structs::SRDF_Dual<FloatType>& SRDF);
}
