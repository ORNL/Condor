#pragma once

#include "impl/Structs/Simdat.hpp"

namespace Condor::impl::Modes::Calibrate {

    template<typename FloatType>
    void Run(Simdat<FloatType>& sim);

}
