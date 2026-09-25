#pragma once

// Include external dependencies
#include <stdexcept>

// Include necessary data structures.
#include "impl/Structs/Simdat.hpp"
#include "impl/Structs/MpiStructs.hpp"
#include "impl/RunModes/Interface/Mode.hpp"
#include "impl/RunModes/Snapshots/Mode.hpp"
#include "impl/RunModes/Calibrate/Mode.hpp"

namespace Condor::impl{
	namespace Modes{
		// Namespace usings
		using Stork::Structs::RDF_Dual;
		using Stork::Structs::SRDF_Dual;

		// Route to one of the supported run modes.
		template<typename FloatType>
		void Simulate(Simdat<FloatType>& sim) {
			// Route to the selected mode's driver function.
			if (sim.files.modeType == "Interface"){
				Interface::Run(sim);
				return;
			}
			else if (sim.files.modeType == "Snapshots"){
				Snapshots::Run(sim);
				return;
			}
			else if (sim.files.modeType == "Calibrate"){
				Calibrate::Run(sim);
				return;
			}
			// Fail loudly when the mode parser handed us anything outside the supported set.
			throw std::runtime_error("Unrecognized simulation mode: " + sim.files.modeType);
		}
	}
}
