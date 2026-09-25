// Include necessary internals
#include "impl/Structs/Simdat.hpp"

// Include necessary externals
#include <iostream>

namespace Condor::impl{
	namespace Out{		
		template<typename FloatType>
		void Progress(int& prog_last_print, const int itert, const FloatType timestep, const Simdat<FloatType>& sim) {
			// Only print first rank
			if (sim.settings.print){
				int prog_now = int(10.0 * itert * timestep / sim.util.allScansEndTime);
				if (prog_now != prog_last_print) {
					prog_last_print = prog_now;
					if (prog_last_print <= 10) {
						std::cout << "Time step: " << itert << "\t\t";
						std::cout << "% of Path: " << 10 * prog_last_print << "%" << "\n";
					}
					else{
						std::cout << "Time step: " << itert << "\t\t";
						std::cout << "Cooling... \n";
					}
				}
			}
		}

		template<typename FloatType>
		void Point_Progress(const Simdat<FloatType>& sim, const int p) {
			// Only print first rank
			if (sim.settings.print){
				static int prog_print_last = 0;
				int prog_now = 10 * p / sim.domain.pnum;
				if (prog_now != prog_print_last) {
					prog_print_last = prog_now;
					std::cout << "% of Points: " << 10 * prog_print_last << "%" << "\n";
				}
			}
		}
	}
}
