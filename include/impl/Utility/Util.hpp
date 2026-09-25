#pragma once

// Include necessary structs
#include "impl/Structs/Simdat.hpp"
#include "impl/Calc/Structs.hpp"
#include "impl/Utility/Grid.hpp"
#include <vector>
#include <string>

namespace Condor::impl{
	namespace Util {
		// Usings
		using std::string;
		using std::vector;
		
		// Turns an integer into a zero-padded string
		string ZeroPadNumber(const int, const int);
		string ZeroPadNumber(const int);
		
		// Turns ijk indices into global point number
		template<typename FloatType>
		int		ijk_to_p(const int, const int, const int, const Simdat<FloatType>&);
		
		// Checks if it is inside the maximum radius
		template<typename FloatType>
		bool	InRMax(const FloatType, const FloatType, const Domain<FloatType>&, const Settings<FloatType>&);
		
		// Calculates the time to integrate back to
		template<typename FloatType>
		FloatType	t0calc(const FloatType, const Beam<FloatType>&, const Material<FloatType>&, const Settings<FloatType>&);
		
		// Gets the maximum allowable step size for a path segment
		template<typename FloatType>
		FloatType	GetRefTime(const FloatType, const int, const vector<path_seg<FloatType>>&, const Beam<FloatType>&);
		
		// Finds the current beam location
		template<typename FloatType>
		int_seg<FloatType> GetBeamLoc(const FloatType, const int, const vector<path_seg<FloatType>>&, const Simdat<FloatType>&);
		
		// Indicates when all points have solidified and the full scan is over
		template<typename FloatType>
		bool	sim_finish(const FloatType, const Simdat<FloatType>&, const int);
		
		// Calculates scan end time
		template<typename FloatType>
		void	Calc_AllScansEndTime(Simdat<FloatType>&);
		
		// Calculates bounds of scan path
		template<typename FloatType>
		void	Calc_ScanBounds(Domain<FloatType>&, const vector<vector<path_seg<FloatType>>>&);
		
		// Calculates the nondimensional integration time
		template<typename FloatType>
		void	Calc_NonD_dt(vector<Beam<FloatType>>&, const Material<FloatType>&);
		
		// Calculates the maximum radius around the domain to be considered
		template<typename FloatType>
		void	Calc_RMax(Simdat<FloatType>&);
	}
}

