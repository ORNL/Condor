
#pragma once

// Include necessary structs
#include "impl/Structs/Simdat.hpp"
#include "impl/Calc/Structs.hpp"
#include <vector>

namespace Condor::impl{
	namespace Calc {
		// Namespace Definitions
		using std::vector;

		// Get quadrature nodes and points for one time
		template <typename FloatType>
		void Integrate_Serial(Nodes<FloatType>&, vector<int>&, const Simdat<FloatType>&, const FloatType, const bool);
	
		// Adaptive Integration Scheme
		template <typename FloatType>
		void GaussIntegrate(Nodes<FloatType>&, vector<int>&, const Simdat<FloatType>&, const FloatType, const bool);

		// Adds Simple boundary conditions (x and y) via method of images
		template <typename FloatType>
		void AddBCs(Nodes<FloatType>&, const Domain<FloatType>&);
	}
}
