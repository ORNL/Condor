#pragma once

// Include Necessary Structs
#include "impl/Structs/Simdat.hpp"
#include "impl/Structs/MpiStructs.hpp"
#include "impl/Initialization/JsonUtility.hpp"
#include <string>
#include <vector>
#include <sstream>

using std::vector;
using std::string;
using std::stringstream;
using std::to_string;

namespace Condor::impl{
	namespace Init {
		template<typename FloatType>
		string	MakeDataDirectory(const string&);
		
		template<typename FloatType>
		void	Initialize_Simdat(Simdat<FloatType>&, const string&, CondorMPI<FloatType>&);

		// Reads in simulation type information
		template<typename FloatType>
		void	FileRead_Mode(Simdat<FloatType>&, const json&);

		template<typename FloatType>
		void	SetDomainParams(Domain<FloatType>&);

		// Reads in all necessary simulation parameters
		template<typename FloatType>
		void	FileRead_Material(Material<FloatType>&, const json&);
		
		template<typename FloatType>
		void	FileRead_Beam(Beam<FloatType>&, const json&);
		
		template<typename FloatType>
		void	FileRead_Path(vector<path_seg<FloatType>>&, const string&);

		// Reads in fine tuned simulation parameters
		template<typename FloatType>
		void	FileRead_Domain(Domain<FloatType>&, const json&);
		
		template<typename FloatType>
		void	FileRead_Settings(Settings<FloatType>&, const json&);

		// Set thermal diffusivity
		template<typename FloatType>
		void	SetDiffusivity(Material<FloatType>&);

		template<typename FloatType>
		void	FileRead_BeamsAndPaths(vector<Beam<FloatType>>&, vector<vector<path_seg<FloatType>>>&, const vector<json>&, const vector<string>&);
	}
}
