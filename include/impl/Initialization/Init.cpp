// Include where functions are defined
#include "impl/Initialization/Init.hpp"

// Internal includes
#include "impl/Initialization/JsonUtility.hpp"
#include "impl/Initialization/Structs.hpp"
#include "impl/Utility/Util.hpp"

// External includes
#include <iostream>
#include <fstream>
#include <initializer_list>
#include <omp.h>
#include <stdexcept>
#include <filesystem>
#include <functional>
#include <sstream>
#include <system_error>
#include <string>
#include <vector>

namespace Condor::impl{
	namespace Init{
		// Usings and namespaces
		
		namespace fs = std::filesystem;
		using string = std::string;
		template<typename T>
		using vector = std::vector<T>;

		// Modern function for making directories (system agnostic)
		template <typename FloatType>
		string MakeDataDirectory(const std::string& configFile) {
			// Setup Data Directory
			fs::path configPath(configFile);
			fs::path dataDir = configPath.parent_path() / "Data";
			
			// Create_directories returns value based on successfullness
			std::error_code ec;
			fs::create_directories(dataDir, ec);
			
			// Check for errors and report them
			if (ec) {
				// Check specifically for permission denied
				if (ec == std::errc::permission_denied) {
					std::cerr << "Error: Permission denied. Cannot create directory: " << dataDir << "\n";
				} 
				else {
					std::cerr << "Error: Could not create directory. " << ec.message() << "\n";
				}
			}

			return dataDir.string();
		}

		template<typename FloatType>
		void SetDomainParams(Domain<FloatType>& domain) {
			domain.xres = domain.res;
			domain.yres = domain.res;
			domain.zres = domain.res;

			domain.xnum = 1 + static_cast<int>(0.5 + (domain.xmax - domain.xmin) / domain.xres);
			domain.ynum = 1 + static_cast<int>(0.5 + (domain.ymax - domain.ymin) / domain.yres);
			domain.znum = 1 + static_cast<int>(0.5 + (domain.zmax - domain.zmin) / domain.zres);

			domain.xmax = domain.xmin + (domain.xnum - 1) * domain.xres;
			domain.ymax = domain.ymin + (domain.ynum - 1) * domain.yres;
			domain.zmax = domain.zmin + (domain.znum - 1) * domain.zres;
			domain.pnum = domain.xnum * domain.ynum * domain.znum;
		}

		// For initializing the simulation
		template <typename FloatType>
		void Initialize_Simdat(Simdat<FloatType>& sim, const string& configFile, CondorMPI<FloatType>& mpi) {
			
			// Make the file reader function
			FileReader reader;

			// If the root rank, read the files
			if (!mpi.rank){
				// Read everything in to the reader
				reader.Initialize(configFile);

				// Make data directory
				reader.dataDir = MakeDataDirectory<FloatType>(configFile);
			}
			
			// If we have multiple ranks
			if (mpi.nproc > 1) {
				// Then we are using mpi
				sim.mpi = true;
				// Broadcast the raw inputs and the separate mode-owned controls.
				reader.Broadcast(0,mpi.comm);
			}

			sim.files.name = reader.name;
			sim.files.dataDir = reader.dataDir;
			sim.files.mode_json = reader.mode_json;

			// Read in beams and paths. Calculate the maximum scan end time across all paths.
			FileRead_BeamsAndPaths(sim.beams, sim.paths, reader.beam_jsons, reader.path_strings); 
			Util::Calc_AllScansEndTime(sim);
			
			// Read in material and set non-dimensional integration time.
			FileRead_Material(sim.material, reader.material_json);
			Util::Calc_NonD_dt(sim.beams, sim.material);
			
			// Read in domain and set domain parameters. Calculate scan bounds if not set
			FileRead_Domain(sim.domain, reader.domain_json); 
			Util::Calc_ScanBounds(sim.domain, sim.paths); 
			SetDomainParams(sim.domain);

			// Read in settings and calculate integration bounds			
			FileRead_Settings(sim.settings, reader.settings_json);
			sim.settings.rank = mpi.rank;
			sim.settings.nproc = mpi.nproc;		
			Util::Calc_RMax(sim);

			// Find out mode type
			FileRead_Mode(sim, reader.mode_json);

			// If we have multiple ranks, make local bounds after settings/domain are initialized.
			if (mpi.nproc > 1) {
				mpi.makeLocalBounds(sim);
				sim.mpi_decomposition.active = true;
				sim.mpi_decomposition.dims[0] = mpi.dims[0];
				sim.mpi_decomposition.dims[1] = mpi.dims[1];
				sim.mpi_decomposition.coords[0] = mpi.coords[0];
				sim.mpi_decomposition.coords[1] = mpi.coords[1];
				for (int n = 0; n < 8; n++) {
					sim.mpi_decomposition.neighbor_ranks[n] = mpi.neighbor_ranks[n];
				}
				sim.mpi_decomposition.i_min = mpi.i_min;
				sim.mpi_decomposition.i_max = mpi.i_max;
				sim.mpi_decomposition.j_min = mpi.j_min;
				sim.mpi_decomposition.j_max = mpi.j_max;
				sim.mpi_decomposition.global_x0 = mpi.header.global_x0();
				sim.mpi_decomposition.global_y0 = mpi.header.global_y0();
				sim.mpi_decomposition.global_z0 = sim.domain.zmin;
				sim.mpi_decomposition.global_i0 = mpi.header.global_i0();
				sim.mpi_decomposition.global_j0 = mpi.header.global_j0();
				sim.mpi_decomposition.global_k0 = static_cast<uint32_t>(0);
				sim.mpi_decomposition.local_inum = mpi.header.local_inum();
				sim.mpi_decomposition.local_jnum = mpi.header.local_jnum();
				sim.mpi_decomposition.local_knum = static_cast<uint32_t>(sim.domain.znum);
				sim.mpi_decomposition.gridResolution = sim.domain.xres;
			}
		}
		
		// Reads in a single beam file
		template<typename FloatType>
		void	FileRead_Beam(Beam<FloatType>& beam, const json& root) {
			// Parse and seperate file
			const json& shape = GetNestedJson(root, "shape", true); // Beam shape parameters.
			const json& intensity = GetNestedJson(root, "intensity", true); // Beam intensity parameters.
			
			// Read the beam shape parameters used by the heat-source model.
			beam.ax = ReadValue<FloatType>(shape, "width_x", (FloatType)0.0, true);
			beam.ay = ReadValue<FloatType>(shape, "width_y", (FloatType)0.0, true);
			beam.az = ReadValue<FloatType>(shape, "width_z", (FloatType)0.0, false);

			// Read the beam power settings before converting them to the runtime form.
			beam.q = ReadValue<FloatType>(intensity, "power", (FloatType)0.0, true);
			beam.eff = ReadValue<FloatType>(intensity, "efficiency", (FloatType)1.0, true);
	
			// Set beam power based on efficiency and half-infinit assumption
			beam.q *= beam.eff * 2.0;
		}

		// Reads a single path file
		template<typename FloatType>
		void FileRead_Path(std::vector<path_seg<FloatType>>& path, const std::string& pathText) {
			// mm to m conversion factor
			FloatType convert = static_cast<FloatType>(1e-3); 

			// Initial segment
			path_seg<FloatType> startSeg{1, 0, 0, 0, 0, 0, 0}; 
			path.push_back(startSeg);

			std::istringstream pathStream(pathText);
			std::string line;
			while (std::getline(pathStream, line)) {
				if (line.empty()) continue;
				
				std::stringstream ss(line);
				path_seg<FloatType> seg;
				if (ss >> seg.smode >> seg.sx >> seg.sy >> seg.sz >> seg.sqmod >> seg.sparam) {
					seg.sx *= convert;
					seg.sy *= convert;
					seg.sz *= convert;
					path.push_back(seg);
				}
			}

			// Calculate segment times
			path[0].seg_time = 0.0;
			for (size_t i = 1; i < path.size(); ++i) {
				if (path[i].smode) { // Spot mode
					path[i].seg_time = path[i-1].seg_time + path[i].sparam;
				} else { // Line mode
					FloatType dx = path[i].sx - path[i-1].sx;
					FloatType dy = path[i].sy - path[i-1].sy;
					FloatType dz = path[i].sz - path[i-1].sz;
					FloatType dist = std::sqrt(dx*dx + dy*dy + dz*dz);
					// Protect against division by zero if speed is 0
					FloatType dt = (path[i].sparam > 0) ? (dist / path[i].sparam) : 0;
					path[i].seg_time = path[i-1].seg_time + dt;
				}
			}
		}

		// Function for reading in (potentially multiple) beams and paths based on patterns. Each beam file must have a corresponding path file.
		template<typename FloatType>
		void FileRead_BeamsAndPaths(vector<Beam<FloatType>>& beams, vector<vector<path_seg<FloatType>>>& paths, const vector<json>& beamJsons, const vector<string>& pathFiles) {
			if (beamJsons.size() != pathFiles.size()) {
				throw std::runtime_error("Beam/path count mismatch.");
			}

			// Process the files
			for (size_t i = 0; i < beamJsons.size(); ++i) {
				// Read Beam
				Beam<FloatType> b;
				FileRead_Beam(b, beamJsons[i]);
				beams.push_back(b);
				// Read Path
				vector<path_seg<FloatType>> p;
				FileRead_Path(p, pathFiles[i]);
				paths.push_back(p);
			}
		}

		// Helper function for setting diffusivity of material based on constants
		template<typename FloatType>
		void	SetDiffusivity(Material<FloatType>& material) {
			material.a = material.kon / (material.rho * material.cps);
		}

		// Reads in the material properties from a JSON file and calculates the diffusivity. Also reads in optional CET parameters.
		template<typename FloatType>
		void	FileRead_Material(Material<FloatType>& material, const json& root) {
			const json& constants = GetNestedJson(root, "constants", true); // Main material constants block.
			
			// Read the thermal constants that control the base material model.
			material.T_init = ReadValue<FloatType>(constants, "T_init", static_cast<FloatType>(1273.0), false);
			material.T_liq = ReadValue<FloatType>(constants, "T_liq", static_cast<FloatType>(1610.0), false);
			material.kon = ReadValue<FloatType>(constants, "k", static_cast<FloatType>(26.6), true);
			material.cps = ReadValue<FloatType>(constants, "c", static_cast<FloatType>(600.0), true);
			material.rho = ReadValue<FloatType>(constants, "p", static_cast<FloatType>(7451.0), true);
			
			// Set the thermal diffusivity based on the conductivity, specific heat, and density.
			SetDiffusivity(material);
		}

		// Read in the domain parameters
		template<typename FloatType>
		void	FileRead_Domain(Domain<FloatType>& domain, const json& root) {
			// For placeholder values
			const FloatType float_type_max = std::numeric_limits<FloatType>::max(); 
			
			// Define internal lambda for bounds parsing
			const std::function<void(const json&, const string&, FloatType&, FloatType&, bool)> readBounds =
				[&](const json& j, const string& key, FloatType& minV, FloatType& maxV, bool isCritical) {
				if (j.contains(key) && j[key].is_array() && j[key].size() >= 2) {
					minV = j[key][0].get<FloatType>();
					maxV = j[key][1].get<FloatType>();
				} else if (isCritical) {
					throw std::runtime_error("Critical Input Error: Domain field '" + key + "' must be an array of size 2.");
				}
			};

			// Parse input file (default is no object)
			const json& domainNode = GetNestedJson(root, "domain", true); 
			const json& boundaryNode = GetNestedJson(root, "boundary_conditions", false);

			// Read the single shared grid resolution before assigning it onto each axis.
			domain.res = ReadValue<FloatType>(domainNode, "resolution", static_cast<FloatType>(50e-6), true);
			readBounds(domainNode, "x", domain.xmin, domain.xmax, false);
			readBounds(domainNode, "y", domain.ymin, domain.ymax, false);
			readBounds(domainNode, "z", domain.zmin, domain.zmax, false);
			
			// If all domain boundaries are set, we don't need to calculate bounds from the path
			if (domain.xmin != float_type_max && domain.xmax != -float_type_max && domain.ymin != float_type_max && domain.ymax != -float_type_max) {
				domain.calc_bounds = false;
			}
			
			// Read the optional reflective boundary-condition limits from the JSON block.
			readBounds(boundaryNode, "x", domain.BC_xmin, domain.BC_xmax, false);
			readBounds(boundaryNode, "y", domain.BC_ymin, domain.BC_ymax, false);
			domain.BC_zmin = ReadValue<FloatType>(boundaryNode, "z_min", float_type_max, false);
			domain.BC_reflections = ReadValue<int>(boundaryNode, "reflections", 1, false);
			
			// If any boundary condition is set, we will use boundary conditions
			if (domain.BC_xmin != float_type_max || domain.BC_xmax != -float_type_max || domain.BC_ymin != float_type_max || domain.BC_ymax != -float_type_max || domain.BC_zmin != float_type_max) {
				domain.use_BCs = true;
			}
		}
		
		// Read in the settings file
		template<typename FloatType>
		void	FileRead_Settings(Settings<FloatType>& settings, const json& root) {
			// Read in an parse the file into its nodes.  Default to empty object if no file provided.
			const json& outputNode = GetNestedJson(root, "output", false); // Output settings
			const json& meltpoolNode = GetNestedJson(root, "meltpool", false); // Meltpool settings.
			const json& quadratureNode = GetNestedJson(root, "quadrature", false); // Temperature solver settings.
			const json& computeNode = GetNestedJson(root, "compute", false); // Path preprocessing settings.
			const json& mpiNode = GetNestedJson(root, "mpi", false); // MPI behavior settings.

			// Read in output settings
			settings.output.dims = ReadValue<string>(outputNode, "dim", "3D", false);
			settings.output.format = ReadValue<string>(outputNode, "format", ".csv", false);
			settings.output.noOutput = ReadValue<bool>(outputNode, "noOutput", false, false);

			// Read in meltpool settings
			settings.radius_check = ReadValue<FloatType>(meltpoolNode, "radius_check", static_cast<FloatType>(0.0), false);
			settings.trace_radius = ReadValue<FloatType>(meltpoolNode, "trace_radius", -static_cast<FloatType>(1.0), false);

			// Read in quadrature settings
			settings.t_hist = ReadValue<FloatType>(quadratureNode, "cutoff_peak", static_cast<FloatType>(1e-9), false);
			settings.p_hist = ReadValue<FloatType>(quadratureNode, "cutoff_t0tl", static_cast<FloatType>(1e-2), false);
			
			// Read in compute settings
			settings.prefetch_threads = ReadValue<int>(computeNode, "prefetch_threads", 2, false);

			// Read in MPI settings
			settings.rank = ReadValue<int>(mpiNode, "rank", 0, false);
			settings.nproc = ReadValue<int>(mpiNode, "nproc", 1, false);
			settings.print = ReadValue<int>(mpiNode, "print", 0, false);
		}

		template<typename FloatType>
		void 	FileRead_Mode(Simdat<FloatType>& sim, const json& root) {
			// Define the "Master List" of supported modes
			const vector<string> supportedModes = {
				"Interface", 
				"Snapshots",
				"Calibrate"
			};

			// Check the file for these specific keywords
			string detectedMode = "";
			int modeCount = 0;
			for (const std::string& candidate : supportedModes) {
				if (root.contains(candidate)) {
					detectedMode = candidate;
					modeCount++;
				}
			}

			// Error if too few or too many modes
			if (modeCount == 0) throw std::runtime_error("File Error: No valid mode found in mode JSON.");
			if (modeCount > 1) throw std::runtime_error("File Error: Multiple modes detected in mode JSON.");

			// Set the mode type (determines switching later in code)
			sim.files.modeType = detectedMode;
		}
	
		// --- Explicit Instantiations (within the namespace, after definitions) ---
		template void	Initialize_Simdat<float>(Simdat<float>&, const string&, CondorMPI<float>&);
		template void	Initialize_Simdat<double>(Simdat<double>&, const string&, CondorMPI<double>&);

		template string	MakeDataDirectory<float>(const string&);
		template string	MakeDataDirectory<double>(const string&);

		template void	FileRead_Mode(Simdat<float>&, const json&);
		template void	FileRead_Mode(Simdat<double>&, const json&);
	
		template void	FileRead_Material(Material<float>&, const json&);
		template void	FileRead_Material(Material<double>&, const json&);
		
		template void	FileRead_Beam(Beam<float>&, const json&);
		template void	FileRead_Beam(Beam<double>&, const json&);
		
		template void	FileRead_Path(vector<path_seg<float>>&, const string&);
		template void	FileRead_Path(vector<path_seg<double>>&, const string&);
	
		template void	FileRead_Domain(Domain<float>&, const json&);
		template void	FileRead_Domain(Domain<double>&, const json&);
		
		template void	FileRead_Settings(Settings<float>&, const json&);
		template void	FileRead_Settings(Settings<double>&, const json&);
	
		template void	SetDiffusivity(Material<float>&);
		template void	SetDiffusivity(Material<double>&);

		template void	SetDomainParams(Domain<float>&);
		template void	SetDomainParams(Domain<double>&);
	
		template void	FileRead_BeamsAndPaths(vector<Beam<float>>&, vector<vector<path_seg<float>>>&, const vector<json>&, const vector<string>&);
		template void	FileRead_BeamsAndPaths(vector<Beam<double>>&, vector<vector<path_seg<double>>>&, const vector<json>&, const vector<string>&);
	}
}
