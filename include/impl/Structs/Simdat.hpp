#pragma once

// Include external dependencies
#include <nlohmann/json.hpp>
#include <cstdint>
#include <string>
#include <vector>
#include <climits>
#include <cfloat>
#include <limits>

// Defines
#define PI 3.14159265358979323846

namespace Condor::impl {

    // Usings
    using string = std::string;
    template<typename T>
    using vector = std::vector<T>;

    // Structure for storing an individual path segment's information.
    template<typename FloatType>
    struct path_seg {
        int smode; // Segment mode (line melt or spot melt).
        FloatType sx, sy, sz; // Segment end coordinates.
        FloatType sqmod; // Segment power modulation.
        FloatType sparam; // Segment time parameter (speed or spot time).
        FloatType seg_time; // Segment end time.
    };

    // Structure to store input filenames.
    struct FileNames {
        string name, dataDir;
        string material, beam, path;
        string domain, settings;
        string mode, modeType;
        nlohmann::json mode_json = nlohmann::json::object();
    };

    // Structure for material constants.
    template<typename FloatType>
    struct Material {
        FloatType kon; // Thermal conductivity.
        FloatType rho; // Density.
        FloatType cps; // Specific heat.
        FloatType T_liq; // Liquidus temperature.
        FloatType T_init; // Initial temperature (preheat or ambient).
        FloatType a; // Thermal diffusivity.
    };

    // Structure for beam parameters.
    template<typename FloatType>
    struct Beam {
        FloatType ax, ay, az; // Beam shape.
        FloatType eff; // Absorption efficiency.
        FloatType q; // Beam power.
        FloatType nond_dt; // Nondimensional time step.
    };

    // Structure for storing domain parameters.
    template<typename FloatType>
    struct Domain {
        // Domain counts.
        int xnum, ynum, znum, pnum;

        // Domain bounds.
        bool calc_bounds = true; // Whether to calculate bounds from the path.
        FloatType xmin = std::numeric_limits<FloatType>::max();
        FloatType xmax = -std::numeric_limits<FloatType>::max();
        FloatType ymin = std::numeric_limits<FloatType>::max();
        FloatType ymax = -std::numeric_limits<FloatType>::max();
        FloatType zmin = static_cast<FloatType>(1e-3);
        FloatType zmax = -static_cast<FloatType>(0.0);

        // Domain resolution.
        FloatType res;
        FloatType xres = static_cast<FloatType>(0.0);
        FloatType yres = static_cast<FloatType>(0.0);
        FloatType zres = static_cast<FloatType>(0.0);

        // Domain reflections.
        bool use_BCs = false;
        int BC_reflections = 1;
        FloatType BC_xmin = std::numeric_limits<FloatType>::max();
        FloatType BC_xmax = -std::numeric_limits<FloatType>::max();
        FloatType BC_ymin = std::numeric_limits<FloatType>::max();
        FloatType BC_ymax = -std::numeric_limits<FloatType>::max();
        FloatType BC_zmin = std::numeric_limits<FloatType>::max();
    };

    // Structure for storing low-level settings.
    template<typename FloatType>
    struct Settings {

        // Output settings
        struct{
            string dims;
            string format;
            bool noOutput;
        } output;

        // Meltpool settings
        FloatType radius_check; 
        FloatType trace_radius = -static_cast<FloatType>(1.0);

        // Quadrature settings
        FloatType t_hist, p_hist;

        // Compute settings
        int prefetch_threads;
        // bool use_pint

        // MPI settings
        int rank;
        int nproc;
        int print; // print=0 no print, 1 first rank, 2 all

        // Radius used by legacy temperature quadrature filters.
        FloatType r_max = -static_cast<FloatType>(1.0);
    };

    // Structure for storing useful runtime values.
    template<typename FloatType>
    struct Utility {
        FloatType allScansEndTime = 0; // Time when all scans are done.
        FloatType approxEndTime = 0; // Approximate end time to simulation.
        bool sol_finish = false; // Whether solidification has finished.
        bool do_sol = false; // Whether solidification refinement is enabled.
        bool surfaceOnly = false; // Whether the domain collapses to a single z layer.
    };

    // Structure for storing the decomposition metadata needed after the local
    // domain bounds have replaced the original global bounds.
    template<typename FloatType>
    struct MpiDecomposition {
        bool active = false;
        int dims[2] = {1, 1};
        int coords[2] = {0, 0};
        int neighbor_ranks[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
        int i_min = 0;
        int i_max = 0;
        int j_min = 0;
        int j_max = 0;
        FloatType global_x0 = static_cast<FloatType>(0.0);
        FloatType global_y0 = static_cast<FloatType>(0.0);
        FloatType global_z0 = static_cast<FloatType>(0.0);
        uint32_t global_i0 = 0;
        uint32_t global_j0 = 0;
        uint32_t global_k0 = 0;
        uint32_t local_inum = 0;
        uint32_t local_jnum = 0;
        uint32_t local_knum = 0;
        FloatType gridResolution = static_cast<FloatType>(0.0);
    };

    // Structure containing the raw simulation inputs shared across every mode.
    template<typename FloatType>
    struct Simdat {
        FileNames files; // Input files.
        
        Material<FloatType> material; // Material information.
        Domain<FloatType> domain; // Domain information.
        
        vector<Beam<FloatType>> beams; // Beam information.
        vector<vector<path_seg<FloatType>>> paths; // Beam path information.
        
        Settings<FloatType> settings; // Execution settings.
        Utility<FloatType> util; // Shared derived values.
        MpiDecomposition<FloatType> mpi_decomposition; // Distributed-domain metadata.
        
        bool mpi = false; // Whether the current run is distributed.
    };
}
