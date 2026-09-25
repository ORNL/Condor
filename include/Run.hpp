// Include definitions
#include "Definitions.hpp"

// Include necessary structs
#include "impl/Structs/Simdat.hpp"
#include "impl/Initialization/Init.hpp"
#include "impl/Output/Logger.hpp"
#include "impl/RunModes/Modes.hpp"
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace Condor::Run{

    template<typename FloatType>
    inline void Classic(int argc, char * argv[])
    {
        // Namespace declarations
        using std::string;

        // Parse command line arguments with lightweight defaults for common runs.
        string inputFile = "ParamInput.json";
        if (argc > 1) {
            inputFile = argv[1];
        }
  
        // Initialize struct for simulation parameters
        impl::CondorMPI<FloatType> mpi(MPI_COMM_WORLD);
        impl::Simdat<FloatType> sim;

        // Initialize Simdat
        impl::Init::Initialize_Simdat(sim, inputFile, mpi);

        // Intialize the logger based on the requested print level and rank
        impl::Out::Logger logger(mpi.rank, sim.settings.print);
        
        // Run the simulation
        impl::Modes::Simulate(sim);

        std::string rank_name = "";
        if (mpi.nproc > 1) {
            rank_name = ".rank" + std::to_string(mpi.rank);
        } 
    }

    namespace detail {
        inline void CheckCoupledMode(const std::string& modeType) {
            if (modeType != "Interface") {
                throw std::runtime_error(
                    "Condor::Run::Coupled requires Interface mode for coupled RDF/SRDF solidification; detected mode: " +
                    modeType);
            }
        }
    }

    template<typename FloatType>
    inline void Coupled(Stork::Structs::RDF_Dual<FloatType>& RDF, const std::string& inputFile = "ParamInput.json")
    {
        // Initialize struct for simulation parameters
        impl::CondorMPI<FloatType> mpi(MPI_COMM_WORLD);
        impl::Simdat<FloatType> sim;

        // Initialize Simdat
        impl::Init::Initialize_Simdat(sim, inputFile, mpi);

        // Coupled RDF/SRDF solidification is implemented by Interface mode.
        detail::CheckCoupledMode(sim.files.modeType);

        // Intialize the logger based on the requested print level and rank
        impl::Out::Logger logger(mpi.rank, sim.settings.print);
        
        // Run Interface while publishing RDF into the caller-owned container.
        impl::Modes::Interface::Run(sim, RDF);
    }

    template<typename FloatType>
    inline void Coupled(Stork::Structs::SRDF_Dual<FloatType>& SRDF, const std::string& inputFile = "ParamInput.json")
    {
        // Initialize struct for simulation parameters
        impl::CondorMPI<FloatType> mpi(MPI_COMM_WORLD, 1);
        impl::Simdat<FloatType> sim;

        // Initialize Simdat
        impl::Init::Initialize_Simdat(sim, inputFile, mpi);

        // Coupled RDF/SRDF solidification is implemented by Interface mode.
        detail::CheckCoupledMode(sim.files.modeType);

        // Intialize the logger based on the requested print level and rank
        impl::Out::Logger logger(mpi.rank, sim.settings.print);
        
        // Run Interface while publishing SRDF into the caller-owned container.
        impl::Modes::Interface::Run(sim, SRDF);
    }
}
