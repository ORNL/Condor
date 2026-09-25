// Include Library
#include <Condor_Core.hpp>
#include <mpi.h>

// TODO::ADJUSTABLE
// Typedef
using FloatType = double;

int main(int argc, char * argv[]) {	
	// Initialize MPI
    MPI_Init(&argc, &argv);
	{
        // Initialize Kokkos
        Kokkos::initialize(argc, argv);
        {   
		    Condor::Run::Classic<FloatType>(argc, argv);
        }
        // Finalize Kokkos
        Kokkos::finalize();
    }
	// Finalize MPI
    MPI_Finalize();

	return 0;
}
