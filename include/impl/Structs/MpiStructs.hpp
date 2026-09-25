#pragma once

// Include Definitions
#include "Definitions.hpp"

// Include necessary structs
#include <algorithm>
#include <mpi.h>
#include <stdexcept>
#include "impl/Structs/Simdat.hpp"

namespace Condor::impl
{

    template<typename FloatType>
    class CondorMPI{
    private:
        int getNeighborRank(MPI_Comm cart_comm, const int shifted_coords[2]) const
        {
            if (shifted_coords[0] < 0 || shifted_coords[0] >= dims[0] ||
                shifted_coords[1] < 0 || shifted_coords[1] >= dims[1]) {
                return MPI_PROC_NULL;
            }

            int neighbor_rank = MPI_PROC_NULL;
            MPI_Cart_rank(cart_comm, shifted_coords, &neighbor_rank);
            return neighbor_rank;
        }

        void makeNeighbors(MPI_Comm cart_comm)
        {
            const int displacements[8][2] = {
                {-1, -1}, {0, -1}, {1, -1}, {1, 0},
                {1, 1}, {0, 1}, {-1, 1}, {-1, 0}
            };

            for (int n = 0; n < 8; n++) {
                const int shifted_coords[2] = {
                    coords[0] + displacements[n][0],
                    coords[1] + displacements[n][1]
                };
                neighbor_ranks[n] = getNeighborRank(cart_comm, shifted_coords);
            }
        }

    public:
        CondorMPI(MPI_Comm inComm, const int overlap_points_in = 0)
            : overlap_points(overlap_points_in)
        {
            if (overlap_points < 0) {
                throw std::invalid_argument("CondorMPI overlap_points must be non-negative.");
            }

            // Set Comm information
            comm = inComm;

            // Get MPI Info
            MPI_Comm_rank(comm, &rank);
            MPI_Comm_size(comm, &nproc);

            // TODO::CUSTOM DECOMPOSITION
            // Get dims for 2D decomposition
            MPI_Dims_create(nproc, 2, dims);

            // Create coms
            int periods[2] = {0,0}; // non-periodic boundaries
            MPI_Comm cart_comm;
            MPI_Cart_create(comm, 2, dims, periods, 0, &cart_comm); 

            // Set coords to coordinate of local process
            MPI_Cart_coords(cart_comm, rank, 2, coords);  

            // Set neighbor ranks in bot-left, down, bot-right, right, up-right, up, up-left, left order.
            makeNeighbors(cart_comm);
        }

        // Communication Stuff
        MPI_Comm comm;

        // MPI info
        int rank;
        int nproc;
        int overlap_points;

        // Ranks of neighboring Cartesian processes.
        int neighbor_ranks[8] = {
            MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL,
            MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL
        };

        // Locality info
        int i_min, i_max;
        int j_min, j_max;

        // Dimensionality of decomposition
        int dims[2] = {0,0};    // Dimensionality of decomposition
        int coords[2] = {0,0};   // Own coordinates

        // Stork Header
        Stork::Structs::RegularGrid_Header<FloatType, host_space> header;

        // Make x-y bounds for local domain
        void makeLocalBounds(Simdat<FloatType>& sim){
            // Global Decomposition
            const int I = dims[0];
            const int J = dims[1];
            
            // Local Decomposition Number
            const int i = coords[0];
            const int j = coords[1];

            // Split by points, then optionally extend non-first ranks backward.
            const int base_i_min = (sim.domain.xnum * i) / I;
            const int base_i_max = (sim.domain.xnum * (i + 1)) / I - 1;

            const int base_j_min = (sim.domain.ynum * j) / J;
            const int base_j_max = (sim.domain.ynum * (j + 1)) / J - 1;

            i_min = (i > 0) ? std::max(0, base_i_min - overlap_points) : base_i_min;
            i_max = base_i_max;

            j_min = (j > 0) ? std::max(0, base_j_min - overlap_points) : base_j_min;
            j_max = base_j_max;

            // Set header info
            header.global_i0() = i_min;
            header.global_j0() = j_min;
            header.global_x0() = sim.domain.xmin;
            header.global_y0() = sim.domain.ymin;
            header.local_inum() = 1+(i_max-i_min);
            header.local_jnum() = 1+(j_max-j_min);

            // Find local domain bounds (no)
            const FloatType xmin = sim.domain.xmin + sim.domain.res*i_min;
            const FloatType xmax = sim.domain.xmin + sim.domain.res*i_max;
            
            const FloatType ymin = sim.domain.ymin + sim.domain.res*j_min;
            const FloatType ymax = sim.domain.ymin + sim.domain.res*j_max;

            // Now adjust local domain
            sim.domain.xmin = xmin; sim.domain.xmax = xmax;
            sim.domain.xnum = 1 + int(0.5 + (sim.domain.xmax - sim.domain.xmin) / sim.domain.res);
            sim.domain.xmax = sim.domain.xmin + (sim.domain.xnum - 1) * sim.domain.res;
            
            sim.domain.ymin = ymin; sim.domain.ymax = ymax;
            sim.domain.ynum = 1 + int(0.5 + (sim.domain.ymax - sim.domain.ymin) / sim.domain.res);
            sim.domain.ymax = sim.domain.ymin + (sim.domain.ynum - 1) * sim.domain.res;

            sim.domain.pnum = sim.domain.xnum*sim.domain.ynum*sim.domain.znum;
        }
    };
}
