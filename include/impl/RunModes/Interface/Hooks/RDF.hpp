#pragma once

// Include definitions.
#include "Definitions.hpp"

// Include the structures needed by the Interface RDF hook.
#include "impl/Meltpool/Tracking.hpp"
#include "impl/RunModes/Interface/Hooks/Solidification.hpp"
#include "impl/Structs/Simdat.hpp"
#include "impl/Utility/Grid.hpp"

#include <mpi.h>
#include <stdexcept>
#include <vector>

namespace Condor::impl::Modes::Interface::Hooks {

    template<typename FloatType>
    class RDF_Hook {
        private:
            struct RDF_Event {
                uint32_t local_p = 0;
                FloatType tm = static_cast<FloatType>(0.0);
                FloatType tl = static_cast<FloatType>(0.0);
                FloatType cr = static_cast<FloatType>(0.0);
            };

            Stork::Structs::RDF_Dual<FloatType>* RDF = nullptr;
            bool shouldSkip = true;
            bool headerInitialized = false;
            std::vector<RDF_Event> events;

            void InitializeHeader(const Simdat<FloatType>& sim) {
                Stork::Structs::RegularGrid_Header<FloatType, host_space>& header = RDF->host_header;

                if (sim.mpi_decomposition.active) {
                    header.global_i0() = sim.mpi_decomposition.global_i0;
                    header.global_j0() = sim.mpi_decomposition.global_j0;
                    header.global_k0() = sim.mpi_decomposition.global_k0;
                    header.local_inum() = sim.mpi_decomposition.local_inum;
                    header.local_jnum() = sim.mpi_decomposition.local_jnum;
                    header.local_knum() = sim.mpi_decomposition.local_knum;
                    header.global_x0() = sim.mpi_decomposition.global_x0;
                    header.global_y0() = sim.mpi_decomposition.global_y0;
                    header.global_z0() = sim.mpi_decomposition.global_z0;
                    header.gridResolution() = sim.mpi_decomposition.gridResolution;
                    return;
                }

                header.global_i0() = static_cast<uint32_t>(0);
                header.global_j0() = static_cast<uint32_t>(0);
                header.global_k0() = static_cast<uint32_t>(0);
                header.local_inum() = static_cast<uint32_t>(sim.domain.xnum);
                header.local_jnum() = static_cast<uint32_t>(sim.domain.ynum);
                header.local_knum() = static_cast<uint32_t>(sim.domain.znum);

                header.global_x0() = (sim.domain.xnum == 1) ? sim.domain.xmax : sim.domain.xmin;
                header.global_y0() = (sim.domain.ynum == 1) ? sim.domain.ymax : sim.domain.ymin;
                header.global_z0() = (sim.domain.znum == 1) ? sim.domain.zmax : sim.domain.zmin;
                header.gridResolution() = (sim.domain.xres != static_cast<FloatType>(0.0)) ? sim.domain.xres : sim.domain.res;
            }

            uint32_t LocalPointFromP(const Simdat<FloatType>& sim, const int p) const {
                const uint32_t local_ijk[3] = {
                    static_cast<uint32_t>(Grid::p_to_i(p, sim.domain.ynum, sim.domain.znum)),
                    static_cast<uint32_t>(Grid::p_to_j(p, sim.domain.ynum, sim.domain.znum)),
                    static_cast<uint32_t>(Grid::p_to_k(p, sim.domain.znum))
                };

                uint32_t local_p = 0;
                RDF->host_header.LOCAL_ijk_to_LOCAL_p(local_p, local_ijk);
                return local_p;
            }

            void SyncSpatialPointOverlap(const Simdat<FloatType>& sim) {
                if (sim.settings.nproc <= 1 || !sim.mpi_decomposition.active) {
                    return;
                }

                Stork::Structs::RegularGrid_Header<FloatType, host_space>& header = RDF->host_header;

                const uint32_t curSize = static_cast<uint32_t>(events.size());

                // Find how many RDF points to send to each neighbor
                uint32_t send_sizes[3] = {0, 0, 0};
                for (uint32_t event = 0; event < curSize; event++) {
                    uint32_t ijk[3];
                    header.LOCAL_p_to_LOCAL_ijk(ijk, events[event].local_p);
                    if (ijk[0] <= 1) {
                        send_sizes[0]++;
                    }
                    if (ijk[0] <= 1 && ijk[1] <= 1) {
                        send_sizes[1]++;
                    }
                    if (ijk[1] <= 1) {
                        send_sizes[2]++;
                    }
                }

                // Make buffers the correct size
                std::vector<uint32_t> send_ijk[3];
                std::vector<FloatType> send_data[3];
                for (int dn = 0; dn < 3; dn++) {
                    send_ijk[dn].resize(3 * send_sizes[dn]);
                    send_data[dn].resize(3 * send_sizes[dn]);
                }

                // Reset send sizes and pack send vectors
                send_sizes[0] = 0;
                send_sizes[1] = 0;
                send_sizes[2] = 0;
                for (uint32_t event = 0; event < curSize; event++) {
                    uint32_t ijk[3];
                    header.LOCAL_p_to_LOCAL_ijk(ijk, events[event].local_p);

                    for (int dn = 0; dn < 3; dn++) {
                        // If we aren't packing this, continue to next iteration
                        const bool should_pack =
                            (dn == 0 && ijk[0] <= 1) ||
                            (dn == 1 && ijk[0] <= 1 && ijk[1] <= 1) ||
                            (dn == 2 && ijk[1] <= 1);
                        if (!should_pack) {continue;}

                        // Increment the send send count for this neighbor and get the index to send to
                        const uint32_t sendNum = send_sizes[dn]++;
                        send_ijk[dn][3 * sendNum + 0] = (dn == 0 || dn == 1) ? ((ijk[0] == 1) ? UINT32_MAX : (UINT32_MAX - 1)) : ijk[0];
                        send_ijk[dn][3 * sendNum + 1] = (dn == 1 || dn == 2) ? ((ijk[1] == 1) ? UINT32_MAX : (UINT32_MAX - 1)) : ijk[1];
                        send_ijk[dn][3 * sendNum + 2] = ijk[2];
                        send_data[dn][3 * sendNum + 0] = events[event].tm;
                        send_data[dn][3 * sendNum + 1] = events[event].tl;
                        send_data[dn][3 * sendNum + 2] = events[event].cr;
                    }
                }

                // --- Now send and recieve with neighbors --- //

                // Comm requests (NOTE::Only left, down-left, down for sending and right, up-right, up for receiving since the other directions are handled by the neighbors)
                const int send_neighbors[3] = {7, 0, 1};
                const int recv_neighbors[3] = {3, 4, 5};
                MPI_Request send_size_requests[3] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL};
                MPI_Request recv_size_requests[3] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL};
                MPI_Request send_data_requests[6] = {
                    MPI_REQUEST_NULL, MPI_REQUEST_NULL,
                    MPI_REQUEST_NULL, MPI_REQUEST_NULL,
                    MPI_REQUEST_NULL, MPI_REQUEST_NULL
                };
                MPI_Request recv_data_requests[6] = {
                    MPI_REQUEST_NULL, MPI_REQUEST_NULL,
                    MPI_REQUEST_NULL, MPI_REQUEST_NULL,
                    MPI_REQUEST_NULL, MPI_REQUEST_NULL
                };

                // Now do nonblocking sends and receives of sizes.
                uint32_t recv_sizes[3] = {0, 0, 0};
                for (int dn = 0; dn < 3; dn++) {
                    const int send_rank = sim.mpi_decomposition.neighbor_ranks[send_neighbors[dn]];
                    const int recv_rank = sim.mpi_decomposition.neighbor_ranks[recv_neighbors[dn]];
                    if (send_rank != MPI_PROC_NULL) {
                        MPI_Isend(&send_sizes[dn], sizeof(uint32_t), MPI_BYTE, send_rank, 0, MPI_COMM_WORLD, &send_size_requests[dn]);
                    }
                    if (recv_rank != MPI_PROC_NULL) {
                        MPI_Irecv(&recv_sizes[dn], sizeof(uint32_t), MPI_BYTE, recv_rank, 0, MPI_COMM_WORLD, &recv_size_requests[dn]);
                    }
                }
                MPI_Waitall(3, recv_size_requests, MPI_STATUSES_IGNORE);
                MPI_Waitall(3, send_size_requests, MPI_STATUSES_IGNORE);

                // Resize host recv buffers based on sizes
                std::vector<uint32_t> recv_ijk[3];
                std::vector<FloatType> recv_data[3];
                for (int dn = 0; dn < 3; dn++) {
                    recv_ijk[dn].resize(3 * recv_sizes[dn]);
                    recv_data[dn].resize(3 * recv_sizes[dn]);
                }

                // Now do nonblocking sends and receives of data.
                for (int dn = 0; dn < 3; dn++) {
                    const int send_rank = sim.mpi_decomposition.neighbor_ranks[send_neighbors[dn]];
                    const int recv_rank = sim.mpi_decomposition.neighbor_ranks[recv_neighbors[dn]];
                    if (send_rank != MPI_PROC_NULL && send_sizes[dn] > 0) {
                        MPI_Isend(send_ijk[dn].data(), static_cast<int>(3 * sizeof(uint32_t) * send_sizes[dn]), MPI_BYTE, send_rank, 1, MPI_COMM_WORLD, &send_data_requests[2 * dn + 0]);
                        MPI_Isend(send_data[dn].data(), static_cast<int>(3 * sizeof(FloatType) * send_sizes[dn]), MPI_BYTE, send_rank, 2, MPI_COMM_WORLD, &send_data_requests[2 * dn + 1]);
                    }
                    if (recv_rank != MPI_PROC_NULL && recv_sizes[dn] > 0) {
                        MPI_Irecv(recv_ijk[dn].data(), static_cast<int>(3 * sizeof(uint32_t) * recv_sizes[dn]), MPI_BYTE, recv_rank, 1, MPI_COMM_WORLD, &recv_data_requests[2 * dn + 0]);
                        MPI_Irecv(recv_data[dn].data(), static_cast<int>(3 * sizeof(FloatType) * recv_sizes[dn]), MPI_BYTE, recv_rank, 2, MPI_COMM_WORLD, &recv_data_requests[2 * dn + 1]);
                    }
                }
                MPI_Waitall(6, recv_data_requests, MPI_STATUSES_IGNORE);
                MPI_Waitall(6, send_data_requests, MPI_STATUSES_IGNORE);

                // Find how many events we need to add and resize if necesary
                uint32_t numToAppend = 0;
                for (int dn = 0; dn < 3; dn++) {
                    numToAppend += recv_sizes[dn];
                }
                events.reserve(events.size() + numToAppend);

                // Turn p into local ijk
                std::vector<uint32_t> local_ijk(3 * curSize);

                // Get ijk of data already in the RDF so we can remap it after we change the header for the new local domain decomposition.
                for (uint32_t event = 0; event < curSize; event++) {
                    uint32_t ijk[3];
                    header.LOCAL_p_to_LOCAL_ijk(ijk, events[event].local_p);
                    local_ijk[3 * event + 0] = ijk[0];
                    local_ijk[3 * event + 1] = ijk[1];
                    local_ijk[3 * event + 2] = ijk[2];
                }

                // Now make local header bigger (to correspond to buffer increase)
                // NOTE::No need to change offsets since the min corner isn't moving
                header.local_inum() += 2*static_cast<uint32_t>(sim.mpi_decomposition.coords[0] != (sim.mpi_decomposition.dims[0] - 1));
                header.local_jnum() += 2*static_cast<uint32_t>(sim.mpi_decomposition.coords[1] != (sim.mpi_decomposition.dims[1] - 1));

                // Now shift current data
                for (uint32_t event = 0; event < curSize; event++) {
                    const uint32_t ijk[3] = {
                        local_ijk[3 * event + 0],
                        local_ijk[3 * event + 1],
                        local_ijk[3 * event + 2]
                    };
                    header.LOCAL_ijk_to_LOCAL_p(events[event].local_p, ijk);
                }

                // Now copy data in
                for (int dn = 0; dn < 3; dn++) {
                    if (recv_sizes[dn] == 0) {continue;}
                    for (uint32_t event = 0; event < recv_sizes[dn]; event++) {
                        uint32_t ijk[3] = {
                            recv_ijk[dn][3 * event + 0],
                            recv_ijk[dn][3 * event + 1],
                            recv_ijk[dn][3 * event + 2]
                        };
                        // Set ijk of points on the boundary to the max local index
                        if (ijk[0] == UINT32_MAX) { ijk[0] = header.local_inum() - 1; }
                        if (ijk[0] == (UINT32_MAX - 1)) { ijk[0] = header.local_inum() - 2; }
                        if (ijk[1] == UINT32_MAX) { ijk[1] = header.local_jnum() - 1; }
                        if (ijk[1] == (UINT32_MAX - 1)) { ijk[1] = header.local_jnum() - 2; }
                        // Shift to new frame;
                        uint32_t local_p = 0;
                        header.LOCAL_ijk_to_LOCAL_p(local_p, ijk);
                        events.push_back(RDF_Event{
                            local_p,
                            recv_data[dn][3 * event + 0],
                            recv_data[dn][3 * event + 1],
                            recv_data[dn][3 * event + 2]
                        });
                    }
                }
            }

            void FinishConstruction() {
                if (shouldSkip) {
                    return;
                }

                const uint32_t numEvents = static_cast<uint32_t>(events.size());
                RDF->template Make_Data_Views<host_space>(numEvents);
                RDF->numEvents = numEvents;

                Kokkos::Profiling::pushRegion("Interface::Hooks::RDF::Construct");
                Stork::Structs::RDF_Data<FloatType, host_space>& data = RDF->host_data;
                for (uint32_t n = 0; n < numEvents; n++) {
                    data.p(n) = events[n].local_p;
                    data.tm(n) = events[n].tm;
                    data.tl(n) = events[n].tl;
                    data.cr(n) = events[n].cr;
                }
                Kokkos::Profiling::popRegion();

                RDF->template Make_Data_Mirrors<host_space, device_space>();
                RDF->template Copy_All<host_space, device_space>();
            }

        public:
            RDF_Hook(Stork::Structs::RDF_Dual<FloatType>* RDF_in)
                : RDF(RDF_in),
                  shouldSkip(RDF_in == nullptr)
            {}

            void Initialize(
                const Simdat<FloatType>& sim,
                Meltpool::Tracking<FloatType>& meltpool,
                Solidification_Hook<FloatType>& solidification_hook)
            {
                if (shouldSkip) { return; }
                
                // Initialize vector size for events
                events.reserve(sim.domain.pnum);

                // Adjust solidification hook's calculators
                solidification_hook.calc.fields.tSol = true;
                solidification_hook.calc.fields.dTdt = true;

                // Adjust meltpool to be RDF mode
                meltpool.calc.Reconstruction = true;
                meltpool.calc.Solidification = true;
                meltpool.calc.RDF = true;
                
                // Initialize the regular grid header
                InitializeHeader(sim);
            }

            void PostStep(
                const Solidification_Hook<FloatType>& solidification_hook,
                const Meltpool::Tracking<FloatType>& meltpool,
                const Simdat<FloatType>& sim)
            {
                // Skip if not doing
                if (shouldSkip) { return;}

                // Skip if there are no events transfered to the host this timestep
                const typename Meltpool::Tracking<FloatType>::Tracking_SolidificationEvents& trackingEvents = meltpool.Events();
                const int eventCount = trackingEvents.host_size;
                if (eventCount <= 0) { return;}
                if (solidification_hook.EventResultCount() != eventCount) {
                    throw std::runtime_error("RDF hook expected event-indexed solidification results for every meltpool event.");
                }

                
                // Add them to a vector
                Kokkos::Profiling::pushRegion("Interface::Hooks::RDF::PostStep");
                for (int n = 0; n < eventCount; n++) {
                    const int p = trackingEvents.event_p_h(n);
                    events.push_back(RDF_Event{
                        LocalPointFromP(sim, p),
                        trackingEvents.event_tm_h(n),
                        solidification_hook.eventTSol(n),
                        solidification_hook.eventDtdt(n)
                    });
                }
                Kokkos::Profiling::popRegion();
            }

            void Finalize(
                const Solidification_Hook<FloatType>& solidification_hook,
                const Meltpool::Tracking<FloatType>& meltpool,
                const Simdat<FloatType>& sim)
            {
                // Skip if not doing
                if (shouldSkip) { return;}

                PostStep(solidification_hook, meltpool, sim);
                Kokkos::Profiling::pushRegion("Interface::Hooks::SRDF::Finalize");
                SyncSpatialPointOverlap(sim);
                Kokkos::Profiling::popRegion();
                FinishConstruction();
            }
    };
}
