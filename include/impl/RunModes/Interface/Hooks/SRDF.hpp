#pragma once

// Include definitions.
#include "Definitions.hpp"

// Include the structures needed by the Interface SRDF hook.
#include "impl/Calc/Structs.hpp"
#include "impl/Calc/Temperature.hpp"
#include "impl/Meltpool/Tracking.hpp"
#include "impl/RunModes/Interface/Hooks/Solidification.hpp"
#include "impl/RunModes/Interface/Structs.hpp"
#include "impl/Structs/Simdat.hpp"
#include "impl/Utility/Grid.hpp"

#include <algorithm>
#include <cstdint>
#include <mpi.h>
#include <optional>

namespace Condor::impl::Modes::Interface::Hooks {

    template<typename FloatType>
    class SRDF_Hook {
        private:
            using floating_deviceView = Kokkos::View<FloatType*, layout, device_memory>;
            using floating_hostView = Kokkos::View<FloatType*, layout, host_memory>;
            using int1_deviceView = Kokkos::View<int[1], layout, device_memory>;
            using int1_hostView = Kokkos::View<int[1], layout, host_memory>;
            using int2_deviceView = Kokkos::View<int[2], layout, device_memory>;
            using int2_hostView = Kokkos::View<int[2], layout, host_memory>;
            using uint32_3_deviceView = Kokkos::View<uint32_t[3], layout, device_memory>;
            using uint32_3_hostView = Kokkos::View<uint32_t[3], layout, host_memory>;
            using team_policy = Kokkos::TeamPolicy<device_exe>;
            using team_member = typename team_policy::member_type;

            Stork::Structs::SRDF_Dual<FloatType>* SRDF = nullptr;
            bool shouldSkip = true;

            // Initial capacity
            uint32_t storkCapacity = 65536;

            // Liquid counters
            bool prevStep_hasLiq = false;
            bool curStep_hasLiq = false;
            int last_itert = -1;
            bool pingPongState = true;

            // Constants
            std::optional<Calc::Device_Constants<FloatType>> const_d;

            // Cell stuff
            int c_pnum = 0;
            uint8_deviceView c_alpha;
            uint8_deviceView c_beta;
            int_deviceView cells_to_add;
            int1_deviceView cell_count_d;
            int1_hostView cell_count_h;
            
            // Vertex stuff
            int v_pnum = 0;
            int_deviceView vertex_prev_stamp;
            int_deviceView vertex_cur_stamp;
            int_deviceView calc_prev_vertex;
            int_deviceView calc_cur_vertex;
            int2_deviceView calc_sizes;
            int2_hostView calc_sizes_h;

            // Initialize SRDF header
            void InitializeHeader(const Simdat<FloatType>& sim) {
                Stork::Structs::RegularGrid_Header<FloatType, host_space>& header = SRDF->host_header;

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
            
            void EnsureCapacity(const int numToAdd) {
                if (numToAdd <= 0) {
                    return;
                }

                Stork::Structs::SRDF_Data<FloatType, device_space>& data = SRDF->template get_data<device_space>();
                if (storkCapacity <= SRDF->numSnaps + static_cast<uint32_t>(numToAdd)) {
                    while (storkCapacity <= SRDF->numSnaps + static_cast<uint32_t>(numToAdd)) {
                        storkCapacity *= 2;
                    }

                    Kokkos::resize(data.cellNum_view, storkCapacity);
                    Kokkos::resize(data.times_view, 2 * storkCapacity);
                    Kokkos::resize(data.thermals_view, 16 * storkCapacity);
                }
            }

            void EnsureCapacity_Exact(const int numToAdd) {
                if (numToAdd <= 0) {
                    return;
                }

                Stork::Structs::SRDF_Data<FloatType, device_space>& data = SRDF->template get_data<device_space>();
                if (storkCapacity <= SRDF->numSnaps + static_cast<uint32_t>(numToAdd)) {
                    // Set capacity to the exact size needed
                    storkCapacity = SRDF->numSnaps + static_cast<uint32_t>(numToAdd);
                    Kokkos::resize(data.cellNum_view, storkCapacity);
                    Kokkos::resize(data.times_view, 2 * storkCapacity);
                    Kokkos::resize(data.thermals_view, 16 * storkCapacity);
                }
            }

        public:
            void SyncSpatialCellOverlap(const Simdat<FloatType>& sim) {
                if (sim.settings.nproc <= 1 || !sim.mpi_decomposition.active) {
                    return;
                }

                Stork::Structs::SRDF_Data<FloatType, device_space>& data = SRDF->template get_data<device_space>();
                Stork::Structs::RegularGrid_Header<FloatType, device_space> header = SRDF->template get_header<device_space>();

                const uint32_t curSize = SRDF->numSnaps;

                // Find how many SRDF cells to send to each neighbor
                uint32_3_deviceView send_sizes_d("interface_hook_srdf_mpi_send_sizes");
                Kokkos::deep_copy(send_sizes_d, static_cast<uint32_t>(0));
                Kokkos::parallel_for(
                    "interface_hook_srdf_mpi_count",
                    Kokkos::RangePolicy<device_exe>(0, curSize),
                    KOKKOS_LAMBDA(const uint32_t snap) {
                        uint32_t ijk[3];
                        header.LOCAL_p_to_LOCAL_ijk(ijk, data.p(snap));
                        if (ijk[0] == 0) {
                            Kokkos::atomic_fetch_add(&send_sizes_d(0), static_cast<uint32_t>(1));
                        }
                        if (ijk[0] == 0 && ijk[1] == 0) {
                            Kokkos::atomic_fetch_add(&send_sizes_d(1), static_cast<uint32_t>(1));
                        }
                        if (ijk[1] == 0) {
                            Kokkos::atomic_fetch_add(&send_sizes_d(2), static_cast<uint32_t>(1));
                        }
                    }
                );
                uint32_3_hostView send_sizes_h = Kokkos::create_mirror_view_and_copy(host_memory(), send_sizes_d);

                // Make buffers the correct size
                uint32_deviceView send_ijk_d[3];
                floating_deviceView send_times_d[3];
                floating_deviceView send_thermals_d[3];
                for (int dn = 0; dn < 3; dn++) {
                    send_ijk_d[dn] = uint32_deviceView(Kokkos::ViewAllocateWithoutInitializing("interface_hook_srdf_mpi_send_ijk"), 3 * send_sizes_h(dn));
                    send_times_d[dn] = floating_deviceView(Kokkos::ViewAllocateWithoutInitializing("interface_hook_srdf_mpi_send_times"), 2 * send_sizes_h(dn));
                    send_thermals_d[dn] = floating_deviceView(Kokkos::ViewAllocateWithoutInitializing("interface_hook_srdf_mpi_send_thermals"), 16 * send_sizes_h(dn));
                }

                // Reset send sizes and pack send vectors
                Kokkos::deep_copy(send_sizes_d, static_cast<uint32_t>(0));
                Kokkos::parallel_for(
                    "interface_hook_srdf_mpi_pack",
                    Kokkos::RangePolicy<device_exe>(0, curSize),
                    KOKKOS_LAMBDA(const uint32_t snap) {
                        uint32_t ijk[3];
                        header.LOCAL_p_to_LOCAL_ijk(ijk, data.p(snap));

                        for (int dn = 0; dn < 3; dn++) {
                            // If we aren't packing this, continue to next iteration
                            const bool should_pack =
                                (dn == 0 && ijk[0] == 0) ||
                                (dn == 1 && ijk[0] == 0 && ijk[1] == 0) ||
                                (dn == 2 && ijk[1] == 0);
                            if (!should_pack) {continue;}
                            
                            // Increment the send send count for this neighbor and get the index to send to
                            const uint32_t sendNum = Kokkos::atomic_fetch_add(&send_sizes_d(dn), static_cast<uint32_t>(1));
                            send_ijk_d[dn](3 * sendNum + 0) = (dn == 0 || dn == 1) ? UINT32_MAX : ijk[0];
                            send_ijk_d[dn](3 * sendNum + 1) = (dn == 1 || dn == 2) ? UINT32_MAX : ijk[1];
                            send_ijk_d[dn](3 * sendNum + 2) = ijk[2];
                            send_times_d[dn](2 * sendNum + 0) = data.t_prev(snap);
                            send_times_d[dn](2 * sendNum + 1) = data.t_cur(snap);
                            for (int q = 0; q < 16; q++) {
                                send_thermals_d[dn](16 * sendNum + q) = data.thermals_view(16 * snap + q);
                            }
                        }
                    }
                );
                Kokkos::fence();

                // Copy each to host
                uint32_hostView send_ijk_h[3];
                floating_hostView send_times_h[3];
                floating_hostView send_thermals_h[3];
                for (int dn = 0; dn < 3; dn++) {
                    send_ijk_h[dn] = Kokkos::create_mirror_view_and_copy(host_memory(), send_ijk_d[dn]);
                    send_times_h[dn] = Kokkos::create_mirror_view_and_copy(host_memory(), send_times_d[dn]);
                    send_thermals_h[dn] = Kokkos::create_mirror_view_and_copy(host_memory(), send_thermals_d[dn]);
                }

                // --- Now send and recieve with neighbors --- //

                // Comm requests (NOTE::Only left, down-left, down for sending and right, up-right, up for receiving since the other directions are handled by the neighbors)
                const int send_neighbors[3] = {7, 0, 1};
                const int recv_neighbors[3] = {3, 4, 5};
                MPI_Request send_size_requests[3] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL};
                MPI_Request recv_size_requests[3] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL};
                MPI_Request send_data_requests[9] = {
                    MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL,
                    MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL,
                    MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL
                };
                MPI_Request recv_data_requests[9] = {
                    MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL,
                    MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL,
                    MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL
                };

                // Now do nonblocking sends and receives of sizes.
                uint32_t recv_sizes[3] = {0, 0, 0};
                for (int dn = 0; dn < 3; dn++) {
                     const int send_rank = sim.mpi_decomposition.neighbor_ranks[send_neighbors[dn]];
                    const int recv_rank = sim.mpi_decomposition.neighbor_ranks[recv_neighbors[dn]];   
                    if (send_rank != MPI_PROC_NULL) {
                        MPI_Isend(&send_sizes_h(dn), sizeof(uint32_t), MPI_BYTE, send_rank, 0, MPI_COMM_WORLD, &send_size_requests[dn]);
                    }
                    if (recv_rank != MPI_PROC_NULL) {
                        MPI_Irecv(&recv_sizes[dn], sizeof(uint32_t), MPI_BYTE, recv_rank, 0, MPI_COMM_WORLD, &recv_size_requests[dn]);
                    }
                }
                MPI_Waitall(3, recv_size_requests, MPI_STATUSES_IGNORE);
                MPI_Waitall(3, send_size_requests, MPI_STATUSES_IGNORE);
                
                // Resize host recv buffers based on sizes
                uint32_hostView recv_ijk_h[3];
                floating_hostView recv_times_h[3];
                floating_hostView recv_thermals_h[3];
                for (int dn = 0; dn < 3; dn++) {
                    recv_ijk_h[dn] = uint32_hostView(Kokkos::ViewAllocateWithoutInitializing("interface_hook_srdf_mpi_recv_ijk"), 3 * recv_sizes[dn]);
                    recv_times_h[dn] = floating_hostView(Kokkos::ViewAllocateWithoutInitializing("interface_hook_srdf_mpi_recv_times"), 2 * recv_sizes[dn]);
                    recv_thermals_h[dn] = floating_hostView(Kokkos::ViewAllocateWithoutInitializing("interface_hook_srdf_mpi_recv_thermals"), 16 * recv_sizes[dn]);
                }

                // Now do nonblocking sends and receives of data.
                for (int dn = 0; dn < 3; dn++) {
                    const int send_rank = sim.mpi_decomposition.neighbor_ranks[send_neighbors[dn]];
                    const int recv_rank = sim.mpi_decomposition.neighbor_ranks[recv_neighbors[dn]];
                    if (send_rank != MPI_PROC_NULL && send_sizes_h(dn) > 0) {
                        MPI_Isend(send_ijk_h[dn].data(), static_cast<int>(3 * sizeof(uint32_t) * send_sizes_h(dn)), MPI_BYTE, send_rank, 1, MPI_COMM_WORLD, &send_data_requests[3 * dn + 0]);
                        MPI_Isend(send_times_h[dn].data(), static_cast<int>(2 * sizeof(FloatType) * send_sizes_h(dn)), MPI_BYTE, send_rank, 2, MPI_COMM_WORLD, &send_data_requests[3 * dn + 1]);
                        MPI_Isend(send_thermals_h[dn].data(), static_cast<int>(16 * sizeof(FloatType) * send_sizes_h(dn)), MPI_BYTE, send_rank, 3, MPI_COMM_WORLD, &send_data_requests[3 * dn + 2]);
                    }
                    if (recv_rank != MPI_PROC_NULL && recv_sizes[dn] > 0) {
                        MPI_Irecv(recv_ijk_h[dn].data(), static_cast<int>(3 * sizeof(uint32_t) * recv_sizes[dn]), MPI_BYTE, recv_rank, 1, MPI_COMM_WORLD, &recv_data_requests[3 * dn + 0]);
                        MPI_Irecv(recv_times_h[dn].data(), static_cast<int>(2 * sizeof(FloatType) * recv_sizes[dn]), MPI_BYTE, recv_rank, 2, MPI_COMM_WORLD, &recv_data_requests[3 * dn + 1]);
                        MPI_Irecv(recv_thermals_h[dn].data(), static_cast<int>(16 * sizeof(FloatType) * recv_sizes[dn]), MPI_BYTE, recv_rank, 3, MPI_COMM_WORLD, &recv_data_requests[3 * dn + 2]);
                    }
                }
                MPI_Waitall(9, recv_data_requests, MPI_STATUSES_IGNORE);
                MPI_Waitall(9, send_data_requests, MPI_STATUSES_IGNORE);

                // Find how many snaps we need to add and resize if necesary
                uint32_t numToAppend = 0;
                for (int dn = 0; dn < 3; dn++) {
                    numToAppend += recv_sizes[dn];
                }
                EnsureCapacity_Exact(static_cast<int>(numToAppend));

                // Turn p into local ijk
                uint32_deviceView local_ijk(Kokkos::ViewAllocateWithoutInitializing("interface_hook_srdf_mpi_local_ijk"), 3 * curSize);
                
                // Get ijk of data already in the SRDF so we can remap it after we change the header for the new local domain decomposition.
                Kokkos::parallel_for(
                    "interface_hook_srdf_mpi_local_ijk",
                    Kokkos::RangePolicy<device_exe>(0, curSize),
                    KOKKOS_LAMBDA(const uint32_t snap) {
                        uint32_t ijk[3];
                        header.LOCAL_p_to_LOCAL_ijk(ijk, data.p(snap));
                        local_ijk(3 * snap + 0) = ijk[0];
                        local_ijk(3 * snap + 1) = ijk[1];
                        local_ijk(3 * snap + 2) = ijk[2];
                    }
                );

                // Now make local header bigger (to correspond to buffer increase)
                // NOTE::No need to change offsets since the min corner isn't moving
                SRDF->template Copy_Header<device_space, host_space>();
                SRDF->host_header.local_inum() += static_cast<uint32_t>(sim.mpi_decomposition.coords[0] != (sim.mpi_decomposition.dims[0] - 1));
                SRDF->host_header.local_jnum() += static_cast<uint32_t>(sim.mpi_decomposition.coords[1] != (sim.mpi_decomposition.dims[1] - 1));
                SRDF->template Copy_Header<host_space, device_space>();

                // Now shift current data
                Kokkos::parallel_for(
                    "interface_hook_srdf_mpi_remap_existing",
                    Kokkos::RangePolicy<device_exe>(0, curSize),
                    KOKKOS_LAMBDA(const uint32_t snap) {
                        const uint32_t ijk[3] = {
                            local_ijk(3 * snap + 0),
                            local_ijk(3 * snap + 1),
                            local_ijk(3 * snap + 2)
                        };
                        header.LOCAL_ijk_to_LOCAL_p(data.p(snap), ijk);
                    }
                );
                Kokkos::fence();

                // Now copy data in
                for (int dn = 0; dn < 3; dn++) {
                    if (recv_sizes[dn] == 0) {continue;}
                    // Reference views
                    uint32_deviceView recv_ijk_d = Kokkos::create_mirror_view_and_copy(device_memory(), recv_ijk_h[dn]);
                    floating_deviceView recv_times_d = Kokkos::create_mirror_view_and_copy(device_memory(), recv_times_h[dn]);
                    floating_deviceView recv_thermals_d = Kokkos::create_mirror_view_and_copy(device_memory(), recv_thermals_h[dn]);
                    const uint32_t appendOffset = SRDF->numSnaps;
                    const uint32_t recvSize = recv_sizes[dn];
                    // Copy data in
                    Kokkos::parallel_for(
                        "interface_hook_srdf_mpi_append",
                        Kokkos::RangePolicy<device_exe>(0, recvSize),
                        KOKKOS_LAMBDA(const uint32_t snap) {
                            // Find index within SRDF to place data
                            const uint32_t idx = appendOffset + snap;
                            uint32_t ijk[3] = {
                                recv_ijk_d(3 * snap + 0),
                                recv_ijk_d(3 * snap + 1),
                                recv_ijk_d(3 * snap + 2)
                            };
                            // Set ijk of points on the boundary to the max local index
                            if (ijk[0] == UINT32_MAX) { ijk[0] = header.local_inum() - 2;}
                            if (ijk[1] == UINT32_MAX) { ijk[1] = header.local_jnum() - 2; }
                            // Shift to new frame;
                            header.LOCAL_ijk_to_LOCAL_p(data.p(idx), ijk);
                            // Set times
                            data.t_prev(idx) = recv_times_d(2 * snap + 0);
                            data.t_cur(idx) = recv_times_d(2 * snap + 1);
                            // Set thermal
                            for (int q = 0; q < 16; q++) {
                                data.thermals_view(16 * idx + q) = recv_thermals_d(16 * snap + q);
                            }
                        }
                    );
                    Kokkos::fence();
                    SRDF->numSnaps += recvSize;
                }
            }

            SRDF_Hook(Stork::Structs::SRDF_Dual<FloatType>* SRDF_in)
                : SRDF(SRDF_in),
                  shouldSkip(SRDF_in == nullptr)
            {}

            void Initialize(
                const Simdat<FloatType>& sim,
                Meltpool::Tracking<FloatType>& meltpool)
            {
                if (shouldSkip) { return; }

                Kokkos::Profiling::pushRegion("Interface::Hooks::SRDF::Initialize");

                // Constants
                const_d.emplace(sim);

                // Number of "cells"
                c_pnum = (sim.domain.xnum - 1) * (sim.domain.ynum - 1) * (sim.domain.znum - 1);

                // Number of verices
                v_pnum = sim.domain.pnum;

                // Counts of liquid cells
                c_alpha = uint8_deviceView(Kokkos::ViewAllocateWithoutInitializing("interface_hook_srdf_c_alpha"), c_pnum);
                c_beta = uint8_deviceView(Kokkos::ViewAllocateWithoutInitializing("interface_hook_srdf_c_beta"), c_pnum);
                cells_to_add = int_deviceView(Kokkos::ViewAllocateWithoutInitializing("interface_hook_srdf_cells_to_add"), c_pnum);
                cell_count_d = int1_deviceView("interface_hook_srdf_cell_count");
                cell_count_h = Kokkos::create_mirror_view(cell_count_d);
                vertex_prev_stamp = int_deviceView(Kokkos::ViewAllocateWithoutInitializing("interface_hook_srdf_vertex_prev_stamp"), v_pnum);
                vertex_cur_stamp = int_deviceView(Kokkos::ViewAllocateWithoutInitializing("interface_hook_srdf_vertex_cur_stamp"), v_pnum);
                calc_prev_vertex = int_deviceView(Kokkos::ViewAllocateWithoutInitializing("interface_hook_srdf_calc_prev_vertex"), v_pnum);
                calc_cur_vertex = int_deviceView(Kokkos::ViewAllocateWithoutInitializing("interface_hook_srdf_calc_cur_vertex"), v_pnum);
                calc_sizes = Kokkos::View<int[2], layout, device_memory>("interface_hook_srdf_calc_sizes");
                calc_sizes_h = Kokkos::create_mirror_view(calc_sizes);
                Kokkos::deep_copy(c_alpha, static_cast<uint8_t>(0));
                Kokkos::deep_copy(c_beta, static_cast<uint8_t>(0));
                Kokkos::deep_copy(vertex_prev_stamp, 0);
                Kokkos::deep_copy(vertex_cur_stamp, 0);

                // Initialize SRDF header
                InitializeHeader(sim);
                SRDF->numSnaps = 0;
                SRDF->T_critical = sim.material.T_liq; 
                SRDF->template Make_Data_Views<device_space>(storkCapacity);
                SRDF->template Copy_Header<host_space, device_space>();
                
                // Request reconstruction from tracking without enabling event storage.
                meltpool.calc.Reconstruction = true;

                Kokkos::Profiling::popRegion();
            }

            void PostStep(
                const Meltpool::Tracking<FloatType>& meltpool,
                const Simdat<FloatType>& sim,
                Interface_State<FloatType>& state)
            {
                if (shouldSkip || c_pnum <= 0) { return; }

                Kokkos::Profiling::pushRegion("Interface::Hooks::SRDF::PostStep");

                const Step_State<FloatType>& step = state.step;
                Thermal_State<FloatType>& thermal = state.thermal;
                const int active_count = meltpool.ActiveColumnCount();
                curStep_hasLiq = (active_count > 0);
                const bool should_process = curStep_hasLiq || prevStep_hasLiq;
                prevStep_hasLiq = curStep_hasLiq;
                if (!should_process) {
                    last_itert = step.iter_cur;
                    Kokkos::Profiling::popRegion();
                    return;
                }

                const int itert = step.iter_cur;
                const FloatType t_prev = step.t_prev;
                const FloatType t_cur = step.t_cur;

                // a->b->a ping pong for minimized memory movement
                uint8_deviceView c_prev = pingPongState ? c_alpha : c_beta;
                uint8_deviceView c_cur = pingPongState ? c_beta : c_alpha;
                int_deviceView cells_to_add_local = cells_to_add;
                int1_deviceView cell_count_local = cell_count_d;
                pingPongState = !pingPongState;

                // Reset counters
                Kokkos::deep_copy(c_cur, static_cast<uint8_t>(0));
                Kokkos::deep_copy(cell_count_local, 0);
                if (last_itert + 1 != itert) {
                    // May not be necessary?
                    Kokkos::deep_copy(c_prev, static_cast<uint8_t>(0));
                }
                
                // Set constants
                const int v_yNum = sim.domain.ynum;
                const int v_zNum = sim.domain.znum;
                const int c_yNum = v_yNum - 1;
                const int c_zNum = v_zNum - 1;
                const int v_yz = v_yNum * v_zNum;
                const int c_yz = c_yNum * c_zNum;
                const int p001_offset = 1;
                const int p010_offset = v_zNum;
                const int p011_offset = v_zNum + 1;
                const int p100_offset = v_yz;
                const int p101_offset = v_yz + 1;
                const int p110_offset = v_yz + v_zNum;
                const int p111_offset = v_yz + v_zNum + 1;

                // Find how many cells to add
                const bool_deviceView& isLiq_cur = thermal.isLiq_cur;
                Kokkos::parallel_for(
                    "interface_hook_srdf_cell_counts",
                    Kokkos::RangePolicy<device_exe>(0, c_pnum),
                    KOKKOS_LAMBDA(const int c) {
                        const int ci = c / c_yz;
                        const int cj = (c / c_zNum) % c_yNum;
                        const int ck = c % c_zNum;

                        const int p000 = ci * v_yz + cj * v_zNum + ck;
                        const int p001 = p000 + p001_offset;
                        const int p010 = p000 + p010_offset;
                        const int p011 = p000 + p011_offset;
                        const int p100 = p000 + p100_offset;
                        const int p101 = p000 + p101_offset;
                        const int p110 = p000 + p110_offset;
                        const int p111 = p000 + p111_offset;

                        const int c_sum_cur =
                            (isLiq_cur(p000) ? 1 : 0) +
                            (isLiq_cur(p001) ? 1 : 0) +
                            (isLiq_cur(p010) ? 1 : 0) +
                            (isLiq_cur(p011) ? 1 : 0) +
                            (isLiq_cur(p100) ? 1 : 0) +
                            (isLiq_cur(p101) ? 1 : 0) +
                            (isLiq_cur(p110) ? 1 : 0) +
                            (isLiq_cur(p111) ? 1 : 0);

                        c_cur(c) = static_cast<uint8_t>(c_sum_cur);
                        const int c_sum_prev = c_prev(c);
                        if ((c_sum_cur != c_sum_prev) || (c_sum_prev % 8) || (c_sum_cur % 8)) {
                            const int pos = Kokkos::atomic_fetch_add(&cell_count_local(0), 1);
                            cells_to_add_local(pos) = c;
                        }
                    }
                );
                Kokkos::deep_copy(cell_count_h, cell_count_d);
                const int numToAdd = cell_count_h(0);

                // If there is nothing to add, stop here
                if (!numToAdd) {
                    last_itert = itert;
                    Kokkos::Profiling::popRegion();
                    return;
                }

                // Make sure the SRDF container is big enough
                EnsureCapacity(numToAdd);
    
                // Counts which vertices need further calculation
                Kokkos::deep_copy(calc_sizes, 0);

                const int_deviceView& T_iter_prev = thermal.T_calc_iter_prev;
                const int_deviceView& T_iter_cur = thermal.T_calc_iter_cur;
                int_deviceView vertex_prev_stamp_local = vertex_prev_stamp;
                int_deviceView vertex_cur_stamp_local = vertex_cur_stamp;
                int_deviceView calc_prev_vertex_local = calc_prev_vertex;
                int_deviceView calc_cur_vertex_local = calc_cur_vertex;
                Kokkos::View<int[2], layout, device_memory> calc_sizes_local = calc_sizes;
                const int iter_prev = step.iter_prev;
                const int iter_cur = step.iter_cur;
                Kokkos::parallel_for(
                    "interface_hook_srdf_queue_vertices",
                    Kokkos::RangePolicy<device_exe>(0, numToAdd),
                    KOKKOS_LAMBDA(const int q) {
                        const int c = cells_to_add_local(q);
                        const int ci = c / c_yz;
                        const int cj = (c / c_zNum) % c_yNum;
                        const int ck = c % c_zNum;

                        const int p000 = ci * v_yz + cj * v_zNum + ck;
                        const int vertices[8] = {
                            p000,
                            p000 + p001_offset,
                            p000 + p010_offset,
                            p000 + p011_offset,
                            p000 + p100_offset,
                            p000 + p101_offset,
                            p000 + p110_offset,
                            p000 + p111_offset
                        };

                        for (int n = 0; n < 8; n++) {
                            const int vp = vertices[n];

                            if (T_iter_prev(vp) != iter_prev) {
                                int old_stamp = Kokkos::atomic_fetch_add(&vertex_prev_stamp_local(vp), 0);
                                while (old_stamp != iter_cur) {
                                    const int observed = Kokkos::atomic_compare_exchange(
                                        &vertex_prev_stamp_local(vp), old_stamp, iter_cur);
                                    if (observed == old_stamp) {
                                        const int pos = Kokkos::atomic_fetch_add(&calc_sizes_local(0), 1);
                                        calc_prev_vertex_local(pos) = vp;
                                        break;
                                    }
                                    old_stamp = observed;
                                }
                            }

                            if (T_iter_cur(vp) != iter_cur) {
                                int old_stamp = Kokkos::atomic_fetch_add(&vertex_cur_stamp_local(vp), 0);
                                while (old_stamp != iter_cur) {
                                    const int observed = Kokkos::atomic_compare_exchange(
                                        &vertex_cur_stamp_local(vp), old_stamp, iter_cur);
                                    if (observed == old_stamp) {
                                        const int pos = Kokkos::atomic_fetch_add(&calc_sizes_local(1), 1);
                                        calc_cur_vertex_local(pos) = vp;
                                        break;
                                    }
                                    old_stamp = observed;
                                }
                            }
                        }
                    }
                );
                
                Kokkos::deep_copy(calc_sizes_h, calc_sizes);
                const int calc_prev_count = calc_sizes_h(0);
                const int calc_cur_count = calc_sizes_h(1);

                // Get more necessary references
                floating_deviceView T_prev = thermal.T_prev;
                floating_deviceView T_cur = thermal.T_cur;
                const Grid::Device_Points<FloatType>& grid = state.grid;

                if (calc_prev_count || calc_cur_count) {
                    const Calc::Device_Constants<FloatType>& const_d_local = *const_d;
                    const Calc::Device_Nodes<FloatType>& nodes_cur_d = state.quad.CurrentSlot().device_nodes;
                    const Calc::Device_Nodes<FloatType>& nodes_prev_d = calc_prev_count ? state.quad.PreviousSlot().device_nodes : nodes_cur_d;
                    const int calc_count = std::max(calc_prev_count, calc_cur_count);
                    Kokkos::parallel_for(
                        "interface_hook_srdf_fill_temperatures",
                        team_policy(calc_count, Kokkos::AUTO),
                        KOKKOS_LAMBDA(const team_member& team) {
                            const int q = team.league_rank();
                            if (q < calc_prev_count) {
                                const int vp = calc_prev_vertex_local(q);
                                const FloatType T_prev_calc = Calc::T_device(grid, nodes_prev_d, const_d_local, vp, team);
                                if (team.team_rank() == 0) {
                                    T_prev(vp) = T_prev_calc;
                                    T_iter_prev(vp) = iter_prev;
                                }
                            }
                            team.team_barrier();
                            if (q < calc_cur_count) {
                                const int vp = calc_cur_vertex_local(q);
                                const FloatType T_cur_calc = Calc::T_device(grid, nodes_cur_d, const_d_local, vp, team);
                                if (team.team_rank() == 0) {
                                    T_cur(vp) = T_cur_calc;
                                    T_iter_cur(vp) = iter_cur;
                                }
                            }
                        }
                    );
                    Kokkos::fence();
                }

                // Loop over all cells and figure out what to add
                Stork::Structs::RegularGrid_Header<FloatType, device_space> header = SRDF->template get_header<device_space>();
                Stork::Structs::SRDF_Data<FloatType, device_space> data = SRDF->template get_data<device_space>();
                const uint32_t numSnaps = SRDF->numSnaps;
                Kokkos::parallel_for(
                    "interface_hook_srdf_place_data",
                    Kokkos::RangePolicy<device_exe>(0, numToAdd),
                    KOKKOS_LAMBDA(const int q) {
                        const int c = cells_to_add_local(q);
                        const int ci = c / c_yz;
                        const int cj = (c / c_zNum) % c_yNum;
                        const int ck = c % c_zNum;

                        uint32_t LOCAL_p = 0;
                        const uint32_t LOCAL_ijk[3] = {
                            static_cast<uint32_t>(ci),
                            static_cast<uint32_t>(cj),
                            static_cast<uint32_t>(ck)
                        };

                        const uint32_t idx = numSnaps + static_cast<uint32_t>(q);
                        header.LOCAL_ijk_to_LOCAL_p(LOCAL_p, LOCAL_ijk);
                        data.p(idx) = LOCAL_p;
                        data.t_prev(idx) = t_prev;
                        data.t_cur(idx) = t_cur;

                        FloatType* T_prev_data = &data.T_prev(idx);
                        FloatType* T_cur_data = &data.T_cur(idx);

                        const int p000 = ci * v_yz + cj * v_zNum + ck;
                        const int p001 = p000 + p001_offset;
                        const int p010 = p000 + p010_offset;
                        const int p011 = p000 + p011_offset;
                        const int p100 = p000 + p100_offset;
                        const int p101 = p000 + p101_offset;
                        const int p110 = p000 + p110_offset;
                        const int p111 = p000 + p111_offset;

                        T_prev_data[0] = T_prev(p000);
                        T_cur_data[0] = T_cur(p000);
                        T_prev_data[1] = T_prev(p001);
                        T_cur_data[1] = T_cur(p001);
                        T_prev_data[2] = T_prev(p010);
                        T_cur_data[2] = T_cur(p010);
                        T_prev_data[3] = T_prev(p011);
                        T_cur_data[3] = T_cur(p011);
                        T_prev_data[4] = T_prev(p100);
                        T_cur_data[4] = T_cur(p100);
                        T_prev_data[5] = T_prev(p101);
                        T_cur_data[5] = T_cur(p101);
                        T_prev_data[6] = T_prev(p110);
                        T_cur_data[6] = T_cur(p110);
                        T_prev_data[7] = T_prev(p111);
                        T_cur_data[7] = T_cur(p111);
                    }
                );
                Kokkos::fence();

                // Update count and iteration
                SRDF->numSnaps += static_cast<uint32_t>(numToAdd);
                last_itert = itert;

                Kokkos::Profiling::popRegion();
            }

            void Finalize(
                const Meltpool::Tracking<FloatType>& meltpool,
                const Simdat<FloatType>& sim,
                const Step_State<FloatType>& step)
            {
                if (shouldSkip) { return; }
                Kokkos::Profiling::pushRegion("Interface::Hooks::SRDF::Finalize");
                SyncSpatialCellOverlap(sim);
                Kokkos::Profiling::popRegion();
            }
    };
}
