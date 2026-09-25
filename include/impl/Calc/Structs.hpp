#pragma once

// Include Definitions
#include "Definitions.hpp"

// Include Necessary Structs
#include "impl/Structs/Simdat.hpp"
#include <utility>
#include <algorithm>
#include <cstring>

namespace Condor::impl
{
    namespace FloatConstants {
        inline constexpr int T_INIT = 0;
        inline constexpr int T_LIQ = 1;
        inline constexpr int DT = 2;
    }

    namespace IntConstants {
        inline constexpr int X_NUM = 0;
        inline constexpr int Y_NUM = 1;
        inline constexpr int Z_NUM = 2;
        inline constexpr int P_NUM = 3;
    }

    // Structure for integration segment
    template<typename FloatType>
    struct int_seg {
        FloatType xb, yb, zb, phix, phiy, phiz, dtau, qmod;
    };

    // Structure to store quadrature nodes
    template<typename FloatType>
    struct Nodes {
        size_t size = 0;
        std::vector<FloatType> xb, yb, zb, phix, phiy, phiz, dtau, expmod;
    };

    // Entire quadrature data structure
    template<typename FloatType>
    struct QuadDat{
        Nodes<FloatType> nodes_alpha;
        Nodes<FloatType> nodes_beta;
        std::vector<Nodes<FloatType>> par_nodes;
        std::vector<int> start_seg;
    };

    namespace Calc
    {

        // Constants for use on the device
        template<typename FloatType>
        struct Device_Constants {
            // Typedefs
            typedef Kokkos::View<FloatType[3], layout, host_memory> hostFloats;
            typedef Kokkos::View<FloatType[3], layout, device_memory> deviceFloats;
            typedef Kokkos::View<int[4], layout, host_memory> hostInts;
            typedef Kokkos::View<int[4], layout, device_memory> deviceInts;

            // Initialize the device constants from raw simulation inputs plus the mode-owned timestep.
            Device_Constants(const Simdat<FloatType>& sim, const FloatType timestep = static_cast<FloatType>(-1.0)){
                hostFloats floats_h(Kokkos::ViewAllocateWithoutInitializing("DeviceConstants_hostFloats_temp"));
                hostInts ints_h(Kokkos::ViewAllocateWithoutInitializing("DeviceConstants_hostInts_temp"));

                // Seed the material and timestep constants used by device-side thermal kernels.
                floats_h(FloatConstants::T_INIT) = sim.material.T_init;
                floats_h(FloatConstants::T_LIQ) = sim.material.T_liq;
                floats_h(FloatConstants::DT) = timestep;

                // Seed the grid dimensions used to decode flattened point indices on device.
                ints_h(IntConstants::X_NUM) = sim.domain.xnum;
                ints_h(IntConstants::Y_NUM) = sim.domain.ynum;
                ints_h(IntConstants::Z_NUM) = sim.domain.znum;
                ints_h(IntConstants::P_NUM) = sim.domain.pnum;

                // Mirror the populated host constants onto the device-side views once per mode run.
                Kokkos::deep_copy(floats, floats_h);
                Kokkos::deep_copy(ints, ints_h);
            }
            deviceFloats floats = deviceFloats(Kokkos::ViewAllocateWithoutInitializing("const_d.floats")); 
            deviceInts ints = deviceInts(Kokkos::ViewAllocateWithoutInitializing("const_d.ints")); 
        };

        // Quadrature Nodes
        template<typename FloatType>
        class Device_Nodes {
        private:
            // Typedefs
            typedef Kokkos::View<const FloatType*, layout, host_memory, Kokkos::MemoryTraits<Kokkos::Unmanaged>> floatType_hostView;
            typedef Kokkos::View<FloatType*, layout, device_memory> floatType_deviceView;
        public:
            int size = 0;
            int capacity = 0;
            floatType_deviceView x_b, y_b, z_b, phi_x, phi_y, phi_z, dtau, expmod;

            // Resizing views
            inline void resize_views(int new_capacity) {
                capacity = new_capacity;

                x_b = floatType_deviceView(Kokkos::ViewAllocateWithoutInitializing("nodes.x_b"), capacity);
                y_b = floatType_deviceView(Kokkos::ViewAllocateWithoutInitializing("nodes.y_b"), capacity);
                z_b = floatType_deviceView(Kokkos::ViewAllocateWithoutInitializing("nodes.z_b"), capacity);
                phi_x = floatType_deviceView(Kokkos::ViewAllocateWithoutInitializing("nodes.phix"), capacity);
                phi_y = floatType_deviceView(Kokkos::ViewAllocateWithoutInitializing("nodes.phiy"), capacity);
                phi_z = floatType_deviceView(Kokkos::ViewAllocateWithoutInitializing("nodes.phiz"), capacity);
                dtau = floatType_deviceView(Kokkos::ViewAllocateWithoutInitializing("nodes.dtau"), capacity);
                expmod = floatType_deviceView(Kokkos::ViewAllocateWithoutInitializing("nodes.expmod"), capacity);
            }

            // Initializing
            Device_Nodes() {
                capacity = 1000;
                resize_views(capacity);
            }

            // Copy one host-side node pack directly onto the device.
            void CopyToDevice(const Nodes<FloatType>& nodes, device_exe* exec = nullptr) {
                // Capture the active node count for this timestep.
                const size_t nodesSize = nodes.size; // Number of quadrature nodes that must be transferred.
                size = static_cast<int>(nodesSize);

                // Exit early when the timestep has no active nodes.
                if (!nodesSize) {
                    return;
                }

                // Grow the reusable staging and device buffers only when this timestep needs more space.
                if (nodesSize > static_cast<size_t>(capacity)) {
                    resize_views(std::max(static_cast<int>(nodesSize * 2), 1));
                }

                // Create unmanaged host views directly over the node storage.
                floatType_hostView x_host(nodes.xb.data(), nodesSize);
                floatType_hostView y_host(nodes.yb.data(), nodesSize);
                floatType_hostView z_host(nodes.zb.data(), nodesSize);
                floatType_hostView phix_host(nodes.phix.data(), nodesSize);
                floatType_hostView phiy_host(nodes.phiy.data(), nodesSize);
                floatType_hostView phiz_host(nodes.phiz.data(), nodesSize);
                floatType_hostView dtau_host(nodes.dtau.data(), nodesSize);
                floatType_hostView expmod_host(nodes.expmod.data(), nodesSize);

                // Restrict the H2D copies to the active node prefix so oversized buffers stay reusable.
                const std::pair<size_t, size_t> subRange = std::make_pair(static_cast<size_t>(0), nodesSize); // Active node range.
                auto x_b_subview = Kokkos::subview(x_b, subRange); // Active device x coordinates.
                auto y_b_subview = Kokkos::subview(y_b, subRange); // Active device y coordinates.
                auto z_b_subview = Kokkos::subview(z_b, subRange); // Active device z coordinates.
                auto phi_x_subview = Kokkos::subview(phi_x, subRange); // Active device x Gaussian factors.
                auto phi_y_subview = Kokkos::subview(phi_y, subRange); // Active device y Gaussian factors.
                auto phi_z_subview = Kokkos::subview(phi_z, subRange); // Active device z Gaussian factors.
                auto dtau_subview = Kokkos::subview(dtau, subRange); // Active device quadrature weights.
                auto expmod_subview = Kokkos::subview(expmod, subRange); // Active device exponential modifiers.

                // Queue the copies on the requested execution space when overlap is possible.
                if (exec) {
                    Kokkos::deep_copy(*exec, x_b_subview, x_host);
                    Kokkos::deep_copy(*exec, y_b_subview, y_host);
                    Kokkos::deep_copy(*exec, z_b_subview, z_host);
                    Kokkos::deep_copy(*exec, phi_x_subview, phix_host);
                    Kokkos::deep_copy(*exec, phi_y_subview, phiy_host);
                    Kokkos::deep_copy(*exec, phi_z_subview, phiz_host);
                    Kokkos::deep_copy(*exec, dtau_subview, dtau_host);
                    Kokkos::deep_copy(*exec, expmod_subview, expmod_host);
                }
                else {
                    Kokkos::deep_copy(x_b_subview, x_host);
                    Kokkos::deep_copy(y_b_subview, y_host);
                    Kokkos::deep_copy(z_b_subview, z_host);
                    Kokkos::deep_copy(phi_x_subview, phix_host);
                    Kokkos::deep_copy(phi_y_subview, phiy_host);
                    Kokkos::deep_copy(phi_z_subview, phiz_host);
                    Kokkos::deep_copy(dtau_subview, dtau_host);
                    Kokkos::deep_copy(expmod_subview, expmod_host);
                }
            }
        };

        namespace Constants { // Assuming you have this namespace
            KOKKOS_INLINE_FUNCTION constexpr int alpha()   { return 0; }
            KOKKOS_INLINE_FUNCTION constexpr int beta()    { return 1; }
            KOKKOS_INLINE_FUNCTION constexpr int nond_dt() { return 2; }
            KOKKOS_INLINE_FUNCTION constexpr int bx2()     { return 3; }
            KOKKOS_INLINE_FUNCTION constexpr int by2()     { return 4; }
            KOKKOS_INLINE_FUNCTION constexpr int bz2()     { return 5; }
            KOKKOS_INLINE_FUNCTION constexpr int NUM_CONSTANTS() {return 6;}
        }        

        template <typename FloatType>
        struct QuadParams {
            using DeviceType = Kokkos::DefaultExecutionSpace; // Or your specific device

            // Members (using Views for device accessibility)
            Kokkos::View<int*, layout, device_space> non_2_nodes_offset;
            Kokkos::View<path_seg<FloatType>*, layout, device_space> path;
            Kokkos::View<int, layout, host_space> last_2_host;
            Kokkos::View<int, layout, device_space> last_2_device;
            Kokkos::View<FloatType[30], layout, device_space> locs;
            Kokkos::View<FloatType[30], layout, device_space> weights;
            Kokkos::View<int[4], layout, device_space> ORDER_OFFSETS;
            Kokkos::View<int[4], layout, device_space> ORDER_COUNTS;
            Kokkos::View<FloatType[Constants::NUM_CONSTANTS()], device_space> consts;
            int numSegs;

            // Constructor (performs initialization)
            QuadParams(const Simdat<FloatType>& sim) :
                non_2_nodes_offset(Kokkos::ViewAllocateWithoutInitializing("non_2_nodes_offset"),1), // Initialize views
                path(Kokkos::ViewAllocateWithoutInitializing("path"),1),
                last_2_host(Kokkos::ViewAllocateWithoutInitializing("last_2_host")),
                last_2_device(Kokkos::ViewAllocateWithoutInitializing("last_2_device")),
                locs(Kokkos::ViewAllocateWithoutInitializing("locs")),
                weights(Kokkos::ViewAllocateWithoutInitializing("weights")),
                ORDER_OFFSETS(Kokkos::ViewAllocateWithoutInitializing("ORDER_OFFSETS")),
                ORDER_COUNTS(Kokkos::ViewAllocateWithoutInitializing("ORDER_COUNTS")),
                consts(Kokkos::ViewAllocateWithoutInitializing("consts")),
                numSegs(0)
                {
                    initialize(sim); // Call initialization
                }

            // Initialization function (separate, for clarity)
            void initialize(const Simdat<FloatType>& sim) 
            {
                
                // Initialize last ones
                last_2_host() = 0;
                Kokkos::deep_copy(last_2_device, last_2_host);

                // Get beam and path stuff 
                const Beam<FloatType>& beam_temp = sim.beams[0]; // Use beamIndex
                const std::vector<path_seg<FloatType>>& path_temp = sim.paths[0]; // Use beamIndex
                numSegs = path_temp.size();

                // Initialize possible
                non_2_nodes_offset = Kokkos::View<int*, Kokkos::LayoutRight, DeviceType>("non_2_nodes_offset", numSegs);
                // Initialize path view
                path = Kokkos::View<path_seg<FloatType>*, Kokkos::LayoutRight, DeviceType>("path", numSegs);
                Kokkos::View<path_seg<FloatType>*, layout, host_memory> h_path = Kokkos::create_mirror_view(path);
                for (int i = 0; i < numSegs; ++i) {
                    h_path(i) = path_temp[i];
                }
                Kokkos::deep_copy(path, h_path);

                // --- Initialize locs, weights, etc. ---
                {  //Use block scope
                    Kokkos::View<FloatType[30], layout, host_memory> h_locs = Kokkos::create_mirror_view(locs);
                    h_locs[0] = -0.5773502691896257;  h_locs[1] = 0.5773502691896257;
                    h_locs[2] = -0.8611363115940526; h_locs[3] = -0.3399810435848563;  h_locs[4] = 0.3399810435848563;  h_locs[5] = 0.8611363115940526;
                    h_locs[6] = -0.9602898564975363; h_locs[7] = -0.7966664774136267; h_locs[8] = -0.5255324099163290; h_locs[9] = -0.1834346424956498;  h_locs[10] = 0.1834346424956498;  h_locs[11] = 0.5255324099163290; h_locs[12] = 0.7966664774136267;  h_locs[13] = 0.9602898564975363;
                    h_locs[14] = -0.9894009349916499; h_locs[15] = -0.9445750230732326; h_locs[16] = -0.8656312023878318; h_locs[17] = -0.7554044083550030; h_locs[18] = -0.6178762444026438; h_locs[19] = -0.4580167776572274; h_locs[20] = -0.2816035507792589; h_locs[21] = -0.0950125098376374; h_locs[22] = 0.0950125098376374; h_locs[23] = 0.2816035507792589; h_locs[24] = 0.4580167776572274; h_locs[25] = 0.6178762444026438; h_locs[26] = 0.7554044083550030; h_locs[27] = 0.8656312023878318; h_locs[28] = 0.9445750230732326; h_locs[29] = 0.9894009349916499;
                    Kokkos::deep_copy(locs, h_locs);

                    Kokkos::View<FloatType[30], layout, host_memory> h_weights = Kokkos::create_mirror_view(weights);
                    h_weights[0] = 1.0000000000000000; h_weights[1] = 1.0000000000000000;
                    h_weights[2] = 0.3478548451374538; h_weights[3] = 0.6521451548625461; h_weights[4] = 0.6521451548625461; h_weights[5] = 0.3478548451374538;
                    h_weights[6] = 0.1012285362903763; h_weights[7] = 0.2223810344533745; h_weights[8] = 0.3137066458778873; h_weights[9] = 0.3626837833783620; h_weights[10] = 0.3626837833783620; h_weights[11] = 0.3137066458778873; h_weights[12] = 0.2223810344533745; h_weights[13] = 0.1012285362903763;
                    h_weights[14] = 0.0271524594117541; h_weights[15] = 0.0622535239386479; h_weights[16] = 0.0951585116824928; h_weights[17] = 0.1246289712555339; h_weights[18] = 0.1495959888165767; h_weights[19] = 0.1691565193950025; h_weights[20] = 0.1826034150449236; h_weights[21] = 0.1894506104550685; h_weights[22] = 0.1894506104550685; h_weights[23] = 0.1826034150449236; h_weights[24] = 0.1691565193950025; h_weights[25] = 0.1495959888165767; h_weights[26] = 0.1246289712555339; h_weights[27] = 0.0951585116824928; h_weights[28] = 0.0622535239386479; h_weights[29] = 0.0271524594117541;
                    Kokkos::deep_copy(weights, h_weights);

                    Kokkos::View<int[4], layout, host_memory> h_ORDER_OFFSETS = Kokkos::create_mirror_view(ORDER_OFFSETS);
                    h_ORDER_OFFSETS[0] = 0; h_ORDER_OFFSETS[1] = 2; h_ORDER_OFFSETS[2] = 6; h_ORDER_OFFSETS[3] = 14;
                    Kokkos::deep_copy(ORDER_OFFSETS, h_ORDER_OFFSETS);

                    Kokkos::View<int[4], layout, host_memory> h_ORDER_COUNTS = Kokkos::create_mirror_view(ORDER_COUNTS);
                    h_ORDER_COUNTS[0] = 2; h_ORDER_COUNTS[1] = 4; h_ORDER_COUNTS[2] = 8; h_ORDER_COUNTS[3] = 16;
                    Kokkos::deep_copy(ORDER_COUNTS, h_ORDER_COUNTS);

                    Kokkos::View<FloatType[Constants::NUM_CONSTANTS()], layout, host_memory> h_consts = Kokkos::create_mirror_view(consts);
                    h_consts(Constants::alpha())   = sim.material.a;
                    h_consts(Constants::beta())    = 0.933162059717596 * beam_temp.q / (sim.material.rho * sim.material.cps);
                    h_consts(Constants::nond_dt()) = beam_temp.nond_dt;
                    h_consts(Constants::bx2())     = beam_temp.ax * beam_temp.ax;
                    h_consts(Constants::by2())     = beam_temp.ay * beam_temp.ay;
                    h_consts(Constants::bz2())     = beam_temp.az * beam_temp.az;
                    Kokkos::deep_copy(consts, h_consts);
                }
            }
        }; // struct QuadParams

    }
}
