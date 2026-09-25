#pragma once

// Include Definitions
#include "Definitions.hpp"
#include "impl/Calc/Structs.hpp"
#include "impl/Structs/Simdat.hpp"
#include "impl/Utility/Grid.hpp"
#include "impl/Utility/Nodes.hpp"

namespace Condor::impl{
    namespace Calc{

        // Evaluate one host-side point temperature from the prepared quadrature nodes.
        template<typename FloatType>
        FloatType T_host(const Nodes<FloatType>&, const Simdat<FloatType>&, const int);
        
        // Temperature Calculation on Device
        template<typename FloatType>
        KOKKOS_INLINE_FUNCTION 
        FloatType T_device(
            const Grid::Device_Points<FloatType>& grid,
            const Device_Nodes<FloatType>& nodes_d,
            const Device_Constants<FloatType>& const_d,
            const int p,
            const typename Kokkos::TeamPolicy<device_exe>::member_type& team)
        {        

            // Initialize temperature
            FloatType dT = 0.0; 

            // Read point coordinates from the precomputed device grid.
            const FloatType xp = grid.x_p(p); // Point x coordinate.
            const FloatType yp = grid.y_p(p); // Point y coordinate.
            const FloatType zp = grid.z_p(p); // Point z coordinate.
            
            // Reduce over quadrature nodes
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team, nodes_d.size),
                [=](int& node, FloatType& dT_th){
                    const FloatType dx = xp - nodes_d.x_b(node);
                    const FloatType dy = yp - nodes_d.y_b(node);
                    const FloatType dz = zp - nodes_d.z_b(node);

                    const FloatType phix = nodes_d.phi_x(node);
                    const FloatType phiy = nodes_d.phi_y(node);
                    const FloatType phiz = nodes_d.phi_z(node);

                    const FloatType expmod = nodes_d.expmod(node);

                    const FloatType dtau = nodes_d.dtau(node);

                    const FloatType phi = Kokkos::exp(-static_cast<FloatType>(3.0)*((dx * dx * phix) + (dy * dy * phiy) + (dz * dz * phiz)) + expmod);

                    const FloatType dT_seg = dtau * phi;

                    dT_th += dT_seg;
                }
            ,dT);
            
            return const_d.floats(FloatConstants::T_INIT) + dT;
        }
    }
}
