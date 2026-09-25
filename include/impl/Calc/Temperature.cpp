// Include where functions are defined
#include "impl/Calc/Temperature.hpp"

#include <cmath>

namespace Condor::impl{
    namespace Calc{

        // Evaluate one host-side point temperature from the prepared quadrature nodes.
        template<typename FloatType>
        FloatType T_host(const Nodes<FloatType>& nodes, const Simdat<FloatType>& sim, const int p) {
            /**************************************************************************************************
            Heat transfer kernel for ellipsoidal Gaussian volumentric heat source
            Adapted from Nguyen et al., Welding Journal, 1999 (Eq. 7)
            **************************************************************************************************/

            // Read the spatial location of the point being reconstructed.
            const int i = Grid::p_to_i(p, sim.domain.ynum, sim.domain.znum);
            const int j = Grid::p_to_j(p, sim.domain.ynum, sim.domain.znum);
            const int k = Grid::p_to_k(p, sim.domain.znum);
            const FloatType x0 = (sim.domain.xnum == 1) ? sim.domain.xmax : sim.domain.xmin;
            const FloatType y0 = (sim.domain.ynum == 1) ? sim.domain.ymax : sim.domain.ymin;
            const FloatType z0 = (sim.domain.znum == 1) ? sim.domain.zmax : sim.domain.zmin;
            const FloatType xp = x0 + static_cast<FloatType>(i) * sim.domain.xres; // Point x-coordinate.
            const FloatType yp = y0 + static_cast<FloatType>(j) * sim.domain.yres; // Point y-coordinate.
            const FloatType zp = z0 + static_cast<FloatType>(k) * sim.domain.zres; // Point z-coordinate.

            // Accumulate the temperature rise contributed by every quadrature node.
            FloatType dT = static_cast<FloatType>(0.0); // Integrated temperature rise above the initial temperature.
            const size_t numNodes = nodes.size; // Number of quadrature nodes active at this time.
            for (size_t node = 0; node < numNodes; ++node) {

                // Measure the offset from the quadrature node to the grid point.
                const FloatType dx = xp - nodes.xb[node]; // Point-to-node offset in x.
                const FloatType dy = yp - nodes.yb[node]; // Point-to-node offset in y.
                const FloatType dz = zp - nodes.zb[node]; // Point-to-node offset in z.

                // Read the anisotropic Gaussian coefficients for this node.
                const FloatType phix = nodes.phix[node]; // Gaussian shape factor in x.
                const FloatType phiy = nodes.phiy[node]; // Gaussian shape factor in y.
                const FloatType phiz = nodes.phiz[node]; // Gaussian shape factor in z.
                const FloatType expmod = nodes.expmod[node]; // Precomputed exponential modifier for this node.
                const FloatType dtau = nodes.dtau[node]; // Time-weighted quadrature weight for this node.

                // Evaluate the node contribution with the natural exponential kernel.
                const FloatType phi = std::exp(-static_cast<FloatType>(3.0)*((dx * dx * phix) + (dy * dy * phiy) + (dz * dz * phiz)) + expmod);                
                const FloatType dT_seg = dtau * phi; // Temperature rise from this quadrature node.
                dT += dT_seg;
            }

            return sim.material.T_init + dT;
        }

        // --- Explicit Instantiations (within the namespace, after definitions) ---
        template float T_host(const Nodes<float>&, const Simdat<float>&, const int);
        template double T_host(const Nodes<double>&, const Simdat<double>&, const int);
    }
}
