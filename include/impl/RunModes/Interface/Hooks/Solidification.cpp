#include "impl/RunModes/Interface/Hooks/Solidification.hpp"

#include "impl/Calc/Quad.hpp"
#include "impl/Calc/Temperature.hpp"
#include "impl/Utility/Grid.hpp"
#include "impl/Utility/Nodes.hpp"
#include "impl/Utility/Util.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace Condor::impl::Modes::Interface::Hooks {

    namespace {

        // For where to start the search each time
        template<typename FloatType>
        std::vector<int> BuildSolidificationSegments(const std::vector<int>& start_seg, const Simdat<FloatType>& sim) {
            std::vector<int> sol_start_seg = start_seg; // Thread-local segment tracker for this refinement.
            if (sol_start_seg.size() != sim.paths.size()) {
                sol_start_seg.assign(sim.paths.size(), 1);
            }
            return sol_start_seg;
        }

        template<typename FloatType>
        FloatType PointX(const Simdat<FloatType>& sim, const int p) {
            const int i = Grid::p_to_i(p, sim.domain.ynum, sim.domain.znum);
            const FloatType x0 = (sim.domain.xnum == 1) ? sim.domain.xmax : sim.domain.xmin;
            const FloatType dx = (sim.domain.xnum == 1) ? static_cast<FloatType>(0.0) : sim.domain.xres;
            return x0 + static_cast<FloatType>(i) * dx;
        }

        template<typename FloatType>
        FloatType PointY(const Simdat<FloatType>& sim, const int p) {
            const int j = Grid::p_to_j(p, sim.domain.ynum, sim.domain.znum);
            const FloatType y0 = (sim.domain.ynum == 1) ? sim.domain.ymax : sim.domain.ymin;
            const FloatType dy = (sim.domain.ynum == 1) ? static_cast<FloatType>(0.0) : sim.domain.yres;
            return y0 + static_cast<FloatType>(j) * dy;
        }

        template<typename FloatType>
        FloatType PointZ(const Simdat<FloatType>& sim, const int p) {
            const int k = Grid::p_to_k(p, sim.domain.znum);
            const FloatType z0 = (sim.domain.znum == 1) ? sim.domain.zmax : sim.domain.zmin;
            const FloatType dz = (sim.domain.znum == 1) ? static_cast<FloatType>(0.0) : sim.domain.zres;
            return z0 + static_cast<FloatType>(k) * dz;
        }

        template<typename FloatType>
        void StoreSolidificationOutputs(
            const Solidification_Fields& fields,
            Solidification_Views<FloatType>& views,
            const FloatType t_sol_exact,
            const FloatType num_melt,
            const SolidificationOutputValues<FloatType>& outputs,
            const int p)
        {
            if (fields.tSol) { views.tSol(p) = t_sol_exact; }
            if (fields.G) { views.G(p) = outputs.G; }
            if (fields.V) { views.V(p) = outputs.V; }
            if (fields.dTdt) { views.dTdt(p) = outputs.dTdt; }
            if (fields.numMelt) { views.numMelt(p) = num_melt; }
            if (fields.G_vec) {
                views.Gx(p) = outputs.Gx;
                views.Gy(p) = outputs.Gy;
                views.Gz(p) = outputs.Gz;
            }
            if (fields.G_unit_vec) {
                views.G_unit_x(p) = outputs.G_unit_x;
                views.G_unit_y(p) = outputs.G_unit_y;
                views.G_unit_z(p) = outputs.G_unit_z;
            }
        }
    }

    template<typename FloatType>
    FloatType FindSolidificationTime(
        const std::vector<int>& start_seg,
        const Simdat<FloatType>& sim,
        const Solidification_Parameters<FloatType>& params,
        const int p,
        const SolidificationBracket<FloatType>& bracket)
    {
        // Seed a local segment tracker so refinement stays thread-safe while preserving fast segment lookup.
        std::vector<int> sol_start_seg = BuildSolidificationSegments(start_seg, sim); // Thread-local segment tracker.

        // Start the root search from the stored liquid/solid bracket for this point.
        const int maxIter = std::max(params.max_iter, 1); // Maximum number of refinement iterations.
        const FloatType T_target = sim.material.T_liq; // Target temperature being solved for.
        FloatType T_cold = bracket.T_cold; // Temperature on the solid side of the bracket.
        FloatType t_cold = bracket.t_cold; // Time on the solid side of the bracket.
        FloatType T_hot = bracket.T_hot; // Temperature on the liquid side of the bracket.
        FloatType t_hot = bracket.t_hot; // Time on the liquid side of the bracket.
        FloatType t_mid = t_cold; // Current candidate solidification time.
        FloatType T_mid = T_cold; // Temperature evaluated at the current candidate time.
        const FloatType err_limit = params.dT_err; // Relative liquidus error tolerance.
        const FloatType slope_limit = std::numeric_limits<FloatType>::epsilon(); // Near-zero slope guard.

        // Handle exact hits and malformed brackets before iterating.
        if (T_hot == T_target) { return t_hot; }
        if (T_cold == T_target) { return t_cold; }
        if (t_hot < static_cast<FloatType>(0.0)) { t_hot = static_cast<FloatType>(0.0); }

        // Reject invalid brackets before any iterative refinement work.
        const FloatType dt_bracket = t_cold - t_hot; // Width of the current time bracket.
        const FloatType dT_bracket = T_cold - T_hot; // Temperature span across the current bracket.
        if ((dt_bracket <= static_cast<FloatType>(0.0)) || (dT_bracket == static_cast<FloatType>(0.0))) {
            return t_cold;
        }

        // Fall back to the stored linear estimate if the saved temperatures do not bracket the target.
        if ((T_hot < T_target) || (T_cold > T_target)) {
            t_mid = t_hot + (T_target - T_hot) * dt_bracket / dT_bracket;
            if (t_mid < t_hot) { t_mid = t_hot; }
            if (t_mid > t_cold) { t_mid = t_cold; }
            return t_mid;
        }

        // Reuse one node container while refining the crossing time.
        Nodes<FloatType> nodes; // Quadrature nodes rebuilt at each candidate time.
        for (int runIter = 0; runIter < maxIter; runIter++) {

            // Predict the crossing time from the current bracket.
            const FloatType m = (T_cold - T_hot) / (t_cold - t_hot); // Current secant slope of the bracket.
            if ((m < slope_limit) && (m > -slope_limit)) {
                t_mid = static_cast<FloatType>(0.5) * (t_hot + t_cold);
            }
            else {
                t_mid = t_hot + ((T_target - T_hot) / m);
            }
            if ((t_mid <= t_hot) || (t_mid >= t_cold)) {
                t_mid = static_cast<FloatType>(0.5) * (t_hot + t_cold);
            }

            // Re-evaluate the temperature at the candidate crossing time.
            Util::ClearNodes(nodes);
            Calc::Integrate_Serial(nodes, sol_start_seg, sim, t_mid, false);
            T_mid = Calc::T_host(nodes, sim, p);

            // Stop once the candidate is close enough to the target temperature.
            const FloatType T_err = std::abs(static_cast<FloatType>(1.0) - T_mid / T_target); // Relative target-temperature error.
            if (T_err < err_limit) {
                return t_mid;
            }

            // Replace whichever side of the bracket shares the candidate temperature state.
            if (T_mid > T_target) {
                t_hot = t_mid;
                T_hot = T_mid;
            }
            else if (T_mid < T_target) {
                t_cold = t_mid;
                T_cold = T_mid;
            }
            else {
                return t_mid;
            }
        }

        // Return the last refined estimate if the iteration cap is reached.
        return t_mid;
    }

    template<typename FloatType>
    SolidificationPrimaryValues<FloatType> EvaluateSolidificationPrimary(const Simdat<FloatType>& sim, const Nodes<FloatType>& nodes, const int p) {
        /**************************************************************************************************
        Heat transfer kernel for ellipsoidal Gaussian volumentric heat source
        Adapted from Nguyen et al., Welding Journal, 1999 (Eq. 7)
        **************************************************************************************************/

        // Read the spatial location of the grid point being reconstructed.
        const FloatType xp = PointX(sim, p); // Point x-coordinate.
        const FloatType yp = PointY(sim, p); // Point y-coordinate.
        const FloatType zp = PointZ(sim, p); // Point z-coordinate.

        // Accumulate the gradient and transient terms over the prepared quadrature nodes.
        SolidificationPrimaryValues<FloatType> primary; // Raw solidification terms at the refined crossing time.
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

            // Evaluate this node's temperature contribution and its spatial derivatives.
            const FloatType phi = std::exp(-static_cast<FloatType>(3.0) * ((dx * dx * phix) + (dy * dy * phiy) + (dz * dz * phiz)) + expmod);
            const FloatType dT_seg = dtau * phi; // Temperature rise from this quadrature node.
            // For exp(-3 * dx_i^2 * phi_i), the first and second derivatives of
            // the logarithm of the spatial kernel are -6 * dx_i * phi_i and -6 * phi_i.
            const FloatType dpx = -static_cast<FloatType>(6.0) * dx * phix; // Logarithmic first derivative in x.
            const FloatType dpy = -static_cast<FloatType>(6.0) * dy * phiy; // Logarithmic first derivative in y.
            const FloatType dpz = -static_cast<FloatType>(6.0) * dz * phiz; // Logarithmic first derivative in z.
            const FloatType ddpx = -static_cast<FloatType>(6.0) * phix; // Logarithmic second derivative in x.
            const FloatType ddpy = -static_cast<FloatType>(6.0) * phiy; // Logarithmic second derivative in y.
            const FloatType ddpz = -static_cast<FloatType>(6.0) * phiz; // Logarithmic second derivative in z.

            // Accumulate the gradient, Laplacian, and transient source terms.
            primary.Gx += dT_seg * dpx;
            primary.Gy += dT_seg * dpy;
            primary.Gz += dT_seg * dpz;
            primary.Laplace += dT_seg * (dpx * dpx + dpy * dpy + dpz * dpz + ddpx + ddpy + ddpz);
            if (dtau == static_cast<FloatType>(0.0)) {
                primary.dT_t += phi;
            }
        }

        return primary;
    }

    template<typename FloatType>
    SolidificationOutputValues<FloatType> BuildSolidificationOutputs(const SolidificationPrimaryValues<FloatType>& primary, const Simdat<FloatType>& sim) {
        // Convert the raw gradient terms into the scalar quantities written to output.
        SolidificationOutputValues<FloatType> outputs; // Final host-side solidification outputs.
        outputs.Gx = primary.Gx;
        outputs.Gy = primary.Gy;
        outputs.Gz = primary.Gz;
        outputs.G = std::sqrt(primary.Gx * primary.Gx + primary.Gy * primary.Gy + primary.Gz * primary.Gz);
        outputs.dTdt = std::abs(sim.material.a * primary.Laplace + primary.dT_t);

        const FloatType eps = std::numeric_limits<FloatType>::epsilon();
        if (outputs.G > eps) {
            outputs.G_unit_x = primary.Gx / outputs.G;
            outputs.G_unit_y = primary.Gy / outputs.G;
            outputs.G_unit_z = primary.Gz / outputs.G;
            outputs.V = outputs.dTdt / outputs.G;
        }
        return outputs;
    }

    template<typename FloatType>
    void Solidification_Hook<FloatType>::RunSolidificationBatch(const Meltpool::Tracking<FloatType>& meltpool, const Simdat<FloatType>& sim, const FloatType timestep, const int eventCount) {

        // Set references
        const Solidification_Fields& fields = calc.fields;
        const Solidification_Parameters<FloatType>& params = config.params;
        const int_hostView event_p_h = meltpool.events.event_p_h;
        const floating_hostView event_tl_h = meltpool.events.event_tl_h;
        const floating_hostView event_Tliq_h = meltpool.events.event_Tliq_h;
        const floating_hostView event_Tsol_h = meltpool.events.event_Tsol_h;
        const floating_hostView event_num_melt_h = meltpool.events.event_num_melt_h;
        Solidification_Views<FloatType>& views = volume;
        floating_hostView eventTSol_local = eventTSol;
        floating_hostView eventDtdt_local = eventDtdt;

        // Loop over all events
        Kokkos::parallel_for(
            "interface_solidification_hook_refine",
            Kokkos::RangePolicy<host_exe, Kokkos::Schedule<Kokkos::Dynamic>>(0, eventCount),
            [&](const int n) {
                // Get point of event
                const int p = event_p_h(n);

                // A negative event time means this point is molten
                if (event_tl_h(n) < static_cast<FloatType>(0.0)) {
                    // Clear previous solidification values and keep the updated melt count
                    const SolidificationOutputValues<FloatType> outputs;
                    StoreSolidificationOutputs(
                        fields,
                        views,
                        static_cast<FloatType>(0.0),
                        event_num_melt_h(n),
                        outputs,
                        p
                    );
                    eventTSol_local(n) = static_cast<FloatType>(0.0);
                    eventDtdt_local(n) = static_cast<FloatType>(0.0);
                    return;
                }
                
                // Make search bracket
                SolidificationBracket<FloatType> bracket;
                bracket.T_hot = event_Tliq_h(n);
                bracket.t_hot = event_tl_h(n) - timestep;
                bracket.T_cold = event_Tsol_h(n);
                bracket.t_cold = event_tl_h(n);

                // Now find the time the event happens
                std::vector<int> start_seg;
                FloatType t_sol_exact = FindSolidificationTime(start_seg, sim, params, p, bracket);
                
                // Get quadrature nodes for that exact time (in solidification mode)
                Nodes<FloatType> nodes; 
                Calc::Integrate_Serial(nodes, start_seg, sim, t_sol_exact, true);
                
                // Get solidification values
                const SolidificationPrimaryValues<FloatType> primary = EvaluateSolidificationPrimary(sim, nodes, p);
                
                // Build output values from primary computed values
                const SolidificationOutputValues<FloatType> outputs = BuildSolidificationOutputs(primary, sim);

                // Store the solidification values
                StoreSolidificationOutputs(fields, views, t_sol_exact, event_num_melt_h(n), outputs, p);
                eventTSol_local(n) = t_sol_exact;
                eventDtdt_local(n) = outputs.dTdt;
            });
        Kokkos::fence();
    }

    template<typename FloatType>
    void Solidification_Hook<FloatType>::PostStep(const Meltpool::Tracking<FloatType>& meltpool, const Simdat<FloatType>& sim, const FloatType timestep) {
        lastEventCount = 0;

        // Skip if there is nothing to do
        if (calc.skip){ return; }
        
        // Get information from meltpool
        const int eventCount = meltpool.events.host_size;
        
        // If no events, skip
        if (eventCount <= 0) {return;}

        eventTSol = floating_hostView("SolidificationHook_eventTSol", eventCount);
        eventDtdt = floating_hostView("SolidificationHook_eventDtdt", eventCount);

        // Otherwise, do solidification on the meltpool views (TODO::asynchronously)...  // Wait if previous task is still pending
        Kokkos::Profiling::pushRegion("Solidification::Run");
        RunSolidificationBatch(meltpool, sim, timestep, eventCount);
        Kokkos::Profiling::popRegion();
        lastEventCount = eventCount;
        lastEventGeneration++;
        // pendingTask = std::async(std::launch::async, [this, meltpool, sim, timestep, eventCount] {
        //     RunSolidificationBatch(meltpool, sim, timestep, eventCount);
        // });
    }

    // Template initializations
    template struct Solidification_Views<float>;
    template struct Solidification_Views<double>;

    template float FindSolidificationTime(
        const std::vector<int>&,
        const Simdat<float>&,
        const Solidification_Parameters<float>&,
        const int,
        const SolidificationBracket<float>&);
    template double FindSolidificationTime(
        const std::vector<int>&,
        const Simdat<double>&,
        const Solidification_Parameters<double>&,
        const int,
        const SolidificationBracket<double>&);

    template SolidificationPrimaryValues<float> EvaluateSolidificationPrimary(const Simdat<float>&, const Nodes<float>&, const int);
    template SolidificationPrimaryValues<double> EvaluateSolidificationPrimary(const Simdat<double>&, const Nodes<double>&, const int);

    template SolidificationOutputValues<float> BuildSolidificationOutputs(const SolidificationPrimaryValues<float>&, const Simdat<float>&);
    template SolidificationOutputValues<double> BuildSolidificationOutputs(const SolidificationPrimaryValues<double>&, const Simdat<double>&);

    template class Solidification_Hook<float>;
    template class Solidification_Hook<double>;
}
