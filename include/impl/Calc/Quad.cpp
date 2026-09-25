// Include where functions are defined
#include "impl/Calc/Quad.hpp"

// Includes necessary functions
#include "impl/Utility/Util.hpp"
#include "impl/Utility/Nodes.hpp"
#include <cmath>
#include <cfloat>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <omp.h>

namespace Condor::impl{
	namespace Calc{
	
		// Integrate serially for this timestep
		template <typename FloatType>
		void Integrate_Serial(Nodes<FloatType>& nodes, vector<int>& start_seg, const Simdat<FloatType>& sim, const FloatType t, const bool isSol) {
			// If it hasn't been initialized yet, initilaize it
			if (!start_seg.size()){
				// This enables the starting search path segment to be quickly initialized each time. 
				start_seg = vector<int>(sim.paths.size(), 1);
			}
			
			// Get dynamic quadrature nodes
			Calc::GaussIntegrate(nodes, start_seg, sim, t, isSol);

			// Add boundary conditions
			if (sim.domain.use_BCs) { Calc::AddBCs(nodes, sim.domain); }
		}

		// Add these helper functions in impl^2 namespace for quadrature calculation
		namespace { // Anonymous namespace

			// Fast log2 floor using bit scanning
			template <typename FloatType>
			KOKKOS_INLINE_FUNCTION
			int log2_floor(FloatType x) {
				return Kokkos::bit_width(static_cast<size_t>(x)) - 1;
			}

			// Fast log2 ceil using bit scanning
			template <typename FloatType>
			KOKKOS_INLINE_FUNCTION
			int log2_ceil(FloatType x) {
				return Kokkos::bit_width(static_cast<size_t>(x) - 1);
			}

			// Analytic determination for spot mode steps
			template <typename FloatType>
			KOKKOS_INLINE_FUNCTION
			int spot_steps(FloatType s_start, FloatType s_end) {
				const int delta = static_cast<int>(std::ceil(s_end - s_start));
				const int prevPow = log2_floor(s_start+static_cast<FloatType>(1.0));
				const int prevMinusOne = (1 << prevPow) - 1;
				return (log2_floor(prevMinusOne + delta)-prevPow) + 1;
			}

			// Upper bound for line mode steps
			template <typename FloatType>
			KOKKOS_INLINE_FUNCTION
			int line_steps(FloatType s_start, FloatType s_end, FloatType V) {
				//constexpr FloatType lineConst_z = 12.0 * std::sqrt(std::log(std::sqrt(2.0)));
				constexpr FloatType lineConst_z = static_cast<FloatType>(7.0644601350928481460694159587581978264843141363149232311351545390001);
				const FloatType A = lineConst_z / V;
				const FloatType z_start = 12.0f * s_start;
				const FloatType z_end = 12.0f * s_end;

				const FloatType t0 = std::sqrt(z_start + 1.0f);
				const FloatType T = std::sqrt(z_end + 1.0f);
				const FloatType denominator = std::sqrt(t0 * t0 + A * t0) - t0;
				
				return static_cast<int>(std::ceil((T - t0) / denominator));
			}
		}

		// Adaptive gaussian quadrature
		template <typename FloatType>
		void GaussIntegrate(Nodes<FloatType>& nodes, vector<int>& start_seg, const Simdat<FloatType>& sim, const FloatType t, const bool isSol) {
				
			static constexpr FloatType lineConst_s = 0.5887050112577373f;
			static constexpr FloatType piConst = 0.933162059717596f;
			static constexpr int ORDER_OFFSETS[4] = {0, 2, 6, 14};
			static constexpr int ORDER_COUNTS[4] = {2, 4, 8, 16};
			
			// Quadrature node locations for order 2, 4, 8, and 16
			static constexpr FloatType locs[30] = {
				-0.5773502691896257,  0.5773502691896257,
				-0.8611363115940526, -0.3399810435848563,  0.3399810435848563,  0.8611363115940526,
				-0.9602898564975363, -0.7966664774136267, -0.5255324099163290, -0.1834346424956498,  0.1834346424956498,  0.5255324099163290, 0.7966664774136267,  0.9602898564975363,
				-0.9894009349916499, -0.9445750230732326, -0.8656312023878318, -0.7554044083550030, -0.6178762444026438, -0.4580167776572274, -0.2816035507792589, -0.0950125098376374, 0.0950125098376374, 0.2816035507792589, 0.4580167776572274, 0.6178762444026438, 0.7554044083550030, 0.8656312023878318, 0.9445750230732326, 0.9894009349916499
			};

			// Quadrature weights for order 2, 4, 8, and 16
			static constexpr FloatType weights[30] = {
				1.0000000000000000, 1.0000000000000000,
				0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538,
				0.1012285362903763, 0.2223810344533745, 0.3137066458778873, 0.3626837833783620, 0.3626837833783620, 0.3137066458778873, 0.2223810344533745, 0.1012285362903763,
				0.0271524594117541, 0.0622535239386479, 0.0951585116824928, 0.1246289712555339, 0.1495959888165767, 0.1691565193950025, 0.1826034150449236, 0.1894506104550685, 0.1894506104550685, 0.1826034150449236, 0.1691565193950025, 0.1495959888165767, 0.1246289712555339, 0.0951585116824928, 0.0622535239386479, 0.0271524594117541
			};

			// Reserve size for nodes
			nodes.xb.reserve(1000);
			nodes.yb.reserve(1000);
			nodes.zb.reserve(1000);
			nodes.phix.reserve(1000);
			nodes.phiy.reserve(1000);
			nodes.phiz.reserve(1000);
			nodes.dtau.reserve(1000);
			nodes.expmod.reserve(1000);

			for (int i = 0; i < sim.paths.size(); i++) {
				const Beam<FloatType>& beam = sim.beams[i];
				const vector<path_seg<FloatType>>& path = sim.paths[i];
				int seg_temp = start_seg[i];

				// Set material constant
				const FloatType beta = piConst * beam.q / (sim.material.rho * sim.material.cps);

				while ((t > path[seg_temp].seg_time) && (seg_temp + 1 < path.size())) {seg_temp++;}
				while ((t < path[seg_temp - 1].seg_time) && (seg_temp - 1 > 0)) {seg_temp--;}
				
				if (!isSol && (omp_get_thread_num() == 0)){start_seg[i] = seg_temp;}

				// Add current beam only if solidifying 
				if (isSol){
					int_seg<FloatType> current_beam = Util::GetBeamLoc(t, seg_temp, path, sim);
					current_beam.qmod *= beta;
					if (t <= path.back().seg_time && current_beam.qmod > 0.0) {
						const FloatType phix = static_cast<FloatType>(1.0) / (beam.ax * beam.ax);
						const FloatType phiy = static_cast<FloatType>(1.0) / (beam.ay * beam.ay);
						const FloatType phiz = static_cast<FloatType>(1.0) / (beam.az * beam.az);
						nodes.size++;
						nodes.xb.push_back(current_beam.xb);
						nodes.yb.push_back(current_beam.yb);
						nodes.zb.push_back(current_beam.zb);
						nodes.phix.push_back(phix);
						nodes.phiy.push_back(phiy);
						nodes.phiz.push_back(phiz);
						nodes.dtau.push_back(static_cast<FloatType>(0.0));
						nodes.expmod.push_back(static_cast<FloatType>(0.5) * std::log(current_beam.qmod * current_beam.qmod * phix * phiy * phiz));
					}
				}

				// Make vectors of points and lines
				const int seg_max = seg_temp;
				vector<int> spot_segs; spot_segs.reserve(seg_max);
				vector<int> line_segs; line_segs.reserve(seg_max);
				for (int seg = 1; seg <= seg_max; seg++) {
					if (path[seg].sqmod == static_cast<FloatType>(0.0)){continue;}
					if (path[seg].smode == static_cast<FloatType>(1.0)){spot_segs.push_back(seg);}
					else{line_segs.push_back(seg);}
				}

				// Set consts and reserve sizes
				const size_t spot_num = spot_segs.size();
				const size_t line_num = line_segs.size();
				const size_t prev_nodeSize = nodes.xb.capacity();

				for (size_t spot_seg=0; spot_seg<spot_num; spot_seg++){
					// Get actual seg
					const int& seg = spot_segs[spot_seg];
					
					// Get position information
					const FloatType& xb = path[seg].sx;
					const FloatType& yb = path[seg].sy;
					const FloatType& zb = path[seg].sz;
					const FloatType qmod = path[seg].sqmod*beta;
					if (!Util::InRMax(xb,yb,sim.domain,sim.settings)){continue;}

					// Set path information
					const FloatType nond_dt = beam.nond_dt;
					FloatType tp_start = std::max(t - path[seg].seg_time, static_cast<FloatType>(0.0));
					FloatType tp_end = t - path[seg - 1].seg_time;
					FloatType sp_start = tp_start / nond_dt;
					FloatType sp_end = tp_end / nond_dt;

					// Calculate maximum number of steps
					const int steps = spot_steps(sp_start, sp_end);
					// const int steps = (sp_start-sp_end < sp_start) ? 1 : spot_steps(sp_start, sp_end);

					// Start stepping
					FloatType s_current = sp_start;
					for (int step = 0; step < steps; ++step) {
						
						// Order and step calculation
						const int flat = log2_floor(s_current + 1.0f);
						const int k = std::max(3-flat,0);
						const int step_size = (1 << flat);
						const int order = ORDER_COUNTS[k];
						const int offset = ORDER_OFFSETS[k];
						const FloatType s_next = std::min(s_current + step_size, sp_end);

						// Precalculate quadrature stuff
						const FloatType s_diff = s_next - s_current;
						const FloatType s_sum = s_next + s_current;
						
						// Quadrature
						for (int q = 0; q < order; ++q) {

							// Quadrature
							const FloatType loc = locs[offset + q];
							const FloatType weight = weights[offset + q];
							const FloatType ct = static_cast<FloatType>(6.0) * (s_diff * loc + s_sum) * sim.material.a * nond_dt;
							const FloatType phix = static_cast<FloatType>(1.0) / (beam.ax * beam.ax + ct); // Unscaled Gaussian shape factor in x.
							const FloatType phiy = static_cast<FloatType>(1.0) / (beam.ay * beam.ay + ct); // Unscaled Gaussian shape factor in y.
							const FloatType phiz = static_cast<FloatType>(1.0) / (beam.az * beam.az + ct); // Unscaled Gaussian shape factor in z.
							const FloatType dtau = static_cast<FloatType>(0.5)*s_diff * nond_dt * weight;

							if (dtau > static_cast<FloatType>(0.0)) {
								// Add information to nodes
								nodes.size++;
								nodes.xb.push_back(xb);
								nodes.yb.push_back(yb);
								nodes.zb.push_back(zb);
								nodes.phix.push_back(phix);
								nodes.phiy.push_back(phiy);	
								nodes.phiz.push_back(phiz);
								nodes.dtau.push_back(dtau);
								nodes.expmod.push_back(static_cast<FloatType>(0.5) * std::log(qmod * qmod * phix * phiy * phiz));
							}
						}

						// Set s for next loop
						s_current = s_next;
					}
				}

				// Do just the lines
				for (size_t line_seg=0; line_seg<line_num; line_seg++){
					const int& seg = line_segs[line_seg];

					// Set object and variables which don't change based on node				
					const FloatType px =  path[seg - 1].sx;
					const FloatType py =  path[seg - 1].sy;
					const FloatType pz =  path[seg - 1].sz;
					const FloatType pt =  path[seg - 1].seg_time;
					const FloatType dx = path[seg].sx - path[seg - 1].sx;
					const FloatType dy = path[seg].sy - path[seg - 1].sy;
					const FloatType dz = path[seg].sz - path[seg - 1].sz;
					const FloatType dt_cur = path[seg].seg_time - path[seg - 1].seg_time;
					const FloatType qmod = path[seg].sqmod*beta;

					// Set path information
					const FloatType nond_dt = beam.nond_dt;
					FloatType tp_start = std::max(t - path[seg].seg_time, static_cast<FloatType>(0.0));
					FloatType tp_end = t - path[seg - 1].seg_time;
					FloatType sp_start = tp_start / nond_dt;
					FloatType sp_end = tp_end / nond_dt;
					const FloatType V = path[seg].sparam * (beam.ax / sim.material.a);

					// Calculate maximum number of steps
					const int steps = line_steps(sp_start, sp_end, V);

					// Start stepping
					FloatType s_current = sp_start;
					for (int step = 0; step < steps; ++step) {
						if (s_current == sp_end) {continue;}

						// Order and step calculation
						const int flat = log2_floor(s_current + static_cast<FloatType>(1.0));
						const int k = std::max(3-flat,0);
						const FloatType step_size = lineConst_s / V * sqrt(static_cast<FloatType>(12.0) * s_current + static_cast<FloatType>(1.0));
						const int order = ORDER_COUNTS[k];
						const int offset = ORDER_OFFSETS[k];
						const FloatType s_next = std::min(s_current + step_size, sp_end);

						// Precalculate quadrature stuff
						const FloatType s_diff = s_next - s_current;
						const FloatType s_sum = s_next + s_current;

						// Quadrature
						for (int q = 0; q < order; ++q) {					

							// Quadrature
							const FloatType loc = locs[offset + q];
							const FloatType weight = weights[offset + q];
							const FloatType tau = static_cast<FloatType>(0.5)*(s_diff * loc + s_sum) * nond_dt;
							const FloatType ct = static_cast<FloatType>(12.0) * sim.material.a * tau;
							const FloatType phix = static_cast<FloatType>(1.0) / (beam.ax * beam.ax + ct); // Unscaled Gaussian shape factor in x.
							const FloatType phiy = static_cast<FloatType>(1.0) / (beam.ay * beam.ay + ct); // Unscaled Gaussian shape factor in y.
							const FloatType phiz = static_cast<FloatType>(1.0) / (beam.az * beam.az + ct); // Unscaled Gaussian shape factor in z.
							const FloatType dtau = static_cast<FloatType>(0.5) * s_diff * nond_dt * weight;

							// Location
							const FloatType tcur = (t - tau) - path[seg - 1].seg_time;
							const FloatType xb = path[seg - 1].sx + (tcur / dt_cur)*dx;
							const FloatType yb = path[seg - 1].sy + (tcur / dt_cur)*dy;
							const FloatType zb = path[seg - 1].sz + (tcur / dt_cur)*dz;

							if (Util::InRMax(xb,yb,sim.domain,sim.settings) && dtau > static_cast<FloatType>(0.0)) {	
								// Add information to nodes
								nodes.size++;
								nodes.xb.push_back(xb);
								nodes.yb.push_back(yb);
								nodes.zb.push_back(zb);
								nodes.phix.push_back(phix);	
								nodes.phiy.push_back(phiy);	
								nodes.phiz.push_back(phiz);	
								nodes.dtau.push_back(dtau);
								nodes.expmod.push_back(static_cast<FloatType>(0.5) * std::log(qmod * qmod * phix * phiy * phiz));
							}
						}

						// Set s for next loop
						s_current = s_next;
					}
				}
			}
		}

		// Adding boundary conditions
		template<typename FloatType>
		void AddBCs(Nodes<FloatType>& nodes, const Domain<FloatType>& domain) {

			vector<vector<int>> allCoords;
			vector<vector<int>> newCoords;
			vector<int> coords = { 0,0,0 };
			newCoords.push_back(coords);
			allCoords.push_back(coords);

			FloatType xmin, xmax;
			FloatType ymin, ymax;
			FloatType zmin, zmax;

			xmin = domain.BC_xmin;
			xmax = domain.BC_xmax;
			ymin = domain.BC_ymin;
			ymax = domain.BC_ymax;
			zmin = domain.BC_zmin;
			zmax = domain.zmax;

			vector<vector<int>> checkCoords = newCoords;
			newCoords.clear();
			// Strength is how many reflections to use
			int iter = 0; int strength = 0; 
			if (domain.BC_reflections != INT_MAX) { strength = domain.BC_reflections; }
			while (checkCoords.size() > 0 && iter < strength) {
				for (int i = 0; i < checkCoords.size(); i++) {
					if (xmin != std::numeric_limits<FloatType>::max()) {
						coords = checkCoords[i];
						coords[0] = (-1 - coords[0]);
						if (std::find(allCoords.begin(), allCoords.end(), coords) == allCoords.end()) {;
							allCoords.push_back(coords);
							newCoords.push_back(coords);
						}
					};
					if (xmax != std::numeric_limits<FloatType>::max()) {
						coords = checkCoords[i];
						coords[0] = (1 - coords[0]);
						if (std::find(allCoords.begin(), allCoords.end(), coords) == allCoords.end()) {
							allCoords.push_back(coords);
							newCoords.push_back(coords);
						}
					};
					if (ymin != std::numeric_limits<FloatType>::max()) {
						coords = checkCoords[i];
						coords[1] = (-1 - coords[1]);
						if (std::find(allCoords.begin(), allCoords.end(), coords) == allCoords.end()) {
							allCoords.push_back(coords);
							newCoords.push_back(coords);
						}
					};
					if (ymax != std::numeric_limits<FloatType>::max()) {
						coords = checkCoords[i];
						coords[1] = (1 - coords[1]);
						if (std::find(allCoords.begin(), allCoords.end(), coords) == allCoords.end()) {
							allCoords.push_back(coords);
							newCoords.push_back(coords);
						}
					};
					if (zmin != std::numeric_limits<FloatType>::max()) {
						coords = checkCoords[i];
						coords[2] = (-1 - coords[2]);
						if (std::find(allCoords.begin(), allCoords.end(), coords) == allCoords.end()) {
							allCoords.push_back(coords);
							newCoords.push_back(coords);
						}
					};
					if (zmax != std::numeric_limits<FloatType>::max()) {
						coords = checkCoords[i];
						coords[2] = (1 - coords[2]);
						
						if (std::find(allCoords.begin(), allCoords.end(), coords) == allCoords.end() && (coords[2]!=1)) {
							allCoords.push_back(coords);
							newCoords.push_back(coords);
						}
					};
				}
				checkCoords = newCoords;
				newCoords.clear();
				iter += 1;
			}

			int org_size = nodes.size;
			Nodes<FloatType> nodes_org = nodes;
			Util::ClearNodes(nodes);

			// Now just convert between refCoords and realCoords
			for (int i = 0; i < allCoords.size(); i++) {
				int xRef = allCoords[i][0];
				int yRef = allCoords[i][1];
				int zRef = allCoords[i][2];	

				int nX = std::abs(xRef % 2);
				int nY = std::abs(yRef % 2);
				int nZ = std::abs(zRef % 2);	

				Nodes<FloatType> nodes2 = nodes_org;
				for (int i = 0; i < org_size; i++) { 
					nodes2.xb[i] = (xRef + nX) * xmax - (xRef - nX) * xmin + (1 - 2 * nX) * nodes2.xb[i];
					nodes2.yb[i] = (yRef + nY) * ymax - (yRef - nY) * ymin + (1 - 2 * nY) * nodes2.yb[i];
					nodes2.zb[i] = (zRef + nZ) * zmax - (zRef - nZ) * zmin + (1 - 2 * nZ) * nodes2.zb[i];
				}
				Util::CombineNodes(nodes, nodes2);	
			}
		}

		// --- Explicit Instantiations (within the namespace, after definitions) ---
		template void Integrate_Serial(Nodes<float>&, vector<int>&, const Simdat<float>&, const float, const bool);
		template void Integrate_Serial(Nodes<double>&, vector<int>&, const Simdat<double>&, const double, const bool);

		template void GaussIntegrate(Nodes<float>&, vector<int>&, const Simdat<float>&, const float, const bool);
		template void GaussIntegrate(Nodes<double>&, vector<int>&, const Simdat<double>&, const double, const bool);

		template void AddBCs(Nodes<float>&, const Domain<float>&);
		template void AddBCs(Nodes<double>&, const Domain<double>&);
	}	
}
