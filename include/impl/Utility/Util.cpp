// Includes where functions are defined
#include "impl/Utility/Util.hpp"

// Include necessary structures
#include <iostream>
#include <iomanip>
#include <sstream>

// Include necessary functions
#include <algorithm>
#include <cmath>

namespace Condor::impl{
	namespace Util{
		// Zero pad a number to specific width
		string ZeroPadNumber(const int num, const int width)
		{
			std::ostringstream ss;
			ss << std::setw(width) << std::setfill('0') << num;
			return ss.str();
		}

		// Zero pad a number (unspecified -> 7)
		string ZeroPadNumber(const int num)
		{
			return ZeroPadNumber(num, 7);
		}

		// Get p from ijk
		template<typename FloatType>
		int		ijk_to_p(const int i, const int j, const int k, const Simdat<FloatType>& sim) {
			return Grid::ijk_to_p(i, j, k, sim.domain.ynum, sim.domain.znum);
		}

		template<typename FloatType>
		void	Calc_AllScansEndTime(Simdat<FloatType>& sim) {
			for (const vector<path_seg<FloatType>>& path : sim.paths) {
				sim.util.allScansEndTime = std::max(sim.util.allScansEndTime, path.back().seg_time);
			}
			sim.util.approxEndTime = sim.util.allScansEndTime;
		}

		template<typename FloatType>
		void	Calc_ScanBounds(Domain<FloatType>& domain, const vector<vector<path_seg<FloatType>>>& paths) {
			// For max
			FloatType FloatType_MAX = std::numeric_limits<FloatType>::max();
			
			// Only do if things aren't set
			const bool xminUnset = (domain.xmin == FloatType_MAX);
			const bool xmaxUnset = (domain.xmax == -FloatType_MAX);
			const bool yminUnset = (domain.ymin == FloatType_MAX);
			const bool ymaxUnset = (domain.ymax == -FloatType_MAX);
			const bool zminUnset = (domain.ymin == FloatType_MAX);
			const bool zmaxUnset = (domain.ymax == -FloatType_MAX);

			// If z isn't set
			if (zmaxUnset) domain.zmax=static_cast<FloatType>(0.0);
			if (zminUnset) domain.zmin=domain.zmax-static_cast<FloatType>(1.0e-3);	
			if (domain.zmax == -FloatType_MAX) { domain.zmax = static_cast<FloatType>(0.0); }
			if (domain.zmin == FloatType_MAX) { domain.zmin = domain.zmax - 250e-6; }
			
			// Skip if all are set
			if (!(xminUnset || xmaxUnset || yminUnset || ymaxUnset)){
				return;
			}

			// Initialize
			FloatType xmin = FloatType_MAX;
			FloatType xmax = -FloatType_MAX;
			FloatType ymin = FloatType_MAX;
			FloatType ymax = -FloatType_MAX;
			
			for (const vector<path_seg<FloatType>>& path : paths) {
				for (const path_seg<FloatType>& seg : path) {
					if (seg.sqmod > 0.0f) {
						if (seg.sx < xmin) { xmin = seg.sx; }
						if (seg.sx > xmax) { xmax = seg.sx; }
						if (seg.sy < ymin) { ymin = seg.sy; }
						if (seg.sy > ymax) { ymax = seg.sy; }
					}
				}
			}

			if (domain.xmin == FloatType_MAX) { domain.xmin = xmin - 500e-6; }
			if (domain.xmax == -FloatType_MAX) { domain.xmax = xmax + 500e-6; }
			if (domain.ymin == FloatType_MAX) { domain.ymin = ymin - 500e-6; }
			if (domain.ymax == -FloatType_MAX) { domain.ymax = ymax + 500e-6; }
		}

		template<typename FloatType>
		void	Calc_NonD_dt(vector<Beam<FloatType>>& beams, const Material<FloatType>& material) {
			for (Beam<FloatType>& beam : beams) {
				beam.nond_dt = beam.ax * beam.ax / static_cast<float>(material.a);
			}
		}

		template<typename FloatType>
		void	Calc_RMax (Simdat<FloatType>& sim){
			sim.settings.t_hist = static_cast<FloatType>(1.0) / sim.settings.t_hist;
			if (sim.settings.r_max<static_cast<FloatType>(0.0)) {
				for (const Beam<FloatType>& beam : sim.beams) {
					//If the temperature never gets to 1/t_hist the peak temperature
					if (sim.settings.t_hist < exp(static_cast<FloatType>(1.5))) { sim.settings.r_max = beam.ax * sqrt(log(sim.settings.t_hist) / static_cast<FloatType>(3.0)); }
					else { sim.settings.r_max = beam.ax * pow(sim.settings.t_hist, (static_cast<FloatType>(1.0) / static_cast<FloatType>(3.0))) / sqrt(static_cast<FloatType>(2.0) * exp(static_cast<FloatType>(1.0))); }
					//If the power never gets to x (K/s)
					FloatType beta = pow(3.0 / 3.14159, 1.5) * beam.q / (sim.material.rho * sim.material.cps);
					FloatType temp_diff = sim.material.T_liq - sim.material.T_init;
					FloatType x = temp_diff * sim.settings.p_hist;
					FloatType r_max_2;
					if (beta / (x * beam.ax * beam.ax * beam.ax) < exp(static_cast<FloatType>(1.5))) { r_max_2 = beam.ax * sqrt(log(beta / (x * beam.ax * beam.ax * beam.ax)) / static_cast<FloatType>(3.0)); }
					else { r_max_2 = pow(beta / x, (static_cast<FloatType>(1.0)/ static_cast<FloatType>(3.0))) / sqrt(static_cast<FloatType>(2.0) * exp(static_cast<FloatType>(1.0))); }

					//Choose the greater of the two
					sim.settings.r_max = std::max(sim.settings.r_max, r_max_2);
					//sim.r_max = sim.ax*pow(sim.t_hist, (1.0 / 3.0)) / sqrt(2.0*exp(1.0)); 
				}
			}
		}

		template<typename FloatType>
		bool	InRMax(const FloatType x, const FloatType y, const Domain<FloatType>& domain, const Settings<FloatType>& settings) {
			if ((x > (domain.xmax + settings.r_max)) || (x < (domain.xmin - settings.r_max))) {return false;}
			else if ((y >(domain.ymax + settings.r_max)) || (y < (domain.ymin - settings.r_max))) {return false; }
			else {return true;}
		}

		template<typename FloatType>
		FloatType	t0calc(const FloatType t, const Beam<FloatType>& beam, const Material<FloatType>& material, const Settings<FloatType>& settings) {
			//Time for beam peak to decay to a fraction (t_hist) of it's initial power
			const FloatType t_hist_t = beam.nond_dt / static_cast<float>(12.0*pow(settings.t_hist, (2.0 / 3.0)) - 1); 

			//Time for beam to never exert more than a fraction (p_hist) of the difference between the preheat and solidus temperature
			const FloatType beta = pow(3.0f / 3.14159f, 1.5f) * beam.q / (material.rho * material.cps);
			const FloatType temp_diff = material.T_liq - material.T_init;
			const FloatType x = temp_diff * settings.p_hist;
			const FloatType y = 432.0f*t*(x*x)*(material.a*material.a*material.a) / (beta*beta);
			const FloatType p_hist_t = t / static_cast<float>((1.0f + sqrt(y))*(1.0f + sqrt(y)));  

			FloatType t0 = t - std::max(t_hist_t, p_hist_t);
			if (t0 < 0.0f) { t0 = 0.0f; }
			t0 = 0.0f;

			return t0;
		}

		template<typename FloatType>
		FloatType	GetRefTime(const FloatType tpp, const int seg, const vector<path_seg<FloatType>>& path, const Beam<FloatType>& beam) {
			
			FloatType ref_t;
			const FloatType spp = std::max(tpp / beam.nond_dt, static_cast<FloatType>(0.0));
			
			//Sets maximum time for line mode (derived from diffusion distance)
			if (path[seg].smode == 0) {
				FloatType t0 = 0.58870501125 * beam.ax / path[seg].sparam; // sqrt(log(sqrt(2)))~0.58870501125
				ref_t = t0 * sqrt(12.0f * spp + 1.0f);
			}
			//Sets maximum time for spot mode (equal to spot time)
			else if (path[seg].smode == 1) {
				ref_t = path[seg].seg_time - path[seg - 1].seg_time;
				if (ref_t == 0) {ref_t = 1.0e-9f;}
			}

			return ref_t;
		}

		template<typename FloatType>
		int_seg<FloatType>	GetBeamLoc(const FloatType time, const int seg, const vector<path_seg<FloatType>>& path, const Simdat<FloatType>& sim) {

			int_seg<FloatType> current_seg;
			if (path[seg].smode) {	//Location calculation for spot mode
				current_seg.xb = path[seg].sx;
				current_seg.yb = path[seg].sy;
				current_seg.zb = path[seg].sz;
			}
			else {							//Location calculation for line mode
				const FloatType dx = path[seg].sx - path[seg - 1].sx;
				const FloatType dy = path[seg].sy - path[seg - 1].sy;
				const FloatType dz = path[seg].sz - path[seg - 1].sz;
				const FloatType tcur = time - path[seg - 1].seg_time;
				const FloatType dt_cur = path[seg].seg_time - path[seg - 1].seg_time;
				current_seg.xb = path[seg - 1].sx + (tcur / dt_cur)*dx;
				current_seg.yb = path[seg - 1].sy + (tcur / dt_cur)*dy;
				current_seg.zb = path[seg - 1].sz + (tcur / dt_cur)*dz;
			}

			// If we are sufficiently outside the domain, set power to zero (so it won't be added to integration)
			if (InRMax(current_seg.xb,current_seg.yb,sim.domain,sim.settings)){ current_seg.qmod = path[seg].sqmod; }
			else { current_seg.qmod = 0.0f; }

			return current_seg;
		}

		template<typename FloatType>
		bool	sim_finish(const FloatType t, const Simdat<FloatType>& sim, const int liq_num) {
			bool isDone = false;
			if (t > sim.util.allScansEndTime && liq_num == 0) {isDone = true;}
			return isDone;
		}

		// --- Explicit Instantiations (within the namespace, after definitions) ---

		template int	ijk_to_p(const int, const int, const int, const Simdat<float>&);
		template int	ijk_to_p(const int, const int, const int, const Simdat<double>&);
		
		template bool	InRMax(const float, const float, const Domain<float>&, const Settings<float>&);
		template bool	InRMax(const double, const double, const Domain<double>&, const Settings<double>&);
		
		template float	t0calc(const float, const Beam<float>&, const Material<float>&, const Settings<float>&);
		template double	t0calc(const double, const Beam<double>&, const Material<double>&, const Settings<double>&);
		
		template float	GetRefTime(const float, const int, const vector<path_seg<float>>&, const Beam<float>&);
		template double	GetRefTime(const double, const int, const vector<path_seg<double>>&, const Beam<double>&);
		
		template int_seg<float> GetBeamLoc(const float, const int, const vector<path_seg<float>>&, const Simdat<float>&);
		template int_seg<double> GetBeamLoc(const double, const int, const vector<path_seg<double>>&, const Simdat<double>&);

		template bool	sim_finish(const float, const Simdat<float>&, const int);
		template bool	sim_finish(const double, const Simdat<double>&, const int);
		
		template void	Calc_AllScansEndTime(Simdat<float>&);
		template void	Calc_AllScansEndTime(Simdat<double>&);
		
		template void	Calc_ScanBounds(Domain<float>&, const vector<vector<path_seg<float>>>&);
		template void	Calc_ScanBounds(Domain<double>&, const vector<vector<path_seg<double>>>&);

		template void	Calc_NonD_dt(vector<Beam<float>>&, const Material<float>&);
		template void	Calc_NonD_dt(vector<Beam<double>>&, const Material<double>&);
		
		template void	Calc_RMax(Simdat<float>&);
		template void	Calc_RMax(Simdat<double>&);
	}
}
