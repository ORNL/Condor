// Internal includes
#include "impl/Meltpool/Beam.hpp"

// External includes
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <utility>

namespace Condor::impl::Meltpool{
	namespace{
		// Subview range type used when copying only active seed entries.
		typedef std::pair<size_t, size_t> SeedRange;
		// Device subview type for active seed transfers.
		typedef decltype(Kokkos::subview(std::declval<int_deviceView>(), std::declval<SeedRange>())) DeviceSeedSubview;
		// Host subview type for active seed transfers.
		typedef decltype(Kokkos::subview(std::declval<int_hostView>(), std::declval<SeedRange>())) HostSeedSubview;

		template<typename FloatType>
		FloatType clamp01(const FloatType value) {
			// Clamp interpolation parameters onto the segment.
			return std::max(static_cast<FloatType>(0.0), std::min(static_cast<FloatType>(1.0), value));
		}

		template<typename FloatType>
		int first_preadvance_step_for_active_time(const FloatType active_time, const FloatType timestep) {
			// Interface asks the tracer for a pre-advance step whose interval is
			// (step * dt, (step + 1) * dt].  An active window that begins just
			// after a grid time belongs to that same pre-advance step, not the
			// next one.
			const FloatType scaled = active_time / timestep;
			const FloatType nearest = std::round(scaled);
			const FloatType snap_tol = static_cast<FloatType>(1.0e-6);
			const FloatType snapped =
				(std::abs(scaled - nearest) <= snap_tol)
					? nearest
					: scaled;
			const int step = static_cast<int>(std::floor(snapped));
			return std::max(0, step);
		}

		template<typename FloatType>
		bool clip_time_to_trace_box(
			const Simdat<FloatType>& sim,
			const FloatType radius,
			const FloatType x0,
			const FloatType y0,
			const FloatType x1,
			const FloatType y1,
			const FloatType t0,
			const FloatType t1,
			FloatType& enter_time,
			FloatType& leave_time)
		{
			// Expanded domain bounds used to decide beam relevance.
			const FloatType xmin = sim.domain.xmin - radius;
			const FloatType xmax = sim.domain.xmax + radius;
			const FloatType ymin = sim.domain.ymin - radius;
			const FloatType ymax = sim.domain.ymax + radius;
			// Segment displacement in the trace plane.
			const FloatType dx = x1 - x0;
			const FloatType dy = y1 - y0;
			// Liang-Barsky interval parameters for the clipped segment.
			FloatType u0 = static_cast<FloatType>(0.0);
			FloatType u1 = static_cast<FloatType>(1.0);

			// Clip one half-space and update the accepted interval.
			const std::function<bool(const FloatType, const FloatType)> clip =
				[&](const FloatType p, const FloatType q) -> bool {
				if (std::abs(p) <= std::numeric_limits<FloatType>::epsilon()) {
					return q >= static_cast<FloatType>(0.0);
				}
				// Crossing fraction for this clipping plane.
				const FloatType u = q / p;
				if (p < static_cast<FloatType>(0.0)) {
					if (u > u1) {
						return false;
					}
					u0 = std::max(u0, u);
				}
				else {
					if (u < u0) {
						return false;
					}
					u1 = std::min(u1, u);
				}
				return true;
			};

			if (!clip(-dx, x0 - xmin) || !clip(dx, xmax - x0) || !clip(-dy, y0 - ymin) || !clip(dy, ymax - y0) || u1 < u0) {
				return false;
			}

			// Physical segment duration used to map clipped fractions to time.
			const FloatType dt = std::max(static_cast<FloatType>(0.0), t1 - t0);
			enter_time = t0 + clamp01(u0) * dt;
			leave_time = t0 + clamp01(u1) * dt;
			return leave_time >= enter_time;
		}

		template<typename FloatType>
		void add_seed_index(const Simdat<FloatType>& sim, const int i, const int j, vector<unsigned char>& touched, vector<int>& seeds) {
			// Reject indices outside the 2D surface grid.
			if (i < 0 || i >= sim.domain.xnum || j < 0 || j >= sim.domain.ynum) {
				return;
			}
			// Flattened column id consumed by the meltpool tracker.
			const int flat = i * sim.domain.ynum + j;
			// Add each surface column once per trace query.
			if (!touched[static_cast<size_t>(flat)]) {
				touched[static_cast<size_t>(flat)] = 1;
				seeds.push_back(flat);
			}
		}

		template<typename FloatType>
		void add_surface_point(const Simdat<FloatType>& sim, const FloatType radius, const FloatType x, const FloatType y, vector<unsigned char>& touched, vector<int>& seeds) {
			// Cell index nearest to the beam position.
			int i0 = static_cast<int>(std::floor((x - sim.domain.xmin) / sim.domain.res));
			int j0 = static_cast<int>(std::floor((y - sim.domain.ymin) / sim.domain.res));

			// Zero-radius traces seed only the local interpolation stencil.
			if (radius <= static_cast<FloatType>(0.0)) {
				i0 = std::max(-1, std::min(sim.domain.xnum - 1, i0));
				j0 = std::max(-1, std::min(sim.domain.ynum - 1, j0));
				for (int di = 0; di <= 1; di++) {
					for (int dj = 0; dj <= 1; dj++) {
						add_seed_index(sim, i0 + di, j0 + dj, touched, seeds);
					}
				}
				return;
			}

			// Interior points also use the local interpolation stencil.
			const bool inside =
				(i0 >= 0) && (i0 < sim.domain.xnum) &&
				(j0 >= 0) && (j0 < sim.domain.ynum);
			if (inside) {
				for (int di = 0; di <= 1; di++) {
					for (int dj = 0; dj <= 1; dj++) {
						add_seed_index(sim, i0 + di, j0 + dj, touched, seeds);
					}
				}
				return;
			}

			// Squared radius used to test edge/outside domain footprints.
			const FloatType r2 = radius * radius;
			// Candidate grid bounds around the requested trace disk.
			const int imin = std::max(0, static_cast<int>(std::floor((x - radius - sim.domain.xmin) / sim.domain.res)));
			const int imax = std::min(sim.domain.xnum - 1, static_cast<int>(std::ceil((x + radius - sim.domain.xmin) / sim.domain.res)));
			const int jmin = std::max(0, static_cast<int>(std::floor((y - radius - sim.domain.ymin) / sim.domain.res)));
			const int jmax = std::min(sim.domain.ynum - 1, static_cast<int>(std::ceil((y + radius - sim.domain.ymin) / sim.domain.res)));
			for (int i = imin; i <= imax; i++) {
				// Candidate surface x coordinate.
				const FloatType xi = sim.domain.xmin + static_cast<FloatType>(i) * sim.domain.res;
				for (int j = jmin; j <= jmax; j++) {
					// Candidate surface y coordinate.
					const FloatType yj = sim.domain.ymin + static_cast<FloatType>(j) * sim.domain.res;
					// Offset from the beam position to the candidate column.
					const FloatType dx = xi - x;
					const FloatType dy = yj - y;
					if (dx * dx + dy * dy <= r2) {
						add_seed_index(sim, i, j, touched, seeds);
					}
				}
			}
		}

		template<typename FloatType>
		void add_x_perimeter_points(const Simdat<FloatType>& sim, const int i, const FloatType r2, const FloatType x, const FloatType y, vector<unsigned char>& touched, vector<int>& seeds) {
			const FloatType xi = sim.domain.xmin + static_cast<FloatType>(i) * sim.domain.xres;
			for (int j = 0; j < sim.domain.ynum; j++) {
				const FloatType yj = sim.domain.ymin + static_cast<FloatType>(j) * sim.domain.yres;
				const FloatType dx = xi - x;
				const FloatType dy = yj - y;
				if (dx * dx + dy * dy <= r2) {
					add_seed_index(sim, i, j, touched, seeds);
				}
			}
		}

		template<typename FloatType>
		void add_y_perimeter_points(const Simdat<FloatType>& sim, const int j, const FloatType r2, const FloatType x, const FloatType y, vector<unsigned char>& touched, vector<int>& seeds) {
			const FloatType yj = sim.domain.ymin + static_cast<FloatType>(j) * sim.domain.yres;
			for (int i = 0; i < sim.domain.xnum; i++) {
				const FloatType xi = sim.domain.xmin + static_cast<FloatType>(i) * sim.domain.xres;
				const FloatType dx = xi - x;
				const FloatType dy = yj - y;
				if (dx * dx + dy * dy <= r2) {
					add_seed_index(sim, i, j, touched, seeds);
				}
			}
		}

		template<typename FloatType>
		void add_perimeter_points(const Simdat<FloatType>& sim, const FloatType radius, const FloatType x, const FloatType y, vector<unsigned char>& touched, vector<int>& seeds) {
			if (radius <= static_cast<FloatType>(0.0)) {
				return;
			}
			const FloatType r2 = radius * radius;
			if (x < sim.domain.xmin) { add_x_perimeter_points(sim, 0, r2, x, y, touched, seeds); }
			if (x > sim.domain.xmax) { add_x_perimeter_points(sim, sim.domain.xnum - 1, r2, x, y, touched, seeds); }
			if (y < sim.domain.ymin) { add_y_perimeter_points(sim, 0, r2, x, y, touched, seeds); }
			if (y > sim.domain.ymax) { add_y_perimeter_points(sim, sim.domain.ynum - 1, r2, x, y, touched, seeds); }
		}

		template<typename FloatType>
		FloatType beam_trace_radius(const Simdat<FloatType>& sim, const int path_index) {
			if (sim.settings.trace_radius >= static_cast<FloatType>(0.0)) {
				return sim.settings.trace_radius;
			}
			const Beam<FloatType>& beam = sim.beams[static_cast<size_t>(path_index)];
			return std::max(beam.ax, beam.ay);
		}

		template<typename FloatType>
		void add_line_trace(
			const Simdat<FloatType>& sim,
			const vector<path_seg<FloatType>>& path,
			const int segment_index,
			const FloatType t_start,
			const FloatType t_end,
			const FloatType radius_check,
			const FloatType trace_radius,
			vector<unsigned char>& touched,
			vector<int>& seeds)
		{
			// Adjacent path records defining the current line segment.
			const path_seg<FloatType>& a = path[static_cast<size_t>(segment_index - 1)];
			const path_seg<FloatType>& b = path[static_cast<size_t>(segment_index)];
			// Segment duration and normalized overlap for this query interval.
			const FloatType dt = b.seg_time - a.seg_time;
			if (dt <= static_cast<FloatType>(0.0)) {
				add_surface_point(sim, trace_radius, b.sx, b.sy, touched, seeds);
				add_perimeter_points(sim, radius_check, b.sx, b.sy, touched, seeds);
				return;
			}
			const FloatType s0 = clamp01((t_start - a.seg_time) / dt);
			const FloatType s1 = clamp01((t_end - a.seg_time) / dt);
			if (s1 < s0) {
				return;
			}
			// Segment displacement used to sample along the traced interval.
			const FloatType dx = b.sx - a.sx;
			const FloatType dy = b.sy - a.sy;
			// Segment length used to normalize trace samples.
			const FloatType length = std::sqrt(dx * dx + dy * dy);
			if (length <= std::numeric_limits<FloatType>::epsilon()) {
				add_surface_point(sim, trace_radius, b.sx, b.sy, touched, seeds);
				add_perimeter_points(sim, radius_check, b.sx, b.sy, touched, seeds);
				return;
			}

			// Trace samples are spaced no coarser than the grid resolution.
			const FloatType trace_length = (s1 - s0) * length;
			const int samples = std::max(1, static_cast<int>(std::ceil(trace_length / std::max(sim.domain.res, std::numeric_limits<FloatType>::epsilon()))));
			for (int sample = 0; sample <= samples; sample++) {
				// Position along the current timestep's line overlap.
				const FloatType s = s0 + (s1 - s0) * static_cast<FloatType>(sample) / static_cast<FloatType>(samples);
				const FloatType x = a.sx + s * dx;
				const FloatType y = a.sy + s * dy;
				add_surface_point(sim, trace_radius, x, y, touched, seeds);
				add_perimeter_points(sim, radius_check, x, y, touched, seeds);
			}
		}

	}

	TransferSeedIndices::TransferSeedIndices(const int initial_capacity)
		// Allocate a reusable host transfer buffer for compact seed lists.
		: values(Kokkos::ViewAllocateWithoutInitializing("beam_trace_seed_transfer"), std::max(initial_capacity, 1)),
			size(0),
			capacity(std::max(initial_capacity, 1)) {}

	void TransferSeedIndices::CopyFromVector(const vector<int>& source) {
		size = static_cast<int>(source.size());
		capacity = size;
		if (size > 0) {
			values = int_hostView(const_cast<int*>(source.data()), static_cast<size_t>(size));
		}
	}

	DeviceSeedIndices::DeviceSeedIndices(const int initial_capacity)
		// Allocate a reusable device seed buffer for meltpool seeding kernels.
		: values("beam_trace_seed_device", std::max(initial_capacity, 1)),
			size(0),
			capacity(std::max(initial_capacity, 1)) {}

	void DeviceSeedIndices::CopyFromHost(const TransferSeedIndices& host_values, device_exe* exec) {
		// Current logical size mirrors the host transfer buffer.
		size = host_values.size;
		// Grow the device buffer only when the staged result exceeds capacity.
		if (size > capacity) {
			capacity = std::max(size * 2, 1);
			values = int_deviceView("beam_trace_seed_device", capacity);
		}
		// Empty seed lists do not need a device copy.
		if (size <= 0) {
			return;
		}

		// Active range limits the copy to valid seed entries.
		const SeedRange active_range = std::make_pair(static_cast<size_t>(0), static_cast<size_t>(size));
		// Destination device subview for the active seed range.
		DeviceSeedSubview dst = Kokkos::subview(values, active_range);
		// Source host subview for the active seed range.
		HostSeedSubview src = Kokkos::subview(host_values.values, active_range);
		// Use the supplied execution space when a caller wants async ordering.
		if (exec != nullptr) {
			Kokkos::deep_copy(*exec, dst, src);
		}
		else {
			Kokkos::deep_copy(dst, src);
		}
	}

	template<typename FloatType>
	BeamTracer<FloatType>::BeamTracer(const Simdat<FloatType>& sim_in)
		// Delegate to the timestep-aware constructor with stepping disabled.
		: BeamTracer(sim_in, static_cast<FloatType>(0.0)) {}

	template<typename FloatType>
	BeamTracer<FloatType>::BeamTracer(const Simdat<FloatType>& sim_in, const FloatType timestep_in)
		// Store Simdat and allocate one-entry staging buffers immediately.
		: sim(&sim_in),
			radius_check(sim_in.settings.radius_check),
			trace_radius(sim_in.settings.trace_radius),
			timestep(timestep_in > static_cast<FloatType>(0.0) ? timestep_in : static_cast<FloatType>(0.0)),
			transfer_h(1),
			seed_indices_d(1),
			indices_d(seed_indices_d.values)
	{
		// Final iteration .
		final_iter = (timestep > static_cast<FloatType>(0.0))
			? std::max(0, static_cast<int>(std::ceil(sim->util.allScansEndTime / timestep)))
			: 0;
		// Build the segment index in the background while other setup continues.
		StartWorker();
	}

	template<typename FloatType>
	BeamTracer<FloatType>::~BeamTracer() {
		// Ensure the background index build cannot outlive the tracer.
		Stop();
	}

	template<typename FloatType>
	void BeamTracer<FloatType>::BuildSegments() {
		// External rebuilds first join any pending worker.
		Stop();
		// Rebuild synchronously so callers can use the index immediately.
		BuildSegmentsFromCurrentThread();
	}

	template<typename FloatType>
	void BeamTracer<FloatType>::BuildSegmentsFromCurrentThread() {
		// Local segment cache avoids exposing partial state during async build.
		vector<BeamTraceSegment<FloatType>> built_segments;
		// Local public window cache aligned with active segments.
		vector<BeamTraceDomainWindow<FloatType>> built_windows;
		// Per-path active segment indices for binary-search queries.
		vector<vector<int>> built_active_segments(sim->paths.size());
		// Per-path active segment event times aligned with built_active_segments.
		vector<vector<FloatType>> built_active_times(sim->paths.size());

		// Walk every beam path and classify each segment once.
		for (int path_index = 0; path_index < static_cast<int>(sim->paths.size()); path_index++) {
			// Source path for this beam.
			const vector<path_seg<FloatType>>& path = sim->paths[static_cast<size_t>(path_index)];
			for (int seg = 1; seg < static_cast<int>(path.size()); seg++) {
				// Previous/current records define the physical segment.
				const path_seg<FloatType>& a = path[static_cast<size_t>(seg - 1)];
				const path_seg<FloatType>& b = path[static_cast<size_t>(seg)];
				// Segment metadata record that will be stored in the cache.
				BeamTraceSegment<FloatType> entry;
				entry.path_index = path_index;
				entry.segment_index = seg;
				entry.smode = b.smode;
				entry.t0 = a.seg_time;
				entry.t1 = b.seg_time;

				// Zero-power segments never contribute static or dynamic beam seeds.
				if (b.sqmod <= static_cast<FloatType>(0.0)) {
					entry.active = false;
					built_segments.push_back(std::move(entry));
					continue;
				}

				// Nonpositive radius treats every segment as active.
				if (radius_check <= static_cast<FloatType>(0.0)) {
					entry.enter_time = entry.t0;
					entry.leave_time = entry.t1;
					entry.active = true;
				}
				// Line segments are clipped against the expanded domain box.
				else if (b.smode == 0) {
					entry.active = clip_time_to_trace_box(*sim, radius_check, a.sx, a.sy, b.sx, b.sy, entry.t0, entry.t1, entry.enter_time, entry.leave_time);
				}
				else {
					// Spot segments are active if the spot lies inside the expanded domain.
					const bool spot_active =
						(b.sx >= sim->domain.xmin - radius_check) && (b.sx <= sim->domain.xmax + radius_check) &&
						(b.sy >= sim->domain.ymin - radius_check) && (b.sy <= sim->domain.ymax + radius_check);
					entry.enter_time = entry.t0;
					entry.leave_time = entry.t1;
					entry.active = spot_active;
				}

				// Inactive segments are kept in the full cache but not indexed.
				if (!entry.active) {
					built_segments.push_back(std::move(entry));
					continue;
				}

				// Event time controls inclusion in (t_start, t_end] queries.
				entry.event_time = entry.leave_time;

				// Public window record describes where the active segment enters the domain.
				BeamTraceDomainWindow<FloatType> window;
				window.path_index = path_index;
				window.segment_index = seg;
				window.enter_time = entry.enter_time;
				window.leave_time = entry.leave_time;
				built_windows.push_back(window);

				// Segment index is captured before moving entry into built_segments.
				const int built_index = static_cast<int>(built_segments.size());
				// Per-path index and time arrays stay aligned for binary search.
				built_active_segments[static_cast<size_t>(path_index)].push_back(built_index);
				built_active_times[static_cast<size_t>(path_index)].push_back(entry.event_time);
				built_segments.push_back(std::move(entry));
			}
		}

		// Publish the completed index atomically with respect to query waiters.
		{
			std::lock_guard<std::mutex> lock(mutex);
			segments = std::move(built_segments);
			domain_windows = std::move(built_windows);
			active_segments_by_path = std::move(built_active_segments);
			active_event_times_by_path = std::move(built_active_times);
			index_ready = true;
			worker_done = true;
			worker_started = false;
		}
		// Wake any query blocked on WaitForIndex.
		cv.notify_all();
	}

	template<typename FloatType>
	void BeamTracer<FloatType>::TraceWindowHost(const FloatType t_start, const FloatType t_end, vector<int>& seed_indices_h) const {
		ComputeTraceHost(t_start, t_end, true, seed_indices_h);
	}

	template<typename FloatType>
	void BeamTracer<FloatType>::WaitForIndex() const {
		// Query calls block until the async index build publishes complete data.
		std::unique_lock<std::mutex> lock(mutex);
		cv.wait(lock, [&]() {
			return index_ready || worker_done;
		});
		// A done worker without a ready index indicates a failed/invalid build.
		if (!index_ready) {
			throw std::runtime_error("BeamTracer segment index is not available");
		}
	}

	template<typename FloatType>
	int BeamTracer<FloatType>::FindActiveSegmentAtTime(const int path_index, const FloatType time) const {
		// Reject invalid path ids before reading the per-path index.
		if (path_index < 0 || path_index >= static_cast<int>(active_segments_by_path.size())) {
			return -1;
		}
		// Active segment ids for this path.
		const vector<int>& active = active_segments_by_path[static_cast<size_t>(path_index)];
		// Event times aligned with the active segment ids.
		const vector<FloatType>& times = active_event_times_by_path[static_cast<size_t>(path_index)];
		// First active segment whose event time is not before the query time.
		const typename vector<FloatType>::const_iterator it = std::lower_bound(times.begin(), times.end(), time);
		if (it == times.end()) {
			return -1;
		}
		// Position in the per-path active arrays.
		const int active_pos = static_cast<int>(std::distance(times.begin(), it));
		// Full segment-cache index corresponding to the active position.
		const int segment_index = active[static_cast<size_t>(active_pos)];
		// Segment metadata used to verify the time is inside the clipped window.
		const BeamTraceSegment<FloatType>& entry = segments[static_cast<size_t>(segment_index)];
		if (time >= entry.enter_time && time <= entry.leave_time && time >= entry.t0 && time <= entry.t1) {
			return segment_index;
		}
		return -1;
	}

	template<typename FloatType>
	void BeamTracer<FloatType>::ComputeTraceHost(const FloatType t_start, const FloatType t_end, const bool include_current_at_end, vector<int>& seed_indices_h) const {
		// Ensure the segment index has been built before reading it.
		WaitForIndex();
		// Per-query duplicate guard across static and dynamic seeds.
		vector<unsigned char> touched(static_cast<size_t>(sim->domain.xnum * sim->domain.ynum), 0);
		// Output vector is owned by the caller and reused between queries.
		seed_indices_h.clear();
		// Reversed intervals intentionally produce no seeds.
		if (t_end < t_start) {
			return;
		}

		if (include_current_at_end) {
			for (int path_index = 0; path_index < static_cast<int>(active_segments_by_path.size()); path_index++) {
				// Source path and configured trace radius for this beam.
				const vector<path_seg<FloatType>>& path = sim->paths[static_cast<size_t>(path_index)];
				const FloatType seed_radius = (trace_radius >= static_cast<FloatType>(0.0))
					? trace_radius
					: beam_trace_radius(*sim, path_index);
				const vector<int>& active = active_segments_by_path[static_cast<size_t>(path_index)];
				for (const int segment_index : active) {
					const BeamTraceSegment<FloatType>& entry = segments[static_cast<size_t>(segment_index)];
					const FloatType overlap_start = std::max(t_start, std::max(entry.enter_time, entry.t0));
					const FloatType overlap_end = std::min(t_end, std::min(entry.leave_time, entry.t1));
					if (overlap_end <= t_start || overlap_end < overlap_start) {
						continue;
					}
					if (entry.smode == 0) {
						add_line_trace(*sim, path, entry.segment_index, overlap_start, overlap_end, radius_check, seed_radius, touched, seed_indices_h);
					}
					else {
						const path_seg<FloatType>& spot = path[static_cast<size_t>(entry.segment_index)];
						add_surface_point(*sim, seed_radius, spot.sx, spot.sy, touched, seed_indices_h);
						add_perimeter_points(*sim, radius_check, spot.sx, spot.sy, touched, seed_indices_h);
					}
				}
			}
		}

		// Stable ordering makes downstream behavior deterministic.
		std::sort(seed_indices_h.begin(), seed_indices_h.end());
	}

	template<typename FloatType>
	void BeamTracer<FloatType>::Stage(const BeamTraceState<FloatType>& state, device_exe* exec) {
		// Copy host vector results into the reusable host view.
		transfer_h.CopyFromVector(state.seed_indices_h);
		// Stage the compact host view to the reusable device view.
		seed_indices_d.CopyFromHost(transfer_h, exec);
		// Cache the valid seed count for tracker access.
		num_indices = seed_indices_d.size;
		// Keep the public view handle in sync after possible reallocation.
		indices_d = seed_indices_d.values;
	}

	template<typename FloatType>
	BeamTraceState<FloatType> BeamTracer<FloatType>::TraceWindowDevice(const FloatType t_start, const FloatType t_end, device_exe* exec) {
		// Temporary state carries host seeds and metadata for this interval.
		BeamTraceState<FloatType> state;
		// Compute interval seeds before staging to device.
		TraceWindowHost(t_start, t_end, state.seed_indices_h);
		// Cache host seed count for return value.
		state.active_seed_count = static_cast<int>(state.seed_indices_h.size());
		// Publish interval seeds to reusable device storage.
		Stage(state, exec);
		return state;
	}

	template<typename FloatType>
	void BeamTracer<FloatType>::StartWorker() {
		// Guard worker launch and avoid rebuilding an already-ready index.
		{
			std::lock_guard<std::mutex> lock(mutex);
			if (worker_started || index_ready) {
				return;
			}
			worker_started = true;
			worker_done = false;
		}
		// One background worker builds the static segment index.
		worker = std::thread([this]() {
			BuildSegmentsFromCurrentThread();
		});
	}

	template<typename FloatType>
	BeamTraceState<FloatType> BeamTracer<FloatType>::FindNextTimestepState(const int requested_iteration, device_exe* exec) {
		// Return object for either the requested step or the next active step.
		BeamTraceState<FloatType> state;
		// Timestep-driven queries require a configured timestep.
		if (timestep <= static_cast<FloatType>(0.0)) {
			throw std::runtime_error("BeamTracer::FindNextTimestepState requires a positive timestep");
		}

		// Requested Interface interval is (t_start, t_end].
		const FloatType t_start = static_cast<FloatType>(requested_iteration) * timestep;
		const FloatType t_end = t_start + timestep;
		// First try to serve the requested timestep directly.
		ComputeTraceHost(t_start, t_end, true, state.seed_indices_h);
		state.active_seed_count = static_cast<int>(state.seed_indices_h.size());
		if (state.active_seed_count > 0) {
			state.iteration = requested_iteration;
			Stage(state, exec);
			return state;
		}

		// Empty requested steps jump to the next active timestep.
		const int next_step = FindNextPreAdvanceStepAfter(t_end);
		if (next_step < 0) {
			state.done = true;
			Stage(state, exec);
			return state;
		}

		// Compute seeds for the active step discovered by the index.
		const FloatType next_start = static_cast<FloatType>(next_step) * timestep;
		ComputeTraceHost(next_start, next_start + timestep, true, state.seed_indices_h);
		state.iteration = next_step;
		state.active_seed_count = static_cast<int>(state.seed_indices_h.size());
		Stage(state, exec);
		return state;
	}

	template<typename FloatType>
	int BeamTracer<FloatType>::FindNextPreAdvanceStepAfter(const FloatType time) const {
		// Search only after indexed data is available.
		WaitForIndex();
		// Nonpositive timestep means step jumps are disabled.
		if (timestep <= static_cast<FloatType>(0.0)) {
			return -1;
		}

		// Best pre-advance Interface step found so far.
		int best_step = -1;
		const std::function<void(const int)> consider_step = [&](const int step) {
			if (step < 0) {
				return;
			}
			if (best_step < 0 || step < best_step) {
				best_step = step;
			}
		};
		// Convert an active physical time into the step whose interval contains it.
		const std::function<void(const FloatType)> consider_time = [&](const FloatType event_time) {
			if (event_time <= time) {
				return;
			}
			// Step interval is (step*timestep, (step+1)*timestep].
			int step = static_cast<int>(std::ceil(event_time / timestep)) - 1;
			if (step < 0) {
				step = 0;
			}
			consider_step(step);
		};

		// Search every path's indexed static events and dynamic active windows.
		for (int path_index = 0; path_index < static_cast<int>(active_segments_by_path.size()); path_index++) {
			// Active segment ids for this path.
			const vector<int>& active = active_segments_by_path[static_cast<size_t>(path_index)];
			// Static event times aligned with active segment ids.
			const vector<FloatType>& times = active_event_times_by_path[static_cast<size_t>(path_index)];

			// First static segment event after the current time.
			typename vector<FloatType>::const_iterator event_it = std::upper_bound(times.begin(), times.end(), time);
			if (event_it != times.end()) {
				consider_time(*event_it);
			}

			// Dynamic current-position seeds can exist before a segment's static event.
			for (const int segment_index : active) {
				// Active segment metadata used to find the next in-window step.
				const BeamTraceSegment<FloatType>& entry = segments[static_cast<size_t>(segment_index)];
				if (entry.leave_time <= time) {
					continue;
				}
				// Candidate dynamic step is the first timestep interval that
				// overlaps the active window after the already-empty query.
				const FloatType active_time = std::max(entry.enter_time, time);
				const int dynamic_step = first_preadvance_step_for_active_time(active_time, timestep);
				const FloatType dynamic_start = static_cast<FloatType>(dynamic_step) * timestep;
				const FloatType dynamic_end = dynamic_start + timestep;
				if (dynamic_end > time && dynamic_start < entry.leave_time) {
					consider_step(dynamic_step);
				}
			}
		}

		// Return -1 when no future active static or dynamic event exists.
		return best_step;
	}

	template<typename FloatType>
	void BeamTracer<FloatType>::Step(const int requested_iteration, device_exe* exec) {
		
		// Only trace when the run loop reaches the next known active iteration.
		if (done || requested_iteration < next_iter) {
			return;
		}

		// Query either this step or the next active step.
		BeamTraceState<FloatType> state = FindNextTimestepState(requested_iteration, exec);
		// Remember the run loop iteration that triggered the query.
		iter = requested_iteration;

		// Done states clear staged seeds and push next_iter past the scan.
		if (state.done) {
			done = requested_iteration >= final_iter;
			next_iter = final_iter + 1;
			Clear(exec);
			return;
		}
		// Active current step keeps the loop marching one step at a time.
		if (state.iteration == requested_iteration) {
			next_iter = requested_iteration + 1;
			done = false;
			return;
		}

		// Future active step means callers should jump there before advancing.
		Clear(exec);
		next_iter = state.iteration;
		done = false;
	}

	template<typename FloatType>
	void BeamTracer<FloatType>::RefreshForTimestep(const Simdat<FloatType>& sim_in, const FloatType t_start, const FloatType timestep_in, device_exe* exec) {
		// Rebuild the index only when the owning Simdat, timestep, or trace gates change.
		if (&sim_in != sim || timestep_in != timestep || sim_in.settings.radius_check != radius_check || sim_in.settings.trace_radius != trace_radius) {
			Stop();
			// Store new simulation and timestep inputs.
			sim = &sim_in;
			radius_check = sim_in.settings.radius_check;
			trace_radius = sim_in.settings.trace_radius;
			timestep = timestep_in;
			// Recompute final iteration for timestep-driven callers.
			final_iter = (timestep > static_cast<FloatType>(0.0))
				? std::max(0, static_cast<int>(std::ceil(sim->util.allScansEndTime / timestep)))
				: 0;
			// Reset step state because the timestep basis changed.
			next_iter = 0;
			done = false;
			// Clear index readiness before launching the new worker.
			{
				std::lock_guard<std::mutex> lock(mutex);
				index_ready = false;
				worker_done = false;
				worker_started = false;
			}
			StartWorker();
		}
		// RefreshForTimestep is only valid for positive-timestep tracing.
		if (timestep <= static_cast<FloatType>(0.0)) {
			throw std::runtime_error("BeamTracer::RefreshForTimestep requires a positive timestep");
		}
		// Convert physical time to the pre-advance Interface iteration.
		Step(std::max(0, static_cast<int>(std::floor(t_start / timestep))), exec);
	}

	template<typename FloatType>
	void BeamTracer<FloatType>::Clear(device_exe* exec) {
		// Empty state clears host/device seed buffers while preserving capacity.
		BeamTraceState<FloatType> empty;
		Stage(empty, exec);
	}

	template<typename FloatType>
	void BeamTracer<FloatType>::Stop() {
		// Join the index worker if it is still running.
		if (worker.joinable()) {
			worker.join();
		}
		// Mark worker state complete for any later readiness checks.
		std::lock_guard<std::mutex> lock(mutex);
		worker_started = false;
		worker_done = true;
		cv.notify_all();
	}

	template<typename FloatType>
	void beam_trace_surface_mask(const Simdat<FloatType>& sim, const FloatType radius_check, const FloatType t_start, const FloatType t_end, int& numTrace, int_2D_hostView& beamTrace_host) {
		// Temporary tracer provides the interval query implementation.
		BeamTracer<FloatType> tracker(sim);
		// Override radius_check for legacy helper callers.
		tracker.radius_check = radius_check;
		// Ensure initial async build finishes before rebuilding with the override.
		tracker.Stop();
		tracker.BuildSegments();

		// Host index list for the requested interval.
		vector<int> indices;
		tracker.TraceWindowHost(t_start, t_end, indices);
		// Return the number of unique traced columns.
		numTrace = static_cast<int>(indices.size());

		// Convert the flat index list into the caller's 2D host mask.
		Kokkos::deep_copy(beamTrace_host, 0);
		for (const int flat : indices) {
			beamTrace_host(flat / sim.domain.ynum, flat % sim.domain.ynum) = 1;
		}
	}

	template<typename FloatType>
	void beam_trace_surface_indices(const Simdat<FloatType>& sim, const FloatType radius_check, const FloatType t_start, const FloatType t_end, vector<int>& beamTrace_indices, int_2D_hostView& beamTrace_host) {
		// Temporary tracer provides the interval query implementation.
		BeamTracer<FloatType> tracker(sim);
		// Override radius_check for legacy helper callers.
		tracker.radius_check = radius_check;
		// Ensure initial async build finishes before rebuilding with the override.
		tracker.Stop();
		tracker.BuildSegments();

		// Fill the caller-owned flat index list.
		tracker.TraceWindowHost(t_start, t_end, beamTrace_indices);
		// Mirror the flat index list into the caller's 2D host mask.
		Kokkos::deep_copy(beamTrace_host, 0);
		for (const int flat : beamTrace_indices) {
			beamTrace_host(flat / sim.domain.ynum, flat % sim.domain.ynum) = 1;
		}
	}

	template void beam_trace_surface_mask(const Simdat<float>&, const float, const float, const float, int&, int_2D_hostView&);
	template void beam_trace_surface_mask(const Simdat<double>&, const double, const double, const double, int&, int_2D_hostView&);
	template void beam_trace_surface_indices(const Simdat<float>&, const float, const float, const float, vector<int>&, int_2D_hostView&);
	template void beam_trace_surface_indices(const Simdat<double>&, const double, const double, const double, vector<int>&, int_2D_hostView&);
	template struct BeamTraceState<float>;
	template struct BeamTraceState<double>;
	template struct BeamTraceDomainWindow<float>;
	template struct BeamTraceDomainWindow<double>;
	template struct BeamTraceSegment<float>;
	template struct BeamTraceSegment<double>;
	template struct BeamTracer<float>;
	template struct BeamTracer<double>;
}
