#pragma once

// Internal includes
#include "Definitions.hpp"
#include "impl/Structs/Simdat.hpp"

// External includes
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

namespace Condor::impl::Meltpool{
	template<typename T>
	using vector = std::vector<T>;

	// Fill a reusable 2D host mask with the unique surface columns touched by the beam trace.
	template<typename FloatType>
	void beam_trace_surface_mask(const Simdat<FloatType>&, const FloatType, const FloatType, const FloatType, int&, int_2D_hostView&);

	// Flatten one timestep's unique surface seeds into reusable host indices.
	template<typename FloatType>
	void beam_trace_surface_indices(const Simdat<FloatType>&, const FloatType, const FloatType, const FloatType, vector<int>&, int_2D_hostView&);

	template<typename FloatType>
	struct BeamTraceState {
		// Host-side seed list computed for one interval or timestep query.
		vector<int> seed_indices_h;
		// Interface iteration represented by this seed list.
		int iteration = -1;
		// Cached seed count used by callers before staging to device.
		int active_seed_count = 0;
		// Marks that no future active beam trace interval exists.
		bool done = false;
	};

	struct TransferSeedIndices {
		// Host staging buffer used for one compact seed transfer.
		int_hostView values;
		// Number of valid entries currently staged.
		int size = 0;
		// Allocated host staging capacity.
		int capacity = 0;

		// Allocate the host transfer buffer with at least one entry.
		explicit TransferSeedIndices(const int initial_capacity = 1);
		// Copy a std::vector seed list into the reusable host view.
		void CopyFromVector(const vector<int>& source);
	};

	struct DeviceSeedIndices {
		// Device buffer consumed by meltpool surface seeding.
		int_deviceView values;
		// Number of valid device entries currently available.
		int size = 0;
		// Allocated device buffer capacity.
		int capacity = 0;

		// Allocate the device seed buffer with at least one entry.
		explicit DeviceSeedIndices(const int initial_capacity = 1);
		// Copy the staged host seed list to the device buffer.
		void CopyFromHost(const TransferSeedIndices& host_values, device_exe* exec = nullptr);
	};

	template<typename FloatType>
	struct BeamTraceDomainWindow {
		// Beam/path index that owns this in-domain interval.
		int path_index = 0;
		// Segment index within the owning path.
		int segment_index = 0;
		// First physical time where the segment can affect the domain.
		FloatType enter_time = static_cast<FloatType>(0.0);
		// Last physical time where the segment can affect the domain.
		FloatType leave_time = static_cast<FloatType>(0.0);
	};

	template<typename FloatType>
	struct BeamTraceSegment {
		// Beam/path index used to recover the source segment.
		int path_index = 0;
		// Segment index within the source path.
		int segment_index = 0;
		// Segment mode: spot or line.
		int smode = 0;
		// Segment start/end times from the input path.
		FloatType t0 = static_cast<FloatType>(0.0);
		FloatType t1 = static_cast<FloatType>(0.0);
		// Clipped in-domain time window expanded by radius_check.
		FloatType enter_time = static_cast<FloatType>(0.0);
		FloatType leave_time = static_cast<FloatType>(0.0);
		// Time used for binary-search interval inclusion.
		FloatType event_time = static_cast<FloatType>(0.0);
		// Whether this segment ever affects the tracked domain.
		bool active = false;
	};

	template<typename FloatType>
	struct BeamTracer {
		// Simulation inputs used for path geometry and grid dimensions.
		const Simdat<FloatType>* sim = nullptr;
		// Distance around the domain used to decide relevant beam segments.
		FloatType radius_check = static_cast<FloatType>(0.0);
		// Radius used to add surface seeds along active beam traces.
		FloatType trace_radius = -static_cast<FloatType>(1.0);
		// Interface timestep used only by Step/NextIteration.
		FloatType timestep = static_cast<FloatType>(0.0);
		// Last possible timestep index for the scan duration.
		int final_iter = 0;

		// Full segment cache, including inactive segments for diagnostics.
		vector<BeamTraceSegment<FloatType>> segments;
		// Public list of active clipped domain windows.
		vector<BeamTraceDomainWindow<FloatType>> domain_windows;
		// Per-path active segment indices into segments.
		vector<vector<int>> active_segments_by_path;
		// Per-path active event times aligned with active_segments_by_path.
		vector<vector<FloatType>> active_event_times_by_path;
		// Reusable host transfer staging for the current seed list.
		TransferSeedIndices transfer_h;
		// Reusable device storage for the current seed list.
		DeviceSeedIndices seed_indices_d;

		// Last requested Interface iteration.
		int iter = -1;
		// Next Interface iteration that may contain active seeds.
		int next_iter = 0;
		// Current number of staged device seeds.
		int num_indices = 0;
		// Device view handle exposed to the meltpool tracker.
		int_deviceView indices_d;
		// Marks that the BeamTracer has no further active work.
		bool done = false;

		// Construct a tracer without timestep-driven stepping.
		BeamTracer(const Simdat<FloatType>& sim_in);
		// Construct a tracer that can serve Interface timestep queries.
		BeamTracer(const Simdat<FloatType>& sim_in, const FloatType timestep_in);
		// Join the preprocessing worker before destruction.
		~BeamTracer();

		// Compute host surface seeds for interval (t_start, t_end].
		void TraceWindowHost(const FloatType t_start, const FloatType t_end, vector<int>& seed_indices_h) const;
		// Compute interval seeds and stage them to device.
		BeamTraceState<FloatType> TraceWindowDevice(const FloatType t_start, const FloatType t_end, device_exe* exec = nullptr);
		// Compatibility overload that ignores the redundant Simdat reference.
		BeamTraceState<FloatType> TraceWindowDevice(const Simdat<FloatType>&, const FloatType t_start, const FloatType t_end, device_exe* exec = nullptr) {
			return TraceWindowDevice(t_start, t_end, exec);
		}
		// Compute interval seeds from scan start to t_end.
		BeamTraceState<FloatType> TraceFromStartDevice(const FloatType t_end, device_exe* exec = nullptr) {
			return TraceWindowDevice(static_cast<FloatType>(0.0), t_end, exec);
		}

		// Stage seeds for one Interface step or jump to the next active step.
		void Step(const int requested_iteration, device_exe* exec = nullptr);
		// Rebind the tracer when the timestep or Simdat object changes.
		void RefreshForTimestep(const Simdat<FloatType>& sim_in, const FloatType t_start, const FloatType timestep_in, device_exe* exec = nullptr);
		// Return seed state for requested_iteration's interval, or for the next active pre-advance step.
		BeamTraceState<FloatType> FindNextTimestepState(const int requested_iteration, device_exe* exec = nullptr);

		// Expose active domain windows for diagnostics and snapshot helpers.
		const vector<BeamTraceDomainWindow<FloatType>>& DomainWindows() const {
			return domain_windows;
		}
		// Report the next pre-advance Interface iteration the run loop should consider.
		int NextIteration() const {
			return next_iter;
		}
		// Report whether the current device seed list is nonempty.
		bool HasActiveSeeds() const {
			return num_indices > 0;
		}
		// Return the current staged seed count.
		int ActiveSeedCount() const {
			return num_indices;
		}
		// Return the current staged device seed view.
		const int_deviceView& ActiveSeedIndices() const {
			return indices_d;
		}
		// Compatibility hook used by callers that treat active seeds as continuous.
		bool Continuous() const {
			return HasActiveSeeds();
		}
		// Clear the current seed list on host and device.
		void Clear(device_exe* exec = nullptr);
		// Join any in-flight index build worker.
		void Stop();

		// Rebuild the host-side segment cache after changing trace settings.
		void BuildSegments();

		private:
		// Synchronizes async segment-index construction.
		mutable std::mutex mutex;
		// Wakes query calls once the segment index is ready.
		mutable std::condition_variable cv;
		// Single preprocessing worker used to build the segment index.
		std::thread worker;
		// True while the preprocessing worker has been launched.
		bool worker_started = false;
		// True after the worker has finished its build attempt.
		bool worker_done = false;
		// True once indexed segment data can answer queries.
		bool index_ready = false;

		// Start the async index build if it is not already available.
		void StartWorker();
		// Block until indexed segment data can be read.
		void WaitForIndex() const;
		// Build the segment index on the calling thread.
		void BuildSegmentsFromCurrentThread();
		// Shared host query for interval and timestep seed generation.
		void ComputeTraceHost(const FloatType t_start, const FloatType t_end, const bool include_current_at_end, vector<int>& seed_indices_h) const;
		// Find the active source segment containing one physical time.
		int FindActiveSegmentAtTime(const int path_index, const FloatType time) const;
		// Find the next pre-advance Interface step with static or dynamic beam seeds.
		int FindNextPreAdvanceStepAfter(const FloatType time) const;
		// Stage a host seed state to the reusable host/device buffers.
		void Stage(const BeamTraceState<FloatType>& state, device_exe* exec = nullptr);
	};

}
