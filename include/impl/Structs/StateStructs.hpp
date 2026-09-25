#pragma once

// Internal includes
#include "Definitions.hpp"

namespace Condor::impl{
    // Own iteration and time logic
    template<typename FloatType>
    struct Step_State {
        // Sometimes used variables
        FloatType dt;
        int outFreq;

        // Always used variables
        FloatType t_prev, t_cur;
        int iter_prev, iter_cur;
        
        // Initialization
        Step_State(const FloatType timestep = static_cast<FloatType>(-1.0), const int outputFrequency = 0){
            // Set always used variables
            dt = timestep;
            outFreq = outputFrequency;

            // Set sometimes used variables
            iter_prev = 0;
            iter_cur = 0;
            t_prev = static_cast<FloatType>(0.0);
            t_cur = static_cast<FloatType>(0.0);  
        }

        // Advance just time and iteration
        void Advance_Iteration(){
            iter_prev = iter_cur;
            iter_cur++;
            t_prev = t_cur;
            t_cur += dt;         
        }   

        void Reset(const int iter){
            iter_prev = iter-1;
            iter_cur = iter;
            t_prev = iter_prev*dt;
            t_cur = iter_cur*dt;
        }

        bool isOutputStep() const {
            return ((outFreq > 0) && !(iter_cur % outFreq));
        }
    };

    // Keep prefetched quadrature slots in one small owner with explicit current/previous access.
    template<typename FloatType>
    struct Quad_State {
        using TimeIteration = typename Calc::AdaptiveQuadraturePrefetcher<FloatType>::TimeIteration;

        int cur_slot = -1; // Current timestep slot index consumed by Interface kernels.
        int prev_slot = -1; // Previous timestep slot index retained for deferred reconstruction.
        Calc::AdaptiveQuadraturePrefetcher<FloatType> prefetcher; // Shared worker pool that prepares future quadrature slots.

        // Build the stopped prefetcher once so Interface can start, stop, or reset it explicitly later.
        Quad_State(const Simdat<FloatType>& sim, const FloatType)
            : prefetcher(sim)
        {}

        // Start uniform prefetching from timestep one using the mode timestep as the first time value.
        void Start(const FloatType timestep) {
            cur_slot = -1;
            prev_slot = -1;
            prefetcher.Start(timestep);
        }

        // Start uniform prefetching from one arbitrary timestep and physical time.
        void Start(const FloatType timestep, const int first_itert, const FloatType first_t) {
            cur_slot = -1;
            prev_slot = -1;
            prefetcher.Start(timestep, first_itert, first_t);
        }

        // Start prefetching from one explicit ordered list of physical times.
        void Start(const std::vector<FloatType>& times, const int first_itert = 1) {
            cur_slot = -1;
            prev_slot = -1;
            prefetcher.Start(times, first_itert);
        }

        // Start prefetching from one explicit ordered list of [time, iteration] requests.
        void Start(const std::vector<TimeIteration>& time_iterations) {
            cur_slot = -1;
            prev_slot = -1;
            prefetcher.Start(time_iterations);
        }

        // Stop the worker pool after returning any retained current or previous slots.
        void Stop() {
            if (cur_slot >= 0) {
                prefetcher.Release(cur_slot);
                cur_slot = -1;
            }
            if (prev_slot >= 0) {
                prefetcher.Release(prev_slot);
                prev_slot = -1;
            }
            prefetcher.Stop();
        }

        // Restart uniform prefetching from one arbitrary timestep and physical time.
        void Reset(const FloatType timestep, const int first_itert, const FloatType first_t) {
            Stop();
            Start(timestep, first_itert, first_t);
        }

        // Restart uniform prefetching from one arbitrary timestep and physical time.
        void Reset(const FloatType timestep, const int first_itert) {
            Reset(timestep, first_itert, timestep*first_itert);
        }

        // Restart explicit-time prefetching from one new ordered list of physical times.
        void Reset(const std::vector<FloatType>& times, const int first_itert = 1) {
            Stop();
            Start(times, first_itert);
        }

        // Restart explicit [time, iteration] prefetching from one new ordered request list.
        void Reset(const std::vector<TimeIteration>& time_iterations) {
            Stop();
            Start(time_iterations);
        }

        // Acquire one prefetched timestep by its iteration.
        void Acquire(const int itert) {
            cur_slot = prefetcher.Acquire(itert);
            if (cur_slot < 0) {
                throw std::runtime_error("Interface prefetch state could not acquire the requested slot");
            }
        }

        // Rotate the slot indices and acquire the next prefetched timestep in order.
        void Advance() {
            if (prev_slot >= 0) {
                prefetcher.Release(prev_slot);
            }

            prev_slot = cur_slot;
            cur_slot = prefetcher.AcquireNext();
            if (cur_slot < 0) {
                throw std::runtime_error("Interface prefetch state could not acquire the next slot");
            }
        }

        // Rotate the slot indices and acquire one explicit timestep.
        void Advance(const int itert) {
            if (prev_slot >= 0) {
                prefetcher.Release(prev_slot);
            }

            prev_slot = cur_slot;
            Acquire(itert);
        }

        // Report whether the current timestep slot is available.
        bool HasCurrent() const {
            return cur_slot >= 0;
        }

        // Report whether the previous timestep slot is available.
        bool HasPrevious() const {
            return prev_slot >= 0;
        }

        // Expose the current timestep slot without leaking ownership outside this wrapper.
        Calc::AdaptiveQuadratureSlot<FloatType>& CurrentSlot() {
            if (cur_slot < 0) {
                throw std::runtime_error("Interface current slot is not available");
            }
            return prefetcher.Slot(cur_slot);
        }

        // Expose the current timestep slot without leaking ownership outside this wrapper.
        const Calc::AdaptiveQuadratureSlot<FloatType>& CurrentSlot() const {
            if (cur_slot < 0) {
                throw std::runtime_error("Interface current slot is not available");
            }
            return prefetcher.Slot(cur_slot);
        }

        // Expose the previous timestep slot without leaking ownership outside this wrapper.
        Calc::AdaptiveQuadratureSlot<FloatType>& PreviousSlot() {
            if (prev_slot < 0) {
                throw std::runtime_error("Interface previous slot is not available");
            }
            return prefetcher.Slot(prev_slot);
        }

        // Expose the previous timestep slot without leaking ownership outside this wrapper.
        const Calc::AdaptiveQuadratureSlot<FloatType>& PreviousSlot() const {
            if (prev_slot < 0) {
                throw std::runtime_error("Interface previous slot is not available");
            }
            return prefetcher.Slot(prev_slot);
        }
    };

    // Own current/previous temperature fields plus their calculation-iteration markers.
    template<typename FloatType>
    struct Thermal_State {
        using float_device_view_t = Kokkos::View<FloatType*, layout, device_memory>;

        float_device_view_t T_cur; // Current-step temperatures used by Interface and deferred hooks.
        float_device_view_t T_prev; // Previous-step temperatures reused for reconstruction across one timestep boundary.
        int_deviceView T_calc_iter_cur; // Iteration stamp that recorded when each current-step temperature was computed.
        int_deviceView T_calc_iter_prev; // Iteration stamp that recorded when each previous-step temperature was computed.
        bool_deviceView isLiq_cur;  // If current points are liquid
        
        // Allocate the necessary views
        Thermal_State(const Simdat<FloatType>& sim)
            : T_cur(Kokkos::ViewAllocateWithoutInitializing("Interface_T_cur"), sim.domain.pnum),
                T_prev(Kokkos::ViewAllocateWithoutInitializing("Interface_T_prev"), sim.domain.pnum),
                T_calc_iter_cur(Kokkos::ViewAllocateWithoutInitializing("Interface_T_calc_iter_cur"), sim.domain.pnum),
                T_calc_iter_prev(Kokkos::ViewAllocateWithoutInitializing("Interface_T_calc_iter_prev"), sim.domain.pnum),
                isLiq_cur(Kokkos::ViewAllocateWithoutInitializing("Interface_isLiq_cur"), sim.domain.pnum)
        {
            // Seed the temperature fields and iteration markers before any timestep work begins.
            Kokkos::deep_copy(T_cur, sim.material.T_init);
            Kokkos::deep_copy(T_prev, sim.material.T_init);
            Kokkos::deep_copy(T_calc_iter_cur, 0);
            Kokkos::deep_copy(T_calc_iter_prev, 0);
            Kokkos::deep_copy(isLiq_cur, false);
        }

        // Rotate the current/previous handles so Interface preserves one-step history without copying.
        void Advance() {
            float_device_view_t T_tmp = T_prev; // Temporary handle used to swap the temperature views.
            T_prev = T_cur;
            T_cur = T_tmp;

            int_deviceView T_calc_iter_tmp = T_calc_iter_prev; // Temporary handle used to swap the iteration markers.
            T_calc_iter_prev = T_calc_iter_cur;
            T_calc_iter_cur = T_calc_iter_tmp;
        }

        // TODO::MAY BE IMPORTANT
        // Restart from a new iteration
        void Reset(const int itert){
            return;
        }
    };
}
