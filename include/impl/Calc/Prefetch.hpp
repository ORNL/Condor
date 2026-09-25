#pragma once

// Internal includes
#include "impl/Calc/Quad.hpp"
#include "impl/Calc/Structs.hpp"
#include "impl/Structs/Simdat.hpp"
#include "impl/Utility/Nodes.hpp"
#include "impl/Utility/Util.hpp"

// External includes
#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace Condor::impl::Calc {

    template<typename FloatType>
    struct AdaptiveQuadratureSlot {
        int itert = 0;
        FloatType t = static_cast<FloatType>(0.0);
        bool ready = false;
        bool in_use = false;
        Calc::Device_Nodes<FloatType> device_nodes;
        device_exe exec;
    };

    template<typename FloatType>
    class AdaptiveQuadraturePrefetcher {
    public:
        using TimeIteration = std::pair<FloatType, int>;

    private:
        const Simdat<FloatType>& sim;
        const int thread_count;

        mutable std::mutex mutex;
        std::condition_variable cv;
        std::vector<std::thread> workers;
        std::vector<AdaptiveQuadratureSlot<FloatType>> slots;
        std::vector<int> free_slots;

        std::vector<TimeIteration> requests;
        std::size_t next_request = 0;
        std::size_t next_acquire = 0;

        bool running = false;
        bool stopping = false;
        bool uniform = false;
        FloatType uniform_dt = static_cast<FloatType>(0.0);
        FloatType next_uniform_t = static_cast<FloatType>(0.0);
        int next_uniform_itert = 1;
        int next_uniform_acquire = 1;

        void ResetSlotsLocked() {
            free_slots.clear();
            for (int i = static_cast<int>(slots.size()) - 1; i >= 0; --i) {
                AdaptiveQuadratureSlot<FloatType>& slot = slots[static_cast<std::size_t>(i)];
                slot.itert = 0;
                slot.t = static_cast<FloatType>(0.0);
                slot.ready = false;
                slot.in_use = false;
                free_slots.push_back(i);
            }
        }

        int FindSlotLocked(const int itert, const bool must_be_ready) const {
            for (int i = 0; i < static_cast<int>(slots.size()); ++i) {
                const AdaptiveQuadratureSlot<FloatType>& slot = slots[static_cast<std::size_t>(i)];
                if (slot.in_use && slot.itert == itert && (!must_be_ready || slot.ready)) {
                    return i;
                }
            }
            return -1;
        }

        bool CanStillProduceLocked(const int itert) const {
            if (FindSlotLocked(itert, false) >= 0) {
                return true;
            }

            for (std::size_t i = next_request; i < requests.size(); ++i) {
                if (requests[i].second == itert) {
                    return true;
                }
            }

            return uniform && itert >= next_uniform_itert;
        }

        bool ClaimWork(int& slot_index, int& itert, FloatType& t) {
            std::unique_lock<std::mutex> lock(mutex);
            cv.wait(lock, [&] {
                return stopping || !running || (!uniform && next_request >= requests.size()) || !free_slots.empty();
            });

            if (stopping || !running || (!uniform && next_request >= requests.size()) || free_slots.empty()) {
                return false;
            }

            if (next_request < requests.size()) {
                t = requests[next_request].first;
                itert = requests[next_request].second;
                ++next_request;
            }
            else if (uniform) {
                t = next_uniform_t;
                itert = next_uniform_itert;
                next_uniform_t += uniform_dt;
                ++next_uniform_itert;
            }
            else {
                return false;
            }

            slot_index = free_slots.back();
            free_slots.pop_back();

            AdaptiveQuadratureSlot<FloatType>& slot = slots[static_cast<std::size_t>(slot_index)];
            slot.itert = itert;
            slot.t = t;
            slot.ready = false;
            slot.in_use = true;
            return true;
        }

        void Worker() {
            std::vector<int> start_seg;
            Nodes<FloatType> host_nodes;

            while (true) {
                int slot_index = -1;
                int itert = 0;
                FloatType t = static_cast<FloatType>(0.0);
                if (!ClaimWork(slot_index, itert, t)) {
                    return;
                }

                Util::ClearNodes(host_nodes);
                Calc::Integrate_Serial(host_nodes, start_seg, sim, t, false);

                AdaptiveQuadratureSlot<FloatType>& slot = slots[static_cast<std::size_t>(slot_index)];
                slot.device_nodes.CopyToDevice(host_nodes, &slot.exec);
                slot.exec.fence();

                {
                    std::lock_guard<std::mutex> lock(mutex);
                    if (!stopping && running) {
                        slot.ready = true;
                    }
                }
                cv.notify_all();
            }
        }

        void StartPoolLocked() {
            if (running) {
                throw std::runtime_error("AdaptiveQuadraturePrefetcher::Start called while already running");
            }

            ResetSlotsLocked();
            running = true;
            stopping = false;
            workers.reserve(static_cast<std::size_t>(thread_count));
            for (int i = 0; i < thread_count; ++i) {
                workers.emplace_back(&AdaptiveQuadraturePrefetcher::Worker, this);
            }
        }

        int AcquireExpected(const int itert) {
            std::unique_lock<std::mutex> lock(mutex);
            cv.wait(lock, [&] {
                return stopping || !running || FindSlotLocked(itert, true) >= 0 || !CanStillProduceLocked(itert);
            });

            const int slot_index = FindSlotLocked(itert, true);
            if (slot_index < 0) {
                return -1;
            }

            slots[static_cast<std::size_t>(slot_index)].ready = false;
            return slot_index;
        }

    public:
        explicit AdaptiveQuadraturePrefetcher(const Simdat<FloatType>& sim_in)
            : sim(sim_in),
              thread_count(std::max(sim_in.settings.prefetch_threads, 1))
        {
            // Add two retained slots for modes that hold current and previous quadrature.
            const int slot_count = std::max(thread_count + 2, 2);
            slots.resize(static_cast<std::size_t>(slot_count));
            free_slots.reserve(static_cast<std::size_t>(slot_count));
            ResetSlotsLocked();
        }

        AdaptiveQuadraturePrefetcher(const Simdat<FloatType>& sim_in, const FloatType timestep)
            : AdaptiveQuadraturePrefetcher(sim_in)
        {
            Start(timestep);
        }

        AdaptiveQuadraturePrefetcher(const Simdat<FloatType>& sim_in, const std::vector<FloatType>& times)
            : AdaptiveQuadraturePrefetcher(sim_in)
        {
            Start(times);
        }

        AdaptiveQuadraturePrefetcher(const AdaptiveQuadraturePrefetcher&) = delete;
        AdaptiveQuadraturePrefetcher& operator=(const AdaptiveQuadraturePrefetcher&) = delete;

        ~AdaptiveQuadraturePrefetcher() {
            Stop();
        }

        void Start(const std::vector<TimeIteration>& time_iterations) {
            std::lock_guard<std::mutex> lock(mutex);
            requests = time_iterations;
            next_request = 0;
            next_acquire = 0;
            uniform = false;
            uniform_dt = static_cast<FloatType>(0.0);
            next_uniform_t = static_cast<FloatType>(0.0);
            next_uniform_itert = 1;
            next_uniform_acquire = 1;
            StartPoolLocked();
        }

        void Start(const std::vector<FloatType>& times, const int first_itert = 1) {
            std::vector<TimeIteration> time_iterations;
            time_iterations.reserve(times.size());
            for (std::size_t i = 0; i < times.size(); ++i) {
                time_iterations.emplace_back(times[i], first_itert + static_cast<int>(i));
            }
            Start(time_iterations);
        }

        void Start(const FloatType timestep) {
            Start(timestep, 1, timestep);
        }

        void Start(const FloatType timestep, const int first_itert, const FloatType first_t) {
            std::lock_guard<std::mutex> lock(mutex);
            requests.clear();
            next_request = 0;
            next_acquire = 0;
            uniform = true;
            uniform_dt = timestep;
            next_uniform_t = first_t;
            next_uniform_itert = first_itert;
            next_uniform_acquire = first_itert;
            StartPoolLocked();
        }

        void Stop() {
            {
                std::lock_guard<std::mutex> lock(mutex);
                if (!running && workers.empty()) {
                    return;
                }
                stopping = true;
                running = false;
            }

            cv.notify_all();
            for (std::thread& worker : workers) {
                if (worker.joinable()) {
                    worker.join();
                }
            }

            std::lock_guard<std::mutex> lock(mutex);
            workers.clear();
            requests.clear();
            next_request = 0;
            next_acquire = 0;
            stopping = false;
            uniform = false;
            uniform_dt = static_cast<FloatType>(0.0);
            next_uniform_t = static_cast<FloatType>(0.0);
            next_uniform_itert = 1;
            next_uniform_acquire = 1;
            ResetSlotsLocked();
        }

        void Reset(const std::vector<TimeIteration>& time_iterations) {
            Stop();
            Start(time_iterations);
        }

        void Reset(const std::vector<FloatType>& times, const int first_itert = 1) {
            Stop();
            Start(times, first_itert);
        }

        void Reset(const FloatType timestep, const int first_itert, const FloatType first_t) {
            Stop();
            Start(timestep, first_itert, first_t);
        }

        bool IsRunning() const {
            std::lock_guard<std::mutex> lock(mutex);
            return running;
        }

        int Acquire(const int itert) {
            return AcquireExpected(itert);
        }

        int AcquireNext() {
            int itert = 0;
            {
                std::lock_guard<std::mutex> lock(mutex);
                if (uniform) {
                    itert = next_uniform_acquire;
                }
                else {
                    if (next_acquire >= requests.size()) {
                        return -1;
                    }
                    itert = requests[next_acquire].second;
                }
            }

            const int slot_index = AcquireExpected(itert);
            if (slot_index < 0) {
                return -1;
            }

            std::lock_guard<std::mutex> lock(mutex);
            if (uniform) {
                ++next_uniform_acquire;
            }
            else {
                ++next_acquire;
            }
            return slot_index;
        }

        AdaptiveQuadratureSlot<FloatType>& Slot(const int slot_index) {
            return slots.at(static_cast<std::size_t>(slot_index));
        }

        const AdaptiveQuadratureSlot<FloatType>& Slot(const int slot_index) const {
            return slots.at(static_cast<std::size_t>(slot_index));
        }

        void Release(const int slot_index) {
            if (slot_index < 0) {
                return;
            }

            std::lock_guard<std::mutex> lock(mutex);
            if (slot_index >= static_cast<int>(slots.size())) {
                return;
            }

            AdaptiveQuadratureSlot<FloatType>& slot = slots[static_cast<std::size_t>(slot_index)];
            if (!slot.in_use) {
                return;
            }

            slot.itert = 0;
            slot.t = static_cast<FloatType>(0.0);
            slot.ready = false;
            slot.in_use = false;
            free_slots.push_back(slot_index);
            cv.notify_all();
        }
    };
}
