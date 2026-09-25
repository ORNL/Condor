#pragma once

#include "impl/Meltpool/Beam.hpp"
#include "impl/Meltpool/Dimensions.hpp"
#include "impl/Meltpool/Tracking.hpp"
#include "impl/RunModes/Calibrate/Evaluators/Base.hpp"
#include "impl/RunModes/Snapshots/Config.hpp"
#include "impl/RunModes/Snapshots/Structs.hpp"
#include "impl/Structs/Simdat.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace Condor::impl::Modes::Calibrate {

    template<typename FloatType>
    class Path_Evaluator final : public Evaluator<FloatType> {
        public:
            using Base = Evaluator<FloatType>;
            using Shared_Config = typename Base::Shared_Config;

            struct Config {
                std::vector<FloatType> times;
                std::vector<FloatType> time_weights;
                std::vector<Dimension_Target<FloatType>> targets;
            };

            Path_Evaluator(
                const Simdat<FloatType>& base,
                const Shared_Config& shared_config,
                const json& config_node)
                : Base(base, shared_config),
                  config(ReadConfig(config_node))
            {
                Validate();
            }

        private:
            Config config;

            Evaluator_Measurements<FloatType> Measure(const Simdat<FloatType>& candidate) const override {
                Snapshots::Snapshots_Config<FloatType> snapshot_config;
                snapshot_config.settings.times = config.times;
                snapshot_config.settings.time_iterations.reserve(config.times.size());
                for (std::size_t i = 0; i < config.times.size(); i++) {
                    snapshot_config.settings.time_iterations.emplace_back(config.times[i], static_cast<int>(i) + 1);
                }
                snapshot_config.settings.fallback_timestep = config.times.empty()
                    ? static_cast<FloatType>(1.0)
                    : std::max(config.times.front(), static_cast<FloatType>(1.0e-9));
                snapshot_config.settings.tracking = Snapshots::Tracking_Mode::Interface;

                Snapshots::Snapshots_State<FloatType> state(candidate, snapshot_config);
                Meltpool::BeamTracer<FloatType> beam_trace(candidate, snapshot_config.settings.fallback_timestep);
                beam_trace.Stop();
                Meltpool::Tracking<FloatType> meltpool(candidate);
                meltpool.Initialize_Solidification();

                Meltpool::Dimensions<FloatType> dimensions(candidate);
                dimensions.calc.MP_Len_or_Wid = true;
                dimensions.calc.MP_Depth = candidate.domain.znum > 1;
                dimensions.Initialize();

                Evaluator_Measurements<FloatType> measurements;
                measurements.samples.resize(config.times.size());
                Boundary_Check boundary_check;
                for (std::size_t snapshot = 0; snapshot < config.times.size(); snapshot++) {
                    const FloatType t_prev = (snapshot == 0)
                        ? static_cast<FloatType>(0.0)
                        : config.times[snapshot - 1];
                    const FloatType t_cur = config.times[snapshot];

                    state.Advance_To(static_cast<int>(snapshot), t_prev, t_cur);
                    beam_trace.TraceWindowDevice(t_prev, t_cur);
                    meltpool.Advance(state, beam_trace);

                    if (Base::HitsBoundary(candidate, meltpool.ActiveColumnCount(), meltpool.ActiveColumns(), meltpool.CurrentDepth(), boundary_check)) {
                        measurements.boundary_hit = true;
                        break;
                    }

                    dimensions.Measure(meltpool, state.thermal, candidate, state.step);
                    Base::UpdateBestDimensions(dimensions, measurements.samples[snapshot]);
                    if (!measurements.samples[snapshot].has_meltpool) {
                        measurements.no_meltpool = true;
                        break;
                    }
                }

                meltpool.Finalize();
                state.Stop_All();
                return measurements;
            }

            FloatType Loss(const Evaluator_Measurements<FloatType>& measurements) const override {
                FloatType loss = static_cast<FloatType>(0.0);
                for (std::size_t i = 0; i < measurements.samples.size(); i++) {
                    const FloatType time_weight = config.time_weights[i];
                    loss += this->LossTerm(measurements.samples[i].length, config.targets[i].length, this->shared_config.weights.length, time_weight);
                    loss += this->LossTerm(measurements.samples[i].width, config.targets[i].width, this->shared_config.weights.width, time_weight);
                    loss += this->LossTerm(measurements.samples[i].depth, config.targets[i].depth, this->shared_config.weights.depth, time_weight);
                }
                return loss;
            }

            static Config ReadConfig(const json& node) {
                Config out;
                out.times = Base::ReadNumberArray(node, "time");
                out.time_weights = Base::ReadNumberArray(node, "weights");
                if (out.time_weights.empty()) {
                    out.time_weights.assign(out.times.size(), static_cast<FloatType>(1.0));
                }

                const std::size_t count = out.times.size();
                const std::vector<Optional_Value<FloatType>> lengths = Base::ReadOptionalSeries(node, "target_length", count);
                const std::vector<Optional_Value<FloatType>> widths = Base::ReadOptionalSeries(node, "target_width", count);
                const std::vector<Optional_Value<FloatType>> depths = Base::ReadOptionalSeries(node, "target_depth", count);
                out.targets.resize(count);
                for (std::size_t i = 0; i < count; i++) {
                    out.targets[i].length = lengths[i];
                    out.targets[i].width = widths[i];
                    out.targets[i].depth = depths[i];
                }
                return out;
            }

            void Validate() const {
                if (config.times.empty()) {
                    throw std::runtime_error("Calibrate evaluator.type.path requires at least one time.");
                }
                if (config.time_weights.size() != config.times.size()) {
                    throw std::runtime_error("Calibrate path weights must match the number of times.");
                }
                bool has_target = false;
                for (std::size_t i = 0; i < config.times.size(); i++) {
                    if (!std::isfinite(static_cast<double>(config.times[i])) || config.times[i] < static_cast<FloatType>(0.0)) {
                        throw std::runtime_error("Calibrate Path times must be finite and non-negative.");
                    }
                    if (i > 0 && config.times[i] <= config.times[i - 1]) {
                        throw std::runtime_error("Calibrate Path times must be strictly increasing.");
                    }
                    has_target = has_target || this->HasWeightedTarget(config.targets[i]);
                }
                if (!has_target) {
                    throw std::runtime_error("Calibrate path evaluator requires at least one enabled target with a non-zero weight.");
                }
            }
    };

}
