#pragma once

#include "impl/Initialization/Init.hpp"
#include "impl/RunModes/Calibrate/Optimizers/Base.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace Condor::impl::Modes::Calibrate {

    template<typename FloatType>
    class Grid_Optimizer final : public Optimizer<FloatType> {
        public:
            using Base = Optimizer<FloatType>;
            using Point = typename Base::Point;
            using Result = typename Base::Result;

            struct Config {
                FloatType initial_span = static_cast<FloatType>(1.0);
                FloatType tolerance = static_cast<FloatType>(1.0e-6);
                int points = 5;
                int iterations = 6;
            };

            Grid_Optimizer(const Optimizer_Shared_Config& shared_config, const json& config_node)
                : Base(shared_config),
                  config(ReadConfig(config_node)),
                  span(config.initial_span)
            {
                Validate();
            }

            std::vector<Point> NextCandidates() override {
                if (done) {
                    return {};
                }
                if (phase == Phase::ProbeCenter) {
                    return {PointFromLog(center_alpha_log, center_beta_log)};
                }

                std::vector<Point> points;
                const int half = config.points / 2;
                points.reserve(static_cast<std::size_t>(config.points * config.points));
                for (int ia = -half; ia <= half; ia++) {
                    for (int ib = -half; ib <= half; ib++) {
                        points.push_back({
                            GridCandidate(center_alpha_log, ia, half),
                            GridCandidate(center_beta_log, ib, half)
                        });
                    }
                }
                return points;
            }

            void RecordResults(const std::vector<Result>& results) override {
                if (results.empty()) {
                    done = true;
                    return;
                }

                if (phase == Phase::ProbeCenter) {
                    const Result& center_result = results.front();
                    Base::CheckInitialResult(center_result);
                    Base::RecordCommon({center_result});
                    phase = Phase::GridBatch;
                    return;
                }

                Base::RecordCommon(results);

                Result iteration_best;
                bool has_iteration_best = false;
                for (const Result& result : results) {
                    if (result.valid && (!has_iteration_best || result.loss < iteration_best.loss)) {
                        iteration_best = result;
                        has_iteration_best = true;
                    }
                }

                if (!has_iteration_best) {
                    done = true;
                    return;
                }

                center_alpha_log = Log2(iteration_best.alpha);
                center_beta_log = Log2(iteration_best.beta);
                span /= static_cast<FloatType>(config.points - 1);
                iteration++;

                if (iteration >= config.iterations || span <= config.tolerance) {
                    done = true;
                    return;
                }

                phase = Phase::GridBatch;
            }

            [[nodiscard]] bool Done() const override { return done; }

            [[nodiscard]] Result Best() const override {
                if (!this->has_best) {
                    throw std::runtime_error("Calibrate optimizer has no evaluated candidates.");
                }
                return this->best;
            }

        private:
            enum class Phase {
                ProbeCenter,
                GridBatch
            };

            Config config;
            Phase phase = Phase::ProbeCenter;
            FloatType center_alpha_log = static_cast<FloatType>(0.0);
            FloatType center_beta_log = static_cast<FloatType>(0.0);
            FloatType span = static_cast<FloatType>(0.0);
            int iteration = 0;
            bool done = false;

            static Config ReadConfig(const json& node) {
                Config out;
                out.initial_span = Init::ReadValue<FloatType>(node, "initial_span", out.initial_span, false);
                out.tolerance = Init::ReadValue<FloatType>(node, "tolerance", out.tolerance, false);
                out.points = Init::ReadValue<int>(node, "points", out.points, false);
                out.iterations = Init::ReadValue<int>(node, "iterations", out.iterations, false);
                return out;
            }

            void Validate() const {
                if (config.points < 3 || config.points % 2 == 0) {
                    throw std::runtime_error("Calibrate optimizer.type.grid points must be an odd integer >= 3.");
                }
                if (!(config.initial_span > static_cast<FloatType>(0.0)) ||
                    !(config.tolerance >= static_cast<FloatType>(0.0)) ||
                    config.iterations < 1 ||
                    config.points < 3) {
                    throw std::runtime_error("Calibrate optimizer.type.grid parameters are invalid.");
                }
            }

            FloatType GridCandidate(const FloatType center_log, const int offset, const int half) const {
                return FactorFromLog(center_log + span * static_cast<FloatType>(offset) / static_cast<FloatType>(half));
            }

            static Point PointFromLog(const FloatType alpha_log, const FloatType beta_log) {
                return {FactorFromLog(alpha_log), FactorFromLog(beta_log)};
            }

            static FloatType FactorFromLog(const FloatType value) {
                return std::max(std::pow(static_cast<FloatType>(2.0), value), std::numeric_limits<FloatType>::epsilon());
            }

            static FloatType Log2(const FloatType value) {
                return std::log(value) / std::log(static_cast<FloatType>(2.0));
            }
    };

}
