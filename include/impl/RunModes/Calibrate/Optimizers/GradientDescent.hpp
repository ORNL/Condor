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
    class Gradient_Descent_Optimizer final : public Optimizer<FloatType> {
        public:
            using Base = Optimizer<FloatType>;
            using Point = typename Base::Point;
            using Result = typename Base::Result;

            struct Config {
                FloatType finite_difference = static_cast<FloatType>(0.01);
                FloatType learning_rate = static_cast<FloatType>(0.25);
                FloatType max_step = static_cast<FloatType>(0.25);
                FloatType tolerance = static_cast<FloatType>(1.0e-3);
                int max_iterations = 1000000;
            };

            Gradient_Descent_Optimizer(const Optimizer_Shared_Config& shared_config, const json& config_node)
                : Base(shared_config),
                  config(ReadConfig(config_node)),
                  learning_rate(config.learning_rate)
            {
                Validate();
            }

            std::vector<Point> NextCandidates() override {
                if (done) {
                    return {};
                }

                if (phase == Phase::ProbeCenter) {
                    return {PointFromLog(alpha_log, beta_log)};
                }
                if (phase == Phase::FiniteDifference) {
                    return {
                        PointFromLog(alpha_log + config.finite_difference, beta_log),
                        PointFromLog(alpha_log, beta_log + config.finite_difference),
                        PointFromLog(alpha_log + config.finite_difference, beta_log + config.finite_difference)
                    };
                }

                return {trial};
            }

            void RecordResults(const std::vector<Result>& results) override {
                if (results.empty()) {
                    done = true;
                    return;
                }

                if (phase == Phase::ProbeCenter) {
                    f0 = results.front();
                    Base::CheckInitialResult(f0);
                    Base::RecordCommon({f0});
                    phase = Phase::FiniteDifference;
                    return;
                }

                if (phase == Phase::FiniteDifference) {
                    if (results.size() != 3) {
                        throw std::runtime_error("Calibrate gradient descent expected three finite-difference results.");
                    }

                    const FloatType dloss_dalpha = (results[0].loss - f0.loss) / config.finite_difference;
                    const FloatType dloss_dbeta = (results[1].loss - f0.loss) / config.finite_difference;
                    const FloatType dloss_dalpha_dbeta = (results[2].loss - results[0].loss - results[1].loss + f0.loss) /
                                                         (config.finite_difference * config.finite_difference);
                    const FloatType cross = config.finite_difference * dloss_dalpha_dbeta;

                    step_alpha = ClipStep(learning_rate * (dloss_dalpha + cross), config.max_step);
                    step_beta = ClipStep(learning_rate * (dloss_dbeta + cross), config.max_step);
                    backtrack = 0;
                    PrepareTrial();
                    phase = Phase::Trial;
                    return;
                }

                const Result& trial_result = results.front();
                if (trial_result.valid && trial_result.loss <= f0.loss) {
                    Base::RecordCommon({trial_result});
                    alpha_log = trial_alpha_log;
                    beta_log = trial_beta_log;
                    const FloatType improvement = f0.loss - trial_result.loss;
                    const FloatType relative_improvement = improvement / std::max(std::abs(f0.loss), std::numeric_limits<FloatType>::epsilon());
                    f0 = trial_result;
                    iteration++;

                    if (iteration >= config.max_iterations ||
                        (std::isfinite(static_cast<double>(relative_improvement)) &&
                         relative_improvement >= static_cast<FloatType>(0.0) &&
                         relative_improvement < config.tolerance)) {
                        done = true;
                        return;
                    }

                    phase = Phase::FiniteDifference;
                    return;
                }

                if (backtrack >= 7) {
                    done = true;
                    return;
                }

                step_alpha *= static_cast<FloatType>(0.5);
                step_beta *= static_cast<FloatType>(0.5);
                learning_rate *= static_cast<FloatType>(0.5);
                backtrack++;
                PrepareTrial();
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
                FiniteDifference,
                Trial
            };

            Config config;
            Phase phase = Phase::ProbeCenter;
            FloatType alpha_log = static_cast<FloatType>(0.0);
            FloatType beta_log = static_cast<FloatType>(0.0);
            FloatType learning_rate = static_cast<FloatType>(0.25);
            FloatType step_alpha = static_cast<FloatType>(0.0);
            FloatType step_beta = static_cast<FloatType>(0.0);
            FloatType trial_alpha_log = static_cast<FloatType>(0.0);
            FloatType trial_beta_log = static_cast<FloatType>(0.0);
            int iteration = 0;
            int backtrack = 0;
            bool done = false;
            Result f0;
            Point trial;

            static Config ReadConfig(const json& node) {
                Config out;
                out.finite_difference = Init::ReadValue<FloatType>(node, "finite_difference", out.finite_difference, false);
                out.learning_rate = Init::ReadValue<FloatType>(node, "learning_rate", out.learning_rate, false);
                out.max_step = Init::ReadValue<FloatType>(node, "max_step", out.max_step, false);
                out.tolerance = Init::ReadValue<FloatType>(node, "tolerance", out.tolerance, false);
                out.max_iterations = Init::ReadValue<int>(node, "max_iterations", out.max_iterations, false);
                return out;
            }

            void Validate() const {
                if (!(config.finite_difference > static_cast<FloatType>(0.0)) ||
                    !(config.learning_rate > static_cast<FloatType>(0.0)) ||
                    !(config.max_step > static_cast<FloatType>(0.0)) ||
                    !(config.tolerance >= static_cast<FloatType>(0.0)) ||
                    config.max_iterations < 1) {
                    throw std::runtime_error("Calibrate optimizer.type.gradient_descent parameters are invalid.");
                }
            }

            static FloatType ClipStep(const FloatType step, const FloatType max_abs_step) {
                return std::max(-max_abs_step, std::min(max_abs_step, step));
            }

            void PrepareTrial() {
                trial_alpha_log = alpha_log - step_alpha;
                trial_beta_log = beta_log - step_beta;
                trial = PointFromLog(trial_alpha_log, trial_beta_log);
            }

            static Point PointFromLog(const FloatType alpha_log_in, const FloatType beta_log_in) {
                return {
                    std::max(std::pow(static_cast<FloatType>(2.0), alpha_log_in), std::numeric_limits<FloatType>::epsilon()),
                    std::max(std::pow(static_cast<FloatType>(2.0), beta_log_in), std::numeric_limits<FloatType>::epsilon())
                };
            }
    };

}
