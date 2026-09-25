#pragma once

#include "Definitions.hpp"
#include "impl/Initialization/Init.hpp"
#include "impl/Meltpool/Beam.hpp"
#include "impl/Meltpool/Dimensions.hpp"
#include "impl/RunModes/Calibrate/Optimizers/Base.hpp"
#include "impl/Structs/Simdat.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace Condor::impl::Modes::Calibrate {

    using json = nlohmann::json;

    template<typename FloatType>
    struct Optional_Value {
        bool enabled = false;
        FloatType value = static_cast<FloatType>(0.0);
    };

    template<typename FloatType>
    struct Dimension_Target {
        Optional_Value<FloatType> length;
        Optional_Value<FloatType> width;
        Optional_Value<FloatType> depth;

        [[nodiscard]] bool any() const {
            return length.enabled || width.enabled || depth.enabled;
        }
    };

    template<typename FloatType>
    struct Measured_Dimensions {
        FloatType length = static_cast<FloatType>(0.0);
        FloatType width = static_cast<FloatType>(0.0);
        FloatType depth = static_cast<FloatType>(0.0);
        bool boundary_hit = false;
        bool has_meltpool = false;
    };

    struct Boundary_Check {
        bool x_min = true;
        bool x_max = true;
        bool y_min = true;
        bool y_max = true;
        bool z_min = true;
    };

    template<typename FloatType>
    struct Evaluator_Measurements {
        std::vector<Measured_Dimensions<FloatType>> samples;
        bool boundary_hit = false;
        bool no_meltpool = false;
    };

    template<typename FloatType>
    struct Evaluator_Shared_Config {
        struct Weights {
            FloatType length = static_cast<FloatType>(1.0);
            FloatType width = static_cast<FloatType>(1.0);
            FloatType depth = static_cast<FloatType>(1.0);
            FloatType distance = static_cast<FloatType>(0.0);
        } weights;
        std::string output_file = "Material-Optimized.json";
        
        enum class Loss_Mode {
            Percent,
            Absolute
        };
        Loss_Mode loss = Loss_Mode::Percent;
    };

    template<typename FloatType>
    class Evaluator {
        public:
            using Point = Candidate_Point<FloatType>;
            using Result = Candidate_Result<FloatType>;
            using Shared_Config = Evaluator_Shared_Config<FloatType>;

            Evaluator(const Simdat<FloatType>& base_in, const Shared_Config& shared_config_in)
                : base(base_in),
                  shared_config(shared_config_in),
                  capacity0(base_in.material.rho * base_in.material.cps),
                  alpha0(base_in.material.kon / (base_in.material.rho * base_in.material.cps))
            {}

            virtual ~Evaluator() = default;

            Result Evaluate(const Point& point) {
                const std::string key = CacheKey(point.alpha, point.beta);
                const auto found = cache.find(key);
                if (found != cache.end()) {
                    return found->second;
                }

                Result result;
                result.alpha = point.alpha;
                result.beta = point.beta;

                const Simdat<FloatType> candidate = CandidateSim(base, point);
                const Evaluator_Measurements<FloatType> measurements = Measure(candidate);
                if (measurements.boundary_hit || measurements.no_meltpool) {
                    result.boundary_hit = measurements.boundary_hit;
                    result.no_meltpool = measurements.no_meltpool;
                    result.loss = LargeLoss();
                    result.valid = false;
                    cache.emplace(key, result);
                    return result;
                }

                result.loss = Loss(measurements) + DistanceLoss(point);
                result.valid = std::isfinite(static_cast<double>(result.loss));
                cache.emplace(key, result);
                return result;
            }

            void WriteMaterial(const Result& result) const {
                const FloatType capacity = capacity0 / result.beta;
                const FloatType alpha = alpha0 * result.alpha;
                const FloatType cps = capacity / base.material.rho;
                const FloatType kon = alpha * capacity;

                json out;
                out["constants"]["T_init"] = base.material.T_init;
                out["constants"]["T_liq"] = base.material.T_liq;
                out["constants"]["k"] = kon;
                out["constants"]["c"] = cps;
                out["constants"]["p"] = base.material.rho;

                out["calibration"]["alpha"] = result.alpha;
                out["calibration"]["beta"] = result.beta;
                out["calibration"]["loss"] = result.loss;

                const std::filesystem::path output_path = ResolveOutputPath(shared_config.output_file);
                if (!output_path.parent_path().empty()) {
                    std::filesystem::create_directories(output_path.parent_path());
                }

                std::ofstream file(output_path);
                if (!file.is_open()) {
                    throw std::runtime_error("Could not open Calibrate output material file: " + output_path.string());
                }
                file << std::setw(4) << out << "\n";
            }

            [[nodiscard]] FloatType Alpha0() const { return alpha0; }
            [[nodiscard]] FloatType Capacity0() const { return capacity0; }

        protected:
            virtual Evaluator_Measurements<FloatType> Measure(const Simdat<FloatType>& candidate) const = 0;
            virtual FloatType Loss(const Evaluator_Measurements<FloatType>& measurements) const = 0;

            const Simdat<FloatType>& base;
            Shared_Config shared_config;
            FloatType capacity0;
            FloatType alpha0;

            static FloatType LargeLoss() {
                return std::numeric_limits<FloatType>::max() / static_cast<FloatType>(1024.0);
            }

            static FloatType PositiveOr(const FloatType value, const FloatType fallback) {
                return (value > static_cast<FloatType>(0.0)) ? value : fallback;
            }

            static FloatType TargetFallback(
                const Optional_Value<FloatType>& target,
                const FloatType fallback)
            {
                return target.enabled
                    ? std::max(std::abs(target.value), fallback)
                    : fallback;
            }

            FloatType LossTerm(
                const FloatType predicted,
                const Optional_Value<FloatType>& target,
                const FloatType weight,
                const FloatType sample_weight = static_cast<FloatType>(1.0)) const
            {
                if (!target.enabled || weight == static_cast<FloatType>(0.0)) {
                    return static_cast<FloatType>(0.0);
                }
                const FloatType diff = std::abs(predicted - target.value);
                const FloatType base_loss = (shared_config.loss == Shared_Config::Loss_Mode::Percent)
                    ? diff / std::max(std::abs(target.value), static_cast<FloatType>(1.0e-30))
                    : diff;
                return base_loss * weight * sample_weight;
            }

            FloatType DistanceLoss(const Point& point) const {
                const FloatType weight = shared_config.weights.distance;
                if (weight == static_cast<FloatType>(0.0)) {
                    return static_cast<FloatType>(0.0);
                }

                const FloatType alpha_log = Log2(point.alpha);
                const FloatType beta_log = Log2(point.beta);
                return weight * std::sqrt(alpha_log * alpha_log + beta_log * beta_log);
            }

            bool HasWeightedTarget(const Dimension_Target<FloatType>& target) const {
                return (target.length.enabled && shared_config.weights.length != static_cast<FloatType>(0.0)) ||
                       (target.width.enabled && shared_config.weights.width != static_cast<FloatType>(0.0)) ||
                       (target.depth.enabled && shared_config.weights.depth != static_cast<FloatType>(0.0));
            }

            static Optional_Value<FloatType> ReadOptionalScalar(const json& node, const std::string& key) {
                Optional_Value<FloatType> out;
                if (!node.contains(key) || node[key].is_null()) {
                    return out;
                }
                if (!node[key].is_number()) {
                    throw std::runtime_error("Calibrate target '" + key + "' must be a number.");
                }
                out.enabled = true;
                out.value = node[key].get<FloatType>();
                return out;
            }

            static std::vector<Optional_Value<FloatType>> ReadOptionalSeries(
                const json& node,
                const std::string& key,
                const std::size_t count)
            {
                std::vector<Optional_Value<FloatType>> values(count);
                if (!node.contains(key) || node[key].is_null()) {
                    return values;
                }

                const json& source = node[key];
                if (source.is_number()) {
                    const FloatType value = source.get<FloatType>();
                    for (Optional_Value<FloatType>& entry : values) {
                        entry.enabled = true;
                        entry.value = value;
                    }
                    return values;
                }

                if (!source.is_array() || source.size() != count) {
                    throw std::runtime_error("Calibrate path target '" + key + "' must be a scalar or an array matching the number of times.");
                }

                for (std::size_t i = 0; i < count; i++) {
                    if (!source[i].is_null()) {
                        if (!source[i].is_number()) {
                            throw std::runtime_error("Calibrate path target '" + key + "' entries must be numbers or null.");
                        }
                        values[i].enabled = true;
                        values[i].value = source[i].get<FloatType>();
                    }
                }
                return values;
            }

            static std::vector<FloatType> ReadNumberArray(const json& node, const std::string& key) {
                if (!node.contains(key) || node[key].is_null()) {
                    return {};
                }
                if (!node[key].is_array()) {
                    throw std::runtime_error("Calibrate field '" + key + "' must be an array.");
                }

                std::vector<FloatType> out;
                out.reserve(node[key].size());
                for (const json& entry : node[key]) {
                    if (!entry.is_number()) {
                        throw std::runtime_error("Calibrate field '" + key + "' entries must be numbers.");
                    }
                    out.push_back(entry.get<FloatType>());
                }
                return out;
            }

            static bool HitsBoundary(
                const Simdat<FloatType>& sim,
                const int active_count,
                const int_deviceView& active_columns,
                const int_2D_deviceView& depth,
                const Boundary_Check& check)
            {
                if (active_count <= 0) {
                    return false;
                }

                int_hostView active_h = Kokkos::create_mirror_view(active_columns);
                int_2D_hostView depth_h = Kokkos::create_mirror_view(depth);
                Kokkos::deep_copy(active_h, active_columns);
                Kokkos::deep_copy(depth_h, depth);

                for (int q = 0; q < active_count; q++) {
                    const int v2d = active_h(q);
                    const int i = v2d / sim.domain.ynum;
                    const int j = v2d % sim.domain.ynum;
                    if (check.x_min && sim.domain.xnum > 1 && i == 0) { return true; }
                    if (check.x_max && sim.domain.xnum > 1 && i + 1 == sim.domain.xnum) { return true; }
                    if (check.y_min && sim.domain.ynum > 1 && j == 0) { return true; }
                    if (check.y_max && sim.domain.ynum > 1 && j + 1 == sim.domain.ynum) { return true; }
                    if (check.z_min && sim.domain.znum > 1 && depth_h(i, j) >= sim.domain.znum) { return true; }
                }
                return false;
            }

            static void UpdateBestDimensions(
                Meltpool::Dimensions<FloatType>& dimensions,
                Measured_Dimensions<FloatType>& best)
            {
                if (dimensions.component_count_h <= 0) {
                    return;
                }

                best.has_meltpool = true;
                Kokkos::deep_copy(dimensions.component_length_h, dimensions.component_length_d);
                Kokkos::deep_copy(dimensions.component_width_h, dimensions.component_width_d);
                if (dimensions.calc.MP_Depth) {
                    Kokkos::deep_copy(dimensions.component_depth_h, dimensions.component_depth_d);
                }

                int best_component = 0;
                FloatType best_area = static_cast<FloatType>(-1.0);
                for (int c = 0; c < dimensions.component_count_h; c++) {
                    const FloatType area = dimensions.component_length_h(c) * dimensions.component_width_h(c);
                    if (area > best_area) {
                        best_area = area;
                        best_component = c;
                    }
                }

                best.length = std::max(best.length, dimensions.component_length_h(best_component));
                best.width = std::max(best.width, dimensions.component_width_h(best_component));
                if (dimensions.calc.MP_Depth) {
                    best.depth = std::max(best.depth, dimensions.component_depth_h(best_component));
                }
            }

        private:
            std::unordered_map<std::string, Result> cache;

            static std::string CacheKey(const FloatType alpha_factor, const FloatType beta_factor) {
                std::ostringstream key;
                key << std::setprecision(12) << alpha_factor << ":" << beta_factor;
                return key.str();
            }

            static FloatType Log2(const FloatType value) {
                return std::log(value) / std::log(static_cast<FloatType>(2.0));
            }

            Simdat<FloatType> CandidateSim(const Simdat<FloatType>& source, const Point& point) const {
                if (!(point.alpha > static_cast<FloatType>(0.0)) || !(point.beta > static_cast<FloatType>(0.0))) {
                    throw std::runtime_error("Calibrate candidate factors must be positive.");
                }

                Simdat<FloatType> sim = source;
                const FloatType capacity = capacity0 / point.beta;
                const FloatType alpha = alpha0 * point.alpha;
                sim.material.cps = capacity / sim.material.rho;
                sim.material.kon = alpha * capacity;
                sim.material.a = alpha;
                sim.settings.output.noOutput = true;
                sim.settings.print = 0;
                RecalculateCandidateDerivedValues(sim);
                return sim;
            }

            static void RecalculateCandidateDerivedValues(Simdat<FloatType>& sim) {
                for (Beam<FloatType>& beam : sim.beams) {
                    beam.nond_dt = beam.ax * beam.ax / sim.material.a;
                }

                FloatType r_max = static_cast<FloatType>(-1.0);
                for (const Beam<FloatType>& beam : sim.beams) {
                    FloatType beam_r = static_cast<FloatType>(0.0);
                    if (sim.settings.t_hist < std::exp(static_cast<FloatType>(1.5))) {
                        beam_r = beam.ax * std::sqrt(std::log(sim.settings.t_hist) / static_cast<FloatType>(3.0));
                    }
                    else {
                        beam_r = beam.ax * std::pow(sim.settings.t_hist, static_cast<FloatType>(1.0) / static_cast<FloatType>(3.0)) /
                            std::sqrt(static_cast<FloatType>(2.0) * std::exp(static_cast<FloatType>(1.0)));
                    }

                    const FloatType beta = std::pow(static_cast<FloatType>(3.0) / static_cast<FloatType>(PI), static_cast<FloatType>(1.5)) * beam.q / (sim.material.rho * sim.material.cps);
                    const FloatType temp_diff = sim.material.T_liq - sim.material.T_init;
                    const FloatType x = temp_diff * sim.settings.p_hist;
                    if (x > static_cast<FloatType>(0.0) && beta > static_cast<FloatType>(0.0)) {
                        FloatType r_max_2 = static_cast<FloatType>(0.0);
                        if (beta / (x * beam.ax * beam.ax * beam.ax) < std::exp(static_cast<FloatType>(1.5))) {
                            r_max_2 = beam.ax * std::sqrt(std::log(beta / (x * beam.ax * beam.ax * beam.ax)) / static_cast<FloatType>(3.0));
                        }
                        else {
                            r_max_2 = std::pow(beta / x, static_cast<FloatType>(1.0) / static_cast<FloatType>(3.0)) /
                                std::sqrt(static_cast<FloatType>(2.0) * std::exp(static_cast<FloatType>(1.0)));
                        }
                        beam_r = std::max(beam_r, r_max_2);
                    }
                    r_max = std::max(r_max, beam_r);
                }
                sim.settings.r_max = r_max;
            }

            std::filesystem::path ResolveOutputPath(const std::string& output) const {
                std::filesystem::path out(output);
                if (out.is_absolute()) {
                    return out;
                }
                std::filesystem::path base_path = ".";
                if (!base.files.dataDir.empty()) {
                    base_path = std::filesystem::path(base.files.dataDir).parent_path();
                    if (base_path.empty()) {
                        base_path = ".";
                    }
                }
                return base_path / out;
            }
    };

}
