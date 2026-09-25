#pragma once

#include "impl/Initialization/Init.hpp"
#include "impl/Meltpool/Beam.hpp"
#include "impl/Meltpool/Dimensions.hpp"
#include "impl/Meltpool/Tracking.hpp"
#include "impl/RunModes/Calibrate/Evaluators/Base.hpp"
#include "impl/RunModes/Snapshots/Config.hpp"
#include "impl/RunModes/Snapshots/Structs.hpp"
#include "impl/Structs/Simdat.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace Condor::impl::Modes::Calibrate {

    template<typename FloatType>
    class Line_Evaluator final : public Evaluator<FloatType> {
        public:
            using Base = Evaluator<FloatType>;
            using Shared_Config = typename Base::Shared_Config;

            struct Config {
                FloatType velocity = static_cast<FloatType>(1.0);
                FloatType length = static_cast<FloatType>(10.0e-3);
                Dimension_Target<FloatType> target;
            };

            Line_Evaluator(
                const Simdat<FloatType>& base,
                const Shared_Config& shared_config,
                const json& config_node)
                : Base(base, shared_config),
                  config(ReadConfig(config_node))
            {
                Validate();
            }

        private:
            struct Domain_Scale {
                FloatType x_neg = static_cast<FloatType>(1.0);
                FloatType x_pos = static_cast<FloatType>(1.0);
                FloatType y_pos = static_cast<FloatType>(1.0);
                FloatType z_neg = static_cast<FloatType>(1.0);
            };

            Config config;

            Evaluator_Measurements<FloatType> Measure(const Simdat<FloatType>& candidate) const override {
                const FloatType check_time = config.length / config.velocity;

                Measured_Dimensions<FloatType> result;
                bool measurements_no_meltpool = false;

                if (config.target.length.enabled || config.target.width.enabled) {
                    Boundary_Check check;
                    check.y_min = false;
                    check.z_min = false;
                    const Measured_Dimensions<FloatType> surface = RunWithAdaptiveDomain(candidate, check_time, true, check);
                    result.length = surface.length;
                    result.width = surface.width;
                    result.boundary_hit = result.boundary_hit || surface.boundary_hit;
                    result.has_meltpool = result.has_meltpool || surface.has_meltpool;
                    measurements_no_meltpool = measurements_no_meltpool || !surface.has_meltpool;
                }

                if (config.target.depth.enabled) {
                    Boundary_Check check;
                    check.y_min = false;
                    check.y_max = false;
                    const Measured_Dimensions<FloatType> depth = RunWithAdaptiveDomain(candidate, check_time, false, check);
                    result.depth = depth.depth;
                    result.boundary_hit = result.boundary_hit || depth.boundary_hit;
                    result.has_meltpool = result.has_meltpool || depth.has_meltpool;
                    measurements_no_meltpool = measurements_no_meltpool || !depth.has_meltpool;
                }

                Evaluator_Measurements<FloatType> measurements;
                measurements.samples.push_back(result);
                measurements.boundary_hit = result.boundary_hit;
                measurements.no_meltpool = measurements_no_meltpool;
                return measurements;
            }

            FloatType Loss(const Evaluator_Measurements<FloatType>& measurements) const override {
                if (measurements.samples.empty()) {
                    return Base::LargeLoss();
                }

                const Measured_Dimensions<FloatType>& line = measurements.samples.front();
                FloatType loss = static_cast<FloatType>(0.0);
                loss += this->LossTerm(line.length, config.target.length, this->shared_config.weights.length);
                loss += this->LossTerm(line.width, config.target.width, this->shared_config.weights.width);
                loss += this->LossTerm(line.depth, config.target.depth, this->shared_config.weights.depth);
                return loss;
            }

            static Config ReadConfig(const json& node) {
                Config out;
                out.velocity = Init::ReadValue<FloatType>(node, "velocity", out.velocity, false);
                out.length = Init::ReadValue<FloatType>(node, "length", out.length, false);
                out.target.length = Base::ReadOptionalScalar(node, "target_length");
                out.target.width = Base::ReadOptionalScalar(node, "target_width");
                out.target.depth = Base::ReadOptionalScalar(node, "target_depth");
                return out;
            }

            void Validate() const {
                if (!(config.velocity > static_cast<FloatType>(0.0)) || !(config.length > static_cast<FloatType>(0.0))) {
                    throw std::runtime_error("Calibrate evaluator.type.line velocity and length must be positive.");
                }
                if (!this->HasWeightedTarget(config.target)) {
                    throw std::runtime_error("Calibrate line evaluator requires at least one enabled target with a non-zero weight.");
                }
            }

            static void SetSingleLinePath(
                Simdat<FloatType>& sim,
                const FloatType line_length,
                const FloatType velocity)
            {
                sim.paths.clear();
                std::vector<path_seg<FloatType>> path;
                path.push_back(path_seg<FloatType>{
                    1,
                    -line_length,
                    static_cast<FloatType>(0.0),
                    static_cast<FloatType>(0.0),
                    static_cast<FloatType>(0.0),
                    static_cast<FloatType>(0.0),
                    static_cast<FloatType>(0.0)});
                path.push_back(path_seg<FloatType>{
                    0,
                    static_cast<FloatType>(0.0),
                    static_cast<FloatType>(0.0),
                    static_cast<FloatType>(0.0),
                    static_cast<FloatType>(1.0),
                    velocity,
                    line_length / velocity});
                sim.paths.push_back(path);
                sim.util.allScansEndTime = line_length / velocity;
                sim.util.approxEndTime = sim.util.allScansEndTime;
            }

            Measured_Dimensions<FloatType> RunWithAdaptiveDomain(
                const Simdat<FloatType>& candidate,
                const FloatType check_time,
                const bool surface_measurement,
                const Boundary_Check& boundary_check) const
            {
                Domain_Scale scale;
                Measured_Dimensions<FloatType> measured;
                constexpr int max_attempts = 4;

                for (int attempt = 0; attempt < max_attempts; attempt++) {
                    Simdat<FloatType> sim = surface_measurement
                        ? MakeLineXYSim(candidate, scale)
                        : MakeLineXZSim(candidate, scale);

                    measured = RunSnapshot(sim, check_time, boundary_check);
                    if (!measured.boundary_hit) {
                        return measured;
                    }

                    ExpandScale(scale, boundary_check);
                }

                return measured;
            }

            static void ExpandScale(Domain_Scale& scale, const Boundary_Check& check) {
                if (check.x_min) { scale.x_neg *= static_cast<FloatType>(2.0); }
                if (check.x_max) { scale.x_pos *= static_cast<FloatType>(2.0); }
                if (check.y_max) { scale.y_pos *= static_cast<FloatType>(2.0); }
                if (check.z_min) { scale.z_neg *= static_cast<FloatType>(2.0); }
            }

            static Measured_Dimensions<FloatType> RunSnapshot(
                Simdat<FloatType> sim,
                const FloatType check_time,
                const Boundary_Check& boundary_check)
            {
                Snapshots::Snapshots_Config<FloatType> snapshot_config;
                snapshot_config.settings.times.push_back(check_time);
                snapshot_config.settings.time_iterations.emplace_back(check_time, 1);
                snapshot_config.settings.fallback_timestep = std::max(check_time, static_cast<FloatType>(1.0e-9));
                snapshot_config.settings.tracking = Snapshots::Tracking_Mode::Interface;

                Snapshots::Snapshots_State<FloatType> state(sim, snapshot_config);
                Meltpool::BeamTracer<FloatType> beam_trace(sim, snapshot_config.settings.fallback_timestep);
                beam_trace.Stop();
                Meltpool::Tracking<FloatType> meltpool(sim);
                meltpool.Initialize_Solidification();

                Meltpool::Dimensions<FloatType> dimensions(sim);
                dimensions.calc.MP_Len_or_Wid = true;
                dimensions.calc.MP_Depth = sim.domain.znum > 1;
                dimensions.Initialize();

                Measured_Dimensions<FloatType> measured;

                state.Advance_To(0, static_cast<FloatType>(0.0), check_time);
                beam_trace.TraceWindowDevice(static_cast<FloatType>(0.0), check_time);
                meltpool.Advance(state, beam_trace);

                if (Base::HitsBoundary(sim, meltpool.ActiveColumnCount(), meltpool.ActiveColumns(), meltpool.CurrentDepth(), boundary_check)) {
                    measured.boundary_hit = true;
                }
                else {
                    dimensions.Measure(meltpool, state.thermal, sim, state.step);
                    Base::UpdateBestDimensions(dimensions, measured);
                }

                meltpool.Finalize();
                state.Stop_All();
                return measured;
            }

            Simdat<FloatType> MakeLineXYSim(
                const Simdat<FloatType>& candidate,
                const Domain_Scale& scale) const
            {
                Simdat<FloatType> sim = candidate;
                SetSingleLinePath(sim, config.length, config.velocity);

                const Beam<FloatType>& beam = sim.beams.front();
                const FloatType width_ref = WidthReference(sim, beam);
                const FloatType x_neg = XNegativeExtent(width_ref) * scale.x_neg;
                const FloatType x_pos = static_cast<FloatType>(2.0) * width_ref * scale.x_pos;
                const FloatType y_pos = static_cast<FloatType>(2.0) * width_ref * scale.y_pos;

                sim.domain.calc_bounds = false;
                sim.domain.xmin = -x_neg;
                sim.domain.xmax = x_pos;
                sim.domain.ymin = static_cast<FloatType>(0.0);
                sim.domain.ymax = y_pos;
                sim.domain.zmin = static_cast<FloatType>(0.0);
                sim.domain.zmax = static_cast<FloatType>(0.0);
                Init::SetDomainParams(sim.domain);
                return sim;
            }

            Simdat<FloatType> MakeLineXZSim(
                const Simdat<FloatType>& candidate,
                const Domain_Scale& scale) const
            {
                Simdat<FloatType> sim = candidate;
                SetSingleLinePath(sim, config.length, config.velocity);

                const Beam<FloatType>& beam = sim.beams.front();
                const FloatType depth_ref = DepthReference(sim, beam);
                const FloatType width_ref = WidthReference(sim, beam);
                const FloatType x_neg = XNegativeExtent(width_ref) * scale.x_neg;
                const FloatType x_pos = static_cast<FloatType>(2.0) * width_ref * scale.x_pos;
                const FloatType z_neg = static_cast<FloatType>(2.0) * depth_ref * scale.z_neg;

                sim.domain.calc_bounds = false;
                sim.domain.xmin = -x_neg;
                sim.domain.xmax = x_pos;
                sim.domain.ymin = static_cast<FloatType>(0.0);
                sim.domain.ymax = static_cast<FloatType>(0.0);
                sim.domain.zmax = static_cast<FloatType>(0.0);
                sim.domain.zmin = -z_neg;
                Init::SetDomainParams(sim.domain);
                return sim;
            }

            FloatType WidthReference(const Simdat<FloatType>& sim, const Beam<FloatType>& beam) const {
                const FloatType fallback = std::max({
                    std::max(beam.ax, beam.ay),
                    sim.domain.res,
                    static_cast<FloatType>(1.0e-12)});
                return config.target.width.enabled
                    ? std::max(std::abs(config.target.width.value), sim.domain.res)
                    : fallback;
            }

            FloatType DepthReference(const Simdat<FloatType>& sim, const Beam<FloatType>& beam) const {
                const FloatType fallback = std::max({
                    beam.az,
                    sim.domain.res,
                    static_cast<FloatType>(1.0e-12)});
                return config.target.depth.enabled
                    ? std::max(std::abs(config.target.depth.value), sim.domain.res)
                    : fallback;
            }

            FloatType XNegativeExtent(const FloatType width_ref) const {
                if (config.target.length.enabled) {
                    return static_cast<FloatType>(2.0) * std::abs(config.target.length.value) +
                           static_cast<FloatType>(2.0) * width_ref;
                }
                return static_cast<FloatType>(20.0) * width_ref;
            }
    };

}
