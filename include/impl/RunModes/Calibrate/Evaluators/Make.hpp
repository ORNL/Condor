#pragma once

#include "impl/Initialization/Init.hpp"
#include "impl/RunModes/Calibrate/Evaluators/Base.hpp"
#include "impl/RunModes/Calibrate/Evaluators/Line.hpp"
#include "impl/RunModes/Calibrate/Evaluators/Path.hpp"
#include "impl/Structs/Simdat.hpp"

#include <algorithm>
#include <cctype>
#include <memory>
#include <stdexcept>
#include <string>

namespace Condor::impl::Modes::Calibrate {

    namespace EvaluatorFactory {

        template<typename FloatType>
        Evaluator_Shared_Config<FloatType> ReadSharedConfig(const json& calibrate_root) {
            const json& evaluator = Init::GetNestedJson(calibrate_root, "evaluator", true);

            Evaluator_Shared_Config<FloatType> config;

            // Weights for length, width, depth, and log-distance from alpha=1 beta=1.
            const json& weights = Init::GetNestedJson(evaluator, "weights", false);
            config.weights.length = Init::ReadValue<FloatType>(weights, "length", config.weights.length, false);
            config.weights.width = Init::ReadValue<FloatType>(weights, "width", config.weights.width, false);
            config.weights.depth = Init::ReadValue<FloatType>(weights, "depth", config.weights.depth, false);
            config.weights.distance = Init::ReadValue<FloatType>(weights, "distance", config.weights.distance, false);
            if (config.weights.distance < static_cast<FloatType>(0.0)) {
                throw std::runtime_error("Calibrate evaluator weight 'distance' must be non-negative.");
            }

            // Loss type
            const std::string loss = Init::ReadValue<std::string>(evaluator, "loss", "", true);
            if (loss == "percent") {
                config.loss = Evaluator_Shared_Config<FloatType>::Loss_Mode::Percent;
            }
            else if (loss == "absolute") {
                config.loss = Evaluator_Shared_Config<FloatType>::Loss_Mode::Absolute;
            }
            else {
                throw std::runtime_error("Calibrate evaluator loss must be 'percent' or 'absolute'.");
            }

            // Material output lives beside the optimizer's other output controls.
            const json& optimizer = Init::GetNestedJson(calibrate_root, "optimizer", true);
            const json& output = Init::GetNestedJson(optimizer, "output", false);
            config.output_file = Init::ReadValue<std::string>(output, "file", config.output_file, false);

            // Return the shared config
            return config;
        }

        inline const json& TypeNode(const json& calibrate_root) {
            const json& evaluator = Init::GetNestedJson(calibrate_root, "evaluator", true);
            const json& type = Init::GetNestedJson(evaluator, "type", true);
            if (!type.is_object() || type.size() != 1) {
                throw std::runtime_error("Calibrate evaluator.type must contain exactly one object: 'line' or 'path'.");
            }
            return type;
        }

    }

    template<typename FloatType>
    std::unique_ptr<Evaluator<FloatType>> MakeEvaluator(
        const Simdat<FloatType>& sim,
        const json& calibrate_root)
    {
        const Evaluator_Shared_Config<FloatType> shared_config = EvaluatorFactory::ReadSharedConfig<FloatType>(calibrate_root);
        const json& type = EvaluatorFactory::TypeNode(calibrate_root);

        // If line, return "line" evaluator unless json node isn't an object
        if (type.contains("line")) {
            if (!type["line"].is_object()) {
                throw std::runtime_error("Calibrate evaluator.type.line must be an object.");
            }
            return std::make_unique<Line_Evaluator<FloatType>>(sim, shared_config, type["line"]);
        }

        // If path, return "path" evaluator unless json node isn't an object
        if (type.contains("path")) {
            if (!type["path"].is_object()) {
                throw std::runtime_error("Calibrate evaluator.type.path must be an object.");
            }
            return std::make_unique<Path_Evaluator<FloatType>>(sim, shared_config, type["path"]);
        }

        // Throw error if we aren't doing "line" or "path" evaluator object
        throw std::runtime_error("Calibrate evaluator.type must be 'line' or 'path'.");
    }

}
