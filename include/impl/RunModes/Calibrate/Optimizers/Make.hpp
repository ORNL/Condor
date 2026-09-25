#pragma once

#include "impl/Initialization/Init.hpp"
#include "impl/RunModes/Calibrate/Optimizers/Base.hpp"
#include "impl/RunModes/Calibrate/Optimizers/GradientDescent.hpp"
#include "impl/RunModes/Calibrate/Optimizers/Grid.hpp"

#include <memory>
#include <stdexcept>
#include <string>

namespace Condor::impl::Modes::Calibrate {

    namespace OptimizerFactory {

        inline Optimizer_Print_Mode ReadPrintMode(const std::string& value) {
            if (value == "on") {
                return Optimizer_Print_Mode::On;
            }
            if (value == "off") {
                return Optimizer_Print_Mode::Off;
            }
            throw std::runtime_error("Calibrate optimizer output print must be 'on' or 'off'.");
        }

        inline Optimizer_Shared_Config ReadSharedConfig(const json& calibrate_root) {
            const json& optimizer = Init::GetNestedJson(calibrate_root, "optimizer", true);
            const json& output = Init::GetNestedJson(optimizer, "output", false);

            Optimizer_Shared_Config config;
            config.print = ReadPrintMode(Init::ReadValue<std::string>(output, "print", "off", false));
            config.history_file = Init::ReadValue<std::string>(output, "history", "", false);
            return config;
        }

        inline const json& TypeNode(const json& calibrate_root) {
            const json& optimizer = Init::GetNestedJson(calibrate_root, "optimizer", true);
            const json& type = Init::GetNestedJson(optimizer, "type", true);
            if (!type.is_object() || type.size() != 1) {
                throw std::runtime_error("Calibrate optimizer.type must contain exactly one object: 'grid' or 'gradient_descent'.");
            }
            return type;
        }

    }

    template<typename FloatType>
    std::unique_ptr<Optimizer<FloatType>> MakeOptimizer(const json& calibrate_root) {
        const Optimizer_Shared_Config shared_config = OptimizerFactory::ReadSharedConfig(calibrate_root);
        const json& type = OptimizerFactory::TypeNode(calibrate_root);

        if (type.contains("grid")) {
            if (!type["grid"].is_object()) {
                throw std::runtime_error("Calibrate optimizer.type.grid must be an object.");
            }
            return std::make_unique<Grid_Optimizer<FloatType>>(shared_config, type["grid"]);
        }

        if (type.contains("gradient_descent")) {
            if (!type["gradient_descent"].is_object()) {
                throw std::runtime_error("Calibrate optimizer.type.gradient_descent must be an object.");
            }
            return std::make_unique<Gradient_Descent_Optimizer<FloatType>>(shared_config, type["gradient_descent"]);
        }

        throw std::runtime_error("Calibrate optimizer.type must be 'grid' or 'gradient_descent'.");
    }

}
