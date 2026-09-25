#pragma once

#include "impl/Initialization/Init.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace Condor::impl::Modes::Calibrate {

    using json = nlohmann::json;

    template<typename FloatType>
    struct Candidate_Point {
        FloatType alpha = static_cast<FloatType>(1.0);
        FloatType beta = static_cast<FloatType>(1.0);
    };

    template<typename FloatType>
    struct Candidate_Result {
        FloatType alpha = static_cast<FloatType>(1.0);
        FloatType beta = static_cast<FloatType>(1.0);
        FloatType loss = std::numeric_limits<FloatType>::infinity();
        bool valid = false;
        bool boundary_hit = false;
        bool no_meltpool = false;
    };

    enum class Optimizer_Print_Mode {
        Off,
        On
    };

    struct Optimizer_Shared_Config {
        Optimizer_Print_Mode print = Optimizer_Print_Mode::Off;
        std::string history_file;
    };

    template<typename FloatType>
    class Optimizer {
        public:
            using Point = Candidate_Point<FloatType>;
            using Result = Candidate_Result<FloatType>;

            explicit Optimizer(const Optimizer_Shared_Config& shared_config_in)
                : shared_config(shared_config_in)
            {
                if (!shared_config.history_file.empty()) {
                    history_stream.open(shared_config.history_file);
                    if (!history_stream.is_open()) {
                        throw std::runtime_error("Could not open Calibrate optimizer history file: " + shared_config.history_file);
                    }
                    history_stream << "alpha,beta,loss,valid,boundary_hit,no_meltpool\n";
                }
            }

            virtual ~Optimizer() = default;

            virtual std::vector<Point> NextCandidates() = 0;
            virtual void RecordResults(const std::vector<Result>& results) = 0;
            [[nodiscard]] virtual bool Done() const = 0;
            [[nodiscard]] virtual Result Best() const = 0;

        protected:
            Optimizer_Shared_Config shared_config;
            std::vector<Result> history;
            std::ofstream history_stream;
            Result best;
            bool has_best = false;

            void Print(const Result& result) const {
                if (shared_config.print != Optimizer_Print_Mode::On) {
                    return;
                }
                std::cout << std::setprecision(10)
                          << result.alpha << " "
                          << result.beta << " "
                          << result.loss << "\n";
            }

            void RecordCommon(const std::vector<Result>& results) {
                for (const Result& result : results) {
                    history.push_back(result);
                    Print(result);
                    if (history_stream.is_open()) {
                        history_stream << std::setprecision(10)
                                       << result.alpha << ","
                                       << result.beta << ","
                                       << result.loss << ","
                                       << result.valid << ","
                                       << result.boundary_hit << ","
                                       << result.no_meltpool << "\n";
                    }
                    if (result.valid && (!has_best || result.loss < best.loss)) {
                        best = result;
                        has_best = true;
                    }
                }
            }

            static void CheckInitialResult(const Result& result) {
                if (result.valid && !result.boundary_hit && !result.no_meltpool) {
                    return;
                }

                if (result.boundary_hit) {
                    throw std::runtime_error("Calibrate initial alpha=1 beta=1 candidate reached the domain boundary.");
                }
                if (result.no_meltpool) {
                    throw std::runtime_error("Calibrate initial alpha=1 beta=1 candidate produced no meltpool.");
                }
                throw std::runtime_error("Calibrate initial alpha=1 beta=1 candidate is invalid.");
            }
    };

}
