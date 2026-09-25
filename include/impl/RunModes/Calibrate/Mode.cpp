#include "impl/RunModes/Calibrate/Mode.hpp"

#include "impl/Initialization/Init.hpp"
#include "impl/RunModes/Calibrate/Evaluators/Make.hpp"
#include "impl/RunModes/Calibrate/Optimizers/Make.hpp"

#include <memory>
#include <vector>

namespace Condor::impl::Modes::Calibrate {
    
    template<typename FloatType>
    void Run(Simdat<FloatType>& sim) {
        // Get the nested jsons
        const json& calibrate_json = Init::GetNestedJson(sim.files.mode_json, "Calibrate", true);

        // Make the evalutor and optimizer based on config
        std::unique_ptr<Evaluator<FloatType>> evaluator = MakeEvaluator<FloatType>(sim, calibrate_json);
        std::unique_ptr<Optimizer<FloatType>> optimizer = MakeOptimizer<FloatType>(calibrate_json);

        // Run until optimizer says we are done
        while (!optimizer->Done()) {
            // Get list of canidates from optimzier
            const std::vector<Candidate_Point<FloatType>> points = optimizer->NextCandidates();
            if (points.empty()) {
                break;
            }

            // Make a list of results and fill for each point
            std::vector<Candidate_Result<FloatType>> results;
            results.reserve(points.size());
            for (const Candidate_Point<FloatType>& point : points) {
                results.push_back(evaluator->Evaluate(point));
            }

            // Record results and loop
            optimizer->RecordResults(results);
        }

        // Get the best result from the optimizer and write it to the file
        const Candidate_Result<FloatType> result = optimizer->Best();
        evaluator->WriteMaterial(result);
    }

    template void Run(Simdat<float>&);
    template void Run(Simdat<double>&);
}
