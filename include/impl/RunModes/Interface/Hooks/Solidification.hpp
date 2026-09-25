#pragma once

#include "Definitions.hpp"
#include "impl/Initialization/Init.hpp"
#include "impl/Meltpool/Tracking.hpp"
#include "impl/Output/Writer.hpp"
#include "impl/RunModes/Interface/Structs.hpp"
#include "impl/Structs/Simdat.hpp"

#include <future>
#include <string>
#include <vector>

namespace Condor::impl::Modes::Interface::Hooks {

    using json = nlohmann::json;
    using string = std::string;

    // Potential Fields
    struct Solidification_Fields {
        bool tSol = false; // Whether to store/output the exact solidification time.
        bool G = false; // Whether to store/output the total gradient magnitude field.
        bool G_vec = false; // Whether to store/output the gradient vector field.
        bool G_unit_vec = false; // Whether to store/output the gradient direction.
        bool V = false; // Whether to store/output the velocity magnitude field.
        bool dTdt = false; // Whether to store/output the cooling rate field.
        bool numMelt = false; // Whether to store/output completed melt-solidification count.

        // Member function to see if any are true
        [[nodiscard]] bool any() const {
            return tSol || G || G_vec || G_unit_vec || V || dTdt || numMelt;
        }
    };

    // Parameters
    template<typename FloatType>
    struct Solidification_Parameters {
        FloatType dT_err = static_cast<FloatType>(1e-3);
        int max_iter = 10;
    };

    // Struct for storing what is read in from json
    template<typename FloatType>
    struct Solidification_Config {
        // Stores params
        Solidification_Parameters<FloatType> params;

        // Stores outputs (one per field)
        Solidification_Fields outputs;

        // Initialization
        void Init(const json& root){
            // Do nothing/keep defaults if object is empty
            if (!root.is_null() && !root.empty()) {
                // Split up CET json node
                const json& paramsNode = Init::GetNestedJson(root, "params", true);
                const json& outputsNode = Init::GetNestedJson(root, "outputs", true);

                // Read the params first
                params.dT_err = Init::ReadValue<FloatType>(paramsNode, "dT_err", static_cast<FloatType>(1e-3), false);
                params.max_iter = Init::ReadValue<int>(paramsNode, "max_iter", static_cast<FloatType>(10), false);
            
                // Set solidification hook to require calculation
                outputs.tSol = Init::ReadValue<bool>(outputsNode, "tSol", false, false);
                outputs.G = Init::ReadValue<bool>(outputsNode, "G", false, false);
                outputs.G_vec = Init::ReadValue<bool>(outputsNode, "G_vector", false, false);
                outputs.G_unit_vec = Init::ReadValue<bool>(outputsNode, "G_unit_vector", false, false);
                outputs.V = Init::ReadValue<bool>(outputsNode, "V", false, false);               
                outputs.dTdt = Init::ReadValue<bool>(outputsNode, "dTdt", false, false);
                outputs.numMelt = Init::ReadValue<bool>(outputsNode, "numMelt", false, false);
            }
        }
    };

    // *Sharable* struct for determining what to calculate
    struct Solidification_Calc {
        // For fields (if needed or not)
        Solidification_Fields fields;

        // If there is nothing to be calculated
        bool skip = true;

        // Larger calculation decisions
        bool time = false;
        bool conditions = false;

        // Initialize the calculation flags from the parsed Solidification hook config.
        void Init() {
            // Set calculation to initially be what is being output
            if (fields.any()) { 
                time = true; 
            }
            if (fields.G ||
                fields.V ||
                fields.dTdt ||
                fields.G_vec ||
                fields.G_unit_vec)
            {
                conditions = true;
            }
            // Now set skip condition
            skip = !(time || conditions);
        }
    };

    // Struct containing Kokkos views
    template<typename FloatType>
    struct Solidification_Views {
        // Quick access
        using floating_deviceView = Kokkos::View<FloatType*, layout, device_memory>;
        using floating_hostView = Kokkos::View<FloatType*, layout, host_memory>;

        // Host mirrors populated by the deferred solidification calculation.
        floating_hostView tSol;
        floating_hostView G;
        floating_hostView V;
        floating_hostView dTdt;
        floating_hostView Gx;
        floating_hostView Gy;
        floating_hostView Gz;
        floating_hostView G_unit_x;
        floating_hostView G_unit_y;
        floating_hostView G_unit_z;
        floating_hostView numMelt;

        // Initialize based on what needs to be calculated
        void Init(const Solidification_Fields& calc, const size_t numPoints, const std::string& label) {
            if (calc.tSol) tSol = floating_hostView(label + "_tSol", numPoints);
            if (calc.G) G = floating_hostView(label + "_G", numPoints);
            if (calc.V) V = floating_hostView(label + "_V", numPoints);  
            if (calc.dTdt) dTdt = floating_hostView(label + "_dTdt", numPoints);   
            if (calc.numMelt) numMelt = floating_hostView(label + "_numMelt", numPoints);
            if (calc.G_vec) {
                Gx = floating_hostView(label + "_Gx", numPoints);
                Gy = floating_hostView(label + "_Gy", numPoints);
                Gz = floating_hostView(label + "_Gz", numPoints);
            } 
            if (calc.G_unit_vec){
                G_unit_x = floating_hostView(label + "_G_unit_x", numPoints);
                G_unit_y = floating_hostView(label + "_G_unit_y", numPoints);
                G_unit_z = floating_hostView(label + "_G_unit_z", numPoints);
            } 
        }

        // Register with the writer
        void Register(const Solidification_Fields& output, Out::Writer<FloatType>& writer){
            if (output.tSol) writer.AddView("tSol",tSol);
            if (output.G) writer.AddView("G",G);
            if (output.V) writer.AddView("V",V);
            if (output.dTdt) writer.AddView("dTdt",dTdt);
            if (output.numMelt) writer.AddView("numMelt",numMelt);
            if (output.G_vec) {
                writer.AddView("Gx",Gx);
                writer.AddView("Gy",Gy);
                writer.AddView("Gz",Gz);
            }
            if (output.G_unit_vec) {
                writer.AddView("G_unit_x",G_unit_x);
                writer.AddView("G_unit_y",G_unit_y);
                writer.AddView("G_unit_z",G_unit_z);
            }
        }
    };

    template<typename FloatType>
    class Solidification_Hook {
        private:
            using floating_hostView = typename Solidification_Views<FloatType>::floating_hostView;

            Solidification_Config<FloatType> config; // Hook-local solidification settings parsed from the Interface mode file.
            int pendingSolidification = 0; // Number of events currently owned by the background calculation.
            int lastEventCount = 0; // Number of events transferred during the most recent PostStep.
            int lastEventGeneration = 0; // Monotonic id used by coupled hooks to avoid duplicate collection.

            int_hostView lastEventPoints;
            floating_hostView lastEventMeltTimes;
            floating_hostView lastEventColdTimes;
            floating_hostView lastEventHotTemps;
            floating_hostView lastEventColdTemps;

            std::future<void> pendingTask;
            void RunSolidificationBatch(const Meltpool::Tracking<FloatType>& meltpool, const Simdat<FloatType>& sim, const FloatType timestep, const int eventCount);

        public:
            // Variables for calculation. Editable by other hooks.
            Solidification_Calc calc;

            // Kokkos views used by Solidification. Only allocated when needed.
            Solidification_Views<FloatType> volume;

            // Event-indexed results from the most recent solidification batch.
            floating_hostView eventTSol;
            floating_hostView eventDtdt;

            int EventResultCount() const { return lastEventCount; }

            // Read the Solidification hook JSON object and apply the meltpool calculation requirements.
            Solidification_Hook(const json& root){
                // If empty, set calculate to false and return (leaves everything set to not calculate)
                if (!root.is_null() && !root.empty()) {
                    // Set up config based on the json node
                    config.Init(root);
                    
                    // Set calculation to initially be only what is output
                    calc.fields = config.outputs;
                }
            }

            // Where we allocate views based on what needs to be calculated and hook to the writer what should be output
            void Initialize(const Simdat<FloatType>& sim, Meltpool::Tracking<FloatType>& meltpool, Out::Writer<FloatType>& writer){
                // Initialize calculation object (everything else should have altered it by this point)
                calc.Init();

                // Initialize views based on what needs to be calculated
                volume.Init(calc.fields, sim.domain.pnum, "SolidificationHook");

                // Register writer based on what will be output
                volume.Register(config.outputs,writer);

                // If we are doing any solidification, we need to reconstruct in the meltpool
                if (calc.fields.any()){
                    meltpool.calc.Reconstruction = true;
                    meltpool.calc.Solidification = true;
                }
            }

            // What to do after each timestep.
            void PostStep(const Meltpool::Tracking<FloatType>& meltpool, const Simdat<FloatType>& sim, const FloatType timestep);
            
            // Finalize (does the same stuff as post-step)
            void Finalize(const Meltpool::Tracking<FloatType>& meltpool, const Simdat<FloatType>& sim, const FloatType timestep){
                PostStep(meltpool, sim, timestep);
            }
    };

    //////////////////////////
    // SOLIDIFICATION LOGIC //
    //////////////////////////

    // Store the hot/cold temperature bracket used to refine one liquidus crossing.
    template<typename FloatType>
    struct SolidificationBracket {
        FloatType T_hot = static_cast<FloatType>(0.0); // Temperature on the liquid side of the bracket.
        FloatType t_hot = static_cast<FloatType>(0.0); // Time on the liquid side of the bracket.
        FloatType T_cold = static_cast<FloatType>(0.0); // Temperature on the solid side of the bracket.
        FloatType t_cold = static_cast<FloatType>(0.0); // Time on the solid side of the bracket.
    };

    // Hold the raw gradient and transient terms reconstructed at the exact solidification time.
    template<typename FloatType>
    struct SolidificationPrimaryValues {
        FloatType Gx = static_cast<FloatType>(0.0); // Temperature gradient in x.
        FloatType Gy = static_cast<FloatType>(0.0); // Temperature gradient in y.
        FloatType Gz = static_cast<FloatType>(0.0); // Temperature gradient in z.
        FloatType Laplace = static_cast<FloatType>(0.0); // Temperature Laplacian at the point.
        FloatType dT_t = static_cast<FloatType>(0.0); // Instantaneous source term contribution.
    };

    // Hold the derived solidification outputs written back onto the hook views.
    template<typename FloatType>
    struct SolidificationOutputValues {
        FloatType G = static_cast<FloatType>(0.0); // Gradient magnitude.
        FloatType Gx = static_cast<FloatType>(0.0); // Gradient vector in x.
        FloatType Gy = static_cast<FloatType>(0.0); // Gradient vector in y.
        FloatType Gz = static_cast<FloatType>(0.0); // Gradient vector in z.
        FloatType G_unit_x = static_cast<FloatType>(0.0); // Unit gradient direction in x.
        FloatType G_unit_y = static_cast<FloatType>(0.0); // Unit gradient direction in y.
        FloatType G_unit_z = static_cast<FloatType>(0.0); // Unit gradient direction in z.
        FloatType dTdt = static_cast<FloatType>(0.0); // Cooling rate magnitude.
        FloatType V = static_cast<FloatType>(0.0); // Solidification velocity.
    };
    
    // Refine one stored liquid/solid bracket into the exact target-temperature crossing time.
    template<typename FloatType>
    FloatType FindSolidificationTime(
        const std::vector<int>& start_seg,
        const Simdat<FloatType>& sim,
        const Solidification_Parameters<FloatType>& params,
        const int p,
        const SolidificationBracket<FloatType>& bracket);

    // Evaluate the raw gradient and transient terms at the exact solidification time.
    template<typename FloatType>
    SolidificationPrimaryValues<FloatType> EvaluateSolidificationPrimary(
        const Simdat<FloatType>& sim,
        const Nodes<FloatType>& nodes,
        const int p);

    // Convert the raw gradient terms into the output values stored on the hook views.
    template<typename FloatType>
    SolidificationOutputValues<FloatType> BuildSolidificationOutputs(
        const SolidificationPrimaryValues<FloatType>& primary,
        const Simdat<FloatType>& sim);

}
