#pragma once

// Include the shared Interface hook parser helpers and solidification hook dependency.
#include "impl/RunModes/Interface/Hooks/Solidification.hpp"
#include <limits>

namespace Condor::impl::Modes::Interface::Hooks {
    // Usings
    using json = nlohmann::json;

    namespace
    {
        // Potential Fields
        struct CET_Fields {
            bool eqFrac = false; // Whether to output the solidification temperature field.
        };

        // Parameters/Settings
        template<typename FloatType>
        struct CET_Params{
            FloatType N0, n, a;
        };

        // Struct for storing what is read in from json
        template<typename FloatType>
        struct CET_Config {
            CET_Params<FloatType> params;
            CET_Fields outputs;

            // Initialize config via json Node
            void Init(const json& root){
                // Do nothing/keep defaults if object is empty
                if (!root.is_null() && !root.empty()) {
                    // Split up CET json node
                    const json& paramsNode = Init::GetNestedJson(root, "params", true);
                    const json& outputsNode = Init::GetNestedJson(root, "outputs", true);

                    // Read the params first
                    params.N0 = Init::ReadValue<FloatType>(paramsNode, "N0", std::numeric_limits<FloatType>::max(), true);
                    params.n = Init::ReadValue<FloatType>(paramsNode, "n", std::numeric_limits<FloatType>::max(), true);
                    params.a = Init::ReadValue<FloatType>(paramsNode, "a", std::numeric_limits<FloatType>::max(), true);
                
                    // Read in the outputs
                    outputs.eqFrac = Init::ReadValue<bool>(outputsNode, "eqFrac", false, true);
                }
            }
        };

        // *Sharable* struct for determining what to calculate
        struct CET_Calc{
            // For fields (if needed or not)
            CET_Fields fields;
            
            // If there is nothing to be calculated
            bool skip = true;

            // Initialize the calculation flags from the parsed CET hook config.     
            void Init() {
                skip = !fields.eqFrac;
            }
        };
    }

     // Struct containing Kokkos views
    template<typename FloatType>
    struct CET_Views {
        // Quick access
        using floating_deviceView = Kokkos::View<FloatType*, layout, device_memory>;
        using floating_hostView = Kokkos::View<FloatType*, layout, host_memory>;

        // Host mirrors populated by the deferred solidification calculation.
        floating_hostView eqFrac;

        // Initialize based on what needs to be calculated
        void Init(const CET_Fields& calc, const size_t numPoints, const std::string& label) {
            if (calc.eqFrac) eqFrac = floating_hostView(label + "_eqFrac", numPoints);
        }

        // Register with the writer
        void Register(const CET_Fields& output, Out::Writer<FloatType>& writer){
            if (output.eqFrac) writer.AddView("eqFrac",eqFrac);
        }
    };
    
    // Hook for determining equiaxed fraction of grains based on analytic model using solidification Gradient and Velocity
    template<typename FloatType>
    class CET_Hook {
        private:
            CET_Config<FloatType> config; // Hook-local CET settings parsed from the Jamie mode file.
            bool shouldSkip = true; // After check to see if it should be skipped or not
        public:
            // Variables for calculation. Editable by other callers.
            CET_Calc calc;

            // Kokkos views used by CET. Only allocated when needed.
            CET_Views<FloatType> volume;
            
            // Read the CET hook JSON object directly and retain the shared solidification hook reference.
            CET_Hook(const json& cetNode)
            {
                // Initialize config based on JSON node
                config.Init(cetNode);
            
                // Set calculation fields based on outputs
                calc.fields = config.outputs;
            }

            // Initialize views and change calc flags
            void Initialize(const Simdat<FloatType>& sim, Solidification_Hook<FloatType>& solidification_hook, Out::Writer<FloatType>& writer){
                // Initialize calculation object
                calc.Init();
                
                // Initialize views based on what needs to be calculated
                volume.Init(calc.fields, sim.domain.pnum, "CETHook");

                // Register writer based on what will be output
                volume.Register(config.outputs, writer);

                // Adjust other hooks based on what is needed
                if (calc.fields.eqFrac){
                    solidification_hook.calc.fields.G = true;
                    solidification_hook.calc.fields.V = true;
                }
            }
            
            // What to do after each timestep.
            void PostStep(const Solidification_Hook<FloatType>& solidification_hook, const Step_State<FloatType>& step, const Simdat<FloatType>& sim, const bool forced = false){
                
                // If not enabled, skip
                if (calc.skip) {return;}
                
                // Only do if an output step or we are forced
                if (step.isOutputStep() || forced) 
                {
                    // Get relevant input and derived variables
                    const int pointCount = sim.domain.pnum;
                    const FloatType N0 = config.params.N0;
                    const FloatType n = config.params.n;
                    const FloatType a = config.params.a;
                    const FloatType coeff = -static_cast<FloatType>(4.0) * static_cast<FloatType>(PI) * N0 / static_cast<FloatType>(3.0);
                    const FloatType nPlusOne = n + static_cast<FloatType>(1.0);
                    const FloatType vPow = static_cast<FloatType>(3.0) / n;
                    const auto G = solidification_hook.volume.G;
                    const auto V = solidification_hook.volume.V;
                    const auto eqFrac = volume.eqFrac;
                    
                    // Calculate volume in parallel
                    Kokkos::parallel_for(
                        "interface_cet_eqFrac",
                        Kokkos::RangePolicy<host_exe>(0, pointCount),
                        [=](const int p) {
                            const FloatType denom = G(p) * nPlusOne;
                            const FloatType denom3 = denom * denom * denom;
                            const FloatType vTerm = std::pow(a * V(p), vPow);
                            eqFrac(p) = static_cast<FloatType>(1.0) - std::exp(coeff * vTerm / denom3);
                        });
                    Kokkos::fence();
                }
            }
            
            // Finalize (does the same stuff as post-step)
            void Finalize(const Solidification_Hook<FloatType>& solidification_hook, const Step_State<FloatType>& step, const Simdat<FloatType>& sim){
                PostStep(solidification_hook, step, sim, true);
            }           
    };
}
