#pragma once

// Include internal dependencies
#include "impl/Initialization/Init.hpp"

// Include extenrnal dependencies
#include "Stork_Core.hpp"
#include <iostream>

namespace Condor::impl::Modes::Interface {
    // Usings
    using string = std::string;
    using json = nlohmann::json;

    template <typename FloatType>
    struct Interface_Config{  
        struct Settings {
            FloatType timestep = static_cast<FloatType>(0.0);
        } settings;
        struct Output {
            int frequency = 1;
            bool temperature = false;
        } output;
        struct Hooks {
            // Input-able, non-unique hooks
            json solidification;
            json meltpool;
            json cet;
            
            // Unique in-memory hooks
            bool do_rdf = false;
            bool do_srdf = false;
        } hooks;

        Interface_Config() = default;
        
        explicit Interface_Config(
            const json& mode_json,
            Stork::Structs::RDF_Dual<FloatType>* RDF_in,
            Stork::Structs::SRDF_Dual<FloatType>* SRDF_in) 
        {
            // Get the root of the Interface mode to initialize config on 
            const json& root = Init::GetNestedJson(mode_json, "Interface", true);

            // Get different sections
            const json& settingsNode = Init::GetNestedJson(root, "settings", true);
            const json& outputNode = Init::GetNestedJson(root, "output", true);
            const json& hooksNode = Init::GetNestedJson(root, "hooks", false);

            // Read the timestep once so Interface can reuse it across all hook logic.
            settings.timestep = Init::ReadValue<FloatType>(settingsNode, "timestep", static_cast<FloatType>(0.0), true);
            
            // Record whether coupled RDF or SRDF output were requested by the caller.
            hooks.do_rdf = (RDF_in != nullptr);
            hooks.do_srdf = (SRDF_in != nullptr);

            // If either RDF or SRDF is triggered, other hooks default to false (let rdf/srdf handle what needs to be computed)
            if (hooks.do_rdf || hooks.do_srdf)
            {
                // Report the coupled hook override so the runtime behavior is explicit in logs.
                std::cout << "In-memory Interface mode detected; disabling Interface file output and non-coupled hooks." << std::endl;

                // Force the output configuration required by the coupled RDF/SRDF path.
                output.frequency = 0;
                output.temperature = false;

                // Clear the optional hook objects so downstream parsing sees them as disabled.
                hooks.solidification = json::object();
                hooks.meltpool = json::object();
                hooks.cet = json::object();
            }
            else
            {
                // Read the shared output controls that Interface owns directly.
                output.frequency = Init::ReadValue<int>(outputNode, "frequency", 0, false);
                output.temperature = Init::ReadValue<bool>(outputNode, "temperature", false, false);
                
                // Cache each hook node so later setup can defer any heavier parsing work.
                hooks.solidification = Init::GetNestedJson(hooksNode, "solidification", false);
                hooks.meltpool = Init::GetNestedJson(hooksNode, "meltpool", false);
                hooks.cet = Init::GetNestedJson(hooksNode, "cet", false);
            }      
        }
    };
}
