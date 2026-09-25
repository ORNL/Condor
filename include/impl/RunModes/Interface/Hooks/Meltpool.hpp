#pragma once

#include "impl/Initialization/Init.hpp"
#include "impl/Meltpool/Dimensions.hpp"
#include "impl/Meltpool/Tracking.hpp"
#include "impl/Output/Writer.hpp"
#include "impl/RunModes/Interface/Structs.hpp"
#include "impl/Structs/Simdat.hpp"

#include <optional>

namespace Condor::impl::Modes::Interface::Hooks {

    using json = nlohmann::json;

    // Potential Fields
    struct Meltpool_Fields {
        bool MP_stats = false;
        bool Col_depth = false;

        // Member function to see if any are true
        [[nodiscard]] bool any() const {
            return MP_stats || Col_depth;
        }

        // Split up Stats
        bool MP_length = false;
        bool MP_width = false;
        bool MP_depth = false;
    };

    // Parameters
    template<typename FloatType>
    struct Meltpool_Parameters{
        string MP_Stats_Output;
    };

    // Struct for storing what is read in from json
    template<typename FloatType>
    struct Meltpool_Config {
        Meltpool_Fields outputs;
        Meltpool_Parameters<FloatType> params;

        void Init(const json& root) {
            if (!root.is_null() && !root.empty()) {
                // Split up meltpool json node
                const json& paramsNode = Init::GetNestedJson(root, "params", false);
                const json& outputsNode = Init::GetNestedJson(root, "outputs", true);

                // Read the params first
                params.MP_Stats_Output = Init::ReadValue<string>(paramsNode, "MP_Stats_Output", "Space", false);
                if (params.MP_Stats_Output != "Space" && params.MP_Stats_Output != "Time" && params.MP_Stats_Output != "SpaceAndTime")
                {
                    throw std::runtime_error("Invalid MP_Stats_Output value: " + params.MP_Stats_Output + ". Must be 'Space', 'Time', or 'SpaceAndTime'.");
                }
    
                // Read in the outputs
                outputs.MP_stats = Init::ReadValue<bool>(outputsNode, "MP_Stats", false, false);
                outputs.Col_depth = Init::ReadValue<bool>(outputsNode, "Col_Depth", false, false);

                // Initialize individual stats
                const bool wants_space = params.MP_Stats_Output == "Space" || params.MP_Stats_Output == "SpaceAndTime";

                outputs.MP_length = outputs.MP_stats && wants_space;
                outputs.MP_width = outputs.MP_stats && wants_space;
                outputs.MP_depth = outputs.MP_stats && wants_space;
            }
        }
    };

    struct Meltpool_Calc {
        // For fields (if needed or not)
        Meltpool_Fields fields;

        // If there is nothing to be calculated
        bool skip = true;

        void Init() {
            // Skip if no fields
            skip = !fields.any();
        }
    };

    template<typename FloatType>
    struct Meltpool_Views{
        // Quick access
        using floating_deviceView = Kokkos::View<FloatType*, layout, device_memory>;

        // Actual Views
        floating_deviceView MP_Length_MAX;
        floating_deviceView MP_Width_MAX;
        floating_deviceView MP_Depth_MAX;
        floating_deviceView Col_Depth_MAX;

        // Initialize based on what needs to be calculated
        void Init(const Meltpool_Fields& calc, const size_t numPoints, const std::string& label) {
            if (calc.MP_length) MP_Length_MAX = floating_deviceView(label + "_MP_Length_MAX", numPoints);
            if (calc.MP_width) MP_Width_MAX = floating_deviceView(label + "_MP_Width_MAX", numPoints);
            if (calc.MP_depth) MP_Depth_MAX = floating_deviceView(label + "_MP_Depth_MAX", numPoints);  
            if (calc.Col_depth) Col_Depth_MAX = floating_deviceView(label + "_Col_Depth_MAX", numPoints);   
        }

        // Register with the writer
        void Register(const Meltpool_Fields& output, Out::Writer<FloatType>& writer){
            if (output.MP_length) writer.AddView("MP_Length",MP_Length_MAX);
            if (output.MP_width) writer.AddView("MP_Width",MP_Width_MAX);
            if (output.MP_depth) writer.AddView("MP_Depth",MP_Depth_MAX);
            if (output.Col_depth) writer.AddView("Col_Depth",Col_Depth_MAX);
        }
    };

    template<typename FloatType>
    class Meltpool_Hook {        
        private:
            // Quick access
            using floating_deviceView = Kokkos::View<FloatType*, layout, device_memory>;
            Meltpool_Config<FloatType> config;
            int xnum, ynum, znum;

        public:
            // Variables for calculation. Editable by other hooks.
            Meltpool_Calc calc;

            // Views for the meltpool
            Meltpool_Views<FloatType> volume;

            // Calculator for meltpool dimensions
            std::optional<Meltpool::Dimensions<FloatType>> dim_calculator;

            // Read the Meltpool hook JSON object
            Meltpool_Hook(const json& root) {
                if (!root.is_null() && !root.empty()){
                    // Read in JSON and set config
                    config.Init(root);
                    // Set calculation fields to initially coincide with outputs
                    calc.fields = config.outputs;
                }   
            }

            void Initialize(
                const Simdat<FloatType>& sim,
                Condor::impl::Meltpool::Tracking<FloatType>& tracking,
                Out::Writer<FloatType>& writer)
            {
                // Initialize calculation
                calc.Init();
                if (calc.skip) {return;}

                // Initialize constants
                xnum = sim.domain.xnum;
                ynum = sim.domain.ynum;
                znum = sim.domain.znum;

                // Initialize views based on what needs to be calculated
                volume.Init(calc.fields, sim.domain.pnum, "MeltpoolHook");

                // Register writer based on what will be output
                volume.Register(config.outputs, writer);

                // Initialize calculator with simdat
                Meltpool::Dimensions<FloatType>& dimensions = dim_calculator.emplace(sim);
                dimensions.calc.MP_Len_or_Wid = calc.fields.MP_stats;
                dimensions.calc.MP_Depth = calc.fields.MP_stats && sim.domain.znum > 1;
                dimensions.calc.Col_Depth = calc.fields.Col_depth;
                dimensions.Initialize();
            }
        
        public:
            void MapToMeltpool(const Condor::impl::Meltpool::Tracking<FloatType>& tracking){
                const bool MP_L = calc.fields.MP_length;
                const bool MP_W = calc.fields.MP_width;
                const bool MP_D = calc.fields.MP_depth;
                const bool C_D = calc.fields.Col_depth;
                const bool MP_ANY = MP_L || MP_W || MP_D;

                if (!(C_D || MP_ANY)) { return; }

                const int active_count = tracking.ActiveColumnCount();
                if (active_count <= 0) { return; }

                const int_2D_deviceView point_depth = tracking.CurrentDepth();
                const int_deviceView active_cols = tracking.ActiveColumns();
                const Meltpool::Dimensions<FloatType>& dimensions = *dim_calculator;
                const floating_deviceView Col_Depth = dimensions.column_depth_d;
                floating_deviceView Col_Depth_MAX = volume.Col_Depth_MAX;

                const int component_count = MP_ANY ? dimensions.component_count_h : 0;
                const int_deviceView meltpool_points = dimensions.component_points_d;
                const int_deviceView meltpool_offsets = dimensions.component_offsets_d;
                const floating_deviceView MP_length = dimensions.component_length_d;
                const floating_deviceView MP_width = dimensions.component_width_d;
                const floating_deviceView MP_depth = dimensions.component_depth_d;

                floating_deviceView MP_Length_MAX = volume.MP_Length_MAX;
                floating_deviceView MP_Width_MAX = volume.MP_Width_MAX;
                floating_deviceView MP_Depth_MAX = volume.MP_Depth_MAX;
                const int ynum_local = ynum;
                const int znum_local = znum;

                Kokkos::parallel_for(
                    "meltpool_hook_map",
                    Kokkos::RangePolicy<device_exe>(0, active_count),
                    KOKKOS_LAMBDA(const int q) {
                        if (C_D) {
                            const int v2d = active_cols(q);
                            const int i = v2d / ynum_local;
                            const int j = v2d % ynum_local;
                            const int depth_active = point_depth(i, j);
                            const FloatType depth_active_interp = Col_Depth(v2d);

                            for (int d = 0; d < depth_active; d++) {
                                const int p = v2d * znum_local + (znum_local - 1 - d);
                                Col_Depth_MAX(p) =
                                    (Col_Depth_MAX(p) > depth_active_interp)
                                        ? Col_Depth_MAX(p)
                                        : depth_active_interp;
                            }
                        }

                        if (MP_ANY && component_count > 0) {
                            const int v2d = meltpool_points(q);
                            const int i = v2d / ynum_local;
                            const int j = v2d % ynum_local;
                            const int depth_active = point_depth(i, j);

                            int mp_num = 0;
                            while ((mp_num + 1) < component_count && meltpool_offsets(mp_num + 1) <= q) {
                                mp_num++;
                            }

                            const FloatType L = MP_L ? MP_length(mp_num) : static_cast<FloatType>(0.0);
                            const FloatType W = MP_W ? MP_width(mp_num) : static_cast<FloatType>(0.0);
                            const FloatType D = MP_D ? MP_depth(mp_num) : static_cast<FloatType>(0.0);

                            for (int d = 0; d < depth_active; d++) {
                                const int p = v2d * znum_local + (znum_local - 1 - d);

                                if (MP_L) {
                                    MP_Length_MAX(p) = (MP_Length_MAX(p) > L) ? MP_Length_MAX(p) : L;
                                }
                                if (MP_W) {
                                    MP_Width_MAX(p) = (MP_Width_MAX(p) > W) ? MP_Width_MAX(p) : W;
                                }
                                if (MP_D) {
                                    MP_Depth_MAX(p) = (MP_Depth_MAX(p) > D) ? MP_Depth_MAX(p) : D;
                                }
                            }
                        }
                    });

                Kokkos::fence();
            }

            void PostStep(
                const Condor::impl::Meltpool::Tracking<FloatType>& tracking,
                const Simdat<FloatType>& sim,
                const Step_State<FloatType>& step,
                const Thermal_State<FloatType>& thermal)
            {
                // If skipping, do nothing
                if (calc.skip) { return; }
                
                // Measure meltpool dimensions
                dim_calculator->Measure(tracking, thermal, sim, step);

                // Record time series data if needed (either Time or SpaceAndTime)
                if (config.params.MP_Stats_Output == "Time" || config.params.MP_Stats_Output == "SpaceAndTime") {
                    dim_calculator->RecordTimeSeries(step, false);
                }

                // Use the measurments for the volume fields (either Space or SpaceAndTime)
                if (config.params.MP_Stats_Output == "Space" || config.params.MP_Stats_Output == "SpaceAndTime") {
                    MapToMeltpool(tracking);
                }
            }

            void Finalize(
                const Condor::impl::Meltpool::Tracking<FloatType>& tracking,
                const Simdat<FloatType>& sim,
                const Step_State<FloatType>& step)
            {
                // If skipping, do nothing
                if (calc.skip) { return; }
                if (config.params.MP_Stats_Output != "Time" && config.params.MP_Stats_Output != "SpaceAndTime") { return; }

                const string output_file = sim.files.dataDir + "/" + sim.files.name + "_MP_Stats.csv";
                std::ofstream file(output_file);
                if (!file.is_open()) {
                    throw std::runtime_error("Could not open meltpool CSV file: " + output_file);
                }
                
                const Meltpool::Dimensions<FloatType>& dimensions = *dim_calculator;
                
                file << std::setprecision(std::numeric_limits<FloatType>::max_digits10);
                file << "t,iter,MP_Count,MP_Length,MP_Width";
                if (dimensions.calc.MP_Depth) { file << ",MP_Depth"; }
                file << "\n";
                
                for (const auto& row : dimensions.mp_stats_rows) {
                    file << row.t << ","
                         << row.iter << ","
                         << row.count << ","
                         << row.length << ","
                         << row.width;
                    if (dimensions.calc.MP_Depth) { file << "," << row.depth; }
                    file << "\n";
                }
            }
    };
}
