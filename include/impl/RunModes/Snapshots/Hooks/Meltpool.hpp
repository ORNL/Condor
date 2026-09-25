#pragma once

#include "impl/Initialization/Init.hpp"
#include "impl/Meltpool/Dimensions.hpp"
#include "impl/Meltpool/Tracking.hpp"
#include "impl/Output/Writer.hpp"
#include "impl/RunModes/Snapshots/Structs.hpp"
#include "impl/Structs/Simdat.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace Condor::impl::Modes::Snapshots::Hooks {

    using json = nlohmann::json;

    // Potential Fields
    struct Meltpool_Fields {
        bool MP_stats = false;
        bool Col_depth = false;

        [[nodiscard]] bool any() const {
            return MP_stats || Col_depth;
        }

        [[nodiscard]] bool any_MP() const {
            return MP_stats;
        }
    };

    // Struct for storing what is read in from json
    template<typename FloatType>
    struct Meltpool_Config {
        Meltpool_Fields outputs;

        void Init(const json& root) {
            if (!root.is_null() && !root.empty()) {
                // Split up meltpool json node
                const json& outputsNode = Init::GetNestedJson(root, "outputs", true);

                // Read in the outputs
                outputs.MP_stats = Init::ReadValue<bool>(outputsNode, "MP_Stats", false, false);
                outputs.Col_depth = Init::ReadValue<bool>(outputsNode, "Col_Depth", false, false);
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
    struct Meltpool_Views {
        // Quick access
        using floating_deviceView = Kokkos::View<FloatType*, layout, device_memory>;

        // Actual Views
        floating_deviceView Col_Depth;

        // Initialize based on what needs to be calculated
        void Init(const Meltpool_Fields& calc, const size_t numPoints, const std::string& label) {
            if (calc.Col_depth) Col_Depth = floating_deviceView(label + "_Col_Depth", numPoints);
        }

        // Register with the writer
        void Register(const Meltpool_Fields& output, Out::Writer<FloatType>& writer) {
            if (output.Col_depth) writer.AddView("Col_Depth", Col_Depth);
        }
    };

    template<typename FloatType>
    class Meltpool_Hook {
        private:
            // Quick access
            using floating_deviceView = Kokkos::View<FloatType*, layout, device_memory>;
            Meltpool_Config<FloatType> config;
            int ynum = 0;
            int znum = 0;

            void ClearViews() {
                if (calc.fields.Col_depth) {
                    Kokkos::deep_copy(volume.Col_Depth, static_cast<FloatType>(0.0));
                }
            }

        public:
            template<typename DimensionsType>
            void MapColDepthToGrid(
                const int active_count,
                const int_deviceView& active_columns,
                const int_2D_deviceView& depth_view,
                const DimensionsType& dimensions)
            {
                if (!calc.fields.Col_depth) { return; }

                if (active_count <= 0) { return; }

                const int_2D_deviceView point_depth = depth_view;
                const int_deviceView active_cols = active_columns;
                const floating_deviceView Col_Depth_Source = dimensions.column_depth_d;
                floating_deviceView Col_Depth = volume.Col_Depth;
                const int ynum_local = ynum;
                const int znum_local = znum;

                Kokkos::parallel_for(
                    "snapshots_meltpool_hook_col_depth_map",
                    Kokkos::RangePolicy<device_exe>(0, active_count),
                    KOKKOS_LAMBDA(const int q) {
                        const int v2d = active_cols(q);
                        const int i = v2d / ynum_local;
                        const int j = v2d % ynum_local;
                        const int depth_active = point_depth(i, j);
                        const FloatType depth_active_interp = Col_Depth_Source(v2d);

                        for (int d = 0; d < depth_active; d++) {
                            const int p = v2d * znum_local + (znum_local - 1 - d);
                            Col_Depth(p) = depth_active_interp;
                        }
                    });
                Kokkos::fence();
            }

        public:
            // Variables for calculation. Editable by other hooks.
            Meltpool_Calc calc;

            // Views for the meltpool
            Meltpool_Views<FloatType> volume;

            // Calculator for snapshot meltpool dimensions
            std::optional<Meltpool::Dimensions<FloatType>> dim_calculator;

            // Read the Meltpool hook JSON object
            Meltpool_Hook(const json& root) {
                if (!root.is_null() && !root.empty()) {
                    // Read in JSON and set config
                    config.Init(root);
                    // Set calculation fields to initially coincide with outputs
                    calc.fields = config.outputs;
                }
            }

            void Initialize(const Simdat<FloatType>& sim, Out::Writer<FloatType>& writer) {
                // Initialize calculation
                calc.Init();
                
                if (calc.skip) { return; }

                // Initialize constants
                ynum = sim.domain.ynum;
                znum = sim.domain.znum;

                // Initialize views based on what needs to be calculated
                volume.Init(calc.fields, sim.domain.pnum, "SnapshotsMeltpoolHook");
                ClearViews();

                // Register writer based on what will be output
                volume.Register(config.outputs, writer);

                // Initialize calculators with simdat
                Meltpool::Dimensions<FloatType>& dimensions = dim_calculator.emplace(sim);
                dimensions.calc.MP_Len_or_Wid = calc.fields.MP_stats;
                dimensions.calc.MP_Depth = calc.fields.MP_stats && sim.domain.znum > 1;
                dimensions.calc.Col_Depth = calc.fields.Col_depth;
                dimensions.Initialize();

            }

            void PostStep(
                const int active_count,
                const int_deviceView& active_columns,
                const int_2D_deviceView& depth_view,
                const Simdat<FloatType>& sim,
                Snapshots_State<FloatType>& state)
            {
                // If skipping, do nothing
                if (calc.skip) { return; }

                ClearViews();

                // Measure meltpool dimensions from the selected snapshot views.
                Meltpool::Dimensions<FloatType>& dimensions = *dim_calculator;
                dimensions.Measure(
                    active_count,
                    active_columns,
                    depth_view,
                    state.thermal.T_cur,
                    state.thermal.isLiq_cur,
                    sim
                );

                // Record time series data and map column depth to the grid for output.
                dimensions.RecordTimeSeries(state.step, true);
                MapColDepthToGrid(active_count, active_columns, depth_view, dimensions);
            }

            void Finalize(const Simdat<FloatType>& sim) const {
                if (!calc.fields.any_MP()) { return; }

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
