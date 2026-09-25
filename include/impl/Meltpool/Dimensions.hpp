#pragma once

#include "Definitions.hpp"
#include "impl/Meltpool/Tracking.hpp"
#include "impl/Output/Writer.hpp"
#include "impl/Structs/Simdat.hpp"
#include "impl/Structs/StateStructs.hpp"
#include "impl/Utility/Grid.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace Condor::impl::Meltpool {

    struct Dimensions_Calc{
        // If it should be skipped
        bool skip = true;

        // Larger calculation decisions
        bool any_MP = false;

        // Outputs
        bool MP_Len_or_Wid = false;
        bool MP_Depth = false;
        bool Col_Depth = false;

        // Initialize
        void Init(){
            // If we have any MP to do
            any_MP = MP_Len_or_Wid || MP_Depth;
            // If anything is needed, don't skip
            skip = !(Col_Depth || MP_Depth || MP_Len_or_Wid);
        }
    };

    template<typename FloatType>
    class Dimensions {
        private:
            using floating_deviceView = Kokkos::View<FloatType*, layout, device_memory>;
            using floating_hostView = Kokkos::View<FloatType*, layout, host_memory>;
            static constexpr int max_components = 64;
                        
            // Dims
            const int xnum, ynum, znum;
            const FloatType x0, y0, res;
            const int total_columns, boundary_capacity;
            
            // Helper function to clamp values between [0,1] (for interpolation)
            KOKKOS_INLINE_FUNCTION
            static FloatType Clamp01(const FloatType value) {
                if (value < static_cast<FloatType>(0.0)) {
                    return static_cast<FloatType>(0.0);
                }
                if (value > static_cast<FloatType>(1.0)) {
                    return static_cast<FloatType>(1.0);
                }
                return value;
            }            
        public:
            // Dimensions Calc
            Dimensions_Calc calc;

            // Label of component (needs reset)
            int_deviceView component_label_d;
            // Count of boundary cells (needs reset)
            int_deviceView boundary_count_d;
            // What cell owns the boundary
            int_deviceView boundary_owner_d;
            // x and y values of boundary cells
            floating_deviceView boundary_x_d;
            floating_deviceView boundary_y_d;
            // maximum depth at column
            floating_deviceView column_depth_d;
            // length, width, depth per component (distinct meltpool)
            floating_deviceView component_length_d;
            floating_deviceView component_width_d;
            floating_deviceView component_depth_d;
            floating_deviceView rectangle_best_area_d;
            floating_deviceView rectangle_best_length_d;
            floating_deviceView rectangle_best_width_d;
            floating_hostView component_length_h;
            floating_hostView component_width_h;
            floating_hostView component_depth_h;
            // Dense roots and packed points for up to max_components meltpools.
            int_deviceView component_roots_d;
            int_deviceView component_count_d;
            int_deviceView component_count_by_root_d;
            int_deviceView component_offsets_d;
            int_deviceView component_points_d;
            int_deviceView rectangle_best_sample_d;
            int_deviceView rectangle_best_lock_d;
            // Host counter
            int component_count_h = 0;
            
            // Initialization
            Dimensions(const Simdat<FloatType>& sim):
                xnum(sim.domain.xnum),
                ynum(sim.domain.ynum),
                znum(sim.domain.znum),
                x0((sim.domain.xnum == 1) ? sim.domain.xmax : sim.domain.xmin),
                y0((sim.domain.ynum == 1) ? sim.domain.ymax : sim.domain.ymin),
                res(sim.domain.xres),
                total_columns(sim.domain.xnum*sim.domain.ynum),
                boundary_capacity(4 * total_columns)
            {}; 

            // Initialize the necessary views
            void Initialize(){
                // Initialize calculation object (everything else should have altered it by this point)
                calc.Init();
                
                // If there is nothing to do, don't initialize
                if (calc.skip){return;}
                
                // Vertical depth at each column (reset=No)
                column_depth_d = floating_deviceView(Kokkos::ViewAllocateWithoutInitializing("meltpool_dimensions_column_depth"), total_columns);

                // Everything below is for meltpool calculation
                if (calc.any_MP){
                    // Label of v2d -> which root each v2d belongs to (reset=No)
                    component_label_d = int_deviceView(Kokkos::ViewAllocateWithoutInitializing("meltpool_dimensions_component_label"), total_columns);
                    // Boundary information (reset=Count Only)
                    boundary_count_d = int_deviceView(Kokkos::ViewAllocateWithoutInitializing("meltpool_dimensions_boundary_count"), 1);
                    boundary_owner_d = int_deviceView(Kokkos::ViewAllocateWithoutInitializing("meltpool_dimensions_boundary_owner"), boundary_capacity);
                    boundary_x_d = floating_deviceView(Kokkos::ViewAllocateWithoutInitializing("meltpool_dimensions_boundary_x"), boundary_capacity);
                    boundary_y_d = floating_deviceView(Kokkos::ViewAllocateWithoutInitializing("meltpool_dimensions_boundary_y"), boundary_capacity);
                    // Count of each meltpool (reset=Yes)
                    component_count_d = int_deviceView(Kokkos::ViewAllocateWithoutInitializing("meltpool_dimensions_component_count"), 1);
                    // Count of each meltpool by root (reset=Partial (only root numbers))
                    component_count_by_root_d = int_deviceView("meltpool_dimensions_component_count_by_root", total_columns);
                    // The roots of each meltpool (reset=No).
                    component_roots_d = int_deviceView(Kokkos::ViewAllocateWithoutInitializing("meltpool_dimensions_component_roots"), max_components);
                    // 1D dense view of meltpool points contiguous by meltpool (reset=No)
                    component_points_d = int_deviceView(Kokkos::ViewAllocateWithoutInitializing("meltpool_dimensions_component_points"), total_columns);
                    // Offset for 1D dense view per root (reset=No)
                    component_offsets_d = int_deviceView(Kokkos::ViewAllocateWithoutInitializing("meltpool_dimensions_component_offsets"), max_components + 1);
                    // Meltpool dimensions (reset=None)
                    component_length_d = floating_deviceView(Kokkos::ViewAllocateWithoutInitializing("meltpool_dimensions_component_length"), max_components);
                    component_width_d = floating_deviceView(Kokkos::ViewAllocateWithoutInitializing("meltpool_dimensions_component_width"), max_components);
                    component_depth_d = floating_deviceView(Kokkos::ViewAllocateWithoutInitializing("meltpool_dimensions_component_depth"), max_components);
                    rectangle_best_area_d = floating_deviceView(Kokkos::ViewAllocateWithoutInitializing("meltpool_dimensions_rectangle_best_area"), 2 * max_components);
                    rectangle_best_length_d = floating_deviceView(Kokkos::ViewAllocateWithoutInitializing("meltpool_dimensions_rectangle_best_length"), 2 * max_components);
                    rectangle_best_width_d = floating_deviceView(Kokkos::ViewAllocateWithoutInitializing("meltpool_dimensions_rectangle_best_width"), 2 * max_components);
                    rectangle_best_sample_d = int_deviceView(Kokkos::ViewAllocateWithoutInitializing("meltpool_dimensions_rectangle_best_sample"), 2 * max_components);
                    rectangle_best_lock_d = int_deviceView(Kokkos::ViewAllocateWithoutInitializing("meltpool_dimensions_rectangle_best_lock"), 2 * max_components);
                    component_length_h = floating_hostView(Kokkos::ViewAllocateWithoutInitializing("meltpool_dimensions_component_length_h"), max_components);
                    component_width_h = floating_hostView(Kokkos::ViewAllocateWithoutInitializing("meltpool_dimensions_component_width_h"), max_components);
                    component_depth_h = floating_hostView(Kokkos::ViewAllocateWithoutInitializing("meltpool_dimensions_component_depth_h"), max_components);
                }                
            }

            // Reset everything necessary for calculation
            void ResetScratch(){
                // Only need to reset meltpool calculations
                if (!calc.any_MP) {
                    return;
                }

                // Get local views
                const int_deviceView& component_roots = component_roots_d;
                const int_deviceView& component_count_by_root = component_count_by_root_d;

                // Reset component counts only at previous roots
                Kokkos::parallel_for(
                    "meltpool_dimensions_reset_root_counts",
                    Kokkos::RangePolicy<device_exe>(0, component_count_h),
                    KOKKOS_LAMBDA(const int c) {
                        component_count_by_root(component_roots(c)) = 0;
                    });
                
                // Zero the boundary count and component counts
                Kokkos::deep_copy(component_count_d, 0);
                Kokkos::deep_copy(boundary_count_d, 0);
                component_count_h = 0;
            }
        public:           
            // Interpolate the geometry and find "interpolated bounds"
            template<bool projectToVolume>
            void BuildInterpolatedGeometry(
                const int active_count,
                const int_deviceView& active_columns_in,
                const int_2D_deviceView& depth_view_in,
                const floating_deviceView& T_view_in,
                const FloatType T_liq_in
            )
            {
                // Get active columns and depth from meltpool tracking
                const int_deviceView& active_columns = active_columns_in;
                const int_2D_deviceView& depth_view = depth_view_in;
            
                // Get temperature from thermal state
                const floating_deviceView& T_cur = T_view_in;

                // Critical temperature from input
                const FloatType T_liq = T_liq_in;

                // Local variables
                const int_deviceView component_label = component_label_d;
                const int_deviceView boundary_owner = boundary_owner_d;
                const int_deviceView boundary_count = boundary_count_d;
                const floating_deviceView boundary_x = boundary_x_d;
                const floating_deviceView boundary_y = boundary_y_d;
                const floating_deviceView column_depth = column_depth_d;
                const int xnum_local = this->xnum;
                const int ynum_local = this->ynum;
                const int znum_local = this->znum;
                const FloatType res = this->res;
                const FloatType x0 = this->x0;
                const FloatType y0 = this->y0;

                // Loop over all the active columns
                Kokkos::parallel_for(
                    "meltpool_dimensions_interpolate",
                    Kokkos::RangePolicy<device_exe>(0, active_count),
                    KOKKOS_LAMBDA(const int q) {
                        // Get column and liquid depth
                        const int v2d = active_columns(q);
                        const int i = v2d / ynum_local;
                        const int j = v2d % ynum_local;
                        const int p_top = Grid::ijk_to_p(i, j, znum_local - 1, ynum_local, znum_local);
                        const int liquid_count = depth_view(i, j);

                        // If on bottom surface, set depth to max
                        if (liquid_count == znum_local) {
                            column_depth(v2d) = (static_cast<FloatType>(znum_local) * res);
                        }
                        // Otherwise, calculate interpolated depth
                        else{
                            // Get the index of the bottom liquid and top solid points
                            const int p_liq = Grid::ijk_to_p(i, j, znum_local - liquid_count, ynum_local, znum_local);
                            const int p_sol = Grid::ijk_to_p(i, j, znum_local - 1 - liquid_count, ynum_local, znum_local);

                            // Find depth
                            const FloatType T_liq_point = T_cur(p_liq);
                            const FloatType T_sol_point = T_cur(p_sol);
                            const FloatType depth_denom = T_sol_point - T_liq_point;
                            const FloatType depth_frac = Clamp01((T_liq - T_liq_point) / depth_denom);
                            column_depth(v2d) = (static_cast<FloatType>(liquid_count - 1) + depth_frac) * res;
                        }

                        // Only do the rest if we need meltpool volume stuff
                        if constexpr(!projectToVolume){return;}
                        
                        // Set component label
                        component_label(v2d) = v2d;

                        // Compute positions and neighbors
                        const FloatType T_center = T_cur(p_top);
                        const FloatType x_center = x0 + static_cast<FloatType>(i) * res;
                        const FloatType y_center = y0 + static_cast<FloatType>(j) * res;

                        // Neighbor column indices
                        const int neighbor_v2d[4] = {
                            (i > 0) ? (v2d - ynum_local) : -1,
                            (i + 1 < xnum_local) ? (v2d + ynum_local) : -1,
                            (j > 0) ? (v2d - 1) : -1,
                            (j + 1 < ynum_local) ? (v2d + 1) : -1
                        };
                        // Neighbor shifts
                        const int neighbor_i[4] = {i - 1, i + 1, i, i};
                        const int neighbor_j[4] = {j, j, j - 1, j + 1};
                        const FloatType x_sign[4] = {
                            static_cast<FloatType>(-1.0),
                            static_cast<FloatType>(1.0),
                            static_cast<FloatType>(0.0),
                            static_cast<FloatType>(0.0)
                        };
                        const FloatType y_sign[4] = {
                            static_cast<FloatType>(0.0),
                            static_cast<FloatType>(0.0),
                            static_cast<FloatType>(-1.0),
                            static_cast<FloatType>(1.0)
                        };

                        for (int face = 0; face < 4; face++) {
                            const int nv2d = neighbor_v2d[face];
                            
                            // If on the edge, set boundary
                            if (nv2d < 0) {
                                // Increment the number of boundaries and set values
                                const int pos = Kokkos::atomic_fetch_add(&boundary_count(0), 1);
                                boundary_x(pos) = x_center;
                                boundary_y(pos) = y_center;
                                boundary_owner(pos) = v2d;
                            }
                            // If neighbor is liquid, then we are on the interior and should skip
                            else if (depth_view(neighbor_i[face], neighbor_j[face])){
                                continue;
                            }
                            // Otherwise interpolate
                            else{
                                // Get neighbor position and temperature
                                const int p_neighbor = Grid::ijk_to_p(neighbor_i[face], neighbor_j[face], znum_local - 1, ynum_local, znum_local);
                                const FloatType T_neighbor = T_cur(p_neighbor);
                                // Interpolate
                                const FloatType denom = T_neighbor - T_center;
                                const FloatType frac = Clamp01((T_liq - T_center) / denom);
                                // Increment the number of boundaries
                                const int pos = Kokkos::atomic_fetch_add(&boundary_count(0), 1);
                                // Set boundary position and owner
                                boundary_x(pos) = x_center + x_sign[face] * frac * res;
                                boundary_y(pos) = y_center + y_sign[face] * frac * res;
                                boundary_owner(pos) = v2d;
                            }              
                        }
                    });
                Kokkos::fence();
            }

            // Determine which columns belong to which meltpools (each column will have a "root")
            int FindConnectedComponents(
                const int active_count, 
                const int_deviceView& active_columns_in,
                const int_2D_deviceView& depth_view_in
            ) {                
                // Get the active columns from meltpool tracking
                const int_deviceView& active_columns = active_columns_in; //tracking.ActiveColumns();
                const int_2D_deviceView& depth_view = depth_view_in; //tracking.CurrentDepth();
                
                // Get local views
                const int_deviceView component_label_local = component_label_d;
                const int_deviceView component_roots = component_roots_d;
                const int_deviceView component_count = component_count_d;
                const int_deviceView component_count_by_root = component_count_by_root_d;
                const int xnum_local = this->xnum;
                const int ynum_local = this->ynum;
                
                // Initially label all columns with their own ID
                Kokkos::parallel_for(
                    "meltpool_dimensions_label_init",
                    Kokkos::RangePolicy<device_exe>(0, active_count),
                    KOKKOS_LAMBDA(const int q) {
                        const int v2d = active_columns(q);
                        component_label_local(v2d) = v2d;
                    }
                );
                
                // Loop over all columns
                Kokkos::parallel_for(
                    "meltpool_dimensions_label_union",
                    Kokkos::RangePolicy<device_exe>(0, active_count),
                    KOKKOS_LAMBDA(const int q) {
                        // Get column numbers
                        const int v2d = active_columns(q);
                        const int i = v2d / ynum_local;
                        const int j = v2d % ynum_local;
                        
                        // Construct top-right neighbors
                        const int neighbors[2] = {
                            (i + 1 < xnum_local) ? (v2d + ynum_local) : -1,
                            (j + 1 < ynum_local) ? (v2d + 1) : -1
                        };
                        const int neighbor_i[2] = {i + 1, i};
                        const int neighbor_j[2] = {j, j + 1};

                        // Loop over neighbors
                        for (int n = 0; n < 2; n++) {
                            const int neighbor = neighbors[n];

                            // Skip if neighbor is OOB or not considered
                            if (neighbor < 0 || !depth_view(neighbor_i[n], neighbor_j[n])) {
                                continue;
                            }

                            // Continue until converged to make the connected graph
                            while (true) {
                                int root_a = v2d;
                                while (component_label_local(root_a) != root_a) {
                                    root_a = component_label_local(root_a);
                                }

                                int root_b = neighbor;
                                while (component_label_local(root_b) != root_b) {
                                    root_b = component_label_local(root_b);
                                }
                                
                                // If neighbor and self have the same value, break
                                if (root_a == root_b) {
                                    break;
                                }
                        
                                const int low_root = (root_a < root_b) ? root_a : root_b;
                                const int high_root = (root_a < root_b) ? root_b : root_a;
                                const int old_root = Kokkos::atomic_compare_exchange(
                                    &component_label_local(high_root),
                                    high_root,
                                    low_root);

                                if (old_root == high_root) {
                                    break;
                                }
                            }
                        }
                    });
                
                // Compress the graph 
                // Calculate "root counts" and "distinct roots"
                Kokkos::parallel_for(
                    "meltpool_dimensions_label_compress",
                    Kokkos::RangePolicy<device_exe>(0, active_count),
                    KOKKOS_LAMBDA(const int q) {
                        const int v2d = active_columns(q);
                        int root = v2d;
                        while (component_label_local(root) != root) {
                            root = component_label_local(root);
                        }

                        int current = v2d;
                        while (component_label_local(current) != current) {
                            const int next = component_label_local(current);
                            component_label_local(current) = root;
                            current = next;
                        }
                        component_label_local(v2d) = root;
                        // After we find the root, add to the count.
                        const int old_count = Kokkos::atomic_fetch_add(&component_count_by_root(root), 1);
                        // If we are the first to count it, add it as a component.
                        if (old_count == 0) {
                            const int component_idx = Kokkos::atomic_fetch_add(&component_count(0), 1);
                            if (component_idx < max_components) {
                                component_roots(component_idx) = root;
                            }
                        }
                    });
                Kokkos::fence();

                int_hostView component_count_host = Kokkos::create_mirror_view(component_count_d);
                Kokkos::deep_copy(component_count_host, component_count_d);
                component_count_h = component_count_host(0);
                return component_count_h;
            }

            // Helper struct for compress parallel scan        
            struct ComponentStuff{
                int_deviceView active_columns;
                int_deviceView component_label;
                int_deviceView component_roots;
                int_deviceView component_count_by_root;
                int_deviceView component_offsets;
                int_deviceView component_points;
                int component_count;
            };

            template<int I>
            struct ComponentPackingScan {
                struct value_type {
                    int count[I];
                };

                // Stuct for easy access
                ComponentStuff stuff;

                KOKKOS_INLINE_FUNCTION
                void init(value_type& v) const {
                    for (int c = 0; c < I; ++c) v.count[c] = 0;
                }

                KOKKOS_INLINE_FUNCTION
                void join(value_type& dst, const value_type& src) const {
                    for (int c = 0; c < I; ++c) dst.count[c] += src.count[c];
                }

                KOKKOS_INLINE_FUNCTION
                void operator()(const int q, value_type& counts, const bool isFinal) const {
                    const int v2d = stuff.active_columns(q);
                    const int myRoot = stuff.component_label(v2d);

                    if (isFinal && q == 0) {
                        stuff.component_offsets(0) = 0;
                    }

                    int base = 0;
                    for (int c = 0; c < stuff.component_count; ++c) {
                        const int compRoot = stuff.component_roots(c);
                        const int compCount = stuff.component_count_by_root(compRoot);

                        if (compRoot == myRoot) {
                            if (isFinal) {
                                if (counts.count[c] == 0) {
                                    stuff.component_offsets(c + 1) = base + compCount;
                                }
                                stuff.component_points(base + counts.count[c]) = v2d;
                            }

                            counts.count[c] += 1;
                            break;
                        }

                        base += compCount;
                    }
                }
            };

            // Takes the "roots" and "root counts" and returns a 1D array of columns indices and offsets per root
            // Basically constructs "contiguous roots"
            int CompressComponents(
                const int active_count,
                const int_deviceView& active_columns,
                const int component_count
            ) {
                // Make "Component Stuff" Container
                const ComponentStuff stuff_local{
                    active_columns,
                    component_label_d,
                    component_roots_d,
                    component_count_by_root_d,
                    component_offsets_d,
                    component_points_d,
                    component_count
                };

                const Kokkos::RangePolicy<device_exe> policy(0, active_count);
                if (component_count <= 4) {
                    Kokkos::parallel_scan(policy, ComponentPackingScan<4>{stuff_local});
                }
                else if (component_count <= 8){
                    Kokkos::parallel_scan(policy, ComponentPackingScan<8>{stuff_local});
                }
                else if (component_count <= 16){
                    Kokkos::parallel_scan(policy, ComponentPackingScan<16>{stuff_local});
                }
                else if (component_count <= 32){
                    Kokkos::parallel_scan(policy, ComponentPackingScan<32>{stuff_local});
                }
                else if (component_count <= 64){
                    Kokkos::parallel_scan(policy, ComponentPackingScan<64>{stuff_local});
                }
                else {
                    throw std::runtime_error("Meltpool dimensions supports at most 64 separate meltpools.");
                }

                Kokkos::fence();

                return component_count;
            }

            struct ProjectionBounds {
                FloatType min_u;
                FloatType max_u;
                FloatType min_v;
                FloatType max_v;
                int found;
            };

            struct ProjectionBoundsReducer {
                using reducer = ProjectionBoundsReducer;
                using value_type = ProjectionBounds;
                value_type* value;

                KOKKOS_INLINE_FUNCTION
                ProjectionBoundsReducer(value_type& value_in) : value(&value_in) {}

                KOKKOS_INLINE_FUNCTION
                void init(value_type& value) const {
                    const FloatType large = static_cast<FloatType>(1.0e30);
                    value.min_u = large;
                    value.max_u = -large;
                    value.min_v = large;
                    value.max_v = -large;
                    value.found = 0;
                }

                KOKKOS_INLINE_FUNCTION
                void join(value_type& dst, const value_type& src) const {
                    if (!src.found) {
                        return;
                    }
                    if (!dst.found) {
                        dst = src;
                        return;
                    }
                    dst.min_u = (dst.min_u < src.min_u) ? dst.min_u : src.min_u;
                    dst.max_u = (dst.max_u > src.max_u) ? dst.max_u : src.max_u;
                    dst.min_v = (dst.min_v < src.min_v) ? dst.min_v : src.min_v;
                    dst.max_v = (dst.max_v > src.max_v) ? dst.max_v : src.max_v;
                    dst.found = 1;
                }

                KOKKOS_INLINE_FUNCTION
                value_type& reference() const {
                    return *value;
                }

                KOKKOS_INLINE_FUNCTION
                bool references_scalar() const {
                    return true;
                }
            };


            // Find the rectangles with the minimum area, per meltpool (principle direction -> length and width)
            void SearchRotatedRectangles(const int component_count) {
                // Capture scratch and output views used by the rectangle kernels.
                const int_deviceView component_label = component_label_d;
                const int_deviceView boundary_owner = boundary_owner_d;
                const int_deviceView boundary_count = boundary_count_d;
                const floating_deviceView boundary_x = boundary_x_d;
                const floating_deviceView boundary_y = boundary_y_d;
                const floating_deviceView component_length = component_length_d;
                const floating_deviceView component_width = component_width_d;
                const int_deviceView component_roots = component_roots_d;
                
                // Return if there are no components
                if (component_count <= 0) { return; }

                int_hostView boundary_count_h = Kokkos::create_mirror_view(boundary_count);
                Kokkos::deep_copy(boundary_count_h, boundary_count);
                const int local_boundary_count = boundary_count_h(0);

                const floating_deviceView rectangle_best_area = rectangle_best_area_d;
                const floating_deviceView rectangle_best_length = rectangle_best_length_d;
                const floating_deviceView rectangle_best_width = rectangle_best_width_d;
                const int_deviceView rectangle_best_sample = rectangle_best_sample_d;
                const int_deviceView rectangle_best_lock = rectangle_best_lock_d;
                const int sample_count = 20;
                const FloatType half_pi = static_cast<FloatType>(0.5 * PI);

                Kokkos::Profiling::pushRegion("SearchRectangles::InitBest");
                Kokkos::parallel_for(
                    "meltpool_dimensions_rectangle_best_init",
                    Kokkos::RangePolicy<device_exe>(0, component_count),
                    KOKKOS_LAMBDA(const int component_idx) {
                        const FloatType large = static_cast<FloatType>(1.0e30);
                        for (int stage = 0; stage < 2; stage++) {
                            const int state_idx = stage * max_components + component_idx;
                            rectangle_best_area(state_idx) = large;
                            rectangle_best_sample(state_idx) = 0;
                            rectangle_best_length(state_idx) = static_cast<FloatType>(0.0);
                            rectangle_best_width(state_idx) = static_cast<FloatType>(0.0);
                            rectangle_best_lock(state_idx) = 0;
                        }
                        component_length(component_idx) = static_cast<FloatType>(0.0);
                        component_width(component_idx) = static_cast<FloatType>(0.0);
                    });
                Kokkos::Profiling::popRegion();

                using team_policy = Kokkos::TeamPolicy<device_exe>;
                using team_member = typename team_policy::member_type;

                auto search_stage = [&](const int stage_idx, const char* profile_label, const char* kernel_label) {
                    Kokkos::Profiling::pushRegion(profile_label);
                    const int sample_total = component_count * sample_count;
                    Kokkos::parallel_for(
                        kernel_label,
                        team_policy(sample_total, Kokkos::AUTO()),
                        KOKKOS_LAMBDA(const team_member& team) {
                            const int sample_id = team.league_rank();
                            const int component_idx = sample_id / sample_count;
                            const int sample = sample_id % sample_count;
                            const int component_root = component_roots(component_idx);

                            FloatType stage_begin = static_cast<FloatType>(0.0);
                            FloatType stage_end = half_pi;
                            if (stage_idx == 1) {
                                const int previous_idx = component_idx;
                                const int best_index = rectangle_best_sample(previous_idx);
                                const FloatType old_step = half_pi / static_cast<FloatType>(sample_count - 1);
                                const int left_index = (best_index > 0) ? (best_index - 1) : best_index;
                                const int right_index = (best_index + 1 < sample_count) ? (best_index + 1) : best_index;
                                stage_begin = static_cast<FloatType>(left_index) * old_step;
                                stage_end = static_cast<FloatType>(right_index) * old_step;
                                if (stage_end <= stage_begin) {
                                    stage_end = stage_begin + old_step;
                                }
                            }

                            const FloatType sample_step = (stage_end - stage_begin) / static_cast<FloatType>(sample_count - 1);
                            const FloatType angle = stage_begin + static_cast<FloatType>(sample) * sample_step;
                            const FloatType c = Kokkos::cos(angle);
                            const FloatType s = Kokkos::sin(angle);

                            ProjectionBounds bounds;
                            ProjectionBoundsReducer bounds_reducer(bounds);
                            Kokkos::parallel_reduce(
                                Kokkos::TeamThreadRange(team, local_boundary_count),
                                [&](const int b, ProjectionBounds& value) {
                                    if (component_label(boundary_owner(b)) != component_root) {
                                        return;
                                    }

                                    const FloatType x = boundary_x(b);
                                    const FloatType y = boundary_y(b);
                                    const FloatType u = c * x + s * y;
                                    const FloatType v = -s * x + c * y;

                                    if (!value.found) {
                                        value.min_u = u;
                                        value.max_u = u;
                                        value.min_v = v;
                                        value.max_v = v;
                                        value.found = 1;
                                        return;
                                    }

                                    value.min_u = (value.min_u < u) ? value.min_u : u;
                                    value.max_u = (value.max_u > u) ? value.max_u : u;
                                    value.min_v = (value.min_v < v) ? value.min_v : v;
                                    value.max_v = (value.max_v > v) ? value.max_v : v;
                                },
                                bounds_reducer);

                            Kokkos::single(Kokkos::PerTeam(team), [&]() {
                                if (!bounds.found) {
                                    return;
                                }

                                const FloatType extent_u = bounds.max_u - bounds.min_u;
                                const FloatType extent_v = bounds.max_v - bounds.min_v;
                                const FloatType length_candidate = (extent_u > extent_v) ? extent_u : extent_v;
                                const FloatType width_candidate = (extent_u > extent_v) ? extent_v : extent_u;
                                const FloatType area_candidate = length_candidate * width_candidate;
                                const int state_idx = stage_idx * max_components + component_idx;

                                // Only attempt to update if we have a better area, but do the atomic dance to make sure only one thread can update at a time
                                if (area_candidate < rectangle_best_area(state_idx)) {
                                
                                    // Keep trying to lock
                                    while (Kokkos::atomic_compare_exchange(&rectangle_best_lock(state_idx), 0, 1) != 0) {}

                                    // Check if we still have a better area after acquiring the lock (another thread may have updated it while we were waiting)
                                    if (area_candidate < Kokkos::atomic_load(&rectangle_best_area(state_idx))) {
                                        // Atomic store the new best area
                                        Kokkos::atomic_store(&rectangle_best_area(state_idx), area_candidate);
                                        rectangle_best_sample(state_idx) = sample;
                                        rectangle_best_length(state_idx) = length_candidate;
                                        rectangle_best_width(state_idx) = width_candidate;
                                        // If this is the last stage, also store the length and width in the component outputs
                                        if (stage_idx == 1) {
                                            component_length(component_idx) = length_candidate;
                                            component_width(component_idx) = width_candidate;
                                        }
                                    }

                                    // Unlock
                                    Kokkos::atomic_exchange(&rectangle_best_lock(state_idx), 0);
                                }
                            });
                        });
                    Kokkos::Profiling::popRegion();
                };

                // 2-stage search
                search_stage(0, "SearchRectangles::Stage0Samples", "meltpool_dimensions_rectangle_stage0_samples");
                search_stage(1, "SearchRectangles::Stage1Samples", "meltpool_dimensions_rectangle_stage1_samples");
                Kokkos::fence();
            }

            void SearchDepth() {
                const int component_count = component_count_h;
                if (component_count <= 0) { return; }

                const floating_deviceView column_depth = column_depth_d;
                const floating_deviceView component_depth = component_depth_d;
                const int_deviceView component_roots = component_roots_d;
                const int_deviceView component_offsets = component_offsets_d;
                const int_deviceView component_points = component_points_d;

                using team_policy = Kokkos::TeamPolicy<device_exe>;
                using team_member = typename team_policy::member_type;

                Kokkos::parallel_for(
                    "meltpool_dimensions_component_depths",
                    team_policy(component_count, Kokkos::AUTO()),
                    KOKKOS_LAMBDA(const team_member& team) {
                        const int component_idx = team.league_rank();
                        const int root = component_roots(component_idx);
                        const int point_begin = component_offsets(component_idx);
                        const int point_end = component_offsets(component_idx + 1);
                        const int point_count = point_end - point_begin;

                        FloatType max_depth = static_cast<FloatType>(0.0);
                        Kokkos::parallel_reduce(
                            Kokkos::TeamThreadRange(team, point_count),
                            [&](const int local_q, FloatType& local_max) {
                                const int v2d = component_points(point_begin + local_q);
                                local_max = (local_max > column_depth(v2d)) ? local_max : column_depth(v2d);
                            },
                            Kokkos::Max<FloatType>(max_depth));

                        Kokkos::single(Kokkos::PerTeam(team), [&]() {
                            component_depth(component_idx) = max_depth;
                        });
                    });
                Kokkos::fence();
            }

            void Measure(
                const int active_count,
                const int_deviceView& active_columns,
                const int_2D_deviceView& depth_view,
                const floating_deviceView& T_view,
                const bool_deviceView& isLiq_view,
                const Simdat<FloatType>& sim
            ){
                // Skip if not activated
                if (calc.skip){ return; }

                // Skip if there are no columns
                if (active_count <= 0) { return; }
                
                // Reset necessary scratch views
                ResetScratch();

                // Do the measurements (depth only if desired)
                Kokkos::Profiling::pushRegion("Dimensions::BuildInterpolatedGeometry");
                if (calc.MP_Len_or_Wid){
                    BuildInterpolatedGeometry<true>(active_count, active_columns, depth_view, T_view, sim.material.T_liq);
                }
                else if (calc.Col_Depth){
                    BuildInterpolatedGeometry<false>(active_count, active_columns, depth_view, T_view, sim.material.T_liq);
                }
                else{
                    throw std::runtime_error("Build Interpolated Geometry has Unknown Template.");
                }
                Kokkos::Profiling::popRegion();
                
                // If we want meltpool geometry, do a lot more stuff
                if (calc.MP_Len_or_Wid){

                    // Have each column have a root
                    Kokkos::Profiling::pushRegion("Dimensions::Label Seperate Meltpool");
                    const int component_count = FindConnectedComponents(active_count, active_columns, depth_view);//FindConnectedComponents(active_count, tracking);
                    Kokkos::Profiling::popRegion();

                    // Get the distinct roots + a 1D array w/ offsets for each root
                    Kokkos::Profiling::pushRegion("Dimensions::Compress Meltpools");
                    CompressComponents(active_count, active_columns, component_count);
                    Kokkos::Profiling::popRegion();

                    // Get the L and W for each root 
                    if (calc.MP_Len_or_Wid){
                        Kokkos::Profiling::pushRegion("Dimensions::Search Rectangles");
                        SearchRotatedRectangles(component_count);
                        Kokkos::Profiling::popRegion();
                    }
                    // Get the D for each root
                    if (calc.MP_Depth){
                        Kokkos::Profiling::pushRegion("Dimensions::Search Depth");
                        SearchDepth();
                        Kokkos::Profiling::popRegion();
                    }              
                } 
            }
            // Allows "Measure" to be called without needing tracking
            void Measure(
                const Tracking<FloatType>& tracking,
                const Thermal_State<FloatType>& thermal,
                const Simdat<FloatType>& sim,
                const Step_State<FloatType>& step)
            {
                Measure(
                    tracking.ActiveColumnCount(),
                    tracking.ActiveColumns(),
                    tracking.CurrentDepth(),
                    thermal.T_cur,
                    thermal.isLiq_cur,
                    sim
                );
            }
            // Structs and functions for recording time series of meltpool dimensions
            struct Meltpool_Time_Row {
                FloatType t = static_cast<FloatType>(0.0);
                int iter = 0;
                int count = 0;
                FloatType length = static_cast<FloatType>(0.0);
                FloatType width = static_cast<FloatType>(0.0);
                FloatType depth = static_cast<FloatType>(0.0);
            };
            vector<Meltpool_Time_Row> mp_stats_rows;
            void RecordTimeSeries(const Step_State<FloatType>& step, const bool onlyBiggest) {
                if (!calc.any_MP) { return; }

                Meltpool_Time_Row row;
                row.t = step.t_cur;
                row.iter = step.iter_cur;
                row.count = component_count_h;

                // If no meltpool, just record empty meltpool
                if (row.count == 0) {
                    mp_stats_rows.push_back(row);
                    return;
                }

                // Copy component dimensions back to host
                Kokkos::deep_copy(component_length_h, component_length_d);
                Kokkos::deep_copy(component_width_h, component_width_d);
                if (calc.MP_Depth) 
                {
                    Kokkos::deep_copy(component_depth_h, component_depth_d);
                }
                

                if (onlyBiggest)
                {
                    int best = 0;
                    FloatType best_area = static_cast<FloatType>(-1.0);
                    for (int c = 0; c < row.count; c++) {
                        const FloatType area = component_length_h(c) * component_width_h(c);
                        if (area > best_area) 
                        {
                            best = c;
                            best_area = area;
                        }
                    }
                    row.length = component_length_h(best);
                    row.width = component_width_h(best);
                    row.depth = calc.MP_Depth ? component_depth_h(best) : static_cast<FloatType>(0.0);
                    mp_stats_rows.push_back(row);
                }
                else
                {
                    for (int c = 0; c < row.count; c++) {
                        row.length = component_length_h(c);
                        row.width = component_width_h(c);
                        row.depth = calc.MP_Depth ? component_depth_h(c) : static_cast<FloatType>(0.0);
                        mp_stats_rows.push_back(row);
                    }
                }          
            }
    };
}
