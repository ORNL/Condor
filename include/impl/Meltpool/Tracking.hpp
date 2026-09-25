#pragma once

#include "Definitions.hpp"
#include "impl/Calc/Structs.hpp"
#include "impl/Meltpool/Beam.hpp"
#include "impl/RunModes/Interface/Structs.hpp"
#include "impl/Structs/Simdat.hpp"

namespace Condor::impl::Meltpool {

    template<typename FloatType>
    class Tracking {
        private:
            using floating_deviceView = Kokkos::View<FloatType*, layout, device_memory>;
            using floating_hostView = Kokkos::View<FloatType*, layout, host_memory>;
            using team_policy = Kokkos::TeamPolicy<device_exe>;
            using team_member = typename team_policy::member_type;

        public:
            // Initialization
            Tracking(const Simdat<FloatType>& sim);     

            // What calculations should be done
            struct Tracking_Calc {
                bool Reconstruction = false;
                bool Solidification = false;
                bool RDF = false;
            } calc;

            // Summary of tracking state
            struct Tracking_Summary {
                // Calculated every step
                int active_columns = 0;
                int solidified_points = 0;
                int solidification_events = 0;
                // Conditionally generated
                int liq_points = -1;
                // Reset (for conditionally generated)
                void Reset(){
                    liq_points = -1;
                }
            } summary;

            // Struct for solidification events
            struct SolCounts{
                enum{
                    needReconstruction = 0,
                    countSolidified = 1,
                    SIZE = 2,
                };
            };
            struct Reconstruct{
                enum{
                    Prev=1,
                    Cur=2,
                    Both=3,
                };
            };
            struct Tracking_SolidificationEvents {
                // Events - Capacity
                int capacity = 0;
                int host_capacity = 0;
                int host_size = 0;
                // Events - Counts
                int_x2_deviceView sol_counts_d;
                int_x2_hostView sol_counts_h;
                int_deviceView needReconstruction_T;
                int_deviceView needReconstruction_mask;
                // Events - Data
                int_deviceView event_p_d;
                floating_deviceView event_tm_d;
                floating_deviceView event_tl_d;
                floating_deviceView event_Tliq_d;
                floating_deviceView event_Tsol_d;
                floating_deviceView event_num_melt_d;
                int_hostView event_p_h;
                floating_hostView event_tm_h;
                floating_hostView event_tl_h;
                floating_hostView event_Tliq_h;
                floating_hostView event_Tsol_h;
                floating_hostView event_num_melt_h;
                // Volume Data
                floating_deviceView volume_tm;
                floating_deviceView volume_num_melt;
                int_deviceView sparse_pointer;  
                // Initialization
                template<bool Reconstruct_Calc, bool Solidification_Calc, bool RDF_Calc>
                void Init(const int capacity); 
                // Increase Capacity
                template<bool Reconstruct_Calc, bool Solidification_Calc, bool RDF_Calc>
                void Increase_Capacity(const int newCapacity);
                // Communicate to host and wipe device
                void PassToHostAndWipe(bool isRDF);
                
                // Lambda function for preparing for reconstruction
                template<bool Reconstruct_Calc, bool Solidification_Calc, bool RDF_Calc>
                KOKKOS_INLINE_FUNCTION
                void PrepareReconstruction(const Step_State<FloatType>& step, const Thermal_State<FloatType>& thermal, const int p) const {
                    if constexpr (!Solidification_Calc) {
                        return;
                    }
                    const bool hasPrevT = (thermal.T_calc_iter_prev(p) == step.iter_prev);
                    const bool hasCurT = (thermal.T_calc_iter_cur(p) == step.iter_cur);
                    if (hasPrevT && hasCurT) {
                        this->template RecordSolidificationEvent<Reconstruct_Calc, Solidification_Calc, RDF_Calc>(
                            p,
                            step.t_cur,
                            thermal.T_prev(p),
                            thermal.T_cur(p)
                        );
                    }
                    else {
                        int mask = 0;
                        if (!hasPrevT) {mask |= Reconstruct::Prev;}
                        if (!hasCurT) {mask |= Reconstruct::Cur;}
                        const int q = Kokkos::atomic_fetch_add(&this->sol_counts_d(SolCounts::needReconstruction),1);
                        this->needReconstruction_T(q) = p;
                        this->needReconstruction_mask(q) = mask;
                    }
                }

                // Lambda for storing a melting event into the Sparse Buffer
                template<bool Reconstruct_Calc, bool Solidification_Calc, bool RDF_Calc>
                KOKKOS_INLINE_FUNCTION
                void RecordMeltingEvent(const FloatType t_cur, const int p) const {
                    // RDF only publishes completed solidification events
                    if constexpr (RDF_Calc) {
                        // Store melt time and increment numMelt when the material actually melts
                        this->volume_tm(p) = t_cur;
                        Kokkos::atomic_fetch_add(
                            &this->volume_num_melt(p),
                            static_cast<FloatType>(1.0)
                        );
                    }
                    // For solidification calc, add numMelt and clear previous events
                    else if constexpr (Solidification_Calc) {
                        // Check sparse pointer index
                        int idx = this->sparse_pointer(p);

                        // If no pending event exists, reserve a new sparse-buffer entry
                        if (idx == -1) {
                            idx = Kokkos::atomic_fetch_add(&sol_counts_d(SolCounts::countSolidified), 1);
                            this->sparse_pointer(p) = idx;
                        }

                        // Increment numMelt when the material actually melts
                        const FloatType num_melt = Kokkos::atomic_fetch_add(
                            &this->volume_num_melt(p),
                            static_cast<FloatType>(1.0)
                        ) + static_cast<FloatType>(1.0);

                        // Add event information and set time to be negative (invalid)
                        this->event_p_d(idx) = p;
                        this->event_tl_d(idx) = static_cast<FloatType>(-1.0);
                        this->event_Tliq_d(idx) = static_cast<FloatType>(0.0);
                        this->event_Tsol_d(idx) = static_cast<FloatType>(0.0);
                        this->event_num_melt_d(idx) = num_melt;
                    }
                }

                // Lambda for storing a solidification event into the Sparse Buffer
                template<bool Reconstruct_Calc, bool Solidification_Calc, bool RDF_Calc>
                KOKKOS_INLINE_FUNCTION
                void RecordSolidificationEvent(
                    const int p,
                    const FloatType t_val,
                    const FloatType Tliq_val,
                    const FloatType Tsol_val) const
                {
                    if constexpr (Solidification_Calc) {
                        // Check
                        int idx;
                        // If rdf calc, always need new events
                        if constexpr (RDF_Calc) {
                            idx = Kokkos::atomic_fetch_add(&sol_counts_d(SolCounts::countSolidified), 1);
                        }
                        // If regular solidification calc, update previous position if it hasn't been communicated yet
                        else {
                            idx = this->sparse_pointer(p);
                            if (idx==-1){ idx = Kokkos::atomic_fetch_add(&sol_counts_d(SolCounts::countSolidified), 1);}
                            this->sparse_pointer(p) = idx;
                        }
                        // numMelt was incremented when the point actually melted
                        const FloatType num_melt = this->volume_num_melt(p);
                        // Add event information
                        this->event_p_d(idx) = p;
                        if constexpr (RDF_Calc) {
                            this->event_tm_d(idx) = this->volume_tm(p);
                        }
                        // A nonnegative time makes this a valid solidification event
                        this->event_tl_d(idx) = t_val;
                        this->event_Tliq_d(idx) = Tliq_val;
                        this->event_Tsol_d(idx) = Tsol_val;
                        this->event_num_melt_d(idx) = num_melt;
                    }
                }

            } events;
            // Initialization Routine
            void Initialize_Solidification();  
            
            // Advance the state
            Tracking_Summary Advance(
                Modes::Interface::Interface_State<FloatType>& state,
                const BeamTracer<FloatType>& beam_trace);
            
            // Finalize by ensuring a flush happens
            void Finalize();

            Tracking_Summary LastSummary() const { return summary; }            
            int ActiveColumnCount() const { return summary.active_columns; }
            int SolidifiedPointCount() const { return summary.solidified_points; }
            const int_deviceView& MaxDepthByColumn() const { return max_depth_by_column_d; }
            const int_2D_deviceView& CurrentDepth() const { return depth_cur; }
            const int_2D_deviceView& PreviousDepth() const { return depth_prev; }
            const int_deviceView& ActiveColumns() const { return depth_queue; }
            const Tracking_SolidificationEvents& Events() const { return events; }

        private:
            const Simdat<FloatType>* sim = nullptr;
            Calc::Device_Constants<FloatType> const_d;

            // For steering vector generation and tracking
            struct TrackingCounts {         
                enum{
                    surfaceSeed = 0,
                    surfaceNext = 1,
                    liqColumns = 2,
                    SIZE = 3,
                };
            };
            int_x3_deviceView track_counts_d;
            int_x3_hostView track_counts_h; 
            int column_count;

            int_2D_deviceView depth_prev;
            int_2D_deviceView depth_cur;
            int_2D_deviceView column_seen;
            int_deviceView surface_queue_a;
            int_deviceView surface_queue_b;
            int_deviceView surface_queue_c;
            int_deviceView surface_queue_cur;
            int_deviceView surface_queue_next;
            int_deviceView depth_queue;
            int_deviceView max_depth_by_column_d;
            floating_deviceView last_liq_time_d;
            floating_deviceView last_liq_temperature_d;

            int_deviceView event_update_queue_d;
            int previous_surface_count = 0;

            void SwapSurfaceQueues();
            void RotateGlobalQueues();
            void SwapDepthState();
            void FinishStep(const Step_State<FloatType>& step, bool force_flush = false);

        public:
            // CUDA extended lambdas require the enclosing member function to be public.
            void SeedPreviousSurface();
            void SeedBeamSurface(const BeamTracer<FloatType>& beam_trace);

            template<bool Reconstruct_Calc, bool Solidification_Calc, bool RDF_Calc>
            Tracking_Summary Run(
                Modes::Interface::Interface_State<FloatType>& state,
                const BeamTracer<FloatType>& beam_trace
            );
            
            template<bool Reconstruct_Calc,bool Solidification_Calc, bool RDF_Calc>
            void TrackSurface(
                const Step_State<FloatType>& step,
                Thermal_State<FloatType>& thermal,
                const Grid::Device_Points<FloatType>& grid,
                const Calc::Device_Constants<FloatType>& consts,
                const Calc::Device_Nodes<FloatType>& nodes_cur_d
            );

            template<bool Reconstruct_Calc, bool Solidification_Calc, bool RDF_Calc>
            void TrackDepth(
                const Step_State<FloatType>& step,
                Thermal_State<FloatType>& thermal,
                const Grid::Device_Points<FloatType>& grid,
                const Calc::Device_Constants<FloatType>& consts,
                const Calc::Device_Nodes<FloatType>& nodes_cur_d
            );

            template<bool Reconstruct_Calc, bool Solidification_Calc, bool RDF_Calc>
            void ReconstructTemperatures(
                const Step_State<FloatType>& step,
                Thermal_State<FloatType>& thermal,
                const Grid::Device_Points<FloatType>& grid,
                const Calc::Device_Constants<FloatType>& const_d,
                const Calc::Device_Nodes<FloatType>& nodes_prev_d,
                const Calc::Device_Nodes<FloatType>& nodes_cur_d
            );

            // Conditional summary generation
            int CountCurrentLiquidPoints()
            {
                // Only calculate if not already calculated
                if (summary.liq_points < 0){
                    const int active_columns = track_counts_h(TrackingCounts::liqColumns);
                    const int_deviceView& depth_queue_local = depth_queue; 
                    const int_2D_deviceView& depth_cur_local = depth_cur;
                    const int yNum = this->sim->domain.ynum;
                    int current_liquid_points = 0;
                    Kokkos::parallel_reduce(
                        "tracking_count_liquid_points",
                        Kokkos::RangePolicy<device_exe>(0, active_columns),
                        KOKKOS_LAMBDA(const int q, int& local_count) {
                            const int v2d = depth_queue_local(q);
                            const int i = v2d / yNum;
                            const int j = v2d % yNum;
                            local_count += depth_cur_local(i, j);
                        },
                    current_liquid_points);
                    summary.liq_points = current_liquid_points;
                }
                
                // Return value
                return summary.liq_points;
            }
    };

}
