#include "impl/Meltpool/Tracking.hpp"
#include "impl/Calc/Temperature.hpp"
#include "impl/Utility/Grid.hpp"

#include <algorithm>
#include <stdexcept>

namespace Condor::impl::Meltpool {

    template<typename FloatType>
    Tracking<FloatType>::Tracking(const Simdat<FloatType>& sim_in)
        : 
        // Store pointer to Simdat
        sim(&sim_in),
        const_d(sim_in),
        column_count(sim_in.domain.xnum * sim_in.domain.ynum),
        depth_prev(Kokkos::ViewAllocateWithoutInitializing("tracking_depth_prev"), sim_in.domain.xnum, sim_in.domain.ynum),
        depth_cur(Kokkos::ViewAllocateWithoutInitializing("tracking_depth_cur"), sim_in.domain.xnum, sim_in.domain.ynum),
        column_seen(Kokkos::ViewAllocateWithoutInitializing("tracking_column_seen"), sim_in.domain.xnum, sim_in.domain.ynum),
        surface_queue_a(Kokkos::ViewAllocateWithoutInitializing("tracking_surface_queue_a"), column_count),
        surface_queue_b(Kokkos::ViewAllocateWithoutInitializing("tracking_surface_queue_b"), column_count),
        surface_queue_c(Kokkos::ViewAllocateWithoutInitializing("tracking_surface_queue_c"), column_count),
        surface_queue_cur(surface_queue_a),
        surface_queue_next(surface_queue_b),
        depth_queue(surface_queue_c),
        max_depth_by_column_d(Kokkos::ViewAllocateWithoutInitializing("tracking_max_depth_by_column"), column_count),
        track_counts_d(Kokkos::ViewAllocateWithoutInitializing("tracking_counts_d")),
        track_counts_h(Kokkos::ViewAllocateWithoutInitializing("tracking_counts_h"))
    {
        Kokkos::deep_copy(depth_prev, 0);
        Kokkos::deep_copy(depth_cur, 0);
        Kokkos::deep_copy(column_seen, 0);
        Kokkos::deep_copy(track_counts_d, 0);
        Kokkos::deep_copy(track_counts_h, 0);
        Kokkos::deep_copy(max_depth_by_column_d, 0);
    }

    template<typename FloatType>
    template<bool Reconstruct_Calc, bool Solidification_Calc, bool RDF_Calc>
    void Tracking<FloatType>::Tracking_SolidificationEvents::Init(const int domain_size){

        // If we dont do anything about the events, return
        if constexpr (!Reconstruct_Calc){return;}
        
        // Set initial capacity and counters
        this->capacity = domain_size;
        sol_counts_d = int_x2_deviceView(Kokkos::ViewAllocateWithoutInitializing("tracking_sol_count_d"));
        sol_counts_h = int_x2_hostView(Kokkos::ViewAllocateWithoutInitializing("tracking_sol_count_h"));
        Kokkos::deep_copy(sol_counts_d, 0);
        Kokkos::deep_copy(sol_counts_h, 0); 
        // Temperature Reconstruction
        needReconstruction_T = int_deviceView(Kokkos::ViewAllocateWithoutInitializing("needReconstruction_T"), domain_size);
        needReconstruction_mask = int_deviceView(Kokkos::ViewAllocateWithoutInitializing("needReconstruction_mask"), domain_size);
        
        // Return if we only care about reconstruction (and not events) then return
        if constexpr (!Solidification_Calc){return;} 
        
        // Events - Data
        event_p_d = int_deviceView(Kokkos::ViewAllocateWithoutInitializing("tracking_event_p_d"), domain_size);
        if constexpr (RDF_Calc){
            event_tm_d = floating_deviceView(Kokkos::ViewAllocateWithoutInitializing("tracking_event_tm_d"), domain_size);
        }
        event_tl_d = floating_deviceView(Kokkos::ViewAllocateWithoutInitializing("tracking_event_tl_d"), domain_size);
        event_Tliq_d = floating_deviceView(Kokkos::ViewAllocateWithoutInitializing("tracking_event_Tliq_d"), domain_size);
        event_Tsol_d = floating_deviceView(Kokkos::ViewAllocateWithoutInitializing("tracking_event_Tsol_d"), domain_size);
        event_num_melt_d = floating_deviceView(Kokkos::ViewAllocateWithoutInitializing("tracking_event_num_melt_d"), domain_size);
        // Volume Data
        if constexpr (RDF_Calc){
            volume_tm = floating_deviceView(Kokkos::ViewAllocateWithoutInitializing("tracking_volume_tm"), domain_size);
        }
        volume_num_melt = floating_deviceView(Kokkos::ViewAllocateWithoutInitializing("tracking_volume_num_melt"), domain_size);
        Kokkos::deep_copy(volume_num_melt, static_cast<FloatType>(0.0));
        if constexpr (!RDF_Calc){
            sparse_pointer = int_deviceView(Kokkos::ViewAllocateWithoutInitializing("tracking_sparse_pointer"), domain_size);
            Kokkos::deep_copy(sparse_pointer, -1);
        }   
    }

    template<typename FloatType>
    template<bool Reconstruct_Calc, bool Solidification_Calc, bool RDF_Calc>
    void Tracking<FloatType>::Tracking_SolidificationEvents::Increase_Capacity(const int newCapacity){
        // Resize is only needed for RDF since it can have multiple remelting events
        if constexpr (!RDF_Calc) { return; }
        if (newCapacity <= this->capacity) { return; }
        this->capacity = newCapacity;
        Kokkos::resize(Kokkos::WithoutInitializing, this->event_p_d, newCapacity);
        Kokkos::resize(Kokkos::WithoutInitializing, this->event_tm_d, newCapacity);
        Kokkos::resize(Kokkos::WithoutInitializing, this->event_tl_d, newCapacity);
        Kokkos::resize(Kokkos::WithoutInitializing, this->event_Tliq_d, newCapacity);
        Kokkos::resize(Kokkos::WithoutInitializing, this->event_Tsol_d, newCapacity);
        Kokkos::resize(Kokkos::WithoutInitializing, this->event_num_melt_d, newCapacity);
    }

    template<typename FloatType>
    void Tracking<FloatType>::Tracking_SolidificationEvents::PassToHostAndWipe(const bool isRDF){
        
        // Use the event count already copied by FinishStep.
        this->host_size = this->sol_counts_h(SolCounts::countSolidified);

        // Do nothing if there are no events
        if (host_size == 0) return;

        // Initially just allocate current size or if we are too small
        if (static_cast<int>(this->host_capacity) < host_size) {
            this->host_capacity = host_size;
            this->event_p_h = int_hostView(Kokkos::ViewAllocateWithoutInitializing("tracking_event_p_h"), host_size);
            if (isRDF){
                this->event_tm_h = floating_hostView(Kokkos::ViewAllocateWithoutInitializing("tracking_event_tm_h"), host_size);
            }
            this->event_tl_h = floating_hostView(Kokkos::ViewAllocateWithoutInitializing("tracking_event_tl_h"), host_size);
            this->event_Tliq_h = floating_hostView(Kokkos::ViewAllocateWithoutInitializing("tracking_event_Tliq_h"), host_size);
            this->event_Tsol_h = floating_hostView(Kokkos::ViewAllocateWithoutInitializing("tracking_event_Tsol_h"), host_size);
            this->event_num_melt_h = floating_hostView(Kokkos::ViewAllocateWithoutInitializing("tracking_event_num_melt_h"), host_size);
        }

        // Transfer data to CPU
        Kokkos::pair<int, int> event_range(0, host_size);
        Kokkos::deep_copy(Kokkos::subview(this->event_p_h, event_range), Kokkos::subview(this->event_p_d, event_range));
        if (isRDF){
            Kokkos::deep_copy(Kokkos::subview(this->event_tm_h, event_range), Kokkos::subview(this->event_tm_d, event_range));
        }
        Kokkos::deep_copy(Kokkos::subview(this->event_tl_h, event_range), Kokkos::subview(this->event_tl_d, event_range));
        Kokkos::deep_copy(Kokkos::subview(this->event_Tliq_h, event_range), Kokkos::subview(this->event_Tliq_d, event_range));
        Kokkos::deep_copy(Kokkos::subview(this->event_Tsol_h, event_range), Kokkos::subview(this->event_Tsol_d, event_range));
        Kokkos::deep_copy(Kokkos::subview(this->event_num_melt_h, event_range), Kokkos::subview(this->event_num_melt_d, event_range));

        // Reset the CPU & GPU Counter
        this->sol_counts_h(SolCounts::countSolidified) = 0;
        Kokkos::deep_copy(this->sol_counts_d, this->sol_counts_h);

        // If RDF is true, sparse_pointer wasn't used, so we skip this.
        if (!isRDF) {
            
            // Capture views by value for the lambda
            int_deviceView event_p_local = this->event_p_d;
            int_deviceView sparse_pointer_local = this->sparse_pointer;

            Kokkos::parallel_for("ResetSparsePointers", host_size, KOKKOS_LAMBDA(const int i) {
                sparse_pointer_local(event_p_local(i)) = -1; 
            });
            
            // Wait for reset to finish before resuming main simulation loop
            Kokkos::fence(); 
        }
    }

    template<typename FloatType>
    void Tracking<FloatType>::Initialize_Solidification(){
        // If we are only determining meltpool dimensions
        if (!this->calc.Reconstruction && !this->calc.Solidification && !this->calc.RDF){
            this->events.template Init<false, false, false>(this->sim->domain.pnum);
        }
        // If we must reconstruct temperatures
        else if (this->calc.Reconstruction && !this->calc.Solidification && !this->calc.RDF){
            this->events.template Init<true, false, false>(this->sim->domain.pnum);
        }
        // If we are doing solidification
        else if (this->calc.Reconstruction && this->calc.Solidification && !this->calc.RDF){
            this->events.template Init<true, true, false>(this->sim->domain.pnum);
        }
        // If we are doing RDF
        else if (this->calc.Reconstruction && this->calc.Solidification && this->calc.RDF){
            this->events.template Init<true, true, true>(this->sim->domain.pnum);
        }
        else{
            throw std::runtime_error("Tracking::Unknown Calculation State");
        }
    }

    template<typename FloatType>
    void Tracking<FloatType>::SeedPreviousSurface() {
        // Reset column seen
        Kokkos::deep_copy(this->column_seen,0);
        // Rotate counts and send back to device
        this->track_counts_h(TrackingCounts::surfaceSeed) = this->track_counts_h(TrackingCounts::liqColumns);
        this->track_counts_h(TrackingCounts::surfaceNext) = 0;
        this->track_counts_h(TrackingCounts::liqColumns) = 0;
        Kokkos::deep_copy(this->track_counts_d,this->track_counts_h);
        // Skip if we don't have any liquid columns
        if (this->track_counts_h(TrackingCounts::surfaceSeed) <= 0) {return;}
        const Calc::Device_Constants<FloatType>& constants = this->const_d;
        int_deviceView& queue = this->surface_queue_cur;
        int_2D_deviceView& column_seen_local = this->column_seen;
        // Set everything in the surface queue to be 1 (so it doesn't get added again)
        Kokkos::parallel_for(
            "tracking_seed_previous_surface",
            Kokkos::RangePolicy<device_exe>(0, this->track_counts_h(TrackingCounts::surfaceSeed)),
            KOKKOS_LAMBDA(const int q) {
                const int v2d = queue(q);
                const int yNum = constants.ints(IntConstants::Y_NUM);
                const int i = v2d / yNum;
                const int j = v2d % yNum;
                column_seen_local(i, j) = 1;
            }
        );
    }

    template<typename FloatType>
    void Tracking<FloatType>::SeedBeamSurface(const BeamTracer<FloatType>& beam_trace) {
        if (!beam_trace.HasActiveSeeds()) {
            return;
        }
        
        int_2D_deviceView& column_seen_local = this->column_seen;
        int_deviceView& queue = this->surface_queue_cur;
        int_x3_deviceView& counts = this->track_counts_d;
        const Calc::Device_Constants<FloatType>& constants = this->const_d;
        const int_deviceView seed_indices = beam_trace.ActiveSeedIndices();
        const int seed_count = beam_trace.ActiveSeedCount();

        Kokkos::parallel_for(
            "tracking_seed_beam_surface",
            Kokkos::RangePolicy<device_exe>(0, seed_count),
            KOKKOS_LAMBDA(const int q) {
                const int v2d = seed_indices(q);
                const int yNum = constants.ints(IntConstants::Y_NUM);
                const int i = v2d / yNum;
                const int j = v2d % yNum;
                if (Kokkos::atomic_compare_exchange(&column_seen_local(i, j), 0, 1) == 0) {
                    queue(Kokkos::atomic_fetch_add(&counts(TrackingCounts::surfaceSeed), 1)) = v2d;
                }
            });
    }

    template<typename FloatType>
    void Tracking<FloatType>::SwapSurfaceQueues() {
        int_deviceView tmp = this->surface_queue_cur;
        this->surface_queue_cur = this->surface_queue_next;
        this->surface_queue_next = tmp;
    }

    template<typename FloatType>
    void Tracking<FloatType>::SwapDepthState() {
        int_2D_deviceView tmp = this->depth_prev;
        this->depth_prev = this->depth_cur;
        this->depth_cur = tmp;
    }

    template<typename FloatType>
    void Tracking<FloatType>::RotateGlobalQueues() {
        int_deviceView tmp = this->depth_queue;
        this->depth_queue = this->surface_queue_next;
        this->surface_queue_next = this->surface_queue_cur;
        this->surface_queue_cur = tmp;
    }

    template<typename FloatType>
    void Tracking<FloatType>::FinishStep(const Step_State<FloatType>& step, bool force_flush) {

        // Copy counts to host
        const int previous_active_columns = this->summary.active_columns;
        Kokkos::deep_copy(this->track_counts_h, this->track_counts_d);
        this->summary.active_columns = this->track_counts_h(TrackingCounts::liqColumns);
        
        // Reset counts if we are empty
        if (previous_active_columns > 0 && this->summary.active_columns == 0) {
            this->track_counts_h(TrackingCounts::surfaceSeed) = 0;
            this->track_counts_h(TrackingCounts::surfaceNext) = 0;
            this->track_counts_h(TrackingCounts::liqColumns) = 0;
            Kokkos::deep_copy(this->track_counts_d, this->track_counts_h);
            Kokkos::deep_copy(this->depth_cur, 0);
            Kokkos::deep_copy(this->depth_prev, 0);
        }

        
        this->summary.solidified_points = 0;
        this->summary.solidification_events = 0;

        // If we are calculating solidification
        if (this->calc.Solidification && !this->calc.RDF){
            // Copy counts to host
            Kokkos::deep_copy(this->events.sol_counts_h, this->events.sol_counts_d);
            const int buffered_events = this->events.sol_counts_h(SolCounts::countSolidified);
            this->summary.solidified_points = buffered_events;
            this->summary.solidification_events = buffered_events;
            // If this is an output step, we need to sync
            if (step.isOutputStep() || force_flush){
                this->events.PassToHostAndWipe(false);
            }
        }

        // If we are doing RDF
        if (this->calc.RDF){
            // Copy counts to host
            Kokkos::deep_copy(this->events.sol_counts_h, this->events.sol_counts_d);
            const int buffered_events = this->events.sol_counts_h(SolCounts::countSolidified);
            const int possible_events = CountCurrentLiquidPoints();
            if (buffered_events + possible_events > this->events.capacity || force_flush){
                this->events.PassToHostAndWipe(true);
            }
        }
    }

    template<typename FloatType>
    void Tracking<FloatType>::Finalize() {
        Kokkos::deep_copy(this->track_counts_h, this->track_counts_d);
        this->summary.active_columns = this->track_counts_h(TrackingCounts::liqColumns);

        if (this->calc.Solidification || this->calc.RDF) {
            Kokkos::deep_copy(this->events.sol_counts_h, this->events.sol_counts_d);
            const int buffered_events = this->events.sol_counts_h(SolCounts::countSolidified);
            this->summary.solidified_points = buffered_events;
            this->summary.solidification_events = buffered_events;
            // Always sync events
            this->events.PassToHostAndWipe(this->calc.RDF);
        }
    }

    template<typename FloatType>
    typename Tracking<FloatType>::Tracking_Summary Tracking<FloatType>::Advance(
        Modes::Interface::Interface_State<FloatType>& state,
        const BeamTracer<FloatType>& beam_trace)
    {
        // If we are only determining meltpool dimensions
        if (!this->calc.Reconstruction && !this->calc.Solidification && !this->calc.RDF){
            return Run<false, false, false>(state, beam_trace);
        }
        // If we must reconstruct temperatures
        else if (this->calc.Reconstruction && !this->calc.Solidification && !this->calc.RDF){
            return Run<true, false, false>(state, beam_trace);
        }
        // If we are doing solidification
        else if (this->calc.Reconstruction && this->calc.Solidification && !this->calc.RDF){
            return Run<true, true, false>(state, beam_trace);
        }
        // If we are doing RDF
        else if (this->calc.Reconstruction && this->calc.Solidification && this->calc.RDF){
            return Run<true, true, true>(state, beam_trace);
        }
        else{
            throw std::runtime_error("Tracking::Unknown Calculation State");
        }    
    }

    template<typename FloatType>
    template<bool Reconstruct_Calc, bool Solidification_Calc, bool RDF_Calc>
    typename Tracking<FloatType>::Tracking_Summary Tracking<FloatType>::Run(
        Modes::Interface::Interface_State<FloatType>& state,
        const BeamTracer<FloatType>& beam_trace)
    {
        Calc::Device_Constants<FloatType> const_d(*this->sim, state.timestep);
        const Grid::Device_Points<FloatType>& grid = state.grid;
        const Calc::Device_Nodes<FloatType>& nodes_cur_d = state.quad.CurrentSlot().device_nodes;

        // Reset variables and rotate pointers for the queues
        Kokkos::Profiling::pushRegion("SeedSurface");  
        this->events.host_size=0;
        this->summary.Reset();
        this->SwapDepthState();
        this->RotateGlobalQueues();
        this->SeedPreviousSurface();
        this->SeedBeamSurface(beam_trace);
        Kokkos::fence();
        Kokkos::Profiling::popRegion();

        // Track molten pool surface
        Kokkos::Profiling::pushRegion("TrackSurface");
        TrackSurface<Reconstruct_Calc, Solidification_Calc, RDF_Calc>(
            state.step,
            state.thermal,
            grid,
            const_d,
            nodes_cur_d);
        Kokkos::Profiling::popRegion();
        
        // Track the molten columns
        Kokkos::Profiling::pushRegion("TrackDepth");
        TrackDepth<Reconstruct_Calc, Solidification_Calc, RDF_Calc>(
            state.step,
            state.thermal,
            grid,
            const_d,
            nodes_cur_d
        );
        Kokkos::Profiling::popRegion();
        
        // Reconstruct necessary temperatures
        Kokkos::Profiling::pushRegion("Reconstruct");
        if (state.step.iter_cur>1){
            const Calc::Device_Nodes<FloatType>& nodes_prev_d = state.quad.PreviousSlot().device_nodes;
            ReconstructTemperatures<Reconstruct_Calc, Solidification_Calc, RDF_Calc>(
                state.step,
                state.thermal,
                grid,
                const_d,
                nodes_prev_d,
                nodes_cur_d
            );
        }
        Kokkos::Profiling::popRegion();

        Kokkos::Profiling::pushRegion("Finalize");
        this->FinishStep(state.step);
        Kokkos::Profiling::popRegion();

        return this->summary;
    }

    template<typename FloatType>
    template<bool Reconstruct_Calc, bool Solidification_Calc, bool RDF_Calc>
    void Tracking<FloatType>::TrackSurface(
        const Step_State<FloatType>& step,
        Thermal_State<FloatType>& thermal,   
        const Grid::Device_Points<FloatType>& grid,
        const Calc::Device_Constants<FloatType>& const_d, 
        const Calc::Device_Nodes<FloatType>& nodes_cur_d)
    {
        Kokkos::deep_copy(this->track_counts_h, this->track_counts_d);
        int surface_count_cur = TrackingCounts::surfaceSeed;
        int surface_count_next = TrackingCounts::surfaceNext;
        int queue_count = this->track_counts_h(surface_count_cur);

        // Step Stuff
        const int iter_prev = step.iter_prev;
        const int iter_cur = step.iter_cur;
        const FloatType t_cur = step.t_cur;

        // Thermal views
        floating_deviceView& T_cur = thermal.T_cur;
        int_deviceView& T_iter_cur = thermal.T_calc_iter_cur;
        int_deviceView& T_iter_prev = thermal.T_calc_iter_prev;
        bool_deviceView& is_liq_cur = thermal.isLiq_cur;

        // Local Views
        int_deviceView& surface_queue = this->surface_queue_cur;
        int_deviceView& surface_next = this->surface_queue_next;
        int_deviceView& depth_queue_local = this->depth_queue;
        int_deviceView& max_depth_by_column_local = this->max_depth_by_column_d;
        int_x3_deviceView& counts = this->track_counts_d;
        int_2D_deviceView& depth_prev_local = this->depth_prev;
        int_2D_deviceView& depth_cur_local = this->depth_cur;
        int_2D_deviceView& column_seen_local = this->column_seen;

        // Solidification event views
        Tracking_SolidificationEvents& events_local = this->events;

        while (queue_count > 0) 
        {
            Kokkos::parallel_for(
                "tracking_surface",
                team_policy(queue_count, Kokkos::AUTO),
                KOKKOS_LAMBDA(const team_member& team) {
                    
                    // Forced capture touches (to proc "ODR-use" rules -> Allows if constexpr)
                    const Step_State<FloatType>& _force_step = step;
                    const Thermal_State<FloatType>& _force_thermal = thermal;
                    const Tracking_SolidificationEvents& _force_events = events_local;
                    const FloatType _force_t_cur = t_cur;

                    // Load the queued surface vertex and its top-of-column point id.
                    const int v2d = surface_queue(team.league_rank());
                    const int xNum = const_d.ints(IntConstants::X_NUM);
                    const int yNum = const_d.ints(IntConstants::Y_NUM);            
                    const int zNum = const_d.ints(IntConstants::Z_NUM);
                    const int i = v2d / yNum;
                    const int j = v2d % yNum;
                    const int p_top = Grid::ijk_to_p(i, j, zNum - 1, yNum, zNum);

                    // Evaluate the surface temperature for this x-y column.
                    const FloatType T = Calc::T_device(grid, nodes_cur_d, const_d, p_top, team);
                    if (team.team_rank() == 0) {
                        if (team.league_rank() == 0){
                            counts(surface_count_cur) = 0;
                        }
                        T_cur(p_top) = T;
                        T_iter_cur(p_top) = iter_cur;
                        const bool isMelted = (T >= const_d.floats(FloatConstants::T_LIQ));
                        const int old_depth = (is_liq_cur(p_top)) ? depth_prev_local(i, j) : 0;
                        if (old_depth == 0){depth_prev_local(i, j) = 0;}
                        if (isMelted) 
                        {
                            // Set current seed depth
                            depth_cur_local(i, j) = (old_depth > 0) ? old_depth : 1;
                            // Update max depth if sim domain is 2D (because depth tracking wont run)
                            if (zNum == 1) {max_depth_by_column_local(v2d) = 1;}
                            // Say that the top is liquid
                            is_liq_cur(p_top) = true;
                            // Add this point to the depth queue
                            depth_queue_local(Kokkos::atomic_fetch_add(&counts(TrackingCounts::liqColumns), 1)) = v2d;
                            // Add the melting event if this point just became liquid
                            if constexpr(Solidification_Calc){
                                if (!old_depth){
                                    events_local.template RecordMeltingEvent<Reconstruct_Calc,Solidification_Calc,RDF_Calc>(t_cur, p_top);
                                }
                            } 
                            // Add neighbors to the next queue
                            for (int di = -1; di <= 1; ++di) {
                                for (int dj = -1; dj <= 1; ++dj) {
                                    const int ni = i + di;
                                    const int nj = j + dj;
                                    // Dont go OOB
                                    if (ni < 0 || ni >= xNum || nj < 0 || nj >= yNum) {
                                        continue;
                                    }
                                    if (column_seen_local(ni, nj)==0){
                                        if (Kokkos::atomic_exchange(&column_seen_local(ni, nj), 1) == 0) {
                                            surface_next(Kokkos::atomic_fetch_add(&counts(surface_count_next), 1)) = ni * yNum + nj;
                                        }
                                    } 
                                }
                            }
                        }   
                        else
                        {
                            // Set to be solid
                            depth_cur_local(i, j) = 0;
                            // Set all points in the column to be solid too 
                            for (int d = 0; d < old_depth; ++d) {
                                const int p = Grid::ijk_to_p(i, j, (zNum - 1) - d, yNum, zNum);
                                is_liq_cur(p) = false;
                                if constexpr(Solidification_Calc){
                                    events_local.template PrepareReconstruction<Reconstruct_Calc, Solidification_Calc, RDF_Calc>(step,thermal,p);
                                }  
                            }
                        }  
                    }
            });

            Kokkos::Profiling::pushRegion("setupForNext");
            Kokkos::deep_copy(this->track_counts_h, this->track_counts_d);
            queue_count = this->track_counts_h(surface_count_next);
            if (queue_count > 0) {
                // a->b switch for reduced cleaning
                SwapSurfaceQueues();
                std::swap(surface_count_cur, surface_count_next);
            }
            Kokkos::Profiling::popRegion();
        }
        Kokkos::fence(); 
    }

    template<typename FloatType>
    template<bool Reconstruct_Calc, bool Solidification_Calc, bool RDF_Calc>
    void Tracking<FloatType>::TrackDepth(
        const Step_State<FloatType>& step,
        Thermal_State<FloatType>& thermal,
        const Grid::Device_Points<FloatType>& grid,
        const Calc::Device_Constants<FloatType>& const_d,
        const Calc::Device_Nodes<FloatType>& nodes_cur_d)
    {
        // For team scratch memory during depth tracking
        struct Scratch {
            enum {
                Depth = 0,
                Melted = 1,
                Low = 2,
                High = 3,
                Step = 4,
                PrevDepth = 5,
                ProbeClamped = 6,
                Count_Int = 7,
            };
        };
        using scratch_memory_space = typename team_member::scratch_memory_space;
        using depth_scratch_intView = Kokkos::View<int[Scratch::Count_Int], scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
      
        // If nothing is liquid or no depth, skip
        const int active_count = this->track_counts_h(TrackingCounts::liqColumns);
        if (!active_count || this->sim->domain.znum==1) {
            return;
        }

        // Step Stuff
        const int iter_cur = step.iter_cur;
        const FloatType t_cur = step.t_cur;

        // Thermal views
        floating_deviceView& T_cur = thermal.T_cur;
        int_deviceView& T_iter_cur = thermal.T_calc_iter_cur;
        bool_deviceView& is_liq_cur = thermal.isLiq_cur;
        int_deviceView& depth_queue_local = this->depth_queue;
        int_2D_deviceView& depth_prev_local = this->depth_prev;
        int_2D_deviceView& depth_cur_local = this->depth_cur;
        int_deviceView& max_depth_by_column_local = this->max_depth_by_column_d;
        
        // Solidification views (object local)
        Tracking_SolidificationEvents& events_local = this->events;

        Kokkos::parallel_for(
            "tracking_depth",
            team_policy(active_count, Kokkos::AUTO).set_scratch_size(0, Kokkos::PerTeam(Scratch::Count_Int * sizeof(int))),
            KOKKOS_LAMBDA(const team_member& team) 
            {
                // Forced capture touches (to proc "ODR-use" rules -> Allows if constexpr)
                const Step_State<FloatType>& _force_step = step;
                const Thermal_State<FloatType>& _force_thermal = thermal;
                const Tracking_SolidificationEvents& _force_events = events_local;
                const FloatType _force_t_cur = t_cur;

                // Load the active column coordinates and the shared scratch state.
                const int v2d = depth_queue_local(team.league_rank());
                const int yNum = const_d.ints(IntConstants::Y_NUM);            
                const int zNum = const_d.ints(IntConstants::Z_NUM);
                const int i = v2d / yNum;
                const int j = v2d % yNum;
                const FloatType T_liq = const_d.floats(FloatConstants::T_LIQ);

                // Set scratch memory
                depth_scratch_intView iscratch(reinterpret_cast<int*>(team.team_shmem().get_shmem(Scratch::Count_Int * sizeof(int))));
                int* depth_ptr = &iscratch.data()[Scratch::Depth];
                int* melted_ptr = &iscratch.data()[Scratch::Melted];
                int* low_ptr = &iscratch.data()[Scratch::Low];
                int* high_ptr = &iscratch.data()[Scratch::High];
                int* step_ptr = &iscratch.data()[Scratch::Step];
                int* probe_clamped_ptr = &iscratch.data()[Scratch::ProbeClamped];
                int* prev_depth_ptr = &iscratch.data()[Scratch::PrevDepth];

                // Carry forward the saved first-solid depth and clamp the first probe to the valid interior range.
                if (team.team_rank() == 0) {
                    *prev_depth_ptr = depth_prev_local(i, j);
                    const int saved_depth = depth_cur_local(i, j);
                    *depth_ptr = ((saved_depth < zNum) ? saved_depth : (zNum - 1));
                }
                team.team_barrier();

                // Evaluate the initial depth probe for this column.
                {
                    const int d = *depth_ptr;
                    const int pCheck = Grid::ijk_to_p(i, j, (zNum - 1) - d, yNum, zNum);
                    const FloatType T = Calc::T_device(grid, nodes_cur_d, const_d, pCheck, team);
                    if (team.team_rank() == 0) {
                        T_cur(pCheck) = T;
                        T_iter_cur(pCheck) = iter_cur;
                        *melted_ptr = (T >= T_liq);
                        if (*melted_ptr) {
                            *low_ptr = *depth_ptr;
                            *high_ptr = *depth_ptr;
                        }
                        else {
                            *low_ptr = 0;
                            *high_ptr = *depth_ptr;
                        }
                    }
                    team.team_barrier();
                }

                // Jump downward in sqrt-sized blocks until a solid point is found.
                if (*melted_ptr) {
                    while (*low_ptr < zNum - 1 && ((*high_ptr == *low_ptr) || (*high_ptr - *low_ptr > 1))) {
                        if (team.team_rank() == 0) {
                            // Keep jumping downward until the first solid point is found.
                            *step_ptr = static_cast<int>(Kokkos::sqrt(static_cast<FloatType>(zNum - 1 - *depth_ptr) + 1));
                            if (*high_ptr == *low_ptr) {
                                *probe_clamped_ptr = *low_ptr + *step_ptr;
                                if (*probe_clamped_ptr <= *low_ptr) {
                                    *probe_clamped_ptr = *low_ptr + 1;
                                }
                                if (*probe_clamped_ptr > zNum - 1) {
                                    *probe_clamped_ptr = zNum - 1;
                                }
                            }
                            // Once a solid point is found, return to the last liquid point and step down one layer at a time.
                            else {
                                *probe_clamped_ptr = *low_ptr + 1;
                            }
                        }
                        team.team_barrier();

                        // Every downward probe is strictly deeper than the previous liquid bound, so it is always new.
                        const int pProbe = Grid::ijk_to_p(i, j, (zNum - 1) - *probe_clamped_ptr, const_d);
                        const FloatType T = Calc::T_device(grid, nodes_cur_d, const_d, pProbe, team);
                        if (team.team_rank() == 0) {
                            T_cur(pProbe) = T;
                            T_iter_cur(pProbe) = iter_cur;
                            // Tighten the bracket around the deepest liquid point found so far.
                            if (T >= T_liq) {
                                *low_ptr = *probe_clamped_ptr;
                                *high_ptr = *low_ptr;
                            }
                            else {
                                *high_ptr = *probe_clamped_ptr;
                            }
                        }
                        team.team_barrier();
                    }
                }
                else {
                    // Solidification needs the conservative layer walk; depth-only tracking can bracket aggressively.
                    while ((*high_ptr - *low_ptr > 1)) {
                        if (team.team_rank() == 0) {
                            *probe_clamped_ptr = *high_ptr - 1;
                        }
                        team.team_barrier();

                        // Every upward probe is strictly shallower than the current solid bound, so it is always new.
                        const int pProbe = Grid::ijk_to_p(i, j, (zNum - 1) - *probe_clamped_ptr, const_d);
                        const FloatType T = Calc::T_device(grid, nodes_cur_d, const_d, pProbe, team);
                        if (team.team_rank() == 0) {
                            T_cur(pProbe) = T;
                            T_iter_cur(pProbe) = iter_cur;
                            // If liquid, low becomes probe which will make it stop
                            if (T >= T_liq) {
                                *low_ptr = *probe_clamped_ptr;
                            }
                            // Otherwise, step down by 1
                            else {
                                *high_ptr = *probe_clamped_ptr;
                            }
                        }
                        team.team_barrier();
                    }
                }

                // Convert the resolved interface to the next probe depth and only apply phase transitions.
                if (team.team_rank() == 0) {
                    // Update depth and max depth
                    const int old_depth = *prev_depth_ptr;
                    const int new_depth = *low_ptr + 1;
                    const bool is_melting = (new_depth > old_depth);
                    int changed_begin = is_melting ? old_depth : new_depth;
                    const int changed_end = is_melting ? new_depth : old_depth;
                    if (changed_begin == 0) { changed_begin = 1; }
                    *depth_ptr = new_depth;
                    depth_cur_local(i, j) = *depth_ptr;
                    if (new_depth > max_depth_by_column_local(v2d)) { max_depth_by_column_local(v2d) = new_depth;}

                    // Reconcile only the layers whose phase changed across the updated interface.
                    for (int d = changed_begin; d < changed_end; d++) {
                        const int pFill = Grid::ijk_to_p(i, j, (zNum - 1) - d, yNum, zNum);
                        
                        // If we are melting
                        if (is_melting)
                        {
                            // Set to be liquid
                            is_liq_cur(pFill) = true;
                            // Melting event bookkeeping
                            if constexpr (Solidification_Calc){
                                events_local.template RecordMeltingEvent<Reconstruct_Calc,Solidification_Calc,RDF_Calc>(t_cur,pFill);
                            }
                        }
                        // If we are solidifying
                        else
                        {
                            // Set not liquid
                            is_liq_cur(pFill) = false;
                            // If we need to calculate solidification
                            if constexpr (Solidification_Calc) {
                                events_local.template PrepareReconstruction<Reconstruct_Calc, Solidification_Calc, RDF_Calc>(step,thermal,pFill);
                            }
                        }
                    }
                }
            }
        );
        Kokkos::fence();
    }

    template<typename FloatType>
    template<bool Reconstruct_Calc, bool Solidification_Calc, bool RDF_Calc>
    void Tracking<FloatType>::ReconstructTemperatures(
        const Step_State<FloatType>& step,
        Thermal_State<FloatType>& thermal,
        const Grid::Device_Points<FloatType>& grid,
        const Calc::Device_Constants<FloatType>& const_d,
        const Calc::Device_Nodes<FloatType>& nodes_prev_d,
        const Calc::Device_Nodes<FloatType>& nodes_cur_d)
    {
        if constexpr (Solidification_Calc) {

            // Copy event counters to host
            Tracking_SolidificationEvents& events_local = this->events;
            Kokkos::deep_copy(events_local.sol_counts_h, events_local.sol_counts_d);
            const int num_need_reconstruction = events_local.sol_counts_h(SolCounts::needReconstruction);

            // Don't do anything if there isn't anything to do
            if (!num_need_reconstruction) {
                return;
            }
           
            // Local views used by deferred reconstruction.
            floating_deviceView& T_cur = thermal.T_cur;
            floating_deviceView& T_prev = thermal.T_prev;
            int_deviceView& T_iter_cur = thermal.T_calc_iter_cur;
            int_deviceView& T_iter_prev = thermal.T_calc_iter_prev;
            FloatType t_cur = step.t_cur;
            int iter_cur = step.iter_cur;
            int iter_prev = step.iter_prev;
            int_deviceView& reconstruct_p = events_local.needReconstruction_T;
            int_deviceView& reconstruct_mask = events_local.needReconstruction_mask;
            
            Kokkos::parallel_for(
                "tracking_reconstruct",
                team_policy(num_need_reconstruction, Kokkos::AUTO),
                KOKKOS_LAMBDA(const team_member& team) {
                    const int q = team.league_rank();
                    const int p = reconstruct_p(q);
                    const int mask = reconstruct_mask(q);
                    
                    // Forced capture touches (to proc "ODR-use" rules -> Allows if constexpr)
                    const Tracking_SolidificationEvents& _force_events = events_local;

                    // T values for reconstruction
                    FloatType T_prev_val = T_prev(p);
                    FloatType T_cur_val = T_cur(p);
                    FloatType _force_t_cur = t_cur;

                    // Reconstruct previous temperature if needed
                    if (mask & Reconstruct::Prev) {
                        T_prev_val = Calc::T_device(grid, nodes_prev_d, const_d, p, team);
                        if (team.team_rank() == 0) {
                            T_prev(p) = T_prev_val;
                            T_iter_prev(p) = iter_prev;
                        }
                    }
                    team.team_barrier();

                    // Reconstruct current temperature if needed
                    if (mask & Reconstruct::Cur) {
                        T_cur_val = Calc::T_device(grid, nodes_cur_d, const_d, p, team);
                        if (team.team_rank() == 0) {
                            T_cur(p) = T_cur_val;
                            T_iter_cur(p) = iter_cur;
                        }
                    }
                    team.team_barrier();
                    
                    // Record event data
                    if constexpr (Solidification_Calc){
                        if (team.team_rank() == 0) {
                            events_local.template RecordSolidificationEvent<Reconstruct_Calc, Solidification_Calc, RDF_Calc>(
                                p,
                                t_cur,
                                T_prev_val,
                                T_cur_val
                            );
                        }
                    }  
                }
            );
            Kokkos::fence();

            // Reset counts and give back to device
            Kokkos::deep_copy(events_local.sol_counts_h, events_local.sol_counts_d);
            events_local.sol_counts_h(SolCounts::needReconstruction) = 0;
            Kokkos::deep_copy(events_local.sol_counts_d, events_local.sol_counts_h);
        }
    }

    // template<typename FloatType>
    // template<bool Reconstruct_Calc, bool Solidification_Calc, bool RDF_Calc>
    // void Tracking<FloatType>::TrackSurface_Experimental(
    //     const Step_State<FloatType>& step,
    //     Thermal_State<FloatType>& thermal,
    //     const Grid::Device_Points<FloatType>& grid,
    //     const Calc::Device_Constants<FloatType>& const_d,
    //     const Calc::Device_Nodes<FloatType>& nodes_cur_d)
    // {
    //     struct IntScratch {
    //         enum {
    //             Claimed = 0,
    //             V2d = 1,
    //             Point = 2,
    //             HasLocal = 3,
    //             LocalV2d = 4,
    //             Count = 5,
    //         };
    //     };
        
    //     Kokkos::deep_copy(this->track_counts_h, this->track_counts_d);
    //     const int queue_count = this->track_counts_h(TrackingCounts::surfaceSeed);
    //     if (queue_count <= 0) {
    //         return;
    //     }
    //     const int iter_cur = step.iter_cur;
    //     const FloatType t_cur = step.t_cur;

    //     // What is this?
    //     this->surface_work_h(SurfaceWorklist::Head) = 0;
    //     this->surface_work_h(SurfaceWorklist::Tail) = queue_count;
    //     this->surface_work_h(SurfaceWorklist::Outstanding) = queue_count;
    //     this->surface_work_h(SurfaceWorklist::Done) = 0;
    //     Kokkos::deep_copy(this->surface_work_d, this->surface_work_h);

    //     int_deviceView surface_ready_stamp_local = this->surface_ready_stamp;
    //     Kokkos::parallel_for(
    //         "tracking_surface_ready_seed",
    //         Kokkos::RangePolicy<device_exe>(0, queue_count),
    //         KOKKOS_LAMBDA(const int q) {
    //             surface_ready_stamp_local(q) = iter_cur;
    //         });

    //     floating_deviceView& T_cur = thermal.T_cur;
    //     int_deviceView& T_iter_cur = thermal.T_calc_iter_cur;
    //     bool_deviceView& is_liq_cur = thermal.isLiq_cur;

    //     int_deviceView& surface_queue = this->surface_queue_cur;
    //     int_deviceView& depth_queue_local = this->depth_queue;
    //     int_x3_deviceView& counts = this->track_counts_d;
    //     int_2D_deviceView& depth_prev_local = this->depth_prev;
    //     int_2D_deviceView& depth_cur_local = this->depth_cur;
    //     int_2D_deviceView& column_seen_local = this->column_seen;
    //     int_deviceView& surface_work = this->surface_work_d;
    //     const int queue_capacity = this->column_count;
    //     Tracking_SolidificationEvents& events_local = this->events;

        
    //     // Help getting leagues size for maximum occupancy
    //     int C = space.concurrency(); // 8
    //     int V = Kokkos::TeamPolicy<Space>::vector_length_max(); 
    //     int T = policy.team_size_recommended(f, tag); 
    //     int L = C / (V * T); 
        
    //     // Persistant workers 
    //     // First Q read their first work to be done (works if Q>league size)
    //     // Others wait
    //     // Spawn points to queue via ??
    //     //

    //     // Wokers
    //     const int worker_count = L;
    //     const std::size_t scratch_bytes = static_cast<std::size_t>(IntScratch::Count) * sizeof(int);
    //     Kokkos::parallel_for(
    //         "tracking_surface_persistent",
    //         team_policy(L, T).set_scratch_size(0, Kokkos::PerTeam(scratch_bytes)),
    //         KOKKOS_LAMBDA(const team_member& team) {
    //             // Forced for constxpr compilation
    //             const Step_State<FloatType>& _force_step = step;
    //             const Thermal_State<FloatType>& _force_thermal = thermal;
    //             const Tracking_SolidificationEvents& _force_events = events_local;
    //             const FloatType _force_t_cur = t_cur;

    //             // Get scratch memory
    //             int* iscratch = reinterpret_cast<int*>(team.team_shmem().get_shmem(IntScratch::Count * sizeof(int)));
    //             if (team.team_rank() == 0) {
    //                 iscratch[IntScratch::HasLocal] = 0;
    //             }
    //             team.team_barrier();

    //             while (Kokkos::atomic_load(&surface_work(SurfaceWorklist::Done)) == 0) {
    //                 if (team.team_rank() == 0) {
    //                     iscratch[IntScratch::Claimed] = 0;
    //                     if (iscratch[IntScratch::HasLocal]) {
    //                         iscratch[IntScratch::V2d] = iscratch[IntScratch::LocalV2d];
    //                         iscratch[IntScratch::HasLocal] = 0;
    //                         iscratch[IntScratch::Claimed] = 1;
    //                     }
    //                     else {
    //                         const int head = Kokkos::atomic_load(&surface_work(SurfaceWorklist::Head));
    //                         const int tail = Kokkos::atomic_load(&surface_work(SurfaceWorklist::Tail));
    //                         if (head < tail) {
    //                             const int observed = Kokkos::atomic_compare_exchange(
    //                                 &surface_work(SurfaceWorklist::Head),
    //                                 head,
    //                                 head + 1);
    //                             if (observed == head) {
    //                                 while (Kokkos::atomic_load(&surface_ready_stamp_local(head)) != iter_cur) {}
    //                                 Kokkos::memory_fence();
    //                                 iscratch[IntScratch::V2d] = surface_queue(head);
    //                                 iscratch[IntScratch::Claimed] = 1;
    //                             }
    //                         }
    //                     }
    //                 }
    //                 team.team_barrier();

    //                 const int claimed = iscratch[IntScratch::Claimed];
    //                 if (claimed == 0) {
    //                     if (team.team_rank() == 0 &&
    //                         Kokkos::atomic_load(&surface_work(SurfaceWorklist::Outstanding)) == 0) {
    //                         Kokkos::atomic_exchange(&surface_work(SurfaceWorklist::Done), 1);
    //                     }
    //                     team.team_barrier();
    //                     continue;
    //                 }

    //                 if (team.team_rank() == 0) {
    //                     const int yNum = const_d.ints(IntConstants::Y_NUM);
    //                     const int zNum = const_d.ints(IntConstants::Z_NUM);
    //                     const int v2d = iscratch[IntScratch::V2d];
    //                     const int i = v2d / yNum;
    //                     const int j = v2d % yNum;
    //                     iscratch[IntScratch::Point] = Grid::ijk_to_p(i, j, zNum - 1, yNum, zNum);
    //                 }
    //                 team.team_barrier();

    //                 const int p_top = iscratch[IntScratch::Point];
    //                 const FloatType T = Calc::T_device(grid, nodes_cur_d, const_d, p_top, team);

    //                 if (team.team_rank() == 0) {
    //                     const int xNum = const_d.ints(IntConstants::X_NUM);
    //                     const int yNum = const_d.ints(IntConstants::Y_NUM);
    //                     const int zNum = const_d.ints(IntConstants::Z_NUM);
    //                     const FloatType T_liq = const_d.floats(FloatConstants::T_LIQ);

    //                     const int v2d = iscratch[IntScratch::V2d];
    //                     const int i = v2d / yNum;
    //                     const int j = v2d % yNum;

    //                     T_cur(p_top) = T;
    //                     T_iter_cur(p_top) = iter_cur;

    //                     const bool isMelted = (T >= T_liq);
    //                     const int old_depth = depth_prev_local(i, j);
    //                     if (isMelted) {
    //                         depth_cur_local(i, j) = (old_depth > 0) ? old_depth : 1;
    //                         is_liq_cur(p_top) = true;
    //                         depth_queue_local(Kokkos::atomic_fetch_add(&counts(TrackingCounts::liqColumns), 1)) = v2d;
    //                         if constexpr(RDF_Calc) {
    //                             if (!old_depth) {
    //                                 events_local.SaveMeltTime(t_cur, p_top);
    //                             }
    //                         }

    //                         bool kept_local = false;
    //                         for (int di = -1; di <= 1; ++di) {
    //                             for (int dj = -1; dj <= 1; ++dj) {
    //                                 const int ni = i + di;
    //                                 const int nj = j + dj;
    //                                 if (ni < 0 || ni >= xNum || nj < 0 || nj >= yNum) {
    //                                     continue;
    //                                 }
    //                                 if (Kokkos::atomic_compare_exchange(&column_seen_local(ni, nj), 0, 1) == 0) {
    //                                     Kokkos::atomic_fetch_add(&surface_work(SurfaceWorklist::Outstanding), 1);
    //                                     const int next_v2d = ni * yNum + nj;
    //                                     if (!kept_local) {
    //                                         iscratch[IntScratch::LocalV2d] = next_v2d;
    //                                         iscratch[IntScratch::HasLocal] = 1;
    //                                         kept_local = true;
    //                                     }
    //                                     else {
    //                                         const int pos = Kokkos::atomic_fetch_add(&surface_work(SurfaceWorklist::Tail), 1);
    //                                         if (pos >= queue_capacity) {
    //                                             Kokkos::abort("tracking_surface_persistent queue overflow");
    //                                         }
    //                                         surface_queue(pos) = next_v2d;
    //                                         Kokkos::memory_fence();
    //                                         Kokkos::atomic_exchange(&surface_ready_stamp_local(pos), iter_cur);
    //                                     }
    //                                 }
    //                             }
    //                         }
    //                     }
    //                     else {
    //                         depth_cur_local(i, j) = 0;
    //                         for (int d = 0; d < old_depth; ++d) {
    //                             const int p = Grid::ijk_to_p(i, j, (zNum - 1) - d, yNum, zNum);
    //                             is_liq_cur(p) = false;
    //                             if constexpr(Solidification_Calc) {
    //                                 events_local.template PrepareReconstruction<Reconstruct_Calc, Solidification_Calc, RDF_Calc>(step, thermal, p);
    //                             }
    //                         }
    //                     }

    //                     Kokkos::atomic_fetch_add(&surface_work(SurfaceWorklist::Outstanding), -1);
    //                 }
    //                 team.team_barrier();
    //             }
    //         });

    //     Kokkos::fence();
    //     Kokkos::deep_copy(this->track_counts_h, this->track_counts_d);
    // }

}

template class Condor::impl::Meltpool::Tracking<float>;
template class Condor::impl::Meltpool::Tracking<double>;
