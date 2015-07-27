// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * bfs_enactor.cuh
 *
 * @brief BFS Problem Enactor
 */

#pragma once

#include <gunrock/util/kernel_runtime_stats.cuh>
#include <gunrock/util/test_utils.cuh>

#include <gunrock/oprtr/advance/kernel.cuh>
#include <gunrock/oprtr/advance/kernel_policy.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>
#include <gunrock/oprtr/filter/kernel_policy.cuh>

#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/bfs/bfs_problem.cuh>
#include <gunrock/app/bfs/bfs_functor.cuh>


namespace gunrock {
namespace app {
namespace bfs {

/**
 * @brief BFS problem enactor class.
 *
 * @tparam INSTRUMWENT Boolean type to show whether or not to collect per-CTA clock-count statistics
 */
template<bool INSTRUMENT>
class BFSEnactor : public EnactorBase
{
    // Members
    protected:

    /**
     * A pinned, mapped word that the traversal kernels will signal when done
     */
    volatile int        *done;
    int                 *d_done;
    cudaEvent_t         throttle_event;

    // Methods
    protected:

    /**
     * @brief Prepare the enactor for BFS kernel call. Must be called prior to each BFS search.
     *
     * @param[in] problem BFS Problem object which holds the graph data and BFS problem data to compute.
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    template <typename ProblemData>
    cudaError_t Setup(
        ProblemData *problem)
    {
        typedef typename ProblemData::SizeT         SizeT;
        typedef typename ProblemData::VertexId      VertexId;
        
        cudaError_t retval = cudaSuccess;

        //initialize the host-mapped "done"
        if (!done) {
            int flags = cudaHostAllocMapped;

            // Allocate pinned memory for done
            if (retval = util::GRError(cudaHostAlloc((void**)&done, sizeof(int) * 1, flags),
                        "PBFSEnactor cudaHostAlloc done failed", __FILE__, __LINE__)) return retval;

            // Map done into GPU space
            if (retval = util::GRError(cudaHostGetDevicePointer((void**)&d_done, (void*) done, 0),
                        "PBFSEnactor cudaHostGetDevicePointer done failed", __FILE__, __LINE__)) return retval;

            // Create throttle event
            if (retval = util::GRError(cudaEventCreateWithFlags(&throttle_event, cudaEventDisableTiming),
                        "PBFSEnactor cudaEventCreateWithFlags throttle_event failed", __FILE__, __LINE__)) return retval;
        }

        done[0] = -1;

            //graph slice
            typename ProblemData::GraphSlice *graph_slice = problem->graph_slices[0];
            typename ProblemData::DataSlice *data_slice = problem->data_slices[0];

        do {

            // Bind row-offsets and bitmask texture
            cudaChannelFormatDesc   row_offsets_desc = cudaCreateChannelDesc<SizeT>();
            gunrock::oprtr::edge_map_forward::RowOffsetTex<SizeT>::ref.channelDesc = row_offsets_desc;
            /*if (retval = util::GRError(cudaBindTexture(
                    0,
                    gunrock::oprtr::edge_map_forward::RowOffsetTex<SizeT>::ref,
                    graph_slice->d_row_offsets,
                    (graph_slice->nodes + 1) * sizeof(SizeT)),
                        "BFSEnactor cudaBindTexture row_offset_tex_ref failed", __FILE__, __LINE__)) break;*/
            cudaBindTexture(
                    0,
                    gunrock::oprtr::edge_map_forward::RowOffsetTex<SizeT>::ref,
                    graph_slice->d_row_offsets,
                    (graph_slice->nodes + 1) * sizeof(SizeT));

            if (ProblemData::ENABLE_IDEMPOTENCE) {
                int bytes = (graph_slice->nodes + 8 - 1) / 8;
                cudaChannelFormatDesc   bitmask_desc = cudaCreateChannelDesc<char>();

                gunrock::oprtr::filter::BitmaskTex<unsigned char>::ref.channelDesc = bitmask_desc;
                if (retval = util::GRError(cudaBindTexture(
                                0,
                                gunrock::oprtr::filter::BitmaskTex<unsigned char>::ref,
                                data_slice->d_visited_mask,
                                bytes),
                            "BFSEnactor cudaBindTexture bitmask_tex_ref failed", __FILE__, __LINE__)) break;
            }

            /*cudaChannelFormatDesc   column_indices_desc = cudaCreateChannelDesc<VertexId>();
            gunrock::oprtr::edge_map_forward::ColumnIndicesTex<SizeT>::ref.channelDesc = column_indices_desc;
            if (retval = util::GRError(cudaBindTexture(
                            0,
                            gunrock::oprtr::edge_map_forward::ColumnIndicesTex<SizeT>::ref,
                            graph_slice->d_column_indices,
                            graph_slice->edges * sizeof(VertexId)),
                        "BFSEnactor cudaBindTexture column_indices_tex_ref failed", __FILE__, __LINE__)) break;*/
        } while (0);
        
        return retval;
    }

    public:

    /**
     * @brief BFSEnactor constructor
     */
    BFSEnactor(bool DEBUG = false) :
        EnactorBase(EDGE_FRONTIERS, DEBUG),
        done(NULL),
        d_done(NULL)
    {}

    /**
     * @brief BFSEnactor destructor
     */
    virtual ~BFSEnactor()
    {
        if (done) {
            util::GRError(cudaFreeHost((void*)done),
                "BFSEnactor cudaFreeHost done failed", __FILE__, __LINE__);

            util::GRError(cudaEventDestroy(throttle_event),
                "BFSEnactor cudaEventDestroy throttle_event failed", __FILE__, __LINE__);
        }
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief Obtain statistics about the last BFS search enacted.
     *
     * @param[out] total_queued Total queued elements in BFS kernel running.
     * @param[out] search_depth Search depth of BFS algorithm.
     * @param[out] avg_duty Average kernel running duty (kernel run time/kernel lifetime).
     */
    template <typename VertexId>
    void GetStatistics(
        long long &total_queued,
        VertexId &search_depth,
        double &avg_duty)
    {
        cudaThreadSynchronize();

        total_queued = enactor_stats.total_queued;
        search_depth = enactor_stats.iteration;

        avg_duty = (enactor_stats.total_lifetimes >0) ?
            double(enactor_stats.total_runtimes) / enactor_stats.total_lifetimes : 0.0;
    }

    /** @} */

    /**
     * @brief Enacts a breadth-first search computing on the specified graph.
     *
     * @tparam AdvanceKernelPolicy Kernel policy for advance operator.
     * @tparam FilterKernelPolicy Kernel policy for filter operator.
     * @tparam BFSProblem BFS Problem type.
     *
     * @param[in] context CudaContext pointer for moderngpu APIs
     * @param[in] problem BFSProblem object.
     * @param[in] src Source node for BFS.
     * @param[in] max_grid_size Max grid size for BFS kernel calls.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    template<
        typename AdvanceKernelPolicy,
        typename FilterKernelPolicy,
        typename BFSProblem>
    cudaError_t EnactBFS(
    CudaContext                          &context,
    BFSProblem                          *problem,
    typename BFSProblem::VertexId       src,
    int                                 max_grid_size = 0)
    {
        typedef typename BFSProblem::SizeT      SizeT;
        typedef typename BFSProblem::VertexId   VertexId;

        typedef BFSFunctor<
            VertexId,
            SizeT,
            VertexId,
            BFSProblem> BfsFunctor;

        cudaError_t retval = cudaSuccess;

        unsigned int    *d_scanned_edges = NULL; 
        do {
            // Determine grid size(s)
            if (DEBUG) {
                printf("Iteration, Edge map queue, Filter queue\n");
                printf("0");
            }


            // Lazy initialization
            if (retval = Setup(problem)) break;

            if (retval = EnactorBase::Setup(problem,
                                            max_grid_size,
                                            AdvanceKernelPolicy::CTA_OCCUPANCY, 
                                            FilterKernelPolicy::CTA_OCCUPANCY,
                                            AdvanceKernelPolicy::LOAD_BALANCED::BLOCKS)) break;


            // Single-gpu graph slice
            typename BFSProblem::GraphSlice *graph_slice = problem->graph_slices[0];
            typename BFSProblem::DataSlice *data_slice = problem->d_data_slices[0];

            frontier_attribute.queue_length         = 1;
            frontier_attribute.queue_index          = 0;        // Work queue index
            frontier_attribute.selector             = 0;

            frontier_attribute.queue_reset = true; 

            if (AdvanceKernelPolicy::ADVANCE_MODE == gunrock::oprtr::advance::LB) {
                if (retval = util::GRError(cudaMalloc(
                                (void**)&d_scanned_edges,
                                graph_slice->edges * sizeof(unsigned int)),
                            "PBFSProblem cudaMalloc d_scanned_edges failed", __FILE__, __LINE__)) return retval;
            }


            
            fflush(stdout);
            // Step through BFS iterations
            unsigned int iteration_ctr = 0;
            
            while (done[0] < 0) {
                ++iteration_ctr;
                // Edge Map
                //struct timeval start, end;
                //cudaDeviceSynchronize();
                //gettimeofday(&start, NULL);
                gunrock::oprtr::advance::LaunchKernel<AdvanceKernelPolicy, BFSProblem, BfsFunctor>(
                    d_done,
                    enactor_stats,
                    frontier_attribute,
                    data_slice,
                    (VertexId*)NULL,
                    (bool*)NULL,
                    (bool*)NULL,
                    d_scanned_edges,
                    graph_slice->frontier_queues.d_keys[frontier_attribute.selector],              // d_in_queue
                    graph_slice->frontier_queues.d_keys[frontier_attribute.selector^1],            // d_out_queue
                    (VertexId*)NULL,          // d_pred_in_queue
                    graph_slice->frontier_queues.d_values[frontier_attribute.selector^1],          // d_pred_out_queue
                    graph_slice->d_row_offsets,
                    graph_slice->d_column_indices,
                    (SizeT*)NULL,
                    (VertexId*)NULL,
                    graph_slice->frontier_elements[frontier_attribute.selector],
                    graph_slice->frontier_elements[frontier_attribute.selector^1],
                    this->work_progress,
                    context,
                    gunrock::oprtr::advance::V2V);
                //cudaDeviceSynchronize();
                //gettimeofday(&end, NULL);
                //std::cout << "advance " << iteration_ctr << " " << (end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec) << std::endl;

                /*gunrock::oprtr::edge_map_forward::Kernel<typename AdvanceKernelPolicy::THREAD_WARP_CTA_FORWARD, BFSProblem, BfsFunctor>
                <<<enactor_stats.advance_grid_size, AdvanceKernelPolicy::THREAD_WARP_CTA_FORWARD::THREADS>>>(
                    frontier_attribute.queue_reset,
                    frontier_attribute.queue_index,
                    enactor_stats.num_gpus,
                    enactor_stats.iteration,
                    frontier_attribute.queue_length,
                    d_done,
                    graph_slice->frontier_queues.d_keys[frontier_attribute.selector],              // d_in_queue
                    graph_slice->frontier_queues.d_values[frontier_attribute.selector^1],          // d_pred_out_queue
                    graph_slice->frontier_queues.d_keys[frontier_attribute.selector^1],            // d_out_queue
                    graph_slice->d_column_indices,
                    data_slice,
                    this->work_progress,
                    graph_slice->frontier_elements[frontier_attribute.selector],
                    graph_slice->frontier_elements[frontier_attribute.selector^1],
                    enactor_stats.advance_kernel_stats,
                    gunrock::oprtr::advance::V2V);*/
 

                // Only need to reset queue for once
                if (frontier_attribute.queue_reset)
                    frontier_attribute.queue_reset = false;

                if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(), "advance::Kernel failed", __FILE__, __LINE__))) break;
                cudaEventQuery(throttle_event);                                 // give host memory mapped visibility to GPU updates 

                frontier_attribute.queue_index++;
                frontier_attribute.selector ^= 1;

                if (AdvanceKernelPolicy::ADVANCE_MODE == gunrock::oprtr::advance::LB) {
                    if (retval = work_progress.GetQueueLength(frontier_attribute.queue_index, frontier_attribute.queue_length)) break;
                }
                
                if (DEBUG) {
                    if (retval = work_progress.GetQueueLength(frontier_attribute.queue_index, frontier_attribute.queue_length)) break;
                    printf(", %lld", (long long) frontier_attribute.queue_length);
                    //util::DisplayDeviceResults(graph_slice->frontier_queues.d_keys[frontier_attribute.selector], frontier_attribute.queue_length);
                    //util::DisplayDeviceResults(graph_slice->frontier_queues.d_values[frontier_attribute.selector], frontier_attribute.queue_length);
                }

                if (INSTRUMENT) {
                    if (retval = enactor_stats.advance_kernel_stats.Accumulate(
                        enactor_stats.advance_grid_size,
                        enactor_stats.total_runtimes,
                        enactor_stats.total_lifetimes)) break;
                }

                // Throttle
                if (enactor_stats.iteration & 1) {
                    if (retval = util::GRError(cudaEventRecord(throttle_event),
                        "BFSEnactor cudaEventRecord throttle_event failed", __FILE__, __LINE__)) break;
                } else {
                    if (retval = util::GRError(cudaEventSynchronize(throttle_event),
                        "BFSEnactor cudaEventSynchronize throttle_event failed", __FILE__, __LINE__)) break;
                }

                // Check if done
                if (done[0] == 0) break;

                // Filter
                unsigned long grid[3], block[3];
                grid[0] = enactor_stats.filter_grid_size; grid[1] = 1; grid[2] = 1;
                block[0] = FilterKernelPolicy::THREADS; block[1] = 1; block[2] = 1;
                KernelIdentifier kid("gunrock::oprtr::filter::Kernel", grid, block);
                long have_run_for=0;
                if(allocate == 0)
                {
                cudaMalloc(&d_yield_point_ret, sizeof(unsigned int));
                cudaMalloc(&d_elapsed_ret, sizeof(int));
                allocate = 1;
                }
                long service_id = -1, service_id_dummy = -1;
                h_yield_point = 0;
                h_elapsed = 0;
		bool global_only = false, yield_global = false, yield_global_select = false, yield_local = false, yield_local_select = false, global_only_select = false;
                if(/*(iteration_ctr == 4)||(iteration_ctr == 5)||(iteration_ctr == 6)*/true)
                {
                    yield_global_select = true;
                    yield_local_select = true;
                    global_only_select = true;
                }
		cudaMemcpy(&d_yield_point_persist, &h_yield_point, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(&d_yield_point, &h_yield_point, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(&d_elapsed, &h_elapsed, sizeof(int), cudaMemcpyHostToDevice);
		while(h_yield_point < grid[0]*grid[1]-1)
		{
			if(yield_global || (global_only && global_only_select))
			{
				if(h_yield_point == 0)
				{
					service_id = EvqueueLaunch(kid, have_run_for, service_id_dummy);
					std::cout << "New filter " << iteration_ctr << " " << h_yield_point << " " << service_id << std::endl;
					assert(service_id != -1);
				}
				else
				{
					service_id_dummy = EvqueueLaunch(kid, have_run_for, service_id);
					std::cout << "In service filter " << iteration_ctr << " " << h_yield_point << " " << have_run_for << " " << service_id << std::endl;
					assert(service_id_dummy == -1);
				}
				assert(service_id != -1);
			}
                        if(yield_local && yield_local_select)
                        {
				unsigned int allotted_slice=1000000; /*1000000000;*/
                struct timeval start, end;
		cudaMemcpy(d_yield_point_ret, &h_yield_point, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_elapsed_ret, &h_elapsed, sizeof(int), cudaMemcpyHostToDevice);
                //cudaDeviceSynchronize();
                gettimeofday(&start, NULL);
                gunrock::oprtr::filter::Kernelinstrumented<FilterKernelPolicy, BFSProblem, BfsFunctor>
                <<<enactor_stats.filter_grid_size, FilterKernelPolicy::THREADS>>>(
                    enactor_stats.iteration+1,
                    frontier_attribute.queue_reset,
                    frontier_attribute.queue_index,
                    enactor_stats.num_gpus,
                    frontier_attribute.queue_length,
                    d_done,
                    graph_slice->frontier_queues.d_keys[frontier_attribute.selector],      // d_in_queue
                    graph_slice->frontier_queues.d_values[frontier_attribute.selector],    // d_pred_in_queue
                    graph_slice->frontier_queues.d_keys[frontier_attribute.selector^1],    // d_out_queue
                    data_slice,
                    problem->data_slices[enactor_stats.gpu_id]->d_visited_mask,
                    work_progress,
                    graph_slice->frontier_elements[frontier_attribute.selector],           // max_in_queue
                    graph_slice->frontier_elements[frontier_attribute.selector^1],         // max_out_queue
                    enactor_stats.filter_kernel_stats,
                        true,
	                allotted_slice,
		        d_yield_point_ret, 
		        d_elapsed_ret);
                //cudaDeviceSynchronize();
				cudaMemcpy(&h_yield_point, d_yield_point_ret, sizeof(unsigned int), cudaMemcpyDeviceToHost);
                gettimeofday(&end, NULL);
				cudaMemcpy(&h_elapsed, d_elapsed_ret, sizeof(int), cudaMemcpyDeviceToHost);
                                assert(h_elapsed != -1);
                std::cout << "l filter " << iteration_ctr << " " << h_yield_point << " " << h_elapsed << " " << enactor_stats.filter_grid_size << " " << (end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec) << std::endl;
                                h_elapsed = -1;
                                break;

                        }
			else if(yield_global && yield_global_select && service_id != 1000000) /*yield needed*/
			{
				struct timeval start, end;
				unsigned int allotted_slice=1000000; /*1000000000;*/
				gettimeofday(&start, NULL);
                gunrock::oprtr::filter::Kernelinstrumented<FilterKernelPolicy, BFSProblem, BfsFunctor>
                <<<enactor_stats.filter_grid_size, FilterKernelPolicy::THREADS>>>(
                    enactor_stats.iteration+1,
                    frontier_attribute.queue_reset,
                    frontier_attribute.queue_index,
                    enactor_stats.num_gpus,
                    frontier_attribute.queue_length,
                    d_done,
                    graph_slice->frontier_queues.d_keys[frontier_attribute.selector],      // d_in_queue
                    graph_slice->frontier_queues.d_values[frontier_attribute.selector],    // d_pred_in_queue
                    graph_slice->frontier_queues.d_keys[frontier_attribute.selector^1],    // d_out_queue
                    data_slice,
                    problem->data_slices[enactor_stats.gpu_id]->d_visited_mask,
                    work_progress,
                    graph_slice->frontier_elements[frontier_attribute.selector],           // max_in_queue
                    graph_slice->frontier_elements[frontier_attribute.selector^1],         // max_out_queue
                    enactor_stats.filter_kernel_stats,
                        true,
	                allotted_slice,
		        d_yield_point_ret, 
		        d_elapsed_ret);
				cudaMemcpy(&h_yield_point, d_yield_point_ret, sizeof(unsigned int), cudaMemcpyDeviceToHost);
				gettimeofday(&end, NULL);
				cudaMemcpy(&h_elapsed, d_elapsed_ret, sizeof(int), cudaMemcpyDeviceToHost);
                                assert(h_elapsed != -1);
				have_run_for+=h_elapsed;
			        std::cout << "g filter " << iteration_ctr << " " << h_yield_point << " " << h_elapsed << " " << enactor_stats.filter_grid_size << " " << (end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec) << std::endl;
                                break;
			}
			else /*first run of this kernel, so don't yield*/
			{
                                //std::cout << "def filter " << iteration_ctr  << std::endl;
                                //struct timeval start, end;
                                //cudaDeviceSynchronize();
                                //gettimeofday(&start, NULL);
                gunrock::oprtr::filter::Kernel<FilterKernelPolicy, BFSProblem, BfsFunctor>
                <<<enactor_stats.filter_grid_size, FilterKernelPolicy::THREADS>>>(
                    enactor_stats.iteration+1,
                    frontier_attribute.queue_reset,
                    frontier_attribute.queue_index,
                    enactor_stats.num_gpus,
                    frontier_attribute.queue_length,
                    d_done,
                    graph_slice->frontier_queues.d_keys[frontier_attribute.selector],      // d_in_queue
                    graph_slice->frontier_queues.d_values[frontier_attribute.selector],    // d_pred_in_queue
                    graph_slice->frontier_queues.d_keys[frontier_attribute.selector^1],    // d_out_queue
                    data_slice,
                    problem->data_slices[enactor_stats.gpu_id]->d_visited_mask,
                    work_progress,
                    graph_slice->frontier_elements[frontier_attribute.selector],           // max_in_queue
                    graph_slice->frontier_elements[frontier_attribute.selector^1],         // max_out_queue
                    enactor_stats.filter_kernel_stats);
                                //cudaDeviceSynchronize();
                                //gettimeofday(&end, NULL);
				//std::cout << "def filter " << iteration_ctr << " " << (end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec) << std::endl;
				break;
			}
		}
                h_yield_point = 0;

                if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(), "filter_forward::Kernel failed", __FILE__, __LINE__))) break;
                cudaEventQuery(throttle_event); // give host memory mapped visibility to GPU updates


                frontier_attribute.queue_index++;
                frontier_attribute.selector ^= 1;

                if (AdvanceKernelPolicy::ADVANCE_MODE == gunrock::oprtr::advance::LB) {
                    if (retval = work_progress.GetQueueLength(frontier_attribute.queue_index, frontier_attribute.queue_length)) break;
                }

                if (INSTRUMENT || DEBUG) {
                    if (retval = work_progress.GetQueueLength(frontier_attribute.queue_index, frontier_attribute.queue_length)) break;
                    enactor_stats.total_queued += frontier_attribute.queue_length;
                    if (DEBUG) printf(", %lld", (long long) frontier_attribute.queue_length);
                    if (INSTRUMENT) {
                        if (retval = enactor_stats.filter_kernel_stats.Accumulate(
                            enactor_stats.filter_grid_size,
                            enactor_stats.total_runtimes,
                            enactor_stats.total_lifetimes)) break;
                    }
                }
                // Check if done
                if (done[0] == 0) break;

                enactor_stats.iteration++;

                if (DEBUG) printf("\n%lld", (long long) enactor_stats.iteration);

            }

            if (retval) break;

            // Check if any of the frontiers overflowed due to redundant expansion
            bool overflowed = false;
            if (retval = work_progress.CheckOverflow<SizeT>(overflowed)) break;
            if (overflowed) {
                retval = util::GRError(cudaErrorInvalidConfiguration, "Frontier queue overflow. Please increase queue-sizing factor.",__FILE__, __LINE__);
                break;
            }
            
        } while(0);

        if (d_scanned_edges) cudaFree(d_scanned_edges);

        if (DEBUG) printf("\nGPU BFS Done.\n");
        return retval;
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief BFS Enact kernel entry.
     *
     * @tparam BFSProblem BFS Problem type. @see BFSProblem
     *
     * @param[in] context CudaContext pointer for moderngpu APIs
     * @param[in] problem Pointer to BFSProblem object.
     * @param[in] src Source node for BFS.
     * @param[in] max_grid_size Max grid size for BFS kernel calls.
     * @param[in] traversal_mode Traversal Mode for advance operator: Load-balanced or Dynamic cooperative
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    template <typename BFSProblem>
    cudaError_t Enact(
        CudaContext                     &context,
        BFSProblem                      *problem,
        typename BFSProblem::VertexId    src,
        int                             max_grid_size = 0,
        int                             traversal_mode = 0)
    {
        if (BFSProblem::ENABLE_IDEMPOTENCE) {
            if (this->cuda_props.device_sm_version >= 300) {
                typedef gunrock::oprtr::filter::KernelPolicy<
                    BFSProblem,                         // Problem data type
                300,                                // CUDA_ARCH
                INSTRUMENT,                         // INSTRUMENT
                0,                                  // SATURATION QUIT
                true,                               // DEQUEUE_PROBLEM_SIZE
                8,                                  // MIN_CTA_OCCUPANCY
                8,                                  // LOG_THREADS
                1,                                  // LOG_LOAD_VEC_SIZE
                0,                                  // LOG_LOADS_PER_TILE
                5,                                  // LOG_RAKING_THREADS
                5,                                  // END_BITMASK_CULL
                8>                                  // LOG_SCHEDULE_GRANULARITY
                    FilterKernelPolicy;

                typedef gunrock::oprtr::advance::KernelPolicy<
                    BFSProblem,                         // Problem data type
                    300,                                // CUDA_ARCH
                    INSTRUMENT,                         // INSTRUMENT
                    8,                                  // MIN_CTA_OCCUPANCY
                    7,                                  // LOG_THREADS
                    8,                                  // LOG_BLOCKS
                    32*128,                                  // LIGHT_EDGE_THRESHOLD (used for partitioned advance mode)
                    1,                                  // LOG_LOAD_VEC_SIZE
                    0,                                  // LOG_LOADS_PER_TILE
                    5,                                  // LOG_RAKING_THREADS
                    32,                            // WARP_GATHER_THRESHOLD
                    128 * 4,                            // CTA_GATHER_THRESHOLD
                    7,                                  // LOG_SCHEDULE_GRANULARITY
                    gunrock::oprtr::advance::TWC_FORWARD>
                        ForwardAdvanceKernelPolicy;

                typedef gunrock::oprtr::advance::KernelPolicy<
                    BFSProblem,                         // Problem data type
                    300,                                // CUDA_ARCH
                    INSTRUMENT,                         // INSTRUMENT
                    8,                                  // MIN_CTA_OCCUPANCY
                    10,                                  // LOG_THREADS
                    8,                                  // LOG_BLOCKS
                    32*128,                                  // LIGHT_EDGE_THRESHOLD (used for partitioned advance mode)
                    1,                                  // LOG_LOAD_VEC_SIZE
                    0,                                  // LOG_LOADS_PER_TILE
                    5,                                  // LOG_RAKING_THREADS
                    32,                            // WARP_GATHER_THRESHOLD
                    128 * 4,                            // CTA_GATHER_THRESHOLD
                    7,                                  // LOG_SCHEDULE_GRANULARITY
                    gunrock::oprtr::advance::LB>
                        LBAdvanceKernelPolicy;

                if (traversal_mode == 0)
                return EnactBFS<LBAdvanceKernelPolicy, FilterKernelPolicy, BFSProblem>(
                        context, problem, src, max_grid_size);
                else
                return EnactBFS<ForwardAdvanceKernelPolicy, FilterKernelPolicy, BFSProblem>(
                        context, problem, src, max_grid_size);
            }
        } else {
                if (this->cuda_props.device_sm_version >= 300) {
                typedef gunrock::oprtr::filter::KernelPolicy<
                    BFSProblem,                         // Problem data type
                300,                                // CUDA_ARCH
                INSTRUMENT,                         // INSTRUMENT
                0,                                  // SATURATION QUIT
                true,                               // DEQUEUE_PROBLEM_SIZE
                8,                                  // MIN_CTA_OCCUPANCY
                8,                                  // LOG_THREADS
                1,                                  // LOG_LOAD_VEC_SIZE
                0,                                  // LOG_LOADS_PER_TILE
                5,                                  // LOG_RAKING_THREADS
                5,                                  // END_BITMASK_CULL
                8>                                  // LOG_SCHEDULE_GRANULARITY
                    FilterKernelPolicy;

                typedef gunrock::oprtr::advance::KernelPolicy<
                    BFSProblem,                         // Problem data type
                    300,                                // CUDA_ARCH
                    INSTRUMENT,                         // INSTRUMENT
                    1,                                  // MIN_CTA_OCCUPANCY
                    7,                                  // LOG_THREADS
                    8,                                  // LOG_BLOCKS
                    32*128,                                  // LIGHT_EDGE_THRESHOLD (used for partitioned advance mode)
                    1,                                  // LOG_LOAD_VEC_SIZE
                    0,                                  // LOG_LOADS_PER_TILE
                    5,                                  // LOG_RAKING_THREADS
                    32,                            // WARP_GATHER_THRESHOLD
                    128 * 4,                            // CTA_GATHER_THRESHOLD
                    7,                                  // LOG_SCHEDULE_GRANULARITY
                    gunrock::oprtr::advance::TWC_FORWARD>
                        ForwardAdvanceKernelPolicy;

                typedef gunrock::oprtr::advance::KernelPolicy<
                    BFSProblem,                         // Problem data type
                    300,                                // CUDA_ARCH
                    INSTRUMENT,                         // INSTRUMENT
                    1,                                  // MIN_CTA_OCCUPANCY
                    10,                                  // LOG_THREADS
                    8,                                  // LOG_BLOCKS
                    32*128,                                  // LIGHT_EDGE_THRESHOLD (used for partitioned advance mode)
                    1,                                  // LOG_LOAD_VEC_SIZE
                    0,                                  // LOG_LOADS_PER_TILE
                    5,                                  // LOG_RAKING_THREADS
                    32,                            // WARP_GATHER_THRESHOLD
                    128 * 4,                            // CTA_GATHER_THRESHOLD
                    7,                                  // LOG_SCHEDULE_GRANULARITY
                    gunrock::oprtr::advance::LB>
                        LBAdvanceKernelPolicy;

                if (traversal_mode == 0)
                return EnactBFS<LBAdvanceKernelPolicy, FilterKernelPolicy, BFSProblem>(
                        context, problem, src, max_grid_size);
                else
                return EnactBFS<ForwardAdvanceKernelPolicy, FilterKernelPolicy, BFSProblem>(
                        context, problem, src, max_grid_size);
            }
        }

        //to reduce compile time, get rid of other architecture for now
        //TODO: add all the kernelpolicy settings for all archs

        printf("Not yet tuned for this architecture\n");
        return cudaErrorInvalidDeviceFunction;
    }

    /** @} */

};

} // namespace bfs
} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
