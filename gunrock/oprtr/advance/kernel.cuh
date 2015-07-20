#pragma once
#include <gunrock/util/basic_utils.h>
#include <gunrock/util/cuda_properties.cuh>
#include <gunrock/util/cta_work_distribution.cuh>
#include <gunrock/util/soa_tuple.cuh>
#include <gunrock/util/srts_grid.cuh>
#include <gunrock/util/srts_soa_details.cuh>
#include <gunrock/util/io/modified_load.cuh>
#include <gunrock/util/io/modified_store.cuh>
#include <gunrock/util/operators.cuh>

#include <gunrock/util/test_utils.cuh>
#include <gunrock/app/problem_base.cuh>
#include <gunrock/app/enactor_base.cuh>

#include <gunrock/util/cta_work_distribution.cuh>
#include <gunrock/util/cta_work_progress.cuh>
#include <gunrock/util/kernel_runtime_stats.cuh>

#include <gunrock/oprtr/edge_map_forward/kernel.cuh>
#include <gunrock/oprtr/edge_map_backward/kernel.cuh>
#include <gunrock/oprtr/edge_map_partitioned_backward/kernel.cuh>
#include <gunrock/oprtr/edge_map_partitioned/kernel.cuh>

#include <gunrock/oprtr/advance/kernel_policy.cuh>

#include <moderngpu.cuh>

#include <sys/time.h>
#include <cstdlib>

#include "EvqueueManager.h"

extern EvqueueManager *evqm;

namespace gunrock {
namespace oprtr {

struct timeval start, end;
/*
unsigned int h_yield_point;
int h_elapsed;
unsigned int *d_yield_point_ret;
int *d_elapsed_ret;
int allocate;
*/
bool done_once=false;

#define Errchk(ans) { DrvAssert((ans), __FILE__, __LINE__); }
inline void DrvAssert( CUresult code, const char *file, int line)
{
    if (code != CUDA_SUCCESS) {
        std::cout << "Error: " << code << " " <<  file << "@" << line << std::endl;
        exit(code);
    } else {
        std::cout << "Success: " << file << "@" << line << std::endl;
    }
}

namespace advance {
/**
 * @brief Advance operator kernel entry point.
 *
 * @tparam KernelPolicy Kernel policy type for advance operator.
 * @tparam ProblemData Problem data type for advance operator.
 * @tparam Functor Functor type for the specific problem type.
 * @tparam Op Operation for gather reduce. mgpu::plus<int> by default.
 *
 * @param[in] d_done                    Pointer of volatile int to the flag to set when we detect incoming frontier is empty
 * @param[in] enactor_stats             EnactorStats object to store enactor related variables and stast
 * @param[in] frontier_attribute        FrontierAttribute object to store frontier attribute while doing the advance operation
 * @param[in] data_slice                Device pointer to the problem object's data_slice member
 * @param[in] backward_index_queue      If backward mode is activated, this is used to store the vertex index. (deprecated)
 * @param[in] backward_frontier_map_in  If backward mode is activated, this is used to store input frontier bitmap
 * @param[in] backward_frontier_map_out If backward mode is activated, this is used to store output frontier bitmap
 * @param[in] partitioned_scanned_edges If load balanced mode is activated, this is used to store the scanned edge number for neighbor lists in current frontier
 * @param[in] d_in_key_queue            Device pointer of input key array to the incoming frontier queue
 * @param[in] d_out_key_queue           Device pointer of output key array to the outgoing frontier queue
 * @param[in] d_in_value_queue          Device pointer of input value array to the incoming frontier queue
 * @param[in] d_out_value_queue         Device pointer of output value array to the outgoing frontier queue
 * @param[in] d_row_offsets             Device pointer of SizeT to the row offsets queue
 * @param[in] d_column_indices          Device pointer of VertexId to the column indices queue
 * @param[in] d_column_offsets          Device pointer of SizeT to the row offsets queue for inverse graph
 * @param[in] d_row_indices             Device pointer of VertexId to the column indices queue for inverse graph
 * @param[in] max_in_queue              Maximum number of elements we can place into the incoming frontier
 * @param[in] max_out_queue             Maximum number of elements we can place into the outgoing frontier
 * @param[in] work_progress             queueing counters to record work progress
 * @param[in] context                   CudaContext pointer for moderngpu APIs
 * @param[in] ADVANCE_TYPE              enumerator of advance type: V2V, V2E, E2V, or E2E
 * @param[in] inverse_graph             whether this iteration of advance operation is in the opposite direction to the previous iteration (false by default)
 * @param[in] REDUCE_OP                 enumerator of available reduce operations: plus, multiplies, bit_or, bit_and, bit_xor, maximum, minimum. none by default.
 * @param[in] REDUCE_TYPE               enumerator of available reduce types: EMPTY(do not do reduce) VERTEX(extract value from |V| array) EDGE(extract value from |E| array)
 * @param[in] d_value_to_reduce         array to store values to reduce
 * @param[out] d_reduce_frontier        neighbor list values for nodes in the output frontier
 * @param[out] d_reduced_value          array to store reduced values
 */

//TODO: Reduce by neighbor list now only supports LB advance mode.
//TODO: Add a switch to enable advance+filter (like in BFS), pissibly moving idempotent ops from filter to advance?
unsigned int iteration_ctr = 0;
template <typename KernelPolicy, typename ProblemData, typename Functor>
    void LaunchKernel(
            volatile int                            *d_done,
            gunrock::app::EnactorStats              &enactor_stats,
            gunrock::app::FrontierAttribute         &frontier_attribute,
            typename ProblemData::DataSlice         *data_slice,
            typename ProblemData::VertexId          *backward_index_queue,
            bool                                    *backward_frontier_map_in,
            bool                                    *backward_frontier_map_out,
            unsigned int                            *partitioned_scanned_edges,
            typename KernelPolicy::VertexId         *d_in_key_queue,
            typename KernelPolicy::VertexId         *d_out_key_queue,
            typename KernelPolicy::VertexId         *d_in_value_queue,
            typename KernelPolicy::VertexId         *d_out_value_queue,
            typename KernelPolicy::SizeT            *d_row_offsets,
            typename KernelPolicy::VertexId         *d_column_indices,
            typename KernelPolicy::SizeT            *d_column_offsets,
            typename KernelPolicy::VertexId         *d_row_indices,
            typename KernelPolicy::SizeT            max_in,
            typename KernelPolicy::SizeT            max_out,
            util::CtaWorkProgress                   work_progress,
            CudaContext                             &context,
            TYPE                                    ADVANCE_TYPE,
            bool                                    inverse_graph = false,
            REDUCE_OP                               R_OP = gunrock::oprtr::advance::NONE,
            REDUCE_TYPE                             R_TYPE = gunrock::oprtr::advance::EMPTY,
            typename KernelPolicy::Value            *d_value_to_reduce = NULL,
            typename KernelPolicy::Value            *d_reduce_frontier = NULL,
            typename KernelPolicy::Value            *d_reduced_value = NULL)

{
    ++iteration_ctr;
    if(iteration_ctr == 13)
       iteration_ctr = 0;
    //std::cout << "LaunchKernel " << iteration_ctr << std::endl;
    switch (KernelPolicy::ADVANCE_MODE)
    {
        case TWC_FORWARD:
        {
            //std::cout << "TWC_FORWARD " << iteration_ctr << std::endl;
            // Load Thread Warp CTA Forward Kernel
            gunrock::oprtr::edge_map_forward::Kernel<typename KernelPolicy::THREAD_WARP_CTA_FORWARD, ProblemData, Functor>
                <<<enactor_stats.advance_grid_size, KernelPolicy::THREAD_WARP_CTA_FORWARD::THREADS>>>(
                    frontier_attribute.queue_reset,
                    frontier_attribute.queue_index,
                    enactor_stats.num_gpus,
                    enactor_stats.iteration,
                    frontier_attribute.queue_length,
                    d_done,
                    d_in_key_queue,              // d_in_queue
                    d_out_value_queue,          // d_pred_out_queue
                    d_out_key_queue,            // d_out_queue
                    d_column_indices,
                    d_row_indices,
                    data_slice,
                    work_progress,
                    max_in,                   // max_in_queue
                    max_out,                 // max_out_queue
                    enactor_stats.advance_kernel_stats,
                    ADVANCE_TYPE,
                    inverse_graph);
            break;
        }
        case LB_BACKWARD:
        {
            //std::cout << "LB_BACKWARD " << iteration_ctr << std::endl;
            // Load Thread Warp CTA Backward Kernel
            typedef typename ProblemData::SizeT         SizeT;
            typedef typename ProblemData::VertexId      VertexId;
            typedef typename KernelPolicy::LOAD_BALANCED LBPOLICY;
            // Load Load Balanced Kernel
            // Get Rowoffsets
            // Use scan to compute edge_offsets for each vertex in the frontier
            // Use sorted sort to compute partition bound for each work-chunk
            // load edge-expand-partitioned kernel
            //util::DisplayDeviceResults(d_in_key_queue, frontier_attribute.queue_length);
            int num_block = (frontier_attribute.queue_length + KernelPolicy::LOAD_BALANCED::THREADS - 1)/KernelPolicy::LOAD_BALANCED::THREADS;
            gunrock::oprtr::edge_map_partitioned_backward::GetEdgeCounts<typename KernelPolicy::LOAD_BALANCED, ProblemData, Functor>
            <<< num_block, KernelPolicy::LOAD_BALANCED::THREADS >>>(
                                        d_column_offsets,
                                        d_row_indices,
                                        d_in_key_queue,
                                        partitioned_scanned_edges,
                                        frontier_attribute.queue_length+1,
                                        max_in,
                                        max_out,
                                        ADVANCE_TYPE);

            Scan<mgpu::MgpuScanTypeExc>((int*)partitioned_scanned_edges, frontier_attribute.queue_length+1, (int)0, mgpu::plus<int>(),
            (int*)0, (int*)0, (int*)partitioned_scanned_edges, context);

            SizeT *temp = new SizeT[1];
            cudaMemcpy(temp,partitioned_scanned_edges+frontier_attribute.queue_length, sizeof(SizeT), cudaMemcpyDeviceToHost);
            SizeT output_queue_len = temp[0];
            //printf("input queue:%d, output_queue:%d\n", frontier_attribute.queue_length, output_queue_len);

            if (frontier_attribute.selector == 1) {
                // Edge Map
                gunrock::oprtr::edge_map_partitioned_backward::RelaxLightEdges<LBPOLICY, ProblemData, Functor>
                <<< num_block, KernelPolicy::LOAD_BALANCED::THREADS >>>(
                        frontier_attribute.queue_reset,
                        frontier_attribute.queue_index,
                        enactor_stats.iteration,
                        d_column_offsets,
                        d_row_indices,
                        (VertexId*)NULL,
                        &partitioned_scanned_edges[1],
                        d_done,
                        d_in_key_queue,
                        backward_frontier_map_in,
                        backward_frontier_map_out,
                        data_slice,
                        frontier_attribute.queue_length,
                        output_queue_len,
                        max_in,
                        max_out,
                        work_progress,
                        enactor_stats.advance_kernel_stats,
                        ADVANCE_TYPE,
                        inverse_graph);
            } else {
                // Edge Map
                gunrock::oprtr::edge_map_partitioned_backward::RelaxLightEdges<LBPOLICY, ProblemData, Functor>
                <<< num_block, KernelPolicy::LOAD_BALANCED::THREADS >>>(
                        frontier_attribute.queue_reset,
                        frontier_attribute.queue_index,
                        enactor_stats.iteration,
                        d_column_offsets,
                        d_row_indices,
                        (VertexId*)NULL,
                        &partitioned_scanned_edges[1],
                        d_done,
                        d_in_key_queue,
                        backward_frontier_map_out,
                        backward_frontier_map_in,
                        data_slice,
                        frontier_attribute.queue_length,
                        output_queue_len,
                        max_in,
                        max_out,
                        work_progress,
                        enactor_stats.advance_kernel_stats,
                        ADVANCE_TYPE,
                        inverse_graph);
            }
            break;
        }
        case TWC_BACKWARD:
        {
            //std::cout << "TWC_BACKWARD " << iteration_ctr << std::endl;
            // Load Thread Warp CTA Backward Kernel
            if (frontier_attribute.selector == 1) {
                // Edge Map
                gunrock::oprtr::edge_map_backward::Kernel<typename KernelPolicy::THREAD_WARP_CTA_BACKWARD, ProblemData, Functor>
                    <<<enactor_stats.advance_grid_size, KernelPolicy::THREAD_WARP_CTA_BACKWARD::THREADS>>>(
                            frontier_attribute.queue_reset,
                            frontier_attribute.queue_index,
                            enactor_stats.num_gpus,
                            frontier_attribute.queue_length,
                            d_done,
                            d_in_key_queue,              // d_in_queue
                            backward_index_queue,            // d_in_index_queue
                            backward_frontier_map_in,
                            backward_frontier_map_out,
                            d_column_offsets,
                            d_row_indices,
                            data_slice,
                            work_progress,
                            enactor_stats.advance_kernel_stats,
                            ADVANCE_TYPE);
            } else {
                // Edge Map
                gunrock::oprtr::edge_map_backward::Kernel<typename KernelPolicy::THREAD_WARP_CTA_BACKWARD, ProblemData, Functor>
                    <<<enactor_stats.advance_grid_size, KernelPolicy::THREAD_WARP_CTA_BACKWARD::THREADS>>>(
                            frontier_attribute.queue_reset,
                            frontier_attribute.queue_index,
                            enactor_stats.num_gpus,
                            frontier_attribute.queue_length,
                            d_done,
                            d_in_key_queue,              // d_in_queue
                            backward_index_queue,            // d_in_index_queue
                            backward_frontier_map_out,
                            backward_frontier_map_in,
                            d_column_offsets,
                            d_row_indices,
                            data_slice,
                            work_progress,
                            enactor_stats.advance_kernel_stats,
                            ADVANCE_TYPE);
            }
            break;
        }
        case LB:
        {
            //std::cout << "LB " << iteration_ctr << std::endl;
            typedef typename ProblemData::SizeT         SizeT;
            typedef typename ProblemData::VertexId      VertexId;
            typedef typename ProblemData::Value         Value;
            typedef typename KernelPolicy::LOAD_BALANCED LBPOLICY;
            int num_block = (frontier_attribute.queue_length + KernelPolicy::LOAD_BALANCED::THREADS - 1)/KernelPolicy::LOAD_BALANCED::THREADS;
            gunrock::oprtr::edge_map_partitioned::GetEdgeCounts<typename KernelPolicy::LOAD_BALANCED, ProblemData, Functor>
            <<< num_block, KernelPolicy::LOAD_BALANCED::THREADS >>>(
                                        d_row_offsets,
                                        d_column_indices,
                                        d_in_key_queue,
                                        partitioned_scanned_edges,
                                        frontier_attribute.queue_length+1,
                                        max_in,
                                        max_out,
                                        ADVANCE_TYPE);

            Scan<mgpu::MgpuScanTypeExc>((int*)partitioned_scanned_edges, frontier_attribute.queue_length+1, (int)0, mgpu::plus<int>(),
            (int*)0, (int*)0, (int*)partitioned_scanned_edges, context);

            SizeT *temp = new SizeT[1];
            cudaMemcpy(temp,partitioned_scanned_edges+frontier_attribute.queue_length, sizeof(SizeT), cudaMemcpyDeviceToHost);
            SizeT output_queue_len = temp[0];

            if (output_queue_len < LBPOLICY::LIGHT_EDGE_THRESHOLD)
            {
		//std::cout << "LB LIGHT_EDGE_THRESHOLD" << iteration_ctr << std::endl;
                gunrock::oprtr::edge_map_partitioned::RelaxLightEdges<LBPOLICY, ProblemData, Functor>
                <<< num_block, KernelPolicy::LOAD_BALANCED::THREADS >>>(
                        frontier_attribute.queue_reset,
                        frontier_attribute.queue_index,
                        enactor_stats.iteration,
                        d_row_offsets,
                        d_column_indices,
                        d_row_indices,
                        &partitioned_scanned_edges[1],
                        d_done,
                        d_in_key_queue,
                        d_out_key_queue,
                        data_slice,
                        frontier_attribute.queue_length,
                        output_queue_len,
                        max_in,
                        max_out,
                        work_progress,
                        enactor_stats.advance_kernel_stats,
                        ADVANCE_TYPE,
                        inverse_graph,
                        R_TYPE,
                        R_OP,
                        d_value_to_reduce,
                        d_reduce_frontier);
            }
            else
            {
		//std::cout << "LB ~LIGHT_EDGE_THRESHOLD" << iteration_ctr << std::endl;
                unsigned int split_val = (output_queue_len + KernelPolicy::LOAD_BALANCED::BLOCKS - 1) / KernelPolicy::LOAD_BALANCED::BLOCKS;
                int num_block = KernelPolicy::LOAD_BALANCED::BLOCKS;
                int nb = (num_block + 1 + KernelPolicy::LOAD_BALANCED::THREADS - 1)/KernelPolicy::LOAD_BALANCED::THREADS;
                gunrock::oprtr::edge_map_partitioned::MarkPartitionSizes<typename KernelPolicy::LOAD_BALANCED, ProblemData, Functor>
                    <<<nb, KernelPolicy::LOAD_BALANCED::THREADS>>>(
                            enactor_stats.d_node_locks,
                            split_val,
                            num_block+1,
                            output_queue_len);
                //util::MemsetIdxKernel<<<128, 128>>>(enactor_stats.d_node_locks, KernelPolicy::LOAD_BALANCED::BLOCKS, split_val);

                SortedSearch<MgpuBoundsLower>(
                        enactor_stats.d_node_locks,
                        KernelPolicy::LOAD_BALANCED::BLOCKS,
                        &partitioned_scanned_edges[1],
                        frontier_attribute.queue_length,
                        enactor_stats.d_node_locks_out,
                        context);
#if 0
                if(!done_once)
                {
                   done_once = true;
                   CUmodule module;
                   std::string path = "test_pr.ptx";
                   std::cout << "Calling cuModuleLoad" << std::endl;
                   CUresult code = cuModuleLoad(&module, path.c_str());
                   if(code == CUDA_SUCCESS)
                   {
                       std::cout << "cuModuleLoad returned CUDA_SUCCESS" << std::endl;
                   }
                   CUfunction func;
                   std::string name;
                   //name = "RelaxPartitionedEdges2instrumented";
                   //name = "_ZN7gunrock5oprtr20edge_map_partitioned34RelaxPartitionedEdges2instrumentedINS1_12KernelPolicyINS_3app2pr9PRProblemIiifEELi300ELb1ELi1ELi7ELi10ELi4096EEES7_NS5_27RemoveZeroDegreeNodeFunctorIiifS7_EEEEvbNT_8VertexIdEiPNSB_5SizeTEPSC_SF_PjSG_jPViSF_SF_PNT0_9DataSliceESD_SD_SD_SD_SD_NS_4util15CtaWorkProgressENSM_18KernelRuntimeStatsENS0_7advance4TYPEEbNSP_11REDUCE_TYPEENSP_9REDUCE_OPEPNSB_5ValueESU_SG_Pi";
                   //name = "_ZN7gunrock5oprtr20edge_map_partitioned22RelaxPartitionedEdges2INS1_12KernelPolicyINS_3app2pr9PRProblemIiifEELi300ELb1ELi1ELi7ELi10ELi4096EEES7_NS5_27RemoveZeroDegreeNodeFunctorIiifS7_EEEEvbNT_8VertexIdEiPNSB_5SizeTEPSC_SF_PjSG_jPViSF_SF_PNT0_9DataSliceESD_SD_SD_SD_SD_NS_4util15CtaWorkProgressENSM_18KernelRuntimeStatsENS0_7advance4TYPEEbNSP_11REDUCE_TYPEENSP_9REDUCE_OPEPNSB_5ValueESU_";
                   //name = "_ZN7gunrock5oprtr20edge_map_partitioned22RelaxPartitionedEdges2INS1_12KernelPolicyINS_3app2pr9PRProblemIiifEELi300ELb1ELi1ELi7ELi10ELi4096EEES7_NS5_9PRFunctorIiifS7_EEEEvbNT_8VertexIdEiPNSB_5SizeTEPSC_SF_PjSG_jPViSF_SF_PNT0_9DataSliceESD_SD_SD_SD_SD_NS_4util15CtaWorkProgressENSM_18KernelRuntimeStatsENS0_7advance4TYPEEbNSP_11REDUCE_TYPEENSP_9REDUCE_OPEPNSB_5ValueESU_";
                   //name = "_ZN7gunrock5oprtr20edge_map_partitioned22RelaxPartitionedEdges2INS1_12KernelPolicyINS_3app2pr9PRProblemIiifEELi300ELb1ELi1ELi10ELi8ELi4096EEES7_NS5_27RemoveZeroDegreeNodeFunctorIiifS7_EEEEvbNT_8VertexIdEiPNSB_5SizeTEPSC_SF_PjSG_jPViSF_SF_PNT0_9DataSliceESD_SD_SD_SD_SD_NS_4util15CtaWorkProgressENSM_18KernelRuntimeStatsENS0_7advance4TYPEEbNSP_11REDUCE_TYPEENSP_9REDUCE_OPEPNSB_5ValueESU_";
                   name = "_ZN7gunrock5oprtr20edge_map_partitioned22RelaxPartitionedEdges2INS1_12KernelPolicyINS_3app2pr9PRProblemIiifEELi300ELb1ELi1ELi10ELi8ELi4096EEES7_NS5_9PRFunctorIiifS7_EEEEvbNT_8VertexIdEiPNSB_5SizeTEPSC_SF_PjSG_jPViSF_SF_PNT0_9DataSliceESD_SD_SD_SD_SD_NS_4util15CtaWorkProgressENSM_18KernelRuntimeStatsENS0_7advance4TYPEEbNSP_11REDUCE_TYPEENSP_9REDUCE_OPEPNSB_5ValueESU_";

                   code = cuModuleGetFunction(&func, module, name.c_str());
                   if(code == CUDA_SUCCESS)
                   {
                       std::cout << "cuModuleGetFunction returned CUDA_SUCCESS" << std::endl;
                   }
                   else
                   {
                       std::cout << "cuModuleGetFunction returned " << code << std::endl;
                   }
                   void **kernel_parameters = new void*[26];
                   //for(int i=0; i<26; i++)
                   //{
                      std::cout << "frontier_attribute.queue_reset " << frontier_attribute.queue_reset << " " << sizeof(frontier_attribute.queue_reset) << std::endl;
                      kernel_parameters[0] = new char[sizeof(frontier_attribute.queue_reset)];
                      std::memcpy(kernel_parameters[0], &frontier_attribute.queue_reset, sizeof(frontier_attribute.queue_reset));
                      
                      std::cout << "frontier_attribute.queue_index " << frontier_attribute.queue_index << " " << sizeof(frontier_attribute.queue_index) << std::endl;
                      kernel_parameters[1] = new char[sizeof(frontier_attribute.queue_index)];
                      std::memcpy(kernel_parameters[1], &frontier_attribute.queue_index, sizeof(frontier_attribute.queue_reset));

                      std::cout << "enactor_stats.iteration " << enactor_stats.iteration << " " << sizeof(enactor_stats.iteration) << std::endl;
                      kernel_parameters[2] = new char[sizeof(frontier_attribute.queue_reset)];
                      std::memcpy(kernel_parameters[2], &frontier_attribute.queue_reset, sizeof(frontier_attribute.queue_reset));

                      std::cout << "d_row_offsets " << std::hex << d_row_offsets << " " << std::dec << sizeof(d_row_offsets) << std::endl;
                      kernel_parameters[3] = new char[sizeof(d_row_offsets)];
                      std::memcpy(kernel_parameters[3], &d_row_offsets, sizeof(d_row_offsets));

                      std::cout << "d_column_indices " << std::hex << d_column_indices << " " << std::dec << sizeof(d_column_indices) << std::endl;
                      kernel_parameters[4] = new char[sizeof(d_column_indices)];
                      std::memcpy(kernel_parameters[4], &d_column_indices, sizeof(d_column_indices));

                      std::cout << "d_row_indices " << std::hex << d_row_indices << " " << std::dec << sizeof(d_row_indices) << std::endl;
                      kernel_parameters[5] = new char[sizeof(d_row_indices)];
                      std::memcpy(kernel_parameters[5], &d_row_indices, sizeof(d_row_indices));

                      std::cout << "&partitioned_scanned_edges[1] " << &partitioned_scanned_edges[1] << std::hex << " " << std::dec << sizeof(void*) << std::endl;
                      kernel_parameters[6] = new char[sizeof(void*)];
                      void *temp = &partitioned_scanned_edges[1];
                      std::memcpy(kernel_parameters[6], &temp, sizeof(void*));

                      std::cout << "enactor_stats.d_node_locks_out " << enactor_stats.d_node_locks_out << " " << sizeof(enactor_stats.d_node_locks_out) << std::endl;
                      kernel_parameters[7] = new char[sizeof(enactor_stats.d_node_locks_out)];
                      std::memcpy(kernel_parameters[7], &enactor_stats.d_node_locks_out, sizeof(enactor_stats.d_node_locks_out));

                      int temp_val = KernelPolicy::LOAD_BALANCED::BLOCKS;
                      std::cout << "KernelPolicy::LOAD_BALANCED::BLOCKS " << KernelPolicy::LOAD_BALANCED::BLOCKS << " " << sizeof(temp_val) << std::endl;
                      kernel_parameters[8] = new char[sizeof(temp_val)];
                      std::memcpy(kernel_parameters[8], &temp_val, sizeof(temp_val));

                      std::cout << "d_done " << std::hex << d_done << " " << std::dec << sizeof(d_done) << std::endl;
                      kernel_parameters[9] = new char[sizeof(d_done)];
                      std::memcpy(kernel_parameters[9], &d_done, sizeof(d_done));

                      std::cout << "d_in_key_queue " << std::hex << d_in_key_queue << " " << std::dec << sizeof(d_in_key_queue) << std::endl;
                      kernel_parameters[10] = new char[sizeof(d_in_key_queue)];
                      std::memcpy(kernel_parameters[10], &d_in_key_queue, sizeof(d_in_key_queue));

                      std::cout << "d_out_key_queue " << std::hex << d_out_key_queue << " " << std::dec << sizeof(d_out_key_queue) << std::endl;
                      kernel_parameters[11] = new char[sizeof(d_out_key_queue)];
                      std::memcpy(kernel_parameters[11], &d_out_key_queue, sizeof(d_out_key_queue));

                      kernel_parameters[12] = new char[sizeof(data_slice)];
                      std::memcpy(kernel_parameters[12], &data_slice, sizeof(data_slice));

                      kernel_parameters[13] = new char[sizeof(frontier_attribute.queue_length)];
                      std::memcpy(kernel_parameters[13], &frontier_attribute.queue_length, sizeof(frontier_attribute.queue_length));

                      kernel_parameters[14] = new char[sizeof(output_queue_len)];
                      std::memcpy(kernel_parameters[14], &output_queue_len, sizeof(output_queue_len));

                      kernel_parameters[15] = new char[sizeof(split_val)];
                      std::memcpy(kernel_parameters[15], &split_val, sizeof(split_val));

                      kernel_parameters[16] = new char[sizeof(max_in)];
                      std::memcpy(kernel_parameters[16], &max_in, sizeof(max_in));

                      kernel_parameters[17] = new char[sizeof(max_out)];
                      std::memcpy(kernel_parameters[17], &max_out, sizeof(max_out));

                      kernel_parameters[18] = new char[sizeof(work_progress)];
                      std::memcpy(kernel_parameters[18], &work_progress, sizeof(work_progress));

                      kernel_parameters[19] = new char[sizeof(enactor_stats.advance_kernel_stats)];
                      std::memcpy(kernel_parameters[19], &enactor_stats.advance_kernel_stats, sizeof(enactor_stats.advance_kernel_stats));

                      int temp_val2 = ADVANCE_TYPE;
                      kernel_parameters[20] = new char[sizeof(int)];
                      std::memcpy(kernel_parameters[20], &temp_val2, sizeof(int));

                      kernel_parameters[21] = new char[sizeof(inverse_graph)];
                      std::memcpy(kernel_parameters[21], &inverse_graph, sizeof(inverse_graph));

                      int temp_val3 = R_TYPE;
                      kernel_parameters[22] = new char[sizeof(int)];
                      std::memcpy(kernel_parameters[22], &temp_val3, sizeof(int));

                      int temp_val4 = R_OP;
                      kernel_parameters[23] = new char[sizeof(int)];
                      std::memcpy(kernel_parameters[23], &temp_val4, sizeof(int));

                      kernel_parameters[24] = new char[sizeof(d_value_to_reduce)];
                      std::memcpy(kernel_parameters[24], &d_value_to_reduce, sizeof(d_value_to_reduce));

                      kernel_parameters[25] = new char[sizeof(d_reduce_frontier)];
                      std::memcpy(kernel_parameters[25], &d_reduce_frontier, sizeof(d_reduce_frontier));

                   //}
                   
		      cudaDeviceSynchronize();
		      gettimeofday(&start, NULL);
                   code = cuLaunchKernel(func, 384, 52, 1, 256, 1, 1, 1024, 0, kernel_parameters, NULL);
                   if(code == CUDA_SUCCESS)
                   {
                       std::cout << "cuLaunchKernel returned CUDA_SUCCESS" << std::endl;
                   }
                   else
                   {
                       std::cout << "cuLaunchKernel returned " << code << std::endl;
                   }
                   cudaDeviceSynchronize();
		   gettimeofday(&end, NULL);
		   std::cout << (end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec) << std::endl;
                   std::exit(-1);
                }
#endif
                /*
                cudaDeviceSynchronize();
                gettimeofday(&start, NULL);
                gunrock::oprtr::edge_map_partitioned::RelaxPartitionedEdges2<typename KernelPolicy::LOAD_BALANCED, ProblemData, Functor>
                <<< num_block, KernelPolicy::LOAD_BALANCED::THREADS >>>(
                                        frontier_attribute.queue_reset,
                                        frontier_attribute.queue_index,
                                        enactor_stats.iteration,
                                        d_row_offsets,
                                        d_column_indices,
                                        d_row_indices,
                                        &partitioned_scanned_edges[1],
                                        enactor_stats.d_node_locks_out,
                                        KernelPolicy::LOAD_BALANCED::BLOCKS,
                                        d_done,
                                        d_in_key_queue,
                                        d_out_key_queue,
                                        data_slice,
                                        frontier_attribute.queue_length,
                                        output_queue_len,
                                        split_val,
                                        max_in,
                                        max_out,
                                        work_progress,
                                        enactor_stats.advance_kernel_stats,
                                        ADVANCE_TYPE,
                                        inverse_graph,
                                        R_TYPE,
                                        R_OP,
                                        d_value_to_reduce,
                                        d_reduce_frontier);
		cudaDeviceSynchronize();
		gettimeofday(&end, NULL);
		std::cout << (end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec) << std::endl;
                */
                
                unsigned long grid[3], block[3];
                grid[0] = num_block; grid[1] = 1; grid[2] = 1;
                block[0] = KernelPolicy::LOAD_BALANCED::THREADS; block[1] = 1; block[2] = 1;
                KernelIdentifier kid("RelaxPartitionedEdges2", grid, block);
                unsigned long have_run_for=0;
                if(allocate == 0)
                {
                cudaMalloc(&d_yield_point_ret, sizeof(int));
                cudaMalloc(&d_elapsed_ret, sizeof(int));
                allocate = 1;
                }
                long service_id = -1, service_id_dummy = -1;
                h_yield_point = 0;
                h_elapsed = 0;
		bool yield_global = true, yield_global_select = false, yield_local = false, yield_local_select = false;
                if(/*iteration_ctr == 11*/true)
                {
                    yield_global_select = true;
                    yield_local_select = true;
                }
		cudaMemcpy(&d_yield_point_persist, &h_yield_point, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(&d_yield_point, &h_yield_point, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(&d_elapsed, &h_elapsed, sizeof(int), cudaMemcpyHostToDevice);
		while(h_yield_point < grid[0]*grid[1]-1)
		{
			if(yield_global)
			{
				if(h_yield_point == 0)
				{
					service_id = EvqueueLaunch(kid, have_run_for, service_id_dummy);
					std::cout << "New " << h_yield_point << " " << service_id << std::endl;
					assert(service_id != -1);
				}
				else
				{
					service_id_dummy = EvqueueLaunch(kid, have_run_for, service_id);
					std::cout << "In service " << h_yield_point << " " << have_run_for << " " << service_id << std::endl;
					assert(service_id_dummy == -1);
				}
				assert(service_id != -1);
			}
                        if(yield_local && yield_local_select)
                        {
                                //std::cout << "Local " << iteration_ctr << " " << yield_local << " " << yield_local_select << std::endl;
				struct timeval start, end;
				cudaMemcpy(d_yield_point_ret, &h_yield_point, sizeof(int), cudaMemcpyHostToDevice);
				cudaMemcpy(d_elapsed_ret, &h_elapsed, sizeof(int), cudaMemcpyHostToDevice);
				//cudaDeviceSynchronize();
				gettimeofday(&start, NULL);
				unsigned int allotted_slice=128000000; /*1000000000;*/
				gunrock::oprtr::edge_map_partitioned::RelaxPartitionedEdges2instrumented<typename KernelPolicy::LOAD_BALANCED, ProblemData, Functor>
					<<< num_block, KernelPolicy::LOAD_BALANCED::THREADS >>>(
							frontier_attribute.queue_reset,
							frontier_attribute.queue_index,
							enactor_stats.iteration,
							d_row_offsets,
							d_column_indices,
							d_row_indices,
							&partitioned_scanned_edges[1],
							enactor_stats.d_node_locks_out,
							KernelPolicy::LOAD_BALANCED::BLOCKS,
							d_done,
							d_in_key_queue,
							d_out_key_queue,
							data_slice,
							frontier_attribute.queue_length,
							output_queue_len,
							split_val,
							max_in,
							max_out,
							work_progress,
							enactor_stats.advance_kernel_stats,
							ADVANCE_TYPE,
							inverse_graph,
							R_TYPE,
							R_OP,
							d_value_to_reduce,
							d_reduce_frontier, 
							allotted_slice,
							d_yield_point_ret, 
							d_elapsed_ret);
				//cudaDeviceSynchronize();
				cudaMemcpy(&h_yield_point, d_yield_point_ret, sizeof(int), cudaMemcpyDeviceToHost);
				gettimeofday(&end, NULL);
				cudaMemcpy(&h_elapsed, d_elapsed_ret, sizeof(int), cudaMemcpyDeviceToHost);
                                assert(h_elapsed != -2);
			        //std::cout << "advance " << iteration_ctr << " " << h_yield_point << " " << h_elapsed << " " << num_block << " " << (end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec) << std::endl;
                                h_elapsed = -2;

                        }
			else if(yield_global && yield_global_select && service_id != 10000000) /*yield needed*/
			{
                                //std::cout << "Global " << iteration_ctr << " " << yield_local << " " << yield_local_select << std::endl;
				unsigned int allotted_slice=1000000; /*1000000000;*/
				gunrock::oprtr::edge_map_partitioned::RelaxPartitionedEdges2instrumented<typename KernelPolicy::LOAD_BALANCED, ProblemData, Functor>
					<<< num_block, KernelPolicy::LOAD_BALANCED::THREADS >>>(
							frontier_attribute.queue_reset,
							frontier_attribute.queue_index,
							enactor_stats.iteration,
							d_row_offsets,
							d_column_indices,
							d_row_indices,
							&partitioned_scanned_edges[1],
							enactor_stats.d_node_locks_out,
							KernelPolicy::LOAD_BALANCED::BLOCKS,
							d_done,
							d_in_key_queue,
							d_out_key_queue,
							data_slice,
							frontier_attribute.queue_length,
							output_queue_len,
							split_val,
							max_in,
							max_out,
							work_progress,
							enactor_stats.advance_kernel_stats,
							ADVANCE_TYPE,
							inverse_graph,
							R_TYPE,
							R_OP,
							d_value_to_reduce,
							d_reduce_frontier, 
							allotted_slice,
							d_yield_point_ret, 
							d_elapsed_ret);
				cudaMemcpy(&h_yield_point, d_yield_point_ret, sizeof(int), cudaMemcpyDeviceToHost);
				cudaMemcpy(&h_elapsed, d_elapsed_ret, sizeof(int), cudaMemcpyDeviceToHost);
				have_run_for+=h_elapsed;
			        std::cout << "g advance " << iteration_ctr << " " << h_yield_point << " " << h_elapsed << " " << num_block << " " << (end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec) << std::endl;
			}
			else /*first run of this kernel, so don't yield*/
			{
                                //std::cout << iteration_ctr << " " << yield_local << " " << yield_local_select << std::endl;
                                //struct timeval start, end;
                                //cudaDeviceSynchronize();
                                //gettimeofday(&start, NULL);
				gunrock::oprtr::edge_map_partitioned::RelaxPartitionedEdges2<typename KernelPolicy::LOAD_BALANCED, ProblemData, Functor>
					<<< num_block, KernelPolicy::LOAD_BALANCED::THREADS >>>(
							frontier_attribute.queue_reset,
							frontier_attribute.queue_index,
							enactor_stats.iteration,
							d_row_offsets,
							d_column_indices,
							d_row_indices,
							&partitioned_scanned_edges[1],
							enactor_stats.d_node_locks_out,
							KernelPolicy::LOAD_BALANCED::BLOCKS,
							d_done,
							d_in_key_queue,
							d_out_key_queue,
							data_slice,
							frontier_attribute.queue_length,
							output_queue_len,
							split_val,
							max_in,
							max_out,
							work_progress,
							enactor_stats.advance_kernel_stats,
							ADVANCE_TYPE,
							inverse_graph,
							R_TYPE,
							R_OP,
							d_value_to_reduce,
							d_reduce_frontier); 
                                //cudaDeviceSynchronize();
                                //gettimeofday(&end, NULL);
				//std::cout << (end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec) << std::endl;
				break;
			}
			//std::cout << (end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec) << std::endl;
			//std::cout << h_yield_point << " " << h_elapsed << std::endl;
                   //std::cout << h_yield_point << " " << grid[0] << " " << grid[1] << " " << grid[0]*grid[1] - 1 << " " << (h_yield_point < grid[0]*grid[1] - 1) << std::endl;
		}
                h_yield_point = 0;

                //util::DisplayDeviceResults(d_out_key_queue, output_queue_len);
            }

            // TODO: switch REDUCE_OP for different reduce operators
            // Do segreduction using d_scanned_edges and d_reduce_frontier
            if (R_TYPE != gunrock::oprtr::advance::EMPTY && d_value_to_reduce && d_reduce_frontier) {
              switch (R_OP) {
                case gunrock::oprtr::advance::PLUS: {
                    SegReduceCsr(d_reduce_frontier, partitioned_scanned_edges, output_queue_len,frontier_attribute.queue_length,
                      false, d_reduced_value, (Value)0, mgpu::plus<typename KernelPolicy::Value>(), context);
                      break;
                }
                case gunrock::oprtr::advance::MULTIPLIES: {
                    SegReduceCsr(d_reduce_frontier, partitioned_scanned_edges, output_queue_len,frontier_attribute.queue_length,
                      false, d_reduced_value, (Value)1, mgpu::multiplies<typename KernelPolicy::Value>(), context);
                      break;
                }
                case gunrock::oprtr::advance::MAXIMUM: {
                    SegReduceCsr(d_reduce_frontier, partitioned_scanned_edges, output_queue_len,frontier_attribute.queue_length,
                      false, d_reduced_value, (Value)INT_MIN, mgpu::maximum<typename KernelPolicy::Value>(), context);
                      break;
                }
                case gunrock::oprtr::advance::MINIMUM: {
                    SegReduceCsr(d_reduce_frontier, partitioned_scanned_edges, output_queue_len,frontier_attribute.queue_length,
                      false, d_reduced_value, (Value)INT_MAX, mgpu::minimum<typename KernelPolicy::Value>(), context);
                      break;
                }
                default:
                    //default operator is plus
                    SegReduceCsr(d_reduce_frontier, partitioned_scanned_edges, output_queue_len,frontier_attribute.queue_length,
                      false, d_reduced_value, (Value)0, mgpu::plus<typename KernelPolicy::Value>(), context);
                      break;
              }
            }
            break;
        }
    }
}


} //advance
} //oprtr
} //gunrock/
