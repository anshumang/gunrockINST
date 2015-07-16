// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * test_bfs.cu
 *
 * @brief Simple test driver program for breadth-first search.
 */

#include <stdio.h>
#include <string>
#include <deque>
#include <vector>
#include <iostream>

#include "EvqueueManager.h"

// Utilities and correctness-checking
#include <gunrock/util/test_utils.cuh>

// Graph construction utils
#include <gunrock/graphio/market.cuh>

// BFS includes
#include <gunrock/app/bfs/bfs_enactor.cuh>
#include <gunrock/app/bfs/bfs_problem.cuh>
#include <gunrock/app/bfs/bfs_functor.cuh>

// DOBFS includes
#include <gunrock/app/dobfs/dobfs_enactor.cuh>
#include <gunrock/app/dobfs/dobfs_problem.cuh>
#include <gunrock/app/dobfs/dobfs_functor.cuh>

// Operator includes
#include <gunrock/oprtr/advance/kernel.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>

#include <moderngpu.cuh>

using namespace gunrock;
using namespace gunrock::util;
using namespace gunrock::oprtr;
using namespace gunrock::app::bfs;
using namespace gunrock::app::dobfs;


/******************************************************************************
 * Defines, constants, globals
 ******************************************************************************/

bool g_verbose;
bool g_undirected;
bool g_quick;
bool g_stream_from_host;
float g_alpha;
float g_beta;

/******************************************************************************
 * Housekeeping Routines
 ******************************************************************************/
void Usage()
{
    printf (
        " test_dobfs <graph type> <graph type args> [--device=<device_index>]\n"
        " [--src=<source_index>] [--instrumented] [--idempotence=<0|1>] [--v]\n"
        " [--undirected] [--iteration-num=<num>] [--quick=<0|1>] [--mark-pred]\n"
        " [--queue-sizing=<scale factor>]\n"
        "\n"
        "Graph types and args:\n"
        "  market <file>\n"
        "    Reads a Matrix-Market coordinate-formatted graph of directed / undirected\n"
        "    edges from stdin (or from the optionally-specified file).\n"
        "  --device=<device_index>   Set GPU device for running the test. [Default: 0].\n"
        "  --undirected              Treat the graph as undirected (symmetric).\n"
        "  --idempotence=<0 or 1>    Enable: 1, Disable: 0 [Default: Enable].\n"
        "  --instrumented            Keep kernels statics [Default: Disable].\n"
        "                            total_queued, search_depth and barrier duty\n"
        "                            (a relative indicator of load imbalance.)\n"
        "  --src=<source vertex id>  Begins BFS from the source [Default: 0].\n"
        "                            If randomize: from a random source vertex.\n"
        "                            If largestdegree: from largest degree vertex.\n"
        "  --quick=<0 or 1>          Skip the CPU validation: 1, or not: 0 [Default: 1].\n"
        "  --mark-pred               Keep both label info and predecessor info.\n"
        "  --queue-sizing=<factor>   Allocates a frontier queue sized at: \n"
        "                            (graph-edges * <scale factor>). [Default: 1.0]\n"
        "  --v                       Print verbose per iteration debug info.\n"
        "  --iteration-num=<number>  Number of runs to perform the test [Default: 1].\n"
        );
}

/**
 * @brief Displays the BFS result (i.e., distance from source)
 *
 * @param[in] source_path Search depth from the source for each node.
 * @param[in] preds Predecessor node id for each node.
 * @param[in] nodes Number of nodes in the graph.
 * @param[in] MARK_PREDECESSORS Whether to show predecessor of each node.
 * @param[in] ENABLE_IDEMPOTENCE Whether to enable idempotence mode.
 */
template<typename VertexId, typename SizeT>
void DisplaySolution (VertexId *source_path,
                      VertexId *preds,
                      SizeT nodes,
                      bool MARK_PREDECESSORS,
                      bool ENABLE_IDEMPOTENCE)
{
    if (nodes > 40) nodes = 40;
    printf("\nFirst %d labels of the GPU result.\n", nodes);
    PrintFormatArray (source_path, nodes, "%4d", 10);
    if (MARK_PREDECESSORS && !ENABLE_IDEMPOTENCE)
    {
        printf("\n First %d predecessors:\n", nodes);
        PrintFormatArray (preds, nodes, "%8d", 5);
    }
    /*
    printf("[");
    for (VertexId i = 0; i < nodes; ++i) {
        PrintValue(i);
        printf(":");
        PrintValue(source_path[i]);
        if (MARK_PREDECESSORS && !ENABLE_IDEMPOTENCE) {
            printf(",");
            PrintValue(preds[i]);
        }
        printf(" ");
    }
    printf("]\n");
    */
}

/**
 * Performance/Evaluation statistics
 */

struct Stats {
    const char *name;
    Statistic rate;
    Statistic search_depth;
    Statistic redundant_work;
    Statistic duty;

    Stats() : name(NULL), rate(), search_depth(), redundant_work(), duty() {}
    Stats(const char *name) : name(name), rate(), search_depth(), redundant_work(), duty() {}
};

/**
 * @brief Displays timing and correctness statistics
 *
 * @tparam MARK_PREDECESSORS
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 *
 * @param[in] stats Reference to the Stats object defined in RunTests
 * @param[in] src Source node where BFS starts
 * @param[in] h_labels Host-side vector stores computed labels for validation
 * @param[in] graph Reference to the CSR graph we process on
 * @param[in] elapsed Total elapsed kernel running time
 * @param[in] search_depth Maximum search depth of the BFS algorithm
 * @param[in] total_queued Total element queued in BFS kernel running process
 * @param[in] avg_duty Average duty of the BFS kernels
 */
template<
    bool MARK_PREDECESSORS,
    typename VertexId,
    typename Value,
    typename SizeT>
void DisplayStats(
    Stats               &stats,
    VertexId            src,
    VertexId            *h_labels,
    const Csr<VertexId, Value, SizeT> &graph,
    double              elapsed,
    VertexId            search_depth,
    long long           total_queued,
    double              avg_duty)
{
    // Compute nodes and edges visited
    SizeT edges_visited = 0;
    SizeT nodes_visited = 0;
    for (VertexId i = 0; i < graph.nodes; ++i) {
        if (h_labels[i] > -1) {
            ++nodes_visited;
            edges_visited += graph.row_offsets[i+1] - graph.row_offsets[i];
        }
    }

    double redundant_work = 0.0;
    if (total_queued > 0) {
        redundant_work = ((double) total_queued - edges_visited) / edges_visited;
        // measure duplicate edges put through queue
    }
    redundant_work *= 100;

    // Display test name
    printf("[%s] finished. ", stats.name);

    // Display statistics
    if (nodes_visited < 5) {
        printf("Fewer than 5 vertices visited.\n");
    } else {
        // Display the specific sample statistics
        double m_teps = (double) edges_visited / (elapsed * 1000.0);
        printf("\n elapsed: %.4f ms, rate: %.4f MiEdges/s", elapsed, m_teps);
        if (search_depth != 0) printf(", search_depth: %lld", (long long) search_depth);
        if (avg_duty != 0) {
            printf("\n avg CTA duty: %.2f%%", avg_duty * 100);
        }
        printf("\n src: %lld, nodes_visited: %lld, edges_visited: %lld",
               (long long) src, (long long) nodes_visited, (long long) edges_visited);
        if (total_queued > 0) {
            printf(", total queued: %lld", total_queued);
        }
        if (redundant_work > 0) {
            printf(", redundant work: %.2f%%", redundant_work);
        }
        printf("\n");
    }

}




/******************************************************************************
 * BFS Testing Routines
 *****************************************************************************/

/**
 * @brief A simple CPU-based reference BFS ranking implementation.
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 *
 * @param[in] graph Reference to the CSR graph we process on
 * @param[in] source_path Host-side vector to store CPU computed labels for each node
 * @param[in] src Source node where BFS starts
 */
template<
    typename VertexId,
    typename Value,
    typename SizeT>
void SimpleReferenceBfs(
    const Csr<VertexId, Value, SizeT>       &graph,
    VertexId                                *source_path,
    VertexId                                src)
{
    // Initialize distances
    for (VertexId i = 0; i < graph.nodes; ++i)
    {
        source_path[i] = -1;
    }
    source_path[src] = 0;
    VertexId search_depth = 0;

    // Initialize queue for managing previously-discovered nodes
    std::deque<VertexId> frontier;
    frontier.push_back(src);

    //
    //Perform BFS
    //

    CpuTimer cpu_timer;
    cpu_timer.Start();
    while (!frontier.empty())
    {

        // Dequeue node from frontier
        VertexId dequeued_node = frontier.front();
        frontier.pop_front();
        VertexId neighbor_dist = source_path[dequeued_node] + 1;

        // Locate adjacency list
        int edges_begin = graph.row_offsets[dequeued_node];
        int edges_end = graph.row_offsets[dequeued_node + 1];

        for (int edge = edges_begin; edge < edges_end; ++edge)
        {
            //Lookup neighbor and enqueue if undiscovered
            VertexId neighbor = graph.column_indices[edge];
            if (source_path[neighbor] == -1)
            {
                source_path[neighbor] = neighbor_dist;
                if (search_depth < neighbor_dist)
                {
                    search_depth = neighbor_dist;
                }
                frontier.push_back(neighbor);
            }
        }
    }

    cpu_timer.Stop();
    float elapsed = cpu_timer.ElapsedMillis();
    search_depth++;

    printf("CPU BFS finished in %lf msec. Search depth is: %d\n",
           elapsed, search_depth);
}

/**
 * @brief Run BFS tests
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 * @tparam INSTRUMENT
 * @tparam MARK_PREDECESSORS
 * @tparam ENABLE_IDEMPOTENCE
 *
 * @param[in] graph Reference to the CSR graph we process on
 * @param[in] inv_graph Reference to the inverse CSC graph we process on
 * @param[in] src Source node where BFS starts
 * @param[in] max_grid_size Maximum CTA occupancy
 * @param[in] num_gpus Number of GPUs
 * @param[in] max_queue_sizing Scaling factor used in edge mapping
 * @param[in] alpha Tuning parameter for switching to reverse bfs
 * @param[in] beta Tuning parameter for switching back to normal bfs
 * @param[in] iterations Number of iterations for running the test
 * @param[in] context CudaContext pointer for moderngpu APIs
 */
template <
    typename VertexId,
    typename Value,
    typename SizeT,
    bool INSTRUMENT,
    bool MARK_PREDECESSORS,
    bool ENABLE_IDEMPOTENCE>
void RunTests(
    const Csr<VertexId, Value, SizeT> &graph,
    const Csr<VertexId, Value, SizeT> &inv_graph,
    VertexId src,
    int max_grid_size,
    int num_gpus,
    double max_queue_sizing,
    float alpha,        // Tuning parameter for switching to reverse bfs
    float beta,         // Tuning parameter for switching back to normal bfs
    int iterations,
    CudaContext& context)
{
    typedef DOBFSProblem<
        VertexId,
        SizeT,
        Value,
        MARK_PREDECESSORS,
        ENABLE_IDEMPOTENCE,
        (MARK_PREDECESSORS && ENABLE_IDEMPOTENCE)> Problem; // does not use double buffer

    // Allocate host-side label array (for both reference and gpu-computed results)
    VertexId    *reference_labels       = (VertexId*)malloc(sizeof(VertexId) * graph.nodes);
    VertexId    *h_labels               = (VertexId*)malloc(sizeof(VertexId) * graph.nodes);
    VertexId    *reference_check        = (g_quick) ? NULL : reference_labels;
    VertexId    *h_preds                = NULL;
    if (MARK_PREDECESSORS)
    {
        h_preds = (VertexId*)malloc(sizeof(VertexId) * graph.nodes);
    }


    // Allocate BFS enactor map
    DOBFSEnactor<INSTRUMENT> dobfs_enactor(g_verbose);

    // Allocate problem on GPU
    Problem *csr_problem = new Problem;

    util::GRError(csr_problem->Init(
                      g_stream_from_host,
                      g_undirected,
                      graph,
                      inv_graph,
                      num_gpus,
                      alpha,
                      beta),
                  "Problem DOBFS Initialization Failed", __FILE__, __LINE__);

    //
    // Compute reference CPU BFS solution for source-distance
    //
    if (reference_check != NULL)
    {
        printf("Computing reference value ...\n");
        SimpleReferenceBfs(
            graph,
            reference_check,
            src);
        printf("\n");
    }

    Stats *stats = new Stats("GPU DOBFS");

    long long           total_queued = 0;
    VertexId            search_depth = 0;
    double              avg_duty = 0.0;

    // Perform BFS
    GpuTimer gpu_timer;

    float elapsed = 0.0f;
    iterations = 100;
    struct timeval start, end;
    for (int iter=0; iter < iterations; ++iter)
    {
        std::cout << "Iteration " << iter << std::endl;
        util::GRError(
            csr_problem->Reset(
                src, dobfs_enactor.GetFrontierType(), max_queue_sizing),
            "DOBFS Problem Data Reset Failed", __FILE__, __LINE__);
        gpu_timer.Start();
        util::GRError(
            dobfs_enactor.template Enact<Problem>(
                context, csr_problem, src, max_grid_size),
            "DOBFS Problem Enact Failed", __FILE__, __LINE__);
        gpu_timer.Stop();
        gettimeofday(&end, NULL);
        std::cerr << "[HITS] ---- " << (end.tv_sec - start.tv_sec)*1000000+(end.tv_usec - start.tv_usec) << std::endl;
        elapsed += gpu_timer.ElapsedMillis();
	EvqueueSynch();
    }
    elapsed /= iterations;

    dobfs_enactor.GetStatistics(total_queued, search_depth, avg_duty);

    // Copy out results
    util::GRError(
        csr_problem->Extract(h_labels, h_preds),
        "DOBFS Problem Data Extraction Failed", __FILE__, __LINE__);

    // Verify the result
    if (reference_check != NULL)
    {
        if (!MARK_PREDECESSORS)
        {
            printf("Validity: ");
            CompareResults(h_labels, reference_check, graph.nodes, true);
        }
    }

    // Display Solution
    DisplaySolution(
        h_labels, h_preds, graph.nodes, MARK_PREDECESSORS, ENABLE_IDEMPOTENCE);

    DisplayStats<MARK_PREDECESSORS>(
        *stats,
        src,
        h_labels,
        graph,
        elapsed,
        search_depth,
        total_queued,
        avg_duty);

    // Cleanup
    delete stats;
    if (csr_problem) delete csr_problem;
    if (reference_labels) free(reference_labels);
    if (h_labels) free(h_labels);
    if (h_preds) free(h_preds);

    cudaDeviceSynchronize();
}

/**
 * @brief RunTests entry
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 *
 * @param[in] graph Reference to the CSR graph we process on
 * @param[in] inv_graph Reference to the CSR graph we process on
 * @param[in] args Reference to the command line arguments
 * @param[in] context CudaContext pointer for moderngpu APIs
 */
template <
    typename VertexId,
    typename Value,
    typename SizeT>
void RunTests(
    Csr<VertexId, Value, SizeT> &graph,
    Csr<VertexId, Value, SizeT> &inv_graph,
    CommandLineArgs &args,
    CudaContext& context)
{
    VertexId    src              = -1;  // Use whatever the specified graph-type's default is
    std::string src_str;
    bool        instrumented     = 0;   // Whether or not to collect instrumentation from kernels
    bool        mark_pred        = 0;   // Whether or not to mark src-distance vs. parent vertices
    bool        idempotence      = 1;   // Whether or not to enable idempotence operation
    int         max_grid_size    = 0;   // maximum grid size (0: leave it up to the enactor)
    int         num_gpus         = 1;   // Number of GPUs for multi-gpu enactor to use
    double      max_queue_sizing = 1.0; // Maximum size scaling factor for work queues (e.g., 1.0 creates n and m-element vertex and edge frontiers).
    int         iterations       = 1;   // Number of runs
    g_quick                      = 1;   // Whether or not to skip ref validation

    instrumented = args.CheckCmdLineFlag("instrumented");
    args.GetCmdLineArgument("src", src_str);
    if (src_str.empty())
    {
        src = 0;
    }
    else if (src_str.compare("randomize") == 0)
    {
        src = graphio::RandomNode(graph.nodes);
    }
    else if (src_str.compare("largestdegree") == 0)
    {
        int temp;
        src = graph.GetNodeWithHighestDegree(temp);
    }
    else
    {
        args.GetCmdLineArgument("src", src);
    }

    mark_pred = args.CheckCmdLineFlag("mark-pred");
    g_verbose = args.CheckCmdLineFlag("v");

    args.GetCmdLineArgument("iteration-num", iterations);
    args.GetCmdLineArgument("grid-size", max_grid_size);
    args.GetCmdLineArgument("idempotence", idempotence);
    args.GetCmdLineArgument("queue-sizing", max_queue_sizing);
    args.GetCmdLineArgument("quick", g_quick);
    args.GetCmdLineArgument("alpha", g_alpha);
    args.GetCmdLineArgument("beta", g_beta);

    if (g_alpha == 0.0f) g_alpha = 12.0f;
    if (g_beta == 0.0f)  g_beta  = 6.0f;

    // printf("alpha: %5f, beta: %5f\n", g_alpha, g_beta);

    if (instrumented)
    {
        if (mark_pred)
        {
            if (idempotence)
            {
                RunTests<VertexId, Value, SizeT, true, true, true>(
                    graph,
                    inv_graph,
                    src,
                    max_grid_size,
                    num_gpus,
                    max_queue_sizing,
                    g_alpha,
                    g_beta,
                    iterations,
                    context);
            }
            else
            {
                RunTests<VertexId, Value, SizeT, true, true, false>(
                    graph,
                    inv_graph,
                    src,
                    max_grid_size,
                    num_gpus,
                    max_queue_sizing,
                    g_alpha,
                    g_beta,
                    iterations,
                    context);
            }
        }
        else
        {
            if (idempotence)
            {
                RunTests<VertexId, Value, SizeT, true, false, true>(
                    graph,
                    inv_graph,
                    src,
                    max_grid_size,
                    num_gpus,
                    max_queue_sizing,
                    g_alpha,
                    g_beta,
                    iterations,
                    context);
            }
            else
            {
                RunTests<VertexId, Value, SizeT, true, false, false>(
                    graph,
                    inv_graph,
                    src,
                    max_grid_size,
                    num_gpus,
                    max_queue_sizing,
                    g_alpha,
                    g_beta,
                    iterations,
                    context);
            }
        }
    }
    else
    {
        if (mark_pred)
        {
            if (idempotence)
            {
                RunTests<VertexId, Value, SizeT, false, true, true>(
                    graph,
                    inv_graph,
                    src,
                    max_grid_size,
                    num_gpus,
                    max_queue_sizing,
                    g_alpha,
                    g_beta,
                    iterations,
                    context);
            }
            else
            {
                RunTests<VertexId, Value, SizeT, false, true, false>(
                    graph,
                    inv_graph,
                    src,
                    max_grid_size,
                    num_gpus,
                    max_queue_sizing,
                    g_alpha,
                    g_beta,
                    iterations,
                    context);
            }
        }
        else
        {
            if (idempotence)
            {
                RunTests<VertexId, Value, SizeT, false, false, true>(
                    graph,
                    inv_graph,
                    src,
                    max_grid_size,
                    num_gpus,
                    max_queue_sizing,
                    g_alpha,
                    g_beta,
                    iterations,
                    context);
            }
            else
            {
                RunTests<VertexId, Value, SizeT, false, false, false>(
                    graph,
                    inv_graph,
                    src,
                    max_grid_size,
                    num_gpus,
                    max_queue_sizing,
                    g_alpha,
                    g_beta,
                    iterations,
                    context);
            }
        }
    }
}

/******************************************************************************
 * Main
 ******************************************************************************/
int main( int argc, char** argv)
{
    EvqueueCreate(2);

    CommandLineArgs args(argc, argv);

    if ((argc < 2) || (args.CheckCmdLineFlag("help")))
    {
        Usage();
        return 1;
    }

    //DeviceInit(args);
    //cudaSetDeviceFlags(cudaDeviceMapHost);

    int dev = 0;
    args.GetCmdLineArgument("device", dev);
    ContextPtr context = mgpu::CreateCudaDevice(dev);

    //srand(0); // Presently deterministic
    //srand(time(NULL));

    // Parse graph-contruction params
    g_undirected = args.CheckCmdLineFlag("undirected");

    std::string graph_type = argv[1];
    int flags = args.ParsedArgc();
    int graph_args = argc - flags - 1;

    if (graph_args < 1)
    {
        Usage();
        return 1;
    }

    //
    // Construct graph and perform search(es)
    //

    if (graph_type == "market")
    {
        // Matrix-market coordinate-formatted graph file

        typedef int VertexId;                   // Use as the node identifier
        typedef int Value;                      // Use as the value type
        typedef int SizeT;                      // Use as the graph size type
        Csr<VertexId, Value, SizeT> csr(false); // default for stream_from_host

        Csr<VertexId, Value, SizeT> inv_csr(false);

        if (graph_args < 1) { Usage(); return 1; }

        char *market_filename = (graph_args == 2) ? argv[2] : NULL;
        if (graphio::BuildMarketGraph<false>(
                market_filename,
                csr,
                g_undirected,
                false) != 0)
        {
            return 1;
        }

        if (!g_undirected)
        {
            if (graphio::BuildMarketGraph<false>(
                    market_filename,
                    inv_csr,
                    g_undirected,
                    true) != 0)
            {
                return 1;
            }
        }

        csr.PrintHistogram();

        if (!g_undirected)
        {
            // Run tests
            RunTests(csr, inv_csr, args, *context);
        }
        else
        {
            RunTests(csr, csr, args, *context);
        }

    }
    else
    {
        fprintf(stderr, "Unspecified graph type\n");
        return 1;
    }
  EvqueueDestroy();
    return 0;
}
