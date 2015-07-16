// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * test_topk.cu
 *
 * @brief Simple test driver program for computing Topk.
 */

#include <stdio.h>
#include <string>
#include <deque>
#include <vector>
#include <utility>
#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <fstream>
#include <map>

#include "EvqueueManager.h"

// Utilities and correctness-checking
#include <gunrock/util/test_utils.cuh>

// Graph construction utils
#include <gunrock/graphio/market.cuh>

// Degree Centrality includes
#include <gunrock/app/topk/topk_enactor.cuh>
#include <gunrock/app/topk/topk_problem.cuh>

// Operator includes
#include <gunrock/oprtr/filter/kernel.cuh>

using namespace gunrock;
using namespace gunrock::util;
using namespace gunrock::oprtr;
using namespace gunrock::app::topk;

/******************************************************************************
 * Defines, constants, globals
 ******************************************************************************/
bool g_verbose;
bool g_undirected;
bool g_quick;
bool g_stream_from_host;

/******************************************************************************
 * Housekeeping Routines
 ******************************************************************************/
void Usage()
{
  printf(
    "\ntest_topk <graph type> <graph type args> [--top=<K_value>] [--device=<device_index>] "
    "[--instrumented] [--quick] "
    "[--v]\n"
    "\n"
    "Graph types and args:\n"
    "  market [<file>]\n"
    "    Reads a Matrix-Market coordinate-formatted graph of directed/undirected\n"
    "    edges from stdin (or from the optionally-specified file).\n"
    "    k value top K value.\n"
    "  --device=<device_index>  Set GPU device for running the graph primitive.\n"
    "  --instrumented If set then kernels keep track of queue-search_depth\n"
    "  and barrier duty (a relative indicator of load imbalance.)\n"
    "  --quick If set will skip the CPU validation code.\n");
}

/**
 * @brief displays the top K results
 *
 */
template<
  typename VertexId,
  typename Value,
  typename SizeT>
void DisplaySolution(
  VertexId *h_node_id,
  Value    *h_degrees_i,
  Value    *h_degrees_o,
  SizeT    num_nodes)
{
  fflush(stdout);
  // at most display the first 100 results
  if (num_nodes > 100) num_nodes = 100;
  printf("==> top %d centrality nodes:\n", num_nodes);
  for (SizeT iter = 0; iter < num_nodes; ++iter)
    printf("%d %d %d\n", h_node_id[iter], h_degrees_i[iter], h_degrees_o[iter]);
}

/******************************************************************************
 * Degree Centrality Testing Routines
 *****************************************************************************/
/**
 * @brief A simple CPU-based reference TOPK implementation.
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 *
 * @param[in] graph Reference to the CSR graph we process on
 */
struct compare_second_only
{
  template <typename T1, typename T2>
  bool operator()(const std::pair<T1, T2>& p1, const std::pair<T1, T2>& p2)
  {
    return p1.second > p2. second;
  }
};

template<
  typename VertexId,
  typename Value,
  typename SizeT>
void SimpleReferenceTopK(
  const Csr<VertexId, Value, SizeT> &graph_original,
  const Csr<VertexId, Value, SizeT> &graph_reversed,
  VertexId *ref_node_id,
  Value    *ref_degrees,
  SizeT    top_nodes)
{
  printf("CPU reference test.\n");
  CpuTimer cpu_timer;

  // malloc degree centrality spaces
  Value *ref_degrees_original =
    (Value*)malloc(sizeof(Value) * graph_original.nodes);
  Value *ref_degrees_reversed =
    (Value*)malloc(sizeof(Value) * graph_reversed.nodes);

  // store reference output results
  std::vector< pair<int, int> > results;

  // calculations
  for (SizeT node = 0; node < graph_original.nodes; ++node)
  {
    ref_degrees_original[node] =
      graph_original.row_offsets[node+1] - graph_original.row_offsets[node];
    ref_degrees_reversed[node] =
      graph_reversed.row_offsets[node+1] - graph_reversed.row_offsets[node];
  }

  cpu_timer.Start();

  // add ingoing degrees and outgoing degrees together
  for (SizeT node = 0; node < graph_original.nodes; ++node)
  {
    ref_degrees_original[node] =
      ref_degrees_original[node] + ref_degrees_reversed[node];
    results.push_back( std::make_pair (node, ref_degrees_original[node]) );
  }

  // pair sort according to second elements - degree centrality
  std::stable_sort(results.begin(), results.end(), compare_second_only());

  for (SizeT itr = 0; itr < top_nodes; ++itr)
  {
    ref_node_id[itr] = results[itr].first;
    ref_degrees[itr] = results[itr].second;
  }

  cpu_timer.Stop();
  float elapsed_cpu = cpu_timer.ElapsedMillis();
  printf("==> CPU Degree Centrality finished in %lf msec.\n", elapsed_cpu);

  // clean up if neccessary
  if (ref_degrees_original) { free(ref_degrees_original); }
  if (ref_degrees_reversed) { free(ref_degrees_reversed); }
  results.clear();
}

/**
 * @brief Run TopK tests
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 * @tparam INSTRUMENT
 *
 * @param[in] graph_original Reference to the CSR graph we process on
 * @param[in] graph_reversed Reference to the inversed CSR graph we process on
 * @param[in] args Reference to the command line arguments
 * @param[in] max_grid_size Maximum CTA occupancy
 * @param[in] num_gpus Number of GPUs
 * @param[in] top_nodes Number of nodes to process for Top-K algorithm
 * @param[in] context CudaContext for moderngpu library
 *
 */
template <
  typename VertexId,
  typename Value,
  typename SizeT,
  bool INSTRUMENT>
void RunTests(
  const Csr<VertexId, Value, SizeT> &graph_original,
  const Csr<VertexId, Value, SizeT> &graph_reversed,
  CommandLineArgs                   &args,
  int                               max_grid_size,
  int                               num_gpus,
  int                               top_nodes,
  CudaContext                       &context)
{

  // define the problem data structure for graph primitive
  typedef TOPKProblem<VertexId, SizeT, Value> Problem;

  // INSTRUMENT specifies whether we want to keep such statistical data
  // Allocate TopK enactor map
  TOPKEnactor<INSTRUMENT> topk_enactor(g_verbose);

  // allocate problem on GPU
  // create a pointer of the TOPKProblem type
  Problem *topk_problem = new Problem;

  // reset top_nodes if input k > total number of nodes
  if (top_nodes > graph_original.nodes)
  {
    top_nodes = graph_original.nodes;
  }

  // malloc host memory
  VertexId *h_node_id   = (VertexId*)malloc(sizeof(VertexId) * top_nodes);
  VertexId *ref_node_id = (VertexId*)malloc(sizeof(VertexId) * top_nodes);
  Value    *h_degrees_i = (Value*)malloc(sizeof(Value) * top_nodes);
  Value    *h_degrees_o = (Value*)malloc(sizeof(Value) * top_nodes);
  Value    *ref_degrees = (Value*)malloc(sizeof(Value) * top_nodes);

  // copy data from CPU to GPU
  // initialize data members in DataSlice for graph
  util::GRError(topk_problem->Init(
    g_stream_from_host,
    graph_original,
    graph_reversed,
    num_gpus),
    "Problem TOPK Initialization Failed", __FILE__, __LINE__);

  // perform topk degree centrality calculations
  GpuTimer gpu_timer; // Record the kernel running time

    struct timeval start, end;
    for (int iter = 0; iter < 1000; ++iter)
    {
        std::cout << "Iteration " << iter << std::endl;
  // reset values in DataSlice for graph
  util::GRError(topk_problem->Reset(
    topk_enactor.GetFrontierType()),
    "TOPK Problem Data Reset Failed", __FILE__, __LINE__);

  gpu_timer.Start();

        gettimeofday(&start, NULL);
  // launch topk enactor
  util::GRError(topk_enactor.template Enact<Problem>(
    topk_problem,
    top_nodes,
    max_grid_size),
    "TOPK Problem Enact Failed", __FILE__, __LINE__);

  gpu_timer.Stop();
        gettimeofday(&end, NULL);
        std::cerr << "[TOPK] ---- " << (end.tv_sec - start.tv_sec)*1000000+(end.tv_usec - start.tv_usec) << std::endl;
	EvqueueSynch();
    }
  float elapsed_gpu = gpu_timer.ElapsedMillis();
  printf("==> GPU TopK Degree Centrality finished in %lf msec.\n", elapsed_gpu);

  // copy out results back to CPU from GPU using Extract
  util::GRError(topk_problem->Extract(
    h_node_id,
    h_degrees_i,
    h_degrees_o,
    top_nodes),
    "TOPK Problem Data Extraction Failed",
    __FILE__, __LINE__);

  // display solution
  DisplaySolution(
    h_node_id,
    h_degrees_i,
    h_degrees_o,
    top_nodes);

  // validation
  SimpleReferenceTopK(
    graph_original,
    graph_reversed,
    ref_node_id,
    ref_degrees,
    top_nodes);

  int error_num = CompareResults(h_node_id, ref_node_id, top_nodes, true);
  if (error_num > 0)
  {
    printf("INCOREECT! %d error(s) occured. \n", error_num);
  }
  printf("\n");

  // cleanup if neccessary
  if (topk_problem) { delete topk_problem; }
  if (h_node_id)    { free(h_node_id);   }
  if (h_degrees_i)  { free(h_degrees_i); }
  if (h_degrees_o)  { free(h_degrees_o); }

  cudaDeviceSynchronize();
}

/**
 * @brief RunTests entry
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 *
 * @param[in] graph_original Reference to the CSR graph we process on
 * @param[in] graph_reversed Reference to the inversed CSR graph we process on
 * @param[in] args Reference to the command line arguments
 * @param[in] top_nodes Number of nodes to process for Top-K algorithm
 * @param[in] context CudaContext for moderngpu library
 */
template <
  typename VertexId,
  typename Value,
  typename SizeT>
void RunTests(
  Csr<VertexId, Value, SizeT> &graph_original,
  Csr<VertexId, Value, SizeT> &graph_reversed,
  CommandLineArgs             &args,
  SizeT                       top_nodes,
  CudaContext                 &context)
{
  bool instrumented  = false;
  int  max_grid_size = 0;
  int  num_gpus      = 1;

  instrumented = args.CheckCmdLineFlag("instrumented");

  g_quick = args.CheckCmdLineFlag("quick");
  g_verbose = args.CheckCmdLineFlag("v");

  if (instrumented)
  {
    RunTests<VertexId, Value, SizeT, true>(
      graph_original,
      graph_reversed,
      args,
      max_grid_size,
      num_gpus,
      top_nodes,
      context);
  }
  else
  {
    RunTests<VertexId, Value, SizeT, false>(
      graph_original,
      graph_reversed,
      args,
      max_grid_size,
      num_gpus,
      top_nodes,
      context);
  }
}

/******************************************************************************
 * Main
 ******************************************************************************/
int main(int argc, char** argv)
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
  int top_nodes;

  args.GetCmdLineArgument("device", dev);
  args.GetCmdLineArgument("top", top_nodes);

  mgpu::ContextPtr context = mgpu::CreateCudaDevice(dev);
  //srand(0); // Presently deterministic
  //srand(time(NULL));

  // Parse graph-contruction params
  g_undirected = false;

  std::string graph_type = argv[1];
  int flags = args.ParsedArgc();
  int graph_args = argc - flags - 1;

  if (graph_args < 1)
  {
    Usage();
    return 1;
  }

  //
  // Construct graph and perform
  //
  if (graph_type == "market")
  {

    // Matrix-market coordinate-formatted graph file

    typedef int VertexId; //!< Use as the node identifier type
    typedef int Value;    //!< Use as the value type
    typedef int SizeT;    //!< Use as the graph size type

    Csr<VertexId, Value, SizeT> csr_original(false);
    Csr<VertexId, Value, SizeT> csr_reversed(false);

    // Default value for stream_from_host is false
    if (graph_args < 1)
    {
      Usage();
      return 1;
    }

    char *market_filename = (graph_args == 2) ? argv[2] : NULL;

    // BuildMarketGraph() reads a mtx file into CSR data structure
    // Template argumet = true because the graph has edge weights
    // read in non-inversed graph
    if (graphio::BuildMarketGraph<true>(
      market_filename,
      csr_original,
      g_undirected,
      false) != 0) // original graph
    {
      return 1;
    }

    // read in inversed graph
    if (graphio::BuildMarketGraph<true>(
      market_filename,
      csr_reversed,
      g_undirected,
      true) != 0) // reversed graph
    {
      return 1;
    }

    //csr_original.DisplayGraph();
    //csr_reversed.DisplayGraph();

    // run gpu tests
    RunTests(csr_original, csr_reversed, args, top_nodes, *context);

  }
  else
  {
    // unknown graph type
    fprintf(stderr, "Unspecified graph type\n");
    return 1;
  }

  EvqueueDestroy();
  return 0;
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
