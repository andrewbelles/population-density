/* 
 * test_graph.cpp  Andrew Belles  Dec 21st, 2025 
 *
 * Unit Test of Generic C++ Graph Backend 
 *
 *
 */

#include <assert.h>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <tuple> 
#include <algorithm>

#include "unittest.h"
#include "graph.hpp"

using namespace topo::graph;  

using edgeTuple = std::tuple<node_index_t, node_index_t, weight_t>; 
using edgePair  = std::pair<node_index_t, node_index_t>; 

static std::vector<edgeTuple> sorted_edges(const std::vector<edgePair>& edges,
                                           const std::vector<weight_t>& weights);
static bool contains_edge(const std::vector<std::pair<node_index_t, node_index_t>>& edges,
                          node_index_t src, node_index_t dst); 


TEST(graph_builder_basic)
{
  BuildOptions opt; 
  opt.directed          = true; 
  opt.sort_by_dst       = true; 
  opt.add_reverse_edges = false;
  opt.allow_self_loops  = true; 
  opt.dedup = DuplicateEdgePolicy::SUM; 

  GraphBuilder builder(3, opt); 
  std::vector<Edge> edges_init = {
    {0, 2, 2.0},
    {0, 1, 1.0},
    {1, 2, 3.0}
  };

  builder.add_edges(edges_init);
  auto graph = builder.build(); 

  assert(graph.num_nodes() == 3); 
  assert(graph.num_edges() == graph.indices().size());

  const auto& ptr = graph.index_ptr(); 

  assert(ptr.size() == 4); // prefix sum is n + 1 
  assert(ptr[0] == 0); 
  assert(ptr[1] == 2);
  assert(ptr[2] == ptr[3] && ptr[2] == 3); 

  const auto& idx = graph.indices(); 
  assert(idx.size() == 3); 
  
  // sorted_by_dst forces node 0 neighbors [1,2]
  assert(idx[ptr[0]] == 1); 
  assert(idx[ptr[0] + 1] == 2); 
  // node 1 has a single neighbor 
  assert(idx[ptr[1]] == 2); 
  
  auto [edges, weights] = graph.to_coo(); 
  assert(edges.size() == 3); 
  assert(weights.size() == 3); 

  assert(edges[0].first == 0 && edges[0].second == 1 && weights[0] == 1.0); 
  assert(edges[1].first == 0 && edges[1].second == 2 && weights[1] == 2.0); 
  assert(edges[2].first == 1 && edges[2].second == 2 && weights[2] == 3.0); 

  // sources should be aligned with prefix sum order 
  auto sources = graph.sources(); 
  assert(sources.size() == 3); 
  assert(sources[0] == 0); 
  assert(sources[1] == 0); 
  assert(sources[2] == 1); 
}

TEST(graph_neighbor_range_bounds) 
{
  BuildOptions opt; 
  GraphBuilder builder(2, opt); 
  builder.add_edge(0, 1, 1.0); 
  auto graph = builder.build(); 

  // in range 
  (void)graph.neighbor_begin(0);
  (void)graph.neighbor_end(0); 

  bool threw = false; 
  try {
    (void)graph.neighbor_begin(graph.indices().size() + 1); 
  } catch (const std::out_of_range&) {
    threw = true; 
  }
  assert(threw); 

  threw = false; 
  try {
    (void)graph.neighbor_end(graph.indices().size() + 1); 
  } catch (const std::out_of_range&) {
    threw = true; 
  }
  assert(threw); 
}

TEST(graph_reverse_edges_undirected)
{
  BuildOptions opt; 
  opt.directed          = false; 
  opt.add_reverse_edges = false;
  opt.sort_by_dst       = true; 
  opt.dedup = DuplicateEdgePolicy::SUM; 

  GraphBuilder builder(3, opt); 
  builder.add_edges({
    {0, 1, 1.0},
    {1, 2, 2.0}
  }); 
  
  auto graph = builder.build(); 
  auto [edges, weights] = graph.to_coo(); 

  assert(contains_edge(edges, 0, 1));
  assert(contains_edge(edges, 1, 0));
  assert(contains_edge(edges, 1, 2));
  assert(contains_edge(edges, 2, 1));

  assert(!contains_edge(edges, 0, 0));
  assert(!contains_edge(edges, 1, 1));
  assert(!contains_edge(edges, 2, 2));

  auto coo = sorted_edges(edges, weights); 
  assert(coo.size() == 4); 
  assert(coo[0] == std::make_tuple((node_index_t)0, (node_index_t)1, 1.0));  
}

TEST(graph_deduplicate_sum)
{
  BuildOptions opt; 
  opt.directed          = true; 
  opt.add_reverse_edges = false;
  opt.sort_by_dst       = true; 
  opt.dedup = DuplicateEdgePolicy::SUM; 

  GraphBuilder builder(2, opt); 
  builder.add_edge(0, 1, 1.0); 
  builder.add_edge(0, 1, 2.0); 
  builder.add_edge(0, 1, 4.0); 

  auto graph = builder.build();
  auto [edges, weights] = graph.to_coo(); 

  assert(edges.size() == 1); 
  assert(weights.size() == 1); 
  assert(edges[0].first == 0 && edges[0].second == 1); 
  assert(weights[0] == 7.0); 
}

TEST(graph_validate_non_finite_weight)
{
  bool threw = false; 
  try {
    Graph g(
      2, 
      std::vector<edge_index_t>{0, 1, 1}, 
      std::vector<node_index_t>{1},
      std::vector<weight_t>{std::numeric_limits<double>::infinity()}, 
      true
    );
    (void)g; 
  } catch (const std::runtime_error&) {
    threw = true; 
  }
  assert(threw); 
}


int main(void)
{
  std::cout << "Running Graph backend tests...\n"; 

  RUN_TEST(graph_builder_basic); 
  RUN_TEST(graph_neighbor_range_bounds);
  RUN_TEST(graph_reverse_edges_undirected);
  RUN_TEST(graph_deduplicate_sum);
  RUN_TEST(graph_validate_non_finite_weight);

  std::cout << "\nAll tests passed\n"; 
  return 0; 
}


/************ sorted_edges ********************************/ 
/* Helper function to sort edges and weights 
 * returned from to_coo */  
static std::vector<edgeTuple> 
sorted_edges(const std::vector<edgePair>& edges, const std::vector<weight_t>& weights)
{
  assert(edges.size() == weights.size()); 
  std::vector<edgeTuple> out; 
  out.reserve(edges.size()); 
  
  for (size_t i = 0; i < edges.size(); i++) {
    out.emplace_back(edges[i].first, edges[i].second, weights[i]); 
  }
  std::sort(out.begin(), out.end()); 
  return out; 
}

/************ contains_edge *******************************/ 
/* Checks if a given set of edges contains a 
 * specific src, dst pair */ 
static bool 
contains_edge(const std::vector<std::pair<node_index_t, node_index_t>>& edges,
              node_index_t src, node_index_t dst)
{
  for (const auto& e : edges) {
    if ( e.first == src && e.second == dst ) {
      return true; 
    } 
  }
  return false; 
}
