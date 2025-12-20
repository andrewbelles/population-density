/*
 * graph.hpp  Andrew Belles  Dec 19th, 2025  
 *  
 * API for Graph Backend leveraged by Graph Neural Network Model   
 *    
 *     
 */    
 
#pragma once 

#include <cstdint> 
#include <vector> 
#include <utility>

namespace topo::graph {

/************ bindings ************************************/ 
using node_index_t = std::uint32_t; 
using edge_index_t = std::uint32_t; 
using weight_t     = double; 

/************ Edge ****************************************/ 
/* Simple weighted Edge structure for use in Graph  
 */
struct Edge {
  node_index_t src{0};
  node_index_t dst{0}; 
  weight_t weight{1.0}; 
  
  // Lexigraphical ordering 
  bool operator<(const Edge& b) {
    if ( src != b.src ) {
      return src < b.src; 
    } else if ( dst != b.dst ) {
      return dst < b.dst; 
    } 
    return weight < b.weight;
  }

}; 

enum class DuplicateEdgePolicy : std::uint8_t {
  ERROR, 
  KEEP_FIRST, 
  KEEP_LAST, 
  SUM, 
  MEAN, 
  MIN, 
  MAX 
}; 

struct BuildOptions {
  bool directed{true}; 
  bool sort_by_dst{true}; 
  bool add_reverse_edges{false}; 
  bool allow_self_loops{true}; 
  DuplicateEdgePolicy dedup{DuplicateEdgePolicy::SUM}; 
}; 


class Graph {
public: 
  Graph() = default; 

  Graph(node_index_t num_nodes,
        std::vector<edge_index_t> index_ptr, // prefix sum for node u's neighbors idx 
        std::vector<node_index_t> indices, 
        std::vector<weight_t> weights,
        bool directed); 

  node_index_t num_nodes() const noexcept { return num_nodes_; }
  edge_index_t num_edges() const noexcept { return num_edges_; }
  bool directed() const noexcept { return directed_; }

  const std::vector<edge_index_t>& index_ptr() const noexcept { return index_ptr_; }
  const std::vector<node_index_t>& indices() const noexcept { return indices_; }

  edge_index_t neighbor_begin(node_index_t u) const; 
  edge_index_t neighbor_end(node_index_t u) const; 

  std::pair<std::vector<std::pair<node_index_t, node_index_t>>, std::vector<weight_t>>
  to_coo() const; 

  std::vector<node_index_t> sources() const; 

  void validate(void) const; 

private: 
  node_index_t num_nodes_{0}; 
  edge_index_t num_edges_{0}; 
  bool directed_{true}; 
  std::vector<edge_index_t> index_ptr_{}; 
  std::vector<node_index_t> indices_{}; 
  std::vector<weight_t> weights_{}; 
}; 


class GraphBuilder {
public: 
  explicit GraphBuilder(node_index_t num_nodeds, BuildOptions options = {}); 

  node_index_t num_nodes() const noexcept { return num_nodes_; }
  const BuildOptions& options() const noexcept { return options_; }

  void reserve_edges(std::size_t n); 

  void add_edge(const Edge& edge); 
  void add_edge(node_index_t src, node_index_t dst, weight_t w = 1.0); 
  void add_edges(const std::vector<Edge>& edges); 

  Graph build(void); 

  void clear(void) noexcept; 

private: 
  node_index_t num_nodes_{0}; 
  BuildOptions options_; 

  std::vector<Edge> edges_{}; 

  void can_add_edge(node_index_t src, node_index_t dst, weight_t w); 

  static void apply_reverse_edges(std::vector<Edge>& edges, const BuildOptions& opt); 
  static void filter_self_loops(std::vector<Edge>& edges, const BuildOptions& opt); 
  static void sort_edges(std::vector<Edge>& edges, bool sort_by_dst); 
  static void deduplicate_edges(std::vector<Edge>&, DuplicateEdgePolicy policy); 
};

} // namespace topo::graph 
