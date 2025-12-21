/*
*
*
* Implementation of API for Graph Backened 
*
* Notes: Only throws errors in functions exposed by Python Bindings 
*/ 

#include "graph.hpp" 

#include <cmath> 
#include <cstddef> 
#include <limits>
#include <stdexcept>
#include <algorithm> 
#include <string>

namespace topo::graph {

/************ Static STL Wrappers *************************/ 

/************ sum_weights *********************************/ 
/* Sums weights given iterators using stl::for_Each */ 
template <typename It> 
static inline weight_t 
sum_weights(It start, It end)
{
  weight_t sum = 0.0; 
  std::for_each(start, end, [&sum](const Edge& e) {
    sum += e.weight; 
  });
  return sum; 
}

/************ min_weights *********************************/ 
/* Get minimum element from vector of Edges */ 
template <typename It> 
static inline weight_t 
min_weights(It start, It end)
{
  auto weight_cmp = [](const Edge& a, const Edge& b) -> bool {
    return a.weight < b.weight;
  };

  auto [s, d, w] = *std::min_element(start, end, weight_cmp);
  return w; 
}

/************ max_weights *********************************/ 
/* Get maximum element from vector of Edges */ 
template <typename It> 
static inline weight_t 
max_weights(It start, It end)
{
  auto weight_cmp = [](const Edge& a, const Edge& b) -> bool {
    return a.weight < b.weight;
  };

  auto [s, d, w] = *std::max_element(start, end, weight_cmp);
  return w; 
}

/************ Graph::Graph ********************************/ 
/* Parameterized ctor for Graph Class.   
 *
 * Throws: 
 *   exceptions from validate() on class construction 
 */ 
Graph::Graph(node_index_t num_nodes, 
             std::vector<edge_index_t> index_ptr,
             std::vector<node_index_t> indices, 
             std::vector<weight_t> weights, 
             bool directed) 
  : num_nodes_(num_nodes),
    num_edges_(static_cast<edge_index_t>(indices.size())),
    directed_(directed), 
    index_ptr_(std::move(index_ptr)), 
    indices_(std::move(indices)), 
    weights_(std::move(weights))
{
  validate(); 
}

/************ Graph::neighbor_begin ***********************/
/* Yields index to first neighbor of node u */ 
edge_index_t 
Graph::neighbor_begin(node_index_t u) const 
{
  if ( u >= num_nodes_ ) {
    throw std::out_of_range("neighbor_begin: node index out of range"); 
  }
  return index_ptr_.at(u); 
}

/************ Graph::neighbor_end *************************/
/* Yields index to index after node u's neighbors */ 
edge_index_t 
Graph::neighbor_end(node_index_t u) const 
{
  if ( u >= num_nodes_ ) {
    throw std::out_of_range("neighbor_begin: node index out of range"); 
  }
  return index_ptr_.at(static_cast<size_t>(u) + 1); 
}

/************ Graph::to_coo *******************************/ 
/* Converts Graph from Compressed Sparse Row 
 * to Coordinate List format 
 */ 
std::pair<std::vector<std::pair<node_index_t, node_index_t>>, std::vector<weight_t>> 
Graph::to_coo(void) const 
{
  std::vector<std::pair<node_index_t, node_index_t>> edges; 
  std::vector<weight_t> weights; 

  edges.reserve(indices_.size());
  weights.reserve(weights_.size()); 

  for (node_index_t u = 0; u < num_nodes_; u++) {
    const size_t u_idx = static_cast<size_t>(u); 
    const auto begin   = static_cast<size_t>(index_ptr_[u_idx]);
    const auto end     = static_cast<size_t>(index_ptr_[u_idx + 1]);

    for (size_t e = begin; e < end; e++) {
      edges.emplace_back(u, indices_[e]); 
      weights.push_back(weights_[e]); 
    }
  }

  return {std::move(edges), std::move(weights)};
}

/************ Graph::sources ******************************/ 
/* Gets vector of nodes that are sources in our Graph Representation 
 */ 
std::vector<node_index_t> 
Graph::sources(void) const 
{
  std::vector<node_index_t> out(indices_.size()); 

  for (node_index_t u = 0; u < num_nodes_; u++) {
    const size_t u_idx = static_cast<size_t>(u); 
    const auto begin   = static_cast<size_t>(index_ptr_[u]);
    const auto end     = static_cast<size_t>(index_ptr_[u_idx + 1]);

    for (size_t e = begin; e < end; e++) {
      out[e] = u;
    }
  }

  return out; 
}

/************ Graph::validate *****************************/ 
/* Ensures Graph ctor does not fail to generate the 
 * CSR representation correctly 
 *
 * Throws: 
 *   runtime_error, length_error, invalid_argument, range_error
 */
void 
Graph::validate(void) const 
{
  if ( !(index_ptr_.size() == static_cast<size_t>(num_nodes_) + 1) ) { 
    throw std::runtime_error("Graph::validate: index_ptr size must be num_nodes + 1"); 
  } 

  if ( !(indices_.size() == weights_.size()) ) {
    throw std::length_error("Graph::validate: indices and weights must have equal length"); 
  }

  if ( num_nodes_ == 0 ) {
    if ( !(index_ptr_.size() == 1) ) { 
      throw std::length_error("Graph::validate: num_nodes=0 requires index_ptr_ size 1"); 
    }

    if ( !(index_ptr_[0] == 0) ) {
      throw std::invalid_argument("Graph::validate: index_ptr[0] must be 0"); 
    }

    if ( !(indices_.empty()) ) {
      throw std::invalid_argument("Graph::validate: num_nodes=0 cannot have edges"); 
    }
  }

  if ( !(index_ptr_.front() == 0) ) {
    throw std::invalid_argument("Graph::validate: index_ptr_[0] must be 0"); 
  }

  for (auto it = index_ptr_.begin(); it != index_ptr_.end() - 1; it++) {
    const auto n = *it; 
    const auto m = *(it + 1); 

    if ( n > m ) {
      throw std::invalid_argument("Graph::validate: index_ptr must be non-decreasing");
    }
  }

  if ( !(static_cast<size_t>(index_ptr_.back()) == indices_.size()) ) { 
    throw std::length_error("Graph::validate: index_ptr_.back() must equal num_edges");
  }

  for (size_t e = 0; e < indices_.size(); e++) {
    if ( (indices_[e] >= num_nodes_) ) {
      throw std::range_error("Graph::validate: edge dst out of range");  
    }

    if ( !std::isfinite(weights_[e]) ) {
      throw std::runtime_error("Graph::validate: weight must be finite"); 
    }
  }
}

/************ GraphBuilder::GraphBuilder ******************/ 
/* Parameterized Ctor */ 
GraphBuilder::GraphBuilder(node_index_t num_nodes, BuildOptions options) 
  : num_nodes_(num_nodes), options_(options) {}

/************ GraphBuilder Public Interfaces **************/ 

/************ GraphBuilder::reserve_edges *****************/ 
void 
GraphBuilder::reserve_edges(size_t n) 
{ 
  edges_.reserve(n); 
}

/************ GraphBuilder::clear *************************/
void 
GraphBuilder::clear(void) noexcept 
{
  edges_.clear(); 
}

/************ GraphBuilder::add_edge **********************/ 
void 
GraphBuilder::add_edge(Edge edge)
{
  can_add_edge(edge.src, edge.dst, edge.weight);
  edges_.push_back(std::move(edge));
}

/************ GraphBuilder::add_edge **********************/ 
void 
GraphBuilder::add_edge(node_index_t src, node_index_t dst, weight_t w)
{
  can_add_edge(src, dst, w); 
  edges_.emplace_back(src, dst, w); 
}

/************ GraphBuilder::add_edges *********************/ 
/* Wrapped call to add_edge over range of edges */  
void 
GraphBuilder::add_edges(const std::vector<Edge>& edges)
{
  reserve_edges(edges_.size() + edges.size()); 
  for (const auto& e : edges) {
    add_edge(e.src, e.dst, e.weight); 
  }
}

/************ GraphBuilder::can_add_edge ******************/ 
/* Ensures an edge can be added given {u, v, w} */ 
void 
GraphBuilder::can_add_edge(node_index_t src, node_index_t dst, weight_t w)
{
  if ( src >= num_nodes_ || dst >= num_nodes_ ) {
    throw std::out_of_range("GraphBuilder::add_edge: node index out of range");  
  }

  if ( !options_.allow_self_loops && src == dst ) {
    throw std::invalid_argument("GraphBuilder::add_edge: self-loops disabled"); 
  }

  if ( !std::isfinite(w) ) {
    throw std::invalid_argument("GraphBuilder::add_edge: weight be finite");
  } 
}

/************ GraphBuilder::apply_reverse_edges ***********/ 
/* Converts a directed graph into undirected by pushing 
 * the reverse of every edge into edges 
 */ 
void 
GraphBuilder::apply_reverse_edges(std::vector<Edge>& edges, const BuildOptions& opt)
{
  const bool need_reverse = opt.add_reverse_edges || !opt.directed; 
  if ( !need_reverse ) {
    return; 
  }

  const size_t original = edges.size(); 
  edges.reserve(original * 2); 
  for (const auto& e : edges) {
    edges.push_back(Edge{e.dst, e.src, e.weight}); 
  }
}

/************ GraphBuilder::filter_self_loops *************/ 
/* Removes any self loops from the graph via stl algorithms */ 
void 
GraphBuilder::filter_self_loops(std::vector<Edge>& edges, const BuildOptions& opt)
{
  if ( opt.allow_self_loops ) {
    return; 
  }

  auto is_loop = [](const Edge& e) -> bool { 
    return e.src == e.dst; 
  }; 

  edges.erase(std::remove_if(edges.begin(), edges.end(), is_loop), edges.end()); 
}

/************ GraphBuilder::sort_edges ********************/ 
/* Wrapper of std/stl sort functions 
 *
 * Caller Provides: 
 *   option to sort on destination node_index_t. 
 *     Edge implements < as lexigraphical sort, so a stable_cmp is instantiated 
 *     as functor. 
 */
void 
GraphBuilder::sort_edges(std::vector<Edge>& edges, bool sort_by_dst)
{
  if ( sort_by_dst ) {
    std::sort(edges.begin(), edges.end()); 
    return; 
  }

  auto stable_cmp = [](const Edge& a, const Edge& b) -> bool {
    return a.src < b.src;
  }; 

  std::stable_sort(edges.begin(), edges.end(), stable_cmp);
}

/************ GraphBuilder::deduplicate_edges *************/ 
/* Handles duplicate edges based on passed policy. 
 *
 * Presorts via lexigraphical sort as ordering (<, E). 
 * Sliding window implementation on node u's dst's 
 *
 * see graph.hpp for options of DuplicateEdgePolicy 
 * 
 * Caller Provides: 
 *   policy for deduplication. 
 */ 
void 
GraphBuilder::deduplicate_edges(std::vector<Edge>& edges, DuplicateEdgePolicy policy)
{
  if ( edges.empty() ) {
    return; 
  }

  std::sort(edges.begin(), edges.end()); 

  size_t write = 0; 
  size_t read  = 0; 

  // Sliding window on dst 
  while ( read < edges.size() ) {
    size_t next = read + 1; 

    // While source nodes match and destinations match (before end of vector) 
    
    while ( next < edges.size() && edges[next].src == edges[read].src 
      && edges[next].dst == edges[read].dst ) {
      next++;  
    }

    // count is # of duplicate edges
    const size_t count = next - read;
    Edge merged = edges[read]; 

    if ( count > 1 ) {
      switch ( policy ) {
        case DuplicateEdgePolicy::ERROR: 
          throw std::runtime_error("GraphBuilder::build: duplicate edges encountered"); 
          break;
        case DuplicateEdgePolicy::KEEP_FIRST: 
          merged.weight = edges[read].weight; 
          break; 
        case DuplicateEdgePolicy::KEEP_LAST: 
          merged.weight = edges[next - 1].weight; 
          break; 
        case DuplicateEdgePolicy::SUM: 
          merged.weight = sum_weights(edges.begin() + read, edges.begin() + next); 
          break; 
        case DuplicateEdgePolicy::MEAN:  
          merged.weight = sum_weights(edges.begin() + read, edges.begin() + next) / count; 
          break; 
        case DuplicateEdgePolicy::MIN: 
          merged.weight = min_weights(edges.begin() + read, edges.begin() + next); 
          break;  
        case DuplicateEdgePolicy::MAX: 
          merged.weight = max_weights(edges.begin() + read, edges.begin() + next);
          break; 
      }
    }

    // Chunk is replaced with merged edge, update window 
    // Count written edges 

    edges[write++] = merged; 
    read = next; 
  }
  
  // Shrink edges to new size 
  edges.resize(write); 
}

/************ GraphBuilder::build *************************/ 
/* Builds Graph from instantiated GraphBuilder. Passes built 
 * parameters to Graph parameterized ctor. 
 *
 * Prefer call to GraphBuilder in python interface 
 */
Graph 
GraphBuilder::build(void)
{
  std::vector<Edge> edges = std::move(edges_); 
  clear(); 

  apply_reverse_edges(edges, options_); 
  filter_self_loops(edges, options_); 
  deduplicate_edges(edges, options_.dedup); 
  sort_edges(edges, options_.sort_by_dst); 

  if ( num_nodes_ == 0 ) {
    if ( edges.empty() ) {
      return Graph(0, std::vector<edge_index_t>{0}, {}, {}, options_.directed); 
    } else {
      throw std::invalid_argument("GraphBuilder::build: num_nodes=0 cannot have edges"); 
    }
  }

  if ( edges.size() > static_cast<size_t>(std::numeric_limits<edge_index_t>::max()) ) {
    throw std::runtime_error("GraphBuilder::build :too many edges for edge_index_t: " 
                             + std::to_string(sizeof(edge_index_t))); 
  }

  std::vector<size_t> degree(num_nodes_, 0); 
  for (const auto& e : edges) {
    degree[e.src]++; 
  }

  std::vector<edge_index_t> index_ptr(static_cast<size_t>(num_nodes_) + 1, 0); 
  size_t running = 0; 
  index_ptr[0] = 0; 

  for (size_t u = 0; u < static_cast<size_t>(num_nodes_); u++) {
    running += degree[u]; 
    if ( running > static_cast<size_t>(std::numeric_limits<edge_index_t>::max()) ) {
      throw std::runtime_error("GraphBuilder::build: index_ptr overflow"); 
    }
    index_ptr[u + 1] = static_cast<edge_index_t>(running); 
  } 

  std::vector<node_index_t> indices(edges.size()); 
  std::vector<weight_t> weights(edges.size()); 

  std::vector<edge_index_t> cursor = index_ptr; 
  for (const auto& e : edges) {
    const auto pos = static_cast<size_t>(cursor[e.src]++); 
    indices[pos] = e.dst; 
    weights[pos] = e.weight; 
  }

  return Graph(num_nodes_, std::move(index_ptr), std::move(indices), 
               std::move(weights), options_.directed); 
}

} // namespace topo::graph 
