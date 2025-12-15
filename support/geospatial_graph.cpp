/* 
 * geospatial_graph.cpp  Andrew Belles  Dec 12th, 2025 
 *
 * Implementation of GeospatialGraph interface with emphasis 
 * on modular formation of adjacencyList from different 
 * metricFunctions 
 */ 

#include "geospatial_graph.hpp"
#include "support.hpp"

#include <cmath> 
#include <algorithm> 
#include <unordered_map>

/************ Compile Time Constants **********************/ 
/* For haversine distance */ 
constexpr double EARTH_RADIUS_KM = 6371.0; 
constexpr double TO_RAD          = M_PI / 180.0; 
constexpr double TO_RAD_2        = M_PI / 360.0; 

/************ Static Helper Functions *********************/ 
static double haversine_distance(const County::Coordinate src, const County::Coordinate rel);

/************ GeospatialGraph ctor ************************/ 
/* Constructs a GeospatialGraph without features given a metricType  
 * and its corresponding parameter/bound to be used 
 *
 * Caller Provides: 
 *    filepath a valid filepath to gazetteer to load county metadata 
 *    metricType defining which metric to define being adjacent 
 *    matric_parameter used by metricFunction 
 */ 
GeospatialGraph::GeospatialGraph(const std::string& filepath, 
                                 const GeospatialGraph::metricType metric_type, 
                                 const double metric_parameter)
  : metric_type_(metric_type), metric_parameter_(metric_parameter)
{
  this->counties_ = load_counties(filepath); 
  
  // Create lookup map of counties indexed on their geoid 
  for (auto& county : this->counties_) {
    this->counties_map_[county.id()] = county;
  }

  this->compute_distance_matrix();
  this->build_adjacency_list();
}

/************ compute_distance_matrix() *******************/ 
/* Distance Matrix using haversine_distance() between two counties 
 * centroids. Haversine distance is the distance between two points on 
 * a sphere given their latitude and longitude. 
 */  
void 
GeospatialGraph::compute_distance_matrix(void)
{
size_t n{this->counties_.size()}; 
  this->distance_matrix_.resize(n, std::vector<double>(n, 0.0));

  for (size_t i = 0; i < n; i++) {
    auto ci     = this->counties_[i].coord(); 
    auto& row_i = this->distance_matrix_[i]; 

    for (size_t j = i + 1; j < n; j++) {
      auto cj     = this->counties_[j].coord(); 
      auto& row_j = this->distance_matrix_[j];  

      double dist = haversine_distance(ci, cj);

      row_i[j] = dist; 
      row_j[i] = dist; 
    }
  }
}

/************ build_adjacency_list() **********************/ 
/* Builds adjacency list using supporting metricFunction.  
 * 
 * We Assume: 
 *   GeospatialGraph was instantiated with a valid metricType 
 */
void 
GeospatialGraph::build_adjacency_list(void)
{
  this->county_adjacency_list_.clear(); 
  size_t n{this->counties_.size()};

  auto metric_fn = this->metric_from_type(this->metric_type_); 

  for (size_t county_idx = 0; county_idx < n; county_idx++) {
    auto neighbors = metric_fn(
      county_idx, 
      this->distance_matrix_, 
      this->metric_parameter_
    );  

    std::vector<GeospatialGraph::stringIndex> geoid_neighbors; 
    for (const auto& [neighbor_idx, distance] : neighbors) {
      geoid_neighbors.emplace_back(this->counties_[neighbor_idx].id(), distance); 
    }

    // get key and move string indexed neighbor list 
    // into adjacency list (represented as hashmap in memory) 
    
    auto county_geoid = this->counties_[county_idx].id(); 
    this->county_adjacency_list_[county_geoid] = std::move(geoid_neighbors);
  }
}

/************ get_edge_indices_and_distances() ************/ 
/* Gets vectors of county indices representing edges and vectors of distances
 * that pytorch needs for most Graph Neural Networks
 */ 
std::pair<std::vector<std::pair<size_t, size_t>>, std::vector<double>>
GeospatialGraph::get_edge_indices_and_distances(void) const
{
  std::vector<std::pair<size_t, size_t>> edges; 
  std::vector<double> distances; 

  size_t estimated_edges = 0; 
  for (const auto& [geoid, neighbors] : this->county_adjacency_list_) {
    estimated_edges += neighbors.size(); 
  }
  edges.reserve(estimated_edges); 
  distances.reserve(estimated_edges); 

  auto geoid_to_idx = get_geoid_to_index(); 
  for (size_t county_idx = 0; county_idx < this->counties_.size(); county_idx++) {
    const auto& geoid = this->counties_[county_idx].id(); 
    
    auto it = this->county_adjacency_list_.find(geoid); 
    if ( it == this->county_adjacency_list_.end() ) {
      continue; 
    } 
    
    for (const auto& [neighbor_geoid, distance] : it->second) {
      auto nit = geoid_to_idx.find(neighbor_geoid); 
      if ( nit == geoid_to_idx.end() ) {
        continue; 
      }

      edges.emplace_back(county_idx, nit->second); 
      distances.push_back(distance); 
    }
  }

  return {std::move(edges), std::move(distances)}; 
}

/************ get_all_coordinates() ***********************/ 
/* Gets all (lat, lon) coordinates as a vector */ 
std::vector<std::pair<double, double>> 
GeospatialGraph::get_all_coordinates(void) const 
{
  std::vector<std::pair<double, double>> coords; 
  coords.reserve(this->counties_.size()); 

  for (const auto& county : this->counties_) {
    coords.push_back(county.coord()); 
  }

  return coords; 
}

/************ geoid_to_idx() ******************************/ 
/* Creates lookup table for converting geoid into index */ 
std::unordered_map<std::string, size_t>
GeospatialGraph::get_geoid_to_index(void) const 
{
  std::unordered_map<std::string, size_t> mapping; 
  mapping.reserve(this->counties_.size()); 
  for (size_t county_idx = 0; county_idx < this->counties_.size(); county_idx++) {
    mapping[this->counties_[county_idx].id()] = county_idx; 
  }
  return mapping; 
}

/************ get_neighbors() *****************************/ 
/* For a county, return all neighbors 
 * in adjacency list as option 
 */
std::optional<const std::vector<County>> 
GeospatialGraph::get_neighbors(const std::string& key) const 
{
  if ( auto it = this->county_adjacency_list_.find(key); it != this->county_adjacency_list_.end() ) {
    std::vector<County> result; 
    result.reserve(it->second.size()); 

    for (const auto& [neighbor_key, _] : it->second) {
      result.push_back(this->counties_map_.at(neighbor_key)); 
    }

    return result; 
  } else {
    return std::nullopt; 
  }
}

/************ metric_from_type ****************************/ 
/* metricType into metricFunction */ 
GeospatialGraph::metricFunction 
GeospatialGraph::metric_from_type(GeospatialGraph::metricType metric_type)
{
  using metricType = GeospatialGraph::metricType; 

  switch ( metric_type ) {
    case metricType::KNN: 
      return GeospatialGraph::knn_metric; 
    case metricType::BOUNDED: 
      return GeospatialGraph::bounded_metric; 
    default: 
    case metricType::STANDARD: 
      return GeospatialGraph::standard_metric; 
  }
}

/************ Supported Metric Functions ******************/ 
/* For all metric functions: 
 *
 * Caller Provides: 
 *    county_idx, the current node of adj list 
 *    vector of county metadata 
 *    precomputed distanceMatrix 
 *    metric parameter which acts as a bound on the specific metric 
 */ 

/************ knn_metric() ********************************/ 
/* Computes the k nearest neighbors from the precomputed 
 * distance matrix 
 *
 * Caller Provides (specific to knn):
 *    k_neighbors specifying the max neighbors to take for a node 
 */
std::vector<GeospatialGraph::integerIndex> 
GeospatialGraph::knn_metric(const size_t county_idx, const distanceMatrix &distances, 
                            const double k_neighbors)
{
  using integerIndex = GeospatialGraph::integerIndex;
  size_t k{static_cast<size_t>(k_neighbors)};   // coerce metric_parameter_ into size_t 

  std::vector<integerIndex> neighbors; 
  const auto& county_distances = distances[county_idx]; 

  size_t n_neighbors{county_distances.size()}; 
  for (size_t neighbor_idx = 0; neighbor_idx < n_neighbors; neighbor_idx++) {
    if ( neighbor_idx != county_idx ) {
      neighbors.emplace_back(neighbor_idx, county_distances[neighbor_idx]);
    } 
  }

  // Push smallest k distances to front and sort subarray. Return that range 
  k = std::min(k, neighbors.size());
  std::partial_sort(neighbors.begin(), neighbors.begin() + k, neighbors.end(), 
                    [](const auto& a, const auto& b) { 
                      return a.second < b.second; 
                    }); 
  return std::vector<integerIndex>(neighbors.begin(), neighbors.begin() + k);  
}

/************ bounded_metric() ***************************/ 
/* Bounds haversine metric on specified threshold. 
 *
 * Caller Provides (specific to haversine bounded metric): 
 *    Let M := distance_threshold. For two counties x,y d(x,y) > M implies 
 *    that x, y are not neighbors. 
 *
 */ 
std::vector<GeospatialGraph::integerIndex> 
GeospatialGraph::bounded_metric(const size_t county_idx, const distanceMatrix &distances, 
                                const double distance_threshold)
{
  using integerIndex = GeospatialGraph::integerIndex;

  std::vector<integerIndex> neighbors; 
  const auto& county_distances = distances[county_idx]; 

  // Only accept counties that are within bounded metric on distance_threshold 
  size_t n_neighbors{county_distances.size()}; 
  for (size_t neighbor_idx = 0; neighbor_idx < n_neighbors; neighbor_idx++) {
    if ( county_distances[neighbor_idx] <= distance_threshold && neighbor_idx != county_idx ) {
      neighbors.emplace_back(neighbor_idx, county_distances[neighbor_idx]);
    } 
  }
 
  // Sort on distances 
  std::sort(neighbors.begin(), neighbors.end(), [](const auto& a, const auto& b){
    return a.second < b.second; 
  });

  return neighbors; 
}

/************ standard_metric *****************************/ 
/* All counties are neighbors, sorted by their standard metric distances 
 * where haversine is considered standard for points on S^2.
 */ 
std::vector<GeospatialGraph::integerIndex> 
GeospatialGraph::standard_metric(const size_t county_idx, const distanceMatrix &distances, 
                                 const double) 
{
  using integerIndex = GeospatialGraph::integerIndex;

  std::vector<integerIndex> neighbors; 
  const auto& county_distances = distances[county_idx]; 
  
  // Take all counties for a complete matrix 
  size_t n_neighbors{county_distances.size()}; 
  for (size_t neighbor_idx = 0; neighbor_idx < n_neighbors; neighbor_idx++) {
    if ( neighbor_idx != county_idx ) {
      neighbors.emplace_back(neighbor_idx, county_distances[neighbor_idx]);
    } 
  }
 
  // Sort on distances 
  std::sort(neighbors.begin(), neighbors.end(), [](const auto& a, const auto& b){
    return a.second < b.second; 
  });

  return neighbors; 
}


/************ haversine_distance() ************************/ 
/* Computes distance between two points given their latitude 
 * and longitude on a sphere. Partially optimized at the cost 
 * of some accuracy by simplifying trig calls in original formula 
 *
 * Caller provides: 
 *   src and rel coordinates which just represent the two points in S^2 
 */ 
static double 
haversine_distance(const County::Coordinate src, const County::Coordinate rel)
{
  const auto [lat_i, lon_i] = src; 
  const auto [lat_j, lon_j] = rel; 

  // precompute all constant values to avoid roundoff 
  const double cos_lat_j_rad  = std::cos(lat_j * TO_RAD); 
  const double cos_lat_i_rad  = std::cos(lat_i * TO_RAD); 
  const double sin_dlat_2     = std::sin((lat_j - lat_i) * TO_RAD_2); 
  const double sin_dlon_2     = std::sin((lon_j - lon_i) * TO_RAD_2); 
  const double a = (sin_dlat_2 * sin_dlat_2) + (cos_lat_i_rad * cos_lat_j_rad * sin_dlon_2 * sin_dlon_2);
  
  return EARTH_RADIUS_KM * 2.0 * std::atan2(std::sqrt(a), std::sqrt(1.0 - a)); 
}
