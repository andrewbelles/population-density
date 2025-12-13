/*
 * geospatial_graph.hpp  Andrew Belles  Dec 12th, 2025 
 *
 * Interface for C++ implementation of GeospatialGraph 
 * for use in python graphing machine learning modeling 
 *
 */ 

#pragma once 

#include <unordered_map>
#include <vector> 
#include <string> 
#include <functional> 
#include <cstdint> 
#include <optional>

struct County; 

/************ GeospatialGraph *****************************/ 
/* Interface for generating graph based on geospatial adjacency. 
 *
 * Adjacency is defined by the choice of metric used. metricType 
 * enumerates available metricFunctions for which neighbors are determined
 *
 * This interface is to be wrapped in a python interface so that features can 
 * be associated with each node 
 */ 
class GeospatialGraph {
private: 
  /********** Type Bindings *******************************/ 

  using integerIndex   = std::pair<size_t, double>; 
  using stringIndex    = std::pair<std::string, double>; 
  using distanceMatrix = std::vector<std::vector<double>>;  
  using adjacencyList  = std::unordered_map<std::string, std::vector<stringIndex>>; 

public: 
  using metricFunction = std::function<std::vector<integerIndex>(
    const size_t,  const distanceMatrix&, const double
  )>; 

  enum class metricType : uint8_t {
    KNN, 
    BOUNDED, 
    STANDARD  
  };
  
  explicit GeospatialGraph(const std::string& filepath, const metricType metric_type, 
                           const double metric_threshold);

  GeospatialGraph(GeospatialGraph&& graph); 
  
  /********** pytorch tensor helper functions *************/ 
  std::pair<std::vector<std::pair<size_t, size_t>>, std::vector<double>> 
  get_edge_indices_and_distances(void) const; 
  
  std::vector<std::pair<double, double>> get_all_coordinates(void) const; 
  std::unordered_map<std::string, size_t> get_geoid_to_index(void) const; 
  
  /********** Getters *************************************/ 

  const std::vector<County>& counties(void) const { return counties_; }
  std::optional<const std::vector<County>> get_neighbors(const std::string& key) const; 

  static std::vector<integerIndex> 
  knn_metric(const size_t county_idx, const distanceMatrix& distances, 
             const double k_neighbors); 

  static std::vector<integerIndex> 
  bounded_metric(const size_t county_idx, const distanceMatrix& distances, 
                 const double k_neighbors); 

  static std::vector<integerIndex> 
  standard_metric(const size_t county_idx, const distanceMatrix& distances, 
                  const double /* unused */); 

private: 
  std::unordered_map<std::string, County> counties_map_; // lookup table on geoid of counties 

  adjacencyList county_adjacency_list_{};  
  distanceMatrix distance_matrix_{}; 

  std::vector<size_t> county_degrees_{}; 
  std::vector<County> counties_;
  metricType metric_type_{};
  double metric_parameter_{}; 

  void compute_distance_matrix(void); 
  void build_adjacency_list(void); 
  static metricFunction metric_from_type(metricType metric_type); 
}; 
