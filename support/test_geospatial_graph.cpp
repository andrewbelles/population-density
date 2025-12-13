/* 
 * test_geospatial_graph.cpp  Andrew Belles  Dec 12th, 2025   
 *   
 * Unit Tests for GeospatialGraph interface 
 *
 *  
 */  

#include <iostream> 
#include <cassert> 
#include <vector> 
#include <string> 

#include "geospatial_graph.hpp"
#include "support.hpp"

/************ Testing Macro ******************************/
#define TEST(name) void test_##name() 
#define RUN_TEST(name) do { \
  std::cout << "Running " #name "..."; \
  test_##name(); \
  std::cout << "PASSED\n"; \
} while(0)


TEST(load_counties) {
  auto counties = load_counties("../data/geography/2020_Gaz_counties_national.txt"); 
  assert( counties.size() > 3200 ); 
  assert( !counties[0].id().empty() ); 
  
  auto [lat, lon] = counties[0].coord(); 
  assert( lat >= -90 && lat <= 90 ); 
  assert( lon >= -180 && lon <= 180 ); 
}

TEST(geospatial_graph_knn_ctor) {
  try {
    auto graph = GeospatialGraph(
      "../data/geography/2020_Gaz_counties_national.txt",  
      GeospatialGraph::metricType::KNN, 5 
    ); 

    if ( auto neighbors = graph.get_neighbors("01001"); neighbors.has_value() ) {
      assert(neighbors.value().size() == 5); 
    } else {
      std::cerr << "FAILURE: key 01001 does not have any neighbors\n"; 
      assert(false);
    }

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << '\n'; 
    assert(false); 
  }
}

TEST(geospatial_graph_bounded_ctor) {
  try {
    auto graph = GeospatialGraph(
      "../data/geography/2020_Gaz_counties_national.txt",  
      GeospatialGraph::metricType::BOUNDED, 5 
    ); 

    if ( auto neighbors = graph.get_neighbors("01001"); neighbors.has_value() ) {
      assert( neighbors.has_value() ); 
    } else {
      std::cerr << "FAILURE: key 01001 does not have any neighbors\n"; 
      assert(false);
    }

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << '\n'; 
    assert(false); 
  }
}

TEST(geospatial_graph_standard_ctor) {
  try {
    auto graph = GeospatialGraph(
      "../data/geography/2020_Gaz_counties_national.txt",  
      GeospatialGraph::metricType::STANDARD, 0 
    ); 

    if ( auto neighbors = graph.get_neighbors("01001"); neighbors.has_value() ) {
      assert( neighbors.value().size() == graph.counties().size() - 1 ); 
    } else {
      std::cerr << "FAILURE: key 01001 does not have any neighbors\n"; 
      assert(false);
    }

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << '\n'; 
    assert(false); 
  }
}

TEST(ensure_edge_indices_and_distances) {
  auto graph = GeospatialGraph(
    "../data/climate_counties.tsv", 
    GeospatialGraph::metricType::KNN, 5
  );

  auto [edges, costs] = graph.get_edge_indices_and_distances(); 
  assert( edges.size() == costs.size() ); 

  // each county has at most 5 neighbors 
  size_t max_degree = graph.counties().size() * 5;
  assert( edges.size() <= max_degree );
}

TEST(ensure_coordinates) {
  auto graph = GeospatialGraph(
    "../data/climate_counties.tsv", 
    GeospatialGraph::metricType::KNN, 1
  );

  // Ensure all coordinates are within expected range 
  auto coordinates = graph.get_all_coordinates(); 
  for (auto& [lat, lon] : coordinates) {
    assert( lat >= 15.0 && lat <= 72.0 );
    assert( lon >= -180.0 && lon <= 180.0 ); 
  }

  assert( coordinates.size() == graph.counties().size() ); 
}

TEST(ensure_geoid_to_index) {
  auto graph = GeospatialGraph(
    "../data/climate_counties.tsv", 
    GeospatialGraph::metricType::KNN, 1
  );

  // Ensure that all geoid's are indexed on an integer properly 
  auto geoid_to_index = graph.get_geoid_to_index(); 
  for (auto& county : graph.counties()) {
    auto geoid = county.id(); 

    auto it = geoid_to_index.find(geoid); 
    assert( it != geoid_to_index.end() ); 
  }
}

int main(void) 
{
  std::cout << "Running GeospatialGraph tests...\n"; 

  try {
    RUN_TEST(load_counties); 
    RUN_TEST(geospatial_graph_knn_ctor); 
    RUN_TEST(geospatial_graph_bounded_ctor); 
    RUN_TEST(geospatial_graph_standard_ctor);
    RUN_TEST(ensure_edge_indices_and_distances); 
    RUN_TEST(ensure_coordinates); 
    RUN_TEST(ensure_geoid_to_index);
  
    std::cout << "\nAll tests passed\n"; 
  } catch (const std::exception& e) {
    std::cerr << "Test failed: " << e.what() << '\n'; 
    exit(1); 
  }

  return 0; 
}
