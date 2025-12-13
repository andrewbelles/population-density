/*
 * python_bindings.cpp  Andrew Belles  Dec 12th, 2025 
 *
 * Bindings for GeospatialGraph struct for clean use with 
 * pytorch for GraphNN modeling 
 *
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h> 
#include <pybind11/numpy.h> 

#include "geospatial_graph.hpp" 
#include "pybind11/detail/common.h"
#include "support.hpp"

namespace py = pybind11;

PYBIND11_MODULE(geospatial_graph_cpp, m) {
  m.doc() = "GeospatialGraph C++ bindings for high-performance spatially aware county graph operations"; 

  // support::County 
  
  py::class_<County>(m, "County") 
    .def(py::init<const std::string&, const std::string&, const std::string&, const County::Coordinate&>())
    .def("id", &County::id, "Get county GEOID") 
    .def("coord", &County::coord, "Get (lat, lon) coordinates")
    .def("__repr__", [](const County& c) {
      auto [lat, lon] = c.coord(); 
      return "County(id='" + c.id() + "', lat=" + std::to_string(lat) + ", lon=" + std::to_string(lon) + ")"; 
    });

  // support::County::Coordinate

  py::class_<County::Coordinate>(m, "Coordinate")
    .def(py::init<double, double>())
    .def("__getitem__", [](const County::Coordinate& c, size_t i) {
      switch (i) {
        case 0: 
          return c.first; 
        case 1: 
          return c.second; 
        default: 
          throw py::index_error("Coordinate index out of range"); 
      }
    })
    .def("__repr__", [](const County::Coordinate& c) {
      return "(" + std::to_string(c.first) + ", " + std::to_string(c.second) + ")"; 
    }); 

  // GeospatialGraph::metricType enum 
  
  py::enum_<GeospatialGraph::metricType>(m, "MetricType") 
    .value("KNN", GeospatialGraph::metricType::KNN)
    .value("BOUNDED", GeospatialGraph::metricType::BOUNDED)
    .value("STANDARD", GeospatialGraph::metricType::STANDARD)
    .export_values(); 

  // GeospatialGraph 
  
  py::class_<GeospatialGraph>(m, "GeospatialGraph")
    .def(py::init<const std::string& , GeospatialGraph::metricType, double>(), 
         "Create GeospatialGraph from county file and metric function", 
         py::arg("filepath"), py::arg("metric_type"), py::arg("parameter")) 
    .def("get_neighbors", &GeospatialGraph::get_neighbors,
         "Get neighbors of a county by GEOID",
         py::arg("geoid")) 
    .def("counties", &GeospatialGraph::counties, 
         "Get all counties", 
         py::return_value_policy::reference_internal)
    .def("get_edge_indices_and_distances", &GeospatialGraph::get_edge_indices_and_distances,
         "Get edge indices as list of (source, target) pairs and their respective distances") 
    .def("get_all_coordinates", &GeospatialGraph::get_all_coordinates, 
         "Get (lat, lon) for all counties")
    .def("get_geoid_to_index", &GeospatialGraph::get_geoid_to_index, 
         "Get mapping from GEOID to index")
    .def("__repr__", [](const GeospatialGraph& g) {
      return "GeospatialGraph(counties=" + std::to_string(g.counties().size()) + ")"; 
    }); 

  // Utility/Support functions 

  m.def("load_counties", &load_counties, 
        "Load counties from gazetteer file",
        py::arg("filepath"));
}
