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

#include <cstdint>
#include <stdexcept>
#include <string> 
#include <vector> 

#include "graph.hpp"

namespace py = pybind11; 
using namespace topo::graph; 

static constexpr py::ssize_t TWO_SSIZE{2}; 

static py::array_t<int64_t> vec_u32_to_i64(const std::vector<uint32_t>& v);
static std::pair<py::array_t<int64_t>, py::array_t<double>> graph_to_coo_numpy(const Graph& g); 
static Graph build_graph_from_numpy(node_index_t num_nodes, py::array edge_index_any, 
                                    py::object edge_weight_any, const BuildOptions& opt);

PYBIND11_MODULE(graph_cpp, m)
{
  m.doc() = "Generic CSR Graph Backend (topo::graph)"; 

  py::enum_<DuplicateEdgePolicy>(m, "DuplicateEdgePolicy")
    .value("ERROR", DuplicateEdgePolicy::ERROR)
    .value("KEEP_FIRST", DuplicateEdgePolicy::KEEP_FIRST)
    .value("KEEP_LAST", DuplicateEdgePolicy::KEEP_LAST)
    .value("SUM", DuplicateEdgePolicy::SUM)
    .value("MEAN", DuplicateEdgePolicy::MEAN)
    .value("MIN", DuplicateEdgePolicy::MIN)
    .value("MAX", DuplicateEdgePolicy::MAX)
    .export_values();

  py::class_<BuildOptions>(m, "BuildOptions")
    .def(py::init<>())
    .def_readwrite("directed", &BuildOptions::directed)
    .def_readwrite("sort_by_dst", &BuildOptions::sort_by_dst)
    .def_readwrite("add_reverse_edges", &BuildOptions::add_reverse_edges)
    .def_readwrite("allow_self_loops", &BuildOptions::allow_self_loops)
    .def_readwrite("dedup", &BuildOptions::dedup);

  py::class_<Edge>(m, "Edge")
    .def(py::init<node_index_t, node_index_t, weight_t>(),
         py::arg("src"), py::arg("dst"), py::arg("weight") = 1.0)
    .def_readwrite("src", &Edge::src)
    .def_readwrite("dst", &Edge::dst)
    .def_readwrite("weight", &Edge::weight);

  py::class_<Graph>(m, "Graph")
    .def(py::init<>())
    .def("num_nodes", &Graph::num_nodes)
    .def("num_edges", &Graph::num_edges)
    .def("directed", &Graph::directed)
    .def("index_ptr", [](const Graph& g) { return vec_u32_to_i64(g.index_ptr()); },
         "CSR row pointer (int64 numpy)")
    .def("indices", [](const Graph& g) { return vec_u32_to_i64(g.indices()); },
         "CSR column indices (int64 numpy)")
    .def("to_coo", &Graph::to_coo,
         "Return COO as (list[(src,dst)], list[weight])")
    .def("to_coo_numpy", [](const Graph& g) { return graph_to_coo_numpy(g); },
         "Return COO as (edge_index[int64,(2,E)], edge_weight[float64,(E,)])")
    .def("__repr__", [](const Graph& g) {
      return "Graph(num_nodes=" + std::to_string(g.num_nodes()) +
             ", num_edges=" + std::to_string(g.num_edges()) +
             ", directed=" + std::string(g.directed() ? "True" : "False") + ")";
    });

  py::class_<GraphBuilder>(m, "GraphBuilder")
    .def(py::init<node_index_t, BuildOptions>(), 
         py::arg("num_nodes"), 
         py::arg("options") = BuildOptions{})
    .def("reserve_edges", &GraphBuilder::reserve_edges, py::arg("n"))
    .def("clear", &GraphBuilder::clear)
    .def("add_edge",
         py::overload_cast<node_index_t, node_index_t, weight_t>(&GraphBuilder::add_edge),
         py::arg("src"), py::arg("dst"), py::arg("weight") = 1.0)
    .def("add_edge",
         py::overload_cast<Edge>(&GraphBuilder::add_edge),
         py::arg("edge"))
    .def("add_edges", &GraphBuilder::add_edges, py::arg("edges"),
         "Add many edges from a list of Edge objects")
    .def("add_edges_numpy",
         [](GraphBuilder& b, py::array edge_index, py::object edge_weight = py::none()) {
           BuildOptions opt = b.options();
           Graph tmp = build_graph_from_numpy(b.num_nodes(), edge_index, edge_weight, opt);
           auto [ei, ew] = graph_to_coo_numpy(tmp);
           auto eip = ei.unchecked<2>();
           auto ewp = ew.unchecked<1>();
           const ssize_t E = ei.shape(1);
           b.reserve_edges(static_cast<size_t>(E));
           for (ssize_t i = 0; i < E; i++) {
             b.add_edge(static_cast<node_index_t>(eip(0, i)),
                        static_cast<node_index_t>(eip(1, i)),
                        static_cast<double>(ewp(i)));
           }
         },
         py::arg("edge_index"), py::arg("edge_weight") = py::none(),
         "Add edges from numpy edge_index (2,E or E,2) and optional edge_weight (E,)")
    .def("build", &GraphBuilder::build);

  m.def("build_graph",
        [](node_index_t num_nodes,
           py::array edge_index,
           py::object edge_weight,
           BuildOptions opt) {
          return build_graph_from_numpy(num_nodes, edge_index, edge_weight, opt);
        },
        py::arg("num_nodes"),
        py::arg("edge_index"),
        py::arg("edge_weight") = py::none(),
        py::arg("options") = BuildOptions{},
        "Build a Graph from COO numpy edge_index and optional edge_weight");
}

static py::array_t<int64_t> 
vec_u32_to_i64(const std::vector<uint32_t>& v)
{
  py::array_t<int64_t> out(v.size()); 
  auto r = out.mutable_unchecked<1>(); 
  for (ssize_t i = 0; i < r.shape(0); i++) {
    r(i) = static_cast<int64_t>(v[static_cast<size_t>(i)]);
  }
  return out; 
}

static std::pair<py::array_t<int64_t>, py::array_t<double>> 
graph_to_coo_numpy(const Graph& graph)
{
  auto [edges, weights] = graph.to_coo(); 
  const ssize_t E = static_cast<ssize_t>(edges.size()); 

  py::array_t<int64_t> edge_index(py::array::ShapeContainer{TWO_SSIZE, E}); 
  py::array_t<double> edge_weight(py::array::ShapeContainer{E}); 

  auto ei = edge_index.mutable_unchecked<2>(); 
  auto ew = edge_weight.mutable_unchecked<1>();

  for (ssize_t i = 0; i < E; i++) {
    ei(0, i) = static_cast<int64_t>(edges[static_cast<size_t>(i)].first); 
    ei(1, i) = static_cast<int64_t>(edges[static_cast<size_t>(i)].second); 
    ew(i) = weights[static_cast<size_t>(i)]; 
  }

  return {std::move(edge_index), std::move(edge_weight)}; 
}


static Graph 
build_graph_from_numpy(node_index_t num_nodes, py::array edge_index_any, 
                       py::object edge_weight_any, const BuildOptions& opt)
{
  auto edge_index = py::array_t<int64_t, py::array::c_style | 
    py::array::forcecast>(edge_index_any); 

  if ( edge_index.ndim() != 2 ) {
    throw std::invalid_argument("edge_index must be a 2d array of shape (2,E) or (E,2)"); 
  }

  const ssize_t d0 = edge_index.shape(0);
  const ssize_t d1 = edge_index.shape(1);
  const bool is_2xE = (d0 == 2);
  const bool is_Ex2 = (d1 == 2);

  if (!is_2xE && !is_Ex2) {
    throw std::invalid_argument("edge_index must have shape (2,E) or (E,2)");
  }

  const ssize_t E = is_2xE ? d1 : d0;

  py::array_t<double, py::array::c_style | py::array::forcecast> edge_weight;
  bool has_weight = !edge_weight_any.is_none();
  if (has_weight) {
    edge_weight = py::array_t<double, py::array::c_style | py::array::forcecast>(edge_weight_any);
    if (edge_weight.ndim() != 1 || edge_weight.shape(0) != E) {
      throw std::invalid_argument("edge_weight must be a 1D array of length E");
    }
  }

  GraphBuilder b(num_nodes, opt);
  b.reserve_edges(static_cast<size_t>(E));

  auto ei = edge_index.unchecked<2>();
  auto ew = [&](ssize_t i) -> double { return edge_weight.unchecked<1>()(i); };

  for (ssize_t i = 0; i < E; i++) {
    const std::int64_t src = is_2xE ? ei(0, i) : ei(i, 0);
    const std::int64_t dst = is_2xE ? ei(1, i) : ei(i, 1);
    const double w = has_weight ? ew(i) : 1.0;

    if (src < 0 || dst < 0) {
      throw std::out_of_range("edge_index contains negative node indices");
    }

    b.add_edge(static_cast<node_index_t>(src), static_cast<node_index_t>(dst), w);
  }

  return b.build();
}
