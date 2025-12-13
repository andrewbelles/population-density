/*
 * support.hpp  Andrew Belles  Dec 12th, 2025 
 *
 * Public Interfaces for supporting C++ modules 
 *
 *
 */ 

#pragma once 

#include <vector> 
#include <string> 

class County {
public: 
  using Coordinate = std::pair<double, double>; 

  County() = default; 
  explicit County(const std::string& st, const std::string& name, const std::string& id, 
                  const Coordinate& loc)
    : state(st), name(name), geoid(id)
  {
    auto [latitude, longitude] = loc; 
    lat = latitude; 
    lon = longitude; 
  }

  Coordinate coord() const noexcept { return {lat, lon}; }
  std::string id() const noexcept { return geoid; }
private: 
  std::string state; 
  std::string name; 
  std::string geoid; 
  double lat{0.0}, lon{0.0}; 
};

std::vector<County> load_counties(const std::string& filepath);
