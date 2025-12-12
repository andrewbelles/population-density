/*
 *
 *
 *
 *
 *
 */ 

#include <vector> 
#include <string> 

class County {
public: 
  using Coordinate = std::pair<double, double>; 

  explicit County(const std::string& st, const std::string& name, const std::string& id, 
                  const Coordinate& c)
    : state(st), name(name), geoid(id)
  {
    auto [latitude, longitude] = c; 
    lat = latitude; 
    lon = longitude; 
  }

  Coordinate coord() const noexcept { return {lat, lon}; }
private: 
  std::string geoid{}; 
  std::string name{}; 
  std::string state{}; 
  double lat{0.0}, lon{0.0}; 
};

std::vector<County> load_counties(const std::string& filepath);
