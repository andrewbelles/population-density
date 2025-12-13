/*
 * support.cpp  Andrew Belles  Dec 12th, 2025 
 *
 * Supporting Modules to C++ implementations
 *
 *
 */ 

#include <fstream> 
#include <sstream> 
#include <iostream> 
#include <optional> 
#include <vector> 
#include <string> 

#include "support.hpp"

static std::optional<County> parse_county(const std::vector<std::string>& fields,
                                          size_t line_num);


/************ load_counties() *****************************/ 
/* Loads county metadata from gazetteer dataset 
 *
 * Caller Provides: 
 *    Const reference to filepath 
 *
 * We Assume: 
 *    filepath is valid and is path to gazetteer tsv 
 *
 * We return: 
 *    Vector of county metadata  
 */  
std::vector<County> 
load_counties(const std::string& filepath)
{
  std::vector<County> counties; 
  std::ifstream file(filepath); 

  // ensure file exists 
  if ( !file.is_open() ) {
    std::cerr << "Failed to open file: " + filepath << '\n'; 
    return {};
  }

  // ensure header exists 
  std::string header, line; 
  if ( !std::getline(file, header) ) {
    std::cerr << "Failed to read header line\n";
    return {}; 
  } 

  // ensure tsv matches format we expect 
  if ( header.find("INTPTLAT") == std::string::npos || 
    header.find("INTPTLONG") == std::string::npos || 
    header.find("GEOID") == std::string::npos ) {
    
    std::cerr << "Header missing required columns: " + header << '\n'; 
    return {}; 
  }

  size_t line_num{1}; 

  // tsv parsing loop 
  while ( std::getline(file, line) ) {
    line_num++; 

    if ( line.empty() ) {
      std::cerr << "Empty line at " << line_num << '\n'; 
      continue; 
    }

    std::istringstream ss(line); 
    std::string token; 
    std::vector<std::string> fields; 

    // Parse all fields for current line 
    while ( std::getline(ss, token, '\t') ) {
      fields.push_back(token); 
    }

    // parse_county returns optional, ensure we have a county and push, or skip row 

    if ( auto county = parse_county(fields, line_num); county.has_value() ) {
      counties.push_back(county.value()); 
    } else {
      std::cerr << "Line at " << line_num << ": failed to parse. Continuing...";
      continue; 
    }
  }
  return counties; 
}

/************ parse_county() ******************************/ 
/* Parses a single county given the read tokens from a single line. Returns 
 * optional value 
 *
 * Caller Provides: 
 *    vector of string tokens representing fields of tsv 
 *    line_num parsing loop is currently on (1's indexed)
 *
 * We return: 
 *    std::nullopt for failure to parse the current county 
 *    the parsed county on success 
 */
static std::optional<County> 
parse_county(const std::vector<std::string>& fields, size_t line_num)
{
  if ( fields.size() < 10 ) {
    std::cerr << "Line " << line_num << ": Expected 10+ fields, got" << fields.size() << '\n'; 
    return std::nullopt; 
  }

  if ( fields[1].empty() || fields[8].empty() || fields[9].empty() ) {
    std::cerr << "Line " << line_num << ": Empty GEOID, lat, or lon\n";  
    return std::nullopt; 
  }

  try {
    
    County::Coordinate coord{std::stod(fields[8]), std::stod(fields[9])}; 
    County county{
      fields[0], // state 
      fields[3], // name 
      fields[1], // geoid  
      coord 
    };  

    // check if (lat, lon) is in CONUS  
    
    if ( auto [lat, _] = county.coord(); lat < 15.0 || lat > 72.0 ) {
      std::cerr << "Line " << line_num << ": Latitude " << lat << " outside US range\n"; 
      return std::nullopt; 
    }

    if ( auto [_, lon] = county.coord(); lon < -180.0 || lon > 180.0 ) {
      std::cerr << "Line " << line_num << ": Longitude " << lon << " outside US range\n"; 
      return std::nullopt; 
    }

    return county; 
  } catch ( const std::invalid_argument& e ) {
    std::cerr << "Line " << line_num << ": Invalid number format: " << e.what() << '\n'; 
    return std::nullopt; 
  } catch ( const std::out_of_range& e ) {
    std::cerr << "Line " << line_num << ": Number out of range: " << e.what() << '\n'; 
    return std::nullopt; 
  }
}
