/*
 *
 *
 *
 *
 *
 */ 

#pragma once 

#include <exception>
#include <functional> 
#include <stdexcept>
#include <string_view>
#include <vector> 
#include <thread> 

#include <boost/json/src.hpp>  
#include "http.hpp"
#include "json.hpp"

namespace crwl {

namespace json = boost::json; 

/*
 * Handler for breaking pulled json into a vector of Items (defined by template)
 *
 */ 
template <class Item> 
using JsonHandler = 
  std::function<std::vector<Item>(const json::object&, const std::vector<std::string>&)>;

/*
 *
 *
 */
template <class Item> 
class Crawler {
public:


private:
  std::string key, base;              // api key and base url to hit  
  std::vector<std::string> endpoints; // all endpoints to hit, 
  std::vector<std::string> fields;    // all fields to access
  JsonHandler<Item> json_handler;     // handles split of json object  
  size_t retries{5}; 

  /*
   * Executes a single request on the specified endpoint, converts into Item vector  
   * 
   * Caller Provides: 
   *   Endpoint of API to hit on 
   *
   * We return: 
   *   Vector of Items (specified by template) or an value error exception 
   */ 
  std::vector<Item>
  fetch(std::string_view endpoint) const 
  {
    const std::string url = base + std::string(endpoint); 
    try {
      // get body of request and parse into value  
      std::string body = request_with_retries_(url);
      const json::object& root = jsc::as_obj(jsc::parse(body)); 
      
      // convert into vector based on input fields 
      return json_handler(root, fields);
    } catch (...) {
      std::throw_with_nested(std::runtime_error("crwl::fetch failed: " + url)); 
    }
  }

  std::string 
  request_with_retries_(const std::string& url) const 
  {
    std::string current = url; 
    auto backoff = std::chrono::milliseconds(200);

    for (size_t attempt{1}; attempt <= retries; attempt++) {
      try {
        auto l_url = htc::parse_url(current); 
        auto [res, status] = htc::request(key, l_url);
        
        // implies a redirect 
        if ( status == 1 ) {
          current = res; 
          attempt -= 1; 
          continue; 
        } else if ( status == 0 ) {
          return std::move(res); 
        }
      } 
      catch (const std::exception& e) {
        if ( attempt == retries ) {
          throw std::runtime_error(
              std::string("failure to complete request: ") + e.what());
        }

        std::this_thread::sleep_for(backoff);
        backoff *= 2; 
        continue; 
      }
    }
  }
}; 

}
