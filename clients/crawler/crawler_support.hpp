/*
 * crawler_support.hpp  Andrew Belles Nov 7th, 2025 
 * 
 * Defines the structures and detached helper functions that the  
 * Crawler Class requires
 *
 */ 

#ifndef __CRAWLER_SUPPORT_HPP
#define __CRAWLER_SUPPORT_HPP


#include <cstdint>  
#include <vector> 
#include <optional> 
#include <string> 
#include <chrono> 
#include <unordered_map>
#include <functional> 
#include <string_view> 
#include <stdexcept>

#include <boost/json.hpp>  

using millis = std::chrono::milliseconds;

/************ supporting structures ***********************/

namespace crwl {

struct PaginationConfig {
  enum class Mode : uint8_t {
     None, 
     PageNumber, 
     Offset, 
     Cursor 
  };

  Mode mode{Mode::None};
  std::string param_name;           // query parameter to advance pagination
  std::string next_token_field;     // JSON field carrying continuation token
  size_t page_size{0};              // preferred items per page; 0 = API default
  size_t max_pages{0};              // hard cap per cycle; 0 = unbounded
  millis page_delay{0};             // optional pause between pages
};

struct Field {
  std::string name;                 // canonical field name for downstream storage
  std::optional<std::string> alias; // optional override when API field differs
};

struct Endpoint {
  std::string path;                 // path appended to base URL
  std::vector<Field> fields;        // columns/items sourced from this endpoint
  PaginationConfig pagination;      // page-handling policy for the endpoint
};

// For state machine  
struct PaginationState {
  size_t page_index{0}; 
  size_t items_seen{0}; 
  std::optional<std::string> cursor{}; 
  bool exhausted{false};               
};

template <class Item> 
struct PageBatch {
  std::vector<std::vector<Item>> items; 
  bool has_more{false}; 
  std::optional<std::string> cursor{}; 
};

struct CrawlerMetadata {
  std::string api_key;              // Authorization header value
  std::string base_url;             // Base domain/schema for requests
  std::vector<Endpoint> endpoints;  // All endpoint definitions to crawl

  void 
  update_endpoints(std::vector<Endpoint> new_endpoints)
  {
    endpoints = std::move(new_endpoints);
  }
};

struct CrawlerConfig {
  size_t retries{5}; 
  std::chrono::seconds backoff{1}; 

  // ??? Anything else 
};

template <class Item> 
class Handlers {
private: 
  using MapType = std::unordered_map<std::string, std::vector<Item>>; 
  using MapIt   = typename MapType::const_iterator; 
// Local type aliases because json_handler has a long ass function prototype 
  using Fields = std::vector<Field>; 

public: 
  std::function<PageBatch<Item>(
     const boost::json::object&, const Fields&, const PaginationState&)> json_handler{};
  std::function<bool(MapIt, MapIt)> sqlite_handler{};
};

/************ build_url_ **********************************/
/* Using the current pagination state build a url string that 
 * accurately reflects the location of the resulting cursor from 
 * the previous request 
 *
 * Caller Provides: 
 *   The base string 
 *   Current endpoint 
 *   State of Pagination FSM 
 *
 * We return: 
 *   String formed to account for state of FSM 
 */ 
static std::string
build_url_(const std::string& base, const Endpoint& endpoint, 
           const PaginationState& state)
{
  std::string url = base;
  if ( url.empty() ) {
    throw std::runtime_error("crwl::build_url_: base_url is empty");
    return ""; 
  } // avoid empty base  

  // locate trailing slash, append endpoint path to url 
  if ( !endpoint.path.empty() ) {
    const bool base_has_slash = !url.empty() && url.back() == '/';
    const bool path_has_slash = endpoint.path.front() == '/';
    if ( base_has_slash && path_has_slash ) {
      url.pop_back();
    } else if ( !base_has_slash && !path_has_slash && endpoint.path.front() != '?' ) {
      url.push_back('/');
    }
    url += endpoint.path;
  }

  auto has_query = url.find('?') != std::string::npos;
  // lambda to help append a parameter safely to url string  
  const auto append_param = [&](std::string_view key, const std::string& value) {
    if ( key.empty() || value.empty() ) {
      return;
    }
    url.push_back(has_query ? '&' : '?');
    has_query = true;
    url.append(key.data(), key.size());
    url.push_back('=');
    url += value;
  };

  // depending on state of FSM, append different result to back of url 
  const auto& cfg = endpoint.pagination;
  switch ( cfg.mode ) {
    case crwl::PaginationConfig::Mode::PageNumber:
      if ( !cfg.param_name.empty() ) {
        append_param(cfg.param_name, std::to_string(state.page_index + 1));
      }
      break;

    case crwl::PaginationConfig::Mode::Offset:
      if ( !cfg.param_name.empty() ) {
        append_param(cfg.param_name, std::to_string(state.items_seen));
      }
      break;

    case crwl::PaginationConfig::Mode::Cursor:
      if ( state.cursor && !cfg.param_name.empty() ) {
        append_param(cfg.param_name, *state.cursor);
      }
      break;

    case crwl::PaginationConfig::Mode::None:
    default:
      break;
  }

  return url;
}

} // end namespace crwl 

#endif // !__CRAWLER_SUPPORT_HPP
