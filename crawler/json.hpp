/*
 *
 *
 *
 *
 *
 */ 

#pragma once 

#include <boost/json.hpp>
#include <stdexcept>
#include <string> 
#include <string_view> 

namespace jsc {

namespace json = boost::json; 

inline json::value 
parse(std::string_view body)
{
  boost::system::error_code err; 
  json::value val = json::parse(body, err); 
  if ( err ) {
    throw std::runtime_error("JSON parse error: " + err.message());
  }
  return val; 
}

inline const json::object& 
as_obj(const json::value& v)
{
  if ( !v.is_object() ) {
    throw std::runtime_error("expected object");
  } else {
    return v.as_object();
  }
}

inline const json::array& 
as_arr(const json::value& v)
{
  if ( !v.is_array() ) {
    throw std::runtime_error("expected object");
  } else {
    return v.as_array();
  }
}

template <class T>
std::optional<T> get_or(const json::object& obj, std::string_view key)
{
  if ( auto* p = obj.if_contains(key) ) {
    if ( !p->is_null() ) {
      return json::value_to<T>(*p); 
    }
  }
  return std::nullopt;  
} 

}
