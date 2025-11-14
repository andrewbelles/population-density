/*
 * http.hpp  Andrew Belles  Nov 7th, 2025 
 * 
 * Provides Interface for exposed http helper functions for Crawler 
 *
 *
 */ 

#ifndef __HTTP_HPP
#define __HTTP_HPP

#include <string_view> 
#include <string> 

namespace htc {

struct Url {
  std::string scheme, host, port, target;
};


/************ parse_url() *********************************/ 
/*
 *
 */ 
Url parse_url(std::string_view url);

/************ request() ***********************************/ 
/*
 *
 */ 
std::pair<std::string, int> request(const std::string& key, const Url& url);

}

#endif // !__HTTP_HPP
