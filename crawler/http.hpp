
#pragma once 

#include <string_view> 

namespace htc {

struct Url {}; 

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
