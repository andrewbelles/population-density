/*
 * http.cpp  Andrew Belles  Nov 7th, 2025 
 *
 * Implementation of request function that Crawler wraps over with 
 * Retry. Provides dynamic interface which can connect to https or http  
 * connections properly. Robust to bad http responses 
 */ 

#pragma once 

#include "http.hpp"

#include <boost/asio.hpp> 
#include <boost/asio/io_context.hpp>
#include <boost/asio/ssl.hpp> 
#include <boost/asio/ssl/error.hpp>
#include <boost/beast/core.hpp> 
#include <boost/beast/core/stream_traits.hpp>
#include <boost/beast/http.hpp> 
#include <boost/beast/ssl.hpp> 
#include <boost/beast/version.hpp>

#include <chrono> 
#include <exception>
#include <stdexcept>

namespace {

/*
 * General checks against status I don't think should be usable outside scope  
 */
/************ is_redirect() ********************************/ 
/*
 * Confirms if a given status implies a redirect 
 */  
static inline bool 
is_redirect(uint32_t status)
{
  return status == 301 || status == 302 || status == 303 || 
         status == 307 || status == 308; {
  }; 
}

/************ is_retryable() ********************************/ 
/*
 * Confirms if a given status can be retried  
 */  
static inline bool 
is_retryable(uint32_t status)
{
  return status == 429 || (status >= 500 && status <= 599);
}

namespace asio  = boost::asio; 
namespace ssl   = asio::ssl; 
namespace beast = boost::beast;  
namespace http  = beast::http; 

}

/************ http for crawler ****************************/ 
namespace htc {

static constexpr auto timeout = std::chrono::seconds(10);
static const std::string bearer = "Bearer "; 

Url 
parse_url(std::string_view url)
{
  // lambda helper to throw exception on parse failure  
  auto bad = [&url]{
    throw std::invalid_argument("invalid URL: " + std::string(url));
  }; 

  const auto scheme_position = url.find("://");
  if ( scheme_position == std::string_view::npos ) {
    bad(); 
    return {};
  }

  Url res{}; 
  res.scheme = std::string(url.substr(0, scheme_position));
  auto trunc = url.substr(scheme_position + 3); 
  
  auto slash_position = trunc.find('/');
  bool slash_last = slash_position == std::string_view::npos; 
  std::string_view host_port = slash_last? trunc : trunc.substr(0, slash_position);
  res.target = slash_last? "/" : std::string(trunc.substr(slash_position));

  auto colon_position = host_port.find(':'); 
  if ( colon_position == std::string_view::npos ) {
    res.host = std::string(host_port); 
    res.port = res.scheme == "https"? "443" : "80";
  } else {
    res.host = std::string(host_port.substr(0, colon_position)); 
    res.port = std::string(host_port.substr(colon_position + 1));
  }

  if ( res.host.empty() ) {
    bad(); 
    return {};
  } else {
    return res; 
  }
}

static inline std::string 
auth(const std::string& key)
{
  if ( key.rfind(bearer, 0) == 0 ) {
    return key; 
  } else {
    return bearer + key; 
  }
}

http::request<http::string_body> 
build_request(const std::string& key, const Url& url)
{
  http::request<http::string_body> req{http::verb::get, url.target, 11}; 
  req.set(http::field::host, url.host); 
  req.set(http::field::user_agent, "crwl/1.0");

  if ( !key.empty() ) {
    req.set(http::field::authorization, auth(key));
  }

  req.set(http::field::accept, "application/json");
  req.set(http::field::connection, "close"); 
  return req; 
}

std::pair<std::string, int> 
handle_status(const http::response<http::string_body>& res)
{
  // check status code and throw appropriate errors 
  if ( is_redirect(res.result_int()) ) {
    if ( auto it = res.find(http::field::location); it != res.end() ) {
      return {it->value(), 1}; // redirect should unwrap on none  
    }
  }

  if ( res.result() != http::status::ok ) {
    auto res_int = std::to_string(res.result_int());
    if ( is_retryable(res.result_int()) ) {
      throw std::runtime_error("retryable HTTP status " + res_int);
    }
    throw std::runtime_error("HTTP error " + res_int);
    return {"", 2}; 
  }

  return {"", 0}; 
}

std::string 
handle_redirect(const Url& url, std::string_view loc)
{
  if ( loc.find("http://") == 0 || loc.find("https://") == 0 ) {
    return std::string(loc); 
  }

  if ( !loc.empty() && loc.front() == '/') {
    return url.scheme + "://" + url.host + ":" + url.port + std::string(loc);  
  } 

  auto path_end = url.target.rfind('/');
  bool is_end = path_end == std::string::npos;
  std::string dir = is_end? "/" : url.target.substr(0, path_end + 1); 
  return url.scheme + "://" + url.host + ":" + url.port + dir + std::string(loc);
}

static std::pair<std::string, int>
https_request_(const std::string& key, const Url& url, asio::io_context& ioc)
{
  asio::ip::tcp::resolver resolver{ioc}; 
  const auto results = resolver.resolve(url.host, url.port); 

  ssl::context ctx{ssl::context::tls_client};
  ctx.set_default_verify_paths(); 
  ctx.set_verify_mode(ssl::verify_peer);
  beast::ssl_stream<beast::tcp_stream> stream{ioc, ctx}; 

  // throw error on failure to setup SNI for SSL prior to TLS handshake  
  if ( !SSL_set_tlsext_host_name(stream.native_handle(), url.host.c_str()) ) {
    throw beast::system_error(
      beast::error_code(
        static_cast<int>(::ERR_get_error()), 
        asio::error::get_ssl_category()
      )
    );
  }

  // perform ssl handshake 
  beast::get_lowest_layer(stream).expires_after(timeout); 
  beast::get_lowest_layer(stream).connect(results); 
  stream.handshake(ssl::stream_base::client);

  // carry out request close, stream  
  auto req = build_request(key, url);
  http::write(stream, req); 

  beast::flat_buffer bufr; 
  http::response<http::string_body> res; 
  http::read(stream, bufr, res); 

  beast::error_code err; 
  stream.shutdown(err);

  auto [loc, status] = handle_status(res); 
  if ( status == 1 ) {
    return {handle_redirect(url, loc), 1};
  } else if ( status == 2 ) {
    std::throw_with_nested(std::runtime_error("HTTP request failure"));
    return {"", 2};
  } 

  return {std::move(res.body()), 0};
}

static std::pair<std::string, int>
http_request_(const std::string& key, const Url& url, asio::io_context& ioc)
{
  asio::ip::tcp::resolver resolver{ioc}; 
  const auto results = resolver.resolve(url.host, url.port); 

  beast::tcp_stream stream{ioc}; 
  stream.expires_after(timeout); 
  stream.connect(results); 

  auto req = build_request(key, url);
  http::write(stream, req); 

  beast::flat_buffer bufr; 
  http::response<http::string_body> res; 
  http::read(stream, bufr, res); 

  beast::error_code err; 
  stream.socket().shutdown(asio::ip::tcp::socket::shutdown_both, err);

  auto [loc, status] = handle_status(res); 
  if ( status == 1 ) {
    return {handle_redirect(url, loc), 1};
  } else if ( status == 2 ) {
    std::throw_with_nested(std::runtime_error("HTTP request failure"));
    return {"", 2};
  } 

  return {std::move(res.body()), 0};
}


/* Differences from proposed implementation 
 * Url is already parsed
 * Assume url is valid 
 */ 
std::pair<std::string, int> 
request(const std::string& key, const Url& url) 
{
  asio::io_context ioc; 
  if ( url.scheme == "https" ) {
    return std::move(https_request_(key, url, ioc)); 
  } else if ( url.scheme == "http" ) {
    return std::move(http_request_(key, url, ioc));
  } else {
    throw std::invalid_argument("Url was malformed, choked on invalid scheme");
  }
  return {"", 3}; // invalid url  
}

}
