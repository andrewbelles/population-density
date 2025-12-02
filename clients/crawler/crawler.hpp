/*
 * crawler.hpp  Andrew Belles  Nov 6th, 2025 
 *
 * Base Crawler. Implements request functionality and allows for custom handlers  
 * Which provide a layer of abstraction for both parsing requested json from some 
 * api and also handling insertion into an sqlite database 
 */ 

#ifndef __CRAWLER_HPP 
#define __CRAWLER_HPP

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <deque>
#include <exception>
#include <functional>
#include <mutex>
#include <optional>
#include <semaphore>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <thread> 
#include <unordered_map>
#include <utility>
#include <vector> 

#include <boost/json.hpp>  
#include "crawler_support.hpp"
#include "http.hpp"
#include "json.hpp"

namespace crwl {


/************ crwl::Crawler *******************************/
/* Crawler Interface
 *
 * Templates: 
 *   Item: A struct that contains a single JSON block from whatever API hit. 
 *
 * TODO: 
 *   - Introduce trait bounds on Item such that we guarantee the json and sqlite 
 *     handlers have deterministic behavior when acting on Item 
 */
template <class Item> 
class Crawler {
public:

  Crawler(
    CrawlerMetadata& m,
    Handlers<Item>& h, 
    CrawlerConfig& cfg 
  ) : meta_(std::move(m)), fns_(std::move(h)), cfg_(std::move(cfg)) {} 

  /********** setters *************************************/ 
  void set_metadata(CrawlerMetadata meta) { meta_ = std::move(meta); }
  void set_endpoints(std::vector<Endpoint> endpoints)
  {
    meta_.update_endpoints(std::move(endpoints));
  }
  void set_api_key(std::string api_key) { meta_.api_key = std::move(api_key); }

  /********** getters *************************************/ 
  const CrawlerMetadata& metadata() const noexcept { return meta_; }

  /********** crwl::cycle() *******************************/ 
  /* Fetches data over all endpoints into map 
   * Sends range of map into sqlite_handler to parse into database 
   */
  void cycle() 
  { 
    item_map_.clear(); 
    try {
      // For each endpoint fetch 
      for (auto& endpoint : meta_.endpoints) {
        drain_endpoint_(endpoint);
      }
    } catch (const std::exception& e) {
      throw std::runtime_error(std::string("crwl::cycle: fetch choked: ") + e.what());
    }
    
    // Call over range of elements in map from previous fetch 
    fns_.sqlite_handler(item_map_.cbegin(), item_map_.cend());
  }

private:
  std::unordered_map<std::string, std::vector<Item>> item_map_{};
  CrawlerMetadata meta_{};
  Handlers<Item> fns_{}; 
  CrawlerConfig cfg_{};

  /********** drain_endpoint_() ***************************/
  /* For each endpoint, make request and drain the endpoint until PaginationState 
   * reaches end state. 
   *
   * Crawler Provides: 
   *   the individual endpoint that should be requested at 
   *
   * Crawler modifies: 
   *   Item_map has items in batch insterted per field 
   */
  void 
  drain_endpoint_(const Endpoint& endpoint)
  {
    PaginationState state{}; 
    do {
      auto batch = fetch_(endpoint, state);
      append_batch_(endpoint.fields, batch); 
      update_state_(endpoint.pagination, batch, state);

      if ( !state.exhausted && endpoint.pagination.page_delay.count() > 0 ) {
        std::this_thread::sleep_for(endpoint.pagination.page_delay);
      }
    } while ( !state.exhausted ); 
  }

  /********** fetch() *************************************/
  /* Executes a single request on the specified endpoint, handles Pagination through
   * a FSM stored in PaginationState struct 
   * 
   * Crawler Provides: 
   *   Endpoint of API to hit on 
   *   Current state of Paging FSM 
   *
   * Returns to Crawler: 
   *   Returns a PageBatch struct to be handled within drain_endpoint_
   */ 
  PageBatch<Item>
  fetch_(const Endpoint& endpoint, const PaginationState& state) 
  {
    const std::string url = build_url_(meta_.base_url, endpoint, state); 
    try {
      // get body of request and parse into value  
      std::string body = request_with_retries_(url);
      const boost::json::object& root = jsc::as_obj(jsc::parse(body)); 

      auto batch = fns_.json_handler(root, endpoint.fields, state); 
      if ( endpoint.fields.size() != batch.items.size() ) {
        throw std::runtime_error("resulting item vector is too small");
      }
      return batch; 
    } catch (...) {
      std::throw_with_nested(std::runtime_error("crwl::fetch failed: " + url)); 
    }
  }

  /********** crwl::request_with_retries_() ***************/ 
  /* Makes requests at the built url for specified number of retries 
   * 
   * Caller Provides: 
   *   String representing the url to be made and requested at 
   *
   * We return: 
   *   The json returned as a string (or "" and exception for error)
   */
  std::string 
  request_with_retries_(const std::string& url) const 
  {
    std::string current = url; 
    auto backoff = cfg_.backoff; 

    // Limit to some maximum number of retries
    for (size_t attempt{1}; attempt <= cfg_.retries; attempt++) {
      // Exception is for is_retryable or critical error 
      try {
        auto l_url = htc::parse_url(current); 
        auto [res, status] = htc::request(meta_.api_key, l_url);
        
        // implies a redirect 
        if ( status == 1 ) {
          current = res; 
          attempt -= 1; 
          continue; 
        } else if ( status == 0 ) {
          return std::move(res); 
        }
      } catch (const std::exception& e) {
        if ( attempt == cfg_.retries ) {
          throw std::runtime_error(
              std::string("failure to complete request: ") + e.what());
        }

        std::this_thread::sleep_for(backoff);
        backoff *= 2; 
        continue; 
      }
    }

    throw std::runtime_error("request_with_retries_: exhausted attemps w/o resp");
  }

  /********** append_batch_() *****************************/ 
  /* Appends an entire batch onto the item_map using the fields associated with
   * the current endpoint. 
   * 
   * Does so safely making the assumption that fields.size() != batch.size() 
   * necessarily 
   *
   * Caller Provides: 
   *   Field vector and current batch from page 
   *
   * We modify: 
   *   Item_map has elements inserted for each field that can be associated with 
   *   an element from the PageBatch 
   */  
  void 
  append_batch_(const std::vector<Field>& fields, const PageBatch<Item>& batch)
  {
    if ( fields.empty() || batch.items.empty() ) {
      return;
    }

    const size_t columns = std::min(fields.size(), batch.items.size());
    for (size_t i{0}; i < columns; i++) {
      const auto& field = fields[i];
      const auto& chunk = batch.items[i];
      auto& bucket = item_map_[field.alias.value_or(field.name)];
      bucket.insert(bucket.end(), chunk.begin(), chunk.end());
    }
  }

  /********** update_state_() *****************************/ 
  /* Updates the state of the PaginationState FSM depending on the previous 
   * batch. Uses metadata stored within the PageBatch struct to make determinations 
   * on how FSM should be updated 
   *
   * Caller Provides: 
   *   PaginationConfig 
   *   Last PageBatch 
   *   Current PaginationState
   *
   * We modify: 
   *   PaginationState to update into next state 
   */ 
  void 
  update_state_(const PaginationConfig& cfg, const PageBatch<Item>& batch,
                PaginationState& state) 
  {
    size_t rows = 0;
    for (const auto& column : batch.items) {
      rows = std::max(rows, column.size());
    }

    // update counters, move cursor
    state.items_seen += rows;
    state.page_index += 1;
    state.cursor      = batch.cursor; 

    // check all possible flags
    const bool reached_max = cfg.max_pages > 0 && state.page_index >= cfg.max_pages;
    const bool pagination_disabled = cfg.mode == PaginationConfig::Mode::None;
    const bool cursor_missing = (cfg.mode == PaginationConfig::Mode::Cursor) &&
                                batch.has_more && !batch.cursor.has_value();
    const bool no_more = pagination_disabled ? true : !batch.has_more;

    // move to end state if any flag is raised 
    state.exhausted = reached_max || cursor_missing || no_more;
  }
}; 

template <class Item, class Seed> 
class Runner {
public: 
  struct Snapshot {
    size_t queued{0}; 
    size_t processed{0}; 
    bool running{false}; 
    std::chrono::system_clock::time_point last_activity{}; 
  };

  using IntoEndpoint = std::function<std::vector<Endpoint>(const Seed&)>; 

  Runner(Crawler<Item> crawler, IntoEndpoint builder, size_t max_queue=512)
    : crawler_(std::move(crawler)), 
      builder_(std::move(builder)), 
      max_queue_(max_queue) 
  {
    if ( max_queue_ == 0 ) {
      throw std::invalid_argument("crwl::Runner max_queue must be greater than zero");
    }

    if ( !builder_ ) {
      throw std::invalid_argument("crwl::Runner requires valid seed to endpoint func");
    }
  }

  void set_on_idle(std::function<void()> idle_callback) 
  {
    on_idle_ = std::move(idle_callback); 
  }

  void set_on_error(std::function<void(const std::exception&)> error_callback) 
  {
    on_error_ = std::move(error_callback); 
  }

  void start(void) 
  {
    bool expected = false; 
    if ( !running_.compare_exchange_strong(expected, true) ) {
      return; 
    }

    stop_requested_.store(false, std::memory_order_relaxed); 
    worker_ = std::jthread([this](std::stop_token token) { worker_loop_(token); });
  }

  void request_stop(void)
  {
    stop_requested_.store(true, std::memory_order_relaxed);
    if ( worker_.joinable() ) {
      worker_.request_stop();
    }
    gate_.release(); 
  }

  void join(void)
  {
    if ( worker_.joinable() ) {
      worker_.join(); 
    }
    running_.store(false, std::memory_order_relaxed); 
  }

  bool enqueue(Seed seed)
  {
    if ( stop_requested_.load(std::memory_order_relaxed) ) {
      return false; 
    }

    {
      std::lock_guard<std::mutex> lock(queue_mu_);
      if ( queue_.size() >= max_queue_ ) {
        return false; 
      }
      queue_.push_back(std::move(seed)); 
    }

    gate_.release();
    return true; 
  }

  size_t backlog(void) const noexcept 
  {
    std::lock_guard<std::mutex> lock(queue_mu_); 
    return queue_.size(); 
  }

  Snapshot snapshot(void) const 
  {
    Snapshot snap{}; 
    {
      std::lock_guard<std::mutex> lock(queue_mu_); 
      snap.queued = queue_.size(); 
    }
    snap.processed = processed_.load(std::memory_order_relaxed); 
    snap.running   = running_.load(std::memory_order_relaxed); 
    {
      std::lock_guard<std::mutex> lock(activity_mu_); 
      snap.last_activity = last_activity_; 
    }
    return snap; 
  }

private: 
  using clock = std::chrono::system_clock; 

  Crawler<Item> crawler_; 
  IntoEndpoint builder_; 
  size_t max_queue_; 

  mutable std::mutex queue_mu_; 
  mutable std::mutex activity_mu_; 
  std::deque<Seed> queue_; 
  std::counting_semaphore<> gate_{0}; 
  std::jthread worker_; 

  std::atomic<bool> running_{false}; 
  std::atomic<bool> stop_requested_{false}; 
  std::atomic<size_t> processed_{0}; 
  std::atomic<size_t> active_{0}; 

  clock::time_point last_activity_{clock::now()}; 

  std::function<void(void)> on_idle_{};
  std::function<void(const std::exception&)> on_error_{};

  void worker_loop_(std::stop_token token)
  {
    while ( true ) {
      gate_.acquire(); 
      
      if ( token.stop_requested() && queue_empty_() ) {
        break; 
      }

      auto seed = pop_seed_(); 
      if ( !seed ) {
        if ( token.stop_requested() ) {
          break;
        }
        continue; 
      }

      active_.fetch_add(1, std::memory_order_relaxed); 
      try {
        process_seed_(*seed, token);
      } catch (const std::exception& e) {
        if ( on_error_ ) {
          on_error_(e);
        }
      }

      active_.fetch_sub(1, std::memory_order_relaxed); 
      processed_.fetch_add(1, std::memory_order_relaxed); 
      update_last_activity_();
      notify_idle_if_needed_();
    }
  }

  void process_seed_(const Seed& seed, std::stop_token token)
  {
    if ( token.stop_requested() ) {
      return; 
    }

    auto endpoints = builder_(seed); 
    if ( endpoints.empty() ) {
      return; 
    }

    crawler_.set_endpoints(std::move(endpoints));
    crawler_.cycle(); 
  }

  std::optional<Seed> pop_seed_(void)
  {
    std::lock_guard<std::mutex> lock(queue_mu_);
    if ( queue_.empty() ) {
      return std::nullopt; 
    }

    Seed seed = std::move(queue_.front());
    queue_.pop_front();
    return seed; 
  }

  bool queue_empty_(void) const 
  {
    std::lock_guard<std::mutex> lock(queue_mu_);
    return queue_.empty();
  }

  void update_last_activity_(void)
  {
    std::lock_guard<std::mutex> lock(activity_mu_); 
    last_activity_ = clock::now(); 
  }

  void notify_idle_if_needed_(void)
  {
    if ( !on_idle_ ) {
      return; 
    } 

    if ( queue_empty_() && !active_.load(std::memory_order_relaxed) ) {
      on_idle_();
    }
  }
};

} // end namespace crwl 

#endif // !__CRAWLER_HPP 
