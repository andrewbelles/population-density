/*
 * climate.cpp  Andrew Belles  Nov 7th, 2025 
 *
 * Both an example of Crawler API and data collection of Climate Data 
 * from NOAA Climate Web Service for collection of medium resolution climate data 
 *
 */ 

#include <array>
#include <chrono>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>
#include <functional> 
#include <algorithm> 
#include <sstream> 

#include "crawler/crawler.hpp"
#include "crawler/database.hpp"
#include "crawler/crawler_support.hpp"

using BBox = std::tuple<double, double, double, double>; 

/************ SQL Queries *********************************/ 
static const std::string upsert_sql = 
R"(
INSERT INTO stations (
  station_id, 
  name, 
  state,
  latitude, 
  longitude, 
  elevation_m, 
  active_start, 
  active_end, 
  metadata, 
  last_ingested
) VALUES (?1, ?2, ?3, ?4, ?5, ?6, json(?7), CURRENT_TIMESTAMP)
  ON CONFLICT(station_id) DO UPDATE SET 
  name          = excluded.name,
  state         = excluded.state, 
  latitude      = excluded.latitude,
  longitude     = excluded.longitude,
  elevation_m   = excluded.elevation_m,
  active_start  = excluded.active_start, 
  active_end    = excluded.active_end, 
  metadata      = excluded.metadata, 
  last_ingested = excluded.last_ingested;
)"; 

/************ Local Data Structures ***********************/ 

/************ StationSeed *********************************/ 
/* Stores data about a single potential station 
 * to make a request at 
 */
struct StationSeed {
  std::string region_id; 
  BBox extent; 
  int limit{1000}; 
};

/************ StationConfig *******************************/
/* Stores data required for configuring Station Client at 
 * construction time 
 */
struct StationConfig {
  std::string base_url{"https://www.ncdc.noaa.gov/cdo-web/api/v2"};
  std::string api_key; 
  size_t max_queue{512}; 
  size_t retries{5}; 
  std::chrono::seconds backoff{1}; 
  millis page_delay{250};
};

/************ StationRecord *******************************/ 
/* Single Station Item to be upserted into climate.db 
 */ 
struct StationRecord {
  std::string station_id; 
  std::string name; 
  std::string state; 
  double latitude; 
  double longitude; 
  double elevation_m; 
  std::string active_start;
  std::string active_end; 
  boost::json::object metadata; 
};

/************ Type Bindings *******************************/ 
using StationMap = std::unordered_map<std::string, std::vector<StationRecord>>;
using StationIterator = StationMap::const_iterator; 
using Fields = std::vector<crwl::Field>;  

/************ Local helper functions **********************/ 
StationRecord parse_station_record(const boost::json::object& row);
crwl::PageBatch<StationRecord> parse_station_page(const boost::json::object& root,
                                                  const Fields& fields,
                                                  const crwl::PaginationState&);

/************ Primary Classes *****************************/ 

/************ ClimateDB ***********************************/ 
/* Child Class of SqliteDB which implements upsert method 
 * and sqlite_handler for climate.db  
 *
 * Finalizes abstract SqliteDB class 
 */ 
class ClimateDB final : public dat::SqliteDB<StationIterator> {
public: 
  using Base = dat::SqliteDB<StationIterator>; 

  // Explicit, Basic Constructor 
  ClimateDB(std::string&& path, 
                     int flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE)
    : Base(std::move(path), flags) {}

  ClimateDB(ClimateDB&&) noexcept = default;

  bool ingest(StationIterator first, StationIterator last)
  {
    typename Base::Tx tx(*this);
    try {
      sqlite_handler(first, last); 
      tx.commit(); 
      return true; 
    } catch (...) {
      tx.rollback();
      throw; 
      return false; 
    }
  }

protected: 
  /********** ClimateDB override : sqlite_handler ***********/ 
  /* Iterates over the Station Map, calling upsert_ for all 
   * non-empty entries. 
   *
   * Caller provides: 
   *   Iterator range from a StationMap object 
   */ 
  void sqlite_handler(StationIterator first, StationIterator last) override 
  {
    for (; first != last; first++) {
      const auto& item = first->second; 
      if ( item.empty() ) {
        continue; 
      }

      const auto& field = first->first; 
      if ( field == "stations" ) {
        upsert_stations_(item);
      }
    }
  }

private: 

  /********** ClimateDB::upsert_ ****************************/ 
  /* Upserts all records from given row into the climate.db  
   * under the stations schema (See climate_init.sql to see schema)
   */ 
  void upsert_stations_(const std::vector<StationRecord>& rows)
  {
    sqlite3_stmt* stmt = nullptr; 
    try {
      stmt = prepare(upsert_sql); 
      for (const auto& record : rows) {
        bind(stmt, 1, record.station_id);
        bind(stmt, 2, record.name); 
        bind(stmt, 3, record.state);

        auto bind_double = [&](int idx, double value) {
          const int code = sqlite3_bind_double(stmt, idx, value); 
          if ( code != SQLITE_OK ) {
            throw std::runtime_error("ClimateDB::upsert_ bind_double failed"); 
          }
        }; 

        bind_double(4, record.latitude); 
        bind_double(5, record.longitude); 
        bind_double(6, record.elevation_m); 
        auto bind_date = [&](int idx, const std::string& value) {
          if ( value.empty() ) {
            const int code = sqlite3_bind_null(stmt, idx); 
            if ( code != SQLITE_OK ) {
              throw std::runtime_error("ClimateDB::upsert_ bind_null failed");
            }
          } else {
            bind(stmt, idx, value); 
          }
        };

        bind_date(7, record.active_start); 
        bind_date(8, record.active_end); 

        const std::string metadata = boost::json::serialize(record.metadata); 
        bind(stmt, 9, metadata); 

        step(stmt); 
        sqlite3_reset(stmt); 
        sqlite3_clear_bindings(stmt); 
      }
    } catch (...) {
      finalize(stmt); 
      throw; 
    }

    finalize(stmt); 
  }
};

/************ StationClient *******************************/ 
/*
 */ 
class StationClient {
private: 
  using Runner = crwl::Runner<StationRecord, StationSeed>;
public: 
  
  StationClient(
      StationConfig cfg, 
      ClimateDB& db, 
      crwl::Handlers<StationRecord> handlers
  ) : cfg_(std::move(cfg)), 
      db_(db), 
    handlers_(std::move(handlers)), 
    runner_(make_runner_()) {} 

  ~StationClient(void)
  {
    request_stop(); 
    join();
  }
  
  /********** getters/setters *****************************/
  bool enqueue(StationSeed seed) { return runner_.enqueue(std::move(seed)); }
  size_t backlog(void) const noexcept { return runner_.backlog(); }
  Runner::Snapshot snapshot(void) const { return runner_.snapshot(); }

  void set_on_idle(std::function<void(void)> idle_callback)
  {
    runner_.set_on_idle(std::move(idle_callback));     
  }

  void set_on_error(std::function<void(const std::exception&)> error_callback)
  {
    runner_.set_on_error(std::move(error_callback));     
  }

  /********** wrapped calls to crwl::Runner ***************/
  void start(void) { runner_.start(); }
  void request_stop(void) { runner_.request_stop(); }
  void join(void) { runner_.join(); }

private: 
  StationConfig cfg_; 
  ClimateDB& db_; 
  crwl::Handlers<StationRecord> handlers_; 
  Runner runner_; 

  Runner make_runner_(void)
  {
    auto meta = make_metadata_();
    crwl::CrawlerConfig crawler_cfg{cfg_.retries, cfg_.backoff};
    crwl::Crawler<StationRecord> crawler(meta, handlers_, crawler_cfg);
    auto builder = [this](const StationSeed& seed) {
      return build_endpoints_(seed); 
    };
    return Runner(std::move(crawler), std::move(builder), cfg_.max_queue);
  }

  crwl::CrawlerMetadata make_metadata_(void) const 
  {
    crwl::CrawlerMetadata meta{}; 
    meta.api_key  = cfg_.api_key; 
    meta.base_url = cfg_.base_url; 
    return meta; 
  }

  std::vector<crwl::Endpoint> build_endpoints_(const StationSeed& seed) const 
  {
    crwl::Endpoint endpoint{}; 
    endpoint.path = build_station_path_(seed); 
    endpoint.fields.emplace_back(crwl::Field{"stations", std::nullopt});
    endpoint.pagination.mode       = crwl::PaginationConfig::Mode::Offset; 
    endpoint.pagination.param_name = "offset"; 
    endpoint.pagination.page_size  = static_cast<size_t>(std::max(seed.limit, 1));
    endpoint.pagination.page_delay = cfg_.page_delay; 

    std::vector<crwl::Endpoint> endpoints; 
    endpoints.push_back(std::move(endpoint));
    return endpoints; 
  }

  std::string build_station_path_(const StationSeed& seed) const 
  {
    std::ostringstream oss; 
    oss << "/stations?datasetid=GHCND"; 
    if ( !seed.region_id.empty() ) {
      oss << "&locationid=" << seed.region_id; 
    }

    const auto [south, west, north, east] = seed.extent; 
    oss << "&extent=" << south << ',' << west << ',' << north << ',' << east; 

    const int limit = seed.limit > 0? seed.limit : 1000; 
    oss << "&limit=" << limit; 

    return oss.str(); 
  }
}; 

crwl::Handlers<StationRecord> make_station_handler(ClimateDB& db)
{
  crwl::Handlers<StationRecord> handlers{}; 
  handlers.json_handler = [](const boost::json::object& root, 
                             const Fields& fields,
                             const crwl::PaginationState& state) {
    return parse_station_page(root, fields, state);
  };

  handlers.sqlite_handler = [&db](StationIterator first, 
                                  StationIterator last) -> bool {
    return db.ingest(first, last); 
  };

  return handlers; 
}

crwl::PageBatch<StationRecord> 
parse_station_page(const boost::json::object& root, const Fields& fields,
                   const crwl::PaginationState&)
{
  crwl::PageBatch<StationRecord> batch{}; 
  batch.items.resize(fields.size()); 

  const auto* results = root.if_contains("results");
  if ( !results || !results->is_array() ) {
    return batch; 
  }

  std::vector<StationRecord> parsed; 
  parsed.reserve(results->as_array().size()); 
  for (const auto& entry : results->as_array()) {
    if ( entry.is_object() ) {
      parsed.emplace_back(parse_station_record(entry.as_object()));
    }
  }

  for (size_t i = 0; i < fields.size(); i++) {
    if ( fields[i].name == "stations" ) {
      batch.items[i] = parsed; 
    }
  }

  if ( const auto* metadata = root.if_contains("metadata"); 
       metadata && metadata->is_object() ) {
    
    if ( const auto* resultset = metadata->as_object().if_contains("resultset");
         resultset && resultset->is_object() ) {

      const auto& rs =resultset->as_object(); 
      const auto offset = static_cast<size_t>(
        std::max<int64_t>(0, jsc::get_or<int64_t>(rs, "offset").value_or(0))
      ); 
      const auto limit  = static_cast<size_t>(
        std::max<int64_t>(0, jsc::get_or<int64_t>(rs, "limit").value_or(0))
      );
      const auto count  = static_cast<size_t>(
        std::max<int64_t>(0, jsc::get_or<int64_t>(rs, "count").value_or(0))
      );

      batch.has_more = (offset + limit) < count; 
    }
  }
  return batch; 
}

StationRecord 
parse_station_record(const boost::json::object& row)
{
  StationRecord rec{}; 
  rec.station_id   = boost::json::value_to<std::string>(row.at("id"));
  rec.name         = jsc::get_or<std::string>(row, "name").value_or("");
  rec.state        = jsc::get_or<std::string>(row, "state").value_or("");
  rec.latitude     = jsc::get_or<double>(row, "latitude").value_or(0.0); 
  rec.longitude    = jsc::get_or<double>(row, "longitude").value_or(0.0); 
  rec.elevation_m  = jsc::get_or<double>(row, "elevation").value_or(0.0); 
  rec.active_start = jsc::get_or<std::string>(row, "mindate").value_or(""); 
  rec.active_end   = jsc::get_or<std::string>(row, "maxdate").value_or("");

  rec.metadata = boost::json::object{}; 
  constexpr std::array<std::string_view, 6> pass = {
    "datacoverage", "elevationUnit", "network",
    "gsnFlag", "hcnFlag", "wmoID"
  };

  for (auto& key : pass) {
    if ( const auto* value = row.if_contains(key) ) {
      rec.metadata.emplace(std::string(key), *value); 
    }
  }
  return rec;
}
