/*
 *
 *
 *
 *
 *
 */ 

#pragma once 

#include <string_view> 
#include <string> 
#include <memory>
#include <stdexcept>
// #include <functional> 
#include <sqlite3.h>

/************ Sqlite Deleter ******************************/ 
/* Simple deleter struct to safely close a unique_ptr to sqlite3* 
 */
struct SqliteDeleter {
  void operator()(sqlite3* db) const noexcept
  {
    if ( db ) {
      sqlite3_close(db); 
    }
  }
};

/************ SqliteDB ************************************/
/* Parent Abstract Class for connecting API Clients to 
 * their respective Databases  
 */
template <class MapIt> 
class SqliteDB {
public:

  /********** Tx ******************************************/ 
  /* Transaction class that holds context of a sqlite3 transaction 
   */ 
  class Tx {
  public: 
    explicit Tx(SqliteDB& db) : db_(&db), active_(false) 
    {
      std::string_view sql = "BEGIN TRANSACTION"; 
      db_->exec(sql); 
      active_ = true; 
    }

    void commit()
    {
      if ( !active_ ) {
        return; 
      }

      if ( db_ == nullptr ) {
        throw std::runtime_error("SqliteDB::Tx::commit: nullptr database handle"); 
      }

      std::string_view sql = "COMMIT"; 
      db_->exec(sql); 
      active_ = false; 
    }

    void rollback()
    {
      if ( !active_ ) {
        return; 
      }

      if ( db_ == nullptr ) {
        throw std::runtime_error("SqliteDB::Tx::rollback: nullptr database handle"); 
      }

      std::string_view sql = "ROLLBACK"; 
      db_->exec(sql); 
      active_ = false; 
    }

  private: 
    SqliteDB* db_; 
    bool active_; 
  };
  
  /********** SqliteDB Constructor ************************/ 
  /*
   * Explicit Constructor for Database RAII wrapper over sqlite3 database 
   * 
   * Throws: 
   *   Runtime error for failure to open sqlite connection 
   */ 
  explicit SqliteDB(std::string&& path, 
                   int flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE)
  {
    sqlite3* sql_raw = nullptr; 
    
    const int code = sqlite3_open_v2(path.c_str(), &sql_raw, flags, nullptr);
    if ( code != SQLITE_OK ) {
      std::string err = sql_raw? sqlite3_errmsg(sql_raw) 
        : ("sqlite open error: " + std::to_string(code)); 
      if ( sql_raw ) {
        sqlite3_close(sql_raw); 
      }
      throw std::runtime_error("SqliteDB construction: " + err); 
    }
    db_.reset(sql_raw); 
  }

  virtual ~SqliteDB() = default; // leave destructor implementation to child 

  // Mutable reference to unique database pointer 
  std::unique_ptr<sqlite3, SqliteDeleter>& ptr() noexcept { return db_; }
  
  /********** exec() **************************************/ 
  /* Executes a single query on the intialized database.
   *
   * Throws: 
   *   runtime error for nullptr database uniq ptr
   *   runtime error for failure to execute query on database  
   */ 
  void exec(std::string_view sql) 
  {
    if ( !db_ ) {
      throw std::runtime_error("SqliteDB::exec database handle is nullptr"); 
    }

    char* errmsg = nullptr; 
    const int bfr_size = static_cast<int>(sql.size()); 
    const int code = sqlite3_exec(db_.get(), sql.data(), nullptr, nullptr, &errmsg); 
    if ( code != SQLITE_OK ) {
      std::string err = errmsg? std::string(errmsg) : sqlite3_errmsg(db_.get()); 

      if ( errmsg ) {
        sqlite3_free(errmsg); 
      } 
      throw std::runtime_error("SqliteDB::exec failed: " + err);
    }
  }

  /********** exec() **************************************/ 
  /* Prepares an sql statement. Returns a raw pointer to the statement.
   *
   * Throws: 
   *   runtime error for nullptr database uniq ptr
   *   runtime error for failure to execute query on database  
   */ 
  sqlite3_stmt* prepare(std::string_view sql)
  {
    if ( !db_ ) {
      throw std::runtime_error("SqliteDB::prepare database handle is nullptr"); 
    }
    
    sqlite3_stmt* stmt = nullptr; 
    const int code = sqlite3_prepare_v2(
      db_.get(), sql.data(), static_cast<int>(sql.size()), &stmt, nullptr
    ); 

    if ( code != SQLITE_OK ) {
      std::string errmsg{sqlite3_errmsg(db_.get())}; 
      if ( stmt ) {
        sqlite3_finalize(stmt); 
      }
      throw std::runtime_error("SqliteDB::prepare failed: " + errmsg); 
    }
    return stmt; 
  }

  /********** finalize() **********************************/ 
  /* Finalizes a given pointer to statement 
   */ 
  void finalize(sqlite3_stmt* stmt) noexcept
  {
    if ( stmt ) {
      sqlite3_finalize(stmt); 
    }
  }

protected: 
  virtual void sqlite_handler(MapIt first, MapIt last) = 0;// 
  
  void bind(sqlite3_stmt* stmt, int idx, std::string_view value)
  {
    if ( !db_ ) {
      throw std::runtime_error("SqliteDB::bind database handle is nullptr"); 
    }

    if ( stmt == nullptr ) {
      throw std::runtime_error("SqliteDB::bind statement is nullptr"); 
    }

    const int code = sqlite3_bind_text(stmt, idx, value.data(),
        static_cast<int>(value.size()), SQLITE_TRANSIENT);
    if ( code != SQLITE_OK ) {
      std::string errmsg{sqlite3_errmsg(db_.get())}; 
      throw std::runtime_error("SqliteDB::bind failed: " + errmsg);
    }

  }

  bool step(sqlite3_stmt* stmt)
  {
    if ( !db_ ) {
      throw std::runtime_error("SqliteDB::step database handle is nullptr"); 
    }

    if ( stmt == nullptr ) {
      throw std::runtime_error("SqliteDB::step statement is nullptr"); 
    }

    const int code = sqlite3_step(stmt); 
    if ( code == SQLITE_ROW ) {
      return true; 
    }

    if ( code == SQLITE_DONE ) {
      return false; 
    }

    std::string errmsg{sqlite3_errmsg(db_.get())}; 
    throw std::runtime_error("SqliteDB::step failed: " + errmsg); 
  }

private: 
  std::unique_ptr<sqlite3, SqliteDeleter> db_{nullptr};
};
