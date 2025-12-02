# C++ Crawler Design Document 

## Planning 

### Data Members 

Preferably, none of the data members of the Crawler should be public. A mock setup could be, 

```c++
class Crawler {
private: 

  std::string key, base_url; 
  std::vector<std::string>> endpoints;

  uint16_t retries; 
};
```

### Methods 

We should implement a minimal number of public methods. I think a single request method should be private, forcing a request with retries. 

```c++ 
class Crawler {
public: 

  void get(db* database);

private: 
  json request_(); 
  json request_with_retry_(); 

};
```

The idea should be that `get()` should insert flexibly into a database instance passed by reference. This allows the output to be flexible. However does impose the requirement that the passed `db*` (through whatever library I use for SQLite integration) is pre-instantiated. Alternatively I could instantiate the database and keep it as a private member. 
