/*
 * unittest.hpp  Andrew Belles  Dec 21st, 2025 
 *  
 * Exposes MACROS, etc. for use in C++ unit tests 
 */ 

#pragma once 

#define TEST(name) void test_##name() 
#define RUN_TEST(name) do { \
  std::cout << "Running " #name "..."; \
  test_##name(); \
  std::cout << "PASSED\n"; \
} while(0)
