# 
# Makefile  Andrew Belles  Nov 29th, 2025 
#
# Build process for all API clients and main api daemon 
#
#
#


CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra
LDLIBS ?= -lsqlite3 -lcurl

.PHONY: all crawler clean

all: crawler climate

crawler:
	$(MAKE) -C crawler

climate: climate.cpp
	$(CXX) $(CXXFLAGS) -Icrawler -o $@ $< $(LDLIBS)

clean:
	$(MAKE) -C crawler clean
	$(RM) climate
