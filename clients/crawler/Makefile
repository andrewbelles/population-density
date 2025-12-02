#
# Makefile (crawler)  Andrew Belles  Nov 29th, 2025 
#
# Main build process for crawler and runner submodules 
# for api clients  
#
# 

.PHONY: all clean

CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra
CPPFLAGS ?= -I.
AR ?= ar
ARFLAGS ?= rcs

SRCS := http.cpp
OBJS := $(SRCS:.cpp=.o)
TARGET := libcrawler.a

all: $(TARGET)

$(TARGET): $(OBJS)
	$(AR) $(ARFLAGS) $@ $^

%.o: %.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

clean:
	$(RM) $(OBJS) $(TARGET)
