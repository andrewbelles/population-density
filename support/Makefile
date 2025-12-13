# 
# Makefile  Andrew Belles  Dec 12th, 2025 
#
# Compiles GeospatialGraph module with extra flags 
# for unit tests 
#
#

CXX 				= g++ 
CXXFLAGS 		= -std=c++17 -Wall -Wpedantic -Wextra -O3 
DEBUG_FLAGS = -g -DDEBUG 
INCLUDES 		= -I.

BUILD_DIR		= build 

SUPPORT_SRC = support.cpp
GEO_SRC 		= geospatial_graph.cpp 
TEST_SRC 		= test_geospatial_graph.cpp 
HEADERS  		= geospatial_graph.hpp support.hpp 

SUPPORT_OBJ = support.o 
GEO_OBJ     = geospatial_graph.o 
TEST_OBJ    = test_geospatial_graph.o 

TEST_TARGET = test_geospatial_graph

### 
# 
# C++ NATIVE CODE COMPILATION RULES  
#
###

all: $(TEST_TARGET) python install-python 

$(BUILD_DIR): 
	mkdir -p $(BUILD_DIR)

$(SUPPORT_OBJ): $(SUPPORT_SRC) support.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $(SUPPORT_SRC) -o $(SUPPORT_OBJ)

$(GEO_OBJ): $(GEO_SRC) geospatial_graph.hpp support.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $(GEO_SRC) -o $(GEO_OBJ)

$(TEST_OBJ): $(TEST_SRC) $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(DEBUG_FLAGS) $(INCLUDES) -c $(TEST_SRC) -o $(TEST_OBJ)

test: $(TEST_TARGET)
	./$(TEST_TARGET)

$(TEST_TARGET): $(SUPPORT_OBJ) $(GEO_OBJ) $(TEST_OBJ)
	$(CXX) $(CXXFLAGS) $(DEBUG_FLAGS) $(INCLUDES) -o $(TEST_TARGET) $(SUPPORT_OBJ) $(GEO_OBJ) $(TEST_OBJ)

debug: CXX += $(DEBUG_FLAG) 
debug: $(MAIN_TARGET)

clean: 
	rm -rf $(BUILD_DIR) $(MAIN_TARGET) $(TEST_TARGET) *.o 
	rm -f $(PYTHON_MODULE) *.so ../models/*.so

valgrind: $(TEST_TARGET)
	valgrind --leak-check=full ./$(TEST_TARGET)

### 
# 
# PYBIND11 COMPILATION RULES 
#
###

PYBIND11_INCLUDES  = $(shell python -m pybind11 --includes) 
PYTHON_SUFFIX 		 = $(shell python3-config --extension-suffix)
PYTHON_MODULE      = geospatial_graph_cpp$(PYTHON_SUFFIX) 

PYTHON_BINDING_SRC = python_bindings.cpp 

python: $(PYTHON_MODULE)

$(PYTHON_MODULE): $(SUPPORT_OBJ) $(GEO_OBJ) $(PYTHON_BINDING_SRC) 
	$(CXX) -O3 -Wall -shared -std=c++17 -fPIC $(PYBIND11_INCLUDES) \
		$(shell python3-config --ldflags) $(PYTHON_BINDING_SRC) $(SUPPORT_OBJ) $(GEO_OBJ) -o $(PYTHON_MODULE)

install-python: $(PYTHON_MODULE)
	cp $(PYTHON_MODULE) ../models/ 

clean-python: 
	rm -f $(PYTHON_MODULE) *.so


.PHONY: python install-python clean-python all test debug clean valgrind 
