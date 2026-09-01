CXX ?= g++
CPPFLAGS := -Iinclude
CXXFLAGS := -std=c++20 -Wall -Wextra -Wpedantic

TARGET := build/model_info
SOURCES := src/main.cpp src/safetensor.cpp

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SOURCES) include/safetensor.hpp include/json.hpp
	mkdir -p build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(SOURCES) -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -rf build
