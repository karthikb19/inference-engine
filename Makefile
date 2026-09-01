NVCC ?= nvcc
CPPFLAGS := -Iinclude
CUDA_ARCH ?= sm_89

TARGET := build/embedding_gpu
SOURCES := src/main.cpp src/safetensor.cpp src/kernels.cu
TEST_TARGET := build/embedding_gather_test
TEST_SOURCES := tests/embedding_gather_test.cu src/kernels.cu
TOKENS ?= 791

.PHONY: all run test clean

all: $(TARGET)

run: $(TARGET)
	./$(TARGET) --tokens $(TOKENS)

$(TARGET): $(SOURCES) include/safetensor.hpp include/kernels.cuh include/json.hpp
	mkdir -p build
	$(NVCC) -std=c++20 -arch=$(CUDA_ARCH) $(CPPFLAGS) $(SOURCES) -o $@

$(TEST_TARGET): $(TEST_SOURCES) include/kernels.cuh
	mkdir -p build
	$(NVCC) -std=c++20 -arch=$(CUDA_ARCH) $(CPPFLAGS) $(TEST_SOURCES) -o $@

test: $(TEST_TARGET)
	./$(TEST_TARGET)

clean:
	rm -rf build
