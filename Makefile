NVCC ?= nvcc
CPPFLAGS := -Iinclude
CUDA_ARCH ?= sm_89

TARGET := build/embedding_gpu
SOURCES := src/main.cpp src/safetensor.cpp src/kernels.cu
TEST_TARGET := build/embedding_gather_test
TEST_SOURCES := tests/embedding_gather_test.cu src/kernels.cu
RMS_NORM_TEST_TARGET := build/rms_norm_test
RMS_NORM_TEST_SOURCES := tests/rms_norm_test.cu src/kernels.cu
ROPE_TEST_TARGET := build/rope_test
ROPE_TEST_SOURCES := tests/rope_test.cu src/kernels.cu
RESIDUAL_ADD_TEST_TARGET := build/residual_add_test
RESIDUAL_ADD_TEST_SOURCES := tests/residual_add_test.cu src/kernels.cu
ATTENTION_TEST_TARGET := build/attention_test
ATTENTION_TEST_SOURCES := tests/attention_test.cu src/kernels.cu
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

$(RMS_NORM_TEST_TARGET): $(RMS_NORM_TEST_SOURCES) include/kernels.cuh
	mkdir -p build
	$(NVCC) -std=c++20 -arch=$(CUDA_ARCH) $(CPPFLAGS) $(RMS_NORM_TEST_SOURCES) -o $@

$(ROPE_TEST_TARGET): $(ROPE_TEST_SOURCES) include/kernels.cuh
	mkdir -p build
	$(NVCC) -std=c++20 -arch=$(CUDA_ARCH) $(CPPFLAGS) $(ROPE_TEST_SOURCES) -o $@

$(RESIDUAL_ADD_TEST_TARGET): $(RESIDUAL_ADD_TEST_SOURCES) include/kernels.cuh
	mkdir -p build
	$(NVCC) -std=c++20 -arch=$(CUDA_ARCH) $(CPPFLAGS) $(RESIDUAL_ADD_TEST_SOURCES) -o $@

$(ATTENTION_TEST_TARGET): $(ATTENTION_TEST_SOURCES) include/kernels.cuh
	mkdir -p build
	$(NVCC) -std=c++20 -arch=$(CUDA_ARCH) $(CPPFLAGS) $(ATTENTION_TEST_SOURCES) -o $@

test: $(TEST_TARGET) $(RMS_NORM_TEST_TARGET) $(ROPE_TEST_TARGET) $(RESIDUAL_ADD_TEST_TARGET)
	./$(TEST_TARGET)
	./$(RMS_NORM_TEST_TARGET)
	./$(ROPE_TEST_TARGET)
	./$(RESIDUAL_ADD_TEST_TARGET)

.PHONY: rms-norm-test
rms-norm-test: $(RMS_NORM_TEST_TARGET)
	./$(RMS_NORM_TEST_TARGET)

.PHONY: rope-test
rope-test: $(ROPE_TEST_TARGET)
	./$(ROPE_TEST_TARGET)

.PHONY: residual-add-test
residual-add-test: $(RESIDUAL_ADD_TEST_TARGET)
	./$(RESIDUAL_ADD_TEST_TARGET)

# This is intentionally opt-in until causal_attention_kernel is implemented.
.PHONY: attention-test
attention-test: $(ATTENTION_TEST_TARGET)
	./$(ATTENTION_TEST_TARGET)

clean:
	rm -rf build
