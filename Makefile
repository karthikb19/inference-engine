NVCC ?= nvcc
CXX ?= g++
CPPFLAGS := -Iinclude
CUDA_ARCH ?= sm_89
LDLIBS := -lcublas -lpcre2-8 -licuuc
TOKENIZER_LDLIBS := -lpcre2-8 -licuuc

TARGET := build/embedding_gpu
SOURCES := src/main.cpp src/inference_engine.cpp src/safetensor.cpp src/tokenizer.cpp src/kernels.cu
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
SOFTMAX_TEST_TARGET := build/softmax_test
SOFTMAX_TEST_SOURCES := tests/softmax_test.cu src/kernels.cu
SWIGLU_TEST_TARGET := build/swiglu_test
SWIGLU_TEST_SOURCES := tests/swiglu_test.cu src/kernels.cu
TOKENIZER_TEST_TARGET := build/tokenizer_test
TOKENIZER_TEST_SOURCES := tests/tokenizer_test.cpp src/tokenizer.cpp
TOKENS ?= 791

.PHONY: all run test clean

all: $(TARGET)

run: $(TARGET)
	./$(TARGET) --tokens $(TOKENS)

$(TARGET): $(SOURCES) include/inference_engine.hpp include/safetensor.hpp include/tokenizer.hpp include/kernels.cuh include/json.hpp
	mkdir -p build
	$(NVCC) -std=c++20 -arch=$(CUDA_ARCH) $(CPPFLAGS) $(SOURCES) -o $@ $(LDLIBS)

$(TEST_TARGET): $(TEST_SOURCES) include/kernels.cuh
	mkdir -p build
	$(NVCC) -std=c++20 -arch=$(CUDA_ARCH) $(CPPFLAGS) $(TEST_SOURCES) -o $@ $(LDLIBS)

$(RMS_NORM_TEST_TARGET): $(RMS_NORM_TEST_SOURCES) include/kernels.cuh
	mkdir -p build
	$(NVCC) -std=c++20 -arch=$(CUDA_ARCH) $(CPPFLAGS) $(RMS_NORM_TEST_SOURCES) -o $@ $(LDLIBS)

$(ROPE_TEST_TARGET): $(ROPE_TEST_SOURCES) include/kernels.cuh
	mkdir -p build
	$(NVCC) -std=c++20 -arch=$(CUDA_ARCH) $(CPPFLAGS) $(ROPE_TEST_SOURCES) -o $@ $(LDLIBS)

$(RESIDUAL_ADD_TEST_TARGET): $(RESIDUAL_ADD_TEST_SOURCES) include/kernels.cuh
	mkdir -p build
	$(NVCC) -std=c++20 -arch=$(CUDA_ARCH) $(CPPFLAGS) $(RESIDUAL_ADD_TEST_SOURCES) -o $@ $(LDLIBS)

$(ATTENTION_TEST_TARGET): $(ATTENTION_TEST_SOURCES) include/kernels.cuh
	mkdir -p build
	$(NVCC) -std=c++20 -arch=$(CUDA_ARCH) $(CPPFLAGS) $(ATTENTION_TEST_SOURCES) -o $@ $(LDLIBS)

$(SOFTMAX_TEST_TARGET): $(SOFTMAX_TEST_SOURCES) include/kernels.cuh
	mkdir -p build
	$(NVCC) -std=c++20 -arch=$(CUDA_ARCH) $(CPPFLAGS) $(SOFTMAX_TEST_SOURCES) -o $@ $(LDLIBS)

$(SWIGLU_TEST_TARGET): $(SWIGLU_TEST_SOURCES) include/kernels.cuh
	mkdir -p build
	$(NVCC) -std=c++20 -arch=$(CUDA_ARCH) $(CPPFLAGS) $(SWIGLU_TEST_SOURCES) -o $@ $(LDLIBS)

$(TOKENIZER_TEST_TARGET): $(TOKENIZER_TEST_SOURCES) include/tokenizer.hpp include/json.hpp
	mkdir -p build
	$(CXX) -std=c++20 $(CPPFLAGS) $(TOKENIZER_TEST_SOURCES) -o $@ $(TOKENIZER_LDLIBS)

test: $(TEST_TARGET) $(RMS_NORM_TEST_TARGET) $(ROPE_TEST_TARGET) $(RESIDUAL_ADD_TEST_TARGET) $(SOFTMAX_TEST_TARGET) $(SWIGLU_TEST_TARGET) $(TOKENIZER_TEST_TARGET)
	./$(TEST_TARGET)
	./$(RMS_NORM_TEST_TARGET)
	./$(ROPE_TEST_TARGET)
	./$(RESIDUAL_ADD_TEST_TARGET)
	./$(SOFTMAX_TEST_TARGET)
	./$(SWIGLU_TEST_TARGET)
	./$(TOKENIZER_TEST_TARGET)

.PHONY: softmax-test swiglu-test
softmax-test: $(SOFTMAX_TEST_TARGET)
	./$(SOFTMAX_TEST_TARGET)

swiglu-test: $(SWIGLU_TEST_TARGET)
	./$(SWIGLU_TEST_TARGET)

.PHONY: tokenizer-test
tokenizer-test: $(TOKENIZER_TEST_TARGET)
	./$(TOKENIZER_TEST_TARGET)

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
