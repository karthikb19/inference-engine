#include "kernels.cuh"

#include <cublas_v2.h>

#include <cmath>
#include <limits>
#include <vector>

namespace inference {

__global__ void softmax_kernel(const float* input,
                               float* output,
                               std::int32_t row_size) {
    extern __shared__ float reduction[];
    const std::int32_t row = blockIdx.x;
    const std::int32_t tid = threadIdx.x;
    const std::int64_t base = static_cast<std::int64_t>(row) * row_size;

    float local_max = -INFINITY;
    for (std::int32_t column = tid; column < row_size; column += blockDim.x) {
        local_max = fmaxf(local_max, input[base + column]);
    }
    reduction[tid] = local_max;
    __syncthreads();
    for (std::int32_t active = blockDim.x / 2; active > 0; active >>= 1) {
        if (tid < active) reduction[tid] = fmaxf(reduction[tid], reduction[tid + active]);
        __syncthreads();
    }
    const float maximum = reduction[0];

    float local_sum = 0.0F;
    for (std::int32_t column = tid; column < row_size; column += blockDim.x) {
        local_sum += expf(input[base + column] - maximum);
    }
    reduction[tid] = local_sum;
    __syncthreads();
    for (std::int32_t active = blockDim.x / 2; active > 0; active >>= 1) {
        if (tid < active) reduction[tid] += reduction[tid + active];
        __syncthreads();
    }
    const float inverse_sum = 1.0F / reduction[0];
    for (std::int32_t column = tid; column < row_size; column += blockDim.x) {
        output[base + column] = expf(input[base + column] - maximum) * inverse_sum;
    }
}

cudaError_t launch_softmax(const float* input,
                           float* output,
                           std::int32_t num_rows,
                           std::int32_t row_size,
                           cudaStream_t stream) {
    if (input == nullptr || output == nullptr || num_rows <= 0 || row_size <= 0) {
        return cudaErrorInvalidValue;
    }
    constexpr std::int32_t threads_per_block = 256;
    softmax_kernel<<<num_rows, threads_per_block,
                     threads_per_block * sizeof(float), stream>>>(input, output, row_size);
    return cudaGetLastError();
}

__global__ void swiglu_kernel(const __nv_bfloat16* gate,
                              const __nv_bfloat16* up,
                              __nv_bfloat16* output,
                              std::int64_t total_elements) {
    const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total_elements) return;

    const float gate_value = __bfloat162float(gate[index]);
    const float up_value = __bfloat162float(up[index]);
    const float silu = gate_value / (1.0F + expf(-gate_value));
    output[index] = __float2bfloat16(silu * up_value);
}

cudaError_t launch_swiglu(const __nv_bfloat16* gate,
                          const __nv_bfloat16* up,
                          __nv_bfloat16* output,
                          std::int32_t num_tokens,
                          std::int32_t intermediate_size,
                          cudaStream_t stream) {
    if (gate == nullptr || up == nullptr || output == nullptr || num_tokens <= 0 ||
        intermediate_size <= 0) {
        return cudaErrorInvalidValue;
    }
    constexpr std::int32_t threads_per_block = 256;
    const std::int64_t total_elements = static_cast<std::int64_t>(num_tokens) * intermediate_size;
    const std::int64_t blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    if (blocks > std::numeric_limits<unsigned int>::max()) return cudaErrorInvalidValue;
    swiglu_kernel<<<static_cast<unsigned int>(blocks), threads_per_block, 0, stream>>>(
        gate, up, output, total_elements);
    return cudaGetLastError();
}

// Embedding gather kernel
// Qwen3-0.6B embedding_table dimensions: [151936 vocabulary tokens, 1024 hidden values]
// Each entry is BF16, so the table is 311,164,928 bytes.
__global__ void embedding_gather_kernel(const __nv_bfloat16* embedding_table, const std::int32_t* token_ids, __nv_bfloat16* output, std::int32_t num_tokens, std::int32_t hidden_size) {
    int tIdx = blockDim.x * blockIdx.x + threadIdx.x;
    int token_idx = blockIdx.x;
    int dim_idx = threadIdx.x;
    if(tIdx < num_tokens * hidden_size){
        output[tIdx] = embedding_table[token_ids[token_idx] * hidden_size + dim_idx];
    }

}

// Embedding gather launch
cudaError_t launch_embedding_gather(const __nv_bfloat16* embedding_table, const std::int32_t* token_ids, __nv_bfloat16* output, std::int32_t num_tokens, std::int32_t hidden_size, cudaStream_t stream) {
    if (embedding_table == nullptr || token_ids == nullptr || output == nullptr ||
        num_tokens <= 0 || hidden_size <= 0 || hidden_size > 1024) {
        return cudaErrorInvalidValue;
    }

    const dim3 grid(num_tokens);
    const dim3 block(hidden_size);
    constexpr std::size_t shared_memory_bytes = 0;

    embedding_gather_kernel<<<grid, block, shared_memory_bytes, stream>>>(
        embedding_table, token_ids, output, num_tokens, hidden_size);

    // cudaGetLastError checks launch errors; it does not wait for kernel completion.
    return cudaGetLastError();
}

// RMSNorm skeleton.  The completed kernel will map one activation row to a
// block, reduce sum(x^2) in FP32, then apply weight / sqrt(mean(x^2) + eps).
__global__ void rms_norm_kernel(const __nv_bfloat16* input,
                                const __nv_bfloat16* weight,
                                __nv_bfloat16* output,
                                std::int32_t num_tokens,
                                std::int32_t hidden_size,
                                float epsilon) {
    // similar to the sum reduction kernel
    // we use that concept and then basically atomic add that stuff and then we can do sync threads go throguh and add epislon
    // and then like divide yk?
    __shared__ float shared[1024];

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int tIdx = threadIdx.x;
    float value = 0.0F;

    if(idx < num_tokens * hidden_size){
        value = __bfloat162float(input[idx]);
        shared[tIdx] = value * value;
    } else{
        shared[tIdx] = 0.0;
    }

    __syncthreads();

    for(int stride = blockDim.x / 2; stride > 0; stride /= 2){
        if(tIdx < stride){
            shared[tIdx] += shared[tIdx + stride];
        }
        __syncthreads();
    }

    const float inv_rms = rsqrtf(shared[0] / static_cast<float>(hidden_size) + epsilon);
    if(idx < num_tokens * hidden_size){
        output[idx] = __float2bfloat16(value * inv_rms *
                                  __bfloat162float(weight[tIdx]));
    }
}

// Launch one block per input row and one thread per hidden dimension:
//   grid.x  = num_tokens
//   block.x = hidden_size (1024 for Qwen3-0.6B)
cudaError_t launch_rms_norm(const __nv_bfloat16* input,
                            const __nv_bfloat16* weight,
                            __nv_bfloat16* output,
                            std::int32_t num_tokens,
                            std::int32_t hidden_size,
                            float epsilon,
                            cudaStream_t stream) {
    if (input == nullptr || weight == nullptr || output == nullptr ||
        num_tokens <= 0 || hidden_size <= 0 || hidden_size > 1024 ||
        (hidden_size & (hidden_size - 1)) != 0 || epsilon <= 0.0F) {
        return cudaErrorInvalidValue;
    }

    const dim3 grid(num_tokens);
    const dim3 block(hidden_size);
    constexpr std::size_t shared_memory_bytes = 0;
    rms_norm_kernel<<<grid, block, shared_memory_bytes, stream>>>(
        input, weight, output, num_tokens, hidden_size, epsilon);

    return cudaGetLastError();
}

// RoPE skeleton: map one thread to one (token, head, half-dimension) pair.
// The test in tests/rope_test.cu is the contract for completing this kernel.
__global__ void rope_kernel(const __nv_bfloat16* input,
                            const std::int32_t* position_ids,
                            const float* cos,
                            const float* sin,
                            __nv_bfloat16* output,
                            std::int32_t num_tokens,
                            std::int32_t num_heads,
                            std::int32_t head_dim) {
    const std::int32_t pair_index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::int32_t half_dim = head_dim / 2;
    const std::int32_t total_pairs = num_tokens * num_heads * half_dim;
    if (pair_index >= total_pairs) return;

    // Turn this thread's global work ID back into [token, head, dim].
    const std::int32_t dim_idx = pair_index % half_dim;
    const std::int32_t head_pair_idx = pair_index / half_dim;
    const std::int32_t head = head_pair_idx % num_heads;
    const std::int32_t token = head_pair_idx / num_heads;

    // A local input row can have a later full-sequence position during decode.
    const std::int32_t position = position_ids[token];
    const std::int32_t vector_base = (token * num_heads + head) * head_dim;
    const std::int32_t table_index = position * half_dim + dim_idx;

    // Load both source values before either write so input == output is safe.
    const float x0 = __bfloat162float(input[vector_base + dim_idx]);
    const float x1 = __bfloat162float(input[vector_base + dim_idx + half_dim]);
    const float c = cos[table_index];
    const float s = sin[table_index];

    output[vector_base + dim_idx] = __float2bfloat16(x0 * c - x1 * s);
    output[vector_base + dim_idx + half_dim] = __float2bfloat16(x0 * s + x1 * c);
}

cudaError_t launch_rope(const __nv_bfloat16* input,
                        const std::int32_t* position_ids,
                        const float* cos,
                        const float* sin,
                        __nv_bfloat16* output,
                        std::int32_t num_tokens,
                        std::int32_t num_heads,
                        std::int32_t head_dim,
                        cudaStream_t stream) {
    if (input == nullptr || position_ids == nullptr || cos == nullptr || sin == nullptr ||
        output == nullptr || num_tokens <= 0 || num_heads <= 0 || head_dim <= 0 ||
        head_dim > 1024 || (head_dim & 1) != 0) {
        return cudaErrorInvalidValue;
    }

    constexpr std::int32_t threads_per_block = 256;
    const std::int32_t total_pairs = num_tokens * num_heads * (head_dim / 2);
    const dim3 grid((total_pairs + threads_per_block - 1) / threads_per_block);
    const dim3 block(threads_per_block);
    rope_kernel<<<grid, block, 0, stream>>>(input, position_ids, cos, sin, output,
                                              num_tokens, num_heads, head_dim);
    return cudaGetLastError();
}

// scores has layout [query_heads, query_tokens, key_value_tokens]. One block
// owns one row scores[query_head, query_token, :]. Thread k changes columns
// k, k + blockDim.x, ... so all future-token logits become -infinity before
// the existing row-softmax kernel turns them into zero probabilities.
__global__ void causal_mask_kernel(float* scores,
                                   std::int32_t query_tokens,
                                   std::int32_t key_value_tokens,
                                   std::int32_t query_start_position) {
    const std::int32_t query_head = blockIdx.x;
    const std::int32_t query_token = blockIdx.y;
    const std::int32_t last_allowed_key = query_start_position + query_token;
    const std::int64_t row_base =
        (static_cast<std::int64_t>(query_head) * query_tokens + query_token) * key_value_tokens;

    for (std::int32_t key_token = threadIdx.x; key_token < key_value_tokens;
         key_token += blockDim.x) {
        if (key_token > last_allowed_key) scores[row_base + key_token] = -INFINITY;
    }
}

__global__ void bf16_to_float_kernel(const __nv_bfloat16* input, float* output,
                                     std::int64_t element_count) {
    const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < element_count) output[index] = __bfloat162float(input[index]);
}

__global__ void float_to_bf16_kernel(const float* input, __nv_bfloat16* output,
                                     std::int64_t element_count) {
    const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < element_count) output[index] = __float2bfloat16(input[index]);
}

class DeviceAllocation {
public:
    DeviceAllocation() = default;
    ~DeviceAllocation() { if (pointer_ != nullptr) cudaFree(pointer_); }
    DeviceAllocation(const DeviceAllocation&) = delete;

    cudaError_t allocate(std::size_t bytes) { return cudaMalloc(&pointer_, bytes); }
    [[nodiscard]] void* get() const { return pointer_; }

private:
    void* pointer_ = nullptr;
};

class CublasHandle {
public:
    ~CublasHandle() { if (handle_ != nullptr) cublasDestroy(handle_); }
    CublasHandle(const CublasHandle&) = delete;
    CublasHandle() = default;

    cublasStatus_t create() { return cublasCreate(&handle_); }
    [[nodiscard]] cublasHandle_t get() const { return handle_; }

private:
    cublasHandle_t handle_ = nullptr;
};

cudaError_t cublas_error(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS: return cudaSuccess;
        case CUBLAS_STATUS_ALLOC_FAILED: return cudaErrorMemoryAllocation;
        case CUBLAS_STATUS_INVALID_VALUE: return cudaErrorInvalidValue;
        case CUBLAS_STATUS_ARCH_MISMATCH:
        case CUBLAS_STATUS_NOT_SUPPORTED: return cudaErrorNotSupported;
        case CUBLAS_STATUS_EXECUTION_FAILED: return cudaErrorLaunchFailure;
        default: return cudaErrorUnknown;
    }
}

cudaError_t launch_causal_attention(const __nv_bfloat16* query,
                                    const __nv_bfloat16* key_cache,
                                    const __nv_bfloat16* value_cache,
                                    __nv_bfloat16* output,
                                    std::int32_t query_tokens,
                                    std::int32_t key_value_tokens,
                                    std::int32_t query_start_position,
                                    std::int32_t query_heads,
                                    std::int32_t key_value_heads,
                                    std::int32_t head_dim,
                                    float attention_scale,
                                    cudaStream_t stream) {
    if (query == nullptr || key_cache == nullptr || value_cache == nullptr || output == nullptr ||
        query_tokens <= 0 || key_value_tokens <= 0 || query_start_position < 0 ||
        query_heads <= 0 || key_value_heads <= 0 || head_dim <= 0 || head_dim > 1024 ||
        query_heads % key_value_heads != 0 ||
        query_start_position > key_value_tokens - query_tokens ||
        !std::isfinite(attention_scale) || attention_scale <= 0.0F) {
        return cudaErrorInvalidValue;
    }

    // Stage 1: cuBLAS computes Q @ K^T for each query head into FP32 scores.
    // Stage 2: causal_mask_kernel and launch_softmax transform scores into P.
    // Stage 3: cuBLAS computes P @ V for each query head.
    const std::size_t score_count = static_cast<std::size_t>(query_heads) * query_tokens * key_value_tokens;
    if (score_count > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        return cudaErrorInvalidValue;
    }

    DeviceAllocation score_storage;
    cudaError_t status = score_storage.allocate(score_count * sizeof(float));
    if (status != cudaSuccess) return status;
    auto* scores = static_cast<float*>(score_storage.get());

    const std::size_t value_count = static_cast<std::size_t>(key_value_tokens) * key_value_heads * head_dim;
    const std::size_t output_count = static_cast<std::size_t>(query_tokens) * query_heads * head_dim;
    if (value_count > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
        output_count > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        return cudaErrorInvalidValue;
    }
    // GemmBatchedEx requires its two input matrices to have the same type.
    // Keep P and the P@V accumulation in FP32 by converting V once here,
    // then round only the final attention output back to BF16.
    DeviceAllocation values_fp32_storage;
    DeviceAllocation output_fp32_storage;
    if ((status = values_fp32_storage.allocate(value_count * sizeof(float))) != cudaSuccess ||
        (status = output_fp32_storage.allocate(output_count * sizeof(float))) != cudaSuccess) {
        return status;
    }
    auto* values_fp32 = static_cast<float*>(values_fp32_storage.get());
    auto* output_fp32 = static_cast<float*>(output_fp32_storage.get());
    constexpr std::int32_t conversion_threads = 256;
    const std::int64_t conversion_blocks =
        (static_cast<std::int64_t>(value_count) + conversion_threads - 1) / conversion_threads;
    bf16_to_float_kernel<<<static_cast<unsigned int>(conversion_blocks), conversion_threads, 0, stream>>>(
        value_cache, values_fp32, value_count);
    if ((status = cudaGetLastError()) != cudaSuccess) return status;

    // GQA repeats each KV-head pointer for its query-head group. Pointer-array
    // batched GEMM expresses that repetition without copying K or V.
    std::vector<const __nv_bfloat16*> query_pointers(query_heads);
    std::vector<const __nv_bfloat16*> key_pointers(query_heads);
    std::vector<float*> value_pointers(query_heads);
    std::vector<float*> score_pointers(query_heads);
    std::vector<float*> output_pointers(query_heads);
    const std::int32_t group_size = query_heads / key_value_heads;
    for (std::int32_t query_head = 0; query_head < query_heads; ++query_head) {
        const std::int32_t kv_head = query_head / group_size;
        query_pointers[query_head] = query + static_cast<std::int64_t>(query_head) * head_dim;
        key_pointers[query_head] = key_cache + static_cast<std::int64_t>(kv_head) * head_dim;
        value_pointers[query_head] = values_fp32 + static_cast<std::int64_t>(kv_head) * head_dim;
        score_pointers[query_head] = scores + static_cast<std::int64_t>(query_head) * query_tokens * key_value_tokens;
        output_pointers[query_head] = output_fp32 + static_cast<std::int64_t>(query_head) * head_dim;
    }

    DeviceAllocation query_pointer_storage;
    DeviceAllocation key_pointer_storage;
    DeviceAllocation value_pointer_storage;
    DeviceAllocation score_pointer_storage;
    DeviceAllocation output_pointer_storage;
    const std::size_t pointer_bytes = static_cast<std::size_t>(query_heads) * sizeof(void*);
    if ((status = query_pointer_storage.allocate(pointer_bytes)) != cudaSuccess ||
        (status = key_pointer_storage.allocate(pointer_bytes)) != cudaSuccess ||
        (status = value_pointer_storage.allocate(pointer_bytes)) != cudaSuccess ||
        (status = score_pointer_storage.allocate(pointer_bytes)) != cudaSuccess ||
        (status = output_pointer_storage.allocate(pointer_bytes)) != cudaSuccess) {
        return status;
    }
    if ((status = cudaMemcpyAsync(query_pointer_storage.get(), query_pointers.data(), pointer_bytes,
                                  cudaMemcpyHostToDevice, stream)) != cudaSuccess ||
        (status = cudaMemcpyAsync(key_pointer_storage.get(), key_pointers.data(), pointer_bytes,
                                  cudaMemcpyHostToDevice, stream)) != cudaSuccess ||
        (status = cudaMemcpyAsync(value_pointer_storage.get(), value_pointers.data(), pointer_bytes,
                                  cudaMemcpyHostToDevice, stream)) != cudaSuccess ||
        (status = cudaMemcpyAsync(score_pointer_storage.get(), score_pointers.data(), pointer_bytes,
                                  cudaMemcpyHostToDevice, stream)) != cudaSuccess ||
        (status = cudaMemcpyAsync(output_pointer_storage.get(), output_pointers.data(), pointer_bytes,
                                  cudaMemcpyHostToDevice, stream)) != cudaSuccess) {
        return status;
    }

    CublasHandle handle;
    if ((status = cublas_error(handle.create())) != cudaSuccess ||
        (status = cublas_error(cublasSetStream(handle.get(), stream))) != cudaSuccess) {
        return status;
    }

    const float zero = 0.0F;

    // Row-major Q[Tq, Dh] is column-major Q^T[Dh, Tq], and likewise for K.
    // Therefore K^T (column-major) times Q^T produces score storage laid out
    // as row-major [Tq, Tk]. GEMM batches over query heads.
    if ((status = cublas_error(cublasGemmBatchedEx(
            handle.get(), CUBLAS_OP_T, CUBLAS_OP_N,
            key_value_tokens, query_tokens, head_dim,
            &attention_scale,
            reinterpret_cast<const void* const*>(key_pointer_storage.get()), CUDA_R_16BF, key_value_heads * head_dim,
            reinterpret_cast<const void* const*>(query_pointer_storage.get()), CUDA_R_16BF, query_heads * head_dim,
            &zero,
            reinterpret_cast<void* const*>(score_pointer_storage.get()), CUDA_R_32F, key_value_tokens,
            query_heads, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT))) != cudaSuccess) {
        return status;
    }

    constexpr std::int32_t mask_threads = 256;
    causal_mask_kernel<<<dim3(query_heads, query_tokens), mask_threads, 0, stream>>>(
        scores, query_tokens, key_value_tokens, query_start_position);
    if ((status = cudaGetLastError()) != cudaSuccess) return status;
    if ((status = launch_softmax(scores, scores, query_heads * query_tokens,
                                 key_value_tokens, stream)) != cudaSuccess) {
        return status;
    }

    const float one = 1.0F;
    // In the same column-major view, V[ Tk, Dh ] becomes [Dh, Tk] and P is
    // [Tk, Tq], so V * P writes the row-major output [Tq, Dh].
    if ((status = cublas_error(cublasSgemmBatched(
            handle.get(), CUBLAS_OP_N, CUBLAS_OP_N,
            head_dim, query_tokens, key_value_tokens,
            &one,
            reinterpret_cast<const float* const*>(value_pointer_storage.get()), key_value_heads * head_dim,
            reinterpret_cast<const float* const*>(score_pointer_storage.get()), key_value_tokens,
            &zero,
            reinterpret_cast<float* const*>(output_pointer_storage.get()), query_heads * head_dim,
            query_heads))) != cudaSuccess) {
        return status;
    }
    const std::int64_t output_blocks =
        (static_cast<std::int64_t>(output_count) + conversion_threads - 1) / conversion_threads;
    float_to_bf16_kernel<<<static_cast<unsigned int>(output_blocks), conversion_threads, 0, stream>>>(
        output_fp32, output, output_count);
    return cudaGetLastError();
}

__global__ void residual_add_kernel(__nv_bfloat16* input,
                                    const __nv_bfloat16* residual,
                                    std::int64_t total_elements) {
    const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total_elements) return;

    const float input_value = __bfloat162float(input[index]);
    const float residual_value = __bfloat162float(residual[index]);
    input[index] = __float2bfloat16(input_value + residual_value);
}

cudaError_t launch_residual_add(__nv_bfloat16* input,
                                const __nv_bfloat16* residual,
                                std::int32_t num_tokens,
                                std::int32_t hidden_size,
                                cudaStream_t stream) {
    if (input == nullptr || residual == nullptr || num_tokens <= 0 || hidden_size <= 0) {
        return cudaErrorInvalidValue;
    }

    constexpr std::int32_t threads_per_block = 256;
    const std::int64_t total_elements = static_cast<std::int64_t>(num_tokens) * hidden_size;
    const std::int64_t blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    if (blocks > std::numeric_limits<unsigned int>::max()) return cudaErrorInvalidValue;

    const dim3 grid(static_cast<unsigned int>(blocks));
    const dim3 block(threads_per_block);
    residual_add_kernel<<<grid, block, 0, stream>>>(input, residual, total_elements);
    return cudaGetLastError();
}

}  // namespace inference
