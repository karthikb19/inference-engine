#include "kernels.cuh"

#include <cmath>
#include <limits>

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

// Attention implementation exercise.
//
// Recommended initial mapping: one CUDA block per (query token, query head),
// with 128 threads for Qwen3-0.6B. A simple first version can have every
// thread cooperate on one key at a time and use shared memory for the online
// softmax state and the output vector. Keep all score/softmax math in FP32.
// The contract and edge cases are encoded in tests/attention_test.cu.
__global__ void causal_attention_kernel(const __nv_bfloat16* query,
                                        const __nv_bfloat16* key_cache,
                                        const __nv_bfloat16* value_cache,
                                        __nv_bfloat16* output,
                                        std::int32_t query_tokens,
                                        std::int32_t key_value_tokens,
                                        std::int32_t query_start_position,
                                        std::int32_t query_heads,
                                        std::int32_t key_value_heads,
                                        std::int32_t head_dim,
                                        float attention_scale) {
    // TODO: Implement causal GQA attention.
    // Grid:  dim3(query_heads, query_tokens)
    // Block: dim3(head_dim) for Qwen3-0.6B (128 threads)
    //
    // query_head = blockIdx.x; query_token = blockIdx.y;
    // kv_head = query_head / (query_heads / key_value_heads);
    // last_key = query_start_position + query_token;
    //
    // Do not write a partial implementation: this inert body intentionally
    // makes attention_test fail until all dimensions are produced correctly.
    (void)query;
    (void)key_cache;
    (void)value_cache;
    (void)output;
    (void)query_tokens;
    (void)key_value_tokens;
    (void)query_start_position;
    (void)query_heads;
    (void)key_value_heads;
    (void)head_dim;
    (void)attention_scale;
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

    // This is intentionally the launch geometry your kernel should target.
    // For Qwen3-0.6B: grid=(16, query_tokens), block=(128).
    const dim3 grid(query_heads, query_tokens);
    const dim3 block(head_dim);
    causal_attention_kernel<<<grid, block, 0, stream>>>(
        query, key_cache, value_cache, output, query_tokens, key_value_tokens,
        query_start_position, query_heads, key_value_heads, head_dim, attention_scale);
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
