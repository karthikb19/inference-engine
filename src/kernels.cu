#include "kernels.cuh"

namespace inference {

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

}  // namespace inference
