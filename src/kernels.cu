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

}  // namespace inference
