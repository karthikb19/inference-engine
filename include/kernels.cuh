#pragma once

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

namespace inference {

// Device-memory contract for Qwen's token embedding lookup:
//
//   embedding_table: [vocab_size, hidden_size] __nv_bfloat16 values
//   token_ids:       [num_tokens] signed 32-bit token IDs
//   output:          [num_tokens, hidden_size] __nv_bfloat16 values
//
// All pointers must point to device memory. The caller validates token IDs
// before launching; the kernel can therefore assume 0 <= token_id < vocab_size.
cudaError_t launch_embedding_gather(const __nv_bfloat16* embedding_table,
                                    const std::int32_t* token_ids,
                                    __nv_bfloat16* output,
                                    std::int32_t num_tokens,
                                    std::int32_t hidden_size,
                                    cudaStream_t stream = nullptr);

}  // namespace inference
