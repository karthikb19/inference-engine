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

// Device-memory contract for RMS normalization:
//
//   input:   [num_tokens, hidden_size] __nv_bfloat16 activations
//   weight:  [hidden_size] __nv_bfloat16 learned scale values
//   output:  [num_tokens, hidden_size] __nv_bfloat16 normalized activations
//
// The reduction and normalization will be performed in FP32.  This interface
// is deliberately separate from residual addition; Qwen's pre-attention and
// pre-MLP norms both use this form.
cudaError_t launch_rms_norm(const __nv_bfloat16* input,
                            const __nv_bfloat16* weight,
                            __nv_bfloat16* output,
                            std::int32_t num_tokens,
                            std::int32_t hidden_size,
                            float epsilon,
                            cudaStream_t stream = nullptr);

// Apply Qwen's half-split rotary positional embedding (RoPE) to Q or K.
//
//   input/output: [num_tokens, num_heads, head_dim] __nv_bfloat16 values
//   position_ids: [num_tokens] int32 positions into the cosine/sine tables
//   cos/sin:      [max_position, head_dim / 2] float values
//
// Qwen pairs dimension i with i + head_dim / 2.  For every token, head, and
// pair i, the intended rotation is:
//   output[i]        = input[i] * cos - input[i + half] * sin
//   output[i + half] = input[i] * sin + input[i + half] * cos
//
// The caller owns the RoPE table and validates that every position ID is in
// range. input and output may alias for an in-place rotation.
cudaError_t launch_rope(const __nv_bfloat16* input,
                        const std::int32_t* position_ids,
                        const float* cos,
                        const float* sin,
                        __nv_bfloat16* output,
                        std::int32_t num_tokens,
                        std::int32_t num_heads,
                        std::int32_t head_dim,
                        cudaStream_t stream = nullptr);

}  // namespace inference
