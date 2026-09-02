#pragma once

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

namespace inference {

// Qwen3-0.6B attention geometry, copied from models/Qwen3-0.6B/config.json.
// It uses grouped-query attention (GQA), not multi-query attention: each of
// the 8 KV heads is shared by 2 of the 16 query heads.
struct AttentionConfig {
    std::int32_t query_heads;
    std::int32_t key_value_heads;
    std::int32_t head_dim;
    float rope_theta;
};

inline constexpr AttentionConfig qwen3_0_6b_attention{
    .query_heads = 16,
    .key_value_heads = 8,
    .head_dim = 128,
    .rope_theta = 1'000'000.0F,
};

inline constexpr std::int32_t qwen3_0_6b_gqa_group_size =
    qwen3_0_6b_attention.query_heads / qwen3_0_6b_attention.key_value_heads;
inline constexpr float qwen3_0_6b_attention_scale = 0.08838834764831843F;  // 1 / sqrt(128)

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

// Causal scaled dot-product attention with grouped-query attention (GQA).
//
//   query:       [query_tokens, query_heads, head_dim] BF16
//   key_cache:   [key_value_tokens, key_value_heads, head_dim] BF16
//   value_cache: [key_value_tokens, key_value_heads, head_dim] BF16
//   output:      [query_tokens, query_heads, head_dim] BF16
//
// key_cache and value_cache contain positions [0, key_value_tokens). The
// first query token corresponds to absolute position query_start_position;
// query token q can attend only to keys <= query_start_position + q. This
// covers both prefill (query_start_position == 0) and decode (normally
// query_tokens == 1 and query_start_position == key_value_tokens - 1).
//
// GQA head mapping is kv_head = query_head / (query_heads / key_value_heads).
// Scores, softmax, and accumulation must be FP32; round only final output to
// BF16. query/key/value/output must point to device memory and output must not
// overlap an input. attention_scale is normally 1 / sqrt(head_dim), i.e.
// qwen3_0_6b_attention_scale for this model.
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
                                    cudaStream_t stream = nullptr);

// In-place residual connection:
//   input    <- input + residual
//   input:      [num_tokens, hidden_size] __nv_bfloat16 values
//   residual:   [num_tokens, hidden_size] __nv_bfloat16 values
//
// Addition is performed in FP32 before the result is rounded back to BF16.
cudaError_t launch_residual_add(__nv_bfloat16* input,
                                const __nv_bfloat16* residual,
                                std::int32_t num_tokens,
                                std::int32_t hidden_size,
                                cudaStream_t stream = nullptr);

}  // namespace inference
