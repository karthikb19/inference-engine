#include "inference_engine.hpp"
#include "kernels.cuh"
#include "safetensor.hpp"

#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

// RMSNorm dispatch scaffolding.  When the first transformer block is wired
// in, these identify its pre-attention norm and preserve Qwen's epsilon.
constexpr char first_attention_norm_name[] = "model.layers.0.input_layernorm.weight";
constexpr float rms_norm_epsilon = 1.0e-6F;
constexpr std::int32_t qwen_layer_count = 28;
constexpr std::int32_t qwen_eos_token_id = 151645;

// CUDA helpers
void cuda_check(cudaError_t status, std::string_view operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) {
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)), "cudaMalloc");
    }

    ~DeviceBuffer() {
        if (data_ != nullptr) cudaFree(data_);
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept : data_(other.data_) { other.data_ = nullptr; }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (data_ != nullptr) cudaFree(data_);
            data_ = other.data_;
            other.data_ = nullptr;
        }
        return *this;
    }

    [[nodiscard]] T* get() const { return data_; }

private:
    T* data_ = nullptr;
};

// Owns all device-resident parameters for one transformer block. DeviceBuffer
// itself only stores a CUDA pointer; moving LayerWeights transfers ownership
// of those pointers without copying any tensor data.
struct LayerWeights {
    DeviceBuffer<__nv_bfloat16> input_norm;
    DeviceBuffer<__nv_bfloat16> q_proj;
    DeviceBuffer<__nv_bfloat16> k_proj;
    DeviceBuffer<__nv_bfloat16> v_proj;
    DeviceBuffer<__nv_bfloat16> o_proj;
    DeviceBuffer<__nv_bfloat16> q_norm;
    DeviceBuffer<__nv_bfloat16> k_norm;
    DeviceBuffer<__nv_bfloat16> post_attention_norm;
    DeviceBuffer<__nv_bfloat16> gate_proj;
    DeviceBuffer<__nv_bfloat16> up_proj;
    DeviceBuffer<__nv_bfloat16> down_proj;
};

void cublas_check(cublasStatus_t status, std::string_view operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed with cuBLAS status " +
                                 std::to_string(static_cast<int>(status)));
    }
}

class CublasHandle {
public:
    CublasHandle() { cublas_check(cublasCreate(&handle_), "cublasCreate"); }
    ~CublasHandle() { if (handle_ != nullptr) cublasDestroy(handle_); }
    CublasHandle(const CublasHandle&) = delete;
    [[nodiscard]] cublasHandle_t get() const { return handle_; }

private:
    cublasHandle_t handle_ = nullptr;
};

// Applies a PyTorch/safetensors linear weight [output_features, input_features]
// to row-major activations [tokens, input_features]. cuBLAS sees the same
// storage as column-major matrices and computes W * X^T.
void linear_bf16(cublasHandle_t handle,
                 const __nv_bfloat16* input,
                 const __nv_bfloat16* weight,
                 __nv_bfloat16* output,
                 std::int32_t tokens,
                 std::int32_t input_features,
                 std::int32_t output_features) {
    const float alpha = 1.0F;
    const float beta = 0.0F;
    cublas_check(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                               output_features, tokens, input_features,
                               &alpha,
                               weight, CUDA_R_16BF, input_features,
                               input, CUDA_R_16BF, input_features,
                               &beta,
                               output, CUDA_R_16BF, output_features,
                               CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
                 "cublasGemmEx linear projection");
}

std::string layer_tensor_name(std::int32_t layer, std::string_view suffix) {
    return "model.layers." + std::to_string(layer) + "." + std::string(suffix);
}

}  // namespace

namespace inference {

InferenceEngine::InferenceEngine(std::filesystem::path model_path)
    : model_path_(std::move(model_path)) {}

GenerationResult InferenceEngine::generate(
    std::span<const std::int32_t> prompt_token_ids,
    std::int32_t max_new_tokens) const {
    if (prompt_token_ids.empty()) {
        throw std::runtime_error("Inference requires at least one prompt token");
    }
    if (max_new_tokens <= 0) {
        throw std::runtime_error("max_new_tokens must be positive");
    }

    std::vector<std::uint64_t> token_ids;
    token_ids.reserve(prompt_token_ids.size() + static_cast<std::size_t>(max_new_tokens));
    for (const auto token_id : prompt_token_ids) {
        if (token_id < 0) throw std::runtime_error("Token IDs must be non-negative");
        token_ids.push_back(static_cast<std::uint64_t>(token_id));
    }

        // Model metadata and host data
        const SafetensorFile model(model_path_);
        constexpr char embedding_name[] = "model.embed_tokens.weight";
        const auto& embedding = model.tensor(embedding_name);
        const auto& attention_norm = model.tensor(first_attention_norm_name);

        if (embedding.dtype != "BF16" || embedding.shape.size() != 2) {
            throw std::runtime_error("Expected a two-dimensional BF16 embedding tensor");
        }
        const auto vocab_size = embedding.shape[0];
        const auto hidden_size = embedding.shape[1];
        if (vocab_size == 0 || hidden_size == 0 ||
            hidden_size > std::numeric_limits<std::uint64_t>::max() / vocab_size / 2 ||
            embedding.byte_size() != vocab_size * hidden_size * 2) {
            throw std::runtime_error("Embedding tensor has an unexpected byte size");
        }
        if (attention_norm.dtype != "BF16" || attention_norm.shape.size() != 1 ||
            attention_norm.shape[0] != hidden_size ||
            attention_norm.byte_size() != hidden_size * sizeof(__nv_bfloat16)) {
            throw std::runtime_error("First attention RMSNorm weight has an unexpected shape or dtype");
        }
        for (const auto token_id : token_ids) {
            if (token_id >= vocab_size || token_id > std::numeric_limits<std::int32_t>::max()) {
                throw std::out_of_range("Token ID " + std::to_string(token_id) +
                                        " is outside the vocabulary of " + std::to_string(vocab_size));
            }
        }
        if (token_ids.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
            throw std::runtime_error("Too many input tokens for the CUDA kernel");
        }

        GenerationResult result;
        result.model = ModelMetadata{
            .path = model_path_,
            .tensor_count = model.tensors().size(),
            .embedding = TensorMetadata{
                .name = embedding_name,
                .dtype = embedding.dtype,
                .shape = embedding.shape,
                .bytes = embedding.byte_size(),
            },
            .first_attention_norm = TensorMetadata{
                .name = first_attention_norm_name,
                .dtype = attention_norm.dtype,
                .shape = attention_norm.shape,
                .bytes = attention_norm.byte_size(),
            },
        };

        const auto prompt_tokens = token_ids.size();
        const auto maximum_cached_tokens =
            prompt_tokens + static_cast<std::size_t>(max_new_tokens) - 1;
        if (maximum_cached_tokens >
            static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
            throw std::runtime_error("Prompt and generated tokens exceed the int32 sequence limit");
        }
        if (hidden_size != inference::qwen3_0_6b_hidden_size) {
            throw std::runtime_error("This executable currently targets Qwen3-0.6B's hidden size");
        }
        if (vocab_size > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
            throw std::runtime_error("Vocabulary size exceeds the cuBLAS int32 limit");
        }

        constexpr auto attention = inference::qwen3_0_6b_attention;
        const std::size_t kv_values_per_token =
            static_cast<std::size_t>(attention.key_value_heads) * attention.head_dim;
        if (maximum_cached_tokens >
            std::numeric_limits<std::size_t>::max() / kv_values_per_token) {
            throw std::runtime_error("KV cache size overflow");
        }
        const std::size_t cache_values = maximum_cached_tokens * kv_values_per_token;

        // One persistent K and V allocation per transformer layer. Prefill writes
        // the prompt at offset zero; every decode pass appends exactly one token.
        std::vector<DeviceBuffer<__nv_bfloat16>> key_caches;
        std::vector<DeviceBuffer<__nv_bfloat16>> value_caches;
        key_caches.reserve(qwen_layer_count);
        value_caches.reserve(qwen_layer_count);
        for (std::int32_t layer = 0; layer < qwen_layer_count; ++layer) {
            key_caches.emplace_back(cache_values);
            value_caches.emplace_back(cache_values);
        }

        std::vector<std::int32_t> host_token_ids;
        host_token_ids.reserve(token_ids.size());
        for (const auto token_id : token_ids) {
            host_token_ids.push_back(static_cast<std::int32_t>(token_id));
        }

        const auto upload_bf16 = [&model](const std::string& name,
                                          std::initializer_list<std::uint64_t> expected_shape) {
            const auto& tensor = model.tensor(name);
            const std::vector<std::uint64_t> shape(expected_shape);
            if (tensor.dtype != "BF16" || tensor.shape != shape ||
                tensor.byte_size() != std::accumulate(shape.begin(), shape.end(),
                                                       std::uint64_t{1}, std::multiplies<>{}) *
                                      sizeof(__nv_bfloat16)) {
                throw std::runtime_error("Unexpected shape or dtype for " + name);
            }
            const auto bytes = model.read_tensor(name);
            DeviceBuffer<__nv_bfloat16> device_weight(bytes.size() / sizeof(__nv_bfloat16));
            cuda_check(cudaMemcpy(device_weight.get(), bytes.data(), bytes.size(), cudaMemcpyHostToDevice),
                       "cudaMemcpy weight H2D");
            return device_weight;
        };

        const auto device_embedding =
            upload_bf16(embedding_name, {vocab_size, hidden_size});

        // Keep every transformer parameter resident for the whole request.
        // Loading happens once here, before either prefill or token decoding.
        std::vector<LayerWeights> layer_weights;
        layer_weights.reserve(qwen_layer_count);
        for (std::int32_t layer = 0; layer < qwen_layer_count; ++layer) {
            layer_weights.push_back(LayerWeights{
                .input_norm = upload_bf16(
                    layer_tensor_name(layer, "input_layernorm.weight"), {hidden_size}),
                .q_proj = upload_bf16(
                    layer_tensor_name(layer, "self_attn.q_proj.weight"),
                    {attention.query_heads * attention.head_dim, hidden_size}),
                .k_proj = upload_bf16(
                    layer_tensor_name(layer, "self_attn.k_proj.weight"),
                    {attention.key_value_heads * attention.head_dim, hidden_size}),
                .v_proj = upload_bf16(
                    layer_tensor_name(layer, "self_attn.v_proj.weight"),
                    {attention.key_value_heads * attention.head_dim, hidden_size}),
                .o_proj = upload_bf16(
                    layer_tensor_name(layer, "self_attn.o_proj.weight"),
                    {hidden_size, attention.query_heads * attention.head_dim}),
                .q_norm = upload_bf16(
                    layer_tensor_name(layer, "self_attn.q_norm.weight"),
                    {attention.head_dim}),
                .k_norm = upload_bf16(
                    layer_tensor_name(layer, "self_attn.k_norm.weight"),
                    {attention.head_dim}),
                .post_attention_norm = upload_bf16(
                    layer_tensor_name(layer, "post_attention_layernorm.weight"),
                    {hidden_size}),
                .gate_proj = upload_bf16(
                    layer_tensor_name(layer, "mlp.gate_proj.weight"),
                    {inference::qwen3_0_6b_mlp_intermediate_size, hidden_size}),
                .up_proj = upload_bf16(
                    layer_tensor_name(layer, "mlp.up_proj.weight"),
                    {inference::qwen3_0_6b_mlp_intermediate_size, hidden_size}),
                .down_proj = upload_bf16(
                    layer_tensor_name(layer, "mlp.down_proj.weight"),
                    {hidden_size, inference::qwen3_0_6b_mlp_intermediate_size}),
            });
        }

        // Reusable device buffers are sized for prefill, the largest query pass.
        const auto maximum_query_tokens = static_cast<std::int32_t>(prompt_tokens);
        const auto output_values = prompt_tokens * static_cast<std::size_t>(hidden_size);
        const std::size_t query_values = prompt_tokens * attention.query_heads * attention.head_dim;
        const std::size_t new_key_value_values = prompt_tokens * kv_values_per_token;
        const std::size_t intermediate_values =
            prompt_tokens * inference::qwen3_0_6b_mlp_intermediate_size;
        DeviceBuffer<std::int32_t> device_token_ids(prompt_tokens);
        DeviceBuffer<__nv_bfloat16> device_hidden(output_values);
        DeviceBuffer<__nv_bfloat16> device_normed(output_values);
        DeviceBuffer<__nv_bfloat16> device_query(query_values);
        DeviceBuffer<__nv_bfloat16> device_key(new_key_value_values);
        DeviceBuffer<__nv_bfloat16> device_value(new_key_value_values);
        DeviceBuffer<__nv_bfloat16> device_attention(query_values);
        DeviceBuffer<__nv_bfloat16> device_projection(output_values);
        DeviceBuffer<__nv_bfloat16> device_gate(intermediate_values);
        DeviceBuffer<__nv_bfloat16> device_up(intermediate_values);
        DeviceBuffer<__nv_bfloat16> device_activated(intermediate_values);
        DeviceBuffer<__nv_bfloat16> device_logits(vocab_size);

        std::vector<float> rope_cos(maximum_cached_tokens * (attention.head_dim / 2));
        std::vector<float> rope_sin(rope_cos.size());
        for (std::int32_t position = 0;
             position < static_cast<std::int32_t>(maximum_cached_tokens); ++position) {
            for (std::int32_t dim = 0; dim < attention.head_dim / 2; ++dim) {
                const float inverse_frequency = std::pow(attention.rope_theta,
                    -2.0F * static_cast<float>(dim) / static_cast<float>(attention.head_dim));
                const float angle = static_cast<float>(position) * inverse_frequency;
                rope_cos[static_cast<std::size_t>(position) * (attention.head_dim / 2) + dim] = std::cos(angle);
                rope_sin[static_cast<std::size_t>(position) * (attention.head_dim / 2) + dim] = std::sin(angle);
            }
        }
        DeviceBuffer<std::int32_t> device_positions(prompt_tokens);
        DeviceBuffer<float> device_rope_cos(rope_cos.size());
        DeviceBuffer<float> device_rope_sin(rope_sin.size());
        cuda_check(cudaMemcpy(device_rope_cos.get(), rope_cos.data(), rope_cos.size() * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy RoPE cos H2D");
        cuda_check(cudaMemcpy(device_rope_sin.get(), rope_sin.data(), rope_sin.size() * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy RoPE sin H2D");

        CublasHandle cublas;
        const auto final_norm = upload_bf16("model.norm.weight", {hidden_size});
        const auto lm_head = upload_bf16("lm_head.weight", {vocab_size, hidden_size});
        std::vector<__nv_bfloat16> last_logits(vocab_size);

        result.token_ids.reserve(max_new_tokens);
        result.steps.reserve(max_new_tokens);
        for (std::int32_t generated = 0; generated < max_new_tokens; ++generated) {
        const bool is_prefill = generated == 0;
        const auto query_tokens = is_prefill ? maximum_query_tokens : 1;
        const auto query_start_position = is_prefill
            ? 0
            : static_cast<std::int32_t>(prompt_tokens) + generated - 1;
        const auto key_value_tokens = query_start_position + query_tokens;

        std::vector<std::int32_t> positions(query_tokens);
        std::iota(positions.begin(), positions.end(), query_start_position);
        if (is_prefill) {
            cuda_check(cudaMemcpy(device_token_ids.get(), host_token_ids.data(),
                                  prompt_tokens * sizeof(std::int32_t), cudaMemcpyHostToDevice),
                       "cudaMemcpy prefill token IDs H2D");
        } else {
            const auto decode_token = static_cast<std::int32_t>(token_ids.back());
            cuda_check(cudaMemcpy(device_token_ids.get(), &decode_token, sizeof(decode_token),
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy decode token ID H2D");
        }
        cuda_check(cudaMemcpy(device_positions.get(), positions.data(),
                              positions.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice),
                   "cudaMemcpy positions H2D");

        cudaEvent_t start_event = nullptr;
        cudaEvent_t stop_event = nullptr;
        cuda_check(cudaEventCreate(&start_event), "cudaEventCreate start");
        cuda_check(cudaEventCreate(&stop_event), "cudaEventCreate stop");
        cuda_check(cudaEventRecord(start_event), "cudaEventRecord start");
        cuda_check(inference::launch_embedding_gather(
                       device_embedding.get(), device_token_ids.get(), device_hidden.get(), query_tokens,
                       static_cast<std::int32_t>(hidden_size)),
                   "launch_embedding_gather");

        for (std::int32_t layer = 0; layer < qwen_layer_count; ++layer) {
            const auto& layer_weight = layer_weights[layer];

            cuda_check(inference::launch_rms_norm(device_hidden.get(), layer_weight.input_norm.get(), device_normed.get(), query_tokens, hidden_size, rms_norm_epsilon), "attention RMSNorm");
            linear_bf16(cublas.get(), device_normed.get(), layer_weight.q_proj.get(), device_query.get(), query_tokens, hidden_size, attention.query_heads * attention.head_dim);
            linear_bf16(cublas.get(), device_normed.get(), layer_weight.k_proj.get(), device_key.get(), query_tokens, hidden_size, attention.key_value_heads * attention.head_dim);
            linear_bf16(cublas.get(), device_normed.get(), layer_weight.v_proj.get(), device_value.get(), query_tokens, hidden_size, attention.key_value_heads * attention.head_dim);
            cuda_check(inference::launch_rms_norm(device_query.get(), layer_weight.q_norm.get(), device_query.get(), query_tokens * attention.query_heads, attention.head_dim, rms_norm_epsilon), "Q RMSNorm");
            cuda_check(inference::launch_rms_norm(device_key.get(), layer_weight.k_norm.get(), device_key.get(), query_tokens * attention.key_value_heads, attention.head_dim, rms_norm_epsilon), "K RMSNorm");
            cuda_check(inference::launch_rope(device_query.get(), device_positions.get(), device_rope_cos.get(), device_rope_sin.get(), device_query.get(), query_tokens, attention.query_heads, attention.head_dim), "Q RoPE");
            cuda_check(inference::launch_rope(device_key.get(), device_positions.get(), device_rope_cos.get(), device_rope_sin.get(), device_key.get(), query_tokens, attention.key_value_heads, attention.head_dim), "K RoPE");

            const auto cache_offset = static_cast<std::size_t>(query_start_position) * kv_values_per_token;
            const auto append_bytes = static_cast<std::size_t>(query_tokens) *
                                      kv_values_per_token * sizeof(__nv_bfloat16);
            cuda_check(cudaMemcpyAsync(key_caches[layer].get() + cache_offset, device_key.get(),
                                       append_bytes, cudaMemcpyDeviceToDevice),
                       "append key cache");
            cuda_check(cudaMemcpyAsync(value_caches[layer].get() + cache_offset, device_value.get(),
                                       append_bytes, cudaMemcpyDeviceToDevice),
                       "append value cache");
            cuda_check(inference::launch_causal_attention(device_query.get(), key_caches[layer].get(), value_caches[layer].get(), device_attention.get(), query_tokens, key_value_tokens, query_start_position, attention.query_heads, attention.key_value_heads, attention.head_dim, inference::qwen3_0_6b_attention_scale), "causal attention");
            linear_bf16(cublas.get(), device_attention.get(), layer_weight.o_proj.get(), device_projection.get(), query_tokens, attention.query_heads * attention.head_dim, hidden_size);
            cuda_check(inference::launch_residual_add(device_hidden.get(), device_projection.get(), query_tokens, hidden_size), "attention residual");

            cuda_check(inference::launch_rms_norm(device_hidden.get(), layer_weight.post_attention_norm.get(), device_normed.get(), query_tokens, hidden_size, rms_norm_epsilon), "MLP RMSNorm");
            linear_bf16(cublas.get(), device_normed.get(), layer_weight.gate_proj.get(), device_gate.get(), query_tokens, hidden_size, inference::qwen3_0_6b_mlp_intermediate_size);
            linear_bf16(cublas.get(), device_normed.get(), layer_weight.up_proj.get(), device_up.get(), query_tokens, hidden_size, inference::qwen3_0_6b_mlp_intermediate_size);
            cuda_check(inference::launch_swiglu(device_gate.get(), device_up.get(), device_activated.get(), query_tokens, inference::qwen3_0_6b_mlp_intermediate_size), "SwiGLU");
            linear_bf16(cublas.get(), device_activated.get(), layer_weight.down_proj.get(), device_projection.get(), query_tokens, inference::qwen3_0_6b_mlp_intermediate_size, hidden_size);
            cuda_check(inference::launch_residual_add(device_hidden.get(), device_projection.get(), query_tokens, hidden_size), "MLP residual");
        }

        // Generation only consumes the final query row, so avoid materializing
        // prompt-length logits during prefill.
        const auto last_hidden_offset = static_cast<std::size_t>(query_tokens - 1) * hidden_size;
        cuda_check(inference::launch_rms_norm(device_hidden.get() + last_hidden_offset,
                                              final_norm.get(), device_normed.get(), 1,
                                              hidden_size, rms_norm_epsilon), "final RMSNorm");
        linear_bf16(cublas.get(), device_normed.get(), lm_head.get(), device_logits.get(),
                    1, hidden_size, static_cast<std::int32_t>(vocab_size));
        cuda_check(cudaEventRecord(stop_event), "cudaEventRecord stop");
        cuda_check(cudaEventSynchronize(stop_event), "cudaEventSynchronize stop");
        float pass_milliseconds = 0.0F;
        cuda_check(cudaEventElapsedTime(&pass_milliseconds, start_event, stop_event), "cudaEventElapsedTime");

        cuda_check(cudaMemcpy(last_logits.data(), device_logits.get(),
                              vocab_size * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost), "cudaMemcpy logits D2H");
        const auto next_token = static_cast<std::size_t>(std::max_element(last_logits.begin(), last_logits.end(),
            [](__nv_bfloat16 left, __nv_bfloat16 right) { return __bfloat162float(left) < __bfloat162float(right); }) - last_logits.begin());
        cuda_check(cudaEventDestroy(start_event), "cudaEventDestroy start");
        cuda_check(cudaEventDestroy(stop_event), "cudaEventDestroy stop");
        result.token_ids.push_back(static_cast<std::int32_t>(next_token));
        result.steps.push_back(GenerationStep{
            .index = generated + 1,
            .token_id = static_cast<std::int32_t>(next_token),
            .sequence_length = key_value_tokens,
            .is_prefill = is_prefill,
            .gpu_milliseconds = pass_milliseconds,
        });
        if (next_token == qwen_eos_token_id) {
            result.stopped_on_eos = true;
            break;
        }
        token_ids.push_back(next_token);
        }
        return result;
}

}  // namespace inference
