#include "kernels.cuh"
#include "safetensor.hpp"

#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
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

void print_shape(const std::vector<std::uint64_t>& shape) {
    std::cout << '[';
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) std::cout << ", ";
        std::cout << shape[i];
    }
    std::cout << ']';
}

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

[[nodiscard]] std::uint64_t parse_token_id(std::string_view text) {
    if (text.empty() || text.front() == '-') {
        throw std::runtime_error("Token IDs must be non-negative integers: " + std::string(text));
    }
    std::size_t parsed{};
    const auto value = std::stoull(std::string(text), &parsed);
    if (parsed != text.size()) {
        throw std::runtime_error("Invalid token ID: " + std::string(text));
    }
    return value;
}

void print_usage(const char* program) {
    std::cerr << "Usage: " << program
              << " [--model PATH] [--max-new-tokens COUNT] --tokens TOKEN_ID [TOKEN_ID ...]\n";
}

}  // namespace

int main(int argc, char** argv) {
    // Command-line input
    std::filesystem::path model_path = "models/Qwen3-0.6B/model.safetensors";
    std::vector<std::uint64_t> token_ids;
    std::int32_t max_new_tokens = 1;

    try {
        for (int index = 1; index < argc; ++index) {
            const std::string_view argument = argv[index];
            if (argument == "--model") {
                if (++index == argc) {
                    print_usage(argv[0]);
                    return 1;
                }
                model_path = argv[index];
            } else if (argument == "--max-new-tokens") {
                if (++index == argc) {
                    print_usage(argv[0]);
                    return 1;
                }
                const auto count = parse_token_id(argv[index]);
                if (count == 0 || count > std::numeric_limits<std::int32_t>::max()) {
                    throw std::runtime_error("--max-new-tokens must be a positive int32 value");
                }
                max_new_tokens = static_cast<std::int32_t>(count);
            } else if (argument == "--tokens") {
                while (++index < argc) {
                    token_ids.push_back(parse_token_id(argv[index]));
                }
            } else {
                print_usage(argv[0]);
                return 1;
            }
        }
        if (token_ids.empty()) {
            print_usage(argv[0]);
            return 1;
        }

        // Model metadata and host data
        const inference::SafetensorFile model(model_path);
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

        std::cout << "Loaded " << model_path << " (" << model.tensors().size() << " tensors)\n";
        std::cout << embedding_name << ": dtype=" << embedding.dtype << ", shape=";
        print_shape(embedding.shape);
        std::cout << ", bytes=" << embedding.byte_size() << '\n';
        std::cout << first_attention_norm_name << ": dtype=" << attention_norm.dtype << ", shape=";
        print_shape(attention_norm.shape);
        std::cout << ", bytes=" << attention_norm.byte_size() << '\n';

        for (std::int32_t generated = 0; generated < max_new_tokens; ++generated) {
        const auto weights = model.read_tensor(embedding_name);
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

        // Device buffers and embedding H2D copy.
        const auto embedding_values = embedding.byte_size() / sizeof(__nv_bfloat16);
        const auto tokens = static_cast<std::int32_t>(host_token_ids.size());
        const auto output_values = host_token_ids.size() * static_cast<std::size_t>(hidden_size);
        DeviceBuffer<__nv_bfloat16> device_embedding(embedding_values);
        DeviceBuffer<std::int32_t> device_token_ids(host_token_ids.size());
        DeviceBuffer<__nv_bfloat16> device_hidden(output_values);
        DeviceBuffer<__nv_bfloat16> device_normed(output_values);
        cuda_check(cudaMemcpy(device_embedding.get(), weights.data(), weights.size(), cudaMemcpyHostToDevice),
                   "cudaMemcpy embedding table H2D");
        cuda_check(cudaMemcpy(device_token_ids.get(), host_token_ids.data(),
                              host_token_ids.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice),
                   "cudaMemcpy token IDs H2D");
        cudaEvent_t start_event = nullptr;
        cudaEvent_t stop_event = nullptr;
        cuda_check(cudaEventCreate(&start_event), "cudaEventCreate start");
        cuda_check(cudaEventCreate(&stop_event), "cudaEventCreate stop");
        cuda_check(cudaEventRecord(start_event), "cudaEventRecord start");
        cuda_check(inference::launch_embedding_gather(
                       device_embedding.get(), device_token_ids.get(), device_hidden.get(), tokens,
                       static_cast<std::int32_t>(hidden_size)),
                   "launch_embedding_gather");

        if (hidden_size != inference::qwen3_0_6b_hidden_size) {
            throw std::runtime_error("This executable currently targets Qwen3-0.6B's hidden size");
        }
        constexpr auto attention = inference::qwen3_0_6b_attention;
        const std::size_t query_values = static_cast<std::size_t>(tokens) * attention.query_heads * attention.head_dim;
        const std::size_t key_value_values = static_cast<std::size_t>(tokens) * attention.key_value_heads * attention.head_dim;
        const std::size_t intermediate_values = static_cast<std::size_t>(tokens) * inference::qwen3_0_6b_mlp_intermediate_size;
        DeviceBuffer<__nv_bfloat16> device_query(query_values);
        DeviceBuffer<__nv_bfloat16> device_key(key_value_values);
        DeviceBuffer<__nv_bfloat16> device_value(key_value_values);
        DeviceBuffer<__nv_bfloat16> device_attention(query_values);
        DeviceBuffer<__nv_bfloat16> device_projection(output_values);
        DeviceBuffer<__nv_bfloat16> device_gate(intermediate_values);
        DeviceBuffer<__nv_bfloat16> device_up(intermediate_values);
        DeviceBuffer<__nv_bfloat16> device_activated(intermediate_values);

        std::vector<std::int32_t> positions(tokens);
        std::vector<float> rope_cos(static_cast<std::size_t>(tokens) * (attention.head_dim / 2));
        std::vector<float> rope_sin(rope_cos.size());
        for (std::int32_t position = 0; position < tokens; ++position) {
            positions[position] = position;
            for (std::int32_t dim = 0; dim < attention.head_dim / 2; ++dim) {
                const float inverse_frequency = std::pow(attention.rope_theta,
                    -2.0F * static_cast<float>(dim) / static_cast<float>(attention.head_dim));
                const float angle = static_cast<float>(position) * inverse_frequency;
                rope_cos[static_cast<std::size_t>(position) * (attention.head_dim / 2) + dim] = std::cos(angle);
                rope_sin[static_cast<std::size_t>(position) * (attention.head_dim / 2) + dim] = std::sin(angle);
            }
        }
        DeviceBuffer<std::int32_t> device_positions(positions.size());
        DeviceBuffer<float> device_rope_cos(rope_cos.size());
        DeviceBuffer<float> device_rope_sin(rope_sin.size());
        cuda_check(cudaMemcpy(device_positions.get(), positions.data(), positions.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice), "cudaMemcpy positions H2D");
        cuda_check(cudaMemcpy(device_rope_cos.get(), rope_cos.data(), rope_cos.size() * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy RoPE cos H2D");
        cuda_check(cudaMemcpy(device_rope_sin.get(), rope_sin.data(), rope_sin.size() * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy RoPE sin H2D");

        CublasHandle cublas;
        for (std::int32_t layer = 0; layer < qwen_layer_count; ++layer) {
            const auto input_norm = upload_bf16(layer_tensor_name(layer, "input_layernorm.weight"), {hidden_size});
            const auto q_weight = upload_bf16(layer_tensor_name(layer, "self_attn.q_proj.weight"), {attention.query_heads * attention.head_dim, hidden_size});
            const auto k_weight = upload_bf16(layer_tensor_name(layer, "self_attn.k_proj.weight"), {attention.key_value_heads * attention.head_dim, hidden_size});
            const auto v_weight = upload_bf16(layer_tensor_name(layer, "self_attn.v_proj.weight"), {attention.key_value_heads * attention.head_dim, hidden_size});
            const auto o_weight = upload_bf16(layer_tensor_name(layer, "self_attn.o_proj.weight"), {hidden_size, attention.query_heads * attention.head_dim});
            const auto q_norm = upload_bf16(layer_tensor_name(layer, "self_attn.q_norm.weight"), {attention.head_dim});
            const auto k_norm = upload_bf16(layer_tensor_name(layer, "self_attn.k_norm.weight"), {attention.head_dim});

            cuda_check(inference::launch_rms_norm(device_hidden.get(), input_norm.get(), device_normed.get(), tokens, hidden_size, rms_norm_epsilon), "attention RMSNorm");
            linear_bf16(cublas.get(), device_normed.get(), q_weight.get(), device_query.get(), tokens, hidden_size, attention.query_heads * attention.head_dim);
            linear_bf16(cublas.get(), device_normed.get(), k_weight.get(), device_key.get(), tokens, hidden_size, attention.key_value_heads * attention.head_dim);
            linear_bf16(cublas.get(), device_normed.get(), v_weight.get(), device_value.get(), tokens, hidden_size, attention.key_value_heads * attention.head_dim);
            cuda_check(inference::launch_rms_norm(device_query.get(), q_norm.get(), device_query.get(), tokens * attention.query_heads, attention.head_dim, rms_norm_epsilon), "Q RMSNorm");
            cuda_check(inference::launch_rms_norm(device_key.get(), k_norm.get(), device_key.get(), tokens * attention.key_value_heads, attention.head_dim, rms_norm_epsilon), "K RMSNorm");
            cuda_check(inference::launch_rope(device_query.get(), device_positions.get(), device_rope_cos.get(), device_rope_sin.get(), device_query.get(), tokens, attention.query_heads, attention.head_dim), "Q RoPE");
            cuda_check(inference::launch_rope(device_key.get(), device_positions.get(), device_rope_cos.get(), device_rope_sin.get(), device_key.get(), tokens, attention.key_value_heads, attention.head_dim), "K RoPE");
            cuda_check(inference::launch_causal_attention(device_query.get(), device_key.get(), device_value.get(), device_attention.get(), tokens, tokens, 0, attention.query_heads, attention.key_value_heads, attention.head_dim, inference::qwen3_0_6b_attention_scale), "causal attention");
            linear_bf16(cublas.get(), device_attention.get(), o_weight.get(), device_projection.get(), tokens, attention.query_heads * attention.head_dim, hidden_size);
            cuda_check(inference::launch_residual_add(device_hidden.get(), device_projection.get(), tokens, hidden_size), "attention residual");

            const auto post_norm = upload_bf16(layer_tensor_name(layer, "post_attention_layernorm.weight"), {hidden_size});
            const auto gate_weight = upload_bf16(layer_tensor_name(layer, "mlp.gate_proj.weight"), {inference::qwen3_0_6b_mlp_intermediate_size, hidden_size});
            const auto up_weight = upload_bf16(layer_tensor_name(layer, "mlp.up_proj.weight"), {inference::qwen3_0_6b_mlp_intermediate_size, hidden_size});
            const auto down_weight = upload_bf16(layer_tensor_name(layer, "mlp.down_proj.weight"), {hidden_size, inference::qwen3_0_6b_mlp_intermediate_size});
            cuda_check(inference::launch_rms_norm(device_hidden.get(), post_norm.get(), device_normed.get(), tokens, hidden_size, rms_norm_epsilon), "MLP RMSNorm");
            linear_bf16(cublas.get(), device_normed.get(), gate_weight.get(), device_gate.get(), tokens, hidden_size, inference::qwen3_0_6b_mlp_intermediate_size);
            linear_bf16(cublas.get(), device_normed.get(), up_weight.get(), device_up.get(), tokens, hidden_size, inference::qwen3_0_6b_mlp_intermediate_size);
            cuda_check(inference::launch_swiglu(device_gate.get(), device_up.get(), device_activated.get(), tokens, inference::qwen3_0_6b_mlp_intermediate_size), "SwiGLU");
            linear_bf16(cublas.get(), device_activated.get(), down_weight.get(), device_projection.get(), tokens, inference::qwen3_0_6b_mlp_intermediate_size, hidden_size);
            cuda_check(inference::launch_residual_add(device_hidden.get(), device_projection.get(), tokens, hidden_size), "MLP residual");
        }

        const auto final_norm = upload_bf16("model.norm.weight", {hidden_size});
        const auto lm_head = upload_bf16("lm_head.weight", {vocab_size, hidden_size});
        DeviceBuffer<__nv_bfloat16> device_logits(static_cast<std::size_t>(tokens) * vocab_size);
        cuda_check(inference::launch_rms_norm(device_hidden.get(), final_norm.get(), device_normed.get(), tokens, hidden_size, rms_norm_epsilon), "final RMSNorm");
        linear_bf16(cublas.get(), device_normed.get(), lm_head.get(), device_logits.get(), tokens, hidden_size, static_cast<std::int32_t>(vocab_size));
        cuda_check(cudaEventRecord(stop_event), "cudaEventRecord stop");
        cuda_check(cudaEventSynchronize(stop_event), "cudaEventSynchronize stop");
        float prefill_milliseconds = 0.0F;
        cuda_check(cudaEventElapsedTime(&prefill_milliseconds, start_event, stop_event), "cudaEventElapsedTime");

        std::vector<__nv_bfloat16> last_logits(vocab_size);
        cuda_check(cudaMemcpy(last_logits.data(), device_logits.get() + static_cast<std::size_t>(tokens - 1) * vocab_size,
                              vocab_size * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost), "cudaMemcpy logits D2H");
        const auto next_token = static_cast<std::size_t>(std::max_element(last_logits.begin(), last_logits.end(),
            [](__nv_bfloat16 left, __nv_bfloat16 right) { return __bfloat162float(left) < __bfloat162float(right); }) - last_logits.begin());
        cuda_check(cudaEventDestroy(start_event), "cudaEventDestroy start");
        cuda_check(cudaEventDestroy(stop_event), "cudaEventDestroy stop");
        std::cout << "generated token " << (generated + 1) << ": id=" << next_token
                  << ", sequence_length=" << tokens
                  << ", prefill_gpu_ms=" << std::fixed << std::setprecision(3)
                  << prefill_milliseconds << '\n';
        if (next_token == qwen_eos_token_id) {
            std::cout << "stopped on EOS token " << qwen_eos_token_id << '\n';
            break;
        }
        token_ids.push_back(next_token);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
