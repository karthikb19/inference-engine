#include "kernels.cuh"
#include "safetensor.hpp"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace {

// RMSNorm dispatch scaffolding.  When the first transformer block is wired
// in, these identify its pre-attention norm and preserve Qwen's epsilon.
constexpr char first_attention_norm_name[] = "model.layers.0.input_layernorm.weight";
constexpr float rms_norm_epsilon = 1.0e-6F;

// Qwen3-0.6B configuration. These will be used when the Q/K projections and
// RoPE frequency tables are added to the first transformer-block dispatch.
constexpr std::int32_t qwen_head_dim = 128;
constexpr std::int32_t qwen_attention_heads = 16;
constexpr std::int32_t qwen_key_value_heads = 8;
constexpr float qwen_rope_theta = 1'000'000.0F;

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
              << " [--model PATH] --tokens TOKEN_ID [TOKEN_ID ...]\n";
}

}  // namespace

int main(int argc, char** argv) {
    // Command-line input
    std::filesystem::path model_path = "models/Qwen3-0.6B/model.safetensors";
    std::vector<std::uint64_t> token_ids;

    try {
        for (int index = 1; index < argc; ++index) {
            const std::string_view argument = argv[index];
            if (argument == "--model") {
                if (++index == argc) {
                    print_usage(argv[0]);
                    return 1;
                }
                model_path = argv[index];
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

        const auto weights = model.read_tensor(embedding_name);
        const auto norm_weights = model.read_tensor(first_attention_norm_name);
        std::vector<std::int32_t> host_token_ids;
        host_token_ids.reserve(token_ids.size());
        for (const auto token_id : token_ids) {
            host_token_ids.push_back(static_cast<std::int32_t>(token_id));
        }

        // Device buffers and H2D copies
        const auto embedding_values = embedding.byte_size() / sizeof(__nv_bfloat16);
        const auto output_values = host_token_ids.size() * static_cast<std::size_t>(hidden_size);
        DeviceBuffer<__nv_bfloat16> device_embedding(embedding_values);
        DeviceBuffer<std::int32_t> device_token_ids(host_token_ids.size());
        DeviceBuffer<__nv_bfloat16> device_output(output_values);
        DeviceBuffer<__nv_bfloat16> device_norm_weight(hidden_size);
        DeviceBuffer<__nv_bfloat16> device_normalized_output(output_values);
        cuda_check(cudaMemcpy(device_embedding.get(), weights.data(), weights.size(), cudaMemcpyHostToDevice),
                   "cudaMemcpy embedding table H2D");
        cuda_check(cudaMemcpy(device_token_ids.get(), host_token_ids.data(),
                              host_token_ids.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice),
                   "cudaMemcpy token IDs H2D");
        cuda_check(cudaMemcpy(device_norm_weight.get(), norm_weights.data(), norm_weights.size(),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy RMSNorm weight H2D");

        // Embedding gather launch
        cuda_check(inference::launch_embedding_gather(
                       device_embedding.get(), device_token_ids.get(), device_output.get(),
                       static_cast<std::int32_t>(host_token_ids.size()),
                       static_cast<std::int32_t>(hidden_size)),
                   "launch_embedding_gather");
        cuda_check(inference::launch_rms_norm(
                       device_output.get(), device_norm_weight.get(), device_normalized_output.get(),
                       static_cast<std::int32_t>(host_token_ids.size()),
                       static_cast<std::int32_t>(hidden_size), rms_norm_epsilon),
                   "launch_rms_norm");

        // Next transformer-block milestone (after the Q/K linear projections):
        //
        //   DeviceBuffer<__nv_bfloat16> device_query(tokens * qwen_attention_heads * qwen_head_dim);
        //   DeviceBuffer<__nv_bfloat16> device_key(tokens * qwen_key_value_heads * qwen_head_dim);
        //   DeviceBuffer<std::int32_t> device_positions(tokens);
        //   DeviceBuffer<float> device_rope_cos(max_positions * (qwen_head_dim / 2));
        //   DeviceBuffer<float> device_rope_sin(max_positions * (qwen_head_dim / 2));
        //
        //   launch_rope(device_query.get(), device_positions.get(), device_rope_cos.get(),
        //               device_rope_sin.get(), device_query.get(), tokens,
        //               qwen_attention_heads, qwen_head_dim);
        //
        // Apply the same operation to device_key with qwen_key_value_heads.
        // Build the shared cos/sin tables once from qwen_rope_theta; position
        // IDs must account for the number of tokens already in the KV cache.
        cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

        std::cout << "Gathered and RMS-normalized " << host_token_ids.size()
                  << " embedding row(s) on the GPU.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
