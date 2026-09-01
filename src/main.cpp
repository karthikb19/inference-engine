#include "safetensor.hpp"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace {

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

[[nodiscard]] float bfloat16_to_float(const std::byte* source) {
    const auto low = std::to_integer<std::uint8_t>(source[0]);
    const auto high = std::to_integer<std::uint8_t>(source[1]);
    const std::uint32_t bits = (static_cast<std::uint32_t>(high) << 24) |
                               (static_cast<std::uint32_t>(low) << 16);
    return std::bit_cast<float>(bits);
}

void print_embedding_preview(const std::vector<std::byte>& weights,
                             std::uint64_t token_id,
                             std::uint64_t hidden_size) {
    constexpr std::uint64_t bfloat16_bytes = 2;
    constexpr std::uint64_t preview_values = 8;
    const auto row_offset = token_id * hidden_size * bfloat16_bytes;

    std::cout << "token " << token_id << " -> [";
    for (std::uint64_t dim = 0; dim < std::min(hidden_size, preview_values); ++dim) {
        if (dim != 0) std::cout << ", ";
        std::cout << bfloat16_to_float(weights.data() + row_offset + dim * bfloat16_bytes);
    }
    std::cout << ", ...]\n";
}

void print_usage(const char* program) {
    std::cerr << "Usage: " << program
              << " [--model PATH] --tokens TOKEN_ID [TOKEN_ID ...]\n";
}

}  // namespace

int main(int argc, char** argv) {
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

        const inference::SafetensorFile model(model_path);
        constexpr char embedding_name[] = "model.embed_tokens.weight";
        const auto& embedding = model.tensor(embedding_name);

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
        for (const auto token_id : token_ids) {
            if (token_id >= vocab_size) {
                throw std::out_of_range("Token ID " + std::to_string(token_id) +
                                        " is outside the vocabulary of " + std::to_string(vocab_size));
            }
        }

        std::cout << "Loaded " << model_path << " (" << model.tensors().size() << " tensors)\n";
        std::cout << embedding_name << ": dtype=" << embedding.dtype << ", shape=";
        print_shape(embedding.shape);
        std::cout << ", bytes=" << embedding.byte_size() << '\n';

        // This CPU reference gather will be replaced by a GPU kernel. Each token
        // selects one contiguous row of [hidden_size] BF16 values.
        const auto weights = model.read_tensor(embedding_name);
        for (const auto token_id : token_ids) {
            print_embedding_preview(weights, token_id, hidden_size);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
