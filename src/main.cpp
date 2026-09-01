#include "safetensor.hpp"

#include <exception>
#include <filesystem>
#include <iostream>
#include <string>

namespace {

void print_shape(const std::vector<std::uint64_t>& shape) {
    std::cout << '[';
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) std::cout << ", ";
        std::cout << shape[i];
    }
    std::cout << ']';
}

}  // namespace

int main(int argc, char** argv) {
    const std::filesystem::path model_path =
        argc > 1 ? argv[1] : "models/Qwen3-0.6B/model.safetensors";

    try {
        const inference::SafetensorFile model(model_path);
        constexpr char embedding_name[] = "model.embed_tokens.weight";
        const auto& embedding = model.tensor(embedding_name);

        std::cout << "Loaded " << model_path << " (" << model.tensors().size() << " tensors)\n";
        std::cout << embedding_name << ": dtype=" << embedding.dtype << ", shape=";
        print_shape(embedding.shape);
        std::cout << ", bytes=" << embedding.byte_size() << '\n';

        // The next milestone will copy this tensor and input IDs to the GPU,
        // then invoke the embedding-gather kernel.
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
