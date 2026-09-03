#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace inference {

struct TensorMetadata {
    std::string name;
    std::string dtype;
    std::vector<std::uint64_t> shape;
    std::uint64_t bytes = 0;
};

struct ModelMetadata {
    std::filesystem::path path;
    std::size_t tensor_count = 0;
    TensorMetadata embedding;
    TensorMetadata first_attention_norm;
};

struct GenerationStep {
    std::int32_t index = 0;
    std::int32_t token_id = 0;
    std::int32_t sequence_length = 0;
    bool is_prefill = false;
    float gpu_milliseconds = 0.0F;
};

struct GenerationResult {
    ModelMetadata model;
    std::vector<std::int32_t> token_ids;
    std::vector<GenerationStep> steps;
    bool stopped_on_eos = false;
};

class InferenceEngine {
public:
    explicit InferenceEngine(std::filesystem::path model_path);

    [[nodiscard]] GenerationResult generate(
        std::span<const std::int32_t> prompt_token_ids,
        std::int32_t max_new_tokens) const;

private:
    std::filesystem::path model_path_;
};

}  // namespace inference
