#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace inference {

struct TensorInfo {
    std::string dtype;
    std::vector<std::uint64_t> shape;
    std::uint64_t begin;
    std::uint64_t end;

    [[nodiscard]] std::uint64_t byte_size() const;
};

// A small reader for the safetensors format. Tensor offsets are relative to the
// beginning of the data section (immediately after the JSON header).
class SafetensorFile {
public:
    explicit SafetensorFile(const std::filesystem::path& path);

    [[nodiscard]] const TensorInfo& tensor(const std::string& name) const;
    [[nodiscard]] bool contains(const std::string& name) const;
    [[nodiscard]] const std::unordered_map<std::string, TensorInfo>& tensors() const;
    [[nodiscard]] std::vector<std::byte> read_tensor(const std::string& name) const;

private:
    std::filesystem::path path_;
    std::uint64_t data_offset_{};
    std::unordered_map<std::string, TensorInfo> tensors_;
};

}  // namespace inference
