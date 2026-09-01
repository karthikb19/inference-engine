#include "safetensor.hpp"

#include "json.hpp"

#include <fstream>
#include <limits>
#include <stdexcept>

namespace inference {
namespace {

std::uint64_t checked_file_size(const std::filesystem::path& path) {
    std::error_code error;
    const auto size = std::filesystem::file_size(path, error);
    if (error) {
        throw std::runtime_error("Could not determine size of " + path.string() + ": " + error.message());
    }
    return size;
}

}  // namespace

std::uint64_t TensorInfo::byte_size() const {
    return end - begin;
}

SafetensorFile::SafetensorFile(const std::filesystem::path& path) : path_(path) {
    const auto size = checked_file_size(path_);
    if (size < sizeof(std::uint64_t)) {
        throw std::runtime_error("Invalid safetensors file (missing header length): " + path_.string());
    }

    std::ifstream file(path_, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Could not open " + path_.string());
    }

    std::uint64_t header_size{};
    file.read(reinterpret_cast<char*>(&header_size), sizeof(header_size));
    if (!file || header_size > size - sizeof(header_size)) {
        throw std::runtime_error("Invalid safetensors header size in " + path_.string());
    }

    std::string header(header_size, '\0');
    file.read(header.data(), static_cast<std::streamsize>(header.size()));
    if (!file) {
        throw std::runtime_error("Could not read safetensors header from " + path_.string());
    }
    data_offset_ = sizeof(header_size) + header_size;

    const nlohmann::json json = nlohmann::json::parse(header);
    if (!json.is_object()) {
        throw std::runtime_error("Safetensors header is not a JSON object: " + path_.string());
    }

    for (const auto& [name, value] : json.items()) {
        if (name == "__metadata__") {
            continue;
        }
        if (!value.is_object() || !value.contains("dtype") || !value.contains("shape") || !value.contains("data_offsets")) {
            throw std::runtime_error("Invalid tensor entry '" + name + "' in " + path_.string());
        }

        const auto& offsets = value.at("data_offsets");
        if (!offsets.is_array() || offsets.size() != 2) {
            throw std::runtime_error("Invalid data_offsets for tensor '" + name + "'");
        }

        TensorInfo info{
            .dtype = value.at("dtype").get<std::string>(),
            .shape = value.at("shape").get<std::vector<std::uint64_t>>(),
            .begin = offsets.at(0).get<std::uint64_t>(),
            .end = offsets.at(1).get<std::uint64_t>(),
        };
        if (info.end < info.begin || info.end > size - data_offset_) {
            throw std::runtime_error("Out-of-bounds data_offsets for tensor '" + name + "'");
        }
        tensors_.emplace(name, std::move(info));
    }
}

const TensorInfo& SafetensorFile::tensor(const std::string& name) const {
    const auto found = tensors_.find(name);
    if (found == tensors_.end()) {
        throw std::out_of_range("Tensor not found: " + name);
    }
    return found->second;
}

bool SafetensorFile::contains(const std::string& name) const {
    return tensors_.contains(name);
}

const std::unordered_map<std::string, TensorInfo>& SafetensorFile::tensors() const {
    return tensors_;
}

std::vector<std::byte> SafetensorFile::read_tensor(const std::string& name) const {
    const auto& info = tensor(name);
    if (info.byte_size() > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error("Tensor is too large to fit in host memory: " + name);
    }

    std::vector<std::byte> bytes(static_cast<std::size_t>(info.byte_size()));
    std::ifstream file(path_, std::ios::binary);
    file.seekg(static_cast<std::streamoff>(data_offset_ + info.begin));
    file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!file) {
        throw std::runtime_error("Could not read tensor data for: " + name);
    }
    return bytes;
}

}  // namespace inference
