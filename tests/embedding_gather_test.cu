#include "kernels.cuh"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

// Test helpers
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

// Embedding gather test case
void test_gather_rows(const std::vector<std::int32_t>& token_ids) {
    constexpr std::int32_t vocab_size = 4;
    constexpr std::int32_t hidden_size = 8;
    std::array<__nv_bfloat16, vocab_size * hidden_size> table;
    for (std::int32_t index = 0; index < vocab_size * hidden_size; ++index) {
        table[index] = __float2bfloat16(static_cast<float>(index));
    }

    DeviceBuffer<__nv_bfloat16> device_table(table.size());
    DeviceBuffer<std::int32_t> device_token_ids(token_ids.size());
    DeviceBuffer<__nv_bfloat16> device_output(token_ids.size() * hidden_size);
    cuda_check(cudaMemcpy(device_table.get(), table.data(), sizeof(table), cudaMemcpyHostToDevice),
               "cudaMemcpy table H2D");
    cuda_check(cudaMemcpy(device_token_ids.get(), token_ids.data(),
                          token_ids.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice),
               "cudaMemcpy token IDs H2D");

    cuda_check(inference::launch_embedding_gather(
                   device_table.get(), device_token_ids.get(), device_output.get(),
                   static_cast<std::int32_t>(token_ids.size()), hidden_size),
               "launch_embedding_gather");
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    std::vector<__nv_bfloat16> output(token_ids.size() * hidden_size);
    cuda_check(cudaMemcpy(output.data(), device_output.get(), output.size() * sizeof(__nv_bfloat16),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy output D2H");

    for (std::size_t position = 0; position < token_ids.size(); ++position) {
        const auto* expected = table.data() + token_ids[position] * hidden_size;
        const auto* actual = output.data() + position * hidden_size;
        if (std::memcmp(actual, expected, hidden_size * sizeof(__nv_bfloat16)) != 0) {
            throw std::runtime_error("gathered row did not match expected embedding row");
        }
    }
}

}  // namespace

int main() {
    try {
        // Multiple blocks, non-sequential vocabulary IDs.
        test_gather_rows({2, 0, 3});
        // One-block case.
        test_gather_rows({1});
        std::cout << "embedding_gather_test: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "embedding_gather_test: FAIL: " << error.what() << '\n';
        return 1;
    }
}
