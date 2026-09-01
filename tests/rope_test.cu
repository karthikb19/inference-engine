#include "kernels.cuh"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

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
    ~DeviceBuffer() { if (data_ != nullptr) cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    [[nodiscard]] T* get() const { return data_; }
private:
    T* data_ = nullptr;
};

void test_rope(std::int32_t num_tokens, std::int32_t num_heads, std::int32_t head_dim) {
    constexpr std::int32_t max_positions = 7;
    const std::int32_t half_dim = head_dim / 2;
    std::vector<__nv_bfloat16> input(num_tokens * num_heads * head_dim);
    std::vector<std::int32_t> positions(num_tokens);
    std::vector<float> cos(max_positions * half_dim);
    std::vector<float> sin(max_positions * half_dim);

    for (std::int32_t token = 0; token < num_tokens; ++token) positions[token] = (token * 3) % max_positions;
    for (std::int32_t position = 0; position < max_positions; ++position) {
        for (std::int32_t dim = 0; dim < half_dim; ++dim) {
            const float angle = 0.13F * position * (dim + 1);
            cos[position * half_dim + dim] = std::cos(angle);
            sin[position * half_dim + dim] = std::sin(angle);
        }
    }
    for (std::size_t index = 0; index < input.size(); ++index) {
        const auto pattern = static_cast<std::int32_t>((index * 7) % 31) - 15;
        input[index] = __float2bfloat16(0.0625F * static_cast<float>(pattern));
    }

    DeviceBuffer<__nv_bfloat16> device_input(input.size());
    DeviceBuffer<std::int32_t> device_positions(positions.size());
    DeviceBuffer<float> device_cos(cos.size());
    DeviceBuffer<float> device_sin(sin.size());
    DeviceBuffer<__nv_bfloat16> device_output(input.size());
    cuda_check(cudaMemcpy(device_input.get(), input.data(), input.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice), "input H2D");
    cuda_check(cudaMemcpy(device_positions.get(), positions.data(), positions.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice), "positions H2D");
    cuda_check(cudaMemcpy(device_cos.get(), cos.data(), cos.size() * sizeof(float), cudaMemcpyHostToDevice), "cos H2D");
    cuda_check(cudaMemcpy(device_sin.get(), sin.data(), sin.size() * sizeof(float), cudaMemcpyHostToDevice), "sin H2D");
    cuda_check(inference::launch_rope(device_input.get(), device_positions.get(), device_cos.get(), device_sin.get(), device_output.get(), num_tokens, num_heads, head_dim), "launch_rope");
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    std::vector<__nv_bfloat16> output(input.size());
    cuda_check(cudaMemcpy(output.data(), device_output.get(), output.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost), "output D2H");
    for (std::int32_t token = 0; token < num_tokens; ++token) {
        for (std::int32_t head = 0; head < num_heads; ++head) {
            const std::int32_t base = (token * num_heads + head) * head_dim;
            for (std::int32_t dim = 0; dim < half_dim; ++dim) {
                const float c = cos[positions[token] * half_dim + dim];
                const float s = sin[positions[token] * half_dim + dim];
                const float x0 = __bfloat162float(input[base + dim]);
                const float x1 = __bfloat162float(input[base + dim + half_dim]);
                const float expected0 = x0 * c - x1 * s;
                const float expected1 = x0 * s + x1 * c;
                if (std::fabs(__bfloat162float(output[base + dim]) - expected0) > 0.01F ||
                    std::fabs(__bfloat162float(output[base + dim + half_dim]) - expected1) > 0.01F) {
                    throw std::runtime_error("RoPE output did not match the FP32 half-split reference");
                }
            }
        }
    }
}

void test_invalid_arguments() {
    DeviceBuffer<__nv_bfloat16> values(8);
    DeviceBuffer<std::int32_t> positions(1);
    DeviceBuffer<float> table(4);
    const auto expect_invalid = [](cudaError_t status) {
        if (status != cudaErrorInvalidValue) throw std::runtime_error("RoPE launcher accepted invalid arguments");
    };
    expect_invalid(inference::launch_rope(nullptr, positions.get(), table.get(), table.get(), values.get(), 1, 1, 8));
    expect_invalid(inference::launch_rope(values.get(), positions.get(), table.get(), table.get(), values.get(), 1, 1, 7));
    expect_invalid(inference::launch_rope(values.get(), positions.get(), table.get(), table.get(), values.get(), 0, 1, 8));
}

}  // namespace

int main() {
    try {
        test_rope(1, 1, 2);
        test_rope(2, 3, 8);
        test_rope(3, 16, 128);  // Qwen3-0.6B Q shape
        test_rope(3, 8, 128);   // Qwen3-0.6B K shape
        test_invalid_arguments();
        std::cout << "rope_test: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "rope_test: FAIL: " << error.what() << '\n';
        return 1;
    }
}
