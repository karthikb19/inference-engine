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

void test_residual_add(std::int32_t num_tokens, std::int32_t hidden_size) {
    const std::size_t count = static_cast<std::size_t>(num_tokens) * hidden_size;
    std::vector<__nv_bfloat16> input(count);
    std::vector<__nv_bfloat16> residual(count);
    for (std::size_t index = 0; index < count; ++index) {
        input[index] = __float2bfloat16(0.125F * static_cast<float>(static_cast<int>(index % 17) - 8));
        residual[index] = __float2bfloat16(0.0625F * static_cast<float>(static_cast<int>((index * 5) % 19) - 9));
    }

    DeviceBuffer<__nv_bfloat16> device_input(count);
    DeviceBuffer<__nv_bfloat16> device_residual(count);
    cuda_check(cudaMemcpy(device_input.get(), input.data(), count * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice), "input H2D");
    cuda_check(cudaMemcpy(device_residual.get(), residual.data(), count * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice), "residual H2D");
    cuda_check(inference::launch_residual_add(device_input.get(), device_residual.get(), num_tokens, hidden_size), "launch_residual_add");
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    std::vector<__nv_bfloat16> output(count);
    cuda_check(cudaMemcpy(output.data(), device_input.get(), count * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost), "output D2H");
    for (std::size_t index = 0; index < count; ++index) {
        const float expected = __bfloat162float(input[index]) + __bfloat162float(residual[index]);
        const float actual = __bfloat162float(output[index]);
        if (std::fabs(actual - expected) > 0.01F) {
            throw std::runtime_error("residual add output did not match the FP32 reference");
        }
    }
}

void test_invalid_arguments() {
    DeviceBuffer<__nv_bfloat16> values(8);
    const auto expect_invalid = [](cudaError_t status) {
        if (status != cudaErrorInvalidValue) throw std::runtime_error("residual add launcher accepted invalid arguments");
    };
    expect_invalid(inference::launch_residual_add(nullptr, values.get(), 1, 8));
    expect_invalid(inference::launch_residual_add(values.get(), nullptr, 1, 8));
    expect_invalid(inference::launch_residual_add(values.get(), values.get(), 0, 8));
    expect_invalid(inference::launch_residual_add(values.get(), values.get(), 1, 0));
}

}  // namespace

int main() {
    try {
        test_residual_add(1, 1);
        test_residual_add(3, 17);     // partial final CUDA block
        test_residual_add(2, 1024);   // Qwen3-0.6B residual-stream width
        test_invalid_arguments();
        std::cout << "residual_add_test: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "residual_add_test: FAIL: " << error.what() << '\n';
        return 1;
    }
}
