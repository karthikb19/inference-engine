#include "kernels.cuh"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

void cuda_check(cudaError_t status, std::string_view operation) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) { cuda_check(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)), "cudaMalloc"); }
    ~DeviceBuffer() { if (data_ != nullptr) cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    [[nodiscard]] T* get() const { return data_; }
private:
    T* data_ = nullptr;
};

void test_swiglu(std::int32_t tokens, std::int32_t width, bool alias_gate) {
    const std::size_t count = static_cast<std::size_t>(tokens) * width;
    std::vector<__nv_bfloat16> gate(count), up(count);
    std::vector<float> expected(count);
    for (std::size_t index = 0; index < count; ++index) {
        const float gate_value = 0.125F * static_cast<float>(static_cast<int>(index % 41) - 20);
        const float up_value = 0.0625F * static_cast<float>(static_cast<int>((index * 7) % 37) - 18);
        gate[index] = __float2bfloat16(gate_value);
        up[index] = __float2bfloat16(up_value);
        const float rounded_gate = __bfloat162float(gate[index]);
        expected[index] = rounded_gate / (1.0F + std::exp(-rounded_gate)) * __bfloat162float(up[index]);
    }
    DeviceBuffer<__nv_bfloat16> device_gate(count), device_up(count), device_output(count);
    cuda_check(cudaMemcpy(device_gate.get(), gate.data(), count * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice), "gate H2D");
    cuda_check(cudaMemcpy(device_up.get(), up.data(), count * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice), "up H2D");
    __nv_bfloat16* output = alias_gate ? device_gate.get() : device_output.get();
    cuda_check(inference::launch_swiglu(device_gate.get(), device_up.get(), output, tokens, width), "launch_swiglu");
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    std::vector<__nv_bfloat16> actual(count);
    cuda_check(cudaMemcpy(actual.data(), output, count * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost), "output D2H");
    for (std::size_t index = 0; index < count; ++index) {
        if (!std::isfinite(__bfloat162float(actual[index])) ||
            std::fabs(__bfloat162float(actual[index]) - expected[index]) > 0.02F) {
            throw std::runtime_error("SwiGLU output did not match FP32 reference");
        }
    }
}

void test_invalid_arguments() {
    DeviceBuffer<__nv_bfloat16> values(4);
    const auto expect_invalid = [](cudaError_t status) {
        if (status != cudaErrorInvalidValue) throw std::runtime_error("SwiGLU launcher accepted invalid arguments");
    };
    expect_invalid(inference::launch_swiglu(nullptr, values.get(), values.get(), 1, 4));
    expect_invalid(inference::launch_swiglu(values.get(), nullptr, values.get(), 1, 4));
    expect_invalid(inference::launch_swiglu(values.get(), values.get(), nullptr, 1, 4));
    expect_invalid(inference::launch_swiglu(values.get(), values.get(), values.get(), 0, 4));
    expect_invalid(inference::launch_swiglu(values.get(), values.get(), values.get(), 1, 0));
}

}  // namespace

int main() {
    try {
        test_swiglu(1, 7, false);
        test_swiglu(3, inference::qwen3_0_6b_mlp_intermediate_size, true);
        test_invalid_arguments();
        std::cout << "swiglu_test: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "swiglu_test: FAIL: " << error.what() << '\n';
        return 1;
    }
}
