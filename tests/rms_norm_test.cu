#include "kernels.cuh"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <iostream>
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

    ~DeviceBuffer() {
        if (data_ != nullptr) cudaFree(data_);
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    [[nodiscard]] T* get() const { return data_; }

private:
    T* data_ = nullptr;
};

void test_rms_norm(std::int32_t num_tokens, std::int32_t hidden_size, bool zero_input = false) {
    constexpr float epsilon = 1.0e-6F;
    std::vector<__nv_bfloat16> input(num_tokens * hidden_size);
    std::vector<__nv_bfloat16> weight(hidden_size);
    for (std::int32_t dim = 0; dim < hidden_size; ++dim) {
        const float scale = dim % 11 == 0 ? 0.0F :
                            0.25F + 0.05F * static_cast<float>(dim % 9);
        weight[dim] = __float2bfloat16(scale);
    }
    for (std::int32_t index = 0; index < num_tokens * hidden_size; ++index) {
        const float value = zero_input ? 0.0F :
                            (index % 2 == 0 ? 1.0F : -1.0F) *
                                (0.03125F * static_cast<float>(1 + (index * 19) % 67));
        input[index] = __float2bfloat16(value);
    }

    DeviceBuffer<__nv_bfloat16> device_input(input.size());
    DeviceBuffer<__nv_bfloat16> device_weight(weight.size());
    DeviceBuffer<__nv_bfloat16> device_output(input.size());
    cuda_check(cudaMemcpy(device_input.get(), input.data(), input.size() * sizeof(__nv_bfloat16),
                          cudaMemcpyHostToDevice), "cudaMemcpy input H2D");
    cuda_check(cudaMemcpy(device_weight.get(), weight.data(), weight.size() * sizeof(__nv_bfloat16),
                          cudaMemcpyHostToDevice), "cudaMemcpy weight H2D");
    cuda_check(inference::launch_rms_norm(device_input.get(), device_weight.get(), device_output.get(),
                                          num_tokens, hidden_size, epsilon),
               "launch_rms_norm");
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    std::vector<__nv_bfloat16> output(input.size());
    cuda_check(cudaMemcpy(output.data(), device_output.get(), output.size() * sizeof(__nv_bfloat16),
                          cudaMemcpyDeviceToHost), "cudaMemcpy output D2H");

    for (std::int32_t row = 0; row < num_tokens; ++row) {
        float sum_squares = 0.0F;
        for (std::int32_t dim = 0; dim < hidden_size; ++dim) {
            const float value = __bfloat162float(input[row * hidden_size + dim]);
            sum_squares += value * value;
        }
        const float inv_rms = 1.0F / std::sqrt(sum_squares / hidden_size + epsilon);
        for (std::int32_t dim = 0; dim < hidden_size; ++dim) {
            const float expected = __bfloat162float(input[row * hidden_size + dim]) * inv_rms *
                                   __bfloat162float(weight[dim]);
            const float actual = __bfloat162float(output[row * hidden_size + dim]);
            if (!std::isfinite(actual) || std::fabs(actual - expected) > 0.02F) {
                throw std::runtime_error("RMSNorm output did not match the FP32 reference");
            }
        }
    }
}

void test_invalid_arguments() {
    constexpr std::int32_t hidden_size = 8;
    constexpr float epsilon = 1.0e-6F;
    DeviceBuffer<__nv_bfloat16> input(hidden_size);
    DeviceBuffer<__nv_bfloat16> weight(hidden_size);
    DeviceBuffer<__nv_bfloat16> output(hidden_size);

    const auto expect_invalid = [](cudaError_t status) {
        if (status != cudaErrorInvalidValue) {
            throw std::runtime_error("RMSNorm launcher accepted invalid arguments");
        }
    };
    expect_invalid(inference::launch_rms_norm(nullptr, weight.get(), output.get(), 1, hidden_size,
                                              epsilon));
    expect_invalid(inference::launch_rms_norm(input.get(), weight.get(), output.get(), 0, hidden_size,
                                              epsilon));
    expect_invalid(inference::launch_rms_norm(input.get(), weight.get(), output.get(), 1, 3, epsilon));
    expect_invalid(inference::launch_rms_norm(input.get(), weight.get(), output.get(), 1, 2048, epsilon));
    expect_invalid(inference::launch_rms_norm(input.get(), weight.get(), output.get(), 1, hidden_size,
                                              0.0F));
}

}  // namespace

int main() {
    try {
        // Smallest supported power-of-two reductions.
        test_rms_norm(1, 1);
        test_rms_norm(2, 2);
        test_rms_norm(1, 8);
        test_rms_norm(3, 8);
        test_rms_norm(5, 32);
        test_rms_norm(2, 256);
        // Exercise Qwen3-0.6B's real 1024-thread launch geometry, including
        // a multi-row case and an all-zero input (epsilon-only denominator).
        test_rms_norm(1, 1024);
        test_rms_norm(3, 1024);
        test_rms_norm(2, 1024, true);
        test_invalid_arguments();
        std::cout << "rms_norm_test: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "rms_norm_test: FAIL: " << error.what() << '\n';
        return 1;
    }
}
