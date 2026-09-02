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

void test_softmax(std::int32_t rows, std::int32_t columns, bool in_place) {
    std::vector<float> input(static_cast<std::size_t>(rows) * columns);
    for (std::size_t index = 0; index < input.size(); ++index) {
        // Include a deliberately very large range to exercise max subtraction.
        input[index] = static_cast<float>((index * 17) % 23) - 11.0F + (index % columns == 0 ? 80.0F : 0.0F);
    }
    std::vector<float> expected(input.size());
    for (std::int32_t row = 0; row < rows; ++row) {
        float maximum = -INFINITY;
        for (std::int32_t column = 0; column < columns; ++column) maximum = std::fmax(maximum, input[row * columns + column]);
        float sum = 0.0F;
        for (std::int32_t column = 0; column < columns; ++column) {
            expected[row * columns + column] = std::exp(input[row * columns + column] - maximum);
            sum += expected[row * columns + column];
        }
        for (std::int32_t column = 0; column < columns; ++column) expected[row * columns + column] /= sum;
    }
    DeviceBuffer<float> device_input(input.size());
    DeviceBuffer<float> device_output(input.size());
    cuda_check(cudaMemcpy(device_input.get(), input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice), "input H2D");
    float* output = in_place ? device_input.get() : device_output.get();
    cuda_check(inference::launch_softmax(device_input.get(), output, rows, columns), "launch_softmax");
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    std::vector<float> actual(input.size());
    cuda_check(cudaMemcpy(actual.data(), output, actual.size() * sizeof(float), cudaMemcpyDeviceToHost), "output D2H");
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (!std::isfinite(actual[index]) || std::fabs(actual[index] - expected[index]) > 1.0e-5F) {
            throw std::runtime_error("softmax output did not match FP32 reference");
        }
    }
}

void test_invalid_arguments() {
    DeviceBuffer<float> data(4);
    const auto expect_invalid = [](cudaError_t status) {
        if (status != cudaErrorInvalidValue) throw std::runtime_error("softmax launcher accepted invalid arguments");
    };
    expect_invalid(inference::launch_softmax(nullptr, data.get(), 1, 4));
    expect_invalid(inference::launch_softmax(data.get(), nullptr, 1, 4));
    expect_invalid(inference::launch_softmax(data.get(), data.get(), 0, 4));
    expect_invalid(inference::launch_softmax(data.get(), data.get(), 1, 0));
}

}  // namespace

int main() {
    try {
        test_softmax(3, 1, false);
        test_softmax(4, 37, false);
        test_softmax(2, 3072, true);
        test_invalid_arguments();
        std::cout << "softmax_test: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "softmax_test: FAIL: " << error.what() << '\n';
        return 1;
    }
}
