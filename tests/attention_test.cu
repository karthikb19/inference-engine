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

std::size_t index_3d(std::int32_t token, std::int32_t head, std::int32_t dim,
                     std::int32_t heads, std::int32_t head_dim) {
    return (static_cast<std::size_t>(token) * heads + head) * head_dim + dim;
}

// Independent FP32 reference. In particular, it derives kv_head from the GQA
// group instead of assuming Q and KV have the same number of heads.
std::vector<float> reference_attention(const std::vector<__nv_bfloat16>& query,
                                       const std::vector<__nv_bfloat16>& keys,
                                       const std::vector<__nv_bfloat16>& values,
                                       std::int32_t query_tokens,
                                       std::int32_t key_value_tokens,
                                       std::int32_t query_start,
                                       std::int32_t query_heads,
                                       std::int32_t key_value_heads,
                                       std::int32_t head_dim,
                                       float scale) {
    std::vector<float> result(query.size());
    const std::int32_t group_size = query_heads / key_value_heads;
    std::vector<float> scores(key_value_tokens);
    for (std::int32_t query_token = 0; query_token < query_tokens; ++query_token) {
        const std::int32_t last_key = query_start + query_token;
        for (std::int32_t query_head = 0; query_head < query_heads; ++query_head) {
            const std::int32_t kv_head = query_head / group_size;
            float maximum = -INFINITY;
            for (std::int32_t key_token = 0; key_token <= last_key; ++key_token) {
                float dot = 0.0F;
                for (std::int32_t dim = 0; dim < head_dim; ++dim) {
                    dot += __bfloat162float(query[index_3d(query_token, query_head, dim, query_heads, head_dim)]) *
                           __bfloat162float(keys[index_3d(key_token, kv_head, dim, key_value_heads, head_dim)]);
                }
                scores[key_token] = dot * scale;
                maximum = std::fmax(maximum, scores[key_token]);
            }
            float denominator = 0.0F;
            for (std::int32_t key_token = 0; key_token <= last_key; ++key_token) {
                scores[key_token] = std::exp(scores[key_token] - maximum);
                denominator += scores[key_token];
            }
            for (std::int32_t dim = 0; dim < head_dim; ++dim) {
                float weighted_sum = 0.0F;
                for (std::int32_t key_token = 0; key_token <= last_key; ++key_token) {
                    weighted_sum += scores[key_token] *
                        __bfloat162float(values[index_3d(key_token, kv_head, dim, key_value_heads, head_dim)]);
                }
                result[index_3d(query_token, query_head, dim, query_heads, head_dim)] =
                    weighted_sum / denominator;
            }
        }
    }
    return result;
}

void test_attention(std::int32_t query_tokens, std::int32_t key_value_tokens,
                    std::int32_t query_start, std::int32_t query_heads,
                    std::int32_t key_value_heads, std::int32_t head_dim) {
    const std::size_t query_count = static_cast<std::size_t>(query_tokens) * query_heads * head_dim;
    const std::size_t cache_count = static_cast<std::size_t>(key_value_tokens) * key_value_heads * head_dim;
    std::vector<__nv_bfloat16> query(query_count), keys(cache_count), values(cache_count);
    for (std::size_t i = 0; i < query.size(); ++i) {
        query[i] = __float2bfloat16(0.03125F * static_cast<float>(1 + (i * 17) % 23));
    }
    for (std::size_t i = 0; i < keys.size(); ++i) {
        keys[i] = __float2bfloat16(0.0625F * static_cast<float>(1 + (i * 11) % 19));
        // Positive and token/head-dependent values make both the causal mask
        // and an incorrect query-head-to-KV-head mapping observable.
        values[i] = __float2bfloat16(0.125F * static_cast<float>(1 + (i * 7) % 29));
    }
    const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
    const auto expected = reference_attention(query, keys, values, query_tokens, key_value_tokens,
                                              query_start, query_heads, key_value_heads, head_dim, scale);

    DeviceBuffer<__nv_bfloat16> device_query(query.size());
    DeviceBuffer<__nv_bfloat16> device_keys(keys.size());
    DeviceBuffer<__nv_bfloat16> device_values(values.size());
    DeviceBuffer<__nv_bfloat16> device_output(query.size());
    cuda_check(cudaMemcpy(device_query.get(), query.data(), query.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice), "query H2D");
    cuda_check(cudaMemcpy(device_keys.get(), keys.data(), keys.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice), "keys H2D");
    cuda_check(cudaMemcpy(device_values.get(), values.data(), values.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice), "values H2D");
    cuda_check(inference::launch_causal_attention(device_query.get(), device_keys.get(), device_values.get(),
                                                  device_output.get(), query_tokens, key_value_tokens,
                                                  query_start, query_heads, key_value_heads, head_dim, scale),
               "launch_causal_attention");
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    std::vector<__nv_bfloat16> output(query.size());
    cuda_check(cudaMemcpy(output.data(), device_output.get(), output.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost), "output D2H");
    for (std::size_t i = 0; i < output.size(); ++i) {
        const float actual = __bfloat162float(output[i]);
        if (!std::isfinite(actual) || std::fabs(actual - expected[i]) > 0.025F) {
            throw std::runtime_error("causal GQA attention output did not match the FP32 reference");
        }
    }
}

void test_invalid_arguments() {
    DeviceBuffer<__nv_bfloat16> values(128);
    const auto expect_invalid = [](cudaError_t status) {
        if (status != cudaErrorInvalidValue) throw std::runtime_error("attention launcher accepted invalid arguments");
    };
    expect_invalid(inference::launch_causal_attention(nullptr, values.get(), values.get(), values.get(), 1, 1, 0, 1, 1, 8, 1.0F));
    expect_invalid(inference::launch_causal_attention(values.get(), values.get(), values.get(), values.get(), 2, 2, 1, 2, 1, 8, 1.0F));
    expect_invalid(inference::launch_causal_attention(values.get(), values.get(), values.get(), values.get(), 1, 1, 0, 3, 2, 8, 1.0F));
    expect_invalid(inference::launch_causal_attention(values.get(), values.get(), values.get(), values.get(), 1, 1, 0, 1, 1, 8, 0.0F));
}

}  // namespace

int main() {
    try {
        // Ordinary causal prefill: A***, AB**, ABC*, ABCD.
        test_attention(4, 4, 0, 4, 2, 8);
        // Prefill with an offset makes the causal boundary different for each
        // query row and uses a nontrivial GQA group of two.
        test_attention(2, 4, 2, 4, 2, 8);
        // Exact Qwen3-0.6B head geometry; this is the decode launch shape.
        test_attention(1, 3, 2, inference::qwen3_0_6b_attention.query_heads,
                       inference::qwen3_0_6b_attention.key_value_heads,
                       inference::qwen3_0_6b_attention.head_dim);
        test_invalid_arguments();
        std::cout << "attention_test: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "attention_test: FAIL: " << error.what() << '\n';
        return 1;
    }
}
