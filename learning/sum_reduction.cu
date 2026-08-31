// #include <cuda_runtime.h>
// #include <iostream>

// __global__ void sum_reduction(const float* A, float* res, int N){
//     // shared memory to define for the sum reduction thing
//     // we define for each block we calculate and then do the atomic add at the end
//     // so for a given block we assign each a location in shared memory so assume we have 256 threads per block
    
//     extern __shared__ float shared[];

//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     int tIdx = threadIdx.x;

//     if(idx < N){
//         shared[tIdx] = A[idx];
//     } else{
//         shared[tIdx] = 0.0;        
//     }

//     __syncthreads();

//     // at this point shared memory is fully populated
//     // cut it by half each time and then sum
//     for(int stride = blockDim.x / 2; stride > 0; stride /= 2){
//         if(tIdx < stride){
//             shared[tIdx] += shared[tIdx + stride];
//         }
//         __syncthreads();
//     }

//     if(tIdx == 0){
//         atomicAdd(res, shared[tIdx]);
//     }
// }

// int main(){

// }

#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

// ------------------------------------------------------------
// CUDA error checking
// ------------------------------------------------------------

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)         \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";   \
            std::exit(1);                                                  \
        }                                                                  \
    } while (0)


// ------------------------------------------------------------
// 1. Naive atomic reduction
//
// Every thread directly atomically adds into the SAME location.
// Correct, but potentially lots of contention.
// ------------------------------------------------------------

__global__ void atomic_sum(
    const float* A,
    float* res,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        atomicAdd(res, A[idx]);
    }
}


// ------------------------------------------------------------
// 2. Shared-memory reduction
//
// Each block:
//     values -> shared memory
//            -> reduce
//            -> one partial sum
//            -> one atomicAdd
// ------------------------------------------------------------

__global__ void sum_reduction(
    const float* A,
    float* res,
    int N
) {
    extern __shared__ float shared[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tIdx = threadIdx.x;

    if (idx < N) {
        shared[tIdx] = A[idx];
    } else {
        shared[tIdx] = 0.0f;
    }

    __syncthreads();

    for (int stride = blockDim.x / 2;
         stride > 0;
         stride /= 2) {

        if (tIdx < stride) {
            shared[tIdx] += shared[tIdx + stride];
        }

        __syncthreads();
    }

    if (tIdx == 0) {
        atomicAdd(res, shared[0]);
    }
}


// ------------------------------------------------------------
// CPU reference
// ------------------------------------------------------------

float cpu_sum(const std::vector<float>& A)
{
    float res = 0.0f;

    for (float x : A) {
        res += x;
    }

    return res;
}


// ------------------------------------------------------------
// Benchmark naive atomicAdd
// ------------------------------------------------------------

float benchmark_atomic(
    const float* d_A,
    float* d_res,
    int N,
    int threads,
    int iterations
) {
    int blocks = (N + threads - 1) / threads;

    // Warmup
    for (int i = 0; i < 10; ++i) {
        CUDA_CHECK(cudaMemset(d_res, 0, sizeof(float)));

        atomic_sum<<<blocks, threads>>>(
            d_A,
            d_res,
            N
        );
    }

    CUDA_CHECK(cudaDeviceSynchronize());


    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float total_ms = 0.0f;

    for (int i = 0; i < iterations; ++i) {

        // Reset OUTSIDE the measured region
        CUDA_CHECK(cudaMemset(d_res, 0, sizeof(float)));

        CUDA_CHECK(cudaEventRecord(start));

        atomic_sum<<<blocks, threads>>>(
            d_A,
            d_res,
            N
        );

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        CUDA_CHECK(cudaGetLastError());

        float ms;

        CUDA_CHECK(
            cudaEventElapsedTime(
                &ms,
                start,
                stop
            )
        );

        total_ms += ms;
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return total_ms / iterations;
}


// ------------------------------------------------------------
// Benchmark shared-memory reduction
// ------------------------------------------------------------

float benchmark_shared(
    const float* d_A,
    float* d_res,
    int N,
    int threads,
    int iterations
) {
    int blocks = (N + threads - 1) / threads;

    size_t shared_bytes =
        threads * sizeof(float);

    // Warmup
    for (int i = 0; i < 10; ++i) {
        CUDA_CHECK(cudaMemset(d_res, 0, sizeof(float)));

        sum_reduction<<<
            blocks,
            threads,
            shared_bytes
        >>>(
            d_A,
            d_res,
            N
        );
    }

    CUDA_CHECK(cudaDeviceSynchronize());


    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float total_ms = 0.0f;

    for (int i = 0; i < iterations; ++i) {

        CUDA_CHECK(cudaMemset(d_res, 0, sizeof(float)));

        CUDA_CHECK(cudaEventRecord(start));

        sum_reduction<<<
            blocks,
            threads,
            shared_bytes
        >>>(
            d_A,
            d_res,
            N
        );

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        CUDA_CHECK(cudaGetLastError());

        float ms;

        CUDA_CHECK(
            cudaEventElapsedTime(
                &ms,
                start,
                stop
            )
        );

        total_ms += ms;
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return total_ms / iterations;
}


// ------------------------------------------------------------
// Test + benchmark one N
// ------------------------------------------------------------

void run_test(int N)
{
    constexpr int THREADS = 256;
    constexpr int ITERATIONS = 100;

    // Simple input so expected sum is obvious.
    std::vector<float> A(N, 1.0f);

    float expected = cpu_sum(A);


    // --------------------------------------------------------
    // Allocate GPU memory
    // --------------------------------------------------------

    float* d_A;
    float* d_res;

    CUDA_CHECK(
        cudaMalloc(
            &d_A,
            N * sizeof(float)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_res,
            sizeof(float)
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            d_A,
            A.data(),
            N * sizeof(float),
            cudaMemcpyHostToDevice
        )
    );


    int blocks =
        (N + THREADS - 1) / THREADS;


    // --------------------------------------------------------
    // Correctness: naive atomic
    // --------------------------------------------------------

    CUDA_CHECK(
        cudaMemset(
            d_res,
            0,
            sizeof(float)
        )
    );

    atomic_sum<<<blocks, THREADS>>>(
        d_A,
        d_res,
        N
    );

    CUDA_CHECK(cudaDeviceSynchronize());

    float atomic_result;

    CUDA_CHECK(
        cudaMemcpy(
            &atomic_result,
            d_res,
            sizeof(float),
            cudaMemcpyDeviceToHost
        )
    );


    // --------------------------------------------------------
    // Correctness: shared reduction
    // --------------------------------------------------------

    CUDA_CHECK(
        cudaMemset(
            d_res,
            0,
            sizeof(float)
        )
    );

    sum_reduction<<<
        blocks,
        THREADS,
        THREADS * sizeof(float)
    >>>(
        d_A,
        d_res,
        N
    );

    CUDA_CHECK(cudaDeviceSynchronize());

    float shared_result;

    CUDA_CHECK(
        cudaMemcpy(
            &shared_result,
            d_res,
            sizeof(float),
            cudaMemcpyDeviceToHost
        )
    );


    // --------------------------------------------------------
    // Benchmark
    // --------------------------------------------------------

    float atomic_ms =
        benchmark_atomic(
            d_A,
            d_res,
            N,
            THREADS,
            ITERATIONS
        );

    float shared_ms =
        benchmark_shared(
            d_A,
            d_res,
            N,
            THREADS,
            ITERATIONS
        );


    // --------------------------------------------------------
    // Results
    // --------------------------------------------------------

    std::cout
        << "\n====================================\n";

    std::cout
        << "N: " << N << "\n";

    std::cout
        << "Blocks: " << blocks << "\n";

    std::cout
        << "Threads/block: " << THREADS << "\n\n";


    std::cout
        << "CPU result:          "
        << expected << "\n";

    std::cout
        << "Atomic result:       "
        << atomic_result << "\n";

    std::cout
        << "Shared result:       "
        << shared_result << "\n\n";


    bool atomic_pass =
        std::fabs(atomic_result - expected) < 1e-3f;

    bool shared_pass =
        std::fabs(shared_result - expected) < 1e-3f;


    std::cout
        << "Atomic correctness:  "
        << (atomic_pass ? "PASS" : "FAIL")
        << "\n";

    std::cout
        << "Shared correctness:  "
        << (shared_pass ? "PASS" : "FAIL")
        << "\n\n";


    std::cout
        << "Atomic time:         "
        << atomic_ms
        << " ms\n";

    std::cout
        << "Shared time:         "
        << shared_ms
        << " ms\n";


    if (shared_ms > 0.0f) {
        std::cout
            << "Speedup:             "
            << atomic_ms / shared_ms
            << "x\n";
    }


    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_res));
}


// ------------------------------------------------------------
// Main
// ------------------------------------------------------------

int main()
{
    // Tiny / correctness-oriented
    run_test(4);

    // Exactly one block
    run_test(256);

    // Partial final block
    run_test(1000);

    // Start seeing contention
    run_test(100000);

    // More meaningful performance comparisons
    run_test(1000000);
    run_test(10000000);

    return 0;
}