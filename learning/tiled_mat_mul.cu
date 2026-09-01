// #include <cuda_runtime.h>
// #include <iostream>

// #define TILE 16

// __global__ void tiled_matrix_multiplication(const float* A, const float* B, float* C, int M, int K, int N){
//     __shared__ float As[TILE][TILE];
//     __shared__ float Bs[TILE][TILE];

//     int row = blockIdx.y * blockDim.y + threadIdx.y;
//     int col = blockIdx.x * blockDim.x + threadIdx.x;

//     // load from block idx . y , block idx. x
//     float res = 0.0f;

//     for(int tile = 0; tile < (K + TILE - 1) / TILE; ++tile){

//         // load tile A 
//         int aColumn = tile * TILE + threadIdx.x;
//         if(row < M and aColumn < K){
//             As[threadIdx.y][threadIdx.x] = A[row * K + aColumn];
//         } else{
//             As[threadIdx.y][threadIdx.x] = 0.0f;
//         }
        
        
//         // load tile B
//         int bRow = tile * TILE + threadIdx.y;
//         if(bRow < K and col < N){
//             Bs[threadIdx.y][threadIdx.x] = B[bRow * N + col];
//         } else{
//             Bs[threadIdx.y][threadIdx.x] = 0.0f;
//         }
        
//         // sync threads
//         __syncthreads();

//         // sum
//         for(int k = 0; k < TILE; ++k){
//             // row fixed, col fixed
//             res += As[threadIdx.y][k] * Bs[k][threadIdx.x];
//         }

//         __syncthreads();

//         // sync threads ot move to next
//     }

    
//     if(row < M and col < N){
//         C[row * N + col] = res; 
//     }
// }

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#define TILE 16

// ============================================================
// CUDA ERROR CHECKING
// ============================================================

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__     \
                      << " -> " << cudaGetErrorString(err) << "\n";           \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)


// ============================================================
// NAIVE MATRIX MULTIPLICATION
//
// A: M x K
// B: K x N
// C: M x N
//
// One thread computes one C[row, col].
// ============================================================

__global__ void naive_matrix_multiplication(
    const float* A,
    const float* B,
    float* C,
    int M,
    int K,
    int N
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {

        float res = 0.0f;

        for (int k = 0; k < K; ++k) {
            res += A[row * K + k] * B[k * N + col];
        }

        C[row * N + col] = res;
    }
}


// ============================================================
// TILED MATRIX MULTIPLICATION
//
// One block computes one TILE x TILE region of C.
//
// Every iteration:
//   1. collaboratively load TILE x TILE A tile
//   2. collaboratively load TILE x TILE B tile
//   3. synchronize
//   4. perform TILE FMAs per thread
//   5. synchronize
// ============================================================

__global__ void tiled_matrix_multiplication(
    const float* A,
    const float* B,
    float* C,
    int M,
    int K,
    int N
) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float res = 0.0f;

    // Move through the K dimension TILE elements at a time.
    for (int tile = 0; tile < (K + TILE - 1) / TILE; ++tile) {

        // ----------------------------------------------------
        // Load A tile
        //
        // A coordinate:
        //     row = output row
        //     col = tile * TILE + threadIdx.x
        // ----------------------------------------------------

        int aColumn = tile * TILE + threadIdx.x;

        if (row < M && aColumn < K) {
            As[threadIdx.y][threadIdx.x]
                = A[row * K + aColumn];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }


        // ----------------------------------------------------
        // Load B tile
        //
        // B coordinate:
        //     row = tile * TILE + threadIdx.y
        //     col = output col
        // ----------------------------------------------------

        int bRow = tile * TILE + threadIdx.y;

        if (bRow < K && col < N) {
            Bs[threadIdx.y][threadIdx.x]
                = B[bRow * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }


        // Make sure BOTH tiles have been fully loaded.
        __syncthreads();


        // ----------------------------------------------------
        // Compute this tile's contribution
        // ----------------------------------------------------

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            res += As[threadIdx.y][k]
                 * Bs[k][threadIdx.x];
        }


        // Make sure nobody is still reading this tile before
        // we overwrite shared memory with the next tile.
        __syncthreads();
    }


    if (row < M && col < N) {
        C[row * N + col] = res;
    }
}


// ============================================================
// CPU REFERENCE IMPLEMENTATION
//
// Only used on smaller matrices because CPU O(MKN) becomes slow.
// ============================================================

void cpu_matrix_multiplication(
    const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<float>& C,
    int M,
    int K,
    int N
) {
    for (int row = 0; row < M; ++row) {

        for (int col = 0; col < N; ++col) {

            float res = 0.0f;

            for (int k = 0; k < K; ++k) {
                res += A[row * K + k]
                     * B[k * N + col];
            }

            C[row * N + col] = res;
        }
    }
}


// ============================================================
// INITIALIZE MATRIX WITH RANDOM VALUES
// ============================================================

void fill_random(std::vector<float>& matrix) {

    static std::mt19937 generator(12345);

    // Keep values relatively small.
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

    for (float& x : matrix) {
        x = distribution(generator);
    }
}


// ============================================================
// MAX ABSOLUTE ERROR
// ============================================================

float max_absolute_error(
    const std::vector<float>& A,
    const std::vector<float>& B
) {
    float maxError = 0.0f;

    for (size_t i = 0; i < A.size(); ++i) {
        maxError = std::max(
            maxError,
            std::abs(A[i] - B[i])
        );
    }

    return maxError;
}


// ============================================================
// BENCHMARK RESULT
// ============================================================

struct BenchmarkResult {
    float ms;
    double gflops;
};


// ============================================================
// BENCHMARK NAIVE
// ============================================================

BenchmarkResult benchmark_naive(
    const float* d_A,
    const float* d_B,
    float* d_C,
    int M,
    int K,
    int N,
    int iterations
) {
    dim3 threads(TILE, TILE);

    dim3 blocks(
        (N + TILE - 1) / TILE,
        (M + TILE - 1) / TILE
    );


    // --------------------------------------------------------
    // Warmup
    //
    // Important because the first launch can include startup
    // overhead that we don't want to benchmark.
    // --------------------------------------------------------

    for (int i = 0; i < 5; ++i) {
        naive_matrix_multiplication<<<blocks, threads>>>(
            d_A,
            d_B,
            d_C,
            M,
            K,
            N
        );
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());


    // --------------------------------------------------------
    // CUDA events
    // --------------------------------------------------------

    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));


    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < iterations; ++i) {
        naive_matrix_multiplication<<<blocks, threads>>>(
            d_A,
            d_B,
            d_C,
            M,
            K,
            N
        );
    }

    CUDA_CHECK(cudaEventRecord(stop));

    // CPU waits until GPU reaches stop event.
    CUDA_CHECK(cudaEventSynchronize(stop));

    CUDA_CHECK(cudaGetLastError());


    float totalMilliseconds = 0.0f;

    CUDA_CHECK(
        cudaEventElapsedTime(
            &totalMilliseconds,
            start,
            stop
        )
    );


    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));


    float averageMilliseconds =
        totalMilliseconds / iterations;


    // Matrix multiplication does roughly:
    //
    // M * N * K multiplications
    // M * N * K additions
    //
    // ≈ 2 * M * N * K FLOPs

    double flops =
        2.0 *
        static_cast<double>(M) *
        static_cast<double>(N) *
        static_cast<double>(K);

    double seconds =
        averageMilliseconds / 1000.0;

    double gflops =
        flops / seconds / 1e9;


    return {
        averageMilliseconds,
        gflops
    };
}


// ============================================================
// BENCHMARK TILED
// ============================================================

BenchmarkResult benchmark_tiled(
    const float* d_A,
    const float* d_B,
    float* d_C,
    int M,
    int K,
    int N,
    int iterations
) {
    dim3 threads(TILE, TILE);

    dim3 blocks(
        (N + TILE - 1) / TILE,
        (M + TILE - 1) / TILE
    );


    // Warmup

    for (int i = 0; i < 5; ++i) {
        tiled_matrix_multiplication<<<blocks, threads>>>(
            d_A,
            d_B,
            d_C,
            M,
            K,
            N
        );
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());


    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));


    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < iterations; ++i) {
        tiled_matrix_multiplication<<<blocks, threads>>>(
            d_A,
            d_B,
            d_C,
            M,
            K,
            N
        );
    }

    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaEventSynchronize(stop));

    CUDA_CHECK(cudaGetLastError());


    float totalMilliseconds = 0.0f;

    CUDA_CHECK(
        cudaEventElapsedTime(
            &totalMilliseconds,
            start,
            stop
        )
    );


    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));


    float averageMilliseconds =
        totalMilliseconds / iterations;


    double flops =
        2.0 *
        static_cast<double>(M) *
        static_cast<double>(N) *
        static_cast<double>(K);

    double seconds =
        averageMilliseconds / 1000.0;

    double gflops =
        flops / seconds / 1e9;


    return {
        averageMilliseconds,
        gflops
    };
}


// ============================================================
// RUN ONE TEST CASE
// ============================================================

void run_test(
    int M,
    int K,
    int N,
    int iterations,
    bool runCPUReference
) {

    std::cout
        << "\n============================================================\n";

    std::cout
        << "Matrix multiplication\n"
        << "A: " << M << " x " << K << "\n"
        << "B: " << K << " x " << N << "\n"
        << "C: " << M << " x " << N << "\n";

    std::cout
        << "Iterations: " << iterations << "\n";

    std::cout
        << "============================================================\n";


    // --------------------------------------------------------
    // Host memory
    // --------------------------------------------------------

    std::vector<float> h_A(
        static_cast<size_t>(M) * K
    );

    std::vector<float> h_B(
        static_cast<size_t>(K) * N
    );

    std::vector<float> h_C_naive(
        static_cast<size_t>(M) * N
    );

    std::vector<float> h_C_tiled(
        static_cast<size_t>(M) * N
    );


    fill_random(h_A);
    fill_random(h_B);


    // --------------------------------------------------------
    // Device memory
    // --------------------------------------------------------

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C_naive = nullptr;
    float* d_C_tiled = nullptr;


    size_t bytesA =
        static_cast<size_t>(M) * K * sizeof(float);

    size_t bytesB =
        static_cast<size_t>(K) * N * sizeof(float);

    size_t bytesC =
        static_cast<size_t>(M) * N * sizeof(float);


    CUDA_CHECK(cudaMalloc(&d_A, bytesA));
    CUDA_CHECK(cudaMalloc(&d_B, bytesB));

    CUDA_CHECK(cudaMalloc(&d_C_naive, bytesC));
    CUDA_CHECK(cudaMalloc(&d_C_tiled, bytesC));


    // --------------------------------------------------------
    // Copy input matrices CPU -> GPU
    // --------------------------------------------------------

    CUDA_CHECK(
        cudaMemcpy(
            d_A,
            h_A.data(),
            bytesA,
            cudaMemcpyHostToDevice
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            d_B,
            h_B.data(),
            bytesB,
            cudaMemcpyHostToDevice
        )
    );


    // --------------------------------------------------------
    // Benchmark naive
    // --------------------------------------------------------

    BenchmarkResult naive =
        benchmark_naive(
            d_A,
            d_B,
            d_C_naive,
            M,
            K,
            N,
            iterations
        );


    // --------------------------------------------------------
    // Benchmark tiled
    // --------------------------------------------------------

    BenchmarkResult tiled =
        benchmark_tiled(
            d_A,
            d_B,
            d_C_tiled,
            M,
            K,
            N,
            iterations
        );


    // --------------------------------------------------------
    // Copy results back
    // --------------------------------------------------------

    CUDA_CHECK(
        cudaMemcpy(
            h_C_naive.data(),
            d_C_naive,
            bytesC,
            cudaMemcpyDeviceToHost
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            h_C_tiled.data(),
            d_C_tiled,
            bytesC,
            cudaMemcpyDeviceToHost
        )
    );


    // --------------------------------------------------------
    // Check naive vs tiled
    // --------------------------------------------------------

    float gpuDifference =
        max_absolute_error(
            h_C_naive,
            h_C_tiled
        );


    std::cout << std::fixed << std::setprecision(4);

    std::cout
        << "\nCorrectness:\n"
        << "  max |naive - tiled| = "
        << gpuDifference
        << "\n";


    if (gpuDifference < 1e-3f) {
        std::cout << "  Naive vs tiled: PASS\n";
    } else {
        std::cout << "  Naive vs tiled: FAIL\n";
    }


    // --------------------------------------------------------
    // Optional CPU verification
    // --------------------------------------------------------

    if (runCPUReference) {

        std::cout
            << "\nRunning CPU reference...\n";

        std::vector<float> h_C_cpu(
            static_cast<size_t>(M) * N
        );

        cpu_matrix_multiplication(
            h_A,
            h_B,
            h_C_cpu,
            M,
            K,
            N
        );


        float naiveError =
            max_absolute_error(
                h_C_cpu,
                h_C_naive
            );

        float tiledError =
            max_absolute_error(
                h_C_cpu,
                h_C_tiled
            );


        std::cout
            << "  max |CPU - naive| = "
            << naiveError
            << "\n";

        std::cout
            << "  max |CPU - tiled| = "
            << tiledError
            << "\n";
    }


    // --------------------------------------------------------
    // Performance
    // --------------------------------------------------------

    std::cout
        << "\nPerformance:\n";

    std::cout
        << "  Naive:\n"
        << "    "
        << naive.ms
        << " ms\n"
        << "    "
        << naive.gflops
        << " GFLOP/s\n";


    std::cout
        << "  Tiled:\n"
        << "    "
        << tiled.ms
        << " ms\n"
        << "    "
        << tiled.gflops
        << " GFLOP/s\n";


    double speedup =
        naive.ms / tiled.ms;


    std::cout
        << "\n  Tiled speedup: "
        << speedup
        << "x\n";


    // --------------------------------------------------------
    // Cleanup
    // --------------------------------------------------------

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C_naive));
    CUDA_CHECK(cudaFree(d_C_tiled));
}


// ============================================================
// MAIN
// ============================================================

int main() {

    // Print GPU information.

    int device = 0;

    CUDA_CHECK(cudaSetDevice(device));


    cudaDeviceProp prop;

    CUDA_CHECK(
        cudaGetDeviceProperties(
            &prop,
            device
        )
    );


    std::cout
        << "GPU: "
        << prop.name
        << "\n";

    std::cout
        << "Compute capability: "
        << prop.major
        << "."
        << prop.minor
        << "\n";

    std::cout
        << "SM count: "
        << prop.multiProcessorCount
        << "\n";

    std::cout
        << "TILE size: "
        << TILE
        << " x "
        << TILE
        << "\n";


    // ========================================================
    // TEST 1
    //
    // Tiny correctness test.
    // ========================================================

    run_test(
        16,
        16,
        16,
        100,
        true
    );


    // ========================================================
    // TEST 2
    //
    // Intentionally NOT divisible by TILE.
    //
    // This tests your boundary handling.
    // ========================================================

    run_test(
        123,
        77,
        91,
        100,
        true
    );


    // ========================================================
    // TEST 3
    //
    // Moderate square matrix.
    // ========================================================

    run_test(
        512,
        512,
        512,
        100,
        true
    );


    // ========================================================
    // TEST 4
    //
    // Bigger benchmark.
    //
    // Skip CPU reference because CPU matmul starts becoming
    // annoying here.
    // ========================================================

    run_test(
        1024,
        1024,
        1024,
        100,
        false
    );


    // ========================================================
    // TEST 5
    //
    // Larger matmul.
    // ========================================================

    run_test(
        2048,
        2048,
        2048,
        50,
        false
    );


    // ========================================================
    // RECTANGULAR TEST
    //
    // A = 1024 x 2048
    // B = 2048 x 512
    // C = 1024 x 512
    // ========================================================

    run_test(
        1024,
        2048,
        512,
        100,
        false
    );


    CUDA_CHECK(cudaDeviceSynchronize());

    std::cout
        << "\nAll tests complete.\n";

    return 0;
}