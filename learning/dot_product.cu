#include <cuda_runtime.h>
#include <iostream>


__global__ void add(const float* A, const float* B, float* C, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N){
        C[idx] = A[idx] + B[idx];
    }
}


int main(){
    int N = 100000;
    size_t N_bytes = N * sizeof(float);

    // cpu memory
    float* h_a = new float[N];
    float* h_b = new float[N];
    float* h_c = new float[N];

    for(int i = 0; i < N; ++i){
        h_a[i] = i;
        h_b[i] = 2 * i;
    }


    // device memory
    float* d_a;
    float* d_b;
    float* d_c;
    cudaMalloc(&d_a, N_bytes);
    cudaMalloc(&d_b, N_bytes);
    cudaMalloc(&d_c, N_bytes);

    // cpu -> gpu
    // dest, src, bytes
    cudaMemcpy(d_a, h_a, N_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N_bytes, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1)/threadsPerBlock;

    add<<<blocks, threadsPerBlock>>>(d_a, d_b, d_c, N);
    std::cout << "hello\n";
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch error: "
                << cudaGetErrorString(err) << "\n";
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Kernel execution error: "
                << cudaGetErrorString(err) << "\n";
    }

    cudaMemcpy(h_c, d_c, N_bytes, cudaMemcpyDeviceToHost);

    std::cout << h_c[10] << "\n";  // 10 + 20 = 30

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
}
