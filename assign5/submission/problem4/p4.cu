// Compile: nvcc -arch=sm_61 -std=c++11 assignment5-p4.cu -o assignment5-p4

#include <cmath>
#include <cuda.h>
#include <iostream>

const uint64_t N = (1 << 10);
const uint64_t BLOCK_SIZE = (1 << 4);

using namespace std;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void kernel1(uint64_t* A, uint64_t* B, uint64_t* C) {
  // SB: Write your code here
  int i =  blockIdx.x*blockDim.x + threadIdx.x;
  int j =  blockIdx.y*blockDim.y + threadIdx.y;
  uint64_t sum = 0;
  for (uint64_t k = 0; k < N; k++) {
    sum += A[i * N + k] * B[k * N + j];
  }
  C[i * N + j] = sum;
}

__global__ void kernel2(uint64_t* A, uint64_t* B, uint64_t* C) {
  // SB: Write your code here
  int i =  blockIdx.x*blockDim.x + threadIdx.x;
  int j =  blockIdx.y*blockDim.y + threadIdx.y;
  uint64_t sum = 0;
  for (uint64_t k = 0; k < N; k++) {
    sum += A[i * N + k] * B[k * N + j];
  }
  C[i * N + j] = sum;
}

__host__ void cpumatMul(uint64_t* A, uint64_t* B, uint64_t* C) {
  for (uint64_t i = 0; i < N; i++) {
    for (uint64_t j = 0; j < N; j++) {
      uint64_t sum = 0;
      for (uint64_t k = 0; k < N; k++) {
        sum += A[i * N + k] * B[k * N + j];
      }
      C[i * N + j] = sum;
    }
  }
}

__host__ void check_result(uint64_t* w_ref, uint64_t* w_opt) {
  for (uint64_t i = 0; i < N; i++) {
    for (uint64_t j = 0; j < N; j++) {
      if (w_ref[i * N + j] != w_opt[i * N + j]) {
        cout << "Difference found\n";
        exit(EXIT_FAILURE);
      }
    }
  }
  cout << "No differences found between base and test versions\n";
}

int main() {
  int SIZE = N * N;
  cudaEvent_t start, end;

  gpuErrchk( cudaEventCreate(&start) );
  gpuErrchk( cudaEventCreate(&end) );

  uint64_t *h_A, *h_B, *h_C1, *h_C2, *cpuResult;

  h_A = (uint64_t*)malloc(SIZE * sizeof(uint64_t));
  h_B = (uint64_t*)malloc(SIZE * sizeof(uint64_t));
  h_C1 = (uint64_t*)malloc(SIZE * sizeof(uint64_t));
  h_C2 = (uint64_t*)malloc(SIZE * sizeof(uint64_t));
  cpuResult = (uint64_t*)malloc(SIZE * sizeof(uint64_t));

  for (uint64_t i = 0; i < N; i++) {
    for (uint64_t j = 0; j < N; j++) {
      h_A[i * N + j] = 1;
      h_B[i * N + j] = 2;
      h_C1[i * N + j] = 0;
      h_C2[i * N + j] = 0;
      cpuResult[i * N + j] = 0;
    }
  }

  cpumatMul(h_A, h_B, cpuResult);

  uint64_t *d_A, *d_B, *d_C1, *d_C2;
  gpuErrchk( cudaMalloc((void**)&d_A, SIZE * sizeof(uint64_t)) );
  gpuErrchk( cudaMalloc((void**)&d_B, SIZE * sizeof(uint64_t)) );
  gpuErrchk( cudaMalloc((void**)&d_C1, SIZE * sizeof(uint64_t)) );
  gpuErrchk( cudaMalloc((void**)&d_C2, SIZE * sizeof(uint64_t)) );

  gpuErrchk( cudaMemcpy(d_A, h_A, SIZE * sizeof(uint64_t), cudaMemcpyHostToDevice) );
  gpuErrchk( cudaMemcpy(d_B, h_B, SIZE * sizeof(uint64_t), cudaMemcpyHostToDevice) );

  dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 blocksPerGrid((N + threadsPerBlock.x - 1)/threadsPerBlock.x, (N + threadsPerBlock.y - 1)/threadsPerBlock.y);

  gpuErrchk( cudaEventRecord(start, 0) );
  kernel1<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C1);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaEventRecord(end, 0) );
  
  gpuErrchk( cudaDeviceSynchronize() );
  float kernel_time = 0;
  gpuErrchk( cudaEventElapsedTime(&kernel_time, start, end) );
  std::cout << "Kernel 1 time (ms): " << kernel_time << "\n";

  gpuErrchk( cudaMemcpy(h_C1, d_C1, SIZE * sizeof(uint64_t), cudaMemcpyDeviceToHost) );

  gpuErrchk( cudaEventRecord(start, 0) );
  kernel2<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C2);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaEventRecord(end, 0) );

  gpuErrchk( cudaDeviceSynchronize() );
  gpuErrchk( cudaEventElapsedTime(&kernel_time, start, end) );
  std::cout << "Kernel 2 time (ms): " << kernel_time << "\n";

  gpuErrchk( cudaMemcpy(h_C2, d_C2, SIZE * sizeof(uint64_t), cudaMemcpyDeviceToHost) );

  gpuErrchk( cudaFree(d_A) );
  gpuErrchk( cudaFree(d_B) );
  gpuErrchk( cudaFree(d_C1) );
  gpuErrchk( cudaFree(d_C2) );

  free(h_A);
  free(h_B);

  check_result(h_C1, cpuResult);
  check_result(h_C2, cpuResult);

  free(cpuResult);
  free(h_C1);
  free(h_C2);

  return EXIT_SUCCESS;
}
