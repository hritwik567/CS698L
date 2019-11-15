// Compile: nvcc -arch=sm_61 -std=c++11 assignment5-p4.cu -o assignment5-p4

#include <cmath>
#include <cuda.h>
#include <iostream>

const uint64_t N = (1 << 10);

using namespace std;

__global__ void kernel1(uint64_t* A, uint64_t* B, uint64_t* C) {
  // SB: Write your code here
}

__global__ void kernel2(uint64_t* A, uint64_t* B, uint64_t* C) {
  // SB: Write your code here
}

__host__ void cpumatMul(uint64_t* A, uint64_t* B, uint64_t* C) {
  for (uint64_t i = 0; i < N; i++) {
    for (uint64_t j = 0; j < N; j++) {
      float sum = 0.0;
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
  cudaError_t status;
  cudaEvent_t start, end;

  cudaEventCreate(&start);
  cudaEventCreate(&end);

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
  status = cudaMalloc((void**)&d_A, SIZE * sizeof(uint64_t));
  if (status != cudaSuccess) {
    cerr << cudaGetErrorString(status) << endl;
  }
  status = cudaMalloc((void**)&d_B, SIZE * sizeof(uint64_t));
  if (status != cudaSuccess) {
    cerr << cudaGetErrorString(status) << endl;
  }
  status = cudaMalloc((void**)&d_C1, SIZE * sizeof(uint64_t));
  if (status != cudaSuccess) {
    cerr << cudaGetErrorString(status) << endl;
  }
  status = cudaMalloc((void**)&d_C2, SIZE * sizeof(uint64_t));
  if (status != cudaSuccess) {
    cerr << cudaGetErrorString(status) << endl;
  }

  status = cudaMemcpy(d_A, h_A, SIZE * sizeof(uint64_t), cudaMemcpyHostToDevice);
  if (status != cudaSuccess) {
    cerr << cudaGetErrorString(status) << endl;
  }
  status = cudaMemcpy(d_B, h_B, SIZE * sizeof(uint64_t), cudaMemcpyHostToDevice);
  if (status != cudaSuccess) {
    cerr << cudaGetErrorString(status) << endl;
  }

  dim3 blocksPerGrid(1);
  dim3 threadsPerBlock(1);
  cudaEventRecord(start, 0);
  kernel1<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C1);
  cudaEventRecord(end, 0);
  float kernel_time;
  cudaEventElapsedTime(&kernel_time, start, end);
  std::cout << "Kernel 1 time (ms): " << kernel_time << "\n";
  cudaMemcpy(h_C1, d_C1, SIZE * sizeof(uint64_t), cudaMemcpyDeviceToHost);

  cudaEventRecord(start, 0);
  kernel2<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C2);
  cudaEventRecord(end, 0);
  cudaEventElapsedTime(&kernel_time, start, end);
  std::cout << "Kernel 2 time (ms): " << kernel_time << "\n";
  cudaMemcpy(h_C2, d_C2, SIZE * sizeof(uint64_t), cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C1);
  cudaFree(d_C2);

  free(h_A);
  free(h_B);

  check_result(h_C1, cpuResult);
  check_result(h_C2, cpuResult);

  free(cpuResult);
  free(h_C1);
  free(h_C2);

  return EXIT_SUCCESS;
}
