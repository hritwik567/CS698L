// Compile: nvcc -arch=sm_61 -std=c++11 assignment5-p4.cu -o assignment5-p4

#include <cmath>
#include <cuda.h>
#include <iostream>

const uint64_t N = (1 << 10);
const uint64_t BLOCK_SIZE = (1 << 4);
const uint64_t TILE_SIZE = (1 << 4);

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
  uint64_t i =  blockIdx.y*blockDim.y + threadIdx.y;
  uint64_t j =  blockIdx.x*blockDim.x + threadIdx.x;
  uint64_t sum = 0;

  for (uint64_t k = 0; k < N; k++) {
    sum += A[i * N + k] * B[k * N + j];
  }

  C[i * N + j] = sum;
}

__global__ void kernel2(uint64_t* A, uint64_t* B, uint64_t* C) {
  // SB: Write your code here
  uint64_t sum = 0;

  uint64_t i = blockIdx.y*blockDim.y + threadIdx.y;
  uint64_t j = blockIdx.x*blockDim.x + threadIdx.x;

  __shared__ uint64_t A_t[TILE_SIZE][TILE_SIZE];
  __shared__ uint64_t B_t[TILE_SIZE][TILE_SIZE];

  for (uint64_t tid = 0; tid < N/blockDim.x; tid++) {
    A_t[threadIdx.y][threadIdx.x] = A[i * N + tid * blockDim.x + threadIdx.x];
    B_t[threadIdx.y][threadIdx.x] = B[(tid * blockDim.y + threadIdx.y) * N + j];
   
    __syncthreads();

    // for (uint64_t k = 0; k < blockDim.x; k++) {
    //   sum += A_t[threadIdx.y][k] * B_t[k][threadIdx.x];
    // }

    sum += A_t[threadIdx.y][0] * B_t[0][threadIdx.x]
          + A_t[threadIdx.y][1] * B_t[1][threadIdx.x]
          + A_t[threadIdx.y][2] * B_t[2][threadIdx.x]
          + A_t[threadIdx.y][3] * B_t[3][threadIdx.x]
          + A_t[threadIdx.y][4] * B_t[4][threadIdx.x]
          + A_t[threadIdx.y][5] * B_t[5][threadIdx.x]
          + A_t[threadIdx.y][6] * B_t[6][threadIdx.x]
          + A_t[threadIdx.y][7] * B_t[7][threadIdx.x]
          + A_t[threadIdx.y][8] * B_t[8][threadIdx.x]
          + A_t[threadIdx.y][9] * B_t[9][threadIdx.x]
          + A_t[threadIdx.y][10] * B_t[10][threadIdx.x]
          + A_t[threadIdx.y][11] * B_t[11][threadIdx.x]
          + A_t[threadIdx.y][12] * B_t[12][threadIdx.x]
          + A_t[threadIdx.y][13] * B_t[13][threadIdx.x]
          + A_t[threadIdx.y][14] * B_t[14][threadIdx.x]
          + A_t[threadIdx.y][15] * B_t[15][threadIdx.x];
    __syncthreads();

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
      h_A[i * N + j] = 3;
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

  dim3 threadsPerBlock1(BLOCK_SIZE, BLOCK_SIZE);
  dim3 blocksPerGrid1((N + threadsPerBlock1.x - 1)/threadsPerBlock1.x, (N + threadsPerBlock1.y - 1)/threadsPerBlock1.y);

  gpuErrchk( cudaEventRecord(start, 0) );
  gpuErrchk( cudaMemcpy(d_A, h_A, SIZE * sizeof(uint64_t), cudaMemcpyHostToDevice) );
  kernel1<<<blocksPerGrid1, threadsPerBlock1>>>(d_A, d_B, d_C1);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaMemcpy(h_C1, d_C1, SIZE * sizeof(uint64_t), cudaMemcpyDeviceToHost) );
  gpuErrchk( cudaEventRecord(end, 0) );
  
  gpuErrchk( cudaDeviceSynchronize() );
  float kernel_time = 0;
  gpuErrchk( cudaEventElapsedTime(&kernel_time, start, end) );
  std::cout << "Kernel 1 time (ms): " << kernel_time << "\n";


  dim3 threadsPerBlock2(TILE_SIZE, TILE_SIZE);
  dim3 blocksPerGrid2((N + threadsPerBlock2.x - 1)/threadsPerBlock2.x, (N + threadsPerBlock2.y - 1)/threadsPerBlock2.y);

  gpuErrchk( cudaEventRecord(start, 0) );
  gpuErrchk( cudaMemcpy(d_B, h_B, SIZE * sizeof(uint64_t), cudaMemcpyHostToDevice) );
  kernel2<<<blocksPerGrid2, threadsPerBlock2>>>(d_A, d_B, d_C2);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaMemcpy(h_C2, d_C2, SIZE * sizeof(uint64_t), cudaMemcpyDeviceToHost) );
  gpuErrchk( cudaEventRecord(end, 0) );

  gpuErrchk( cudaDeviceSynchronize() );
  gpuErrchk( cudaEventElapsedTime(&kernel_time, start, end) );
  std::cout << "Kernel 2 time (ms): " << kernel_time << "\n";


  gpuErrchk( cudaFree(d_A) );
  gpuErrchk( cudaFree(d_B) );
  gpuErrchk( cudaFree(d_C1) );
  gpuErrchk( cudaFree(d_C2) );

  free(h_A);
  free(h_B);

  // check_result(h_C1, cpuResult);
  // check_result(h_C2, cpuResult);
  // check_result(h_C1, h_C2);

  free(cpuResult);
  free(h_C1);
  free(h_C2);

  return EXIT_SUCCESS;
}
