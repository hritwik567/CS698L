// Compile: nvcc -arch=sm_61 -std=c++11 assignment5-p3.cu -o assignment5-p3

#include <cmath>
#include <cstdint>
#include <iostream>
#include <sys/time.h>

#define SIZE 1024
#define BLOCK_SIZE 16
#define THRESHOLD (0.000001)

using namespace std;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

double rtclock() {
  struct timezone Tzp;
  struct timeval Tp;
  int stat;
  stat = gettimeofday(&Tp, &Tzp);
  if (stat != 0) {
    cout << "Error return from gettimeofday: " << stat << "\n";
  }
  return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

__host__ void ATAonCPU(double* M, double* P) {
  for (int k = 0; k < SIZE; k++) {
    for (int i = 0; i < SIZE; i++) {
      for (int j = 0; j < SIZE; j++)
        P[i*SIZE + j] += M[k*SIZE + i] * M[k*SIZE + j];
    }
  }
}

__host__ void check_result(double* Test, double* Ref) {
  double maxdiff = 0, rel_diff = 0;
  int numdiffs = 0;

  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      rel_diff = (Test[i*SIZE + j] - Ref[i*SIZE + j]);
      if (fabs(rel_diff) > THRESHOLD) {
        printf("%f %f %f\n",Test[i*SIZE + j], Ref[i*SIZE + j], rel_diff);
        numdiffs++;
        if (rel_diff > maxdiff)
          maxdiff = rel_diff;
      }
    }
  }
  if (numdiffs > 0)
    cout << numdiffs << " Diffs found over THRESHOLD " << THRESHOLD << " Max Diff = " << maxdiff
         << "\n";
  else
    cout << "No differences found between base and test versions\n";
}

// SB: Implement your kernel here
__global__ void ATAkernel(double* M, double* P) {

  if(blockIdx.x < blockIdx.y) return;
  double sum = 0;

  uint64_t i = blockIdx.y*blockDim.y + threadIdx.y;
  uint64_t j = blockIdx.x*blockDim.x + threadIdx.x;

  __shared__ double A_t[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ double B_t[BLOCK_SIZE][BLOCK_SIZE];

  for (uint64_t tid = 0; tid < SIZE/blockDim.x; tid++) {
    A_t[threadIdx.y][threadIdx.x] = M[(tid * blockDim.x + threadIdx.x) * SIZE + i];
    B_t[threadIdx.y][threadIdx.x] = M[(tid * blockDim.y + threadIdx.y) * SIZE + j];
   
    __syncthreads();

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
  
  P[i * SIZE + j] = sum;
  if(blockIdx.x > blockIdx.y) P[j * SIZE + i] = sum;
}

int main() {
  cout << "Matrix Size = " << SIZE << "\n";

  double* A = new double[SIZE*SIZE];

  double* O_s = new double[SIZE*SIZE];

  double* O_p = new double[SIZE*SIZE];

  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      // A[i*SIZE + j] = random() * 0.25;
      A[i*SIZE + j] = i * (j-i) * 0.25;
      O_s[i*SIZE + j] = 0;
      O_p[i*SIZE + j] = 0;
    }
  }

  double clkbegin, clkend;
  double t;

  clkbegin = rtclock();
  ATAonCPU(A, O_s);
  clkend = rtclock();
  t = clkend - clkbegin;
  cout << "A^T.A on CPU: " << (2.0 * SIZE * SIZE * SIZE / t / 1.0e9)
       << " GFLOPS; Time = " << t * 1000 << " msec\n";

  cudaEvent_t start, end;

  gpuErrchk( cudaEventCreate(&start) );
  gpuErrchk( cudaEventCreate(&end) );
  
  // SB: Write your GPU kernel here
  double *O_p_c, *A_c;
  gpuErrchk( cudaMalloc((void**)&O_p_c, SIZE*SIZE*sizeof(double)) );
  gpuErrchk( cudaMalloc((void**)&A_c, SIZE*SIZE*sizeof(double)) );
  dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridSize((SIZE + blockSize.x - 1)/blockSize.x, (SIZE + blockSize.y - 1)/blockSize.y);
  
  gpuErrchk( cudaEventRecord(start, 0) );
  gpuErrchk( cudaMemcpy(O_p_c, O_p, SIZE*SIZE*sizeof(double), cudaMemcpyHostToDevice) );
  gpuErrchk( cudaMemcpy(A_c, A, SIZE*SIZE*sizeof(double), cudaMemcpyHostToDevice) );
  ATAkernel<<<gridSize, blockSize>>>(A_c, O_p_c);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaMemcpy(O_p, O_p_c, SIZE*SIZE*sizeof(double), cudaMemcpyDeviceToHost) );
  gpuErrchk( cudaEventRecord(end, 0) );

  
  gpuErrchk( cudaDeviceSynchronize() );

  float kernel_time = 0;
  gpuErrchk( cudaEventElapsedTime(&kernel_time, start, end) );

  cout << "A^T.A on GPU: " << (2.0 * SIZE * SIZE * SIZE / t / 1.0e9)
       << " GFLOPS; Time = " << kernel_time << " msec\n";

  check_result(O_p, O_s);

  gpuErrchk( cudaFree(O_p_c) );
  gpuErrchk( cudaFree(A_c) );
  
  free(O_s);
  free(O_p);
  free(A);

  return EXIT_SUCCESS;
}
