// Compile: nvcc -arch=sm_61 -std=c++11 assignment5-p3.cu -o assignment5-p3

#include <cmath>
#include <cstdint>
#include <iostream>
#include <sys/time.h>

#define SIZE 128
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

  int i =  blockIdx.y*blockDim.y + threadIdx.y;
  int j =  blockIdx.x*blockDim.x + threadIdx.x;
  
  if(i < SIZE and j < SIZE) {
    for (int k = 0; k < SIZE; k++)
      P[i*SIZE + j] += M[k*SIZE + i] * M[k*SIZE + j];
  }
}

int main() {
  cout << "Matrix Size = " << SIZE << "\n";

  double* A = new double[SIZE*SIZE];

  double* O_s = new double[SIZE*SIZE];

  double* O_p = new double[SIZE*SIZE];

  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      A[i*SIZE + j] = i * j * 0.25;
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
  gpuErrchk( cudaMemcpy(O_p_c, O_p, SIZE*SIZE*sizeof(double), cudaMemcpyHostToDevice) );
  gpuErrchk( cudaMemcpy(A_c, A, SIZE*SIZE*sizeof(double), cudaMemcpyHostToDevice) );
  dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridSize((SIZE + blockSize.x - 1)/blockSize.x, (SIZE + blockSize.y - 1)/blockSize.y);
  
  gpuErrchk( cudaEventRecord(start, 0) );
  ATAkernel<<<gridSize, blockSize>>>(A_c, O_p_c);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaEventRecord(end, 0) );

  gpuErrchk( cudaMemcpy(O_p, O_p_c, SIZE*SIZE*sizeof(double), cudaMemcpyDeviceToHost) );
  
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
