// Compile: nvcc -arch=sm_61 -std=c++11 assignment5-p2.cu -o assignment5-p2

#include <cmath>
#include <cstdint>
#include <iostream>
#include <sys/time.h>

#define THRESHOLD (0.000001)

#define SIZE1 4096
#define SIZE2 4097
#define ITER 100

using namespace std;

__global__ void kernel1(double** A) {
  // SB: Write the first kernel here
  int j = threadIdx.x;
  for (int k = 0; k < ITER; k++) {
    for (int i = 1; i < SIZE1; i++) {
      A[i][j + 1] = A[i - 1][j + 1] + A[i][j + 1];
    }
  }
}

__global__ void kernel2(double** A) {
  // SB: Write the second kernel here
  int j = threadIdx.x;
  for (int k = 0; k < ITER; k++) {
    for (int i = 1; i < SIZE1; i++) {
      A[i][j + 1] = A[i - 1][j + 1] + A[i][j + 1];
    }
  }
}

__host__ void serial(double** A) {
  for (int k = 0; k < ITER; k++) {
    for (int i = 1; i < SIZE1; i++) {
      for (int j = 0; j < SIZE1 - 1; j++) {
        A[i][j + 1] = A[i - 1][j + 1] + A[i][j + 1];
      }
    }
  }
}

__host__ void check_result(double** w_ref, double** w_opt, uint64_t size) {
  double maxdiff = 0.0, this_diff = 0.0;
  int numdiffs = 0;

  for (uint64_t i = 0; i < size; i++) {
    for (uint64_t j = 0; j < size; j++) {
      this_diff = w_ref[i][j] - w_opt[i][j];
      if (fabs(this_diff) > THRESHOLD) {
        numdiffs++;
        if (this_diff > maxdiff)
          maxdiff = this_diff;
      }
    }
  }

  if (numdiffs > 0) {
    cout << numdiffs << " Diffs found over THRESHOLD " << THRESHOLD << "; Max Diff = " << maxdiff
         << endl;
  } else {
    cout << "No differences found between base and test versions\n";
  }
}

__host__ double rtclock() {
  struct timezone Tzp;
  struct timeval Tp;
  int stat;
  stat = gettimeofday(&Tp, &Tzp);
  if (stat != 0) {
    cout << "Error return from gettimeofday: " << stat << "\n";
  }
  return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

int main() {
  double** A_ser = new double*[SIZE1];
  double** A_k1 = new double*[SIZE1];
  for (int i = 0; i < SIZE1; i++) {
    A_ser[i] = new double[SIZE1];
    A_k1[i] = new double[SIZE1];
  }

  double** A_k2 = new double*[SIZE2];
  for (int i = 0; i < SIZE2; i++) {
    A_k2[i] = new double[SIZE2];
  }

  for (int i = 0; i < SIZE1; i++) {
    for (int j = 0; i < SIZE1; j++) {
      A_ser[i][j] = i + j;
      A_k1[i][j] = i + j;
    }
  }
  for (int i = 0; i < SIZE2; i++) {
    for (int j = 0; i < SIZE2; j++) {
      A_k2[i][j] = i + j;
    }
  }

  double clkbegin, clkend;
  double t;

  clkbegin = rtclock();
  serial(A_ser);
  clkend = rtclock();
  t = clkend - clkbegin;
  cout << "Serial code on CPU: " << (1.0 * SIZE1 * SIZE1 * ITER / t / 1.0e9)
       << " GFLOPS; Time = " << t * 1000 << " msec" << endl;

  // cudaError_t status;
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start, 0);
  // SB: Write your first GPU kernel here
  kernel1<<<1, SIZE1 - 1>>>(A_k1);
  cudaEventRecord(end, 0);
  float kernel_time;
  cudaEventElapsedTime(&kernel_time, start, end);
  check_result(A_ser, A_k1, SIZE1);
  cout << "Kernel 1 on GPU: " << (1.0 * SIZE1 * SIZE1 * ITER / t / 1.0e9)
       << " GFLOPS; Time = " << kernel_time << " msec" << endl;

  cudaEventRecord(start, 0);
  // SB: Write your second GPU kernel here
  kernel2<<<1, SIZE2 - 1>>>(A_k2);
  cudaEventRecord(end, 0);
  cudaEventElapsedTime(&kernel_time, start, end);
  check_result(A_ser, A_k2, SIZE2);
  cout << "Kernel 2 on GPU: " << (1.0 * SIZE2 * SIZE2 * ITER / t / 1.0e9)
       << " GFLOPS; Time = " << kernel_time << " msec" << endl;

  return EXIT_SUCCESS;
}
