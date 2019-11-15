// Compile: nvcc -arch=sm_61 -std=c++11 assignment5-p3.cu -o assignment5-p3

#include <cmath>
#include <iostream>
#include <sys/time.h>

#define SIZE 4096
#define THRESHOLD (0.000001)

using namespace std;

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

__host__ void ATAonCPU(double** M, double** P) {
  for (int k = 0; k < SIZE; k++) {
    for (int i = 0; i < SIZE; i++) {
      for (int j = 0; j < SIZE; j++)
        P[i][j] += M[k][i] * M[k][j];
    }
  }
}

__host__ void check_result(double** Test, double** Ref) {
  double maxdiff = 0, rel_diff = 0;
  int numdiffs = 0;

  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; i < SIZE; j++) {
      rel_diff = (Test[i][j] - Ref[i][j]);
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
__global__ void ATAkernel(double* A, double* B) {}

int main() {
  cout << "Matrix Size = " << SIZE << "\n";

  double** A = new double*[SIZE];
  for (int i = 0; i < SIZE; i++) {
    A[i] = new double[SIZE];
  }

  double** O_s = new double*[SIZE];
  for (int i = 0; i < SIZE; i++) {
    O_s[i] = new double[SIZE];
  }

  double** O_p = new double*[SIZE];
  for (int i = 0; i < SIZE; i++) {
    O_p[i] = new double[SIZE];
  }

  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      A[i][j] = i * j * 0.25;
      O_s[i][j] = 0;
      O_p[i][j] = 0;
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

  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start, 0);
  // SB: Write your GPU kernel here
  cudaEventRecord(end, 0);
  float kernel_time;
  cudaEventElapsedTime(&kernel_time, start, end);
  cout << "A^T.A on GPU: " << (2.0 * SIZE * SIZE * SIZE / t / 1.0e9)
       << " GFLOPS; Time = " << kernel_time << " msec\n";

  check_result(O_p, O_s);

  return EXIT_SUCCESS;
}
