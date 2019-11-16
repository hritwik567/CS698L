#include <cassert>
#include <iostream>
#include <omp.h>

using namespace std;

void oddEvenSort_ref(uint32_t* A, uint32_t N) {
  uint32_t tmp;
  for (int phase = 0; phase < N; phase++) {
    if (phase % 2 == 0) {
      for (int i = 1; i < N; i += 2) {
        if (A[i - 1] > A[i]) {
          tmp = A[i];
          A[i] = A[i - 1];
          A[i - 1] = tmp;
        }
      }
    } else {
      for (int i = 1; i < N - 1; i += 2) {
        if (A[i] > A[i + 1]) {
          tmp = A[i + 1];
          A[i + 1] = A[i];
          A[i] = tmp;
        }
      }
    }
  }
}

void OddEvenSort_omp(uint32_t* A, uint32_t N) {
  // SB: Write your OpenMP code here
  uint32_t tmp;
  for (int phase = 0; phase < N; phase++) {
    if (phase % 2 == 0) {
      #pragma omp parallel for private(tmp)
      for (int i = 1; i < N; i += 2) {
        if (A[i - 1] > A[i]) {
          tmp = A[i];
          A[i] = A[i - 1];
          A[i - 1] = tmp;
        }
      }
    } else {
      #pragma omp parallel for private(tmp)
      for (int i = 1; i < N - 1; i += 2) {
        if (A[i] > A[i + 1]) {
          tmp = A[i + 1];
          A[i + 1] = A[i];
          A[i] = tmp;
        }
      }
    }
  }
}

void check_result(uint32_t* w_ref, uint32_t* w_opt, uint32_t N) {
  for (int i = 0; i < N; i++) {
    if (w_opt[i] != w_ref[i]) {
      cout << "Differences found between the two arrays.\n";
      break;
    }
  }
  cout << "No differences found between base and test versions\n";
}

int main() {
  const uint32_t N = (1 << 14);

  uint32_t* A_ref = new uint32_t[N];
  uint32_t* A_omp1 = new uint32_t[N];
  uint32_t* A_omp2 = new uint32_t[N];
  for (int i = 0; i < N; i++) {
    A_ref[i] = N - i;
    A_omp1[i] = N - i;
    A_omp2[i] = N - i;
  }

  double start = omp_get_wtime();
  oddEvenSort_ref(A_ref, N);
  double end = omp_get_wtime();
  cout << "Time for reference version: " << end - start << " seconds\n";

  start = omp_get_wtime();
  OddEvenSort_omp(A_omp1, N);
  end = omp_get_wtime();
  check_result(A_ref, A_omp1, N);
  cout << "Time for OpenMP version: " << end - start << " seconds\n";
}
