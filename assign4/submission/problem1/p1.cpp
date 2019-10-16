#include <cassert>
#include <iostream>
#include <omp.h>

#define N (1 << 12)
#define ITER 100

using namespace std;

void check_result(uint32_t **w_ref, uint32_t **w_opt) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      assert(w_ref[i][j] == w_opt[i][j]);
    }
  }
  cout << "No differences found between base and test versions\n";
}

void reference(uint32_t **A) {
  int i, j, k;
  for (k = 0; k < ITER; k++) {
    for (i = 1; i < N; i++) {
      for (j = 0; j < (N - 1); j++) {
        A[i][j + 1] = A[i - 1][j + 1] + A[i][j + 1];
      }
    }
  }
}

// SB: MAKE YOUR CHANGES TO OPTIMIZE THIS FUNCTION
void omp_version(uint32_t **A) {
  int i, j, k;
  for (k = 0; k < ITER; k++) {
    for (i = 1; i < N; i++) {
      #pragma omp parallel for
      for (j = 0; j < (N - 1); j++) {
        A[i][j + 1] = A[i - 1][j + 1] + A[i][j + 1];
      }
    }
  }
}

int main() {
  uint32_t **A_ref = new uint32_t *[N];
  for (int i = 0; i < N; i++) {
    A_ref[i] = new uint32_t[N];
  }

  uint32_t **A_omp = new uint32_t *[N];
  for (int i = 0; i < N; i++) {
    A_omp[i] = new uint32_t[N];
  }

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A_ref[i][j] = i + j + 1;
      A_omp[i][j] = i + j + 1;
    }
  }

  double start = omp_get_wtime();
  reference(A_ref);
  double end = omp_get_wtime();
  cout << "Time for reference version: " << end - start << " seconds\n";

  start = omp_get_wtime();
  omp_version(A_omp);
  end = omp_get_wtime();

  check_result(A_ref, A_omp);
  cout << "Time for OpenMP version: " << end - start << " seconds\n";
}
